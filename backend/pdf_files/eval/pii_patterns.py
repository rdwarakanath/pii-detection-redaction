"""
pii_patterns.py - detection building blocks for the redaction evaluator.

Contains (nothing here is tied to one document):
  * canonical PII types + risk weights + severity tiers
  * checksum validators  (Verhoeff/Aadhaar, Luhn, GSTIN, PESEL, IBAN mod-97, PAN, SSN)
  * context-aware regex rules (Aadhaar, PAN, GSTIN, IFSC, UPI, phone, e-mail, SSN, card,
    IBAN, passport, DOB, address, bank account, IP, employee id, ...)
  * label maps that translate model labels (gravitee / GLiNER / PII-Tracer / any HF model)
    into the canonical types
  * span merging: regex + model spans -> one clean entity list
  * user config (extra rules, weights, allowlist) loaded from a JSON file

Everything is pure python (re / math / json) so this file imports without any ML library.
"""
from __future__ import annotations

import bisect
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

# =============================================================================
# 1. Canonical types, risk weights, tiers
# =============================================================================
# Base risk weight 0-10: how harmful it is if ONE such value leaks.
RISK_WEIGHTS: dict[str, float] = {
    "AADHAAR": 10, "SSN": 10, "CARD": 10, "PASSWORD": 10,
    "BANK_ACCOUNT": 9, "PASSPORT": 9, "NATIONAL_ID": 9,
    "DRIVER_LICENSE": 8, "TAX_ID": 8, "PAN": 8, "IBAN": 8,
    "FINANCIAL_ID": 6, "DOB": 6,
    "UPI": 5, "ADDRESS": 5, "GSTIN": 5,
    "USERNAME": 4, "PHONE": 4, "EMAIL": 3.5,
    "PERSON": 3, "EMPLOYEE_ID": 3, "PERSONAL_DATE": 3, "OTHER_PII": 3, "DEVICE_ID": 3,
    "IP_ADDRESS": 2, "IFSC": 2, "URL_PERSONAL": 2,
    "PARTIAL_ID": 1.5, "AGE": 1.5, "LOCATION": 0.8,
}

TIER_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


def tier_of(weight: float) -> str:
    if weight >= 8:
        return "CRITICAL"
    if weight >= 5:
        return "HIGH"
    if weight >= 3:
        return "MEDIUM"
    return "LOW"


# Types compared by normalised text (names, e-mail ...) vs "identifier-like" types
TEXT_TYPES = {"PERSON", "ADDRESS", "LOCATION", "EMAIL", "UPI", "URL_PERSONAL", "USERNAME",
              "OTHER_PII", "AGE", "PERSONAL_DATE"}
# identifier types where a *partially* surviving number (e.g. 8 of 12 digits) still counts as a leak
PARTIAL_RUN_TYPES = {"AADHAAR", "SSN", "CARD", "BANK_ACCOUNT", "NATIONAL_ID", "PHONE", "PASSPORT",
                     "PAN", "TAX_ID", "IBAN", "DRIVER_LICENSE", "GSTIN"}


# =============================================================================
# 2. Data classes
# =============================================================================
@dataclass
class RawSpan:
    start: int
    end: int
    text: str
    etype: str
    conf: float
    source: str                  # "regex:pan", "model:gravitee", ...
    validated: bool = False      # passed a checksum / strict format check


@dataclass
class Entity:
    start: int
    end: int
    text: str
    etype: str
    conf: float
    sources: set = field(default_factory=set)
    validated: bool = False
    line: int = 0
    key: str = ""                # normalised value, used to match before/after
    weight: float = 0.0          # filled by the scorer

    def to_dict(self) -> dict:
        return {"start": self.start, "end": self.end, "text": self.text, "etype": self.etype,
                "conf": round(self.conf, 4), "sources": sorted(self.sources),
                "validated": self.validated, "line": self.line}

    @staticmethod
    def from_dict(d: dict) -> "Entity":
        e = Entity(d["start"], d["end"], d["text"], d["etype"], d["conf"],
                   set(d["sources"]), d["validated"], d.get("line", 0))
        e.key = normalize_value(e.etype, e.text)
        return e


# =============================================================================
# 3. Validators
# =============================================================================
_VD = [[0,1,2,3,4,5,6,7,8,9],[1,2,3,4,0,6,7,8,9,5],[2,3,4,0,1,7,8,9,5,6],[3,4,0,1,2,8,9,5,6,7],
       [4,0,1,2,3,9,5,6,7,8],[5,9,8,7,6,0,4,3,2,1],[6,5,9,8,7,1,0,4,3,2],[7,6,5,9,8,2,1,0,4,3],
       [8,7,6,5,9,3,2,1,0,4],[9,8,7,6,5,4,3,2,1,0]]
_VP = [[0,1,2,3,4,5,6,7,8,9],[1,5,7,6,2,8,3,0,9,4],[5,8,0,3,7,9,6,1,4,2],[8,9,1,6,0,4,3,5,2,7],
       [9,4,5,3,1,2,6,8,7,0],[4,2,8,6,5,7,3,9,0,1],[2,7,9,3,8,0,6,4,1,5],[7,0,4,6,9,1,3,2,5,8]]


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def verhoeff_ok(value: str) -> bool:
    d = _digits(value)
    if len(d) != 12:
        return False
    c = 0
    for i, ch in enumerate(reversed(d)):
        c = _VD[c][_VP[i % 8][int(ch)]]
    return c == 0


def luhn_ok(value: str) -> bool:
    d = _digits(value)
    if not 13 <= len(d) <= 19:
        return False
    total = 0
    for i, ch in enumerate(reversed(d)):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


_CARD_PREFIX = re.compile(r"^(?:4|5[1-5]|2[2-7]|3[47]|6011|65|64[4-9]|35|30[0-5]|36|38|39|62|60|81|82|508)")


def card_ok(value: str) -> bool:
    d = _digits(value)
    return luhn_ok(d) and bool(_CARD_PREFIX.match(d))


def pan_ok(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Z]{5}\d{4}[A-Z]", value)) and value[3] in "ABCFGHJLPT"


_GST_CH = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def gstin_ok(value: str) -> bool:
    g = value.upper()
    if not re.fullmatch(r"\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]", g):
        return False
    total = 0
    for i, c in enumerate(g[:14]):
        v = _GST_CH.index(c) * (1 if i % 2 == 0 else 2)
        total += v // 36 + v % 36
    return _GST_CH[(36 - total % 36) % 36] == g[14]


def pesel_ok(value: str) -> bool:
    d = _digits(value)
    if len(d) != 11:
        return False
    w = [1, 3, 7, 9, 1, 3, 7, 9, 1, 3]
    if (10 - sum(int(a) * b for a, b in zip(d, w)) % 10) % 10 != int(d[10]):
        return False
    return 1 <= int(d[2:4]) % 20 <= 12 and 1 <= int(d[4:6]) <= 31


def iban_ok(value: str) -> bool:
    s = re.sub(r"\s", "", value).upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}", s):
        return False
    r = s[4:] + s[:4]
    return int("".join(str(int(c, 36)) for c in r)) % 97 == 1


def ssn_strict_ok(value: str) -> bool:
    d = _digits(value)
    if len(d) != 9:
        return False
    area, grp, ser = int(d[:3]), int(d[3:5]), int(d[5:])
    return area not in (0, 666) and area < 900 and grp != 0 and ser != 0


def phone_ok(value: str) -> bool:
    d = _digits(value)
    return 8 <= len(d) <= 15 and len(set(d)) > 2


# Length-preserving clean-up so OCR oddities (full-width brackets/digits, odd dashes, nbsp) don't hide PII.
_NORM = str.maketrans({
    "\uff08": "(", "\uff09": ")", "\uff0b": "+", "\uff20": "@", "\uff1a": ":",
    "\u2013": "-", "\u2212": "-", "\u2011": "-", "\u2010": "-",
    "\u00a0": " ", "\u202f": " ", "\u2009": " ",
    **{chr(0xFF10 + i): str(i) for i in range(10)},
})


def normalize_for_detection(text: str) -> str:
    return text.translate(_NORM)


# =============================================================================
# 4. Normalisation (used to match "the same value" before vs after)
# =============================================================================
def normalize_value(etype: str, text: str) -> str:
    text = normalize_for_detection(text)
    if etype in ("EMAIL", "UPI"):
        return text.strip().lower()
    if etype == "PHONE":
        d = _digits(text)
        return d[-10:] if len(d) >= 10 else d
    if etype in TEXT_TYPES:
        t = re.sub(r"[^\w\s@.\-]", " ", text.casefold())
        return " ".join(t.split())
    return re.sub(r"[^A-Za-z0-9]", "", text).upper()


# =============================================================================
# 5. Redaction-mask markers (so a masked value is never mistaken for PII)
# =============================================================================
MASK_RE = re.compile(
    r"\[(?:REDACTED|MASKED|PII|HIDDEN|REMOVED|[A-Z][A-Z_ ]{2,24})\]"
    r"|<(?:REDACTED|MASKED|[A-Z][A-Z_]{2,24})>"
    r"|\{\{[A-Za-z_ ]{2,25}\}\}"
    r"|[█■▇▉]{2,}|\bX{4,}\b|\bx{4,}\b|\*{3,}|#{5,}"
)


# =============================================================================
# 6. Regex rules
# =============================================================================
def context_hit(text: str, start: int, end: int, rx: re.Pattern, before: int = 60, after: int = 20) -> bool:
    return bool(rx.search(text[max(0, start - before):start] + " " + text[end:end + after]))


@dataclass
class Rule:
    name: str
    etype: str
    rx: re.Pattern
    conf: float
    group: int = 0
    validator: Optional[Callable[[str], bool]] = None
    valid_conf: Optional[float] = None       # confidence when validator passes
    invalid_conf: Optional[float] = None     # confidence when it fails (None = drop the match)
    ctx: Optional[re.Pattern] = None         # nearby keywords that raise confidence
    ctx_boost: float = 0.2
    ctx_required: bool = False
    pre: Optional[Callable] = None           # (val, s, e) -> (val, s, e) tweak


def _trim_phone(val: str, s: int, e: int):
    while sum(c.isdigit() for c in val) > 15:
        m = re.search(r"[\s.\-]?\(?\d+\)?\s*$", val)
        if not m or m.start() == 0:
            break
        val = val[:m.start()]
    while val.count("(") != val.count(")"):          # drop dangling "(24" / "...)"
        if val.endswith(")") and val.count(")") > val.count("("):
            val = val[:-1]
        elif "(" in val and val.count("(") > val.count(")"):
            val = val[:val.rindex("(")]
        else:
            break
    val = val.rstrip(" .-(")
    return val, s, s + len(val)


def _c(p: str, flags: int = 0) -> re.Pattern:
    return re.compile(p, flags)


_DATE = (r"(?:\d{1,2}[\s/\-.](?:\d{1,2}|[A-Za-z]{3,9})[\s/\-.,]*\d{2,4}"
         r"|[A-Za-z]{3,9}\.?\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2})")
_STREET = (r"(?:Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Drive|Dr|Court|Ct|Crescent|Boulevard|Blvd|Way|"
           r"Place|Pl|Terrace|Square|Sq|Highway|Hwy|Nagar|Marg|Colony|Layout|Cross)")
_UPI_HANDLES = ("oksbi|okhdfcbank|okicici|okaxis|ybl|ibl|axl|paytm|apl|upi|sbi|hdfcbank|icici|axisbank|"
                "pnb|fbl|barodampay|kotak|idfcbank|yesbank|cnrb|unionbank|aubank|jupiteraxis|freecharge")

BUILTIN_RULES: list[Rule] = [
    Rule("email", "EMAIL",
         _c(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}"), 0.97),
    Rule("upi_known", "UPI",
         _c(rf"(?<![\w.%+\-@])[A-Za-z0-9._\-]{{2,64}}@(?:{_UPI_HANDLES})(?![\w]|\.[A-Za-z])", re.I), 0.92),
    Rule("upi_generic", "UPI",
         _c(r"(?<![\w.%+\-@])[A-Za-z0-9._\-]{2,64}@[A-Za-z]{2,20}(?![\w]|\.[A-Za-z])"), 0.5,
         ctx=_c(r"\b(?:upi|vpa|gpay|phonepe|bhim|pay)\b", re.I), ctx_boost=0.4),

    Rule("phone_intl", "PHONE",
         _c(r"(?<![\w+])\+\d{1,3}(?:[ .\-]?\(?\d{1,5}\)?){2,6}"), 0.95,
         validator=phone_ok, valid_conf=0.95, pre=_trim_phone),
    Rule("phone_us", "PHONE",
         _c(r"(?<![\w])(?:\(\d{3}\)\s?|\d{3}[\s.\-])\d{3}[\s.\-]\d{4}(?!\d)"), 0.72,
         ctx=_c(r"\b(?:phone|tel|mobile|cell|call|contact|fax)\b", re.I), ctx_boost=0.2),
    Rule("phone_in_mobile", "PHONE",
         _c(r"(?<![\d\w])(?:(?:\+?91|0)[\s\-]?)?[6-9]\d{4}[\s\-]?\d{5}(?!\d)"), 0.55,
         ctx=_c(r"\b(?:phone|tel|mobile|cell|call|contact|whatsapp)\b", re.I), ctx_boost=0.35),

    Rule("aadhaar", "AADHAAR",
         _c(r"(?<![\d\-])(?<!\d )(?<!\d\-)[2-9]\d{3}[\s\-]?\d{4}[\s\-]?\d{4}(?![\d\-])(?! \d)"), 0.30,
         validator=verhoeff_ok, valid_conf=0.95, invalid_conf=0.30,
         ctx=_c(r"\b(?:aadhaar|aadhar|uidai|uid|आधार)\b", re.I), ctx_boost=0.45),
    Rule("pan", "PAN",
         _c(r"(?<![A-Za-z0-9])[A-Z]{5}\d{4}[A-Z](?![A-Za-z0-9])"), 0.55,
         validator=pan_ok, valid_conf=0.95, invalid_conf=0.55,
         ctx=_c(r"\b(?:pan|permanent account|income tax)\b", re.I), ctx_boost=0.2),
    Rule("tan", "TAX_ID",
         _c(r"(?<![A-Za-z0-9])[A-Z]{4}\d{5}[A-Z](?![A-Za-z0-9])"), 0.45,
         ctx=_c(r"\b(?:tan|tax deduction)\b", re.I), ctx_boost=0.4),
    Rule("gstin", "GSTIN",
         _c(r"(?<![A-Za-z0-9])\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z](?![A-Za-z0-9])"), 0.6,
         validator=gstin_ok, valid_conf=0.98, invalid_conf=0.6,
         ctx=_c(r"\b(?:gstin|gst)\b", re.I), ctx_boost=0.3),
    Rule("ifsc", "IFSC",
         _c(r"(?<![A-Za-z0-9])[A-Z]{4}0[A-Z0-9]{6}(?![A-Za-z0-9])"), 0.7,
         ctx=_c(r"\bifsc\b", re.I), ctx_boost=0.28),

    Rule("ssn", "SSN",
         _c(r"(?<![\d\-])\d{3}[\-\s]\d{2}[\-\s]\d{4}(?![\d\-])"), 0.55,
         validator=ssn_strict_ok, valid_conf=0.75, invalid_conf=0.55,
         ctx=_c(r"\b(?:ssn|social\s+security|soc\.?\s*sec)\b", re.I), ctx_boost=0.4),
    Rule("card", "CARD",
         _c(r"(?<![\d\-])(?<!\d )(?:\d{4}([ \-])\d{4}\1\d{4}\1\d{1,7}|\d{4}([ \-])\d{6}\2\d{5}|\d{13,19})(?![\d\-])(?! \d)"), 0.4,
         validator=card_ok, valid_conf=0.93, invalid_conf=None,
         ctx=_c(r"\b(?:card|visa|mastercard|amex|rupay|pan\s+no|credit|debit)\b", re.I), ctx_boost=0.06),
    Rule("card_luhn_only", "CARD",
         _c(r"(?<![\d\-])(?<!\d )(?:\d{4}([ \-])\d{4}\1\d{4}\1\d{1,7}|\d{4}([ \-])\d{6}\2\d{5}|\d{13,19})(?![\d\-])(?! \d)"), 0.5,
         validator=luhn_ok, valid_conf=0.5, invalid_conf=None,
         ctx=_c(r"\b(?:card|credit|debit)\b", re.I), ctx_boost=0.4, ctx_required=True),
    Rule("iban", "IBAN",
         _c(r"(?<![A-Za-z0-9])[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,4})?(?![A-Za-z0-9])"), 0.9,
         validator=iban_ok, valid_conf=0.97, invalid_conf=None),
    Rule("pesel", "NATIONAL_ID",
         _c(r"(?<![\d\w])\d{11}(?!\d)"), 0.6,
         validator=pesel_ok, valid_conf=0.7, invalid_conf=None,
         ctx=_c(r"\b(?:pesel|national\s+id|nric|id\s+number)\b", re.I), ctx_boost=0.27),
    Rule("nino", "NATIONAL_ID",
         _c(r"(?<![A-Za-z])[A-CEGHJ-PR-TW-Z][A-CEGHJ-NPR-TW-Z]\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D](?![A-Za-z0-9])"), 0.8),

    Rule("passport_ctx", "PASSPORT",
         _c(r"(?i:\bpassport)(?i:\s*(?:no\.?|number|num|#|id))?\s*[:#\-]?\s*((?=[A-Z0-9]*\d)[A-Z0-9]{6,9})(?![A-Za-z0-9])"),
         0.92, group=1),
    Rule("passport_in", "PASSPORT",
         _c(r"(?<![A-Za-z0-9])[A-PR-WY][1-9]\d{6}(?![A-Za-z0-9])"), 0.4,
         ctx=_c(r"\bpassport\b", re.I), ctx_boost=0.5),
    Rule("voter_id", "NATIONAL_ID",
         _c(r"(?<![A-Za-z0-9])[A-Z]{3}\d{7}(?![A-Za-z0-9])"), 0.4,
         ctx=_c(r"\b(?:voter|epic)\b", re.I), ctx_boost=0.5),
    Rule("driving_in", "DRIVER_LICENSE",
         _c(r"(?<![A-Za-z0-9])[A-Z]{2}\d{2}[\s\-]?\d{11}(?![A-Za-z0-9])"), 0.5,
         ctx=_c(r"\b(?:driving|driver|licen[cs]e|dl)\b", re.I), ctx_boost=0.4),
    Rule("driver_ctx", "DRIVER_LICENSE",
         _c(r"(?i:\b(?:driver'?s?|driving)\s+licen[cs]e(?:\s+(?:no\.?|number|#))?)\s*[:#\-]?\s*((?=[A-Z0-9\-]*\d)[A-Z0-9][A-Z0-9\-]{5,17})(?![A-Za-z0-9])"),
         0.9, group=1),
    Rule("tax_id_ctx", "TAX_ID",
         _c(r"(?i:\b(?:tax\s+(?:id|identification\s+number|identifier)|taxpayer\s+id|tin|ein|itin)\b)\s*[:#\-]?\s*((?=[A-Z0-9\-]*\d)[A-Z0-9][A-Z0-9\-]{5,15})(?![A-Za-z0-9])"),
         0.85, group=1),
    Rule("national_id_ctx", "NATIONAL_ID",
         _c(r"(?i:\b(?:national\s+(?:id|identity|identifier)(?:\s+(?:no\.?|number))?|id\s+card\s+(?:no\.?|number))\b)\s*[:#\-]?\s*((?=[A-Z0-9\-]*\d)[A-Z0-9][A-Z0-9\-]{5,17})(?![A-Za-z0-9])"),
         0.85, group=1),

    Rule("bank_account_ctx", "BANK_ACCOUNT",
         _c(r"(?i:\b(?:a/c|acct|account|iban|bank\s+account)(?:\s*(?:no\.?|number|num|#))?)\s*[:#\-]?\s*(\d[\d\- ]{7,22}\d)(?!\d)"),
         0.88, group=1),
    Rule("partial_ending", "PARTIAL_ID",
         _c(r"(?i:\b(?:card|account|acct|a/c)\b[^.\n\d]{0,40}?\bending(?:\s+(?:in|with))?)\s+(\d{4})(?!\d)"), 0.9, group=1),
    Rule("partial_last4", "PARTIAL_ID",
         _c(r"(?i:last\s+(?:four|4)\s+digits[^()\n]{0,40}\()\s*(\d{4})\s*\)"), 0.9, group=1),

    Rule("dob_ctx", "DOB",
         _c(rf"(?i:\b(?:dob|d\.o\.b\.?|date\s+of\s+birth|birth\s*date|born(?:\s+on)?|birthday)\b)\s*[:\-,]?\s*({_DATE})"),
         0.95, group=1),

    Rule("address_street", "ADDRESS",
         _c(rf"\b\d{{1,5}}[A-Za-z]?\s+(?:[A-Z][A-Za-z'.\-]*\s+){{1,4}}{_STREET}\b\.?"
            rf"(?:,?\s+(?:Apartment|Apt|Suite|Ste|Unit|Flat|Floor|Fl|#)\s*[\w\-]+)?"
            rf"(?:,\s*[A-Z][A-Za-z.'\-]*(?: [A-Z][A-Za-z.'\-]*){{0,3}}){{0,2}}"
            rf"(?:,?\s*[A-Z]{{2}}\s+\d{{5}}(?:-\d{{4}})?)?"), 0.8),
    Rule("ipv4", "IP_ADDRESS",
         _c(r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\d.])"), 0.7),

    Rule("employee_prefix", "EMPLOYEE_ID",
         _c(r"(?<![A-Za-z0-9])(?:EMP|EMPID|EID|STAFF|STF|DIR|VEN|VND)[-_ ]?\d{3,8}(?![A-Za-z0-9])"), 0.8),
    Rule("employee_ctx", "EMPLOYEE_ID",
         _c(r"(?i:\b(?:employee|emp|staff|vendor|worker|contractor)\s*(?:id|no\.?|number|#))\s*[:#\-]?\s*((?=[A-Z0-9\-_]*\d)[A-Z0-9][A-Z0-9\-_]{3,17})(?![A-Za-z0-9])"),
         0.85, group=1),

    Rule("secret_ctx", "PASSWORD",
         _c(r"(?i:\b(?:password|passwd|pwd|secret|api[_\- ]?key|token)\b)\s*[:=]\s*(\S{6,})"), 0.8, group=1),
    Rule("aws_key", "PASSWORD", _c(r"(?<![A-Z0-9])AKIA[0-9A-Z]{16}(?![A-Z0-9])"), 0.95),

    # light name heuristics so a regex-only run is not blind to people
    Rule("name_honorific", "PERSON",
         _c(r"\b(?:Mr|Mrs|Ms|Miss|Dr|Prof|Shri|Smt|Sri)\.?\s+[A-Z][a-z]+(?:[\s\-][A-Z][a-z]+){0,2}"), 0.6),
    Rule("name_labelled", "PERSON",
         _c(r"(?i:\b(?:name|customer|client|applicant|holder|beneficiary|attested\s+by)\s*:\s*)([A-Z][a-z]+(?:\s+[A-Z]\.?)?(?:\s+[A-Z][a-z'\-]+)+)"),
         0.6, group=1),
]


def compile_extra_rules(items: list[dict]) -> list[Rule]:
    rules = []
    for it in items or []:
        rules.append(Rule(
            name=it["name"], etype=it["type"].upper(), rx=re.compile(it["regex"]),
            conf=float(it.get("conf", 0.8)), group=int(it.get("group", 0)),
            ctx=re.compile(it["context"], re.I) if it.get("context") else None,
            ctx_boost=float(it.get("context_boost", 0.2)),
            ctx_required=bool(it.get("context_required", False)),
        ))
    return rules


def find_regex_spans(text: str, rules: list[Rule]) -> list[RawSpan]:
    out: list[RawSpan] = []
    orig, text = text, normalize_for_detection(text)
    for rule in rules:
        for m in rule.rx.finditer(text):
            s, e = m.span(rule.group)
            if s < 0:
                continue
            val = text[s:e]
            if rule.pre:
                val, s, e = rule.pre(val, s, e)
            conf, validated = rule.conf, False
            if rule.validator:
                if rule.validator(val):
                    conf, validated = (rule.valid_conf if rule.valid_conf is not None else rule.conf), True
                elif rule.invalid_conf is None:
                    continue
                else:
                    conf = rule.invalid_conf
            if rule.ctx is not None:
                if context_hit(text, s, e, rule.ctx):
                    conf = min(0.99, conf + rule.ctx_boost)
                elif rule.ctx_required:
                    continue
            out.append(RawSpan(s, e, orig[s:e], rule.etype, conf, f"regex:{rule.name}", validated))
    return out


# =============================================================================
# 7. Model label maps
# =============================================================================
# model label (upper-case, '-'/' ' -> '_')  ->  (canonical type | None, confidence multiplier)
_LABELS: dict[str, tuple[Optional[str], float]] = {}


def _reg(etype: Optional[str], labels: str, mult: float = 1.0):
    for lb in labels.split():
        _LABELS[lb] = (etype, mult)


_reg("PERSON", "PERSON PER NAME GIVENNAME SURNAME FIRSTNAME LASTNAME FIRST_NAME LAST_NAME MIDDLE_NAME "
               "FULL_NAME PERSON_NAME PRIVATE_PERSON PATIENT DOCTOR NAME_PERSON")
_reg(None, "ORGANIZATION ORG COMPANY COMPANY_NAME NRP DATE_TIME DATE TIME URL GPE")  # not personal
_reg("LOCATION", "LOCATION LOC CITY STATE COUNTRY COORDINATE", 0.7)
_reg("ADDRESS", "ADDRESS STREET STREET_ADDRESS PRIVATE_ADDRESS BUILDINGNUM ZIPCODE POSTCODE POSTAL_CODE HOME_ADDRESS")
_reg("EMAIL", "EMAIL EMAIL_ADDRESS PRIVATE_EMAIL")
_reg("PHONE", "PHONE PHONE_NUMBER TELEPHONENUM PRIVATE_PHONE MOBILE_NUMBER MOBILE FAX_NUMBER TELEPHONE")
_reg("DOB", "DATE_OF_BIRTH DOB DATEOFBIRTH BIRTHDATE BIRTH_DATE")
_reg("PERSONAL_DATE", "PRIVATE_DATE", 0.8)
_reg("SSN", "US_SSN SSN SOCIALNUM SOCIAL_SECURITY_NUMBER")
_reg("PASSPORT", "US_PASSPORT PASSPORT PASSPORT_NUMBER UK_PASSPORT")
_reg("DRIVER_LICENSE", "US_DRIVER_LICENSE DRIVERLICENSENUM DRIVER_LICENSE DRIVERS_LICENSE DRIVING_LICENSE")
_reg("TAX_ID", "TAXNUM TAX_ID TIN US_ITIN TAX_IDENTIFICATION_NUMBER IN_TAN")
_reg("PAN", "IN_PAN PAN PAN_NUMBER")
_reg("AADHAAR", "IN_AADHAAR AADHAAR AADHAAR_NUMBER AADHAR")
_reg("NATIONAL_ID", "IDCARDNUM ID_NUMBER NATIONAL_ID NATIONAL_ID_NUMBER ID_CARD_NUMBER IN_VOTER UK_NHS UK_NINO")
_reg("CARD", "CREDIT_CARD CREDITCARDNUMBER CREDIT_CARD_NUMBER CARD_NUMBER DEBIT_CARD")
_reg("IBAN", "IBAN_CODE IBAN")
_reg("BANK_ACCOUNT", "ACCOUNT_NUMBER ACCOUNTNUM BANK_ACCOUNT BANK_ACCOUNT_NUMBER PRIVATE_ACCOUNT ACCOUNT")
_reg("FINANCIAL_ID", "FINANCIAL", 0.9)
_reg("IP_ADDRESS", "IP_ADDRESS IP IPV4 IPV6")
_reg("DEVICE_ID", "IMEI MAC_ADDRESS DEVICE_ID")
_reg("PASSWORD", "PASSWORD SECRET API_KEY SECRET_KEY CREDENTIAL")
_reg("USERNAME", "USERNAME USER_NAME USER_ID")
_reg("EMPLOYEE_ID", "EMPLOYEE_ID EMPLOYEE_NUMBER STAFF_ID")
_reg("AGE", "AGE")
_reg("URL_PERSONAL", "PRIVATE_URL PERSONAL_URL")
_reg("OTHER_PII", "OTHER_PII", 0.9)


def map_model_label(label: str) -> tuple[Optional[str], float]:
    lb = re.sub(r"^[BIES]-", "", label.strip().upper()).replace("-", "_").replace(" ", "_")
    if lb in _LABELS:
        return _LABELS[lb]
    return "OTHER_PII", 0.7          # unknown label: keep, but with reduced confidence


# GLiNER is zero-shot: natural-language label -> canonical type
GLINER_LABELS: dict[str, str] = {
    "person": "PERSON", "email address": "EMAIL", "phone number": "PHONE",
    "social security number": "SSN", "passport number": "PASSPORT",
    "aadhaar number": "AADHAAR", "pan card number": "PAN",
    "bank account number": "BANK_ACCOUNT", "credit card number": "CARD",
    "date of birth": "DOB", "street address": "ADDRESS",
    "tax identification number": "TAX_ID", "driver license number": "DRIVER_LICENSE",
    "ip address": "IP_ADDRESS", "employee id": "EMPLOYEE_ID",
    "national identification number": "NATIONAL_ID", "username": "USERNAME", "password": "PASSWORD",
}


# =============================================================================
# 8. Merging regex + model spans into entities
# =============================================================================
def _overlap(a_s: int, a_e: int, b_s: int, b_e: int) -> bool:
    return a_s < b_e and b_s < a_e


def _noisy_or(*confs: float) -> float:
    p = 1.0
    for c in confs:
        p *= (1 - min(max(c, 0.0), 0.99))
    return min(0.99, 1 - p)


def merge_spans(text: str, regex_spans: list[RawSpan], model_spans: list[RawSpan]) -> list[Entity]:
    # ---- a) resolve regex overlaps (validated > confidence > length)
    ranked = sorted(regex_spans, key=lambda s: (-int(s.validated), -s.conf, -(s.end - s.start)))
    accepted: list[RawSpan] = []
    for sp in ranked:
        if not any(_overlap(sp.start, sp.end, a.start, a.end) for a in accepted):
            accepted.append(sp)
    ents = [Entity(a.start, a.end, a.text, a.etype, a.conf, {a.source}, a.validated) for a in accepted]
    ents.sort(key=lambda e: e.start)
    starts = [e.start for e in ents]

    # ---- b) model spans: merge adjacent fragments of the same model/type
    by_model: dict[str, list[RawSpan]] = {}
    for sp in model_spans:
        by_model.setdefault(sp.source, []).append(sp)
    fused: list[RawSpan] = []
    for src, spans in by_model.items():
        spans.sort(key=lambda s: (s.etype, s.start))
        cur: Optional[RawSpan] = None
        for sp in spans:
            gap_txt = text[cur.end:sp.start] if cur and sp.start >= cur.end else None
            if (cur and cur.etype == sp.etype and
                    ((gap_txt is not None and len(gap_txt) <= 2 and gap_txt.strip(" ,\t") == "")
                     or sp.start < cur.end)):
                cur = RawSpan(cur.start, max(cur.end, sp.end), text[cur.start:max(cur.end, sp.end)],
                              cur.etype, (cur.conf + sp.conf) / 2, src)
            else:
                if cur:
                    fused.append(cur)
                cur = sp
        if cur:
            fused.append(cur)

    # ---- c) model spans vs regex entities
    free: list[RawSpan] = []
    for sp in fused:
        hit = None
        i = bisect.bisect_left(starts, sp.start)
        for j in range(max(0, i - 2), min(len(ents), i + 6)):
            if _overlap(sp.start, sp.end, ents[j].start, ents[j].end):
                hit = ents[j]
                break
        if hit is None:
            free.append(sp)
        elif hit.etype == sp.etype or (hit.etype in ("PHONE", "AADHAAR", "BANK_ACCOUNT", "NATIONAL_ID", "CARD")
                                       and sp.etype in ("PHONE", "AADHAAR", "BANK_ACCOUNT", "NATIONAL_ID", "CARD")):
            hit.conf = _noisy_or(hit.conf, sp.conf)
            hit.sources.add(sp.source)
        # else: regex wins, model span discarded

    # ---- d) cluster remaining model spans across models
    free.sort(key=lambda s: s.start)
    clusters: list[list[RawSpan]] = []
    for sp in free:
        if clusters and sp.start < max(x.end for x in clusters[-1]):
            clusters[-1].append(sp)
        else:
            clusters.append([sp])
    for cl in clusters:
        by_type: dict[str, list[RawSpan]] = {}
        for sp in cl:
            by_type.setdefault(sp.etype, []).append(sp)
        best_t = max(by_type, key=lambda t: _noisy_or(*[s.conf for s in by_type[t]]))
        grp = by_type[best_t]
        s0, e0 = min(x.start for x in grp), max(x.end for x in grp)
        ents.append(Entity(s0, e0, text[s0:e0], best_t,
                           _noisy_or(*[x.conf for x in grp]), {x.source for x in grp}, False))

    ents.sort(key=lambda e: e.start)
    nl = [m.start() for m in re.finditer(r"\n", text)]
    for e in ents:
        e.line = bisect.bisect_right(nl, e.start) + 1
        e.key = normalize_value(e.etype, e.text)
    return ents


# =============================================================================
# 9. Config (extra rules / weights / allowlist)
# =============================================================================
@dataclass
class Config:
    rules: list = field(default_factory=list)
    weights: dict = field(default_factory=dict)
    allowlist_exact: set = field(default_factory=set)
    allowlist_regex: list = field(default_factory=list)
    raw: dict = field(default_factory=dict)

    def allowed(self, ent: Entity) -> bool:
        t = " ".join(ent.text.casefold().split())
        if t in self.allowlist_exact:
            return True
        return any(rx.search(ent.text) for rx in self.allowlist_regex)


def load_config(path: Optional[str]) -> Config:
    cfg = Config()
    if not path:
        return cfg
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cfg.raw = data
    cfg.rules = compile_extra_rules(data.get("rules", []))
    cfg.weights = {k.upper(): float(v) for k, v in data.get("weights", {}).items()}
    for item in data.get("allowlist", []):
        if item.startswith("re:"):
            cfg.allowlist_regex.append(re.compile(item[3:], re.I))
        else:
            cfg.allowlist_exact.add(" ".join(item.casefold().split()))
    return cfg


def weight_for(etype: str, cfg: Optional[Config] = None) -> float:
    if cfg and etype in cfg.weights:
        return cfg.weights[etype]
    return RISK_WEIGHTS.get(etype, 3.0)


def mask_preview(text: str) -> str:
    t = text.strip()
    if len(t) <= 4:
        return "*" * len(t)
    return t[:2] + "*" * min(len(t) - 4, 10) + t[-2:]
