"""
pii_helpers.py - rule/validator based PII detection for plain strings.

Complements Presidio / BERT. Every function takes a str (or list[str]) and returns
Span objects (character offsets), so main() can merge them with other detectors.

Typical flow in main():
    registry = build_name_registry(all_texts)            # pass 1 (document level)
    for text in all_texts:                                # pass 2
        spans = detect_pii(text, registry)
        spans = merge_detections(spans, presidio_spans)   # optional
"""
from __future__ import annotations

import datetime
import difflib
import re
from dataclasses import dataclass
from typing import Iterable, Optional

try:                                   # pip install phonenumbers
    import phonenumbers as _pn
except ImportError:
    _pn = None
try:                                   # pip install gender-guesser   (offline first-name list, ~40k names)
    import gender_guesser.detector as _gg
except ImportError:
    _gg = None
try:                                   # pip install rapidfuzz        (OCR-tolerant surname matching)
    from rapidfuzz import fuzz as _fuzz
except ImportError:
    _fuzz = None

# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Span:
    start: int
    end: int
    entity: str
    score: float
    source: str  # which rule produced it (for the audit log; never log raw text)


# --------------------------------------------------------------------------- #
# Validators (checksums) - these are what kill false positives
# --------------------------------------------------------------------------- #

def luhn_ok(digits: str) -> bool:
    d = [int(c) for c in digits if c.isdigit()]
    if not 13 <= len(d) <= 19:
        return False
    total = 0
    for i, n in enumerate(reversed(d)):
        if i % 2:
            n = n * 2 - 9 if n * 2 > 9 else n * 2
        total += n
    return total % 10 == 0


_VD = [[0,1,2,3,4,5,6,7,8,9],[1,2,3,4,0,6,7,8,9,5],[2,3,4,0,1,7,8,9,5,6],
       [3,4,0,1,2,8,9,5,6,7],[4,0,1,2,3,9,5,6,7,8],[5,9,8,7,6,0,4,3,2,1],
       [6,5,9,8,7,1,0,4,3,2],[7,6,5,9,8,2,1,0,4,3],[8,7,6,5,9,3,2,1,0,4],
       [9,8,7,6,5,4,3,2,1,0]]
_VP = [[0,1,2,3,4,5,6,7,8,9],[1,5,7,6,2,8,3,0,9,4],[5,8,0,3,7,9,6,1,4,2],
       [8,9,1,6,0,4,3,5,2,7],[9,4,5,3,1,2,6,8,7,0],[4,2,8,6,5,7,3,9,0,1],
       [2,7,9,3,8,0,6,4,1,5],[7,0,4,6,9,1,3,2,5,8]]


def verhoeff_ok(num: str) -> bool:  # Aadhaar
    digits = re.sub(r"\D", "", num)
    if len(digits) != 12 or digits[0] in "01":
        return False
    c = 0
    for i, ch in enumerate(reversed(digits)):
        c = _VD[c][_VP[i % 8][int(ch)]]
    return c == 0


def pesel_ok(s: str) -> bool:  # Poland
    if not re.fullmatch(r"\d{11}", s):
        return False
    w = [1, 3, 7, 9, 1, 3, 7, 9, 1, 3]
    return (10 - sum(int(a) * b for a, b in zip(s, w)) % 10) % 10 == int(s[-1])


def iban_ok(s: str) -> bool:
    s = re.sub(r"\s", "", s).upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}", s):
        return False
    t = s[4:] + s[:4]
    return int("".join(str(int(c, 36)) for c in t)) % 97 == 1


def pan_ok(s: str) -> bool:  # India PAN: 4th char = holder type
    return bool(re.fullmatch(r"[A-Z]{3}[ABCFGHLJPTK][A-Z]\d{4}[A-Z]", s))


def ssn_ok(s: str) -> tuple[bool, float]:
    """Returns (valid_shape, score). 9xx area numbers are never issued, so they
    get a lower base score (context words can still lift them - useful for
    synthetic data like this PDF)."""
    a, g, n = s.split("-")
    if a == "000" or a == "666" or g == "00" or n == "0000":
        return False, 0.0
    return True, (0.55 if a.startswith("9") else 0.75)


# --------------------------------------------------------------------------- #
# Context words (lift or require)
# --------------------------------------------------------------------------- #

def _window(text: str, start: int, end: int, before: int = 45, after: int = 12) -> str:
    return text[max(0, start - before): start].lower() + " | " + text[end: end + after].lower()


CTX = {
    "US_SSN": r"ssn|social security",
    "IN_PAN": r"\bpan\b|permanent account|tax id|tin\b",
    "PASSPORT": r"passport",
    "DATE_OF_BIRTH": r"\bdob\b|date of birth|born|birth",
    "TAX_ID": r"\btin\b|tax id|tax identifier|national tax",
    "PL_PESEL": r"pesel",
    "IN_AADHAAR": r"aadhaar|aadhar|uidai",
}


def _boost(text: str, s: int, e: int, entity: str, base: float, require: bool = False) -> float:
    pat = CTX.get(entity)
    if not pat:
        return base
    hit = re.search(pat, _window(text, s, e))
    if require and not hit:
        return 0.0
    return min(1.0, base + 0.2) if hit else base


# --------------------------------------------------------------------------- #
# Regex detectors for structured identifiers
# --------------------------------------------------------------------------- #

MONTHS = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
          r"Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)")

STREET_SUFFIX = (r"(?:Street|St|Road|Rd|Avenue|Ave|Court|Ct|Crescent|Lane|Ln|Drive|Dr|"
                 r"Boulevard|Blvd|Way|Place|Terrace|Close)")

_CITY = r"[A-Z][A-Za-z.]*(?:\s[A-Z][A-Za-z.]*){0,2}"
ADDRESS_RE = re.compile(
    rf"\b\d{{1,5}}\s+(?:[A-Z][\w'’\-]*\s+){{1,4}}?{STREET_SUFFIX}\b"
    rf"(?:,\s*(?:Apt\.?|Apartment|Suite|Unit|Flat)\s*\w+)?"
    rf"(?:,\s*{_CITY},\s*[A-Z]{{2}}\s+\d{{5}}(?:-\d{{4}})?)?"
)

PHONE_INTL = re.compile(r"(?<![\w+])\+\d[\d\s().\-]{7,17}\d(?!\d)")
PHONE_US = re.compile(r"(?<![\d-])\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?!\d)")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}")
IPV4_RE = re.compile(r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\d.])")

# group-capturing patterns: only the captured group is redacted
LAST4_RES = [
    re.compile(r"(?i:last (?:four|4) digits(?: of [^()\d]{0,40})?)\s*\(?(\d{4})\)?"),
    re.compile(r"(?i:ending(?: in)?)\s+(\d{4})\b"),
    re.compile(r"(?i:social security number)\s*\((\d{4})\)"),
]


def _date_spans(text: str, today_year: int, date_mode: str = "all"):
    """date_mode: 'none' | 'dob' (only birth dates) | 'all' (every date -> entity DATE).
    Tolerates OCR glue such as '14Mar2021' and '02Sep 2022'."""
    if date_mode == "none":
        return
    M = r"(?i:" + MONTHS + r")"
    pats = [
        rf"(?<!\d)\d{{1,2}}(?:st|nd|rd|th)?\s*{M}\.?,?\s*(\d{{4}})(?!\d)",
        rf"(?<![A-Za-z]){M}\.?\s*\d{{1,2}}(?:st|nd|rd|th)?,?\s*(\d{{4}})(?!\d)",
        r"(?<![\d.])\d{1,2}[/.-]\d{1,2}[/.-](\d{4})(?!\d)",
        r"(?<!\d)(\d{4})-\d{2}-\d{2}(?!\d)",
    ]
    if date_mode == "all":
        pats.append(rf"(?<![A-Za-z]){M}\.?\s*(\d{{4}})(?!\d)")        # 'March 2026'
    for p in pats:
        for m in re.finditer(p, text):
            year = int(m.group(1))
            ctx = bool(re.search(CTX["DATE_OF_BIRTH"], _window(text, m.start(), m.end(), 25, 0)))
            plausible = 1930 <= year <= today_year - 16
            if ctx:
                yield Span(m.start(), m.end(), "DATE_OF_BIRTH", 0.95, "dob_ctx")
            elif plausible and date_mode != "none":
                yield Span(m.start(), m.end(), "DATE_OF_BIRTH", 0.60, "dob_year")
            if date_mode == "all" and 1900 <= year <= 2100:
                yield Span(m.start(), m.end(), "DATE", 0.70, "date_re")


# --------------------------------------------------------------------------- #
# Phone numbers: regex candidates + libphonenumber validation (OCR tolerant)
# --------------------------------------------------------------------------- #

_PHONE_PLUS = re.compile(r"(?<![\w+])\+\d[\d\s().\-]{7,19}\d(?!\d)")
_PHONE_PAREN = re.compile(r"(?<![\d])\(\d{3}\)\s*\d{3}[\s.-]?\d{4}(?!\d)")   # (212)555-0108 / (212) 555-0108
_PHONE_US = re.compile(r"(?<![\d-])\d{3}[\s.-]\d{3}[\s.-]\d{4}(?![\d-])")
_MONTH_AHEAD = re.compile(r"\s*(?i:" + MONTHS + r")")


def _best_phone_prefix(cand: str):
    """Longest prefix of `cand` that libphonenumber accepts as VALID (else POSSIBLE).
    Stops a greedy regex from swallowing the day number of a following date."""
    ends = [i + 1 for i, c in enumerate(cand) if c.isdigit()]
    for strict in (True, False):
        for e in reversed(ends):
            try:
                n = _pn.parse(cand[:e], None)
            except Exception:
                continue
            if (_pn.is_valid_number(n) if strict else _pn.is_possible_number(n)):
                return cand[:e], ("valid" if strict else "possible")
    return None, None


def validate_phone(raw: str, region: Optional[str] = None) -> str:
    """-> 'valid' | 'possible' | 'invalid'  (needs `phonenumbers`; otherwise a digit-count check)."""
    digits = re.sub(r"\D", "", raw)
    if _pn is None:
        return "possible" if 9 <= len(digits) <= 15 else "invalid"
    try:
        n = _pn.parse(raw, None if raw.strip().startswith("+") else (region or "US"))
    except Exception:
        return "invalid"
    if _pn.is_valid_number(n):
        return "valid"
    return "possible" if _pn.is_possible_number(n) else "invalid"


def detect_phones(text: str) -> list[Span]:
    out: list[Span] = []
    for m in _PHONE_PLUS.finditer(text):
        raw, end = m.group(), m.end()
        ndig = len(re.sub(r"\D", "", raw))
        if _pn is not None:
            pref, kind = _best_phone_prefix(raw)
            if pref is None:
                if 9 <= ndig <= 15:
                    out.append(Span(m.start(), end, "PHONE", 0.6, "phone_plus_unvalidated"))
                continue
            end = m.start() + len(pref)
            tail = re.search(r"\s\d{1,2}$", pref)                 # '...0177 07 Jun 2024' -> drop '07'
            if tail and _MONTH_AHEAD.match(text, end):
                end -= len(tail.group())
            out.append(Span(m.start(), end, "PHONE", 0.95 if kind == "valid" else 0.85, "phone_" + kind))
        elif 9 <= ndig <= 15:
            out.append(Span(m.start(), end, "PHONE", 0.85, "phone_plus"))
    for rx in (_PHONE_PAREN, _PHONE_US):
        for m in rx.finditer(text):
            v = validate_phone(m.group(), "US")
            if v != "invalid":
                out.append(Span(m.start(), m.end(), "PHONE", 0.8 if v == "valid" else 0.75, "phone_us_" + v))
    return out


def detect_structured(text: str, today_year: Optional[int] = None, date_mode: str = "all") -> list[Span]:
    today_year = today_year or datetime.date.today().year
    out: list[Span] = []

    def add(s, e, ent, sc, src):
        if sc > 0:
            out.append(Span(s, e, ent, round(sc, 2), src))

    for m in EMAIL_RE.finditer(text):
        add(m.start(), m.end(), "EMAIL", 0.99, "email_re")

    for m in re.finditer(r"\b(?:EMP|DIR)-\d{4,6}\b|\bMER-[A-Z]{2}-\d{3,5}\b", text):
        add(m.start(), m.end(), "EMPLOYEE_ID", 0.97, "empid_re")

    for m in re.finditer(r"(?<![\d-])\d{3}-\d{2}-\d{4}(?![\d-])", text):
        ok, base = ssn_ok(m.group())
        if ok:
            add(m.start(), m.end(), "US_SSN", _boost(text, m.start(), m.end(), "US_SSN", base), "ssn_re")

    for m in re.finditer(r"\b[A-Z]{5}\d{4}[A-Z]\b", text):
        if pan_ok(m.group()):
            add(m.start(), m.end(), "IN_PAN", _boost(text, m.start(), m.end(), "IN_PAN", 0.8), "pan_re")

    # passports: letters+digits are too generic -> require the word 'passport'
    for m in re.finditer(r"\b[A-Z]{1,2}\d{6,9}\b", text):
        sc = _boost(text, m.start(), m.end(), "PASSPORT", 0.4, require=True)
        add(m.start(), m.end(), "PASSPORT", sc, "passport_ctx")

    for m in re.finditer(r"\b\d{6}[xX*]{5}\b", text):  # masked PESEL still reveals DOB
        add(m.start(), m.end(), "PL_PESEL", 0.85, "pesel_masked")
    for m in re.finditer(r"(?<!\d)\d{11}(?!\d)", text):
        if pesel_ok(m.group()):
            add(m.start(), m.end(), "PL_PESEL", _boost(text, m.start(), m.end(), "PL_PESEL", 0.6), "pesel_chk")

    for m in re.finditer(r"(?<!\d)\d{4}[\s-]?\d{4}[\s-]?\d{4}(?!\d)", text):
        if verhoeff_ok(m.group()):           # ~10% of random 12-digit strings pass: demand context
            sc = _boost(text, m.start(), m.end(), "IN_AADHAAR", 0.45)
            add(m.start(), m.end(), "IN_AADHAAR", sc if sc > 0.5 else 0.0, "aadhaar_chk")

    for m in re.finditer(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)", text):
        if luhn_ok(m.group()):               # same story for Luhn: need card wording or 4-4-4-4 grouping
            ctx = re.search(r"card|visa|mastercard|amex|credit|debit|pan\b", _window(text, m.start(), m.end()))
            grouped = re.fullmatch(r"\d{4}([ -])\d{4}\1\d{4}\1\d{1,4}", m.group().strip())
            add(m.start(), m.end(), "CREDIT_CARD", 0.9 if (ctx or grouped) else 0.0, "luhn")

    for m in re.finditer(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{2,4}){3,8}\b", text):
        if iban_ok(m.group()):
            add(m.start(), m.end(), "IBAN", 0.95, "iban_mod97")

    for m in IPV4_RE.finditer(text):
        add(m.start(), m.end(), "IP_ADDRESS", 0.8, "ipv4_re")

    out.extend(detect_phones(text))

    for rx in LAST4_RES:  # "card ending 2218", "last four digits ... (8827)"
        for m in rx.finditer(text):
            add(m.start(1), m.end(1), "PARTIAL_ACCOUNT_ID", 0.8, "last4_ctx")

    for m in ADDRESS_RE.finditer(text):
        add(m.start(), m.end(), "ADDRESS", 0.85, "address_re")

    out.extend(_date_spans(text, today_year, date_mode))
    return out


# --------------------------------------------------------------------------- #
# Person names: no-model heuristics + document-level registry
# --------------------------------------------------------------------------- #

NAME_TOKEN = r"[A-Z][a-zà-ÿ]+(?:['’\-][A-Z]?[a-zà-ÿ]+)*"
INITIAL = r"[A-Z]\.(?:-[A-Z]\.)?"
ID_AFTER = r"\s*\((?:EMP|DIR|MER)-"

# words that look like name tokens but are roles / org words
STOP = {w.lower() for w in """
Group Chief Risk Officer Head Enterprise Framework Committee Board Director Manager Financial Services
Limited Ltd Inc Policy Internal Audit Auditor Compliance Privacy Security Information Operations
Operational Technology Divisional Third Party Executive Identity Access Management Engineer Analyst
Secretary Counsel General Legal Reviewer Standing Cadence Halloran Pike Meridian Staffing Northwind
Analytics Veritrace Screening Document Appendix Section Level Critical High Medium Low Elevated
Moderate Policy Version Contingent Asset Consumer Banking Investment Capital Markets Functions Global
Review Approved Effective Classification Training Notice Figure Table Page Dear Thank
Chair Chairman Chaired Attendee Deputy Alternate Primary Lead Sponsor Independent Non Mandatory Responsible
""".split()}

SALUTATION_RE = re.compile(rf"\b(?:Mr|Mrs|Ms|Miss|Mx|Dr|Prof)\.?\s+({NAME_TOKEN}(?:\s+{NAME_TOKEN})?)")
RELATION_RE = re.compile(
    rf"\b(?:husband|wife|spouse|partner|son|daughter|mother|father|brother|sister|"
    rf"administrator|attention|contact|nominee)\b\s*,?\s*(?:of [^,]{{0,30}},\s*)?"
    rf"({NAME_TOKEN}(?:\s+(?:{INITIAL}|{NAME_TOKEN})){{1,2}})")
NAME_BEFORE_ID_RE = re.compile(rf"((?:{NAME_TOKEN}|{INITIAL})(?:\s+(?:{NAME_TOKEN}|{INITIAL})){{1,3}}){ID_AFTER}")
INITIALS_NAME_RE = re.compile(rf"(?<![A-Z.])((?:{NAME_TOKEN}\s*)?(?:{INITIAL}\s*){{1,3}}{NAME_TOKEN})")
FIRSTLAST_RE = re.compile(rf"\b({NAME_TOKEN})\s+(?:({INITIAL})\s*)?({NAME_TOKEN})\b")


_TOK_RE = re.compile(rf"{INITIAL}|{NAME_TOKEN}")


def name_parts(span_text: str) -> list[str]:
    """'AnthonyP.Brightwell' -> ['Anthony', 'P.', 'Brightwell'] (works with or without spaces)."""
    return [m.group() for m in _TOK_RE.finditer(span_text)]


def _clean_name(span_text: str, start: int, min_tokens: int = 2):
    """Trim stop-words from both ends; reject if stop-words remain inside."""
    toks = [(m.group(), start + m.start(), start + m.end()) for m in _TOK_RE.finditer(span_text)]
    low = lambda t: t.lower().strip(".,")
    while toks and low(toks[0][0]) in STOP:
        toks.pop(0)
    while toks and low(toks[-1][0]) in STOP:
        toks.pop()
    if len(toks) < min_tokens or any(low(t) in STOP for t, _, _ in toks):
        return None
    return toks[0][1], toks[-1][2]


class FirstNameChecker:
    """Is this token a given name?  Sources (all optional, combined):
       - gender_guesser  (offline list of ~40k first names)
       - names_dataset   (pip install names-dataset; large) if use_names_dataset=True
       - names learned from this document (NameRegistry.firsts)
       - your own list: extra={'devansh','ananya',...} or a text file (one name per line)."""

    def __init__(self, extra: Optional[Iterable[str]] = None, names_file: Optional[str] = None,
                 use_names_dataset: bool = False):
        self.extra = {n.lower() for n in (extra or [])}
        if names_file:
            self.extra |= {l.strip().lower() for l in open(names_file, encoding="utf-8") if l.strip()}
        self._gg = _gg.Detector(case_sensitive=False) if _gg else None
        self._nd = None
        if use_names_dataset:
            from names_dataset import NameDataset
            self._nd = NameDataset()
        self.learned: set[str] = set()

    def __contains__(self, lname: str) -> bool:
        n = lname.lower()
        if len(n) < 3 or n in STOP:
            return False
        if n in self.extra or n in self.learned:
            return True
        if self._gg and self._gg.get_gender(n.capitalize()) != "unknown":
            return True
        if self._nd:
            r = self._nd.search(n.capitalize()).get("first_name")
            return bool(r and r.get("rank"))
        return False


# words that are capitalised in headings/tables and must never be taken as a surname
COMMON_CAPS = {w.lower() for w in """Period Rating Report Status Review Owner Summary Date Details Detail Limit Action
Actions Notes Note Page Table Figure Open Closed Pending Approved Version Event Events Loss Data Account
Accounts Control Controls Issue Issues Incident Incidents Service Services System Systems Process
Regulator Contact Position Outcome Outcomes Matter Measure Target Scope Type Level Band Score Policy""".split()}


def detect_names_strong(text: str, first_names: Optional[set[str]] = None) -> list[Span]:
    """High-precision person patterns. `first_names` = optional lowercase gazetteer
    (e.g. from the `names-dataset` package) to catch 'Helena Vorstermann'-style names."""
    out = []
    for rx, grp, score, src in (
        (NAME_BEFORE_ID_RE, 1, 0.95, "name_before_id"),
        (INITIALS_NAME_RE, 1, 0.88, "name_initials"),
        (SALUTATION_RE, 1, 0.90, "name_salutation"),
        (RELATION_RE, 1, 0.85, "name_relation"),
    ):
        for m in rx.finditer(text):
            r = _clean_name(m.group(grp), m.start(grp), 1 if src == "name_salutation" else 2)
            if r:
                out.append(Span(r[0], r[1], "PERSON", score, src))
    if first_names:
        for m in FIRSTLAST_RE.finditer(text):
            sur = m.group(3).lower()
            if m.group(1).lower() in first_names and sur not in STOP and sur not in COMMON_CAPS and len(sur) >= 3:
                out.append(Span(m.start(), m.end(), "PERSON", 0.75, "name_gazetteer"))
        # OCR glued 'HelenaVorstermann'
        for m in re.finditer(r"(?<![A-Za-z])([A-Z][a-z]{2,})([A-Z][a-z]{3,})(?![a-z])", text):
            if m.group(1).lower() in first_names and m.group(2).lower() not in STOP | COMMON_CAPS:
                out.append(Span(m.start(), m.end(), "PERSON", 0.65, "name_gazetteer_glued"))
    return out


def _ratio(a: str, b: str) -> float:
    return _fuzz.ratio(a, b) if _fuzz else 100 * difflib.SequenceMatcher(None, a, b).ratio()


class NameRegistry:
    """Document-level memory: surname -> first-name initials, plus full first names seen."""

    def __init__(self):
        self.surnames: dict[str, set[str]] = {}
        self.firsts: set[str] = set()
        self._rx: dict[str, re.Pattern] = {}

    def add(self, surname: str, initial: Optional[str] = None):
        s = re.sub(r"[^a-z]", "", surname.lower())
        if len(s) < 4 or s in STOP:
            return
        self.surnames.setdefault(s, set())
        if initial:
            self.surnames[s].add(initial.lower())

    def add_first(self, name: str):
        n = name.strip(".")
        if len(n) >= 4 and n.lower() not in STOP and n.lower() not in COMMON_CAPS:
            self.firsts.add(n)

    def pattern(self, s: str) -> re.Pattern:
        if s not in self._rx:
            body = "-?".join(re.escape(c) for c in s[1:])  # tolerate Osei-Tutu vs oseitutu
            self._rx[s] = re.compile(rf"\b[{s[0].upper()}](?i:{'-?' if body else ''}{body})(?:-{NAME_TOKEN})?\b")
        return self._rx[s]


def build_name_registry(texts: Iterable[str], first_names=None) -> NameRegistry:
    """PASS 1: learn people from e-mail local parts (h.vorstermann@...) and strong name patterns.
    `first_names` may be a FirstNameChecker: names learned here are added to it (Devansh, Ananya...)."""
    reg = NameRegistry()
    for t in texts:
        for m in EMAIL_RE.finditer(t):
            mm = re.fullmatch(r"([a-z])\.([a-z][a-z'\-]+)", m.group().split("@")[0])
            if mm:
                reg.add(mm.group(2), mm.group(1))
        for sp in detect_names_strong(t, first_names):
            parts = name_parts(t[sp.start:sp.end])
            if len(parts) >= 2:
                reg.add(parts[-1], parts[0][0])
                if not parts[0].endswith("."):
                    reg.add_first(parts[0])
                    if isinstance(first_names, FirstNameChecker):
                        first_names.learned.add(parts[0].lower())
    return reg


def _extend_left(text: str, start: int, initials: set[str]):
    """Include a first name / initials sitting directly left of a surname. Returns (new_start, found)."""
    pre = text[max(0, start - 60): start]
    lm = re.search(rf"(?:({NAME_TOKEN})\s*)?(?:{INITIAL}\s*)*$", pre)
    if not lm or not lm.group():
        return start, False
    first = lm.group(1)
    if first and (first.lower() in STOP or (initials and first[0].lower() not in initials)):
        cut = lm.start() + len(first)
        while cut < len(pre) and pre[cut].isspace():
            cut += 1
        return start - (len(pre) - cut), cut < len(pre)
    return start - (len(pre) - lm.start()), True


def propagate_names(text: str, reg: NameRegistry, fuzzy: bool = True) -> list[Span]:
    """PASS 2: every other occurrence of a known person: surname alone, 'Helena Vorstermann',
    'T. Lindqvist', OCR-damaged surnames ('Ouarshie' ~ 'Quarshie'), bare first names seen with a surname."""
    out, covered = [], []
    for s, initials in reg.surnames.items():
        for m in reg.pattern(s).finditer(text):
            st, found = _extend_left(text, m.start(), initials)
            out.append(Span(st, m.end(), "PERSON", 0.85 if found else 0.70, "name_registry"))
            covered.append((m.start(), m.end()))
    if fuzzy and reg.surnames:
        for m in re.finditer(r"[A-Za-z][A-Za-z'\-]{5,}", text):
            tok = m.group()
            low = re.sub(r"[^a-z]", "", tok.lower())
            if (not tok[0].isupper() or low in reg.surnames or low in STOP or low in COMMON_CAPS
                    or any(a <= m.start() < b for a, b in covered)):
                continue
            best = max(((s, _ratio(low, s)) for s in reg.surnames if abs(len(s) - len(low)) <= 1 and len(s) >= 6),
                       key=lambda x: x[1], default=None)
            if best and best[1] >= 86:
                st, found = _extend_left(text, m.start(), reg.surnames[best[0]])
                out.append(Span(st, m.end(), "PERSON", 0.80 if found else 0.68, "name_fuzzy"))
    for f in reg.firsts:
        for m in re.finditer(rf"(?<![A-Za-z]){re.escape(f)}(?![a-z])", text):
            out.append(Span(m.start(), m.end(), "PERSON", 0.65, "first_name_seen"))
    return out


# --------------------------------------------------------------------------- #
# Combine, de-duplicate, filter
# --------------------------------------------------------------------------- #

DEFAULT_ALLOWLIST = [
    "245 Hudson Street",          # corporate HQ - not personal data
    "Cadence Financial Group",
]


def resolve_overlaps(spans: Iterable[Span], min_score: float = 0.5) -> list[Span]:
    """Greedy: highest score wins, then longest. Result sorted by position."""
    keep: list[Span] = []
    for sp in sorted(spans, key=lambda s: (-s.score, -(s.end - s.start))):
        if sp.score < min_score:
            continue
        if all(sp.end <= k.start or sp.start >= k.end for k in keep):
            keep.append(sp)
    return sorted(keep, key=lambda s: s.start)


def apply_allowlist(text: str, spans: list[Span], allow: Iterable[str] = DEFAULT_ALLOWLIST) -> list[Span]:
    al = [a.lower() for a in allow]
    return [s for s in spans if not any(a in text[s.start:s.end].lower() for a in al)]


def merge_detections(mine: list[Span], other: list[Span], min_score: float = 0.5) -> list[Span]:
    """Merge with Presidio/BERT output (convert theirs to Span first)."""
    return resolve_overlaps([*mine, *other], min_score)


def detect_pii(text: str, registry: Optional[NameRegistry] = None,
               first_names=None, min_score: float = 0.5,
               allowlist: Iterable[str] = DEFAULT_ALLOWLIST,
               today_year: Optional[int] = None, date_mode: str = "all") -> list[Span]:
    """first_names: a FirstNameChecker (recommended) or a set of lowercase names.
       date_mode : 'all' (every date), 'dob' (birth dates only), 'none'."""
    spans = detect_structured(text, today_year, date_mode)
    spans += detect_names_strong(text, first_names)
    if registry:
        spans += propagate_names(text, registry)
    spans = resolve_overlaps(spans, min_score)
    return apply_allowlist(text, spans, allowlist)


# --------------------------------------------------------------------------- #
# Char spans -> word boxes (for your JSON: [{"text":..., "bbox":[x0,y0,x1,y1]}, ...])
# --------------------------------------------------------------------------- #

def build_offsets(words: list[dict], key: str = "text"):
    """Join word texts with single spaces; remember each word's char range."""
    pos, offs, parts = 0, [], []
    for w in words:
        t = w[key]
        offs.append((pos, pos + len(t)))
        parts.append(t)
        pos += len(t) + 1
    return " ".join(parts), offs


def spans_to_word_hits(spans: list[Span], offs: list[tuple[int, int]]):
    """-> [(word_index, span, frac_start, frac_end)]; fractions allow partial-word boxes
    (e.g. only '8827' inside '(8827)')."""
    hits = []
    for sp in spans:
        for i, (a, b) in enumerate(offs):
            if a < sp.end and b > sp.start:
                n = max(1, b - a)
                hits.append((i, sp, (max(sp.start, a) - a) / n, (min(sp.end, b) - a) / n))
    return hits


def sub_bbox(bbox, f0: float, f1: float, pad: float = 0.5):
    """Shrink a word bbox horizontally to a character fraction range (approximate)."""
    x0, y0, x1, y1 = bbox
    w = x1 - x0
    return [x0 + w * f0 - pad, y0 - pad, x0 + w * f1 + pad, y1 + pad]


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    samples = [
        "Policy owner: Miriam T. Achebe, Group Chief Risk Officer (EMP-38104) m.achebe@cadencefg.com | +1 (212) 555-0147 | mobile +1 (917) 555-0143",
        "Author: Tobias Lindqvist, Head of Enterprise Risk Framework (EMP-40522), t.lindqvist@cadencefg.com, +44 20 7555 0171",
        "Interpretation questions are directed to Helena Vorstermann and L.-E. Brannstrom or T. Lindqvist.",
        "contact Sofia Marchetti-Lang on personal mobile +1 (917) 555-0146 (home address 88 Willowbank Court, Hoboken, NJ 07030).",
        "Priyanka Raghavan (EMP-40318, DOB 14 July 1991), Adaeze Onwuka (EMP-40766, SSN 900-55-2043), Hiroshi Yamagata (EMP-41310, passport TK4820913) and Rohan Deshmukh (EMP-41488, PAN AQBPR7741L). SSN 900-42-7781, 1180 Ashwood Lane, Apartment 4C, Jersey City, NJ 07302.",
        "Dear Ms Prieto-Ramos, ... card ending 2218 ... replacement card to your home at 2214 Palmetto Crescent, Coral Gables, FL 33134. husband, Emilio Ramos-Vega, is a joint holder; savings ending 7741; mobile (305) 555-0143; last four digits of your social security number (8827) and your date of birth, 3 August 1996.",
        "PESEL 890412xxxxx +48 512 555 0180 | Cadence Financial Group, Inc. | 245 Hudson Street, New York, NY 10013",
        "Approved on 18 June 2026; version 1.0 dated 1 July 2018; 1,840 items; USD 2.4m; ISS-2026-0112; ip=10.20.4.17",
    ]
    reg = build_name_registry(samples)
    for s in samples:
        print("-" * 70)
        for sp in detect_pii(s, reg, today_year=2026):
            print(f"{sp.entity:18} {sp.score:.2f} {sp.source:15} {s[sp.start:sp.end]!r}")
