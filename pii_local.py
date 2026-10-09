"""Offline PII detection using regex, contextual rules, and optional spaCy NER."""
import json, os, re, sys


# ---------- Stage 1: patterns ----------
PATTERNS = {
    "EMAIL": r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b",
    "PHONE": r"(?<!\w)\+?\d[\d\s().-]{8,16}\d(?!\w)",
    "EMPLOYEE_ID": r"\bEMP-\d{4,8}\b",
    "DIRECTOR_ID": r"\bDIR-\d{4,8}\b",
    "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
    "PAN": r"\b[A-Z]{5}\d{4}[A-Z]\b",
    "PASSPORT_NUMBER": r"\b[A-Z]{1,2}\d{6,8}\b",
    "PESEL": r"\b\d{11}\b",
    "CARD_NUMBER": r"\b(?:\d[ -]?){13,19}\b",
    "ACCOUNT_NUMBER": r"\b\d{9,18}\b",
}

# ---------- Stage 2: context keywords ----------
CONTEXT = {
    "EMAIL": [r"e-?mail", r"contact"],
    "PHONE": [r"phone", r"mobile", r"tel", r"call", r"contact"],
    "EMPLOYEE_ID": [r"employee", r"emp"],
    "DIRECTOR_ID": [r"director", r"din"],
    "SSN": [r"ssn", r"social security"],
    "PAN": [r"\bpan\b", r"permanent account"],
    "PASSPORT_NUMBER": [r"passport"],
    "PESEL": [r"pesel"],
    "CARD_NUMBER": [r"card", r"visa", r"mastercard", r"amex"],
    "ACCOUNT_NUMBER": [r"account", r"a/c", r"iban", r"acct"],
}
DOB_RE = r"(?i)\b(?:dob|d\.o\.b|date of birth|born(?: on)?)\b[:\s-]*([^\n,;)]{6,25})"


def has_context(text, start, end, kind, window=50):
    ctx = text[max(0, start - window):end + window].lower()
    return any(re.search(k, ctx) for k in CONTEXT.get(kind, []))


# ---------- Stage 3: validators ----------
def luhn(num):
    d = [int(c) for c in re.sub(r"\D", "", num)][::-1]
    return len(d) >= 13 and sum(d[0::2]) + sum(sum(divmod(x * 2, 10)) for x in d[1::2]) % 10 == 0 \
        if False else (sum(d[0::2]) + sum(sum(divmod(x * 2, 10)) for x in d[1::2])) % 10 == 0


def pesel_ok(n):
    if not re.fullmatch(r"\d{11}", n):
        return False
    w = [1, 3, 7, 9, 1, 3, 7, 9, 1, 3]
    return (10 - sum(int(a) * b for a, b in zip(n, w)) % 10) % 10 == int(n[-1])


def pan_ok(p):
    # 4th char = holder type, 5th = first letter of surname/name
    return bool(re.fullmatch(r"[A-Z]{3}[ABCFGHLJPT][A-Z]\d{4}[A-Z]", p))


def ssn_ok(s):
    a, g, ser = s.split("-")
    return a not in ("000", "666") and g != "00" and ser != "0000"


def validate(kind, val, has_ctx):
    """returns (validation_label, confidence)"""
    if kind == "CARD_NUMBER":
        ok = luhn(val)
        return ("LUHN_VALID" if ok else "LUHN_FAILED"), (0.97 if ok and has_ctx else 0.9 if ok else 0.3)
    if kind == "PESEL":
        ok = pesel_ok(val)
        return ("CHECKSUM_VALID" if ok else "CHECKSUM_FAILED"), (0.97 if ok and has_ctx else 0.75 if ok else 0.2)
    if kind == "PAN":
        ok = pan_ok(val)
        return ("VALID_PAN_STRUCTURE" if ok else "INVALID_STRUCTURE"), (0.97 if ok and has_ctx else 0.7 if ok else 0.2)
    if kind == "SSN":
        ok = ssn_ok(val)
        return ("VALID_SSN_RULES" if ok else "INVALID_SSN_RULES"), (0.97 if ok and has_ctx else 0.8 if ok else 0.2)
    if kind in ("EMAIL", "EMPLOYEE_ID", "DIRECTOR_ID"):
        return "VALID_PATTERN" + ("_AND_CONTEXT" if has_ctx else ""), 0.99 if kind != "EMAIL" else 0.98
    # weak-shape types: require context, otherwise it's just a number
    return ("VALID_PATTERN_AND_CONTEXT", 0.88) if has_ctx else ("PATTERN_ONLY_NO_CONTEXT", 0.35)


def level(c):
    return "HIGH" if c >= 0.9 else "MEDIUM" if c >= 0.7 else "LOW"


def regex_stage(text):
    found, taken = [], []
    for kind, pat in PATTERNS.items():
        for m in re.finditer(pat, text):
            s, e = m.span()
            if any(s < te and e > ts for ts, te in taken):  # overlap with a stronger match
                continue
            ctx = has_context(text, s, e, kind)
            label, conf = validate(kind, m.group(), ctx)
            if conf < 0.3:
                continue
            found.append(dict(
                text=m.group(), type=kind if conf >= 0.7 else "UNCONFIRMED", confidence=conf,
                confidence_level=level(conf), detection_method="REGEX",
                context=text[max(0, s - 40):e + 40].replace("\n", " "),
                validation=label, start=s, end=e))
            if conf >= 0.7:
                taken.append((s, e))
    for m in re.finditer(DOB_RE, text):
        s, e = m.span(1)
        found.append(dict(text=m.group(1).strip(), type="DATE_OF_BIRTH", confidence=0.95,
                          confidence_level="HIGH", detection_method="REGEX_CONTEXT",
                          context=text[max(0, s - 40):e + 40].replace("\n", " "),
                          validation="DOB_KEYWORD_CONTEXT", start=s, end=e))
    return found

# ---------- Offline name / address detection (no LLM) ----------
def ent(src, s, e, type_, conf, method, validation):
    return dict(text=src[s:e], type=type_, confidence=conf, confidence_level=level(conf),
                detection_method=method, validation=validation,
                context=src[max(0, s - 40):e + 40].replace("\n", " "), start=s, end=e)

NAME = r"([A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+){0,2})"
RX_NAME_LABEL = re.compile(r"\b(?i:name|employee|emp|manager|contact person|author|owner|prepared by|approved by|"
                           r"reviewed by|submitted by|requested by|signed by)\b\.?[ \t]*[:\-][ \t]*" + NAME)
RX_HONORIFIC = re.compile(r"\b(?i:mr|mrs|ms|miss|dr|shri|smt|prof)\b\.?[ \t]+" + NAME)
RX_ROLE_NAME = re.compile(r"\b(?i:employee|manager|engineer|developer|analyst|director|intern|colleague|"
                          r"candidate|applicant|staff member|supervisor)\b[ \t]+" + NAME)
# Lowercase names can be accepted when a role cue and a following verb make the
# name position clear (for example, "employee ram working ...").
RX_LOWER_ROLE_NAME = re.compile(
    r"\b(?i:employee|emp|manager|engineer|developer|analyst|director|intern|colleague|"
    r"candidate|applicant|staff member|supervisor)\b[ \t]+([a-z]{2,})"
    r"(?=[ \t]+(?i:working|works|is|belongs|from|in)\b)"
)
STOP = {"Handbook", "Policy", "Report", "Process", "Data", "Details", "Information", "Name", "Id", "Benefits",
        "Records", "Number", "Email", "Phone", "Department", "Team", "Manager", "Employee", "Director", "Dob"}
EMAIL_STOP = {"info", "admin", "support", "hr", "team", "contact", "sales", "noreply", "hello", "mail",
              "office", "help", "careers", "service", "services", "accounts", "billing"}

def find_names(text):
    out = []
    for rx, conf, why in ((RX_NAME_LABEL, 0.9, "LABEL_BEFORE_NAME"), (RX_HONORIFIC, 0.9, "HONORIFIC_BEFORE_NAME"),
                          (RX_ROLE_NAME, 0.7, "ROLE_WORD_BEFORE_NAME")):
        for m in rx.finditer(text):
            words = [w for w in m.group(1).split() if w not in STOP]
            if words:
                s = text.find(words[0], m.start(1)); e = text.find(words[-1], s) + len(words[-1])
                out.append(ent(text, s, e, "PERSON", conf, "RULE", why))
    for m in RX_LOWER_ROLE_NAME.finditer(text):
        out.append(ent(text, m.start(1), m.end(1), "PERSON", 0.7, "ROLE_CUE", "LOWERCASE_NAME_AFTER_ROLE"))
    # names hidden in e-mail addresses: rahul.kumar@x.com -> hide "rahul" / "kumar" everywhere
    for m in re.finditer(PATTERNS["EMAIL"], text):
        for tok in re.split(r"[._+-]+", m.group().split("@")[0]):
            if tok.isalpha() and len(tok) >= 3 and tok.lower() not in EMAIL_STOP:
                out += [ent(text, x.start(), x.end(), "PERSON", 0.65, "EMAIL_DERIVED", "NAME_FROM_EMAIL")
                        for x in re.finditer(rf"\b{re.escape(tok)}\b", text, flags=re.I)]
    return out

_nlp = None
def spacy_entities(text):
    """Run local NER for people and places; map organizations to departments only in context."""
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load(os.getenv("SPACY_MODEL", "en_core_web_sm"))
        except Exception as ex:
            print("spaCy NER unavailable; using contextual rules only:", ex, file=sys.stderr)
            _nlp = False
    if not _nlp:
        return []

    out = []
    for span in _nlp(text).ents:
        if span.label_ == "PERSON":
            type_, validation = "PERSON", "SPACY_PERSON"
        elif span.label_ in ("GPE", "LOC", "FAC"):
            type_, validation = "LOCATION", f"SPACY_{span.label_}"
        elif span.label_ == "ORG":
            nearby = text[max(0, span.start_char - 35):span.end_char + 35]
            if not re.search(r"\b(?:department|dept|division|team|unit)\b", nearby, re.I):
                continue
            type_, validation = "DEPARTMENT", "SPACY_ORG_DEPARTMENT_CONTEXT"
        else:
            continue
        out.append(ent(text, span.start_char, span.end_char, type_, 0.82, "SPACY_NER", validation))
    return out

RX_ADDR = re.compile(r"\b(?i:address|residence|resides at|lives at|home)\b[ \t]*[:\-][ \t]*([^\n]{8,})")
RX_PIN = re.compile(r"\b(?i:pin(?:code)?|zip(?: code)?|postal code)\b[ \t]*[:\-]?[ \t]*(\d{5,6}(?:-\d{4})?)")
RX_DEPARTMENT = re.compile(
    r"\b(?i:in|from|of)[ \t]+([A-Za-z][A-Za-z0-9&./-]*(?:[ \t]+[A-Za-z][A-Za-z0-9&./-]*){0,3})"
    r"[ \t]+(?i:department|dept)\b"
)
RX_LOCATION = re.compile(
    r"\b(?i:belongs to|based in|located in|lives in|resides in|from|city|location)"
    r"[ \t]+([A-Za-z][A-Za-z.-]*(?:[ \t]+[A-Za-z][A-Za-z.-]*){0,2})\b"
)

def find_addresses(text):
    return [ent(text, m.start(1), m.end(1), "HOME_ADDRESS", 0.85, "RULE", "ADDRESS_LABEL")
            for m in RX_ADDR.finditer(text)] + \
           [ent(text, m.start(1), m.end(1), "HOME_ADDRESS", 0.8, "RULE", "POSTAL_CODE_CONTEXT")
            for m in RX_PIN.finditer(text)]

def find_departments_and_locations(text):
    departments = [ent(text, m.start(1), m.end(1), "DEPARTMENT", 0.85, "RULE", "DEPARTMENT_LABEL")
                   for m in RX_DEPARTMENT.finditer(text)]
    locations = [ent(text, m.start(1), m.end(1), "LOCATION", 0.85, "RULE", "LOCATION_CUE")
                 for m in RX_LOCATION.finditer(text)]
    return departments + locations + spacy_entities(text)
