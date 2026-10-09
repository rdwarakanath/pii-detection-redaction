"""
Cadence PII Guard - Case Study 2 pipeline
File -> detect type -> extract text (OCR if needed) -> detect PII -> classify
     -> normalize -> detect and classify PII -> redact -> report + sanitized text

Setup:
  pip install pymupdf python-pptx python-docx pytesseract pillow gradio spacy
  python3 -m spacy download en_core_web_sm
  apt-get install tesseract-ocr          (Colab/Linux)
Needs pii_local.py in the same folder. Detection and redaction are local; no LLM/API calls.
Run:  python pipeline.py file.pdf     (CLI)   |   python pipeline.py   (Gradio demo)
"""
import collections, hashlib, io, json, os, re, sys
import unicodedata
from pathlib import Path
import pii_local as eng

MIN_CONF = 0.5          # recall-first: anything >= 0.5 is redacted

# ================= 1. FILE INGESTION + EXTRACTION =================
def detect_type(path):
    ext = Path(path).suffix.lower()
    head = open(path, "rb").read(8)
    if head.startswith(b"%PDF"): return "pdf"
    if head.startswith(b"PK"): return {".docx": "docx", ".pptx": "pptx"}.get(ext, "unknown")
    if ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"): return "image"
    if ext in (".txt", ".md", ".csv"): return "text"
    return "unknown"

def ocr_image(img):
    import pytesseract
    from PIL import ImageOps
    return pytesseract.image_to_string(ImageOps.autocontrast(ImageOps.grayscale(img)))

def extract_pdf(path):
    import fitz
    from PIL import Image
    segs = []
    for i, page in enumerate(fitz.open(path), 1):
        blocks = sorted(page.get_text("blocks"), key=lambda b: (round(b[1]), b[0]))  # reading order
        text = "\n".join(b[4].strip() for b in blocks if b[6] == 0 and b[4].strip())
        method = "pdf_text"
        if len(text) < 30:                       # scanned page -> OCR
            pix = page.get_pixmap(dpi=300)
            text = ocr_image(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
            method = "ocr"
        segs.append(dict(loc=f"page {i}", method=method, text=text))
    return segs

def _shapes(shapes):
    for sh in shapes:
        if sh.shape_type == 6: yield from _shapes(sh.shapes)     # grouped shapes
        else: yield sh

def extract_pptx(path):
    from pptx import Presentation
    from PIL import Image
    segs = []
    for i, slide in enumerate(Presentation(path).slides, 1):
        parts, method = [], "pptx_text"
        for sh in _shapes(slide.shapes):
            if sh.has_text_frame and sh.text_frame.text.strip():
                parts.append(sh.text_frame.text)
            elif sh.has_table:
                parts += [" | ".join(c.text for c in r.cells) for r in sh.table.rows]
            elif sh.shape_type == 13:            # picture -> OCR
                try: parts.append(ocr_image(Image.open(io.BytesIO(sh.image.blob)))); method = "pptx_text+ocr"
                except Exception: pass
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame.text.strip():
            parts.append("[NOTES] " + slide.notes_slide.notes_text_frame.text)
        segs.append(dict(loc=f"slide {i}", method=method, text="\n".join(parts)))
    return segs

def extract_docx(path):
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    d, parts = Document(path), []
    for child in d.element.body.iterchildren():   # keeps paragraph/table order
        if child.tag.endswith("}p"): parts.append(Paragraph(child, d).text)
        elif child.tag.endswith("}tbl"):
            parts += [" | ".join(c.text for c in r.cells) for r in Table(child, d).rows]
    for s in d.sections:
        parts += [p.text for p in s.header.paragraphs + s.footer.paragraphs if p.text.strip()]
    return [dict(loc="document", method="docx_text", text="\n".join(parts))]

def extract(path):
    kind = detect_type(path)
    if kind == "pdf": return kind, extract_pdf(path)
    if kind == "pptx": return kind, extract_pptx(path)
    if kind == "docx": return kind, extract_docx(path)
    if kind == "image":
        from PIL import Image
        return kind, [dict(loc="image", method="ocr", text=ocr_image(Image.open(path)))]
    if kind == "text": return kind, [dict(loc="text", method="plain", text=open(path, errors="ignore").read())]
    raise ValueError(f"Unsupported file type: {path}")

# ================= 2. DETECTION =================
_D = [[0,1,2,3,4,5,6,7,8,9],[1,2,3,4,0,6,7,8,9,5],[2,3,4,0,1,7,8,9,5,6],[3,4,0,1,2,8,9,5,6,7],[4,0,1,2,3,9,5,6,7,8],
      [5,9,8,7,6,0,4,3,2,1],[6,5,9,8,7,1,0,4,3,2],[7,6,5,9,8,2,1,0,4,3],[8,7,6,5,9,3,2,1,0,4],[9,8,7,6,5,4,3,2,1,0]]
_P = [[0,1,2,3,4,5,6,7,8,9],[1,5,7,6,2,8,3,0,9,4],[5,8,0,3,7,9,6,1,4,2],[8,9,1,6,0,4,3,5,2,7],
      [9,4,5,3,1,2,6,8,7,0],[4,2,8,6,5,7,3,9,0,1],[2,7,9,3,8,0,6,4,1,5],[7,0,4,6,9,1,3,2,5,8]]
def verhoeff(num):
    c = 0
    for i, ch in enumerate(reversed(num)): c = _D[c][_P[i % 8][int(ch)]]
    return c == 0

def mk(src, s, e, type_, conf, method, validation):
    return dict(text=src[s:e], type=type_, confidence=conf, confidence_level=eng.level(conf),
                detection_method=method, validation=validation,
                context=src[max(0, s - 40):e + 40].replace("\n", " "), start=s, end=e)

LABELLED = re.compile(r"(?i)\b(?:govt?\.?|government|national|tax)\s*(?:id|identification|number|no\.?)\s*[:#-]\s*([A-Za-z0-9*-]{4,})")

def extra_regex(text):
    out = []
    for m in re.finditer(r"\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b", text):          # Aadhaar
        if verhoeff(re.sub(r"\s", "", m.group())):
            out.append(mk(text, m.start(), m.end(), "AADHAAR", 0.92, "REGEX", "VERHOEFF_VALID"))
    for m in LABELLED.finditer(text):                                          # "Government ID: XXXX"
        out.append(mk(text, m.start(1), m.end(1), "NATIONAL_ID", 0.9, "REGEX_CONTEXT", "LABELLED_VALUE"))
    return out

def propagate_names(text, ents):
    names = {e["text"] for e in ents if e["type"] == "PERSON"}
    for n in list(names): names.update(t.strip(".,") for t in n.split() if len(t.strip(".,")) > 2)
    return [mk(text, m.start(), m.end(), "PERSON", 0.85, "PROPAGATED", "SAME_NAME_ELSEWHERE")
            for n in names for m in re.finditer(rf"\b{re.escape(n)}\b", text, flags=re.I)]

def resolve(ents):
    ents = [e for e in ents if e["type"] != "UNCONFIRMED" and e["confidence"] >= MIN_CONF]
    ents.sort(key=lambda e: (-(e["end"] - e["start"]), -e["confidence"]))
    kept = []
    for e in ents:
        if not any(e["start"] < k["end"] and e["end"] > k["start"] for k in kept): kept.append(e)
    return sorted(kept, key=lambda e: e["start"])

def detect(text):
    ents = (eng.regex_stage(text) + extra_regex(text) + eng.find_names(text) +
            eng.find_addresses(text) + eng.find_departments_and_locations(text))
    ents = resolve(ents)
    return resolve(ents + propagate_names(text, ents))

def normalize_text(text):
    """Normalize Unicode and line endings while retaining readable text layout."""
    return unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")

# ================= 3. CLASSIFY + REDACT + REPORT =================
CATEGORY = {"PERSON": "PERSON_NAME", "PHONE": "PHONE_NUMBER", "SSN": "GOVERNMENT_ID", "PAN": "GOVERNMENT_ID",
            "PASSPORT_NUMBER": "GOVERNMENT_ID", "PESEL": "GOVERNMENT_ID", "TIN": "GOVERNMENT_ID",
            "AADHAAR": "GOVERNMENT_ID", "NATIONAL_ID": "GOVERNMENT_ID", "CARD_NUMBER": "FINANCIAL_ID",
            "ACCOUNT_NUMBER": "FINANCIAL_ID", "HOME_ADDRESS": "ADDRESS"}
cat = lambda e: CATEGORY.get(e["type"], e["type"])

def redact(text, ents):
    for e in sorted(ents, key=lambda e: -e["start"]):     # right-to-left keeps offsets valid
        text = text[:e["start"]] + f"[REDACTED_{cat(e)}]" + text[e["end"]:]
    return text

def risk(counts):
    total = sum(counts.values())
    if {"GOVERNMENT_ID", "FINANCIAL_ID", "DATE_OF_BIRTH"} & set(counts) or total >= 10: return "HIGH"
    return "MEDIUM" if total >= 3 else "LOW" if total else "NONE"

def process(path, display_name=None):
    kind, segs = extract(path)
    filename = display_name or Path(path).name
    audit, safe, counts = [], [], collections.Counter()
    for seg in segs:
        text = normalize_text(seg["text"])
        ents = detect(text)
        safe.append(f"--- {seg['loc']} ---\n" + redact(text, ents))
        for e in ents:
            counts[cat(e)] += 1
            audit.append(dict(file=filename, location=seg["loc"], extraction=seg["method"],
                              start=e["start"], end=e["end"], category=cat(e), type=e["type"],
                              confidence=e["confidence"], method=e["detection_method"],
                              validation=e["validation"],
                              value_sha256=hashlib.sha256(e["text"].encode()).hexdigest()[:12],  # no raw PII in log
                              masked_preview=e["text"][:1] + "*" * (len(e["text"]) - 1)))
    lines = [f"File: {filename}  ({kind}, {len(segs)} segment(s))", "PII detected:", "-" * 32]
    lines += [f"{k:<16}: {v}" for k, v in counts.most_common()] or ["none"]
    lines += ["-" * 32, f"Risk: {risk(counts)}",
              "Action: PII redacted locally" if counts else "Action: none required"]
    return dict(report="\n".join(lines), redacted_text="\n\n".join(safe), audit=audit)

# ================= 4. DEMO =================
def _ui(f):
    r = process(f if isinstance(f, str) else f.name)
    return r["report"], r["redacted_text"], json.dumps(r["audit"], indent=2)

def launch():
    import gradio as gr
    gr.Interface(_ui, gr.File(label="Upload PDF / PPTX / DOCX / image"),
                 [gr.Textbox(label="PII report", lines=10),
                  gr.Textbox(label="Sanitized document", lines=18),
                  gr.Code(label="Audit log", language="json")],
                 title="Local PII Guard").launch()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        r = process(sys.argv[1]); print(r["report"]); print(); print(r["redacted_text"])
    else:
        launch()
