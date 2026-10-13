"""
redact_pdf.py - redact PII in a scanned PDF using your OCR JSON word boxes.

  python First_pipeline/firstpp_fixed/processing/redact_pdf.py \
      in.pdf pages.json out.pdf [--bert] [--figures review|blank]

Pipeline per page
  words_text (JSON)  -> detect_pii (rules + name registry [+ BERT/Presidio])
  char spans         -> JSON words via word['start'/'end'] (no re-joining needed)
  words              -> PDF-point rects (fixing coarse fallback boxes)
  rects              -> merge per line, pad -> PyMuPDF true redaction (image pixels)
  output             -> scrub, save, verify pixels are black, write audit JSON
"""
from __future__ import annotations

import argparse
import json
import re
import sys

import pymupdf as fitz  # PyMuPDF >= 1.24 (import fitz also works)

from pii_helpers import (Span, FirstNameChecker, build_name_registry, detect_pii, apply_allowlist,
                         name_parts, DEFAULT_ALLOWLIST)

PAD = 1.5            # points added around every box
LINE_GAP = 6.0       # merge boxes on one line if gap smaller than this (pt)
FIGURE_LABELS = {"image", "chart"}
MAX_COARSE_AREA = 6000.0   # pt^2; larger "coarse" boxes are reported instead of drawn


# figures whose caption says they are evidence captures (IDs, scans, screenshots, logs) are blanked entirely
EVIDENCE_HINTS = re.compile(r"identity record|credential|badge|attestation|screenshot|configuration|audit log|"
                            r"evidence|scan|signed|photograph", re.I)


_FIG_CAP = re.compile(r"^\s*figure\s*\d+", re.I)


def caption_for(blocks: list[dict], frect: "fitz.Rect") -> str:
    """Text of the nearest 'Figure N - ...' caption directly below a region."""
    best, gap_best = "", 40.0
    for b in blocks:
        t = (b.get("plain_text") or b.get("text") or "")
        if _FIG_CAP.match(t):
            gap = b["bbox_pdf"][1] - frect.y1
            if -5 <= gap < gap_best:
                best, gap_best = t, gap
    return best


def figure_blocks(blocks: list[dict]) -> list[dict]:
    """image/chart blocks, plus any block (e.g. a dark screenshot the layout model called 'table')
    that sits directly above a 'Figure N' caption."""
    out = []
    for b in blocks:
        if b["label"] in FIGURE_LABELS:
            out.append(b)
        elif b["label"] in ("table", "text") and caption_for(blocks, fitz.Rect(b["bbox_pdf"])) \
                and b["bbox_pdf"][3] - b["bbox_pdf"][1] > 60:
            out.append(b)
    return out


# --------------------------------------------------------------------------- #
# 1. word -> rect in PDF points (top-left origin, same as PyMuPDF for rotation 0)
# --------------------------------------------------------------------------- #

def _norm(s: str) -> str:
    return re.sub(r"\W", "", s).lower()


def word_rect(word: dict, block: dict | None, dpi: float):
    """Returns (fitz.Rect, quality) with quality in {'exact','estimated','line','block-line','coarse'}.
    The OCR JSON gives coarse boxes for fallback words (whole line / whole block), which would
    black out far too much (or too little). Narrow them down when possible."""
    status = word["status"]
    r = fitz.Rect(word["bbox_pdf"])
    if status in ("ok", "split", "locator-only"):
        return r, "exact"
    if word.get("est_bbox_px"):                       # proportional estimate inside the line
        x0, y0, x1, y1 = [v * 72.0 / dpi for v in word["est_bbox_px"]]
        return fitz.Rect(x0, y0, x1, y1), "estimated"
    if status == "fallback-line":
        return r, "line"                               # one line: acceptable, flagged
    # fallback-block: bbox is the entire block -> find the line_box that contains the word
    if block:
        w = _norm(word["text"])
        for lb in block.get("line_boxes", []):
            t = _norm(lb["text"])
            i = t.find(w) if w else -1
            if i >= 0:
                x0, y0, x1, y1 = lb["bbox_pdf"]
                n = max(1, len(t))
                return fitz.Rect(x0 + (x1 - x0) * i / n, y0, x0 + (x1 - x0) * (i + len(w)) / n, y1), "block-line"
    return r, "coarse"                                 # fail closed, but must be reviewed


# --------------------------------------------------------------------------- #
# 2. char spans -> rects
# --------------------------------------------------------------------------- #

def page_geometry(words: list[dict], blocks: list[dict], dpi: float):
    """One (Rect, quality) per word. Words with coarse OCR boxes (whole line / whole block)
    are re-positioned by interpolating between the nearest well-located neighbours on the same line."""
    geo = []
    for w in words:
        blk = blocks[w["block"]] if w["block"] < len(blocks) else None
        geo.append(word_rect(w, blk, dpi))
    good = lambda q: q in ("exact", "estimated", "block-line")
    n = len(words)
    i = 0
    while i < n:
        if good(geo[i][1]):
            i += 1
            continue
        j = i
        while j + 1 < n and not good(geo[j + 1][1]):
            j += 1
        L = geo[i - 1][0] if i > 0 else None
        R = geo[j + 1][0] if j + 1 < n else None
        if L is not None and R is not None:
            overlap = min(L.y1, R.y1) - max(L.y0, R.y0)
            if overlap > 0.5 * min(L.height, R.height) and R.x0 > L.x1:
                y0, y1 = min(L.y0, R.y0), max(L.y1, R.y1)
                weights = [len(words[k]["text"]) + 1 for k in range(i, j + 1)]
                tot, x = float(sum(weights)), L.x1
                for k, wt in zip(range(i, j + 1), weights):
                    w_ = (R.x0 - L.x1) * wt / tot
                    geo[k] = (fitz.Rect(x, y0, x + w_, y1), "interpolated")
                    x += w_
        i = j + 1
    return geo


def span_rects(span: Span, words: list[dict], geo: list):
    out = []
    for w, (rect, q) in zip(words, geo):
        if w["start"] < span.end and w["end"] > span.start:
            n = max(1, w["end"] - w["start"])
            f0 = (max(span.start, w["start"]) - w["start"]) / n
            f1 = (min(span.end, w["end"]) - w["start"]) / n
            if (f0 > 0 or f1 < 1) and q in ("exact", "estimated", "interpolated"):
                # OCR often glues tokens ('a.x@y.com,+1'); char-proportional boxes drift, so snap to the
                # word edge when only a couple of characters remain and otherwise widen by ~1 char.
                cw = (rect.x1 - rect.x0) / n
                if f0 * n <= 3:
                    f0 = 0.0
                if (1 - f1) * n <= 3:
                    f1 = 1.0
                x0 = rect.x0 + (rect.x1 - rect.x0) * f0 - (cw * 1.2 if f0 > 0 else 0)
                x1 = rect.x0 + (rect.x1 - rect.x0) * f1 + (cw * 1.2 if f1 < 1 else 0)
                rect = fitz.Rect(max(rect.x0, x0), rect.y0, min(rect.x1, x1), rect.y1)
            out.append((rect, q, w["text"], w["block"]))
    return out


def merge_rects(rects: list[fitz.Rect]) -> list[fitz.Rect]:
    """Merge neighbouring boxes on the same text line into one rectangle."""
    rects = sorted((fitz.Rect(r) for r in rects), key=lambda r: (round(r.y0), r.x0))
    merged: list[fitz.Rect] = []
    for r in rects:
        for i, m in enumerate(merged):
            v_overlap = min(m.y1, r.y1) - max(m.y0, r.y0)
            if v_overlap > 0.5 * min(m.height, r.height) and r.x0 - m.x1 < LINE_GAP and m.x0 - r.x1 < LINE_GAP:
                merged[i] = m | r          # Rect has no in-place |=, so assign back
                break
        else:
            merged.append(fitz.Rect(r))
    return merged


# --------------------------------------------------------------------------- #
# 2b. last-resort localisation: re-OCR only the block that contains the word
# --------------------------------------------------------------------------- #

def ocr_find(page, block_rect, token: str, cache: dict, dpi: int = 300):
    """Tesseract on the block crop; returns rects (PDF pt) of every OCR word matching `token`.
    Over-redacts repeated tokens inside that block, which is the safe direction."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return []
    clip = fitz.Rect(block_rect) & page.rect
    key = (page.number, tuple(round(v) for v in clip))
    if key not in cache:
        pix = page.get_pixmap(dpi=dpi, clip=clip, colorspace=fitz.csGRAY)
        img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
        d = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config="--psm 6")
        k = 72.0 / dpi
        cache[key] = [(_norm(t), fitz.Rect(clip.x0 + l * k, clip.y0 + tp * k, clip.x0 + (l + w) * k, clip.y0 + (tp + h) * k))
                      for t, l, tp, w, h in zip(d["text"], d["left"], d["top"], d["width"], d["height"]) if t.strip()]
    tok = _norm(token)
    if len(tok) < 3:                      # '%', 'a', initials: never match alone (too many false hits)
        return []
    items = cache[key]
    def _near(a, b):                      # tolerate OCR slips: 'Szewezyk' ~ 'Szewczyk'
        if len(a) < 5 or len(b) < 5:
            return False
        if a in b or b in a:
            return True
        try:
            from rapidfuzz import fuzz
            return fuzz.ratio(a, b) >= 82
        except ImportError:
            import difflib
            return difflib.SequenceMatcher(None, a, b).ratio() >= 0.82
    hits = [r for t, r in items if t == tok or _near(tok, t)]
    out = list(hits)
    for h in hits:                        # attach initials sitting just left of a matched surname ('F. L. Nakagawa')
        for t, r in items:
            same_line = min(r.y1, h.y1) - max(r.y0, h.y0) > 0.5 * min(r.height, h.height)
            if len(t) <= 2 and same_line and 0 <= h.x0 - r.x1 < 12 and r not in out:
                out.append(r)
    return out


# --------------------------------------------------------------------------- #
# 3. detection (rules, optionally BERT through Presidio)
# --------------------------------------------------------------------------- #

def detect_all(texts: list[str], use_bert: bool, date_mode: str = "all", extra_first_names=None):
    checker = FirstNameChecker(extra=extra_first_names)
    registry = build_name_registry(texts, checker)
    rules = [detect_pii(t, registry, checker, today_year=2026, allowlist=(), date_mode=date_mode) for t in texts]
    if not use_bert:
        return rules, registry, checker
    from presidio_bert_combo import build_analyzer, bert_spans, merge_spans   # lazy import
    analyzer = build_analyzer()
    bert = [bert_spans(analyzer, t) for t in texts]
    for t, bs in zip(texts, bert):                       # BERT names feed the registry too
        for sp in bs:
            pt = name_parts(t[sp.start:sp.end])
            if len(pt) >= 2:
                registry.add(pt[-1], pt[0][0])
                if not pt[0].endswith("."):
                    registry.add_first(pt[0]); checker.learned.add(pt[0].lower())
    rules = [detect_pii(t, registry, checker, today_year=2026, allowlist=(), date_mode=date_mode) for t in texts]
    return [merge_spans(r_, b) for r_, b in zip(rules, bert)], registry, checker


# --------------------------------------------------------------------------- #
# 3b. figures / screenshots / scans: OCR the crop at high DPI, detect, box the words
# --------------------------------------------------------------------------- #

def ocr_detect(page, clip, registry, checker, date_mode="all", dpi=300, variants=("n6", "n11", "i11")):
    """Tesseract on `clip` of the page as it is NOW, then the same PII detectors.
    Returns [(Rect, entity, source)]. variants: n=normal/i=inverted image, number = --psm."""
    try:
        import pytesseract
        from PIL import Image, ImageOps
    except ImportError:
        return []
    clip = fitz.Rect(clip) & page.rect
    if clip.is_empty:
        return []
    pix = page.get_pixmap(dpi=dpi, clip=clip, colorspace=fitz.csGRAY)
    img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
    inv = None
    k = 72.0 / dpi
    out = []
    for v in variants:
        if v[0] == "i":
            inv = inv or ImageOps.invert(img)
        im = inv if v[0] == "i" else img
        d = pytesseract.image_to_data(im, output_type=pytesseract.Output.DICT, config=f"--psm {v[1:]}")
        words, text, last = [], "", None
        for i, t in enumerate(d["text"]):
            if not t.strip() or float(d["conf"][i]) < 20:
                continue
            key = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
            if last is not None:
                text += "\n" if key != last else " "
            last = key
            words.append((len(text), len(text) + len(t),
                          fitz.Rect(clip.x0 + d["left"][i] * k, clip.y0 + d["top"][i] * k,
                                    clip.x0 + (d["left"][i] + d["width"][i]) * k,
                                    clip.y0 + (d["top"][i] + d["height"][i]) * k)))
            text += t
        for sp in detect_pii(text, registry, checker, today_year=2026, allowlist=DEFAULT_ALLOWLIST, date_mode=date_mode):
            for a, b, rc in words:
                if a < sp.end and b > sp.start:
                    out.append((rc, sp.entity, "ocr_" + v))
    return out


def residual_loop(pdf_out, registry, checker, date_mode, audit, max_rounds=2):
    """Verify-and-fix: re-OCR the OUTPUT, detect PII again, black out whatever is still readable.
    Independent of the JSON geometry, so it catches mis-placed boxes. Returns a log per round."""
    import os
    log = []
    for rnd in range(max_rounds + 1):
        doc = fitz.open(pdf_out)
        found = 0
        for pno, page in enumerate(doc):
            hits = ocr_detect(page, page.rect, registry, checker, date_mode, 300, ("n6", "n11"))
            if not hits or rnd == max_rounds:
                found += len(hits)
                continue
            for rc, ent, src in hits:
                audit["pages"][pno]["items"].append({"entity": ent, "score": 0.8, "source": src,
                                                     "quality": "residual-ocr", "rect": [round(v, 1) for v in rc]})
            for m in merge_rects([h[0] for h in hits]):
                page.add_redact_annot(fitz.Rect(m.x0 - PAD, m.y0 - PAD, m.x1 + PAD, m.y1 + PAD) & page.rect, fill=(0, 0, 0))
            page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_PIXELS, graphics=0, text=0)
            found += len(hits)
        log.append({"round": rnd, "pii_still_readable": found, "applied": rnd < max_rounds and found > 0})
        if found == 0 or rnd == max_rounds:
            doc.close()
            break
        tmp = pdf_out + ".tmp"
        doc.save(tmp, garbage=4, deflate=True, clean=True)
        doc.close()
        os.replace(tmp, pdf_out)
    return log


def figure_pii(page, rect, registry, checker, date_mode="all", dpi=600):
    """Returns [(Rect, entity, source)] found INSIDE an image region (text via Tesseract, faces,
    barcodes/QR). Tries normal + inverted polarity (dark UI themes, terminals)."""
    out = []
    clip = fitz.Rect(rect) & page.rect
    if clip.is_empty:
        return out
    out.extend(ocr_detect(page, clip, registry, checker, date_mode, dpi, ("n11", "n6", "i11")))
    try:                                                # faces + barcodes/QR (needs opencv-python-headless)
        import cv2, numpy as np
        pix = page.get_pixmap(dpi=200, clip=clip, colorspace=fitz.csGRAY)
        g = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
        s = 72.0 / 200
        face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        for (x, y, w, h) in face.detectMultiScale(g, 1.1, 5, minSize=(24, 24)):
            out.append((fitz.Rect(clip.x0 + x * s, clip.y0 + y * s, clip.x0 + (x + w) * s, clip.y0 + (y + h) * s), "FACE", "haar"))
        for det in (cv2.QRCodeDetector(),):
            ok, pts = det.detect(g)
            if ok and pts is not None:
                for q in pts:
                    xs, ys = q[:, 0], q[:, 1]
                    out.append((fitz.Rect(clip.x0 + xs.min() * s, clip.y0 + ys.min() * s, clip.x0 + xs.max() * s, clip.y0 + ys.max() * s), "QR_CODE", "cv2"))
        if hasattr(cv2, "barcode"):
            ok, _, _, pts = cv2.barcode.BarcodeDetector().detectAndDecode(g)
            if pts is not None:
                for q in pts:
                    xs, ys = q[:, 0], q[:, 1]
                    out.append((fitz.Rect(clip.x0 + xs.min() * s, clip.y0 + ys.min() * s, clip.x0 + xs.max() * s, clip.y0 + ys.max() * s), "BARCODE", "cv2"))
    except Exception:
        pass
    return out


# --------------------------------------------------------------------------- #
# 4. main redaction
# --------------------------------------------------------------------------- #

def redact(pdf_in: str, json_in: str, pdf_out: str, use_bert: bool = False,
           figures: str = "hybrid", audit_path: str | None = None, date_mode: str = "all"):
    data = json.load(open(json_in, encoding="utf-8"))
    pages = data["pages"]
    doc = fitz.open(pdf_in)
    assert len(pages) == len(doc), "JSON and PDF page counts differ"

    texts = [p["words_text"] for p in pages]
    all_spans, registry, checker = detect_all(texts, use_bert, date_mode)

    audit = {"pages": [], "review": []}
    ocr_cache: dict = {}
    for p, spans in zip(pages, all_spans):
        page = doc[p["page_no"] - 1]
        assert page.rotation == 0, "rotated page: apply page.derotation_matrix to rects first"
        dpi = p["stats"].get("selected_dpi", 300)
        text = p["words_text"]
        spans = apply_allowlist(text, spans, DEFAULT_ALLOWLIST)

        boxes, entries = [], []
        geo = page_geometry(p["words"], p["blocks"], dpi)
        for sp in spans:
            for rect, q, tok, bidx in span_rects(sp, p["words"], geo):
                # 'line' boxes (no estimate) are often one shared box for every identical token on the page,
                # so they can point at the wrong row. Treat them like 'coarse': re-locate inside the block.
                if q in ("line", "coarse"):
                    blk = p["blocks"][bidx] if bidx < len(p["blocks"]) else None
                    search = fitz.Rect(blk["bbox_pdf"]) if blk else rect
                    found = ocr_find(page, search, tok, ocr_cache)
                    if found:
                        for fr in found:
                            boxes.append(fr)
                            entries.append({"entity": sp.entity, "score": sp.score, "source": sp.source,
                                            "quality": "ocr-relocated", "rect": [round(v, 1) for v in fr]})
                        continue
                    if q == "coarse" and len(_norm(tok)) < 3:
                        continue          # lone initials ('M.'): covered via the surname box (initials are attached)
                    if q == "coarse" and rect.get_area() > MAX_COARSE_AREA:
                        audit.setdefault("unredacted", []).append(
                            {"page": p["page_no"], "entity": sp.entity, "reason": "no usable word box",
                             "block_rect": [round(v, 1) for v in rect]})
                        continue
                boxes.append(rect)
                entries.append({"entity": sp.entity, "score": sp.score, "source": sp.source,
                                "quality": q, "rect": [round(v, 1) for v in rect]})
                if q in ("line", "coarse"):               # coarse geometry -> human check
                    audit["review"].append({"page": p["page_no"], "entity": sp.entity,
                                            "quality": q, "rect": entries[-1]["rect"]})

        # figures: your page OCR often misses text inside screenshots / scans / badges
        for b in figure_blocks(p["blocks"]):
            frect = fitz.Rect(b["bbox_pdf"])
            cap = caption_for(p["blocks"], frect)
            evidence = bool(EVIDENCE_HINTS.search(cap))
            if figures == "blank" or (figures == "hybrid" and evidence):
                boxes.append(frect)
                entries.append({"entity": "FIGURE", "score": 1.0, "source": "figure_blank" + ("_evidence" if evidence else ""),
                                "quality": "block", "rect": [round(v, 1) for v in frect]})
            elif figures in ("ocr", "hybrid"):
                for rc, ent, src in figure_pii(page, frect, registry, checker, date_mode):
                    boxes.append(rc)
                    entries.append({"entity": ent, "score": 0.8, "source": src, "quality": "figure",
                                    "rect": [round(v, 1) for v in rc]})
            else:
                audit["review"].append({"page": p["page_no"], "entity": "FIGURE_UNVERIFIED",
                                        "rect": [round(v, 1) for v in frect]})

        for r in merge_rects(boxes):
            r = fitz.Rect(r.x0 - PAD, r.y0 - PAD, r.x1 + PAD, r.y1 + PAD) & page.rect
            page.add_redact_annot(r, fill=(0, 0, 0))
        # one apply per page: blanks pixels in the scanned image, removes any text under the box,
        # keeps vector graphics (graphics=0)
        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_PIXELS, graphics=0, text=0)
        audit["pages"].append({"page": p["page_no"], "n_boxes": len(entries), "items": entries})

    doc.set_metadata({})
    doc.scrub(metadata=True, xml_metadata=True, attached_files=True, embedded_files=True,
              hidden_text=True, redact_images=0)
    doc.save(pdf_out, garbage=4, deflate=True, clean=True)

    audit["residual_rounds"] = residual_loop(pdf_out, registry, checker, date_mode, audit)
    left = audit["residual_rounds"][-1]["pii_still_readable"]
    problems = verify(pdf_out, audit)
    if left:
        problems.append({"issue": f"{left} PII hits still readable after OCR re-check (see residual_rounds)"})
    for u in audit.get("unredacted", []):
        problems.append({"page": u["page"], "issue": "PII found but could not be located", "entity": u["entity"]})
    audit["verify_problems"] = problems
    json.dump(audit, open(audit_path or pdf_out + ".audit.json", "w"), indent=1)
    return audit


# --------------------------------------------------------------------------- #
# 5. verification (never skip)
# --------------------------------------------------------------------------- #

def verify(pdf_out: str, audit: dict, scale: float = 2.0) -> list[dict]:
    """(a) output has no extractable text; (b) every redaction box is really black pixels."""
    problems = []
    doc = fitz.open(pdf_out)
    for pinfo in audit["pages"]:
        page = doc[pinfo["page"] - 1]
        if page.get_text("text").strip():
            problems.append({"page": pinfo["page"], "issue": "text layer present"})
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csGRAY)
        for it in pinfo["items"]:
            x0, y0, x1, y1 = [int(v * scale) for v in it["rect"]]
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(pix.width, x1), min(pix.height, y1)
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            dark = 0
            total = (x1 - x0) * (y1 - y0)
            for yy in range(y0, y1, 2):
                row = pix.samples[yy * pix.stride + x0: yy * pix.stride + x1]
                dark += sum(1 for v in row[::2] if v < 40) * 4
            if dark / total < 0.85:
                problems.append({"page": pinfo["page"], "issue": "box not fully black",
                                 "entity": it["entity"], "rect": it["rect"]})
    return problems


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf_in"); ap.add_argument("json_in"); ap.add_argument("pdf_out")
    ap.add_argument("--bert", action="store_true", help="add BERT via Presidio (needs model download)")
    ap.add_argument("--figures", choices=["hybrid", "ocr", "blank", "review"], default="hybrid",
                    help="hybrid (default): blank evidence figures (IDs, scans, screenshots, logs), OCR the rest | ocr | blank | review")
    ap.add_argument("--dates", choices=["all", "dob", "none"], default="all")
    a = ap.parse_args()
    res = redact(a.pdf_in, a.json_in, a.pdf_out, a.bert, a.figures, date_mode=a.dates)
    n = sum(p["n_boxes"] for p in res["pages"])
    print(f"{n} boxes | {len(res['review'])} items for review | {len(res['verify_problems'])} verify problems")
    sys.exit(1 if res["verify_problems"] else 0)
