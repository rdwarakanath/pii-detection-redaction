"""Run from the repository root:  python -m pytest First_pipeline/firstpp_fixed/processing/test_wordbox.py -q
(the Tesseract test is skipped when the binary / pytesseract is missing)."""
import shutil
import numpy as np
import pytest
from . import wordbox as wb

L = wb.LocWord


def _line(words, y=0, key=0, w=50, gap=5):
    out, x = [], 0
    for t in words:
        out.append(L(t, (x, y, x + w, y + 10), key)); x += w + gap
    return out


def test_misread_glued_and_split():
    ref = "Miriam Achebe, EMP-38104 Documentcontrol Version 6.0"
    loc = _line(["Miriam", "Achehe,", "EMP-3B104", "Document", "control"]) + [L("Version6.0", (400, 0, 500, 10), 0)]
    toks, unclaimed = wb.align(ref, loc)
    st = {t["text"]: t for t in toks}
    assert st["EMP-38104"]["status"] == "ok"                       # one wrong char does not lose the box
    assert len(st["Documentcontrol"]["pieces"]) == 1 and st["Documentcontrol"]["status"] == "ok"
    assert st["Version"]["status"] == st["6.0"]["status"] == "split"
    v, n = st["Version"]["pieces"][0][1], st["6.0"]["pieces"][0][1]
    assert v[2] <= n[0] + 1e-6 and v[0] == 400 and n[2] == 500     # slices tile the locator word
    assert unclaimed == []


def test_wrapped_token_gives_one_box_per_line():
    toks, _ = wb.align("Achebe now", [L("Ache-", (0, 0, 40, 10), 0), L("be", (0, 20, 15, 30), 1), L("now", (20, 20, 50, 30), 1)])
    assert len(toks[0]["pieces"]) == 2


def test_failsafe_uses_line_box_never_drops():
    txt, words, cnt = wb.words_for_block(
        block_idx=3, text="call +1 (212) 555-0147 now", bbox=(0, 0, 500, 100),
        line_boxes=[{"text": "call +1 (212) 555-0147 now", "bbox_px": [10, 20, 400, 50]}], loc=[], to_pdf=lambda *b: b, native=False)
    assert cnt["fallback"] == cnt["tokens"] > 0
    w = [x for x in words if x["text"] == "555-0147"][0]
    assert w["status"] == "fallback-line" and w["bbox_px"] == [10, 20, 400, 50] and w["est_bbox_px"]
    assert txt[w["start"]:w["end"]] == "555-0147"


def test_locator_only_words_are_kept_and_flagged():
    txt, words, _ = wb.words_for_block(block_idx=0, text="hello world", bbox=(0, 0, 9, 9), line_boxes=[],
                                       loc=_line(["hello", "world", "SECRET123"]), to_pdf=lambda *b: b, native=False)
    extra = [w for w in words if w["status"] == "locator-only"]
    assert [w["text"] for w in extra] == ["SECRET123"] and txt[extra[0]["start"]:extra[0]["end"]] == "SECRET123"


def test_large_block_uses_anchored_path_and_stays_correct():
    ref = " ".join(["Policy owner Miriam Achebe EMP-38104 contact"] * 400)
    loc = []
    for k in range(400):
        loc += [L(t, (i * 60, k * 20, i * 60 + 50, k * 20 + 10), k) for i, t in
                enumerate("Policy owner Miriam Achebe EMP-38104 contact".split())]
    toks, _ = wb.align(ref, loc)
    assert sum(t["status"] == "ok" for t in toks) == len(toks)


def test_native_page_words_offsets():
    pymupdf = pytest.importorskip("pymupdf")
    d = pymupdf.open(); p = d.new_page(); p.insert_text((50, 100), "Miriam Achebe EMP-38104", fontsize=12)
    text, words = wb.native_page_words(p.get_text("words"))
    assert [text[w["start"]:w["end"]] for w in words] == ["Miriam", "Achebe", "EMP-38104"]
    assert all(w["status"] == "exact" and w["bbox_pdf"] for w in words)


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract binary missing")
def test_locator_with_real_tesseract():
    PIL = pytest.importorskip("PIL.ImageFont"); from PIL import Image, ImageDraw, ImageFont
    from .config import Config
    from .ocr_engine import Runner
    font = ImageFont.load_default(size=34)
    im = Image.new("L", (1100, 120), 255); ImageDraw.Draw(im).text((30, 30), "Contact m.achebe@cadencefg.com today", font=font, fill=0)
    cfg = Config(); loc = wb.locate_block(np.array(im), (10, 10, 1090, 110), Runner(cfg), cfg, label="text", n_lines=1)
    _, words, cnt = wb.words_for_block(block_idx=0, text="Contact m.achebe@cadencefg.com today", bbox=(10, 10, 1090, 110),
                                       line_boxes=[], loc=loc, to_pdf=lambda *b: b, native=False)
    assert cnt["fallback"] == 0 and cnt["weak"] == 0
