"""Unified text layer: reading order + text / markdown renderers.

This module turns positioned words into Markdown. In the PaddleOCR pipeline its job is the *native* (digital-page) path: for a
page with a trusted text layer, pipeline.native_markdown() hands over PyMuPDF words and ``page_markdown()`` renders them. OCR
pages are rendered block by block in ocr_engine.py, so both paths must produce the same kind of Markdown; this module therefore
shares its paragraph / list / hyphen rules with ``ocr_engine.join_lines`` and infers the same two heading levels
(``#`` document title, ``##`` section title) that PP-StructureV3 labels give the OCR path.

What the native path does (``reading_order`` / ``page_markdown``)
  * works in VISUAL space: PyMuPDF reports word boxes in the *unrotated* page space even when the page is rotated, so boxes are
    mapped through the page rotation before anything is sorted (otherwise a 90-degree page is read sideways);
  * splits a PyMuPDF block where the line height jumps (a heading set tight against its paragraph becomes its own block);
  * orders blocks with an XY-cut: a vertical gutter that no block crosses separates columns (read column by column), full-width
    bands (title, footnote) are read top to bottom; table-like grids of short cells are NOT split into columns and are read row by
    row. When nothing separates the blocks they are read in rows, top to bottom and left to right;
  * keeps list items on their own lines (bullet glyphs become ``- ``), merges wrapped lines into paragraphs, removes hyphenation
    (soft hyphens always, a trailing hyphen before a lower-case word), expands ligatures and drops invisible characters;
  * marks short blocks set in clearly larger type as headings (needs at least 15 words on the page to know the body size).

Words that carry OCR column information (``page.columns > 1`` or a non-zero ``Word.col``) keep the previous behaviour exactly
(group by ``(col, block)``, sort by column then position), so legacy callers are unaffected.

Not done here: tables on digital pages. ``page.tables`` is rendered when present, but the native path supplies none, so grids on
digital pages are rendered row-wise as separate paragraphs. Detecting them needs the PyMuPDF page (``page.find_tables()``) and
therefore belongs in pipeline.native_markdown().
"""
from __future__ import annotations
import re
import statistics
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple
from .models import PageResult, Table, Word


@dataclass
class Block:
    kind: str                          # text | table
    bbox: tuple                        # native path: VISUAL page space (rotation applied); legacy path: as supplied
    lines: List[List[Word]]
    col: int = 0
    table: Optional[Table] = None
    level: int = 0                     # heading level inferred from type size (0 = body text, 1 = "#", 2 = "##")


def _bbox(ws: List[Word]):
    return (min(w.bbox[0] for w in ws), min(w.bbox[1] for w in ws), max(w.bbox[2] for w in ws), max(w.bbox[3] for w in ws))


def rebuild_tables(page: PageResult) -> None:
    """Refresh cell text after post-processing."""
    cells: Dict = {}
    for w in page.words:
        if w.table:
            cells.setdefault(w.table, []).append(w)
    for t in page.tables:
        for c in t.cells:
            ws = cells.get((t.id, c.row, c.col), [])
            ws.sort(key=lambda w: (round(w.bbox[1] / 6), w.bbox[0]))
            c.text = " ".join(w.text for w in ws)


# ====================================================================== text joining (shared rules)
_SOFT_HYPHEN = "\u00ad"
_HYPHENS = "-\u2010\u2011"
_INVISIBLE = dict.fromkeys(map(ord, "\u00ad\u200b\u2060\ufeff\x00"))        # ZWNJ / ZWJ are kept: they carry meaning in some scripts
_LIGATURES = {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl", "\ufb05": "st", "\ufb06": "st"}
_LIGATURE_RE = re.compile("|".join(map(re.escape, _LIGATURES)))
# Same pattern as ocr_engine._LIST_START (keep the two in sync); _BULLET_START adds bullet glyphs glued to the text and the
# private-use bullets Word / Symbol / Wingdings fonts produce.
_LIST_START = re.compile(r"^\s*(?:[\u2022\u00b7\u25cf\u25aa*-]\s+|\(?\d{1,3}[.)]\s+|\(?[a-zA-Z][.)]\s+)")
_BULLET_GLYPHS = "\u2022\u00b7\u25cf\u25aa\u25e6\u2023\u2043\uf0b7\uf0a7\uf0d8\uf076"
_BULLET_START = re.compile(r"^\s*[" + _BULLET_GLYPHS + r"]")
_BULLET_PREFIX = re.compile(r"^\s*[" + _BULLET_GLYPHS + r"]\s*(?=\S)")


def _clean(text: str) -> str:
    """Invisible characters out, ligatures and non-breaking spaces to their plain forms."""
    text = text.translate(_INVISIBLE).replace("\u00a0", " ").replace("\u202f", " ")
    return _LIGATURE_RE.sub(lambda m: _LIGATURES[m.group()], text)


def _is_list_start(line: str) -> bool:
    return bool(_LIST_START.match(line) or _BULLET_START.match(line))


def _dehyph(lines: List[str]) -> List[str]:
    """Join words split across lines: a soft hyphen at the end of a line always joins; a hyphen joins when the next line starts
    in lower case and the hyphen follows a letter (so "well-" + "Known" and "- item" are left alone)."""
    out: List[str] = []
    for l in lines:
        prev = out[-1] if out else ""
        if prev.endswith(_SOFT_HYPHEN) and l:
            out[-1] = prev[:-1] + l
        elif prev and prev[-1] in _HYPHENS and l[:1].islower() and len(prev) > 2 and prev[-2].isalpha():
            out[-1] = prev[:-1] + l
        else:
            out.append(l)
    return out


def join_lines(lines: List[str], markdown: bool = False) -> str:
    """Join one block's lines: paragraphs merge, list items keep their own line, hyphenation is removed. With markdown=True,
    bullet glyphs become ``- `` so the result renders as a list. Same rules as ocr_engine.join_lines (which keeps bullet glyphs)."""
    lines = [l.strip() for l in lines if l.strip()]
    merged: List[str] = []
    for l in _dehyph(lines):
        if merged and not _is_list_start(l):
            merged[-1] += " " + l
        else:
            merged.append(l)
    text = _clean("\n".join(merged))
    if markdown:
        text = "\n".join(_BULLET_PREFIX.sub("- ", l) for l in text.split("\n"))
    return text


# ====================================================================== geometry: visual space
def _visual_box(bbox, rot: int, W: float, H: float):
    """Unrotated page space -> the page as displayed. W, H = UNROTATED page width / height; rot = /Rotate (clockwise)."""
    x0, y0, x1, y1 = bbox
    if rot == 90:
        return (H - y1, x0, H - y0, x1)
    if rot == 180:
        return (W - x1, H - y1, W - x0, H - y0)
    if rot == 270:
        return (y0, W - x1, y1, W - x0)
    return (x0, y0, x1, y1)


def _page_geometry(page: PageResult) -> Tuple[int, float, float, float]:
    """(rotation, unrotated W, unrotated H, displayed width). PageResult.width / height are the DISPLAYED page's (page.rect):
    the unrotated size swaps them for 90 / 270. Zero sizes are recovered from the word extents."""
    rot = int(page.rotation or 0) % 360
    rot = rot if rot in (90, 180, 270) else 0
    w, h = float(page.width or 0), float(page.height or 0)
    if w <= 0 or h <= 0:
        xs = [x.bbox[2] for x in page.words] or [1.0]
        ys = [x.bbox[3] for x in page.words] or [1.0]
        w, h = (max(xs), max(ys)) if rot in (0, 180) else (max(ys), max(xs))
    W, H = (h, w) if rot in (90, 270) else (w, h)
    return rot, W, H, w


# ====================================================================== native path: blocks
def _height(ws: Sequence[Word]) -> float:
    return float(statistics.median(w.bbox[3] - w.bbox[1] for w in ws))


def _group_lines(ws: List[Word]) -> List[List[Word]]:
    """Words of one PyMuPDF block -> lines (left to right), top to bottom, in visual space."""
    by_line: Dict = {}
    for w in ws:
        by_line.setdefault(w.line, []).append(w)
    lines = [sorted(l, key=lambda w: w.bbox[0]) for l in by_line.values()]
    lines.sort(key=lambda l: (sum(w.bbox[1] + w.bbox[3] for w in l) / (2 * len(l)), l[0].bbox[0]))
    return lines


def _split_runs(lines: List[List[Word]]) -> List[List[List[Word]]]:
    """Cut a block where the line height jumps by more than 18%: a heading set tight against its paragraph is its own block."""
    runs = [[lines[0]]]
    for prev, cur in zip(lines, lines[1:]):
        hp, hc = _height(prev), _height(cur)
        if max(hp, hc) / max(min(hp, hc), 1e-6) > 1.18:
            runs.append([cur])
        else:
            runs[-1].append(cur)
    return runs


def _heading_level(b: Block, body_h: float, page_words: int) -> int:
    """1 = clearly the largest type on the page, 2 = clearly larger than body text, else 0. Conservative: needs a few words of
    context, a short block of at most three lines, real letters, and not a list item or a sentence set in large type."""
    if b.kind != "text" or not b.lines or page_words < 15 or body_h <= 0:
        return 0
    ws = [w for l in b.lines for w in l]
    text = " ".join(w.text for w in ws)
    if len(b.lines) > 3 or len(ws) > 30 or sum(c.isalpha() for c in text) < 4 or _is_list_start(text):
        return 0
    if text.rstrip().endswith(".") and len(ws) > 8:
        return 0
    r = _height(ws) / body_h
    return 1 if r >= 1.55 else 2 if r >= 1.2 else 0


def _split_axis(blocks: List[Block], axis: int, min_gap: float) -> List[List[Block]]:
    """Group blocks into runs separated by a gap (> min_gap) that NO block crosses. axis 0 = left-to-right, 1 = top-to-bottom."""
    lo, hi = (0, 2) if axis == 0 else (1, 3)
    bs = sorted(blocks, key=lambda b: (b.bbox[lo], b.bbox[hi]))
    groups, cur, cur_hi = [], [bs[0]], bs[0].bbox[hi]
    for b in bs[1:]:
        if b.bbox[lo] > cur_hi + min_gap:
            groups.append(cur)
            cur, cur_hi = [b], b.bbox[hi]
        else:
            cur.append(b)
            cur_hi = max(cur_hi, b.bbox[hi])
    groups.append(cur)
    return groups


def _rows(blocks: List[Block]) -> List[Block]:
    """Top to bottom; blocks that overlap vertically by at least half form a row, read left to right."""
    rows: List[list] = []
    for b in sorted(blocks, key=lambda b: (b.bbox[1], b.bbox[0])):
        y0, y1 = b.bbox[1], b.bbox[3]
        if rows:
            r = rows[-1]
            if min(y1, r[1]) - max(y0, r[0]) >= 0.5 * min(y1 - y0, r[1] - r[0]):
                r[2].append(b)
                r[0], r[1] = min(r[0], y0), max(r[1], y1)
                continue
        rows.append([y0, y1, [b]])
    return [b for r in rows for b in sorted(r[2], key=lambda b: b.bbox[0])]


def _column_split_ok(groups: List[List[Block]]) -> bool:
    """A vertical gutter separates real text columns only if at least two sides hold running text and the blocks are not a table
    (short cells lined up across the gutter: those are read row by row instead)."""
    lines = [sum(len(b.lines) for b in g) for g in groups]
    if sum(1 for n in lines if n >= 3) < 2:
        return False
    flat = [(i, b) for i, g in enumerate(groups) for b in g]
    n_lines = sum(len(b.lines) for _, b in flat)
    n_words = sum(len(l) for _, b in flat for l in b.lines)
    if n_lines and n_words / n_lines < 4.0:                                 # short cells, not prose lines
        aligned = 0
        for i, b in flat:
            for j, o in flat:
                if i != j and min(b.bbox[3], o.bbox[3]) - max(b.bbox[1], o.bbox[1]) >= 0.6 * min(b.bbox[3] - b.bbox[1], o.bbox[3] - o.bbox[1]):
                    aligned += 1
                    break
        if aligned >= 0.6 * len(flat):
            return False
    return True


def _xy_order(blocks: List[Block], gutter: float, depth: int = 0) -> List[Block]:
    """Recursive XY-cut: a vertical gutter spanning every block first (columns), else a horizontal gap (bands), else rows."""
    if len(blocks) <= 1:
        return list(blocks)
    if depth < 40:
        cols = _split_axis(blocks, 0, gutter)
        if len(cols) > 1 and _column_split_ok(cols):
            return [b for g in cols for b in _xy_order(g, gutter, depth + 1)]
        bands = _split_axis(blocks, 1, 0.0)
        if len(bands) > 1:
            return [b for g in bands for b in _xy_order(g, gutter, depth + 1)]
    return _rows(blocks)


def _native_reading_order(page: PageResult) -> List[Block]:
    rot, W, H, shown_w = _page_geometry(page)
    words = [replace(w, bbox=_visual_box(w.bbox, rot, W, H)) for w in page.words if not w.table and w.text and w.text.strip()]
    blocks: List[Block] = []
    groups: Dict = {}
    for w in words:
        groups.setdefault(w.block, []).append(w)
    body_h = float(statistics.median(w.bbox[3] - w.bbox[1] for w in words)) if words else 0.0
    for ws in groups.values():
        for run in _split_runs(_group_lines(ws)):
            blocks.append(Block("text", _bbox([w for l in run for w in l]), run, 0))
    for b in blocks:
        b.level = _heading_level(b, body_h, len(words))
    for t in page.tables:
        blocks.append(Block("table", _visual_box(t.bbox, rot, W, H), [], 0, t))
    if not blocks:
        return []
    return _xy_order(blocks, max(body_h, 0.015 * shown_w))


def _legacy_reading_order(page: PageResult) -> List[Block]:
    """Words that carry OCR column information: group by (col, block), sort by column then position (previous behaviour)."""
    groups: Dict = {}
    for w in page.words:
        if w.table:
            continue
        groups.setdefault((w.col, w.block), []).append(w)
    blocks = []
    for (col, _), ws in groups.items():
        lines: Dict = {}
        for w in ws:
            lines.setdefault(w.line, []).append(w)
        L = sorted(([sorted(l, key=lambda w: w.bbox[0]) for l in lines.values()]), key=lambda l: l[0].bbox[1])
        blocks.append(Block("text", _bbox(ws), L, col))
    for t in page.tables:
        blocks.append(Block("table", t.bbox, [], 0, t))
    multi = page.columns > 1
    blocks.sort(key=lambda b: ((b.col if multi and b.kind == "text" else 0), b.bbox[1], b.bbox[0]))
    return blocks


def reading_order(page: PageResult) -> List[Block]:
    legacy = page.columns > 1 or any(w.col for w in page.words if not w.table)
    return _legacy_reading_order(page) if legacy else _native_reading_order(page)


# ====================================================================== renderers
def page_text(page: PageResult) -> str:
    parts = []
    for b in reading_order(page):
        if b.kind == "table":
            parts.append("\n".join("\t".join(c.replace("\n", " ") for c in r) for r in b.table.grid()))
        else:
            parts.append(_clean("\n".join(_dehyph([" ".join(w.text for w in l) for l in b.lines]))))
    return "\n\n".join(p for p in parts if p.strip())


def page_markdown(page: PageResult) -> str:
    parts = []
    for b in reading_order(page):
        if b.kind == "table":
            parts.append(b.table.to_markdown())
            continue
        text = join_lines([" ".join(w.text for w in l) for l in b.lines], markdown=True)
        if b.level and text.strip():
            text = "#" * b.level + " " + " ".join(text.split())               # a heading is one line
        parts.append(text)
    return "\n\n".join(p for p in parts if p.strip())
