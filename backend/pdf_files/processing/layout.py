"""Layout analysis: ruled tables, multi-column splitting, borderless-table detection."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import cv2
import numpy as np
from . import preprocess as pp
from .config import Config


@dataclass
class TableCand:
    bbox: Tuple[int, int, int, int]                 # x0,y0,x1,y1 (px)
    row_spans: List[Tuple[int, int]] = field(default_factory=list)   # from rules; may be empty
    col_lines: List[int] = field(default_factory=list)               # x of vertical rules; may be empty
    method: str = "ruled-h"


def _runs(mask1d: np.ndarray) -> List[Tuple[int, int]]:
    out, start = [], None
    for i, v in enumerate(mask1d):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i)); start = None
    if start is not None:
        out.append((start, len(mask1d)))
    return out


def detect_tables(g: np.ndarray, dpi: int, cfg: Config) -> List[TableCand]:
    H, W = g.shape
    kh = max(40, int(W * 0.035))
    kv = max(30, int(dpi * 0.10))
    hmask, vmask = pp.line_masks(g, kh, kv)
    gap = int(cfg.table_row_gap_in * dpi)
    joined = cv2.dilate(hmask, cv2.getStructuringElement(cv2.MORPH_RECT, (31, gap)))
    cnts, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cap = pp.estimate_cap_height(g) or 25
    out: List[TableCand] = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w < W * cfg.table_min_width_frac:
            continue
        sub = hmask[y:y + h, x:x + w] > 0
        cover = sub.sum(axis=1) >= 0.5 * w
        lines = _runs(cover)
        if len(lines) < cfg.table_min_lines:
            continue
        cols_any = sub.any(axis=0)
        xs = np.where(cols_any)[0]
        tx0, tx1 = x + int(xs.min()), x + int(xs.max()) + 1
        ty0, ty1 = y + lines[0][0], y + lines[-1][1]
        # separators: thin rule -> its centre; thick band (filled header) -> both edges (the band is a row)
        seps = []
        thick = max(6, int(cap * 0.45))
        for a, b in lines:
            if b - a >= thick:
                seps += [y + a, y + b]
            else:
                seps.append(y + (a + b) // 2)
        seps = sorted(seps)
        min_row = max(8, int(cap * 0.8))
        rows = [(s0, s1) for s0, s1 in zip(seps, seps[1:]) if s1 - s0 >= min_row]
        # vertical rules
        vs = vmask[ty0:ty1, tx0:tx1] > 0
        vcov = vs.sum(axis=0) >= 0.6 * (ty1 - ty0)
        vx = [tx0 + (a + b) // 2 for a, b in _runs(vcov)]
        method = "ruled" if len(vx) >= 3 else "ruled-h"
        out.append(TableCand((tx0, ty0, tx1, ty1), rows, vx if method == "ruled" else [], method))
    out.sort(key=lambda t: t.bbox[1])
    return out


def find_columns(g: np.ndarray, cap: float, min_cols_ink: float = 0.15) -> List[Tuple[int, int]]:
    """Return x-ranges of text columns. Single [(0,W)] when the page is one column."""
    H, W = g.shape
    ink = (g < 190).astype(np.uint8)
    smear = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(cap * 3))))
    proj = smear.sum(axis=0)
    empty = proj < 0.02 * H
    gaps = [(a, b) for a, b in _runs(empty) if b - a >= 2.0 * cap and a > W * 0.12 and b < W * 0.88]
    if not gaps:
        return [(0, W)]
    cuts = [0] + [(a + b) // 2 for a, b in gaps] + [W]
    ranges = list(zip(cuts, cuts[1:]))
    tot = ink.sum() or 1
    ranges = [r for r in ranges if ink[:, r[0]:r[1]].sum() / tot >= min_cols_ink]
    return ranges if len(ranges) >= 2 else [(0, W)]


def detect_borderless(words, cap: float) -> List[Tuple[int, int, int, int]]:
    """Find runs of >=3 consecutive text lines whose words fall into >=3 consistently aligned columns."""
    from .ocr_engine import _lines
    ls = sorted(_lines(words), key=lambda l: l[1])
    gap_thr = 1.6 * cap
    info = []
    for x0, y0, x1, y1, ws in ls:
        ws = sorted(ws, key=lambda w: w.x0)
        starts = [ws[0].x0]
        for a, b in zip(ws, ws[1:]):
            if b.x0 - a.x1 > gap_thr:
                starts.append(b.x0)
        info.append((x0, y0, x1, y1, starts, ws))
    runs, cur = [], []
    for item in info:
        if len(item[4]) >= 3 and (not cur or item[1] - cur[-1][3] < 2.5 * cap):
            cur.append(item)
        else:
            if len(cur) >= 3:
                runs.append(cur)
            cur = [item] if len(item[4]) >= 3 else []
    if len(cur) >= 3:
        runs.append(cur)
    out = []
    tol = 0.8 * cap
    for run in runs:
        allstarts = [s for it in run for s in it[4]]
        support = 0
        for s in set(round(a / tol) for a in allstarts):
            n = sum(1 for it in run if any(abs(a - s * tol) <= tol for a in it[4]))
            if n >= 0.6 * len(run):
                support += 1
        if support >= 3:
            out.append((int(min(i[0] for i in run)), int(min(i[1] for i in run)),
                        int(max(i[2] for i in run)), int(max(i[3] for i in run))))
    return out
