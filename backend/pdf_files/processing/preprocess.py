"""Image preprocessing primitives. All functions take/return 8-bit grayscale (dark text on light bg) unless noted."""
from __future__ import annotations
from typing import List, Optional, Tuple
import cv2
import numpy as np


# ---------------------------------------------------------------- basic filters
def illumination_normalize(g: np.ndarray) -> np.ndarray:
    """Divide by an estimated background (removes shadows / uneven scans / tinted paper)."""
    small = cv2.resize(g, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    bg = cv2.GaussianBlur(bg, (0, 0), 5)
    bg = cv2.resize(bg, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_LINEAR)
    return cv2.divide(g, np.maximum(bg, 1), scale=255)


def clahe(g: np.ndarray, clip: float = 2.0) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(g)


def unsharp(g: np.ndarray, amount: float = 0.8, sigma: float = 1.0) -> np.ndarray:
    return cv2.addWeighted(g, 1 + amount, cv2.GaussianBlur(g, (0, 0), sigma), -amount, 0)


def denoise(g: np.ndarray) -> np.ndarray:
    if g.size < 3_000_000:
        return cv2.fastNlMeansDenoising(g, None, h=8, templateWindowSize=5, searchWindowSize=15)
    return cv2.medianBlur(g, 3)


def sauvola(g: np.ndarray, window: int = 31, k: float = 0.2, R: float = 128.0) -> np.ndarray:
    window |= 1
    f = g.astype(np.float32)
    mean = cv2.boxFilter(f, -1, (window, window))
    sq = cv2.boxFilter(f * f, -1, (window, window))
    std = np.sqrt(np.maximum(sq - mean * mean, 0))
    thr = mean * (1 + k * (std / R - 1))
    return ((f > thr) * 255).astype(np.uint8)


def otsu(g: np.ndarray) -> np.ndarray:
    return cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def pad(g: np.ndarray, n: int = 12) -> np.ndarray:
    return cv2.copyMakeBorder(g, n, n, n, n, cv2.BORDER_CONSTANT, value=255)


def resize(g: np.ndarray, f: float) -> np.ndarray:
    if abs(f - 1) < 0.02:
        return g
    return cv2.resize(g, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC if f > 1 else cv2.INTER_AREA)


# ---------------------------------------------------------------- measurements
def estimate_cap_height(g: np.ndarray) -> Optional[float]:
    """80th percentile height of text-like connected components ~ capital/ascender height in px."""
    bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    n, _, st, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if n < 25:
        return None
    h, w, a = st[1:, cv2.CC_STAT_HEIGHT], st[1:, cv2.CC_STAT_WIDTH], st[1:, cv2.CC_STAT_AREA]
    m = (h >= 5) & (h <= 150) & (w <= h * 6) & (a >= 10)
    if m.sum() < 20:
        return None
    return float(np.percentile(h[m], 80))


def scale_for(g: np.ndarray, target: float, lo: float, hi: float) -> float:
    cap = estimate_cap_height(g)
    if not cap:
        return 1.0
    s = float(np.clip(target / cap, lo, hi))
    return 1.0 if 0.9 < s < 1.12 else s


def ink_fraction(g: np.ndarray) -> float:
    return float((g < 200).mean())


# ---------------------------------------------------------------- geometry
def estimate_skew(g: np.ndarray, max_deg: float = 6.0) -> float:
    """Projection-profile skew search. Returns angle (deg, cv2 convention) that straightens the page."""
    s = 900.0 / max(g.shape)
    small = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else g
    bw = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    h, w = bw.shape

    def score(a: float) -> float:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1)
        r = cv2.warpAffine(bw, M, (w, h), flags=cv2.INTER_NEAREST)
        p = r.sum(axis=1, dtype=np.float64)
        return float(np.sum(np.diff(p) ** 2))

    coarse = max(np.arange(-max_deg, max_deg + 0.01, 0.5), key=score)
    return float(max(np.arange(coarse - 0.5, coarse + 0.51, 0.1), key=score))


def rotate_affine(g: np.ndarray, angle: float) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate keeping size. Returns (image, 3x3 matrix mapping NEW px -> OLD px)."""
    h, w = g.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    out = cv2.warpAffine(g, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    inv = cv2.invertAffineTransform(M)
    return out, np.vstack([inv, [0, 0, 1]])


def rotate_k90(g: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate clockwise by 90*k. Returns (image, 3x3 matrix mapping NEW px -> OLD px)."""
    k %= 4
    h, w = g.shape
    if k == 0:
        return g, np.eye(3)
    if k == 1:   # new(x',y') = old(y', h-1-x')  -> old_x = y', old_y = h-1-x'
        return cv2.rotate(g, cv2.ROTATE_90_CLOCKWISE), np.array([[0, 1, 0], [-1, 0, h - 1], [0, 0, 1]], float)
    if k == 2:
        return cv2.rotate(g, cv2.ROTATE_180), np.array([[-1, 0, w - 1], [0, -1, h - 1], [0, 0, 1]], float)
    return cv2.rotate(g, cv2.ROTATE_90_COUNTERCLOCKWISE), np.array([[0, -1, w - 1], [1, 0, 0], [0, 0, 1]], float)


# ---------------------------------------------------------------- polarity
def find_reverse_blocks(g: np.ndarray, thresh: int, min_w: int, min_h: int, min_fill: float) -> List[Tuple[int, int, int, int]]:
    """Solid filled rectangles with lighter text inside (table headers, pills, dark screenshots)."""
    mask = (g < thresh).astype(np.uint8) * 255
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w < min_w or h < min_h or w * h > 0.8 * g.size:
            continue
        fill = (mask[y:y + h, x:x + w] > 0).mean()
        if fill >= min_fill and g[y:y + h, x:x + w].mean() < thresh:
            out.append((x, y, x + w, y + h))
    return out


def invert_blocks(g: np.ndarray, blocks) -> np.ndarray:
    out = g.copy()
    for x0, y0, x1, y1 in blocks:
        reg = 255 - out[y0:y1, x0:x1]
        lo, hi = np.percentile(reg, 1), np.percentile(reg, 99)
        if hi - lo > 20:
            reg = np.clip((reg.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
        out[y0:y1, x0:x1] = reg
    return out


# ---------------------------------------------------------------- lines
def line_masks(g: np.ndarray, kh: int, kv: int, thresh: int = 238) -> Tuple[np.ndarray, np.ndarray]:
    ink = (g < thresh).astype(np.uint8) * 255
    h = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(kh, 3), 1)))
    v = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(kv, 3))))
    return h, v


def remove_lines(g: np.ndarray, kh: int = 40, kv: int = 40) -> np.ndarray:
    h, v = line_masks(g, kh, kv, thresh=225)
    m = cv2.dilate(cv2.bitwise_or(h, v), np.ones((3, 3), np.uint8))
    out = g.copy()
    out[m > 0] = 255
    return out


# ---------------------------------------------------------------- named variants
VARIANTS = ("gray", "norm", "clahe", "otsu", "sauvola", "denoise")


def make_variant(g: np.ndarray, name: str) -> np.ndarray:
    if name == "gray":
        return g
    n = illumination_normalize(g)
    if name == "norm":
        return n
    if name == "clahe":
        return unsharp(clahe(n))
    if name == "otsu":
        return otsu(denoise(n))
    if name == "sauvola":
        return sauvola(n, window=int(max(25, (estimate_cap_height(n) or 20) * 1.8)))
    if name == "denoise":
        return denoise(n)
    raise ValueError(name)
