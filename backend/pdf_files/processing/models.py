from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

BBox = Tuple[float, float, float, float]   # PDF points, *unrotated* page space


@dataclass
class Word:
    text: str
    conf: float                  # 0-100 (native text = 100)
    bbox: BBox
    source: str = "ocr"          # native | ocr | ocr-cell | ocr-refine | alt:<engine>
    block: int = 0
    line: int = 0
    col: int = 0
    table: Optional[Tuple[int, int, int]] = None   # (table_id, row, col)
    raw: Optional[str] = None    # text before post-processing, if changed


@dataclass
class Cell:
    row: int
    col: int
    text: str
    conf: float
    bbox: BBox
    reocr: bool = False


@dataclass
class Table:
    id: int
    bbox: BBox
    n_rows: int
    n_cols: int
    cells: List[Cell]
    method: str                  # ruled | ruled-h | borderless
    score: float = 0.0

    def grid(self) -> List[List[str]]:
        g = [["" for _ in range(self.n_cols)] for _ in range(self.n_rows)]
        for c in self.cells:
            g[c.row][c.col] = c.text
        return g

    def to_markdown(self) -> str:
        g = self.grid()
        if not g:
            return ""
        esc = lambda s: s.replace("|", "\\|").replace("\n", " ")
        out = ["| " + " | ".join(esc(x) for x in g[0]) + " |", "|" + "---|" * self.n_cols]
        out += ["| " + " | ".join(esc(x) for x in r) + " |" for r in g[1:]]
        return "\n".join(out)


@dataclass
class PageResult:
    page_no: int                 # 1-based
    width: float
    height: float
    rotation: int
    page_type: str               # digital | scanned | hybrid | vector | blank
    words: List[Word] = field(default_factory=list)
    tables: List[Table] = field(default_factory=list)
    columns: int = 1
    score: float = 100.0
    log: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
