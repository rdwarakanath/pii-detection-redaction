#!/usr/bin/env python3
"""
md_to_text.py - Convert OCR'd Markdown (e.g. finance PDFs run through an OCR
pipeline) into clean plain text, keeping the real content and table structure.

What it removes
  * OCR/page marker comments   <!-- page 3 | source=... -->   (and any other <!-- -->)
  * Markdown heading markers   '## 1.1 Objectives'  ->  '1.1 Objectives'
  * Table separator rows       |---|---|
  * Bold / italic / inline-code markers, image syntax, link URLs, known HTML tags
  * Horizontal rules and code-fence lines
  * Running page headers/footers that repeat on most pages
    ("Group Risk Management Policy", "Page 2of35", ...)  [switch off with --keep-running]

What it does NOT do
  * It never collapses or re-flows spaces inside your text - OCR text is kept
    character-for-character.
  * It never strips arbitrary <...> - so "< USD 100k" or "<10 pts" survive.

Tables
  Every cell stays a separate cell (empty cells keep their column position).
  --table-style aligned  (default) padded columns, 3-space gap
                 tsv     tab-separated, best for pasting into Excel / parsing
                 pipe    "cell | cell | cell"

Output
  <name>.txt is written in the SAME folder as the input file (or --out-dir).

Usage
  python md_to_text.py sample.md
  python md_to_text.py a.md b.md --table-style tsv
  python md_to_text.py                # converts every .md in the current folder
  python md_to_text.py docs/          # converts every .md in that folder
"""

import argparse
import html
import re
import sys
from collections import Counter
from pathlib import Path

# ----------------------------------------------------------------------------
# Regexes
# ----------------------------------------------------------------------------
PAGE_MARKER = re.compile(r"<!--\s*page\s+(\d+)[^>]*?-->", re.I)
HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)

# Only genuine HTML tags (whitelist) are removed. OCR text such as "<Collapse ...>"
# or "< USD 100k" is real content and must survive.
HTML_TAGS = (
    "br|p|div|span|b|i|u|s|strong|em|sub|sup|small|font|center|"
    "table|thead|tbody|tfoot|tr|td|th|caption|colgroup|col|"
    "ul|ol|li|h[1-6]|img|a|hr|pre|code|blockquote|section|article|figure|figcaption"
)
BR_TAG = re.compile(r"<br\s*/?>", re.I)
HTML_TAG = re.compile(rf"</?(?:{HTML_TAGS})(?:\s[^<>]*)?/?>", re.I)

HEADING = re.compile(r"^\s{0,3}#{1,6}(?:\s+|$)(.*?)(?:\s+#+)?\s*$")
HRULE = re.compile(r"^\s*([-*_])(?:\s*\1){2,}\s*$")
FENCE = re.compile(r"^\s*(```|~~~)")
TABLE_SEP_CELL = re.compile(r"^\s*:?-{2,}:?\s*$")

IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
LINK = re.compile(r"\[([^\]]+)\]\((?:[^)\s]+)(?:\s+\"[^\"]*\")?\)")
BOLD_STAR = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*")
BOLD_UNDER = re.compile(r"(?<!\w)__(?=\S)(.+?)(?<=\S)__(?!\w)")
INLINE_CODE = re.compile(r"`([^`\n]+)`")
MD_ESCAPE = re.compile(r"\\([*_#\[\]()`~|])")


# ----------------------------------------------------------------------------
# Inline cleanup (never touches whitespace inside the text)
# ----------------------------------------------------------------------------
def clean_inline(s: str, in_table: bool = False) -> str:
    s = BR_TAG.sub(" " if in_table else "\n", s)
    s = HTML_TAG.sub("", s)
    s = IMAGE.sub(r"\1", s)
    s = LINK.sub(r"\1", s)
    s = BOLD_STAR.sub(r"\1", s)
    s = BOLD_UNDER.sub(r"\1", s)
    s = INLINE_CODE.sub(r"\1", s)
    s = MD_ESCAPE.sub(r"\1", s)
    s = html.unescape(s)
    return s


# ----------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------
def split_row(line: str) -> list[str]:
    """Split a markdown table row into cells (respects \\| escapes)."""
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not s.endswith("\\|"):
        s = s[:-1]
    cells = re.split(r"(?<!\\)\|", s)
    return [c.strip() for c in cells]


def is_separator_row(cells: list[str]) -> bool:
    return bool(cells) and all(TABLE_SEP_CELL.match(c) for c in cells if c != "" ) \
        and any(c for c in cells)


def render_table(rows: list[list[str]], style: str, max_pad: int) -> list[str]:
    ncols = max(len(r) for r in rows)
    rows = [r + [""] * (ncols - len(r)) for r in rows]

    if style == "tsv":
        return ["\t".join(r).rstrip("\t") if any(r) else "" for r in rows]
    if style == "pipe":
        return [" | ".join(r) for r in rows]

    # aligned
    widths = [min(max(len(r[c]) for r in rows), max_pad) for c in range(ncols)]
    out = []
    for r in rows:
        parts = [cell.ljust(widths[i]) for i, cell in enumerate(r)]
        out.append("   ".join(parts).rstrip())
    return out


# ----------------------------------------------------------------------------
# Page parsing
# ----------------------------------------------------------------------------
def split_pages(text: str):
    """Return [(page_number_or_None, chunk_text), ...]."""
    marks = list(PAGE_MARKER.finditer(text))
    if not marks:
        return [(None, text)]
    pages = []
    if text[: marks[0].start()].strip():
        pages.append((None, text[: marks[0].start()]))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        pages.append((int(m.group(1)), text[m.end():end]))
    return pages


def parse_page(chunk: str, table_style: str, max_pad: int):
    """
    Turn one page of markdown into a list of blocks:
      ("text", str) | ("heading", str) | ("table", [lines]) | ("blank", "")
    """
    chunk = HTML_COMMENT.sub("", chunk)
    blocks = []
    table_rows: list[list[str]] = []
    in_fence = False

    def flush_table():
        nonlocal table_rows
        if table_rows:
            blocks.append(("blank", ""))
            blocks.append(("table", render_table(table_rows, table_style, max_pad)))
            blocks.append(("blank", ""))
            table_rows = []

    for raw in chunk.split("\n"):
        line = raw.rstrip()

        if FENCE.match(line):
            flush_table()
            in_fence = not in_fence
            continue
        if in_fence:
            blocks.append(("text", line))
            continue

        stripped = line.strip()

        # ---- table rows
        if stripped.startswith("|"):
            cells = split_row(stripped)
            if is_separator_row(cells):
                continue
            table_rows.append([clean_inline(c, in_table=True).strip() for c in cells])
            continue
        flush_table()

        if not stripped:
            blocks.append(("blank", ""))
            continue
        if HRULE.match(stripped):
            continue

        m = HEADING.match(line)
        if m:
            txt = clean_inline(m.group(1)).strip()
            if txt:
                blocks.append(("heading", txt))
            continue

        txt = clean_inline(line)
        if txt.strip():
            blocks.append(("text", txt))
        else:
            blocks.append(("blank", ""))

    flush_table()
    return blocks


# ----------------------------------------------------------------------------
# Running header / footer detection
# ----------------------------------------------------------------------------
def norm_key(s: str) -> str:
    """Whitespace-free, digit-insensitive key so 'Page 2of35' == 'Page 3of35'
    and 'GRP-POL-001|Version6.0|Internal' == 'GRP-POL-001 | Version 6.0 | Internal'."""
    return re.sub(r"\d+", "#", re.sub(r"\s+", "", s)).lower()


def find_running_keys(pages_blocks, edge: int = 4, min_pages: int = 3, ratio: float = 0.4):
    """Plain-text lines that sit at the top/bottom edge of many pages."""
    n_pages = len(pages_blocks)
    if n_pages < min_pages:
        return set()
    seen = Counter()
    for blocks in pages_blocks:
        idx = [i for i, (k, _) in enumerate(blocks) if k in ("text", "heading", "table")]
        edge_idx = set(idx[:edge]) | set(idx[-edge:])
        keys_on_page = set()
        for i in edge_idx:
            kind, val = blocks[i]
            if kind == "text" and len(val) <= 120:
                keys_on_page.add(norm_key(val))
        seen.update(keys_on_page)
    need = max(min_pages, int(n_pages * ratio))
    return {k for k, c in seen.items() if c >= need}


# ----------------------------------------------------------------------------
# Main conversion
# ----------------------------------------------------------------------------
def convert(md_text: str, table_style="aligned", max_pad=40,
            keep_running=False, page_markers=False) -> str:
    md_text = md_text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")

    pages = split_pages(md_text)
    parsed = [parse_page(chunk, table_style, max_pad) for _, chunk in pages]

    running = set() if keep_running else find_running_keys(parsed)

    out: list[str] = []
    for (pno, _), blocks in zip(pages, parsed):
        if page_markers and pno is not None:
            out.append("")
            out.append(f"[Page {pno}]")
            out.append("")
        for i, (kind, val) in enumerate(blocks):
            if kind == "table":
                out.extend(val)
            elif kind in ("text", "heading"):
                if kind == "text" and norm_key(val) in running:
                    continue
                out.append(val)
            else:
                out.append("")
        out.append("")  # page boundary

    # tidy vertical whitespace only: max one blank line in a row
    result, blank = [], 0
    for ln in out:
        if ln.strip() == "":
            blank += 1
            if blank <= 1:
                result.append("")
        else:
            blank = 0
            result.append(ln.rstrip())
    return "\n".join(result).strip("\n") + "\n"


def process_file(path: Path, args) -> Path:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    plain = convert(
        text,
        table_style=args.table_style,
        max_pad=args.max_col_width,
        keep_running=args.keep_running,
        page_markers=args.page_markers,
    )
    out_dir = Path(args.out_dir) if args.out_dir else path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (path.stem + ".txt")
    out_path.write_text(plain, encoding="utf-8", newline="\n")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="OCR'd Markdown -> clean plain text (tables preserved).")
    ap.add_argument("inputs", nargs="*", help=".md file(s) or folder(s). Default: current folder.")
    ap.add_argument("--table-style", choices=["aligned", "tsv", "pipe"], default="aligned")
    ap.add_argument("--max-col-width", type=int, default=40,
                    help="max padding width per column in 'aligned' mode (default 40)")
    ap.add_argument("--keep-running", action="store_true",
                    help="keep repeating page headers/footers (default: remove them)")
    ap.add_argument("--page-markers", action="store_true",
                    help="insert a [Page N] line where each page starts")
    ap.add_argument("--out-dir", help="write .txt here instead of next to each input")
    args = ap.parse_args()

    files: list[Path] = []
    for item in (args.inputs or ["."]):
        p = Path(item)
        if p.is_dir():
            files.extend(sorted(p.glob("*.md")))
        elif p.is_file():
            files.append(p)
        else:
            print(f"Not found: {p}", file=sys.stderr)
    if not files:
        sys.exit("No .md files found.")

    for f in files:
        out = process_file(f, args)
        print(f"{f}  ->  {out}")


if __name__ == "__main__":
    main()
