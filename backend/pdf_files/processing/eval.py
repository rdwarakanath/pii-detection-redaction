"""Evaluate page Markdown against manually corrected page references.

Example:
    python -m First_pipeline.firstpp_fixed.processing.eval \
        --predictions First_pipeline/firstpp_fixed/output/run_sample_35p_wordboxes/_internal/sample.pages.json \
        --reference-dir reference_pages

Reference files must be named page_0001.md, page_0001.txt, etc. This reports
character/word error rates and the per-page engine, runtime, and review status.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

try:
    from rapidfuzz.distance import Levenshtein
except ImportError:  # Keep the evaluator usable before optional acceleration installs.
    Levenshtein = None


_PAGE_FILE = re.compile(r"^page[_-]?0*(\d+)\.(?:md|markdown|txt)$", re.IGNORECASE)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_HTML_TAG = re.compile(r"</?[^>]+>")


def plain_text(value: str) -> str:
    """Remove Markdown presentation syntax while retaining readable text and table cells."""
    value = _HTML_COMMENT.sub(" ", value)
    value = _LINK.sub(r"\1", value)
    value = _HTML_TAG.sub(" ", value)
    value = re.sub(r"^\s{0,3}#{1,6}\s*", "", value, flags=re.MULTILINE)
    value = re.sub(r"^\s{0,3}(?:[-+*]|\d+[.)])\s+", "", value, flags=re.MULTILINE)
    value = re.sub(r"[`*_~]", "", value)
    value = value.replace("|", " ")
    return re.sub(r"\s+", " ", value).strip()


def edit_distance(reference, prediction) -> int:
    if Levenshtein is not None:
        return int(Levenshtein.distance(reference, prediction))
    # Memory is O(min(n, m)); time is O(n*m). For large evaluation sets, install
    # install rapidfuzz for its optimized exact distance.
    if len(reference) < len(prediction):
        reference, prediction = prediction, reference
    previous = list(range(len(prediction) + 1))
    for i, ref_char in enumerate(reference, 1):
        current = [i]
        for j, pred_char in enumerate(prediction, 1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (ref_char != pred_char),
            ))
        previous = current
    return previous[-1]


def error_rate(reference, prediction) -> float:
    if not reference:
        return 0.0 if not prediction else 1.0
    return edit_distance(reference, prediction) / len(reference)


def read_records(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    pages = data.get("pages") if isinstance(data, dict) else None
    if not isinstance(pages, list):
        raise ValueError("Predictions JSON must contain a 'pages' list")
    out = []
    for page in pages:
        if isinstance(page, dict) and isinstance(page.get("page_no"), int):
            out.append(page)
    if not out:
        raise ValueError("Predictions JSON contains no page records with integer page_no values")
    return out


def read_references(directory: Path) -> dict[int, Path]:
    refs: dict[int, Path] = {}
    for path in directory.iterdir():
        match = _PAGE_FILE.match(path.name)
        if path.is_file() and match:
            page_no = int(match.group(1))
            if page_no in refs:
                raise ValueError(f"More than one reference file found for page {page_no}")
            refs[page_no] = path
    return refs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path,
                        help="internal page JSON written by the pipeline")
    parser.add_argument("--reference-dir", required=True, type=Path,
                        help="directory of corrected page_####.md/.txt references")
    args = parser.parse_args(argv)

    try:
        pages = read_records(args.predictions)
        refs = read_references(args.reference_dir)
        evaluated = []
        missing = []
        for page in pages:
            no = page["page_no"]
            ref_path = refs.get(no)
            if ref_path is None:
                missing.append(no)
                continue
            ref = plain_text(ref_path.read_text(encoding="utf-8"))
            pred = plain_text(str(page.get("markdown") or ""))
            ref_words, pred_words = ref.split(), pred.split()
            cer = error_rate(ref, pred)
            wer = error_rate(ref_words, pred_words)
            stats = page.get("stats") or {}
            evaluated.append({
                "page_no": no,
                "cer": cer,
                "wer": wer,
                "text_accuracy_pct": max(0.0, 1.0 - cer) * 100.0,
                "source": page.get("source", "unknown"),
                "seconds": float(stats.get("seconds", 0) or 0),
                "needs_review": bool(stats.get("needs_review") or stats.get("needs_vlm") or page.get("source") == "error"),
            })
        if not evaluated:
            raise ValueError("No prediction pages matched reference filenames")

        # Micro-averaged CER/WER: longer reference pages contribute proportionally.
        refs_text = []
        preds_text = []
        for row in evaluated:
            page_no = row["page_no"]
            refs_text.append(plain_text(refs[page_no].read_text(encoding="utf-8")))
            preds_text.append(plain_text(str(next(p for p in pages if p["page_no"] == page_no).get("markdown") or "")))
        all_ref = "\n".join(refs_text)
        all_pred = "\n".join(preds_text)
        aggregate_cer = error_rate(all_ref, all_pred)
        ref_tokens, pred_tokens = all_ref.split(), all_pred.split()
        aggregate_wer = error_rate(ref_tokens, pred_tokens)
        payload = {
            "pages_evaluated": len(evaluated),
            "pages_missing_reference": missing,
            "cer": aggregate_cer,
            "wer": aggregate_wer,
            "text_accuracy_pct": max(0.0, 1.0 - aggregate_cer) * 100.0,
            "target_85_percent_met": aggregate_cer <= 0.15,
            "evaluated_page_seconds": round(sum(r["seconds"] for r in evaluated), 2),
            "pages": evaluated,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if payload["target_85_percent_met"] else 1
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"evaluation error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
