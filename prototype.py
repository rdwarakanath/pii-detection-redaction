"""Compatibility entry point for the shared OCR pipeline CLI.

All routing, configuration, checkpoints, fallback verification, and output
assembly live in cli.py / pipeline.py. This wrapper intentionally avoids a
second independent OCR implementation that could drift from the production
path.

Examples:
    python -m First_pipeline.prototype input.pdf --fallback-engine vl
    python First_pipeline/prototype.py input.pdf --fallback-engine vl
"""
from __future__ import annotations

import sys
from pathlib import Path


def _load_main():
    if __package__:
        from .cli import main
        return main
    # Support direct-script execution from the repository root as well as
    # `python -m First_pipeline.prototype` without masking import errors from
    # the CLI's own dependencies.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from First_pipeline.cli import main
    return main


if __name__ == "__main__":
    raise SystemExit(_load_main()())
