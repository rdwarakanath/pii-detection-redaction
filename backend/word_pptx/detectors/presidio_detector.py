
import re
from typing import Any

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerResult
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from models.document import DocumentBlock, NormalizedDocument


PLACEHOLDER = "xxxxxxxxxx"

ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "US_SSN",
]


class PresidioDetector:
    """Detect and anonymize PII in normalized document blocks."""

    def __init__(self, language: str = "en"):
        self.known_names: set[str] = set()
        self.language = language
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()

        # Use Presidio's pattern-recognizer extension point to reliably
        # cover email addresses and phone formats common in our test pack.
        email_recognizer = PatternRecognizer(
            supported_entity="EMAIL_ADDRESS",
            patterns=[
                Pattern(
                    name="SecureDoc email address",
                    regex=(
                        r"\b[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+"
                        r"@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+\b"
                    ),
                    score=1.0,
                )
            ],
            supported_language=language,
        )

        phone_recognizer = PatternRecognizer(
            supported_entity="PHONE_NUMBER",
            patterns=[
                Pattern(
                    name="SecureDoc synthetic sample phone",
                    regex=r"(?<![\w+])\+1[ \t]+\(555\)[ \t]+\d{4}(?![\w\d]|[ \t.-]+\d)",
                    score=1.0,
                ),
                Pattern(
                    name="SecureDoc international-style phone",
                    regex=(
                        r"(?<!\w)(?:\+\d{1,3}\s*)?"
                        r"(?:\(\d{2,4}\)|\d{2,4})"
                        r"[\s.-]+\d{3,4}[\s.-]+\d{3,4}(?!\w)"
                    ),
                    score=0.8,
                )
            ],
            supported_language=language,
        )

        self.analyzer.registry.add_recognizer(email_recognizer)
        self.analyzer.registry.add_recognizer(phone_recognizer)

        self.operators = {
            "DEFAULT": OperatorConfig(
                "replace",
                {"new_value": PLACEHOLDER},
            )
        }

    def analyze_block(self, block: DocumentBlock) -> dict[str, Any]:
        """Analyze one block and preserve the source location."""
        results = self.analyzer.analyze(
            text=block.text,
            language=self.language,
            entities=ENTITIES,
        )

        # Standalone directory column labels are not personal names.
        if block.text.strip().casefold() in {"email", "name", "mobile", "role", "division"}:
            results = [result for result in results if result.entity_type != "PERSON"]
        for name in self.known_names:
            for match in re.finditer(r"(?<!\w)" + re.escape(name) + r"(?!\w)", block.text, re.IGNORECASE):
                results.append(RecognizerResult("PERSON", match.start(), match.end(), 1.0))

        # Resolve overlaps consistently before exposing findings.
        results = self._resolve_overlaps(results)

        return {
            "source_file": block.source_file,
            "file_type": block.file_type,
            "location_type": block.location_type,
            "location": block.location,
            "text": block.text,
            "findings": [
                {
                    "entity_type": result.entity_type,
                    "start": result.start,
                    "end": result.end,
                    "score": result.score,
                    "text": block.text[result.start:result.end],
                }
                for result in results
            ],
        }

    def prepare_document(self, document: NormalizedDocument) -> None:
        """Learn full names and initial/surname aliases from labeled directories.

        The scope resets for each document. Names must be in a Name column
        whose same row contains an email, avoiding broad capitalized-word rules.
        """
        self.known_names = set()
        headers = {}
        rows = {}
        for block in document.blocks:
            if block.location_type != "table_cell":
                continue
            loc = block.location
            key = (loc.get("slide"), loc.get("shape"))
            row = int(loc["row"])
            column = int(loc["column"])
            if block.text.strip().casefold() == "name":
                headers[key] = (row, column)
            rows.setdefault((key, row), {})[column] = block.text
        for (key, row), cells in rows.items():
            if key not in headers:
                continue
            header_row, column = headers[key]
            name = cells.get(column, "").strip()
            if row <= header_row or not any("@" in text for text in cells.values()):
                continue
            parts = name.split()
            if len(parts) < 2 or not all(re.fullmatch(r"[^\W\d_]+(?:[-'?][^\W\d_]+)*", part) for part in parts):
                continue
            self.known_names.add(name)
            self.known_names.add(f"{parts[0][0]}. {' '.join(parts[1:])}")

    def analyze_document(
        self,
        document: NormalizedDocument,
    ) -> list[dict[str, Any]]:
        """Analyze every block in a normalized document."""
        self.prepare_document(document)
        return [self.analyze_block(block) for block in document.blocks]

    @staticmethod
    def _resolve_overlaps(results):
        """
        Keep the strongest non-overlapping detections.

        This avoids attempting to replace the same characters twice.
        """
        candidates = sorted(
            results,
            key=lambda result: (
                -result.score,
                result.start,
                -(result.end - result.start),
            ),
        )

        selected = []

        for candidate in candidates:
            if any(
                candidate.start < existing.end
                and candidate.end > existing.start
                for existing in selected
            ):
                continue
            selected.append(candidate)

        return sorted(selected, key=lambda result: result.start)

    def anonymize_block(self, block: DocumentBlock) -> str:
        """Anonymize a block using Presidio's replacement operator."""
        analysis = self.analyze_block(block)

        if not analysis["findings"]:
            return block.text

        results = [
            RecognizerResult(item["entity_type"], item["start"], item["end"], item["score"])
            for item in analysis["findings"]
        ]

        return self.anonymizer.anonymize(
            text=block.text,
            analyzer_results=results,
            operators=self.operators,
        ).text
