
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from models.document import DocumentBlock, NormalizedDocument


class PresidioDetector:
    """Detect and anonymize PII in normalized document blocks."""

    def __init__(self, language: str = "en"):
        self.language = language
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()

    def analyze_block(self, block: DocumentBlock) -> dict:
        """Analyze one extracted block and preserve its source location."""

        results = self.analyzer.analyze(
            text=block.text,
            language=self.language,
            entities=[
                "PERSON",
                "EMAIL_ADDRESS",
                "PHONE_NUMBER",
                "CREDIT_CARD",
                "US_SSN",
            ],
        )

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

    def analyze_document(self, document: NormalizedDocument) -> list[dict]:
        """Analyze every block in a normalized document."""

        return [
            self.analyze_block(block)
            for block in document.blocks
        ]

    def anonymize_block(self, block: DocumentBlock) -> str:
        """Replace detected PII with readable placeholders."""

        results = self.analyzer.analyze(
            text=block.text,
            language=self.language,
            entities=[
                "PERSON",
                "EMAIL_ADDRESS",
                "PHONE_NUMBER",
                "CREDIT_CARD",
                "US_SSN",
            ],
        )

        if not results:
            return block.text

        operators = {
            "PERSON": OperatorConfig(
                "replace", {"new_value": "[PERSON]"}
            ),
            "EMAIL_ADDRESS": OperatorConfig(
                "replace", {"new_value": "[EMAIL]"}
            ),
            "PHONE_NUMBER": OperatorConfig(
                "replace", {"new_value": "[PHONE]"}
            ),
            "CREDIT_CARD": OperatorConfig(
                "replace", {"new_value": "[CREDIT_CARD]"}
            ),
            "US_SSN": OperatorConfig(
                "replace", {"new_value": "[SSN]"}
            ),
        }

        result = self.anonymizer.anonymize(
            text=block.text,
            analyzer_results=results,
            operators=operators,
        )

        return result.text
