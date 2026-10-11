from models.document import DocumentBlock
from detectors.presidio_detector import PresidioDetector, PLACEHOLDER


def make_block(text: str) -> DocumentBlock:
    return DocumentBlock(
        text=text,
        source_file="sample.txt",
        file_type="TXT",
        location_type="test",
        location={"test_case": 1},
        extraction_method="manual",
    )


def main():
    detector = PresidioDetector()

    samples = [
        (
            "Contact m.vossberg@cadencefg.example today.",
            "EMAIL_ADDRESS",
        ),
        (
            "Mobile: +1 (555) 0114",
            "PHONE_NUMBER",
        ),
        (
            "Reach Daniela Figueroa for assistance.",
            "PERSON",
        ),
    ]

    for text, expected_entity in samples:
        block = make_block(text)
        result = detector.analyze_block(block)

        matching = [
            finding
            for finding in result["findings"]
            if finding["entity_type"] == expected_entity
        ]

        print(f"\nInput: {text}")
        print(f"Expected entity: {expected_entity}")
        print(f"Findings: {result['findings']}")

        assert matching, (
            f"Expected {expected_entity} to be detected in: {text}"
        )

        anonymized = detector.anonymize_block(block)

        assert PLACEHOLDER in anonymized, (
            f"Expected placeholder {PLACEHOLDER!r} in: {anonymized}"
        )
        assert text != anonymized, (
            f"Expected the input to be anonymized: {text}"
        )

        if expected_entity in {"EMAIL_ADDRESS", "PHONE_NUMBER"}:
            assert matching[0]["text"] not in anonymized, (
                f"PII was not removed: {matching[0]['text']}"
            )

        print(f"Anonymized: {anonymized}")

    assert not detector.analyze_block(make_block("Email"))["findings"]
    assert not any(
        finding["entity_type"] == "PHONE_NUMBER"
        for finding in detector.analyze_block(make_block("Version 1.0, 2026-10-12, 555 0114"))["findings"]
    )
    print("\nAll focused Presidio tests passed.")


if __name__ == "__main__":
    main()