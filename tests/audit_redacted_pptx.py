import re
from pathlib import Path

from extractors.pptx_extractor import extract_pptx


INPUT_FILE = Path(
    "data/output/"
    "Cadence_Financial_Group_Organizational_Pack_redacted.pptx"
)

EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+"
    r"@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+\b"
)

PHONE_PATTERN = re.compile(
    r"(?<!\w)"
    r"(?:\+\d{1,3}[\s.-]*)?"
    r"(?:\(\d{2,4}\)|\d{2,4})"
    r"[\s.-]+\d{3,4}"
    r"(?:[\s.-]+\d{3,4})?"
    r"(?!\w)"
)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Redacted PPTX not found: {INPUT_FILE}\n"
            "Run the PPTX redaction test first."
        )

    document = extract_pptx(str(INPUT_FILE))

    email_findings = []
    phone_findings = []

    for block in document.blocks:
        for match in EMAIL_PATTERN.finditer(block.text):
            email_findings.append(
                {
                    "text": match.group(),
                    "location_type": block.location_type,
                    "location": block.location,
                }
            )

        for match in PHONE_PATTERN.finditer(block.text):
            phone_findings.append(
                {
                    "text": match.group(),
                    "location_type": block.location_type,
                    "location": block.location,
                }
            )

    print("===== REDACTED PPTX AUDIT =====")
    print(f"File: {INPUT_FILE}")
    print(f"Total extracted blocks: {len(document.blocks)}")
    print(f"Possible residual emails: {len(email_findings)}")
    print(f"Possible residual phone numbers: {len(phone_findings)}")

    if email_findings:
        print("\n--- Possible residual emails ---")
        for finding in email_findings:
            print(
                f"{finding['text']} | "
                f"{finding['location_type']} | "
                f"{finding['location']}"
            )

    if phone_findings:
        print("\n--- Possible residual phone numbers ---")
        for finding in phone_findings:
            print(
                f"{finding['text']} | "
                f"{finding['location_type']} | "
                f"{finding['location']}"
            )

    if email_findings or phone_findings:
        print(
            "\nAUDIT RESULT: REVIEW REQUIRED — "
            "possible PII remains."
        )
    else:
        print(
            "\nAUDIT RESULT: No email or phone matches found "
            "by these patterns."
        )
        print(
            "This does not prove that all PII is removed; "
            "names, IDs, images, and other sensitive content "
            "need separate checks."
        )


if __name__ == "__main__":
    main()