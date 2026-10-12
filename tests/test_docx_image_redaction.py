"""Verify the reviewed sample image policy without relying on OCR accuracy."""
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

from PIL import Image, ImageChops, ImageDraw
from redaction.image_redactor import redact_reviewed_image


def main(output_path="data/output/Cadence_TPRM_Training_redacted.docx"):
    manifest = json.loads(Path("config/docx_image_redactions.json").read_text())["images"]
    changed, regions = 0, 0
    with ZipFile("data/input/Cadence_TPRM_Training.docx") as source, ZipFile(output_path) as output:
        for part, entry in manifest.items():
            original, result = source.read(part), output.read(part)
            assert sha256(original).hexdigest() == entry["sha256"]
            if not entry["regions"]:
                assert result == original, f"Unexpected change to {part}"
                continue
            assert result == redact_reviewed_image(original, entry)
            before, after = Image.open(BytesIO(original)).convert("RGB"), Image.open(BytesIO(result)).convert("RGB")
            assert before.size == after.size
            differences = ImageChops.difference(before, after)
            for region in entry["regions"]:
                left, top, right, bottom = region["box"]
                assert ImageChops.difference(before.crop((left, top, right, bottom)), after.crop((left, top, right, bottom))).getbbox()
                ImageDraw.Draw(differences).rectangle((left, top, right - 1, bottom - 1), fill="black")
            assert differences.getbbox() is None, f"Changed pixels outside masks: {part}"
            try:
                redact_reviewed_image(original + b"changed", entry)
            except ValueError:
                pass
            else:
                raise AssertionError("Unsafe mismatched source was accepted")
            changed += 1
            regions += len(entry["regions"])
    print(f"{len(manifest)} media assets checked; {changed} screenshots and {regions} regions redacted: PASS")
    print("Image dimensions, unaffected pixels/assets, and source digest guard: PASS")


if __name__ == "__main__":
    main()
