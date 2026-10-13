"""Apply reviewed image regions only when the exact source pixels match."""
from hashlib import sha256
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from detectors.presidio_detector import PLACEHOLDER


def redact_reviewed_image(data: bytes, entry: dict) -> bytes:
    if sha256(data).hexdigest() != entry["sha256"]:
        raise ValueError("Image digest differs from reviewed source; coordinates are unsafe")
    if not entry.get("regions"):
        return data
    with Image.open(BytesIO(data)) as original:
        if original.format != "PNG":
            raise ValueError("Reviewed image masking currently supports PNG only")
        image = original.convert("RGBA" if "A" in original.getbands() else "RGB")
        draw = ImageDraw.Draw(image)
        for region in entry["regions"]:
            left, top, right, bottom = region["box"]
            if not (0 <= left < right <= image.width and 0 <= top < bottom <= image.height):
                raise ValueError("Invalid reviewed image rectangle")
            width, height = right - left, bottom - top
            font_path = Path("C:/Windows/Fonts/arial.ttf")
            for size in range(min(18, height - 2), 3, -1):
                font = ImageFont.truetype(str(font_path), size) if font_path.is_file() else ImageFont.load_default(size=size)
                bounds = draw.textbbox((0, 0), PLACEHOLDER, font=font)
                if bounds[2] - bounds[0] <= width - 4 and bounds[3] - bounds[1] <= height - 2:
                    break
            else:
                raise ValueError("Reviewed rectangle cannot fit the required placeholder")
            # Replace pixel data, not an overlay that leaves recoverable content.
            draw.rectangle((left, top, right - 1, bottom - 1), fill="white")
            draw.text((left + 2, top + (height - (bounds[3] - bounds[1])) / 2 - bounds[1]), PLACEHOLDER, font=font, fill="black")
        output = BytesIO()
        # Saving without image metadata prevents an original thumbnail surviving.
        image.save(output, format="PNG")
        return output.getvalue()
