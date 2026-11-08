"""Contact sheet generator for extracted video frames."""

from pathlib import Path
from typing import List, Tuple
from PIL import Image, ImageDraw, ImageFont


def generate_contact_sheet(
    frame_entries: List[Tuple[Path, float]],
    output_path: Path,
    columns: int = 4,
    thumb_max_dim: int = 360,
    padding: int = 16,
    label_height: int = 28,
    jpeg_quality: int = 90,
) -> Path:
    """Generate a grid contact sheet image showing all extracted frames in order.

    Args:
        frame_entries: List of (frame_path, timestamp_seconds) in chronological order.
        output_path: Destination path for the contact_sheet.jpg file.
        columns: Number of columns in the grid.
        thumb_max_dim: Maximum width or height of each frame thumbnail in pixels.
        padding: Padding in pixels around cells.
        label_height: Height reserved below each thumbnail for its filename/timestamp label.
        jpeg_quality: JPEG compression quality (1-100).

    Returns:
        Path to the saved contact sheet.
    """
    if not frame_entries:
        raise ValueError("Cannot generate a contact sheet with no frames.")

    first_frame_path = frame_entries[0][0]
    with Image.open(first_frame_path) as sample_img:
        orig_w, orig_h = sample_img.size

    if orig_w <= 0 or orig_h <= 0:
        raise ValueError(f"Invalid frame dimensions: {orig_w}x{orig_h}")

    # Preserve original aspect ratio for thumbnails
    scale = min(thumb_max_dim / orig_w, thumb_max_dim / orig_h)
    thumb_w = max(1, int(round(orig_w * scale)))
    thumb_h = max(1, int(round(orig_h * scale)))

    # Adjust column count automatically for vertical vs horizontal videos if helpful
    num_frames = len(frame_entries)
    cols = min(columns, num_frames)
    rows = (num_frames + cols - 1) // cols

    cell_w = thumb_w
    cell_h = thumb_h + label_height

    sheet_w = padding + cols * (cell_w + padding)
    sheet_h = padding + rows * (cell_h + padding)

    sheet = Image.new("RGB", (sheet_w, sheet_h), color=(24, 24, 28))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for idx, (frame_path, _) in enumerate(frame_entries):
        row = idx // cols
        col = idx % cols

        x = padding + col * (cell_w + padding)
        y = padding + row * (cell_h + padding)

        with Image.open(frame_path) as img:
            img_rgb = img.convert("RGB")
            if img_rgb.size != (thumb_w, thumb_h):
                thumb = img_rgb.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            else:
                thumb = img_rgb

        sheet.paste(thumb, (x, y))

        # Draw subtle border around thumbnail
        draw.rectangle(
            [x, y, x + thumb_w - 1, y + thumb_h - 1],
            outline=(70, 70, 80),
            width=1,
        )

        # Label with frame number and timestamp (derived from filename stem)
        label_text = frame_path.stem
        text_x = x + 4
        text_y = y + thumb_h + 6
        draw.text((text_x, text_y), label_text, fill=(225, 225, 230), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="JPEG", quality=jpeg_quality, optimize=True)
    return output_path

