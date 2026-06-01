"""
generate_app_icon.py

Renders the iOS app icon for the Symptom NER app and writes every size the
existing AppIcon.appiconset expects, plus updates its Contents.json to point at
the generated files. Re-runnable — overwrites in place.

Design: a charcoal rounded-square "bracket" frame whose interior is split
horizontally into the app's two semantic colors — top half green (POS / patient
reported), bottom half red (NEG / patient denied). White background. Carries the
NER-tagging metaphor (the bracket frames an entity) and the existing in-app
color language in one tile that scales cleanly to the smallest icon sizes.

Usage (from repo root):
    python mobile_app/app/SymptomNerApp/scripts/generate_app_icon.py
"""

import json
from pathlib import Path
from PIL import Image, ImageDraw

# --- Design tokens (hex matches the in-app palette) ---------------------------
COLOR_BG = (255, 255, 255)        # white
COLOR_BORDER = (17, 24, 39)       # #111827 charcoal
COLOR_POS = (22, 163, 74)         # #16a34a green
COLOR_NEG = (220, 38, 38)         # #dc2626 red

# Master canvas size for the master render; everything else is resampled down.
MASTER_SIZE = 1024
# Frame inset from edges, corner radius, and stroke width — proportional to
# MASTER_SIZE so the resampled icons keep the same visual weight.
FRAME_INSET = 96
FRAME_RADIUS = 144
FRAME_STROKE = 36

# Per Contents.json — pixel sizes iOS asks for (size * scale, deduplicated).
ICON_SIZES = [40, 58, 60, 80, 87, 120, 180, 1024]


def render_master() -> Image.Image:
    """Draws the bracket icon at MASTER_SIZE and returns the RGB image."""
    img = Image.new("RGB", (MASTER_SIZE, MASTER_SIZE), COLOR_BG)

    # 1. Build a rounded-rectangle mask covering the frame interior.
    mask = Image.new("L", (MASTER_SIZE, MASTER_SIZE), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (FRAME_INSET, FRAME_INSET, MASTER_SIZE - FRAME_INSET, MASTER_SIZE - FRAME_INSET),
        radius=FRAME_RADIUS,
        fill=255,
    )

    # 2. Build a full-canvas fill split horizontally green-over-red.
    fill = Image.new("RGB", (MASTER_SIZE, MASTER_SIZE), COLOR_BG)
    fdraw = ImageDraw.Draw(fill)
    mid = MASTER_SIZE // 2
    fdraw.rectangle((0, 0, MASTER_SIZE, mid), fill=COLOR_POS)
    fdraw.rectangle((0, mid, MASTER_SIZE, MASTER_SIZE), fill=COLOR_NEG)

    # 3. Paste the split fill through the rounded mask, then stroke the frame.
    img.paste(fill, (0, 0), mask)
    ImageDraw.Draw(img).rounded_rectangle(
        (FRAME_INSET, FRAME_INSET, MASTER_SIZE - FRAME_INSET, MASTER_SIZE - FRAME_INSET),
        radius=FRAME_RADIUS,
        outline=COLOR_BORDER,
        width=FRAME_STROKE,
    )
    return img


def write_sized_pngs(master: Image.Image, out_dir: Path) -> dict[int, str]:
    """
    Resamples the master to each size iOS needs and writes them as PNGs.
    Returns a {pixel_size: filename} map used to rewrite Contents.json.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    sized: dict[int, str] = {}
    for size in ICON_SIZES:
        filename = f"AppIcon-{size}.png"
        resized = master.resize((size, size), Image.Resampling.LANCZOS)
        resized.save(out_dir / filename, format="PNG")
        sized[size] = filename
    return sized


def update_contents_json(contents_path: Path, sized: dict[int, str]) -> None:
    """
    Adds a `filename` field to every image entry in AppIcon's Contents.json,
    pointing at the matching generated file. Leaves any unrecognized fields alone.
    """
    contents = json.loads(contents_path.read_text())
    for entry in contents["images"]:
        size_dim = int(entry["size"].split("x")[0])
        scale = int(entry["scale"].rstrip("x"))
        pixel_size = size_dim * scale
        entry["filename"] = sized[pixel_size]
    contents_path.write_text(json.dumps(contents, indent=2) + "\n")


def main() -> None:
    """Renders the master icon, writes all sizes, and updates Contents.json."""
    project_root = Path(__file__).resolve().parents[1]
    appicon_dir = project_root / "ios" / "SymptomNerApp" / "Images.xcassets" / "AppIcon.appiconset"
    if not appicon_dir.is_dir():
        raise FileNotFoundError(f"AppIcon.appiconset not found at {appicon_dir}")
    master = render_master()
    sized = write_sized_pngs(master, appicon_dir)
    update_contents_json(appicon_dir / "Contents.json", sized)
    print(f"Wrote {len(sized)} icon PNGs to {appicon_dir}")
    print(f"Sizes: {sorted(sized)}")


if __name__ == "__main__":
    main()
