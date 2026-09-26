"""Placeholder ad creatives.

The catalogue references creatives (e.g. ads/brand_a/a_15s_bn.mp4) whose files were not supplied.
Any missing file is rendered as a plain slate with the synthetic brand name, category and length,
so the manifest and player always work, including for a brand added later.
"""

import hashlib
import shutil
import subprocess
from pathlib import Path

from .config import REPO_ROOT

ADS_ROOT = REPO_ROOT / "data"
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
]


def creative_path(creative: dict) -> Path:
    return ADS_ROOT / creative["url"].lstrip("/")


def _font() -> str | None:
    for candidate in FONT_CANDIDATES:
        if Path(candidate).exists():
            return candidate.replace(":", "\\:")
    return None


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\u2019").replace("%", "\\%")


def render_placeholder(brand: dict, creative: dict, dst: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH")
    dst.parent.mkdir(parents=True, exist_ok=True)
    hue = int(hashlib.md5(brand["brand_id"].encode()).hexdigest()[:2], 16) / 255
    r, g, b = (int(40 + 120 * abs(((hue * 6 + k) % 6) - 3) / 3) for k in (0, 4, 2))
    color = f"0x{r:02x}{g:02x}{b:02x}"
    font = _font()
    fontfile = f":fontfile='{font}'" if font else ""
    duration = creative["duration_sec"]
    lines = [
        (brand["display_name"], 72, -60),
        (brand["category"], 34, 30),
        (f"{duration}s ad  |  {creative['id']}  |  synthetic placeholder", 24, 90),
    ]
    text = ",".join(
        f"drawtext=text='{_escape(t)}'{fontfile}:fontsize={size}:fontcolor=white:"
        f"x=(w-text_w)/2:y=(h-text_h)/2+{dy}"
        for t, size, dy in lines
    )
    countdown = (f"drawtext=text='Ad %{{eif\\:{duration}-t\\:d}}'{fontfile}:fontsize=26:fontcolor=white@0.8:"
                 f"x=w-text_w-24:y=24")
    subprocess.run([
        ffmpeg, "-y", "-v", "error",
        "-f", "lavfi", "-i", f"color=c={color}:s=960x540:r=25:d={duration}",
        "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={duration}",
        "-filter_complex", f"[0:v]{text},{countdown}[v];[1:a]volume=0.05[a]",
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-preset", "veryfast", "-tune", "stillimage", "-crf", "30", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "64k", "-movflags", "+faststart", str(dst),
    ], check=True, capture_output=True)


def ensure_creatives(brands: list[dict]) -> list[Path]:
    """Render any missing creative file; returns the paths that were created."""
    created = []
    for brand in brands:
        for creative in brand["creatives"]:
            path = creative_path(creative)
            if not path.exists():
                render_placeholder(brand, creative, path)
                created.append(path)
    return created
