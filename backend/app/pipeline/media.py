"""ffmpeg / ffprobe helpers: probing, hashing, audio extraction, frame sampling.

Everything here is written to stay cheap on a small server (0.1 CPU / 512 MB):
full-length work is audio-only; video frames are only decoded in short windows.
"""

import hashlib
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

AUDIO_SR = 16_000
THUMB_WIDTH = 160


def _bin(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"{name} not found on PATH")
    return path


def _run(cmd: list[str]) -> bytes:
    if Path(cmd[0]).stem == "ffmpeg":
        # Containers report the host's core count; unbounded threads mean unbounded frame buffers.
        cmd = [cmd[0], "-threads", "2", *cmd[1:]]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", "replace")[-2000:]
        raise RuntimeError(f"{Path(cmd[0]).stem} failed ({proc.returncode}): {err}")
    return proc.stdout


@dataclass
class MediaInfo:
    duration: float
    width: int
    height: int
    fps: float
    video_codec: str
    audio_codec: str | None
    container: str

    def to_dict(self) -> dict:
        return asdict(self)


def probe(path: Path) -> MediaInfo:
    out = _run([
        _bin("ffprobe"), "-v", "error",
        "-show_entries", "format=duration,format_name:stream=codec_type,codec_name,width,height,avg_frame_rate",
        "-of", "json", str(path),
    ])
    data = json.loads(out)
    video = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
    if video is None:
        raise ValueError(f"{path.name} has no video stream")
    audio = next((s for s in data["streams"] if s["codec_type"] == "audio"), None)
    num, den = (video.get("avg_frame_rate") or "25/1").split("/")
    fps = float(num) / float(den) if float(den) else 25.0
    return MediaInfo(
        duration=float(data["format"]["duration"]),
        width=int(video["width"]),
        height=int(video["height"]),
        fps=fps,
        video_codec=video["codec_name"],
        audio_codec=audio["codec_name"] if audio else None,
        container=data["format"]["format_name"],
    )


def content_hash(path: Path) -> str:
    """Hash of the file bytes: cache key, so results can never be tied to a file *name*."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def extract_audio(src: Path, dst: Path) -> None:
    """Mono 16 kHz 16-bit WAV for VAD / ASR / loudness."""
    _run([
        _bin("ffmpeg"), "-y", "-v", "error", "-i", str(src),
        "-vn", "-ac", "1", "-ar", str(AUDIO_SR), "-c:a", "pcm_s16le", str(dst),
    ])


def make_playback(src: Path, dst: Path, info: MediaInfo) -> None:
    """Browser-playable MP4 with the moov atom up front.

    H.264/AAC MP4 input is only remuxed (no re-encode); anything else is transcoded to 540p.
    """
    if info.video_codec == "h264" and info.audio_codec in ("aac", None) and "mp4" in info.container:
        codec_args = ["-c", "copy"]
    else:
        codec_args = [
            "-vf", "scale=-2:'min(540,ih)'", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
            "-c:a", "aac", "-b:a", "128k",
        ]
    _run([_bin("ffmpeg"), "-y", "-v", "error", "-i", str(src), *codec_args, "-movflags", "+faststart", str(dst)])


def read_gray_frames(src: Path, start: float, duration: float, info: MediaInfo) -> tuple[np.ndarray, np.ndarray]:
    """Decode a short window as small grayscale frames.

    Returns (times, frames[n, h, w] uint8). Input-side -ss seeks accurately in modern ffmpeg.
    """
    start = max(0.0, start)
    height = max(2, round(THUMB_WIDTH * info.height / info.width / 2) * 2)
    raw = _run([
        _bin("ffmpeg"), "-v", "error", "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{duration:.3f}",
        "-an", "-vf", f"scale={THUMB_WIDTH}:{height},format=gray", "-f", "rawvideo", "-",
    ])
    frame_size = THUMB_WIDTH * height
    n = len(raw) // frame_size
    frames = np.frombuffer(raw[: n * frame_size], dtype=np.uint8).reshape(n, height, THUMB_WIDTH)
    times = start + np.arange(n) / info.fps
    return times, frames


def extract_jpeg(src: Path, t: float, dst: Path, width: int = 480) -> None:
    """Single frame as JPEG (thumbnails for the UI and frames for LLM prompts)."""
    _run([
        _bin("ffmpeg"), "-y", "-v", "error", "-ss", f"{max(0.0, t):.3f}", "-i", str(src),
        "-frames:v", "1", "-vf", f"scale={width}:-2", "-q:v", "4", str(dst),
    ])


def make_analysis_proxy(src: Path, dst: Path, height: int) -> None:
    """Small same-timeline copy for upload to the scene model (it samples ~1 fps at low resolution)."""
    _run([
        _bin("ffmpeg"), "-y", "-v", "error", "-i", str(src),
        "-vf", f"scale=-2:{height},fps=5", "-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
        "-c:a", "aac", "-b:a", "48k", "-ac", "1", "-movflags", "+faststart", str(dst),
    ])
