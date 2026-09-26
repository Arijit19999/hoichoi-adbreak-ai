"""Pipeline orchestrator.

Each stage writes a JSON artifact into outputs/<video_id>/ and is skipped when that artifact
already exists, so a crashed or rate-limited run resumes where it stopped. video_id is the
content hash, so results follow the bytes of the video, never its file name.

    uv run python -m app.pipeline.run ../data/samples/mohanagar.mp4 [--force]
"""

import argparse
import json
import logging
import shutil
import time
from collections.abc import Callable
from pathlib import Path

from .. import brands, llm
from ..config import get_settings
from . import media, scenes, speech

Progress = Callable[[str, str], None]


def _print_progress(stage: str, message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {stage:<8} {message}", flush=True)


def write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ingest(src: Path, progress: Progress = _print_progress) -> Path:
    """Hash + probe the upload and create the per-video work dir with playback copy and audio."""
    progress("ingest", f"hashing {src.name}")
    video_id = media.content_hash(src)[:16]
    work = get_settings().outputs / video_id
    work.mkdir(parents=True, exist_ok=True)

    meta_path = work / "meta.json"
    if not meta_path.exists():
        info = media.probe(src)
        if info.audio_codec is None:
            raise ValueError("video has no audio track; dialogue-aware break placement needs audio")
        write_json(meta_path, {"video_id": video_id, "source_name": src.name, **info.to_dict()})

    if not (work / "video.mp4").exists():
        progress("ingest", "preparing playback copy")
        media.make_playback(src, work / "video.mp4", media.probe(src))
    if not (work / "audio.wav").exists():
        progress("ingest", "extracting 16 kHz audio")
        media.extract_audio(src, work / "audio.wav")
    return work


def run_pipeline(src: Path, force: bool = False, progress: Progress = _print_progress) -> Path:
    if force:
        work = get_settings().outputs / media.content_hash(src)[:16]
        shutil.rmtree(work, ignore_errors=True)
    work = ingest(src, progress)

    speech_path = work / "speech.json"
    if not speech_path.exists():
        progress("speech", "detecting speech / silence (VAD)")
        started = time.perf_counter()
        result = speech.detect_speech(work / "audio.wav")
        write_json(speech_path, result)
        progress("speech", f"{len(result['speech'])} speech segments, {len(result['gaps'])} gaps, "
                           f"speech ratio {result['speech_ratio']:.0%} ({time.perf_counter() - started:.1f}s)")

    scenes_path = work / "scenes.json"
    if not scenes_path.exists():
        info = media.probe(work / "video.mp4")
        progress("upload", "uploading video to Gemini")
        remote = llm.upload_video(work / "video.mp4", work / "gemini_file.json")

        raw_path = work / "scenes_raw.json"
        if not raw_path.exists():
            progress("scenes", "segmenting into scenes")
            write_json(raw_path, scenes.segment(remote, work / "video.mp4", info))
        raw = read_json(raw_path)
        progress("scenes", f"{len(raw['scenes'])} scenes from {raw['proposals']} proposals; analysing each scene")

        negative, target = brands.vocabulary(brands.load_catalogue())
        analysed = scenes.understand(remote, raw["scenes"], negative, target)
        write_json(scenes_path, {"negative_vocabulary": negative, "target_vocabulary": target, "scenes": analysed})

    progress("done", str(work))
    return work


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ad-break pipeline on one video")
    parser.add_argument("video", type=Path)
    parser.add_argument("--force", action="store_true", help="discard cached results for this video")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("google_genai").setLevel(logging.WARNING)
    run_pipeline(args.video.resolve(), force=args.force)


if __name__ == "__main__":
    main()
