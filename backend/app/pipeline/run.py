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

from .. import ads, brands, llm
from ..config import get_settings
from . import breaks, media, plan, scenes, speech, vmap

Progress = Callable[[str, str], None]


def _print_progress(stage: str, message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {stage:<8} {message}", flush=True)


def write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ingest(src: Path, progress: Progress = _print_progress, source_name: str | None = None) -> Path:
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
        write_json(meta_path, {"video_id": video_id, "source_name": source_name or src.name, **info.to_dict()})

    if not (work / "video.mp4").exists():
        progress("ingest", "preparing playback copy")
        media.make_playback(src, work / "video.mp4", media.probe(src))
    if not (work / "audio.wav").exists():
        progress("ingest", "extracting 16 kHz audio")
        media.extract_audio(src, work / "audio.wav")
    return work


def run_pipeline(src: Path, force: bool = False, replan: bool = False, rebuild_scenes: bool = False,
                 progress: Progress = _print_progress, source_name: str | None = None) -> Path:
    if force:
        work = get_settings().outputs / media.content_hash(src)[:16]
        shutil.rmtree(work, ignore_errors=True)
    work = ingest(src, progress, source_name)

    speech_path = work / "speech.json"
    if not speech_path.exists():
        progress("speech", "detecting speech / silence (VAD)")
        started = time.perf_counter()
        result = speech.detect_speech(work / "audio.wav")
        write_json(speech_path, result)
        progress("speech", f"{len(result['speech'])} speech segments, {len(result['gaps'])} gaps, "
                           f"speech ratio {result['speech_ratio']:.0%} ({time.perf_counter() - started:.1f}s)")

    scenes_path = work / "scenes.json"
    if rebuild_scenes and scenes_path.exists():
        progress("scenes", "rebuilding scenes from cached model answers (no new model calls)")
        result = scenes.build_scenes(read_json(scenes_path), work / "video.mp4", media.probe(work / "video.mp4"),
                                     read_json(speech_path)["gaps"])
        write_json(scenes_path, result)
        progress("scenes", f"{len(result['scenes'])} scenes, {sum(len(s['beats']) for s in result['scenes'])} beats, "
                           f"{len(result['dropped_boundaries'])} boundaries dropped")
        replan = True
    if not scenes_path.exists():
        info = media.probe(work / "video.mp4")
        upload = work / "video.mp4"
        proxy_height = get_settings().gemini_proxy_height
        if proxy_height > 0:
            upload = work / "analysis_proxy.mp4"
            if not upload.exists():
                progress("upload", f"making {proxy_height}p analysis proxy")
                media.make_analysis_proxy(work / "video.mp4", upload, proxy_height)
        progress("upload", f"uploading {upload.stat().st_size / 1e6:.0f} MB to Gemini")
        remote = llm.upload_video(upload, work / "gemini_file.json")

        progress("scenes", "segmenting and analysing scenes")
        started = time.perf_counter()
        negative, target = brands.vocabulary(brands.load_catalogue())
        result = scenes.analyse(remote, work / "video.mp4", info, negative, target,
                                gaps=read_json(speech_path)["gaps"])
        write_json(scenes_path, result)
        models = sorted({m for s in result["scenes"] for m in s.get("models", [])})
        progress("scenes", f"{len(result['scenes'])} scenes, {sum(len(s['beats']) for s in result['scenes'])} "
                           f"in-scene beats from {len(result['windows'])} windows "
                           f"via {models} ({time.perf_counter() - started:.0f}s)")

    if replan or not (work / "breaks.json").exists():
        make_plan(work, progress)

    progress("done", str(work))
    return work


def make_plan(work: Path, progress: Progress = _print_progress, rules: breaks.PacingRules | None = None,
              catalogue: list[dict] | None = None, use_asr: bool = True) -> dict:
    """Break plan + manifest from cached analysis. Re-run alone when the catalogue or rules change."""
    catalogue = catalogue if catalogue is not None else brands.load_catalogue()
    rules = rules or breaks.PacingRules()
    progress("plan", "scoring break candidates, matching brands, auditing")
    started = time.perf_counter()
    ads.ensure_creatives(catalogue)
    result, debug = plan.build_plan(work, read_json(work / "scenes.json"), read_json(work / "speech.json"),
                                    read_json(work / "meta.json")["duration"], catalogue, rules, use_asr=use_asr)
    write_json(work / "breaks.json", result)
    write_json(work / "debug.json", debug)
    (work / "manifest.vmap.xml").write_bytes(vmap.build_vmap(result, get_settings().public_base_url))
    placed = ", ".join(f"{b['id']}@{b['time']:.1f}s={b['brand_id']}" for b in result["breaks"]) or "none"
    progress("plan", f"{len(result['breaks'])} breaks ({placed}), ad load {result['ad_load_pct']}% "
                     f"({time.perf_counter() - started:.0f}s)")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ad-break pipeline on one video")
    parser.add_argument("video", type=Path)
    parser.add_argument("--force", action="store_true", help="discard cached results for this video")
    parser.add_argument("--replan", action="store_true", help="recompute breaks / brands / manifest only")
    parser.add_argument("--rebuild-scenes", action="store_true",
                        help="recompute scene boundaries from cached model answers, then re-plan")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("google_genai").setLevel(logging.WARNING)
    run_pipeline(args.video.resolve(), force=args.force, replan=args.replan, rebuild_scenes=args.rebuild_scenes)


if __name__ == "__main__":
    main()
