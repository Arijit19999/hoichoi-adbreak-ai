"""Independent self-check of every processed video against the judging rules. No model calls.

    uv run python -m app.verify            # all processed videos
    uv run python -m app.verify <video_id>

Checks each placed break the way a judge would:
  cut        a real shot change / fade at the break time (re-measured from the frames)
  dialogue   nobody is speaking at the cut (speech timeline)
  safety     the brand's negative contexts are not present/possible before, nor present after
  pacing     count, gap, position and ad-load rules of the plan
  manifest   VMAP parses, matches the plan, and every creative file exists with the right length
  scenes     scenes tile the whole video without gaps or overlaps
"""

import json
import subprocess
import sys
from pathlib import Path

from lxml import etree

from .ads import creative_path
from .brands import load_catalogue
from .config import get_settings
from .pipeline import breaks as breaks_mod
from .pipeline.cuts import find_cut_near
from .pipeline.matching import safety
from .pipeline.media import _bin, probe

VMAP_NS = "{http://www.iab.net/videosuite/vmap}"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _offset(value: str) -> float:
    h, m, s = value.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _media_duration(path: Path) -> float:
    out = subprocess.run([_bin("ffprobe"), "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                          str(path)], capture_output=True, text=True)
    return float(out.stdout.strip() or 0)


def verify(work: Path, catalogue: list[dict]) -> list[tuple[str, bool, str]]:
    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, ok, detail))

    plan = _load(work / "breaks.json")
    scenes = _load(work / "scenes.json")["scenes"]
    speech = _load(work / "speech.json")["speech"]
    duration = plan["duration"]
    brands = {b["brand_id"]: b for b in catalogue}
    info = probe(work / "video.mp4")

    # scenes tile the video
    tiled = abs(scenes[0]["start"]) < 0.01 and abs(scenes[-1]["end"] - duration) < 0.5 and all(
        abs(a["end"] - b["start"]) < 0.01 for a, b in zip(scenes, scenes[1:]))
    check("scenes", tiled, f"{len(scenes)} scenes cover 0-{duration:.0f}s")

    for brk in plan["breaks"]:
        t, tag = brk["time"], f"{brk['id']}@{brk['time']:.1f}s"

        cut = find_cut_near(work / "video.mp4", t, info, window=0.4)
        check(f"cut {tag}", cut is not None and abs(cut.time - t) <= 0.08,
              f"{cut.kind} at {cut.time:.2f}s" if cut else "no shot change at the break")

        before = [e for s, e in speech if e <= t]
        after = [s for s, e in speech if s >= t]
        crossing = [(s, e) for s, e in speech if s < t < e]
        gap_before = t - max(before) if before else t
        check(f"dialogue {tag}", not crossing and gap_before >= breaks_mod.PRE_SILENCE_S,
              f"last speech ends {gap_before:.2f}s before, next starts "
              f"{(min(after) - t) if after else float('inf'):.2f}s after" + (" - SPEECH ACROSS CUT" if crossing else ""))

        brand = brands.get(brk["brand_id"])
        if brand is None:
            check(f"safety {tag}", False, f"{brk['brand_id']} is not in the catalogue")
        else:
            reasons = safety(scenes[brk["scene_before"]], scenes[brk["scene_after"]], brand)
            check(f"safety {tag}", not reasons, f"{brk['brand_id']}: " + ("; ".join(reasons) if reasons else
                  f"none of {brand['negative_contexts']} near the break"))

    rules = plan["rules"]
    times = [b["time"] for b in plan["breaks"]]
    limit = int(duration / 3600 * rules["max_breaks_per_hour"] + 1e-9)
    gaps = [b - a for a, b in zip(times, times[1:])]
    ad_load = sum(b["creative"]["duration_sec"] for b in plan["breaks"])
    check("pacing count", len(times) <= limit, f"{len(times)} breaks, max {limit}")
    check("pacing gap", all(g >= rules["min_gap_s"] for g in gaps),
          f"gaps {[round(g) for g in gaps]}s, min {rules['min_gap_s']:.0f}s")
    check("pacing position", all(rules["no_break_before_s"] <= x <= duration - rules["no_break_in_last_s"]
                                 for x in times), f"breaks at {[round(x) for x in times]}s")
    check("ad load", ad_load <= rules["max_ad_load_pct"] / 100 * duration,
          f"{ad_load}s = {100 * ad_load / duration:.2f}% (max {rules['max_ad_load_pct']}%)")

    try:
        root = etree.parse(str(work / "manifest.vmap.xml")).getroot()
        ad_breaks = root.findall(f"{VMAP_NS}AdBreak")
        offsets = [_offset(b.get("timeOffset")) for b in ad_breaks]
        matches = len(ad_breaks) == len(plan["breaks"]) and all(
            abs(o - b["time"]) < 0.002 for o, b in zip(offsets, plan["breaks"]))
        required = all(b.find(f".//{tag}") is not None for b in ad_breaks
                       for tag in ("AdSystem", "AdTitle", "Impression", "Duration", "MediaFile"))
        check("manifest", root.tag == f"{VMAP_NS}VMAP" and matches and required,
              f"{len(ad_breaks)} AdBreak(s) at {[round(o, 3) for o in offsets]}")
    except (OSError, etree.XMLSyntaxError, ValueError) as e:
        check("manifest", False, str(e))

    for brk in plan["breaks"]:
        path = creative_path(brk["creative"])
        length = _media_duration(path) if path.exists() else 0
        check(f"creative {brk['creative']['id']}", path.exists() and abs(length - brk["creative"]["duration_sec"]) < 0.6,
              f"{path.name}: {length:.1f}s (declared {brk['creative']['duration_sec']}s)")
    return results


def main() -> int:
    outputs = get_settings().outputs
    wanted = sys.argv[1:]
    catalogue = load_catalogue()
    failures = 0
    for plan_path in sorted(outputs.glob("*/breaks.json")):
        work = plan_path.parent
        if wanted and work.name not in wanted:
            continue
        meta = _load(work / "meta.json")
        print(f"\n== {meta['source_name']} ({work.name}, {meta['duration'] / 60:.1f} min)")
        for name, ok, detail in verify(work, catalogue):
            failures += not ok
            print(f"  {'PASS' if ok else 'FAIL'}  {name:<28} {detail}")
    print(f"\n{'ALL CHECKS PASSED' if failures == 0 else f'{failures} CHECK(S) FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
