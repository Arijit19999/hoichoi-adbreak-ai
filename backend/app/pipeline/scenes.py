"""Scene segmentation + scene understanding with Gemini (video + audio), one call per window.

Each overlapping window returns its scenes (clip-relative MM:SS) with a full analysis.
1. Scene starts from all windows are merged and snapped to the exact shot cut / fade.
2. Each final scene takes its description from the window-scene that overlaps it most, and the
   WORST negative-context verdict from every window-scene that overlaps it (safety is a union).
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from google.genai import types
from pydantic import BaseModel, Field

from .. import llm
from .cuts import find_cut_near
from .media import MediaInfo

log = logging.getLogger("scenes")

WINDOW_S = 360.0
OVERLAP_S = 45.0
EDGE_MARGIN_S = 6.0      # scene starts this close to a window edge are left to the overlapping window
MERGE_S = 6.0            # scene starts closer than this are the same boundary
REFINE_WINDOW_S = 4.0
MIN_SCENE_S = 12.0
MIN_OVERLAP_S = 3.0
PARALLEL_CALLS = 3

VERDICT_RANK = {"absent": 0, "possible": 1, "present": 2}


class ContextVerdict(BaseModel):
    context: str
    verdict: Literal["absent", "possible", "present"]
    evidence: str = Field(description="what you saw or heard; empty if absent")


class Activity(BaseModel):
    activity: str
    share: float = Field(ge=0, le=1, description="fraction of the scene's screen time")


class WindowScene(BaseModel):
    start: str = Field(description="MM:SS from the start of THIS clip")
    end: str = Field(description="MM:SS from the start of THIS clip")
    starts_before_clip: bool = Field(description="true if the scene was already running at 00:00")
    continues_after_clip: bool = Field(description="true if the scene is still running when the clip ends")
    start_confidence: float = Field(ge=0, le=1, description="confidence that a NEW scene really starts at `start`")
    start_transition: Literal["hard_cut", "fade", "dissolve", "other"]
    summary: str = Field(description="2-3 sentences, English")
    dialogue_gist: str = Field(description="what is being said, in English, 1-2 sentences")
    setting: str
    time_of_day: str
    characters: list[str]
    dominant_activity: str = Field(description="the ONE activity that occupies most of the scene")
    activities: list[Activity]
    mood: str
    emotional_intensity: int = Field(ge=0, le=10)
    sensitive_events: list[str] = Field(description="death, grief, illness, injury, violence, crime, accident, "
                                                    "bathroom/toilet, nudity, alcohol/drugs, money trouble... if any")
    negative_context_checks: list[ContextVerdict] = Field(description="one entry for EVERY listed negative context")
    target_contexts_present: list[str] = Field(description="listed target contexts that clearly appear")
    ending: Literal["resolved", "open", "cliffhanger", "mid_action"]
    interruptibility: int = Field(ge=0, le=10, description="10 = perfectly natural to cut to a commercial break "
                                                           "right after this scene ends")
    ending_note: str


class WindowAnalysis(BaseModel):
    scenes: list[WindowScene]


SYSTEM = """You are a senior broadcast editor preparing a Bengali TV drama for ad-break planning.
Watch the picture and listen to the Bengali dialogue. Split the clip into SCENES and analyse each one.

A scene is one continuous dramatic unit: same location, continuous time, same core characters. A new scene
starts when the location changes, time jumps, or a clearly new dramatic unit begins. NOT scene changes:
camera angle changes, close-ups, reaction shots, brief cutaways, or cross-cutting between the two sides of
one phone call. Title sequences, credits and recaps are scenes of their own. Timestamps must be precise.

Negative-context checks protect viewers and advertisers. Mark a context "present" if it is shown OR clearly
referred to (characters talking about a death counts as grief), "possible" if there are hints, and "absent"
only when you are confident it is not there. When unsure, choose "possible"."""


def _mmss_to_s(value: str) -> float | None:
    try:
        parts = [float(p) for p in value.strip().split(":")]
    except ValueError:
        return None
    seconds = 0.0
    for p in parts:
        seconds = seconds * 60 + p
    return seconds


def _fmt(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


def _windows(duration: float) -> list[tuple[float, float]]:
    windows, start = [], 0.0
    while True:
        end = min(duration, start + WINDOW_S)
        windows.append((start, end))
        if end >= duration:
            return windows
        start += WINDOW_S - OVERLAP_S


def _timestamp_shift(times: list[tuple[float | None, float | None]], start: float, length: float) -> float:
    """0 if the model answered clip-relative (as asked), `start` if it answered in absolute video time.

    Weaker fallback models sometimes ignore the instruction; pick the reading under which the
    timestamps actually fit inside the clip.
    """
    valid = [(a, b) for a, b in times if a is not None and b is not None]
    if start <= 0 or not valid:
        return 0.0

    def fits(shift: float) -> int:
        return sum(-2 <= a - shift < b - shift <= length + 2 for a, b in valid)

    return start if fits(start) > fits(0.0) else 0.0


def _analyse_window(remote: types.File, start: float, end: float, negative: list[str], target: list[str]) -> list[dict]:
    length = end - start
    result, model = llm.generate_json(
        [llm.video_part(remote, start, end),
         f"This clip is {length:.0f} seconds long: its first frame is 00:00 and its last frame is {_fmt(length)}. "
         f"List every scene in it, in order, covering the whole clip. All timestamps are MM:SS measured from "
         f"the first frame of THIS clip (never from the start of the full episode).\n"
         f"Negative contexts to check for every scene (one verdict each): {negative}\n"
         f"Target contexts (list only those clearly present): {target}"],
        WindowAnalysis,
        system=SYSTEM,
    )
    times = [(_mmss_to_s(s.start), _mmss_to_s(s.end)) for s in result.scenes]
    shift = _timestamp_shift(times, start, length)
    if shift:
        log.warning("%s answered in absolute video time for window %s; converting", model, _fmt(start))
    scenes = []
    for s, (raw_start, raw_end) in zip(result.scenes, times):
        if raw_start is None or raw_end is None:
            continue
        rel_start, rel_end = raw_start - shift, raw_end - shift
        if not (-2 <= rel_start < rel_end <= length + 2):
            log.warning("dropping scene with invalid times %s-%s from %s", s.start, s.end, model)
            continue
        rel_start = max(0.0, rel_start)
        data = s.model_dump()
        checked = {c["context"].lower().strip() for c in data["negative_context_checks"]}
        for context in negative:  # a skipped check must never read as "absent"
            if context not in checked:
                data["negative_context_checks"].append(
                    {"context": context, "verdict": "possible", "evidence": "not answered by model"})
        scenes.append({**data, "abs_start": start + rel_start, "abs_end": start + min(rel_end, length),
                       "window": [start, end], "model": model})
    return scenes


def _boundaries(window_scenes: list[dict], duration: float) -> list[dict]:
    proposals = []
    for s in window_scenes:
        w_start, w_end = s["window"]
        rel = s["abs_start"] - w_start
        if s["starts_before_clip"] or s["abs_start"] <= 0.5:
            continue
        if w_start > 0 and rel < EDGE_MARGIN_S:
            continue
        if w_end < duration and w_end - s["abs_start"] < EDGE_MARGIN_S:
            continue
        proposals.append(s)

    merged: list[dict] = []
    for p in sorted(proposals, key=lambda p: p["abs_start"]):
        if merged and p["abs_start"] - merged[-1]["abs_start"] < MERGE_S:
            if p["start_confidence"] > merged[-1]["start_confidence"]:
                merged[-1] = p
            continue
        merged.append(p)
    return [{"time": p["abs_start"], "confidence": p["start_confidence"],
             "transition": p["start_transition"], "model": p["model"]} for p in merged]


def _combine(scene: dict, window_scenes: list[dict]) -> dict:
    def overlap(ws: dict) -> float:
        return min(scene["end"], ws["abs_end"]) - max(scene["start"], ws["abs_start"])

    sources = [ws for ws in window_scenes if overlap(ws) >= min(MIN_OVERLAP_S, (scene["end"] - scene["start"]) / 2)]
    if not sources:
        return {**scene, "summary": "(not analysed)", "dominant_activity": "unknown", "activities": [],
                "sensitive_events": [], "negative_context_checks": [], "target_contexts_present": [],
                "ending": "mid_action", "interruptibility": 0, "analysis_missing": True}

    main = max(sources, key=overlap)
    closing = min(sources, key=lambda ws: abs(ws["abs_end"] - scene["end"]))

    worst: dict[str, dict] = {}
    for ws in sources:
        for check in ws["negative_context_checks"]:
            key = check["context"].lower().strip()
            if key not in worst or VERDICT_RANK[check["verdict"]] > VERDICT_RANK[worst[key]["verdict"]]:
                worst[key] = {**check, "context": key}

    return {
        **scene,
        **{k: main[k] for k in ("summary", "dialogue_gist", "setting", "time_of_day", "characters",
                                "dominant_activity", "activities", "mood")},
        "emotional_intensity": max(ws["emotional_intensity"] for ws in sources),
        "sensitive_events": sorted({e for ws in sources for e in ws["sensitive_events"]}),
        "negative_context_checks": sorted(worst.values(), key=lambda c: c["context"]),
        "target_contexts_present": sorted({c.lower() for ws in sources for c in ws["target_contexts_present"]}),
        **{k: closing[k] for k in ("ending", "interruptibility", "ending_note")},
        "models": sorted({ws["model"] for ws in sources}),
        "source_windows": [[round(ws["abs_start"], 1), round(ws["abs_end"], 1)] for ws in sources],
    }


def analyse(remote: types.File, video: Path, info: MediaInfo, negative: list[str], target: list[str]) -> dict:
    duration = info.duration
    windows = _windows(duration)
    with ThreadPoolExecutor(PARALLEL_CALLS) as pool:
        per_window = list(pool.map(lambda w: _analyse_window(remote, *w, negative, target), windows))
    window_scenes = [s for chunk in per_window for s in chunk]

    # Snap each boundary to the real frame-accurate cut.
    boundaries = []
    for b in _boundaries(window_scenes, duration):
        cut = find_cut_near(video, b["time"], info, window=REFINE_WINDOW_S)
        boundaries.append({**b, "proposed_time": round(b["time"], 2),
                           "time": cut.time if cut else b["time"], "cut": cut.to_dict() if cut else None})
    boundaries.sort(key=lambda b: b["time"])

    # Drop the weakest boundary around any scene that is too short, until none are.
    while boundaries:
        edges = [0.0, *[b["time"] for b in boundaries], duration]
        short = [i for i in range(len(edges) - 1) if edges[i + 1] - edges[i] < MIN_SCENE_S]
        if not short:
            break
        i = short[0]
        candidates = [j for j in (i - 1, i) if 0 <= j < len(boundaries)]
        boundaries.pop(min(candidates, key=lambda j: boundaries[j]["confidence"]))

    edges = [0.0, *[b["time"] for b in boundaries], duration]
    scenes = [
        _combine({"index": i, "start": round(edges[i], 3), "end": round(edges[i + 1], 3),
                  "boundary_in": boundaries[i - 1] if i > 0 else None}, window_scenes)
        for i in range(len(edges) - 1)
    ]
    return {
        "windows": windows,
        "negative_vocabulary": negative,
        "target_vocabulary": target,
        "window_scenes": window_scenes,
        "scenes": scenes,
    }
