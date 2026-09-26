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
WIDE_REFINE_WINDOW_S = 8.0
MIN_SCENE_S = 12.0
MIN_OVERLAP_S = 3.0
PARALLEL_CALLS = 3

VERDICT_RANK = {"absent": 0, "possible": 1, "present": 2}

BEAT_EDGE_S = 20.0           # beats this close to a scene boundary are redundant
LONG_SCENE_S = 360.0         # scenes longer than this with no AI beats get silence-based pause points
PAUSE_MIN_GAP_S = 1.2
PAUSES_PER_LONG_SCENE = 4


class ContextVerdict(BaseModel):
    context: str
    verdict: Literal["absent", "possible", "present"]
    evidence: str = Field(description="what you saw or heard; empty if absent")


class Activity(BaseModel):
    activity: str
    share: float = Field(ge=0, le=1, description="fraction of the scene's screen time")


class Beat(BaseModel):
    timestamp: str = Field(description="MM:SS from the start of THIS clip")
    description: str
    strength: float = Field(ge=0, le=1, description="1 = an obvious, satisfying pause point")


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
    beats: list[Beat] = Field(description="moments INSIDE this scene where the action clearly wraps up and a "
                                          "commercial break would feel natural (a round / dish / topic / "
                                          "conversation ends); empty if none")


class WindowAnalysis(BaseModel):
    scenes: list[WindowScene]


SYSTEM = """You are a senior broadcast editor preparing a Bengali TV drama for ad-break planning.
Watch the picture and listen to the Bengali dialogue. Split the clip into SCENES and analyse each one.

A scene is one continuous dramatic unit: same location, continuous time, same core characters. A new scene
starts when the location changes, time jumps, or a clearly new dramatic unit begins. NOT scene changes:
camera angle changes, close-ups, reaction shots, brief cutaways, or cross-cutting between the two sides of
one phone call. Title sequences, credits and recaps are scenes of their own. Timestamps must be precise.
The video may also be non-fiction (cooking, game, travel or talk show): there, each new segment (a new round,
dish, guest, topic or location) is a new scene. Inside long scenes, also mark "beats": moments where a piece of
action clearly wraps up (a round or dish is finished, a conversation concludes) so a break would feel natural.

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
        beats = []
        for beat in data.pop("beats", []):
            rel = _mmss_to_s(beat["timestamp"])
            if rel is not None and rel_start + 5 < rel - shift < rel_end - 5:
                beats.append({**beat, "time": start + rel - shift})
        scenes.append({**data, "beats": beats, "abs_start": start + rel_start, "abs_end": start + min(rel_end, length),
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


def analyse(remote: types.File, video: Path, info: MediaInfo, negative: list[str], target: list[str],
            gaps: list[list[float]] | None = None) -> dict:
    windows = _windows(info.duration)
    with ThreadPoolExecutor(PARALLEL_CALLS) as pool:
        per_window = list(pool.map(lambda w: _analyse_window(remote, *w, negative, target), windows))
    window_scenes = [s for chunk in per_window for s in chunk]
    return build_scenes(
        {"windows": windows, "negative_vocabulary": negative, "target_vocabulary": target,
         "window_scenes": window_scenes},
        video, info, gaps or [],
    )


def _refine(video: Path, info: MediaInfo, b: dict) -> dict:
    """Snap a proposed boundary to the real cut, widening the search once.

    A boundary with no cut nearby is kept as a scene division (it still separates the contexts on
    either side, which matters for brand safety) but can never host a break.
    """
    cut = find_cut_near(video, b["time"], info, window=REFINE_WINDOW_S)
    if cut is None:
        cut = find_cut_near(video, b["time"], info, window=WIDE_REFINE_WINDOW_S)
    return {**b, "proposed_time": round(b["time"], 2),
            "time": cut.time if cut else b["time"], "cut": cut.to_dict() if cut else None}


def build_scenes(result: dict, video: Path, info: MediaInfo, gaps: list[list[float]]) -> dict:
    """Final scenes from the per-window model answers (re-runnable from cache, no model calls)."""
    duration = info.duration
    window_scenes = result["window_scenes"]

    boundaries = sorted((_refine(video, info, b) for b in _boundaries(window_scenes, duration)),
                        key=lambda b: b["time"])
    dropped = []

    # Drop the weakest boundary around any scene that is too short, until none are.
    while boundaries:
        edges = [0.0, *[b["time"] for b in boundaries], duration]
        short = [i for i in range(len(edges) - 1) if edges[i + 1] - edges[i] < MIN_SCENE_S]
        if not short:
            break
        i = short[0]
        candidates = [j for j in (i - 1, i) if 0 <= j < len(boundaries)]
        dropped.append({**boundaries.pop(min(candidates, key=lambda j: boundaries[j]["confidence"])),
                        "reason": f"scene shorter than {MIN_SCENE_S:.0f}s"})

    edges = [0.0, *[b["time"] for b in boundaries], duration]
    scenes = [
        _combine({"index": i, "start": round(edges[i], 3), "end": round(edges[i + 1], 3),
                  "boundary_in": boundaries[i - 1] if i > 0 else None}, window_scenes)
        for i in range(len(edges) - 1)
    ]
    _attach_beats(scenes, window_scenes, video, info, gaps)
    return {**result, "dropped_boundaries": dropped, "scenes": scenes}


def _attach_beats(scenes: list[dict], window_scenes: list[dict], video: Path, info: MediaInfo,
                  gaps: list[list[float]]) -> None:
    """Pause points inside scenes: AI-marked beats, or long silences on a cut for long beat-less scenes.

    Long continuous formats (cooking / game / talk shows) can run 10+ minutes without a scene change;
    these give the break planner somewhere natural to cut.
    """
    proposals = sorted((b for ws in window_scenes for b in ws.get("beats", [])), key=lambda b: b["time"])
    merged: list[dict] = []
    for b in proposals:
        if merged and b["time"] - merged[-1]["time"] < MERGE_S:
            if b["strength"] > merged[-1]["strength"]:
                merged[-1] = b
            continue
        merged.append(b)

    for scene in scenes:
        scene["beats"] = []
    for b in merged:
        cut = find_cut_near(video, b["time"], info, window=3.0)
        if cut is None:
            continue
        scene = next((s for s in scenes if s["start"] + BEAT_EDGE_S < cut.time < s["end"] - BEAT_EDGE_S), None)
        if scene is not None:
            scene["beats"].append({"time": cut.time, "proposed_time": round(b["time"], 2), "cut": cut.to_dict(),
                                   "strength": b["strength"], "description": b["description"], "source": "ai"})

    for scene in scenes:
        if scene["beats"] or scene["end"] - scene["start"] < LONG_SCENE_S:
            continue
        inside = [g for g in gaps if scene["start"] + 30 < g[0] and g[1] < scene["end"] - 30
                  and g[1] - g[0] >= PAUSE_MIN_GAP_S]
        for g_start, g_end in sorted(inside, key=lambda g: g[1] - g[0], reverse=True)[:PAUSES_PER_LONG_SCENE]:
            mid = (g_start + g_end) / 2
            cut = find_cut_near(video, mid, info, window=min(2.5, (g_end - g_start) / 2))
            if cut and g_start + 0.4 <= cut.time <= g_end - 0.15:
                scene["beats"].append({"time": cut.time, "proposed_time": round(mid, 2), "cut": cut.to_dict(),
                                       "strength": round(min(0.6, (g_end - g_start) / 5), 2),
                                       "description": f"{g_end - g_start:.1f}s pause in dialogue on a shot change",
                                       "source": "silence"})
        scene["beats"].sort(key=lambda b: b["time"])
