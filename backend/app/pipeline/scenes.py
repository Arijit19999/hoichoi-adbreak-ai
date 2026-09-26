"""Scene segmentation and scene understanding with Gemini (video + audio).

1. segment():    overlapping windows -> scene-start proposals (MM:SS, clip-relative)
                 -> absolute time -> merged -> snapped to the exact shot cut / fade.
2. understand(): one call per final scene -> summary, dominant activity, mood, and an explicit
                 absent/possible/present verdict for every negative context in the catalogue.
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

WINDOW_S = 300.0
OVERLAP_S = 40.0
EDGE_MARGIN_S = 6.0      # proposals this close to a window edge are left to the overlapping window
MERGE_S = 6.0            # proposals closer than this are the same boundary
REFINE_WINDOW_S = 4.0
MIN_SCENE_S = 12.0
PARALLEL_CALLS = 4


# ---------- 1. segmentation ----------

class BoundaryProposal(BaseModel):
    timestamp: str = Field(description="MM:SS from the start of THIS clip where the new scene starts")
    confidence: float = Field(ge=0, le=1)
    transition: Literal["hard_cut", "fade", "dissolve", "other"]
    before: str = Field(description="the scene that ends, in a few words")
    after: str = Field(description="the scene that starts, in a few words")


class WindowSegmentation(BaseModel):
    boundaries: list[BoundaryProposal]


SEGMENT_SYSTEM = """You are a senior broadcast editor segmenting a Bengali TV drama into SCENES for ad-break planning.
A scene is one continuous dramatic unit: the same location, continuous time, and the same core characters.
A new scene starts when the location changes, time jumps, or a clearly new dramatic unit begins.
NOT scene changes: camera angle changes, close-ups, reaction shots, brief cutaways, or cross-cutting between
the two sides of one phone call. Title cards, credits and recaps are their own scenes.
Use both the picture and the audio (dialogue, music cues). Be precise with timestamps."""


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


def _segment_window(remote: types.File, start: float, end: float, duration: float) -> list[dict]:
    length = end - start
    result, model = llm.generate_json(
        [llm.video_part(remote, start, end),
         f"This clip is {length:.0f} seconds long (00:00 to {_fmt(length)}). List the start of every NEW scene "
         f"inside it (not 00:00 itself). Timestamps are MM:SS measured from the start of this clip."],
        WindowSegmentation,
        system=SEGMENT_SYSTEM,
    )
    lo = 0.0 if start <= 0 else EDGE_MARGIN_S
    hi = length if end >= duration else length - EDGE_MARGIN_S
    proposals = []
    for b in result.boundaries:
        rel = _mmss_to_s(b.timestamp)
        if rel is None or not lo < rel < hi:
            continue  # unparseable or outside the clip (seen with weaker fallback models)
        proposals.append({**b.model_dump(), "time": start + rel, "model": model})
    return proposals


def segment(remote: types.File, video: Path, info: MediaInfo) -> dict:
    duration = info.duration
    windows, start = [], 0.0
    while start < duration:
        windows.append((start, min(duration, start + WINDOW_S)))
        if start + WINDOW_S >= duration:
            break
        start += WINDOW_S - OVERLAP_S

    with ThreadPoolExecutor(PARALLEL_CALLS) as pool:
        proposals = [p for chunk in pool.map(lambda w: _segment_window(remote, *w, duration), windows) for p in chunk]

    # Merge duplicates from overlapping windows, keeping the most confident proposal.
    merged: list[dict] = []
    for p in sorted(proposals, key=lambda p: p["time"]):
        if merged and p["time"] - merged[-1]["time"] < MERGE_S:
            if p["confidence"] > merged[-1]["confidence"]:
                merged[-1] = p
            continue
        merged.append(p)

    # Snap each boundary to the real frame-accurate cut.
    boundaries = []
    for p in merged:
        cut = find_cut_near(video, p["time"], info, window=REFINE_WINDOW_S)
        boundaries.append({
            "time": cut.time if cut else p["time"],
            "proposed_time": round(p["time"], 2),
            "cut": cut.to_dict() if cut else None,
            "confidence": p["confidence"],
            "transition": p["transition"],
            "before": p["before"],
            "after": p["after"],
            "model": p["model"],
        })
    boundaries.sort(key=lambda b: b["time"])

    # Drop the weakest boundary around any scene that is too short, until none are.
    while True:
        edges = [0.0, *[b["time"] for b in boundaries], duration]
        short = [i for i in range(len(edges) - 1) if edges[i + 1] - edges[i] < MIN_SCENE_S]
        if not short or not boundaries:
            break
        i = short[0]
        candidates = [j for j in (i - 1, i) if 0 <= j < len(boundaries)]
        boundaries.pop(min(candidates, key=lambda j: boundaries[j]["confidence"]))

    edges = [0.0, *[b["time"] for b in boundaries], duration]
    scenes = [
        {"index": i, "start": round(edges[i], 3), "end": round(edges[i + 1], 3),
         "boundary_in": boundaries[i - 1] if i > 0 else None}
        for i in range(len(edges) - 1)
    ]
    return {"windows": windows, "proposals": len(proposals), "scenes": scenes}


# ---------- 2. understanding ----------

class ContextVerdict(BaseModel):
    context: str
    verdict: Literal["absent", "possible", "present"]
    evidence: str = Field(description="what you saw or heard; empty if absent")


class Activity(BaseModel):
    activity: str
    share: float = Field(ge=0, le=1, description="fraction of the scene's screen time")


class SceneUnderstanding(BaseModel):
    summary: str = Field(description="2-3 sentences, English")
    dialogue_gist: str = Field(description="what is being said, in English, 1-2 sentences")
    setting: str
    time_of_day: str
    characters: list[str]
    dominant_activity: str = Field(description="the ONE activity that occupies most of the scene")
    activities: list[Activity]
    mood: str
    emotional_intensity: int = Field(ge=0, le=10)
    sensitive_events: list[str] = Field(description="death, grief, illness, injury, violence, crime, accidents, "
                                                    "bathroom, nudity, substance use, financial distress... if any")
    negative_context_checks: list[ContextVerdict] = Field(description="one entry for EVERY listed negative context")
    target_contexts_present: list[str] = Field(description="listed target contexts that clearly appear")
    ending: Literal["resolved", "open", "cliffhanger", "mid_action"]
    interruptibility: int = Field(ge=0, le=10, description="10 = a perfectly natural place for a commercial break "
                                                           "right after this scene ends")
    ending_note: str


UNDERSTAND_SYSTEM = """You analyse scenes of Bengali TV dramas for a broadcaster's ad-placement system.
Watch the picture and listen to the Bengali dialogue. Be literal and evidence-based.
Negative-context checks protect viewers and advertisers: mark a context "present" if it is shown OR clearly
referred to (e.g. characters talking about a death counts as grief/funeral context), "possible" if there are
hints, and "absent" only when you are confident it is not there. When unsure, choose "possible"."""


def _understand_scene(remote: types.File, scene: dict, negative: list[str], target: list[str]) -> dict:
    start, end = scene["start"], scene["end"]
    result, model = llm.generate_json(
        [llm.video_part(remote, start, end, fps=1.0),
         "Analyse this scene.\n"
         f"Negative contexts to check (one verdict each): {negative}\n"
         f"Target contexts (list only those clearly present): {target}"],
        SceneUnderstanding,
        system=UNDERSTAND_SYSTEM,
    )
    data = result.model_dump()
    checked = {c["context"].lower().strip() for c in data["negative_context_checks"]}
    for context in negative:  # a skipped check must never read as "absent"
        if context not in checked:
            data["negative_context_checks"].append(
                {"context": context, "verdict": "possible", "evidence": "not answered by model"})
    return {**scene, **data, "model": model}


def understand(remote: types.File, scenes: list[dict], negative: list[str], target: list[str]) -> list[dict]:
    with ThreadPoolExecutor(PARALLEL_CALLS) as pool:
        return list(pool.map(lambda s: _understand_scene(remote, s, negative, target), scenes))
