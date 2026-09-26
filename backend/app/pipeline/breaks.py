"""Break candidates, scoring and pacing-constrained selection.

Every scene boundary is a candidate. Hard gates (any failure = rejected, with the reason kept):
  * the boundary sits on a real shot cut or fade (frame-accurate)
  * no speech across the cut, per VAD
  * no speech across the cut, per word-timestamped ASR (second opinion on quiet / music-covered lines)
Survivors get a 0-1 score; selection then maximises total score under the pacing rules.
"""

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .. import llm
from .speech import wav_clip

log = logging.getLogger("breaks")

PRE_SILENCE_S = 0.4      # no speech in the last 0.4 s before the cut
POST_SILENCE_S = 0.15    # ... nor starting within 0.15 s after it
ASR_LOOKBACK_S = 8.0

ENDING_SCORE = {"resolved": 1.0, "cliffhanger": 0.75, "open": 0.6, "mid_action": 0.0}
TRANSITION_SCORE = {"fade_black": 1.0, "hard_cut": 0.6}
WEIGHTS = {
    "interruptibility": 0.35, "ending": 0.15, "transition": 0.15,
    "silence_before": 0.15, "silence_after": 0.05, "loudness_dip": 0.10, "boundary_confidence": 0.05,
}


@dataclass
class PacingRules:
    max_breaks_per_hour: float = 6.0
    min_gap_s: float = 300.0
    max_ad_load_pct: float = 10.0
    no_break_before_s: float = 180.0
    no_break_in_last_s: float = 90.0
    min_break_score: float = 0.45

    def to_dict(self) -> dict:
        return asdict(self)

    def max_breaks(self, duration: float) -> int:
        return int(duration / 3600 * self.max_breaks_per_hour + 1e-9)


def _speech_around(speech: list[list[float]], t: float) -> tuple[float, float, list[float] | None]:
    """(last speech end before t, next speech start after t, first speech segment inside the guard window)."""
    last_end, next_start, crossing = 0.0, float("inf"), None
    for s, e in speech:
        if crossing is None and s < t + POST_SILENCE_S and e > t - PRE_SILENCE_S:
            crossing = [s, e]
        if e <= t:
            last_end = max(last_end, e)
        if s >= t:
            next_start = min(next_start, s)
    return last_end, next_start, crossing


def _asr_speech_at_cut(wav: Path, t: float) -> tuple[bool | None, str]:
    """Word-level check that nobody is mid-word at the cut. None = ASR unavailable."""
    start = max(0.0, t - ASR_LOOKBACK_S)
    result = llm.transcribe(wav_clip(wav, start, t + 1.0))
    if result is None:
        return None, "asr unavailable"
    words = result.get("words") or []
    for w in words:
        w_start, w_end = start + w["start"], start + w["end"]
        if w_start < t + POST_SILENCE_S and w_end > t - PRE_SILENCE_S / 2:
            return True, f"word '{w['word'].strip()}' at {w_start:.2f}-{w_end:.2f}s crosses the cut"
    tail = " ".join(w["word"].strip() for w in words[-6:])
    return False, f"last words before cut: …{tail}" if tail else "no words detected"


def _loudness_dip(loudness: dict, t: float) -> float:
    values = np.array(loudness["values"])
    if not len(values):
        return 0.0
    hop = loudness["hop_s"]
    lo, hi = int(max(0, t - 1.0) / hop), int((t + 1.0) / hop) + 1
    local = values[lo:hi].mean() if hi > lo else values.mean()
    return float(np.clip((np.median(values) - local) / 12.0, 0, 1))


def _evaluate(cand: dict, cut: dict | None, speech: dict, wav: Path, use_asr: bool,
              judgement: dict[str, float]) -> None:
    """Apply the hard gates, then score. `judgement` holds the AI features (0-1) for this point."""
    t = cand["time"]
    if not cut:
        cand["rejected"].append("no clean shot change at this boundary")
        return
    cand["transition"] = cut["kind"]

    last_end, next_start, crossing = _speech_around(speech["speech"], t)
    cand["silence_before"] = round(t - last_end, 2)
    cand["silence_after"] = round(next_start - t, 2) if next_start != float("inf") else None
    if crossing:
        cand["rejected"].append(f"speech too close to the cut (VAD): speech {crossing[0] - t:+.2f}s to "
                                f"{crossing[1] - t:+.2f}s around it (needs {PRE_SILENCE_S}s before, "
                                f"{POST_SILENCE_S}s after)")
        return

    if use_asr:
        crossing, note = _asr_speech_at_cut(wav, t)
        cand["notes"].append(f"ASR: {note}")
        if crossing:
            cand["rejected"].append(f"speech across the cut (ASR): {note}")
            return

    features = {
        **judgement,
        "transition": TRANSITION_SCORE.get(cut["kind"], 0.5),
        "silence_before": min(t - last_end, 2.0) / 2.0,
        "silence_after": min(next_start - t, 1.0),
        "loudness_dip": _loudness_dip(speech["loudness_db"], t),
    }
    cand["features"] = {k: round(v, 3) for k, v in features.items()}
    cand["score"] = round(sum(WEIGHTS[k] * v for k, v in features.items()), 3)


def candidates(scenes: list[dict], speech: dict, wav: Path, use_asr: bool = True) -> list[dict]:
    """Scene boundaries plus in-scene beats (natural pauses inside long scenes)."""
    out = []
    for prev, nxt in zip(scenes, scenes[1:]):
        boundary = nxt["boundary_in"] or {}
        cand = {"time": nxt["start"], "kind": "scene_boundary", "scene_before": prev["index"],
                "scene_after": nxt["index"], "rejected": [], "notes": []}
        out.append(cand)
        _evaluate(cand, boundary.get("cut"), speech, wav, use_asr, {
            "interruptibility": prev["interruptibility"] / 10,
            "ending": ENDING_SCORE.get(prev["ending"], 0.0),
            "boundary_confidence": boundary.get("confidence", 0.5),
        })

    for scene in scenes:
        for beat in scene.get("beats", []):
            # Same scene on both sides: its negative contexts apply before and after the break.
            cand = {"time": beat["time"], "kind": f"beat ({beat['source']})", "scene_before": scene["index"],
                    "scene_after": scene["index"], "rejected": [], "notes": [f"beat: {beat['description']}"]}
            out.append(cand)
            _evaluate(cand, beat["cut"], speech, wav, use_asr, {
                "interruptibility": beat["strength"],
                "ending": beat["strength"],
                "boundary_confidence": beat["strength"],
            })
    return sorted(out, key=lambda c: c["time"])


def position_problem(c: dict, duration: float, rules: PacingRules) -> str | None:
    if c["time"] < rules.no_break_before_s:
        return f"pacing: within the first {rules.no_break_before_s:.0f}s"
    if c["time"] > duration - rules.no_break_in_last_s:
        return f"pacing: within the last {rules.no_break_in_last_s:.0f}s"
    if c["score"] < rules.min_break_score:
        return f"score {c['score']:.2f} below minimum {rules.min_break_score:.2f}"
    return None


def select(cands: list[dict], duration: float, rules: PacingRules) -> list[dict]:
    """Max-total-value subset under count / gap / position rules (small DP over candidates).

    Value is `selection_score` (break quality + best safe brand fit) when present, else `score`.
    """
    eligible = []
    for c in cands:
        if c["rejected"] or c.get("safe_brands", 1) == 0:
            continue
        problem = position_problem(c, duration, rules)
        if problem:
            c["rejected"].append(problem)
        else:
            eligible.append(c)
    eligible.sort(key=lambda c: c["time"])
    limit = rules.max_breaks(duration)
    if not eligible or limit == 0:
        return []

    # best[i][k] = (total score, chosen indices) using k breaks with the last one at candidate i
    n = len(eligible)
    best: list[dict[int, tuple[float, list[int]]]] = [dict() for _ in range(n)]
    value = [c.get("selection_score", c["score"]) for c in eligible]
    for i, c in enumerate(eligible):
        best[i][1] = (value[i], [i])
        for j in range(i):
            if c["time"] - eligible[j]["time"] < rules.min_gap_s:
                continue
            for k, (total, chosen) in best[j].items():
                if k + 1 <= limit and total + value[i] > best[i].get(k + 1, (-1, []))[0]:
                    best[i][k + 1] = (total + value[i], chosen + [i])
    _, chosen = max((entry for b in best for entry in b.values()), key=lambda e: e[0])

    picked = {id(eligible[i]) for i in chosen}
    for c in eligible:
        if id(c) not in picked:
            c["rejected"].append("pacing: a higher-scoring combination of breaks was chosen")
    return [eligible[i] for i in chosen]
