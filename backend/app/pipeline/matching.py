"""Brand safety, brand fit, independent audit, creative selection.

Safety is enforced in code, not left to a model's judgement:
  * scene BEFORE the break: a negative context that is "present" OR "possible" blocks the brand
  * scene AFTER the break:  "present" blocks the brand
  * a negative context that also appears in the scene's sensitive_events text blocks the brand
  * contexts the scene analysis never checked (e.g. a brand added later) are checked from the
    scene description by a text model; anything it cannot rule out counts as "possible"
Then a different model audits every final placement; a flagged placement is swapped or dropped.
"""

import logging
from typing import Literal

from pydantic import BaseModel, Field

from .. import llm

log = logging.getLogger("matching")

FIT_LLM_WEIGHT = 0.75
BLOCK_BEFORE = {"present", "possible"}
BLOCK_AFTER = {"present"}


# ---------- context verdicts ----------

class ExtraVerdict(BaseModel):
    context: str
    verdict: Literal["absent", "possible", "present"]
    reason: str


class ExtraVerdicts(BaseModel):
    verdicts: list[ExtraVerdict]


def _scene_brief(scene: dict) -> str:
    return (f"Setting: {scene.get('setting', '')}, {scene.get('time_of_day', '')}\n"
            f"Summary: {scene.get('summary', '')}\n"
            f"Dialogue: {scene.get('dialogue_gist', '')}\n"
            f"Dominant activity: {scene.get('dominant_activity', '')}\n"
            f"Other activities: {[a['activity'] for a in scene.get('activities', [])]}\n"
            f"Mood: {scene.get('mood', '')} (intensity {scene.get('emotional_intensity', '?')}/10)\n"
            f"Sensitive events: {scene.get('sensitive_events', [])}")


def fill_unchecked_contexts(scenes: list[dict], brands: list[dict], scene_ids: set[int]) -> None:
    """Add verdicts for negative contexts the video analysis did not cover (unseen brands)."""
    wanted = {c for b in brands for c in b["negative_contexts"]}
    for scene in scenes:
        if scene["index"] not in scene_ids:
            continue
        known = {c["context"] for c in scene.get("negative_context_checks", [])}
        missing = sorted(wanted - known)
        if not missing:
            continue
        try:
            result, model = llm.text_json(
                f"Scene description:\n{_scene_brief(scene)}\n\nContexts to check: {missing}",
                ExtraVerdicts,
                system="You check TV scenes for contexts an advertiser must avoid. For each context give "
                       "'present' if the description shows or clearly implies it, 'possible' if there is any "
                       "hint or you cannot rule it out, 'absent' only when clearly not there.",
            )
            got = {v.context.lower().strip(): v for v in result.verdicts}
        except Exception as e:  # noqa: BLE001 - any failure must fall back to the cautious verdict
            log.warning("context check failed for scene %s: %s", scene["index"], e)
            got, model = {}, "none"
        for context in missing:
            v = got.get(context)
            scene.setdefault("negative_context_checks", []).append({
                "context": context,
                "verdict": v.verdict if v else "possible",
                "evidence": f"text check ({model}): {v.reason}" if v else "could not be checked",
            })


def _verdict(scene: dict, context: str) -> tuple[str, str]:
    for c in scene.get("negative_context_checks", []):
        if c["context"] == context:
            return c["verdict"], c.get("evidence", "")
    return "possible", "not checked"


def safety(before: dict, after: dict, brand: dict) -> list[str]:
    """Reasons this brand is blocked at this break (empty list = safe)."""
    reasons = []
    events_before = " | ".join(before.get("sensitive_events", [])).lower()
    for context in brand["negative_contexts"]:
        verdict, evidence = _verdict(before, context)
        if verdict in BLOCK_BEFORE:
            reasons.append(f"'{context}' {verdict} in scene before: {evidence}".strip())
        elif context in events_before:
            reasons.append(f"'{context}' listed in sensitive events of scene before")
        verdict, evidence = _verdict(after, context)
        if verdict in BLOCK_AFTER:
            reasons.append(f"'{context}' {verdict} in scene after: {evidence}".strip())
    return reasons


# ---------- fit ----------

class BrandFit(BaseModel):
    brand_id: str
    fit: int = Field(ge=0, le=10, description="10 = the scene's dominant activity is exactly this brand's context")
    reason: str


class FitResult(BaseModel):
    fits: list[BrandFit]


FIT_SYSTEM = """You place ads in Bengali TV dramas. Score how well each brand fits a commercial break that comes
right after the given scene. The scene's DOMINANT activity decides the fit: a context that only appears
briefly or in the background counts for little. Score 0-10 per brand, using only the brand data given."""


def _overlap(scene: dict, brand: dict) -> float:
    present = set(scene.get("target_contexts_present", []))
    text = f"{scene.get('dominant_activity', '')} {scene.get('setting', '')}".lower()
    hits = sum(1 for c in brand["target_contexts"] if c in present or c in text)
    return min(1.0, hits / 2)


def score_fit(before: dict, brands: list[dict]) -> tuple[list[dict], str]:
    if not brands:
        return [], "none"
    brand_lines = "\n".join(
        f"- {b['brand_id']} ({b['display_name']}, {b['category']}): fits {b['target_contexts']}" for b in brands)
    try:
        result, model = llm.text_json(f"Scene:\n{_scene_brief(before)}\n\nBrands:\n{brand_lines}",
                                      FitResult, system=FIT_SYSTEM)
        llm_fit = {f.brand_id: f for f in result.fits}
    except Exception as e:  # noqa: BLE001 - fall back to the deterministic overlap score
        log.warning("fit scoring failed: %s", e)
        llm_fit, model = {}, "none"

    scored = []
    for b in brands:
        f = llm_fit.get(b["brand_id"])
        overlap = _overlap(before, b)
        llm_score = f.fit / 10 if f else overlap
        scored.append({
            "brand_id": b["brand_id"],
            "fit": round(FIT_LLM_WEIGHT * llm_score + (1 - FIT_LLM_WEIGHT) * overlap, 3),
            "llm_fit": f.fit if f else None,
            "context_overlap": overlap,
            "reason": f.reason if f else "keyword overlap only",
        })
    scored.sort(key=lambda s: s["fit"], reverse=True)
    return scored, model


# ---------- audit ----------

class Audit(BaseModel):
    verdict: Literal["safe", "violation"]
    violated_context: str = ""
    reason: str


AUDIT_SYSTEM = """You are an independent brand-safety auditor for TV ad breaks. An ad for the given brand will play
right after scene BEFORE and right before scene AFTER. The brand must never appear next to any of its
negative contexts. Answer "violation" if scene BEFORE shows or implies any of them, or scene AFTER clearly
shows one. If in doubt, answer "violation"."""


def audit(before: dict, after: dict, brand: dict) -> tuple[Audit, str]:
    prompt = (f"Brand: {brand['display_name']} ({brand['category']})\n"
              f"Negative contexts: {brand['negative_contexts']}\n\n"
              f"Scene BEFORE:\n{_scene_brief(before)}\n\nScene AFTER:\n{_scene_brief(after)}")
    try:
        return llm.text_json(prompt, Audit, system=AUDIT_SYSTEM, prefer="audit")
    except Exception as e:  # noqa: BLE001 - an audit that cannot run must not approve anything
        log.warning("audit failed: %s", e)
        return Audit(verdict="violation", reason=f"audit unavailable: {e}"), "none"


# ---------- creative / ad load ----------

def pick_creative(brand: dict, budget_s: float) -> dict | None:
    fitting = [c for c in brand["creatives"] if c["duration_sec"] <= budget_s]
    return max(fitting, key=lambda c: c["duration_sec"]) if fitting else None
