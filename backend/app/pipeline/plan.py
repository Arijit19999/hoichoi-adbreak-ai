"""Break plan: candidates -> brand safety -> pacing selection -> brand fit -> audit -> creatives."""

import logging
from pathlib import Path

from . import breaks, matching

log = logging.getLogger("plan")

AUDIT_TOP_N = 3            # brands tried per break before the break is dropped
DIVERSITY_MARGIN = 0.10    # accept the runner-up brand this close to avoid back-to-back repeats
FIT_IN_SELECTION = 0.30    # weight of the best safe brand's fit when choosing between break points


def build_plan(work: Path, scenes_doc: dict, speech_doc: dict, duration: float, catalogue: list[dict],
               rules: breaks.PacingRules, use_asr: bool = True) -> tuple[dict, dict]:
    scenes = scenes_doc["scenes"]
    by_id = {b["brand_id"]: b for b in catalogue}

    cands = breaks.candidates(scenes, speech_doc, work / "audio.wav", use_asr=use_asr)

    # Brand safety for every candidate that survived the audio/visual gates.
    live = [c for c in cands if not c["rejected"]]
    matching.fill_unchecked_contexts(scenes, catalogue, {i for c in live for i in (c["scene_before"], c["scene_after"])})
    for c in live:
        before, after = scenes[c["scene_before"]], scenes[c["scene_after"]]
        c["brand_safety"] = {b["brand_id"]: matching.safety(before, after, b) for b in catalogue}
        c["safe_brands"] = sum(1 for reasons in c["brand_safety"].values() if not reasons)
        if c["safe_brands"] == 0:
            c["rejected"].append("no brand in the catalogue is safe next to these scenes")

    # Brand fit for every candidate that could still be picked, so selection weighs "where" and "what".
    for c in live:
        if c["rejected"] or breaks.position_problem(c, duration, rules):
            continue
        safe = [by_id[b] for b, reasons in c["brand_safety"].items() if not reasons]
        c["brand_ranking"], c["fit_model"] = matching.score_fit(scenes[c["scene_before"]], safe)
        c["best_fit"] = c["brand_ranking"][0]["fit"]
        c["selection_score"] = round(c["score"] + FIT_IN_SELECTION * c["best_fit"], 3)

    chosen = breaks.select(cands, duration, rules)

    budget_left = rules.max_ad_load_pct / 100 * duration
    previous_brand = None
    placed = []
    for n, c in enumerate(chosen):
        before, after = scenes[c["scene_before"]], scenes[c["scene_after"]]
        ranking = c["brand_ranking"]

        order = [r["brand_id"] for r in ranking]
        if len(ranking) > 1 and order[0] == previous_brand and ranking[1]["fit"] >= ranking[0]["fit"] - DIVERSITY_MARGIN:
            order[0], order[1] = order[1], order[0]
            c["notes"].append(f"runner-up chosen to avoid repeating {previous_brand} back to back")

        per_break_budget = budget_left / (len(chosen) - n)
        c["audits"] = []
        for brand_id in order[:AUDIT_TOP_N]:
            brand = by_id[brand_id]
            creative = matching.pick_creative(brand, per_break_budget) or matching.pick_creative(brand, budget_left)
            if creative is None:
                c["notes"].append(f"{brand_id}: no creative fits the remaining ad-load budget ({budget_left:.0f}s)")
                continue
            result, audit_model = matching.audit(before, after, brand)
            c["audits"].append({"brand_id": brand_id, "model": audit_model, **result.model_dump()})
            if result.verdict != "safe":
                continue
            fit = next(r for r in ranking if r["brand_id"] == brand_id)
            c["placement"] = {"brand_id": brand_id, "display_name": brand["display_name"],
                              "category": brand["category"], "creative": creative,
                              "fit": fit["fit"], "reason": fit["reason"]}
            budget_left -= creative["duration_sec"]
            previous_brand = brand_id
            break
        else:
            c["rejected"].append("no safe brand passed the independent audit within the ad-load budget")
            continue
        placed.append(c)

    plan = {
        "duration": duration,
        "rules": rules.to_dict(),
        "ad_load_s": round(sum(c["placement"]["creative"]["duration_sec"] for c in placed), 1),
        "ad_load_pct": round(100 * sum(c["placement"]["creative"]["duration_sec"] for c in placed) / duration, 2),
        "breaks": [
            {"id": f"midroll-{i + 1}", "time": round(c["time"], 3), "score": c["score"],
             "selection_score": c.get("selection_score"),
             "transition": c.get("transition"), "scene_before": c["scene_before"], "scene_after": c["scene_after"],
             **c["placement"]}
            for i, c in enumerate(placed)
        ],
    }
    debug = {
        "rules": rules.to_dict(),
        "scoring_weights": breaks.WEIGHTS,
        "scenes": [
            {k: s.get(k) for k in ("index", "start", "end", "summary", "dominant_activity", "mood", "ending",
                                   "interruptibility", "sensitive_events", "target_contexts_present", "models")}
            | {"flagged_contexts": [c for c in s.get("negative_context_checks", []) if c["verdict"] != "absent"],
               "boundary_in": s.get("boundary_in")}
            for s in scenes
        ],
        "candidates": cands,
        "plan": plan,
    }
    return plan, debug
