"""Fuse the deterministic ontology verdict into the LLM judges' scores, and pull
out every record where the two disagree.
"""

from __future__ import annotations

import statistics

from judge.clients import score_judge_result
from judge.ontology_check import WEIGHT

ONTOLOGY_LEVEL_ORDER = {"full": 3, "core": 2, "partial": 1}

DISAGREEMENT_GAP = 0.35

SURE_ONTOLOGY_WEIGHT = 0.85

UNSURE_JUDGE_T = 0.50

SURE_JUDGE_T = 0.85


def ontology_floor(verdict: dict | None) -> str | None:
    """An ontology_check verdict -> the lowest match level it justifies."""
    if not verdict:
        return None
    name = str(verdict.get("verdict") or "").upper()
    if name == "FULL":
        return "full" if str(verdict.get("confidence") or "").lower() == "high" else "core"
    if name == "CORE":
        return "core"
    return None


def _match_levels(detail: dict) -> list[str]:
    """The graded match levels behind a detail dict, as a flat list."""
    if any(k in detail for k in ("full_count", "core_count", "partial_count")):
        return (
            ["full"] * int(detail.get("full_count") or 0)
            + ["core"] * int(detail.get("core_count") or 0)
            + ["partial"] * int(detail.get("partial_count") or 0)
        )
    matches = detail.get("matches")
    if not isinstance(matches, list):
        return []
    levels = [str((m or {}).get("level", "")).lower() for m in matches]
    return [lv for lv in levels if lv in ONTOLOGY_LEVEL_ORDER]


def _is_blank(detail: dict, levels: list[str], gt: int, pred: int) -> bool:
    """Is this detail a judgment that found nothing, or no judgment at all?"""
    return str(detail.get("scored_from") or "").lower() == "unreachable" or (
        not levels and gt == 0 and pred == 0
    )


def _score(
    levels: list[str], gt: int, pred: int, detail: dict, reward_min: float, reward_max: float
) -> dict:
    """Levels -> a full detail dict, through the eval's own scorer."""
    return score_judge_result(
        {"gt_count": gt, "pred_count": pred, "matches": [{"level": lv} for lv in levels]},
        reward_min,
        reward_max,
        core_weight=float(detail.get("core_weight", 0.85)),
        partial_weight=float(detail.get("partial_weight", 0.50)),
        return_details=True,
    )


def _raise_matches(matches: list, floor: str, injected: bool) -> list:
    """Rewrite a `matches` list to the floor, for the callers that carry one."""
    rank = ONTOLOGY_LEVEL_ORDER[floor]
    out = []
    for m in matches:
        m = dict(m or {})
        if ONTOLOGY_LEVEL_ORDER.get(str(m.get("level", "")).lower(), 0) < rank:
            m["level"] = floor
            m["raised_by"] = "ontology_floor"
        out.append(m)
    if injected:
        out.append({"level": floor, "raised_by": "ontology_floor", "injected": True})
    return out


def fuse_detail(
    judge_detail: dict, verdict: dict | None, *, reward_min: float = 0.0, reward_max: float = 1.0
) -> dict:
    """One judge's detail dict + one ontology verdict -> a NEW detail dict."""
    floor = ontology_floor(verdict)
    levels = _match_levels(judge_detail)
    gt = int(judge_detail.get("gt_count") or 0)
    pred = int(judge_detail.get("pred_count") or 0)

    base = judge_detail
    if "t_reward" not in judge_detail:
        base = dict(judge_detail)
        base.update(_score(levels, gt, pred, judge_detail, reward_min, reward_max))

    fused_levels, injected = levels, False
    if floor is not None and not _is_blank(judge_detail, levels, gt, pred):
        rank = ONTOLOGY_LEVEL_ORDER[floor]
        fused_levels = [lv if ONTOLOGY_LEVEL_ORDER.get(lv, 0) >= rank else floor for lv in levels]
        if not levels:
            fused_levels, injected = [floor], True
            gt, pred = max(1, gt), max(1, pred)

    changed = injected or fused_levels != levels

    out = dict(base)
    if changed:
        out.update(_score(fused_levels, gt, pred, judge_detail, reward_min, reward_max))
        if isinstance(judge_detail.get("matches"), list):
            out["matches"] = _raise_matches(judge_detail["matches"], floor, injected)

    out["verdict_source"] = "ontology_floor" if changed else "judges"
    out["ontology"] = verdict
    out["prefusion"] = {
        "t_reward": float(base.get("t_reward") or 0.0),
        "full_count": levels.count("full"),
        "core_count": levels.count("core"),
        "partial_count": levels.count("partial"),
        "matched_count": len(levels),
        "gt_count": int(base.get("gt_count") or 0),
        "pred_count": int(base.get("pred_count") or 0),
    }
    return out


def _prefusion_t(fused: dict) -> float:
    """A fused detail's pre-floor t_reward, tolerating a hand-built one."""
    pre = fused.get("prefusion")
    if isinstance(pre, dict) and pre.get("t_reward") is not None:
        return float(pre["t_reward"])
    return float(fused.get("t_reward") or 0.0)


def fuse_case(
    judge_details: dict[str, dict],
    verdict: dict | None,
    *,
    reward_min: float = 0.0,
    reward_max: float = 1.0,
) -> dict:
    """Fuse every judge's detail for one pair, then aggregate to one number."""
    if not judge_details:
        raise ValueError("fuse_case needs at least one judge detail")

    fused = {
        name: fuse_detail(detail, verdict, reward_min=reward_min, reward_max=reward_max)
        for name, detail in judge_details.items()
    }

    prefusion = statistics.median([_prefusion_t(f) for f in fused.values()])
    median = statistics.median([float(f["t_reward"]) for f in fused.values()])
    source = (
        "ontology_floor"
        if any(f["verdict_source"] == "ontology_floor" for f in fused.values())
        else "judges"
    )
    none_conflict = (
        verdict is not None
        and str(verdict.get("verdict") or "").upper() == "NONE"
        and prefusion >= SURE_JUDGE_T
    )

    return {
        "judges": fused,
        "median_t_reward": median,
        "median_t_reward_prefusion": prefusion,
        "verdict_source": source,
        "none_conflict": bool(none_conflict),
        "n_judges": len(fused),
        "judges_present": sorted(fused),
    }


def classify_disagreement(median_t_prefusion: float, verdict: dict | None) -> str | None:
    """Compare the judges' unaided median against the ontology's weight."""
    if not verdict:
        return None
    weight = WEIGHT.get(str(verdict.get("verdict") or "").upper())
    if weight is None:
        return None

    median = float(median_t_prefusion)
    if abs(median - weight) < DISAGREEMENT_GAP:
        return None
    if (weight >= SURE_ONTOLOGY_WEIGHT and median < UNSURE_JUDGE_T) or (
        weight == 0.0 and median >= SURE_JUDGE_T
    ):
        return "hard"
    return "soft"


def _level_summary(fused_detail: dict) -> str:
    """What a judge actually graded, in one phrase: "1 core + 2 partial"."""
    counts = fused_detail.get("prefusion") or fused_detail
    parts = [
        f"{int(counts.get(lv + '_count') or 0)} {lv}"
        for lv in ("full", "core", "partial")
        if int(counts.get(lv + "_count") or 0)
    ]
    return " + ".join(parts) if parts else "no match"


def _evidence(verdict: dict | None) -> dict | None:
    """The auditable core of a verdict, without the per-source concept dumps."""
    if not verdict:
        return None
    nested = verdict.get("evidence")
    source = nested if isinstance(nested, dict) else verdict
    evidence = {
        "relations": source.get("relations"),
        "flags": list(source.get("flags") or []),
        "direction": source.get("direction"),
        "depth": source.get("depth"),
    }
    if source.get("head_check"):
        evidence["head_check"] = source["head_check"]
    return evidence


def _panel_test(full_panel: int | list | set | tuple):
    """-> (is_full(judges_present, n_judges) -> bool, expected_n)."""
    if isinstance(full_panel, (set, frozenset, list, tuple)):
        expected = {str(name) for name in full_panel}
        return (lambda present, n: set(present) == expected), len(expected)
    return (lambda present, n: n >= int(full_panel)), int(full_panel)


def build_disagreements(
    fused_rows: list[dict],
    *,
    full_panel: int | list | set | tuple,
    strip_judge_verdicts: bool = True,
) -> dict:
    """Every record where the judges and the ontology disagree -> the
    judge_disagreement.json payload.
    """
    panel_is_full, _expected_n = _panel_test(full_panel)
    summary = {
        "n_records": len(fused_rows),
        "n_records_full_panel": 0,
        "n_ontology_decisive": 0,
        "n_disagreements": 0,
        "hard": 0,
        "soft": 0,
        "none_conflicts": 0,
        "by_verdict": {"FULL": 0, "CORE": 0, "PARTIAL": 0, "NONE": 0},
    }
    rows: list[dict] = []

    for row in fused_rows:
        fusion = row.get("fusion") or {}
        judges = fusion.get("judges") or {}
        verdict = row.get("ontology")
        name = str((verdict or {}).get("verdict") or "").upper()
        weight = WEIGHT.get(name) if verdict else None

        n_judges = int(fusion.get("n_judges", len(judges)))
        judges_present = list(fusion.get("judges_present") or sorted(judges))
        is_full = panel_is_full(judges_present, n_judges)
        if is_full:
            summary["n_records_full_panel"] += 1
        if verdict is not None and weight is not None:
            summary["n_ontology_decisive"] += 1
        if fusion.get("none_conflict"):
            summary["none_conflicts"] += 1

        per_judge = {
            judge: {"level_summary": _level_summary(detail), "t_reward": _prefusion_t(detail)}
            for judge, detail in judges.items()
        }
        if strip_judge_verdicts:
            for detail in judges.values():
                detail.pop("ontology", None)

        prefusion = float(fusion.get("median_t_reward_prefusion") or 0.0)
        kind = classify_disagreement(prefusion, verdict)
        if kind is None:
            continue

        summary[kind] += 1
        if name in summary["by_verdict"]:
            summary["by_verdict"][name] += 1
        rows.append(
            {
                "case_id": row.get("case_id"),
                "pmcid": row.get("pmcid"),
                "sample_idx": row.get("sample_idx"),
                "pred": row.get("pred"),
                "gold": row.get("gold"),
                "type": kind,
                "ontology_verdict": name,
                "ontology_confidence": verdict.get("confidence"),
                "ontology_weight": weight,
                "ontology_evidence": _evidence(verdict),
                "judge_median_t_prefusion": prefusion,
                "judge_median_t_fused": float(fusion.get("median_t_reward") or 0.0),
                "per_judge": per_judge,
                "none_conflict": bool(fusion.get("none_conflict")),
                "n_judges": n_judges,
                "judges_present": judges_present,
                "partial_panel": not is_full,
            }
        )

    summary["n_disagreements"] = len(rows)
    rows.sort(
        key=lambda r: (
            0 if r["type"] == "hard" else 1,
            -abs(r["judge_median_t_prefusion"] - r["ontology_weight"]),
            str(r["case_id"] if r["case_id"] is not None else ""),
            r["sample_idx"] if isinstance(r["sample_idx"], int) else -1,
        )
    )
    return {"summary": summary, "disagreements": rows}
