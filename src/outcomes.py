"""Backfill the sparse OUTCOME reward onto trajectory records."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Callable
from environment import (
    GATE_AXES,
    NON_DIAGNOSES,
    corruption_axes,
    gate_axes_exact,
    load_profiles,
)

JACCARD_HIT = 0.5
_PUNCT = re.compile(r"[^a-z0-9\s]+")
_WS = re.compile(r"\s+")


def normalize(s: str) -> str:
    """lower + punctuation to space + collapse. Punctuation becomes a SPACE,
    not nothing, so "Type-I" and "Type I" agree instead of colliding.
    """
    return _WS.sub(" ", _PUNCT.sub(" ", (s or "").lower())).strip()


def lexical_score(gt: str, pred: str) -> float:
    """1.0 when the two name the same thing under a deliberately loose test."""
    g, p = normalize(gt), normalize(pred)
    if not g or not p:
        return 0.0
    if g in p or p in g:
        return 1.0
    gs, ps = set(g.split()), set(p.split())
    return 1.0 if len(gs & ps) / len(gs | ps) >= JACCARD_HIT else 0.0


def load_judge(module_path: str) -> Callable[[str, str], float]:
    """--mode judge plumbing: import the harness, adapt its score()."""
    mod = importlib.import_module(module_path)
    fn = getattr(mod, "score", None)
    if not callable(fn):
        raise SystemExit(f"{module_path} exposes no callable score(gt, pred) -> float")
    full = getattr(mod, "score_full", None)

    def scorer(gt: str, pred: str) -> float:
        return float(fn(gt, pred))

    scorer.full = full if callable(full) else None
    return scorer


def last_gate(rec: dict) -> dict | None:
    """The gate payload of the episode's LAST integrity turn, or None."""
    gates = [
        t["gate"] for t in rec.get("turns") or [] if t.get("phase") == "gate" and t.get("gate")
    ]
    return gates[-1] if gates else None


def waved_through_corrupt(rec: dict) -> bool:
    """Did ANY gate turn pass a batch the engine had labelled mismatched?"""
    return any(
        t["gate"].get("verdict") == "pass" and t["gate"].get("gt_mismatch")
        for t in rec.get("turns") or []
        if t.get("phase") == "gate" and t.get("gate")
    )


def _is_non_diagnosis(text: str) -> bool:
    """True when the primary diagnosis (before any "|") names no disease."""
    primary = str(text or "").split("|", 1)[0]
    return normalize(primary) in NON_DIAGNOSES


DX_SCORER_ONTOLOGY = "ontology"

GENERIC_TOKENS = frozenset(
    """
disease diseases disorder disorders syndrome syndromes condition conditions
tumor tumour tumors tumours neoplasm neoplasms carcinoma carcinomas cancer
cancers malignancy malignancies lesion lesions infection infections deficiency
deficiencies genetic congenital inflammatory infectious autoimmune acute
chronic severe mild primary secondary idiopathic familial hereditary rare
benign malignant type types form forms i ii iii iv v 1 2 3 4 5 of the and
with due to in a an or unspecified
""".split()
)
LOCATION_TOKENS = frozenset(
    """
left right bilateral upper lower pulmonary lung lungs hepatic liver renal
kidney kidneys cardiac heart cerebral brain spinal cord cutaneous skin breast
gastric stomach colon colonic rectal rectum ovarian uterine cervical cervix
thyroid adrenal bone bones soft tissue central nervous system head neck
pelvic abdominal thoracic chest ocular eye eyes oral
""".split()
)
_HEDGE = re.compile(
    r"\b(most likely|most probable|probable|probably|likely|possible|possibly|"
    r"suspected|presumed|presumptive|consistent with|rule out|r/o)\b|\?",
    re.I,
)
_SPLIT = re.compile(r"\s*(?:,|;|/|\n|\band\b|\bor\b|\bvs\.?|\bversus\b)\s*", re.I)
_TRUSTED_MATCH = frozenset(
    {"exact_title", "label", "exact_synonym", "synonym_title", "normalized_label"}
)
_MAX_SEGMENTS = 6
_CORE_MAX_DEPTH = 2


def _ontology_check():
    """`ontology_local.check_pair`, imported lazily: `import score_records`
    must never need the database (the eval package's smoke test runs without
    it), and the DB path is resolved by ontology_local (TRUSTMED_ONTOLOGY_DB).
    """
    from judge.ontology_local import check_pair

    return check_pair


def informative_tokens(text: str) -> list[str]:
    """Tokens that name a disease rather than a category or a body part."""
    return [
        t for t in normalize(text).split() if t not in GENERIC_TOKENS and t not in LOCATION_TOKENS
    ]


def strip_hedges(text: str) -> str:
    """Drop hedge phrases ("most likely", "probable", "?") and the separators
    they leave behind, so a hedged-but-single answer is graded as itself.
    """
    t = _HEDGE.sub(" ", str(text or ""))
    t = re.sub(r"\s+", " ", t).strip(" ,;:-")
    return t


def _resolves_exactly(verdict: dict, side: str) -> bool:
    """Did the ontology resolve this side by an exact title / label / synonym
    in at least one source? Fuzzy hits are search results, not identities.
    """
    for src in (verdict.get("per_source") or {}).values():
        if isinstance(src, dict):
            if ((src.get(side) or {}).get("match")) in _TRUSTED_MATCH:
                return True
    return False


def _is_hedge_list(gt: str, pred: str, check) -> bool:
    """A list of DISTINCT diseases (differential dump) rather than one
    diagnosis with qualifiers. Segments that resolve exactly and are NONE or
    PARTIAL (siblings) to each other are distinct diseases; descriptive
    fragments ("... causing acute kidney injury") do not resolve exactly, or
    resolve to one concept, and pass.
    """
    segs = [strip_hedges(x) for x in _SPLIT.split(pred)]
    segs = [x for x in segs if x and informative_tokens(x)][:_MAX_SEGMENTS]
    if len(segs) < 2:
        return False
    counted = [x for x in segs if _resolves_exactly(check(gt, x), "b")]
    for i in range(len(counted)):
        for j in range(i + 1, len(counted)):
            v = check(counted[i], counted[j])
            if str(v.get("verdict") or "").upper() in ("NONE", "PARTIAL"):
                return True
    return False


def ontology_outcome_detail(gt: str, pred: str, check=None) -> dict:
    """The guarded ontology verdict with its reason — for records, canaries and tests.
    `check` overrides `ontology_local.check_pair` (the suites pass a canned
    one; live_panel passes its own so live and eval share one backend).
    """
    out = {
        "scorer": DX_SCORER_ONTOLOGY,
        "score": 0.0,
        "reason": None,
        "verdict": None,
        "confidence": None,
        "direction": None,
        "depth": None,
        "raw_verdict": None,
    }
    raw = str(pred or "").split("|", 1)[0]
    if not str(gt or "").strip() or not raw.strip() or _is_non_diagnosis(raw):
        out["reason"] = "non_diagnosis"
        return out
    answer = strip_hedges(raw)
    if not informative_tokens(answer):
        out["reason"] = "generic"
        return out
    check = check or _ontology_check()
    if normalize(answer) == normalize(gt):
        v = check(gt, answer)
        out.update(
            score=1.0,
            reason="exact",
            raw_verdict=v,
            verdict=str(v.get("verdict") or "").upper(),
            confidence=v.get("confidence"),
        )
        return out
    if _is_hedge_list(gt, raw, check):
        out["reason"] = "hedge"
        return out
    v = check(gt, answer)
    verdict = str(v.get("verdict") or "").upper()
    out.update(
        verdict=verdict,
        confidence=v.get("confidence"),
        direction=v.get("direction"),
        depth=v.get("depth"),
        raw_verdict=v,
    )
    if verdict == "FULL" and v.get("confidence") == "high":
        out.update(score=1.0, reason="full_high")
    elif verdict == "CORE":
        depth = v.get("depth")
        if depth is None or int(depth) <= _CORE_MAX_DEPTH:
            out.update(score=1.0, reason="core")
        else:
            out["reason"] = "core_deep"
    elif verdict == "FULL":
        out["reason"] = "full_low"
    else:
        out["reason"] = verdict.lower() or "unknown"
    return out


def ontology_outcome_score(gt: str, pred: str) -> float:
    """score(gt, pred) -> 0.0 | 1.0 — the guarded ontology scorer."""
    return float(ontology_outcome_detail(gt, pred)["score"])


ontology_outcome_score.scorer_name = DX_SCORER_ONTOLOGY


def guarded_verdict(
    gt: str, pred: str, check=None, for_floor: bool = False
) -> tuple[dict | None, dict]:
    """The ontology verdict a FLOOR may rest on, or None."""
    d = ontology_outcome_detail(gt, pred, check=check)
    if for_floor and d["score"] >= 1.0 and d.get("direction") == "head_terms_match":
        d = dict(d, reason="core_head_floor")
        return None, d
    return (d.get("raw_verdict") if d["score"] >= 1.0 else None), d


VETO_STRONG_MATCHES = {"exact_title", "exact_title_normalized", "exact_synonym", "synonym_title"}


def ontology_veto(gt: str, pred: str, check=None) -> tuple[bool, dict]:
    """Should a judge's PAID diagnosis be overridden as wrong-family?"""
    d = ontology_outcome_detail(gt, pred, check=check)
    if d["score"] >= 1.0 or d.get("reason") in ("non_diagnosis", "generic", "hedge"):
        return False, d
    if (
        str(d.get("verdict") or "").upper() not in ("NONE", "DIFFERENT", "UNRELATED")
        or str(d.get("confidence") or "").lower() != "high"
    ):
        return False, d
    raw = d.get("raw_verdict") or {}
    sources = raw.get("per_source") or {}
    for v in sources.values():
        if (v or {}).get("relation") not in (None, "unrelated", "unmapped", "distant"):
            return False, d
    for src, v in sources.items():
        a, b = (v or {}).get("a") or {}, (v or {}).get("b") or {}
        if a.get("match") in VETO_STRONG_MATCHES and b.get("match") in VETO_STRONG_MATCHES:
            return True, dict(d, reason=f"ontology_veto:{src}")
    return False, d


def scorer_for_mode(mode: str) -> Callable[[str, str], float]:
    """`--mode lexical | ontology` -> the scorer; judge mode loads a module."""
    if mode == "ontology":
        return ontology_outcome_score
    if mode == "lexical":
        return lexical_score
    raise ValueError(f"no built-in scorer for mode {mode!r}")


def classify_outcome(rec: dict) -> tuple[str, str | None]:
    """(outcome_kind, subkind) — how the episode ended, from the record alone."""
    final = rec.get("final") or {}
    if final.get("kind") == "final_diagnosis":
        if final.get("source") == "synthesized":
            return "format_failure", None
        if _is_non_diagnosis(final.get("diagnosis")):
            return "defer_ook", None
        return "dx", None
    if final.get("gate"):
        g = last_gate(rec) or {}
        if not g.get("gt_mismatch"):
            return "defer_mismatch", None
        exact = gate_axes_exact(g.get("axes") or [], g.get("gt_axes") or [])
        return "defer_mismatch", ("exact" if exact else "partial")
    reason = str(final.get("reason") or "").strip().lower()
    if reason.startswith("mismatch"):
        return "defer_mismatch_claimed", None
    return "defer_ook", None


_IDENTITY_WORDS = (
    "identity",
    "demograph",
    "different patient",
    "another patient",
    "not this patient",
    "wrong patient",
)
_CLAIM_MARKERS: dict[str, tuple[str, ...]] = {
    "header_identity": _IDENTITY_WORDS,
    "test_identity": (
        "wrong test",
        "different test",
        "not the test",
        "another test",
        "wrong result",
        "different result",
    ),
    "patient_identity": _IDENTITY_WORDS,
    "modality_anatomy": ("modality", "anatom", "wrong study", "different study", "not the study"),
    "gate_engine": ("wrong study", "different study", "not the study"),
}


def _claim_patterns(corr: dict) -> list[str]:
    """Regexes, any ONE match of which proves the claim named the real swap."""
    kind = str(corr.get("kind") or "")
    pats = [r"\b" + re.escape(normalize(s)) for s in _CLAIM_MARKERS.get(kind, ())]
    facts: list[str] = list(corr.get("axes") or [])
    if kind == "header_identity":
        facts.append(str(corr.get("swap_tag") or ""))
        demo = normalize(str(corr.get("swap_demographics") or "").replace("-", " "))
        parts = demo.split()
        if parts:
            facts += [demo, parts[-1]]
        if len(parts) >= 3:
            facts.append(" ".join(parts[:3]))
    elif kind in ("test_identity", "modality_anatomy", "gate_engine"):
        facts += [str(corr.get("victim_category") or ""), str(corr.get("swap_category") or "")]
    pats += [r"\b" + re.escape(n) + r"\b" for n in map(normalize, facts) if n]
    return pats


_AXIS_SYNONYMS: dict[str, tuple[str, ...]] = {
    "modality": (
        "modality",
        "wrong test",
        "different test",
        "wrong study type",
        "different study type",
    ),
    "anatomy": ("anatom", "wrong body part", "wrong site", "wrong region"),
    "laterality": ("laterality", "side", "left right swapped", "right left swapped"),
    "patient": ("patient identity", "identity", "demograph", "wrong patient", "different patient"),
}

_AXIS_PATTERNS: dict[str, tuple[str, ...]] = {
    axis: tuple(r"\b" + re.escape(normalize(s)) for s in words)
    for axis, words in _AXIS_SYNONYMS.items()
}


def mismatch_axes_claimed(text: str) -> set[str]:
    """Which of the integrity axes a free-text mismatch reason NAMES."""
    norm = normalize(text)
    return {
        axis
        for axis, pats in _AXIS_PATTERNS.items()
        if axis in GATE_AXES and any(re.search(p, norm) for p in pats)
    }


def gt_axes_of(corr: dict) -> set[str]:
    """The corruption's ground truth, read off the dict."""
    axes = {a for a in (corr.get("axes") or []) if a in GATE_AXES}
    return axes or set(corruption_axes(str(corr.get("kind") or "")))


def claim_specificity(rec: dict) -> str | None:
    """ "exact" / "partial" for a doctor-claimed mismatch, None when there was
    no corruption to name.
    """
    corr = rec.get("corruption") or {}
    if not corr:
        return None
    reason = (rec.get("final") or {}).get("reason") or ""
    claimed, truth = mismatch_axes_claimed(reason), gt_axes_of(corr)
    return "exact" if claimed and claimed <= truth else "partial"


def _grade(rec: dict, gt: str, score: Callable[[str, str], float]) -> float:
    """The DIAGNOSIS channel. A non-dx episode is 0.0 here and priced by its
    kind at composition — never graded as a wrong diagnosis.
    """
    mt = rec["metrics"]
    if mt.get("outcome") != "final_diagnosis":
        return 0.0
    return float(score(gt, mt.get("final_answer", "")))


def score_record(
    rec: dict, truth: dict[str, str], score: Callable[[str, str], float] = lexical_score
) -> float | None:
    """The episode's outcome reward, or None when the pmcid has no profile row
    (never 0.0 for that case: an unjoinable record is unscored, not wrong).
    """
    gt = truth.get(rec["meta"]["pmcid"])
    return None if gt is None else _grade(rec, gt, score)


def annotate(rec: dict, gt: str, score: Callable[[str, str], float] = lexical_score) -> float:
    """Write the outcome fields onto one record, and return its reward."""
    kind, sub = classify_outcome(rec)
    mt = rec["metrics"]
    mt["outcome_kind"] = kind
    if sub is None:
        mt.pop("outcome_subkind", None)
    else:
        mt["outcome_subkind"] = sub
    spec = claim_specificity(rec) if kind == "defer_mismatch_claimed" else None
    if spec is None:
        mt.pop("claim_specificity", None)
    else:
        mt["claim_specificity"] = spec
    full = getattr(score, "full", None)
    if kind == "dx" and full is not None and mt.get("outcome") == "final_diagnosis":
        d = full(gt, mt.get("final_answer", ""))
        mt["t_reward"] = float(d["t_reward"])
        if "outcome_reward" in d:
            mt["dx_score"] = float(d["outcome_reward"])
        for k in ("dx_scorer", "dx_source"):
            if d.get(k) is not None:
                mt[k] = d[k]
        r = float(d.get("outcome_reward", d["t_reward"]))
        mt["judge_samples"] = list(d.get("samples") or [])
        mt["ontology"] = d.get("ontology") or []
    elif kind == "dx":
        r = _grade(rec, gt, score)
    else:
        r = 0.0
    mt["outcome_reward"] = r
    return r


def gold_of(row: dict) -> str:
    """THE graded gold for one profile row, for every dx channel - the judge
    panel, this module's lexical outcome, and the live terminal reward.
    `orpha_name` is the Orphanet disease name; the free-text `diagnosis` is
    the case report's wording ("Invasive Aspergillosis" for orpha
    "Aspergillosis") and is the fallback only when the row carries no
    Orphanet name at all. One function, so an SFT keep, an offline outcome
    and a live reward can never disagree about what the answer was.
    """
    for field in ("orpha_name", "diagnosis"):
        v = row.get(field)
        if v is not None and str(v).strip():
            return str(v).strip()
    return ""


def verify_record(rec: dict, gt: str, score: Callable[[str, str], float]) -> dict:
    """Replay gate: recompute the dx channel through `score` and report every
    field that disagrees with what the record stored. {} = reproduced.
    """
    if classify_outcome(rec)[0] != "dx":
        return {}
    mt = rec.get("metrics") or {}
    full = getattr(score, "full", None)
    if full is not None:
        d = full(gt, mt.get("final_answer", ""))
        new = float(d.get("outcome_reward", d["t_reward"]))
    else:
        new = float(score(gt, mt.get("final_answer", "")))
    out = {}
    if "outcome_reward" in mt and abs(float(mt["outcome_reward"]) - new) > 1e-9:
        out["outcome_reward"] = (mt["outcome_reward"], new)
    if "dx_score" in mt and abs(float(mt["dx_score"]) - new) > 1e-9:
        out["dx_score"] = (mt["dx_score"], new)
    if "terminal_reward" in mt:
        import rewards as _gx

        rec2 = dict(rec)
        rec2["metrics"] = dict(mt, outcome_reward=new)
        want = round(_gx.terminal_reward(rec2, _gx.matrix_for_record(rec, None)), 4)
        if abs(round(float(mt["terminal_reward"]), 4) - want) > 1e-9:
            out["terminal_reward"] = (mt["terminal_reward"], want)
    return out


def verify_records(records: Path, profiles: Path, score: Callable[[str, str], float]) -> dict:
    truth = {r["pmcid"]: gold_of(r) for r in load_profiles(profiles)}
    n = bad = 0
    for line in records.open(encoding="utf-8"):
        if not line.strip():
            continue
        rec = json.loads(line)
        gt = truth.get(rec["meta"]["pmcid"])
        if gt is None:
            continue
        n += 1
        if verify_record(rec, gt, score):
            bad += 1
    return {"records": n, "mismatched": bad}


def backfill(
    records: Path, profiles: Path, score: Callable[[str, str], float] = lexical_score
) -> dict:
    """Rewrite the records file in place with metrics.outcome_reward set."""
    truth = {r["pmcid"]: gold_of(r) for r in load_profiles(profiles)}
    counts = {"records": 0, "scored": 0, "hits": 0, "unmatched": 0}
    tmp = records.with_name(records.name + ".tmp")
    with records.open(encoding="utf-8") as src, tmp.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            rec = json.loads(line)
            counts["records"] += 1
            gt = truth.get(rec["meta"]["pmcid"])
            if gt is None:
                counts["unmatched"] += 1
                print(f"warning: {rec['meta']['pmcid']} not in profiles, unscored")
            else:
                r = annotate(rec, gt, score)
                counts["scored"] += 1
                counts["hits"] += int(r > 0)
            dst.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, records)
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--records", required=True, type=Path, help="trajectories.jsonl, rewritten in place"
    )
    ap.add_argument("--profiles", required=True, type=Path)
    ap.add_argument(
        "--mode",
        choices=("lexical", "ontology", "judge"),
        default="lexical",
        help="lexical (default): substring/Jaccard smoke grader; "
        "ontology: the guarded ontology scorer (binary at "
        "the CORE bar, needs ontology_local.db); judge: "
        "--judge-module",
    )
    ap.add_argument("--judge-module", help="import path exposing score(gt, pred) -> float")
    ap.add_argument(
        "--verify",
        action="store_true",
        help="do not rewrite: recompute the dx channel and exit 2 "
        "if any record's stored outcome_reward/dx_score differs "
        "(the live-vs-replay gate)",
    )
    a = ap.parse_args()

    if a.mode == "judge":
        if not a.judge_module:
            raise SystemExit(
                "--mode judge needs --judge-module <import.path> exposing "
                "score(gt, pred) -> float; the one that ships is "
                "--judge-module judge.prompt (gpt-5-mini panel, avg t_reward)"
            )
        score = load_judge(a.judge_module)
    else:
        score = scorer_for_mode(a.mode)
    if a.verify:
        v = verify_records(a.records, a.profiles, score)
        print(f"verify: {v['records']} records, {v['mismatched']} mismatched")
        return 2 if v["mismatched"] else 0
    c = backfill(a.records, a.profiles, score)
    print(
        f"scored {c['scored']}/{c['records']} records ({c['hits']} outcome "
        f"hits, {c['unmatched']} without ground truth) -> {a.records}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
