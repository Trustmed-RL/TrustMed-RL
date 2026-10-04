"""The diagnosis judge behind `score_records.py --mode judge`."""

from __future__ import annotations

import json
import os
import re
from statistics import fmean
from typing import Callable

from judge.rubric import _JUDGE_INSTRUCTIONS, compute_diagnosis_reward

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-5-mini")
JUDGE_SAMPLES = 3
JUDGE_TEMPERATURE = 1.0
JUDGE_MAX_TOKENS = 4096
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$")


def _client():
    """Built per call, not at import: score_records imports this module to read
    score/score_full, and a missing OPENAI_API_KEY must not break that import.
    """
    from openai import OpenAI

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "judge_adapter needs OPENAI_API_KEY in the environment to reach the gpt-5-mini judge"
        )
    return OpenAI(api_key=key, timeout=600.0, max_retries=3)


def _default_sample(prompt: str) -> str:
    rsp = _client().chat.completions.create(
        model=JUDGE_MODEL,
        temperature=JUDGE_TEMPERATURE,
        max_completion_tokens=JUDGE_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    return rsp.choices[0].message.content or ""


RL_RUBRIC_ADDENDUM = """

### 4. Strict entity rule

Include a pair in "matches" only when the prediction and the ground truth name
the SAME disease entity. Two DIFFERENT diseases of the same organ, system, or
histologic category are NOT a match at any level -- leave such a pair out of
"matches" entirely.

Still the same entity -- match it normally under the rubric above:
* synonyms and abbreviations (pancreatic neuroendocrine tumor = neuroendocrine
  neoplasm of the pancreas), spelling variants
* a site, form, or qualifier variant of one disease (ocular tuberculosis vs
  tuberculosis = core; ectopic thyroid vs congenital thyroid malformation = core)
* a specific disease against a generic ground-truth category it belongs to by
  disease classification, not merely by location (primary cardiac angiosarcoma
  vs rare cardiac tumor = core)

Different entities -- never a match, at any level:
* glioma, astrocytoma, medulloblastoma, or ependymoma vs meningioma
* leukemias of a different lineage or tempo: AML vs ALL, AML vs CML, AML vs CMML
* gastrointestinal stromal tumor (GIST) vs leiomyosarcoma, or vs a
  neuroendocrine tumor
* Best vitelliform macular dystrophy vs Stargardt disease
* cholangiocarcinoma vs biliary cystadenocarcinoma
* Kaposi sarcoma vs Castleman disease

Sharing anatomy, imaging appearance, or a differential-diagnosis list does not
make two different diseases a match.
"""


def judge_prompt(gt: str, pred: str) -> str:
    """The rubric with its two slots filled."""
    t = (_JUDGE_INSTRUCTIONS + RL_RUBRIC_ADDENDUM).replace("{", "{{").replace("}", "}}")
    t = t.replace("{{ground_truth}}", "{ground_truth}").replace("{{predicted}}", "{predicted}")
    return t.format(ground_truth=gt, predicted=pred)


def extract_json(text: str) -> dict:
    """The judge is told to return only JSON and mostly does; a fence or a
    sentence of preamble is not worth a retry, so both are stripped before the
    outermost object is parsed. Raises ValueError when nothing parses.
    """
    t = _FENCE.sub("", (text or "").strip())
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j <= i:
        raise ValueError(f"no JSON object in judge reply: {text[:200]!r}")
    obj = json.loads(t[i : j + 1])
    if not isinstance(obj, dict):
        raise ValueError(f"judge returned {type(obj).__name__}, not an object")
    return obj


def _one_sample(prompt: str, sample: Callable[[str], str]) -> dict | None:
    """One judge verdict, with ONE retry: a truncated or chatty reply is a
    sampling accident and re-asking usually fixes it, but a judge that fails
    twice is failing systematically and burning more calls on it just delays
    the report.
    """
    for _ in range(2):
        try:
            return extract_json(sample(prompt))
        except (ValueError, json.JSONDecodeError):
            continue
    return None


def _ontology_rows(judge_result: dict, check: Callable[[str, str], dict] | None) -> list[dict]:
    """Audit column: what the ontologies say about the pairs the judge called
    FULL or CORE. PARTIAL pairs are skipped — the rubric defines them as
    "a component of", which is not a relation MedGen/MONDO can confirm or
    deny, so a lookup there would only manufacture disagreement.
    """
    rows: list[dict] = []
    for mt in judge_result.get("matches") or []:
        level = str(mt.get("level", "")).lower()
        if level not in ("full", "core"):
            continue
        a, b = str(mt.get("ground_truth", "")), str(mt.get("predicted", ""))
        row = {"ground_truth": a, "predicted": b, "judge_level": level}
        try:
            if check is None:
                raise RuntimeError("ontology_check unavailable")
            v = check(a, b)
            row.update(
                ontology=v.get("verdict"),
                confidence=v.get("confidence"),
                flags=v.get("flags") or [],
            )
        except Exception as e:
            row.update(ontology="unavailable", error=str(e)[:200])
        rows.append(row)
    return rows


def _default_check() -> Callable[[str, str], dict] | None:
    try:
        from judge.ontology_check import default_check_pair

        return default_check_pair()
    except Exception:
        return None


def score_full(
    gt: str,
    pred: str,
    _sample: Callable[[str], str] | None = None,
    _check: Callable[[str, str], dict] | None = None,
) -> dict:
    """{"t_reward", "samples", "ontology"} for one (ground truth, prediction)."""
    sample = _sample or _default_sample
    check = _check if _check is not None else _default_check()
    prompt = judge_prompt(gt, pred)

    results = [
        r for r in (_one_sample(prompt, sample) for _ in range(JUDGE_SAMPLES)) if r is not None
    ]
    if not results:
        raise RuntimeError(
            f"judge produced no parseable verdict in {JUDGE_SAMPLES} samples "
            f"(x2 attempts each) for {pred!r} vs {gt!r}"
        )
    t: list[float] = []
    for r in results:
        try:
            t.append(float(compute_diagnosis_reward(r, return_details=True)["t_reward"]))
        except (KeyError, TypeError, ValueError):
            continue
    if not t:
        raise RuntimeError(f"every judge verdict was self-inconsistent for {pred!r} vs {gt!r}")
    return {"t_reward": fmean(t), "samples": t, "ontology": _ontology_rows(results[0], check)}


def score(gt: str, pred: str) -> float:
    """score_records' `--judge-module` contract: the panel mean, in [0, 1]."""
    return float(score_full(gt, pred)["t_reward"])
