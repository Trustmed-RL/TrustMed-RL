import argparse
import json
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path

from dotenv import load_dotenv

import sys as _sys

_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from judge.prompt import judge_prompt

load_dotenv()
load_dotenv(Path(__file__).resolve().parents[2] / ".env")
load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def extract_diagnosis(text: str) -> str | None:
    """Extract diagnosis from [DIAGNOSIS: ...] format."""
    if text is None:
        return None
    match = re.search(r"\[DIAGNOSIS:\s*(.+?)\]", text, re.IGNORECASE)
    return match.group(1).strip() if match else None


def score_judge_result(
    judge_result: dict,
    reward_min: float = 0.0,
    reward_max: float = 1.0,
    core_weight: float = 0.85,
    partial_weight: float = 0.50,
    return_details: bool = False,
) -> float | dict:
    """Compute diagnosis rewards from four-level LLM judge results."""

    gt = judge_result["gt_count"]
    pred = judge_result["pred_count"]
    matches = judge_result["matches"]

    full_count = sum(m["level"].lower() == "full" for m in matches)

    core_count = sum(m["level"].lower() == "core" for m in matches)

    partial_count = sum(m["level"].lower() == "partial" for m in matches)

    matched_count = full_count + core_count + partial_count

    unmatched_gt_count = gt - matched_count
    unmatched_pred_count = pred - matched_count

    if unmatched_gt_count < 0 or unmatched_pred_count < 0:
        raise ValueError(
            "Invalid judge output: number of matched pairs cannot exceed gt_count or pred_count."
        )

    soft_tp = full_count + core_weight * core_count + partial_weight * partial_count

    soft_fn = gt - soft_tp

    soft_fp = pred - soft_tp

    soft_fn = max(0.0, soft_fn)
    soft_fp = max(0.0, soft_fp)

    def safe_divide(
        numerator: float,
        denominator: float,
        empty_value: float = 1.0,
    ) -> float:
        if denominator == 0:
            return empty_value
        return numerator / denominator

    def tversky(alpha: float, beta: float) -> float:
        return safe_divide(
            soft_tp,
            soft_tp + alpha * soft_fp + beta * soft_fn,
        )

    tversky_balanced = tversky(
        alpha=0.50,
        beta=0.50,
    )

    tversky_fn_heavy = tversky(
        alpha=0.30,
        beta=0.70,
    )

    tversky_fp_heavy = tversky(
        alpha=0.70,
        beta=0.30,
    )

    t_reward = tversky_fn_heavy

    jaccard_weighted = safe_divide(
        soft_tp,
        soft_tp + soft_fp + soft_fn,
    )

    jaccard_count = safe_divide(
        float(matched_count),
        float(gt + pred - matched_count),
    )

    dice_f1 = safe_divide(
        2.0 * soft_tp,
        2.0 * soft_tp + soft_fp + soft_fn,
    )

    acc = safe_divide(
        soft_tp,
        max(gt, pred),
    )

    acc_exact = float(full_count == gt and gt == pred)

    acc_strict = float(full_count == gt)

    acc_lenient = float(full_count + core_count == gt)

    reward = reward_min + (reward_max - reward_min) * t_reward

    details = {
        "reward": reward,
        "t_reward": t_reward,
        "tversky_balanced_0.5_0.5": tversky_balanced,
        "tversky_fn_heavy_0.3_0.7": tversky_fn_heavy,
        "tversky_fp_heavy_0.7_0.3": tversky_fp_heavy,
        "jaccard_weighted": jaccard_weighted,
        "jaccard_count": jaccard_count,
        "dice_f1": dice_f1,
        "acc": acc,
        "acc_exact": acc_exact,
        "acc_strict": acc_strict,
        "acc_lenient": acc_lenient,
        "gt_count": gt,
        "pred_count": pred,
        "full_count": full_count,
        "core_count": core_count,
        "partial_count": partial_count,
        "matched_count": matched_count,
        "unmatched_gt_count": unmatched_gt_count,
        "unmatched_pred_count": unmatched_pred_count,
        "soft_tp": soft_tp,
        "soft_fp": soft_fp,
        "soft_fn": soft_fn,
        "full_weight": 1.0,
        "core_weight": core_weight,
        "partial_weight": partial_weight,
    }

    return details if return_details else reward


import math
from typing import Any


def aggregate_case_scores(
    judge_scores: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate all diagnosis-evaluation metrics across J judges."""

    if not judge_scores:
        raise ValueError("judge_scores must contain at least one judge result.")

    J = len(judge_scores)
    majority_threshold = math.ceil(J / 2)

    def mean_metric(name: str) -> float | None:
        """Average a metric across all judges, or None if no judge reports it."""
        values = [
            float(result[name])
            for result in judge_scores
            if name in result and result[name] is not None
        ]
        return sum(values) / len(values) if values else None

    def majority_metric(name: str) -> int | None:
        """Aggregate a binary metric by majority vote, or None if unreported."""
        values = [
            int(result[name])
            for result in judge_scores
            if name in result and result[name] is not None
        ]
        if not values:
            return None

        if any(value not in {0, 1} for value in values):
            raise ValueError(f"{name} must be binary (0 or 1) for majority-vote aggregation.")

        return int(sum(values) >= majority_threshold)

    t_reward = mean_metric("t_reward")

    tversky_balanced = mean_metric("tversky_balanced_0.5_0.5")

    tversky_fn_heavy = mean_metric("tversky_fn_heavy_0.3_0.7")

    tversky_fp_heavy = mean_metric("tversky_fp_heavy_0.7_0.3")

    jaccard_weighted = mean_metric("jaccard_weighted")
    jaccard_count = mean_metric("jaccard_count")

    dice_f1 = mean_metric("dice_f1")

    graded_acc = mean_metric("graded_acc")

    acc = majority_metric("acc")

    acc_exact = majority_metric("acc_exact")

    acc_strict = majority_metric("acc_strict")

    acc_lenient = mean_metric("acc_lenient")

    return {
        "num_judges": J,
        "majority_threshold": majority_threshold,
        "t_reward": t_reward,
        "tversky_balanced_0.5_0.5": tversky_balanced,
        "tversky_fn_heavy_0.3_0.7": tversky_fn_heavy,
        "tversky_fp_heavy_0.7_0.3": tversky_fp_heavy,
        "jaccard_weighted": jaccard_weighted,
        "jaccard_count": jaccard_count,
        "dice_f1": dice_f1,
        "graded_acc": graded_acc,
        "acc": acc,
        "acc_exact": acc_exact,
        "acc_strict": acc_strict,
        "acc_lenient": acc_lenient,
        "per_judge": {
            "t_reward": [s.get("t_reward") for s in judge_scores],
            "tversky_balanced_0.5_0.5": [s.get("tversky_balanced_0.5_0.5") for s in judge_scores],
            "tversky_fn_heavy_0.3_0.7": [s.get("tversky_fn_heavy_0.3_0.7") for s in judge_scores],
            "tversky_fp_heavy_0.7_0.3": [s.get("tversky_fp_heavy_0.7_0.3") for s in judge_scores],
            "jaccard_weighted": [s.get("jaccard_weighted") for s in judge_scores],
            "jaccard_count": [s.get("jaccard_count") for s in judge_scores],
            "dice_f1": [s.get("dice_f1") for s in judge_scores],
            "graded_acc": [s.get("graded_acc") for s in judge_scores],
            "acc": [s.get("acc") for s in judge_scores],
            "acc_exact": [s.get("acc_exact") for s in judge_scores],
            "acc_strict": [s.get("acc_strict") for s in judge_scores],
            "acc_lenient": [s.get("acc_lenient") for s in judge_scores],
        },
        "acc_positive_votes": sum(int(s["acc"]) for s in judge_scores),
        "acc_exact_positive_votes": sum(int(s["acc_exact"]) for s in judge_scores),
        "acc_strict_positive_votes": sum(int(s["acc_strict"]) for s in judge_scores),
    }


JUDGE_TEMPERATURE = 0.9
JUDGE_MAX_TOKENS = 150


GPT5_REASONING_EFFORT = "low"
GPT5_MAX_COMPLETION_TOKENS = 2000


def _parse_judge_result(response: str) -> dict | None:
    """Raw judge text -> the rubric's dict with `matches`, or None."""
    text = (response or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("matches"), list):
        return None
    obj.setdefault("gt_count", 1)
    obj.setdefault("pred_count", 1)
    return obj


def _parse_judge_response(response: str) -> tuple[int, int, int]:
    """Parse a judge response into (gt_count, pred_count, matched)."""
    text = (response or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            gt = max(1, int(obj.get("gt_count", 1)))
            pred = max(1, int(obj.get("pred_count", 1)))
            matches = obj.get("matches")
            if isinstance(matches, list):
                matched = len(matches)
            else:
                matched = max(0, int(obj.get("matched", 0)))
            return gt, pred, min(matched, gt, pred)
        except (ValueError, TypeError, AttributeError):
            pass

    gt_count, pred_count, matched = 1, 1, 0
    for line in response.split("\n"):
        line = line.strip().lower()
        if line.startswith("gt_count:"):
            try:
                gt_count = max(1, int(line.split(":")[1].strip()))
            except (ValueError, IndexError):
                pass
        elif line.startswith("pred_count:"):
            try:
                pred_count = max(1, int(line.split(":")[1].strip()))
            except (ValueError, IndexError):
                pass
        elif line.startswith("matched:"):
            try:
                matched = max(0, int(line.split(":")[1].strip()))
            except (ValueError, IndexError):
                pass

    matched = min(matched, gt_count, pred_count)
    return gt_count, pred_count, matched


def _call_openai(client, model: str, predicted: str, ground_truth: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": judge_prompt(ground_truth, predicted)},
        ],
        max_tokens=JUDGE_MAX_TOKENS,
        temperature=JUDGE_TEMPERATURE,
    )
    return resp.choices[0].message.content or ""


def _call_gpt5(client, model: str, predicted: str, ground_truth: str) -> str:
    """OpenAI reasoning models (gpt-5 family). Same endpoint as _call_openai,
    different parameter contract -- which is why it is its own provider rather
    than a model swap on `openai`:
    """
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": judge_prompt(ground_truth, predicted)},
        ],
        max_completion_tokens=GPT5_MAX_COMPLETION_TOKENS,
        reasoning_effort=GPT5_REASONING_EFFORT,
    )
    return resp.choices[0].message.content or ""


_CALL_FN = {"openai": _call_openai, "gpt5": _call_gpt5}


_LAST_JUDGE_RESULT: dict = {}


def judge_diagnosis_score(
    predicted: str,
    ground_truth: str,
    provider: str,
    client,
    model_name: str,
    max_retries: int = 3,
) -> tuple[int, int, int]:
    """One retry/parse loop for every provider -- swap `provider` and the only
    thing that changes is which _CALL_FN entry actually makes the request.
    """
    call = _CALL_FN[provider]
    for attempt in range(max_retries):
        try:
            text = call(client, model_name, predicted, ground_truth).strip()
            counts = _parse_judge_response(text)
            _LAST_JUDGE_RESULT[(provider, model_name, predicted, ground_truth)] = (
                _parse_judge_result(text)
            )
            return counts
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
            else:
                print(
                    f"[{provider}:{model_name}] judge API request failed after {max_retries} attempts: {e}"
                )
                return 1, 1, 0


BATCH_CAPABLE = ("openai",)

BATCH_CHUNK_SIZE = 40_000
BATCH_POLL_SECONDS = 10


def _openai_batch_submit(client, model: str, items: list[tuple[str, str, str]]) -> str:
    import io

    lines = []
    for cid, predicted, ground_truth in items:
        lines.append(
            json.dumps(
                {
                    "custom_id": cid,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "messages": [
                            {"role": "user", "content": judge_prompt(ground_truth, predicted)},
                        ],
                        "max_tokens": JUDGE_MAX_TOKENS,
                        "temperature": JUDGE_TEMPERATURE,
                    },
                }
            )
        )
    buf = io.BytesIO(("\n".join(lines) + "\n").encode())
    buf.name = "judge_batch.jsonl"
    file = client.files.create(file=buf, purpose="batch")
    batch = client.batches.create(
        input_file_id=file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    return batch.id


def _openai_batch_status(client, handle: str) -> tuple[bool, str]:
    batch = client.batches.retrieve(handle)
    c = batch.request_counts
    ended = batch.status in ("completed", "failed", "expired", "cancelled")
    return ended, f"{batch.status} completed={c.completed} failed={c.failed} total={c.total}"


def _openai_batch_results(client, handle: str) -> dict[str, str | None]:
    batch = client.batches.retrieve(handle)
    out: dict[str, str | None] = {}
    if not batch.output_file_id:
        return out
    content = client.files.content(batch.output_file_id).text
    for line in content.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cid = row["custom_id"]
        if row.get("error"):
            out[cid] = None
            continue
        out[cid] = row["response"]["body"]["choices"][0]["message"]["content"]
    return out


_BATCH_ADAPTERS = {
    "openai": (_openai_batch_submit, _openai_batch_status, _openai_batch_results),
}


_BATCH_JUDGE_RESULT: dict = {}


def run_judge_batch(
    provider: str,
    client,
    model_name: str,
    items: list[tuple[str, str, str]],
    poll_seconds: int = BATCH_POLL_SECONDS,
) -> dict[str, tuple[int, int, int]]:
    """items: (custom_id, predicted, ground_truth) triples. Submits, polls,
    and parses one provider's whole judge run as a Batch API job (or several
    chunks of one, if items exceeds BATCH_CHUNK_SIZE) -- the control flow is
    identical for both batch-capable providers; only _BATCH_ADAPTERS[provider]'s
    three functions differ. Blocks until every chunk ends, so call this from
    a background thread per provider if more than one provider should run
    concurrently rather than serialize behind each other (see run_all).
    """
    if not items:
        return {}
    submit, status, fetch = _BATCH_ADAPTERS[provider]

    scores: dict[str, tuple[int, int, int]] = {}
    for i in range(0, len(items), BATCH_CHUNK_SIZE):
        chunk = items[i : i + BATCH_CHUNK_SIZE]
        handle = submit(client, model_name, chunk)
        print(f"[{provider} batch] submitted {handle} ({len(chunk)} requests)", flush=True)

        ended, line = status(client, handle)
        print(f"[{provider} batch] {handle}: {line}", flush=True)
        while not ended:
            time.sleep(poll_seconds)
            ended, line = status(client, handle)
            print(f"[{provider} batch] {handle}: {line}", flush=True)

        texts = fetch(client, handle)
        for cid, _, _ in chunk:
            text = texts.get(cid)
            scores[cid] = _parse_judge_response(text) if text else (1, 1, 0)
            if text:
                _BATCH_JUDGE_RESULT[(provider, cid)] = _parse_judge_result(text)
    return scores


def build_clients(args, providers=None) -> dict:
    """One client per REQUESTED provider, built once and reused across calls."""
    from openai import OpenAI

    wanted = set(providers) if providers else set(_CALL_FN)
    clients: dict = {}

    if "openai" in wanted:
        clients["openai"] = OpenAI(base_url=args.judge_base_url, timeout=1800.0, max_retries=2)
    if "gpt5" in wanted:
        clients["gpt5"] = OpenAI(base_url=args.judge_base_url, timeout=1800.0, max_retries=2)
    return clients


def _score_from_counts(
    gt_count: int,
    pred_count: int,
    matched: int,
    reward_min: float = 0.0,
    reward_max: float = 1.0,
) -> dict:
    """Counts-only fallback -> the SAME detail dict score_judge_result returns."""
    matched = min(matched, gt_count, pred_count)
    detail = score_judge_result(
        {"gt_count": gt_count, "pred_count": pred_count, "matches": [{"level": "full"}] * matched},
        reward_min,
        reward_max,
        return_details=True,
    )
    detail["scored_from"] = "counts_only"
    return detail


def _unreachable_detail(reward_min: float = 0.0) -> dict:
    """The detail dict for a record no judgment exists for -- predicted was
    None (nothing to judge), or, in the batch path, a provider's whole
    submission failed. Zeros under the same key schema score_judge_result
    emits (counts included: 0, not the parser's floor of 1), so the summary,
    pairwise_compare, aggregate_case_scores and the CSV exports all read one
    schema no matter how a record was (not) judged.
    """
    return {
        "reward": reward_min,
        "t_reward": 0.0,
        "tversky_balanced_0.5_0.5": 0.0,
        "tversky_fn_heavy_0.3_0.7": 0.0,
        "tversky_fp_heavy_0.7_0.3": 0.0,
        "jaccard_weighted": 0.0,
        "jaccard_count": 0.0,
        "dice_f1": 0.0,
        "acc": 0.0,
        "acc_exact": 0.0,
        "acc_strict": 0.0,
        "acc_lenient": 0.0,
        "gt_count": 0,
        "pred_count": 0,
        "full_count": 0,
        "core_count": 0,
        "partial_count": 0,
        "matched_count": 0,
        "unmatched_gt_count": 0,
        "unmatched_pred_count": 0,
        "soft_tp": 0.0,
        "soft_fp": 0.0,
        "soft_fn": 0.0,
        "full_weight": 1.0,
        "core_weight": 0.85,
        "partial_weight": 0.50,
        "scored_from": "unreachable",
    }


def _detail_from_judge(
    parsed: dict | None,
    counts: tuple[int, int, int] | None,
    reward_min: float = 0.0,
    reward_max: float = 1.0,
) -> dict:
    """One judge reply -> its detail dict; the single scoring step both modes
    call. `parsed` is _parse_judge_result's level-aware view of the reply,
    `counts` is _parse_judge_response's (gt, pred, matched) view of the same
    text. Preference order:
    """
    if parsed is not None:
        try:
            detail = score_judge_result(parsed, reward_min, reward_max, return_details=True)
            detail["scored_from"] = "matches"
            return detail
        except (ValueError, TypeError, KeyError, AttributeError):
            pass
    if counts is None:
        return _unreachable_detail(reward_min)
    return _score_from_counts(*counts, reward_min=reward_min, reward_max=reward_max)


def compute_diagnosis_reward(
    predicted: str | None,
    ground_truth: str,
    provider: str,
    client,
    model_name: str,
    reward_min: float = 0.0,
    reward_max: float = 1.0,
    return_details: bool = False,
) -> float | dict:
    """Live mode: run one judge's synchronous LLM-as-a-judge call and score
    it via the shared _detail_from_judge. Same shape as medexagent's
    compute_diagnosis_reward, generalized to dispatch to whichever provider
    this judge is via judge_diagnosis_score.
    """
    if predicted is None:
        detail = _unreachable_detail(reward_min)
        return detail if return_details else reward_min

    gt_count, pred_count, matched = judge_diagnosis_score(
        predicted,
        ground_truth,
        provider,
        client,
        model_name,
    )
    result = _LAST_JUDGE_RESULT.pop((provider, model_name, predicted, ground_truth), None)
    detail = _detail_from_judge(result, (gt_count, pred_count, matched), reward_min, reward_max)
    return detail if return_details else detail["reward"]


def score_detail(
    predicted: str | None,
    counts: tuple[int, int, int] | None,
    parsed: dict | None = None,
    reward_min: float = 0.0,
    reward_max: float = 1.0,
) -> dict:
    """Batch mode's wrapper around the same _detail_from_judge the live path
    uses -- run_all_batches hands it the parsed level-aware reply alongside
    the counts. Both views absent means the record was never submitted
    (predicted None, the same short-circuit the live path takes) or its
    custom_id went missing from that provider's batch results; both score as
    judge-unreachable.
    """
    if predicted is None or (counts is None and parsed is None):
        return _unreachable_detail(reward_min)
    return _detail_from_judge(parsed, counts, reward_min, reward_max)


_embed_model = None


def _get_embed_model():
    """Lazy-load the MedEmbed sentence-transformer (singleton)."""
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer

        _embed_model = SentenceTransformer("abhinand/MedEmbed-base-v0.1")
    return _embed_model


def compute_diagnosis_embed_scores_batch(
    pairs: list[tuple[str | None, str]],
    batch_size: int = 64,
) -> list[float]:
    """Vectorised version of compute_diagnosis_embed_score for a list of pairs.
    Judge-independent (predicted/gold text only), so this runs once, not
    once per judge.
    """
    if not pairs:
        return []
    from sentence_transformers.util import cos_sim

    model = _get_embed_model()

    nonempty_idx, texts = [], []
    for i, (p, g) in enumerate(pairs):
        if p is None or not str(p).strip():
            continue
        nonempty_idx.append(i)
        texts.append(str(p))
        texts.append(str(g))

    out = [0.0] * len(pairs)
    if not texts:
        return out
    embs = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=False,
    )
    for k, idx in enumerate(nonempty_idx):
        sim = float(cos_sim(embs[2 * k], embs[2 * k + 1]).item())
        out[idx] = max(0.0, sim)
    return out


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    denom = (vx * vy) ** 0.5
    return cov / denom if denom > 0 else 0.0


def _cohens_kappa(xs: list[float], ys: list[float]) -> float:
    """xs, ys: parallel 0/1 sequences from two judges. Chance-corrected
    agreement -- 0 means no better than the agreement expected from each
    judge's own marginal rate, 1 means perfect agreement.
    """
    n = len(xs)
    if n == 0:
        return 0.0
    po = sum(1 for x, y in zip(xs, ys) if x == y) / n
    p_x1, p_y1 = sum(xs) / n, sum(ys) / n
    pe = p_x1 * p_y1 + (1 - p_x1) * (1 - p_y1)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def pairwise_compare(results: list[dict], judge_keys: list[str]) -> dict:
    """dict["judge_a__vs__judge_b"] -> agreement metrics, one entry per pair
    of the (usually three) judges. Computed on the exact same
    predicted/ground_truth text every judge scored, so any disagreement is
    attributable to the judges' own behavior, not to different inputs.
    """
    comparisons = {}
    for a, b in combinations(judge_keys, 2):
        rewards_a = [r[a]["reward"] for r in results]
        rewards_b = [r[b]["reward"] for r in results]
        exact_a = [r[a].get("acc_exact", 0.0) for r in results]
        exact_b = [r[b].get("acc_exact", 0.0) for r in results]
        strict_a = [r[a]["acc_strict"] for r in results]
        strict_b = [r[b]["acc_strict"] for r in results]
        lenient_a = [r[a]["acc_lenient"] for r in results]
        lenient_b = [r[b]["acc_lenient"] for r in results]
        n = len(results) or 1
        comparisons[f"{a}__vs__{b}"] = {
            "n": len(results),
            "reward_pearson_r": round(_pearson(rewards_a, rewards_b), 4),
            "reward_mean_abs_diff": round(
                sum(abs(x - y) for x, y in zip(rewards_a, rewards_b)) / n, 4
            ),
            "accuracy_exact_agreement": round(
                sum(1 for x, y in zip(exact_a, exact_b) if x == y) / n, 4
            ),
            "accuracy_exact_kappa": round(_cohens_kappa(exact_a, exact_b), 4),
            "accuracy_strict_agreement": round(
                sum(1 for x, y in zip(strict_a, strict_b) if x == y) / n, 4
            ),
            "accuracy_strict_kappa": round(_cohens_kappa(strict_a, strict_b), 4),
            "accuracy_lenient_agreement": round(
                sum(1 for x, y in zip(lenient_a, lenient_b) if x == y) / n, 4
            ),
            "accuracy_lenient_kappa": round(_cohens_kappa(lenient_a, lenient_b), 4),
        }
    return comparisons


def _load_parquet(path: Path) -> list[dict]:
    """One parquet file. pyarrow is only imported here, so a plain jsonl/json
    run never needs it installed.
    """
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def _load_parquet_dir(path: Path) -> list[dict]:
    """A directory of parquet part files -- e.g. a rarearena_vl run dir,
    which holds runs/<run_id>/generations/part-*.parquet (schema.py). Reads
    generations/*.parquet if that subdirectory exists, else every *.parquet
    directly under path, and concatenates every part (a resumed/crashed run
    can have more than one).
    """
    gen_dir = path / "generations"
    parts = sorted((gen_dir if gen_dir.is_dir() else path).glob("*.parquet"))
    if not parts:
        raise SystemExit(f"no .parquet files found under {path}")
    records: list[dict] = []
    for part in parts:
        records.extend(_load_parquet(part))
    return records


def load_records(path: Path) -> list[dict]:
    """.jsonl / .json (a list) / a single .parquet file / a directory of
    parquet part files -- see pick_field for how differently-named
    prediction/gold columns across these sources (e.g. rarearena_vl's
    answer/gold_diagnosis vs osce's own prediction/gold) are reconciled
    without a separate conversion step.
    """
    if path.is_dir():
        return _load_parquet_dir(path)
    if path.suffix == ".parquet":
        return _load_parquet(path)
    text = path.read_text()
    if text.lstrip().startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def pick_field(rec: dict, names: list[str]) -> str:
    """First of `names` this record fills. Empty string when none of them do."""
    for name in names:
        v = rec.get(name)
        if v is not None and str(v).strip():
            return str(v).strip()
    return ""


def prepare_records(
    records: list[dict],
    pred_fields: list[str],
    gold_fields: list[str],
    strict_extract: bool,
) -> list[dict]:
    """Extraction only -- no judge calls. Needed up front regardless of
    --mode: batch mode needs every record's predicted/gold before it can
    build the batch, and live mode needs it before dispatching to the
    thread pool.
    """
    out = []
    for rec in records:
        raw_pred = pick_field(rec, pred_fields)
        gold = pick_field(rec, gold_fields)
        predicted = extract_diagnosis(raw_pred)
        if predicted is None and not strict_extract and raw_pred:
            predicted = raw_pred
        out.append(
            {
                "idx": rec.get("idx"),
                "case_id": rec.get("case_id"),
                "predicted_diagnosis": predicted,
                "ground_truth_diagnosis": gold,
                "diagnosis_found": 1.0 if predicted is not None else 0.0,
            }
        )
    return out


def run_all_batches(
    prelim: list[dict],
    judge_specs: list[tuple[str, str]],
    clients: dict,
    poll_seconds: int,
) -> tuple[list[dict[str, dict]], dict[str, str]]:
    """One run_judge_batch() per provider in judge_specs, each in its own
    background thread so they run concurrently instead of serializing
    behind whichever is slowest. Callers only ever pass batch-capable
    providers here (see run_all) -- this function itself doesn't filter.
    """
    items = [
        (str(i), r["predicted_diagnosis"], r["ground_truth_diagnosis"])
        for i, r in enumerate(prelim)
        if r["predicted_diagnosis"] is not None
    ]
    print(f"batch: {len(items)}/{len(prelim)} records need a judge call", flush=True)

    provider_scores: dict[str, dict[str, tuple[int, int, int]]] = {}
    provider_errors: dict[str, str] = {}

    def _worker(provider: str, model_name: str):
        try:
            provider_scores[provider] = run_judge_batch(
                provider,
                clients[provider],
                model_name,
                items,
                poll_seconds=poll_seconds,
            )
        except Exception as e:
            print(
                f"[{provider} batch] FAILED, no scores collected for this provider: {e}", flush=True
            )
            provider_scores[provider] = {}
            provider_errors[provider] = str(e)

    threads = [threading.Thread(target=_worker, args=(p, m), daemon=True) for p, m in judge_specs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    scores = []
    for i, r in enumerate(prelim):
        row = {}
        for provider, model_name in judge_specs:
            counts = provider_scores[provider].get(str(i))
            parsed = _BATCH_JUDGE_RESULT.pop((provider, str(i)), None)
            row[model_name] = score_detail(r["predicted_diagnosis"], counts, parsed)
        scores.append(row)
    return scores, provider_errors


def run_all_live(
    prelim: list[dict],
    judge_specs: list[tuple[str, str]],
    clients: dict,
    workers: int,
) -> list[dict[str, dict]]:
    """One synchronous call per record per judge in judge_specs, all of a
    record's judges run one after another inside run_one, but different
    records run concurrently across the thread pool. Returns scores[i] =
    {model_name: detail, ...}, same shape as run_all_batches -- see its
    docstring for why record fields aren't part of this return value.
    """
    done = 0

    def run_one(i: int) -> dict:
        nonlocal done
        r = prelim[i]
        row = {}
        for provider, model_name in judge_specs:
            row[model_name] = compute_diagnosis_reward(
                r["predicted_diagnosis"],
                r["ground_truth_diagnosis"],
                provider,
                clients[provider],
                model_name,
                return_details=True,
            )
        done += 1
        if done == 1 or done % 10 == 0 or done == len(prelim):
            print(f"judge {done}/{len(prelim)}", flush=True)
        return row

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run_one, range(len(prelim))))


def run_all(
    prelim: list[dict],
    judge_specs: list[tuple[str, str]],
    clients: dict,
    mode: str,
    workers: int,
    poll_seconds: int,
) -> tuple[list[dict], dict[str, str]]:
    """Splits judge_specs into a batch-capable group and a live-only group,
    runs both concurrently (one thread each), and merges their per-judge
    scores onto one full result list per record -- the only place in this
    file that builds the complete per-record dict (prelim's fields plus
    every judge's detail), so run_all_batches/run_all_live only need to
    hand back scores, not reconstruct records around them.
    """
    batch_specs = [(p, m) for p, m in judge_specs if p in BATCH_CAPABLE] if mode == "batch" else []
    live_specs = [spec for spec in judge_specs if spec not in batch_specs]

    results = [dict(r) for r in prelim]
    provider_errors: dict[str, str] = {}
    threads = []

    if batch_specs:

        def _run_batch_group():
            batch_scores, errs = run_all_batches(prelim, batch_specs, clients, poll_seconds)
            for row, batch_row in zip(results, batch_scores):
                row.update(batch_row)

            failed = [(p, m) for p, m in batch_specs if p in errs]
            if failed:
                for p, m in failed:
                    print(
                        f"[{p} batch] unavailable ({errs[p][:90]}); falling back to live calls",
                        flush=True,
                    )
                fallback_scores = run_all_live(prelim, failed, clients, workers)
                for row, live_row in zip(results, fallback_scores):
                    row.update(live_row)
                errs = {p: e for p, e in errs.items() if p not in {fp for fp, _ in failed}}
            provider_errors.update(errs)

        threads.append(threading.Thread(target=_run_batch_group, daemon=True))

    if live_specs:

        def _run_live_group():
            live_scores = run_all_live(prelim, live_specs, clients, workers)
            for row, live_row in zip(results, live_scores):
                row.update(live_row)

        threads.append(threading.Thread(target=_run_live_group, daemon=True))

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return results, provider_errors


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "input",
        type=Path,
        help="gen.json (list), .jsonl, a single .parquet file, or a "
        "directory of parquet part files (e.g. a rarearena_vl run dir)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="results file (default: <input dir>/eval_<input stem>[_n<limit>].json)",
    )
    ap.add_argument(
        "--pred-field",
        default="pred,prediction,answer",
        help="comma-separated; first field the record fills wins "
        "(e.g. pred_short,pred to prefer the <diagnosis_short> tag). "
        "`answer` covers rarearena_vl's generations parquet with no "
        "extra flag needed",
    )
    ap.add_argument(
        "--gold-field",
        default="orpha_name,Orpha_name,gold_dx,gold,gold_diagnosis",
        help="comma-separated; first field the record fills wins. "
        "`orpha_name` leads because Orphanet's canonical rare-"
        "disease name is a cleaner grading target than the free-"
        "text `diagnosis`, which varies in wording, casing and "
        "abbreviation across records. Both capitalisations are "
        "listed: the RareArena parquets spell it `Orpha_name`, "
        "rarearena_vl's generations spell it `orpha_name`. Pass "
        "--gold-field gold_diagnosis to score against the free-"
        "text diagnosis instead",
    )
    ap.add_argument("--openai-model", default="gpt-4.1-mini", help="OpenAI chat-completions judge")
    ap.add_argument(
        "--gpt5-model",
        default="gpt-5-mini",
        help="OpenAI reasoning judge, separate from --openai-model so both can sit on the board",
    )
    ap.add_argument(
        "--judges", default="gpt5", help="comma-separated subset of the registered judges to run"
    )
    ap.add_argument("--judge-base-url", default="https://api.openai.com/v1")
    ap.add_argument(
        "--mode",
        choices=["batch", "live"],
        default="batch",
        help="batch (default): the chat judge runs as one OpenAI Batch API job; live: one "
        "synchronous call per record on a thread pool",
    )
    ap.add_argument("--workers", type=int, default=8, help="thread pool size for live judging")
    ap.add_argument(
        "--poll-seconds",
        type=int,
        default=BATCH_POLL_SECONDS,
        help="--mode batch only: how often to poll each provider's job",
    )
    ap.add_argument("--limit", type=int, default=None, help="only eval first N records")
    ap.add_argument(
        "--strict-extract",
        action="store_true",
        help="medexagent-exact: pred without [DIAGNOSIS: ...] tag scores 0",
    )
    ap.add_argument(
        "--embed",
        action="store_true",
        help="also score MedEmbed cosine (needs sentence-transformers; "
        "downloads abhinand/MedEmbed-base-v0.1 on first run)",
    )
    args = ap.parse_args()

    pred_fields = [f.strip() for f in args.pred_field.split(",") if f.strip()]
    gold_fields = [f.strip() for f in args.gold_field.split(",") if f.strip()]
    all_specs = [("openai", args.openai_model), ("gpt5", args.gpt5_model)]

    wanted = [j.strip() for j in args.judges.split(",") if j.strip()]
    unknown = [j for j in wanted if j not in _CALL_FN]
    if unknown:
        raise SystemExit(f"unknown judge(s): {unknown}. known: {sorted(_CALL_FN)}")
    judge_specs = [(p, m) for p, m in all_specs if p in wanted]

    records = load_records(args.input)
    if args.limit:
        records = records[: args.limit]
    print(
        f"{len(records)} records, mode={args.mode}, judges="
        + ", ".join(f"{p}:{m}" for p, m in judge_specs)
    )

    for label, names in (("pred", pred_fields), ("gold", gold_fields)):
        used = Counter(
            next((n for n in names if str(r.get(n) or "").strip()), "<none>") for r in records
        )
        print(f"{label} field: " + ", ".join(f"{k}={v}" for k, v in used.most_common()))

    clients = build_clients(args, [p for p, _ in judge_specs])
    prelim = prepare_records(records, pred_fields, gold_fields, args.strict_extract)

    results, provider_errors = run_all(
        prelim,
        judge_specs,
        clients,
        mode=args.mode,
        workers=args.workers,
        poll_seconds=args.poll_seconds,
    )

    n = len(results) or 1
    found_rate = sum(r["diagnosis_found"] for r in results) / n

    judges_summary = {}
    for provider, model_name in judge_specs:
        judges_summary[model_name] = {
            "provider": provider,
            "diagnosis_reward": sum(r[model_name]["reward"] for r in results) / n,
            "t_reward": sum(r[model_name]["t_reward"] for r in results) / n,
            "jaccard_count": sum(r[model_name]["jaccard_count"] for r in results) / n,
            "jaccard_weighted": sum(r[model_name]["jaccard_weighted"] for r in results) / n,
            "diagnosis_accuracy_exact": sum(r[model_name].get("acc_exact", 0.0) for r in results)
            / n,
            "diagnosis_accuracy_strict": sum(r[model_name]["acc_strict"] for r in results) / n,
            "diagnosis_accuracy_lenient": sum(r[model_name]["acc_lenient"] for r in results) / n,
        }
        if provider in provider_errors:
            judges_summary[model_name]["error"] = provider_errors[provider]

    if args.embed:
        print("embedding (MedEmbed-base-v0.1)...", flush=True)
        scores = compute_diagnosis_embed_scores_batch(
            [(r["predicted_diagnosis"], r["ground_truth_diagnosis"]) for r in results]
        )
        for r, s in zip(results, scores):
            r["diagnosis_embed_score"] = s

    pairwise = pairwise_compare(results, [m for _, m in judge_specs])

    summary = {
        "n": len(results),
        "mode": args.mode,
        "strict_extract": args.strict_extract,
        "diagnosis_found_rate": found_rate,
        "judges": judges_summary,
        "pairwise": pairwise,
    }
    if args.embed:
        summary["diagnosis_embed_score"] = sum(scores) / n

    print(f"diagnosis_found_rate: {found_rate:.3f}  (a prediction existed at all)")
    for model_name, s in judges_summary.items():
        if "error" in s:
            print(
                f"[{model_name}] SUBMISSION FAILED -- scores below are the "
                f"judge-unreachable fallback, not real judgments: {s['error']}"
            )
        print(
            f"[{model_name}] t_reward={s['t_reward']:.3f} "
            f"jaccard_count={s['jaccard_count']:.3f} "
            f"jaccard_weighted={s['jaccard_weighted']:.3f} "
            f"acc_exact={s['diagnosis_accuracy_exact']:.3f} "
            f"acc_strict={s['diagnosis_accuracy_strict']:.3f} "
            f"acc_lenient={s['diagnosis_accuracy_lenient']:.3f}"
        )
    if args.embed:
        print(f"diagnosis_embed_score: {summary['diagnosis_embed_score']:.3f}  (MedEmbed cosine)")
    for pair_key, c in pairwise.items():
        print(
            f"[{pair_key}] reward_pearson_r={c['reward_pearson_r']:.3f} "
            f"strict_agreement={c['accuracy_strict_agreement']:.3f} (kappa={c['accuracy_strict_kappa']:.3f}) "
            f"lenient_agreement={c['accuracy_lenient_agreement']:.3f} (kappa={c['accuracy_lenient_kappa']:.3f})"
        )

    suffix = f"_n{args.limit}" if args.limit else ""
    default_out = args.input.parent / f"eval_{args.input.stem}{suffix}.json"
    out = args.out or default_out
    out.write_text(json.dumps({"summary": summary, "results": results}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
