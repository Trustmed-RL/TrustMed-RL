JUDGE_INSTRUCTIONS = """
You are a medical expert evaluating a predicted diagnosis against a ground-truth diagnosis.

Ground truth:
{ground_truth}

Prediction:
{predicted}

### 1. Identify conditions

Identify all distinct medical conditions in the ground truth and prediction.

Count conditions by clinical meaning, not wording. Do not treat subtype, site, etiology, pathology, grade, stage, severity, or complication as a separate condition unless it is an independent diagnosis.

### 2. Match conditions

Match conditions one-to-one using the best overall assignment. Each condition can be matched at most once.

Classify each matched pair as:

* FULL: Same fully specified disease. Synonyms, abbreviations, spelling variants, and clinically equivalent wording count as FULL.

* CORE: captures core disease, but one or more clinically relevant qualifiers are missing or incorrect, such as subtype, site, etiology, pathology, severity, grade, stage, or complication.

For example:
{
"ground_truth": "Erysipelothrix bacteremia with endocarditis",
"predicted": "Erysipelothrix rhusiopathiae bacteremia",
"level": "core"
}
{
"ground_truth": "Catastrophic antiphospholipid syndrome",
"predicted": "Antiphospholipid syndrome",
"level": "core"
},

* PARTIAL: The core disease is not fully identified, but the prediction correctly identifies a meaningful component of the ground-truth diagnosis, such as an organism, complication, manifestation, or syndrome component.

A match requires either the same core disease or a specific component of the ground-truth diagnosis.


### 3. Output

Return only valid JSON using this format:

{
"gt_count": int,
"pred_count": int,
"matches": [
{
"ground_truth": "",
"predicted": "",
"level": ""
},
]
}

Report the number of distinct ground-truth and predicted conditions in "gt_count" and "pred_count".
Level must be one of "full", "core", or "partial".
Include only valid matched pairs in "matches". Unmatched conditions should not appear in "matches".


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


def compute_diagnosis_reward(
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

    jaccard_reward = safe_divide(
        soft_tp,
        soft_tp + soft_fp + soft_fn,
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
        "jaccard_reward": jaccard_reward,
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

    def mean_metric(name: str) -> float:
        """Average a metric across all judges."""
        values = [float(result[name]) for result in judge_scores]
        return sum(values) / J

    def majority_metric(name: str) -> int:
        """Aggregate a binary metric using majority vote."""
        values = [int(result[name]) for result in judge_scores]

        if any(value not in {0, 1} for value in values):
            raise ValueError(f"{name} must be binary (0 or 1) for majority-vote aggregation.")

        return int(sum(values) >= majority_threshold)

    t_reward = mean_metric("t_reward")

    tversky_balanced = mean_metric("tversky_balanced_0.5_0.5")

    tversky_fn_heavy = mean_metric("tversky_fn_heavy_0.3_0.7")

    tversky_fp_heavy = mean_metric("tversky_fp_heavy_0.7_0.3")

    jaccard_reward = mean_metric("jaccard_reward")

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
        "jaccard_reward": jaccard_reward,
        "dice_f1": dice_f1,
        "graded_acc": graded_acc,
        "acc": acc,
        "acc_exact": acc_exact,
        "acc_strict": acc_strict,
        "acc_lenient": acc_lenient,
        "per_judge": {
            "t_reward": [s["t_reward"] for s in judge_scores],
            "tversky_balanced_0.5_0.5": [s["tversky_balanced_0.5_0.5"] for s in judge_scores],
            "tversky_fn_heavy_0.3_0.7": [s["tversky_fn_heavy_0.3_0.7"] for s in judge_scores],
            "tversky_fp_heavy_0.7_0.3": [s["tversky_fp_heavy_0.7_0.3"] for s in judge_scores],
            "jaccard_reward": [s["jaccard_reward"] for s in judge_scores],
            "dice_f1": [s["dice_f1"] for s in judge_scores],
            "graded_acc": [s["graded_acc"] for s in judge_scores],
            "acc": [s["acc"] for s in judge_scores],
            "acc_exact": [s["acc_exact"] for s in judge_scores],
            "acc_strict": [s["acc_strict"] for s in judge_scores],
            "acc_lenient": [s["acc_lenient"] for s in judge_scores],
        },
        "acc_positive_votes": sum(int(s["acc"]) for s in judge_scores),
        "acc_exact_positive_votes": sum(int(s["acc_exact"]) for s in judge_scores),
        "acc_strict_positive_votes": sum(int(s["acc_strict"]) for s in judge_scores),
    }
