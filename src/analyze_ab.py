"""Compare sp_consult arms (v2 vs v3/rl_mode) from their trajectory records."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from rewards import episode_group_uid

SCHEMA = "sp_consult/v3"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_records(path: Path) -> list[dict]:
    return [r for r in load_jsonl(path) if r.get("schema") == SCHEMA]


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def opt_mean(mt: list[dict], key: str, spec: str = "+.3f") -> str:
    """Mean of a metric the arm may predate, or a statement that it does not
    carry it. Pre-v4 records have no invalid_turns, and printing 0.000 for that
    would read as "the policy never erred" instead of "nobody counted".
    """
    vals = [float(x[key]) for x in mt if key in x]
    return format(mean(vals), spec) if vals else "(not recorded)"


def action_text(turn: dict) -> str:
    """verb + payload as the menus name them (a list payload is joined)."""
    p = turn.get("payload")
    p = ", ".join(p) if isinstance(p, list) else str(p or "")
    return f"{turn['verb']}: {p}" if p else turn["verb"]


def first_workup_action(rec: dict) -> str | None:
    """The first EXECUTED workup action: the zero-cost begin_workup transition
    is not one (it consumes no budget and returns no result), and a forfeit is
    not one either — this measures where the policy opens the workup, and both
    would answer a different question.
    """
    for t in rec["turns"]:
        if t.get("phase") == "workup" and t["verb"] not in ("begin_workup", "invalid"):
            return action_text(t)
    return None


def usage_by_phase(records: Path) -> dict[str, dict[str, float]]:
    """Mean doctor input/output tokens per phase from the sibling usage log."""
    path = records.parent / "state" / "usage.jsonl"
    if not path.exists():
        return {}
    by: dict[str, list[dict]] = {}
    for u in load_jsonl(path):
        if u.get("agent") != "doctor":
            continue
        by.setdefault(u.get("phase") or "(none)", []).append(u)
    return {
        ph: {
            "calls": len(rows),
            "in": mean([float(u.get("input_tokens") or 0) for u in rows]),
            "cached": mean([float(u.get("cached_tokens") or 0) for u in rows]),
            "out": mean([float(u.get("output_tokens") or 0) for u in rows]),
        }
        for ph, rows in sorted(by.items())
    }


def anchor_report(recs: list[dict], opening_anchor: str | None = None) -> dict:
    """Anchor collisions inside each episode group."""
    groups: dict[str, list[int]] = {}
    for i, rec in enumerate(recs):
        groups.setdefault(episode_group_uid(rec), []).append(i)
    sizes = Counter(len(v) for v in groups.values())
    occ_of: dict[tuple[str, str], int] = {}
    seen_in: dict[tuple[str, str], set] = {}
    phase_of: dict[tuple[str, str], str] = {}
    same_traj = 0
    n_unanchored = 0
    for uid, members in groups.items():
        if len(members) < 2:
            continue
        opener = opening_anchor or next(
            (t.get("anchor") for i in members for t in recs[i]["turns"] if t.get("anchor")), None
        )
        for i in members:
            per_rollout: dict[str, int] = {}
            for t in recs[i]["turns"]:
                a = t.get("anchor")
                if not a:
                    n_unanchored += 1
                    continue
                if a == opener:
                    continue
                occ_of[(uid, a)] = occ_of.get((uid, a), 0) + 1
                seen_in.setdefault((uid, a), set()).add(i)
                per_rollout[a] = per_rollout.get(a, 0) + 1
                phase_of.setdefault((uid, a), t.get("phase") or "(none)")
            for c in per_rollout.values():
                if c >= 2:
                    same_traj += c - 1
    distinct = len(occ_of)
    collided = sum(1 for v in occ_of.values() if v >= 2)
    eligible_collided = sum(1 for r in seen_in.values() if len(r) >= 2)
    by_phase: dict[str, list[int]] = {}
    for k, v in occ_of.items():
        row = by_phase.setdefault(phase_of[k], [0, 0])
        row[0] += 1
        row[1] += int(v >= 2)
    return {
        "groups": len(groups),
        "sizes": dict(sorted(sizes.items())),
        "multi_groups": sum(1 for v in groups.values() if len(v) > 1),
        "distinct_anchors": distinct,
        "collided_anchors": collided,
        "collision_rate": (collided / distinct) if distinct else 0.0,
        "eligible_distinct": distinct,
        "eligible_collided": eligible_collided,
        "eligible_rate": (eligible_collided / distinct) if distinct else 0.0,
        "same_traj_recurrences": same_traj,
        "unanchored_turns": n_unanchored,
        "by_phase": {
            ph: {"distinct": d, "collided": c, "rate": (c / d) if d else 0.0}
            for ph, (d, c) in sorted(by_phase.items())
        },
    }


def report_arm(records: Path, steps: Path | None) -> None:
    recs = load_records(records)
    print(f"=== {records} ===")
    if not recs:
        print("  no sp_consult/v3 records")
        return
    mt = [r["metrics"] for r in recs]
    b = recs[0]["budgets"]
    print(
        f"  episodes            {len(recs)}  "
        f"(prompt_version {b.get('prompt_version')}, "
        f"max_workup_turns {b.get('max_workup_turns')})"
    )
    outcomes = Counter(x["outcome"] for x in mt)
    print("  outcomes            " + ", ".join(f"{k} {v}" for k, v in outcomes.most_common()))
    print(f"  fact_recall         {mean([x['fact_recall'] for x in mt]):.3f}")
    print(f"  asks_used           {mean([x['asks_used'] for x in mt]):.2f}")
    print(f"  workup_turns        {mean([x['workup_turns'] for x in mt]):.2f}")
    print(
        f"  doctor_retries      {sum(x['doctor_retries'] for x in mt)}  "
        f"(fallbacks {sum(x['doctor_fallbacks'] for x in mt)})"
    )
    inv = [x["invalid_turns"] for x in mt if "invalid_turns" in x]
    print(
        f"  invalid_turns       {sum(inv)}  ({mean(inv):.2f}/episode)"
        if inv
        else "  invalid_turns       (not recorded)"
    )
    print("  step_reward_sum     " + opt_mean(mt, "step_reward_sum"))
    gate = "  gate_reward_sum     " + opt_mean(mt, "gate_reward_sum")
    if any("gate_reward_sum" in x for x in mt):
        gate += (
            f"  ({sum(x.get('gate_turns', 0) for x in mt)} gate turns, "
            f"{sum(x.get('gate_correct', 0) for x in mt)} correct)"
        )
    print(gate)
    scored = [x["outcome_reward"] for x in mt if "outcome_reward" in x]
    if scored:
        print(f"  outcome_reward      {mean(scored):+.3f}  ({len(scored)}/{len(mt)} scored)")
    else:
        print("  outcome_reward      (unscored — run score_records.py first)")

    firsts = Counter(a for a in (first_workup_action(r) for r in recs) if a)
    print(f"  first workup action ({sum(firsts.values())}/{len(recs)} episodes reached one)")
    for act, n in firsts.most_common():
        print(f"      {n:3d}  {act[:60]}")

    use = usage_by_phase(records)
    if use:
        print("  doctor tokens/call  (state/usage.jsonl)")
        for ph, u in use.items():
            print(
                f"      {ph:<8} {u['calls']:4d} calls  in {u['in']:8.0f}  "
                f"cached {u['cached']:7.0f}  out {u['out']:6.0f}"
            )
    else:
        print("  doctor tokens/call  (no state/usage.jsonl beside these records)")

    a = anchor_report(recs)
    print(
        f"  anchor groups       {a['groups']} "
        f"({a['multi_groups']} with >1 rollout), sizes {a['sizes']}"
    )
    if a["multi_groups"]:
        print(
            f"      anchors {a['distinct_anchors']}, "
            f"collided {a['collided_anchors']}, "
            f"rate {a['collision_rate']:.3f}"
            + (f", {a['unanchored_turns']} unanchored turns" if a["unanchored_turns"] else "")
        )
        for ph, r in a["by_phase"].items():
            print(f"      {ph:<8} {r['collided']}/{r['distinct']} = {r['rate']:.3f}")
    else:
        print(
            "      no group has 2+ rollouts — collect with --rollouts N "
            "--freeze-opening to measure step-level overlap"
        )

    if steps is not None:
        st = load_jsonl(steps)
        rew = [float(s.get("reward", 0.0)) for s in st]
        print(
            f"  steps               {len(st)} in "
            f"{len({s['episode_group_uid'] for s in st})} group(s), "
            f"reward mean {mean(rew):+.3f} / sum {sum(rew):+.2f}"
        )
        print(
            f"      terminal {sum(1 for s in st if s.get('done'))}, "
            f"unanchored {sum(1 for s in st if not s.get('anchor'))}, "
            f"with images {sum(1 for s in st if s.get('images'))}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--arm",
        required=True,
        action="append",
        type=Path,
        help="trajectories.jsonl of one arm; repeat per arm",
    )
    ap.add_argument(
        "--steps",
        action="append",
        type=Path,
        help="steps.jsonl from export_gigpo, paired with the "
        "--arm flags in order (give fewer than --arm to "
        "skip the rest)",
    )
    a = ap.parse_args()
    steps = a.steps or []
    for i, arm in enumerate(a.arm):
        if i:
            print()
        report_arm(arm, steps[i] if i < len(steps) else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
