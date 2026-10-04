"""Offline GiGPO advantage simulator — test the reward/credit DESIGN decisions
before the 2B pilot exists, on synthetic (or replayed) episode groups.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass


@dataclass
class Traj:
    """One rollout: per-step rewards (forfeits are -0.1), a terminal scalar added to the last step, an anchor per step, and its clinical outcome value (the dx/mismatch/ook number before the coverage multiplier), used only by the outcome-variance filter."""

    step_rewards: list[float]
    terminal: float
    anchors: list[str]
    outcome: float = 0.0

    def __post_init__(self):
        assert len(self.step_rewards) == len(self.anchors)

    @property
    def stream(self) -> list[float]:
        r = list(self.step_rewards)
        r[-1] += self.terminal
        return r

    @property
    def total(self) -> float:
        return sum(self.stream)


def _norm(values: list[float], mode: str, std_floor: float = 0.1) -> list[float]:
    if len(values) < 2:
        return [0.0] * len(values)
    mu = statistics.fmean(values)
    if mode == "mean":
        return [v - mu for v in values]
    sd = statistics.stdev(values)
    if mode == "mean_std_floor":
        sd = max(sd, std_floor)
    if sd == 0:
        return [0.0] * len(values)
    return [(v - mu) / sd for v in values]


def record_pen_local(rec: dict, knobs: dict | None = None) -> bool:
    """Which charging path a stored record was trained under: its own budgets stamp wins; a
    launch_knobs.json PEN_LOCAL is the fallback for records that predate the stamp; absent
    everywhere = the global path.
    """
    b = rec.get("budgets") or {}
    if "local_penalty" in b:
        return bool(b["local_penalty"])
    if knobs and knobs.get("PEN_LOCAL") not in (None, ""):
        return str(knobs["PEN_LOCAL"]).strip() == "1"
    return False


def trainer_returns(
    rec: dict,
    gamma: float = 0.99,
    pen_local: bool | None = None,
    coef: float | None = None,
    knobs: dict | None = None,
) -> list[float]:
    """Per-row discounted return exactly as the trainer's A^S input saw it: the offline trainer truth for step_telemetry, collisions_by_phase and rollout_probe."""
    turns = rec.get("turns") or []
    if pen_local is None:
        pen_local = record_pen_local(rec, knobs)
    rewards = [float(t.get("reward") or 0.0) for t in turns]
    term = float((rec.get("metrics") or {}).get("terminal_reward") or 0.0)
    if coef is None:
        coef = -min(rewards) if rewards and min(rewards) < 0 else 0.1
    stream = ([0.0] * len(rewards)) if pen_local else list(rewards)
    if stream:
        stream[-1] += term
    G, out = 0.0, [0.0] * len(stream)
    for i in range(len(stream) - 1, -1, -1):
        G = stream[i] + gamma * G
        out[i] = G
    if pen_local:
        for i, r in enumerate(rewards):
            if r < 0:
                out[i] -= coef
    return out


def advantages(
    group: list[Traj],
    mode: str,
    gamma: float = 0.99,
    omega: float = 1.0,
    std_floor: float = 0.1,
    pen_local: bool = False,
    coef: float = 0.1,
) -> list[list[float]]:
    """A_i,t = A^E_i + omega * A^S_i,t."""
    if pen_local:
        traj_score = [t.terminal for t in group]
        mu = statistics.fmean(traj_score) if len(traj_score) > 1 else 0.0
        sd = statistics.stdev(traj_score) if len(traj_score) > 1 else 0.0
        if mode == "mean_std_floor":
            sd = max(sd, std_floor)

        def _row_ae(t: Traj, k: int) -> float:
            score = t.terminal - (coef if t.step_rewards[k] < 0 else 0.0)
            if len(group) < 2:
                return 0.0
            if mode == "mean":
                return score - mu
            return 0.0 if sd == 0 else (score - mu) / sd

        a_e_rows = [[_row_ae(t, k) for k in range(len(t.anchors))] for t in group]
    else:
        a_e = _norm([t.total for t in group], mode, std_floor)
        a_e_rows = [[a_e[i]] * len(t.anchors) for i, t in enumerate(group)]
    disc: list[list[float]] = []
    for t in group:
        if pen_local:
            s = [0.0] * len(t.step_rewards)
            s[-1] += t.terminal
        else:
            s = t.stream
        g = [0.0] * len(s)
        acc = 0.0
        for k in range(len(s) - 1, -1, -1):
            acc = s[k] + gamma * acc
            g[k] = acc
        if pen_local:
            for k, r in enumerate(t.step_rewards):
                if r < 0:
                    g[k] -= coef
        disc.append(g)
    opener = group[0].anchors[0] if group else None
    buckets: dict[str, list[tuple[int, int]]] = {}
    for i, t in enumerate(group):
        for k, a in enumerate(t.anchors):
            if a == opener:
                continue
            buckets.setdefault(a, []).append((i, k))
    a_s = [[0.0] * len(t.anchors) for t in group]
    for a, members in buckets.items():
        if len(members) < 2:
            continue
        gs = [disc[i][k] for i, k in members]
        normed = _norm(gs, mode, std_floor)
        for (i, k), val in zip(members, normed):
            a_s[i][k] = val
    return [
        [a_e_rows[i][k] + omega * a_s[i][k] for k in range(len(group[i].anchors))]
        for i in range(len(group))
    ]


def keep_total_variance(group: list[Traj]) -> bool:
    """The fork rule (PATCHES.md): keep iff total episode rewards vary."""
    return len({round(t.total, 6) for t in group}) > 1


def keep_outcome_variance(group: list[Traj]) -> bool:
    """The proposed gate: keep iff the CLINICAL outcome varies (a forfeit-only
    difference is not a training signal about diagnosis).
    """
    return len({round(t.outcome, 6) for t in group}) > 1


def _mk(n, terminals, forfeits=None, anchors=None, outcomes=None) -> list[Traj]:
    forfeits = forfeits or [0] * n
    out = []
    for i in range(n):
        steps = [0.0, 0.0, 0.0]
        for _ in range(forfeits[i]):
            steps.insert(1, -0.1)
        anch = anchors[i] if anchors else ["s1"] + ["s2"] * (len(steps) - 1)
        out.append(
            Traj(
                step_rewards=steps,
                terminal=terminals[i],
                anchors=anch,
                outcome=(outcomes[i] if outcomes else terminals[i]),
            )
        )
    return out


SHAPES = {
    "all-correct, ONE forfeit (decision 12/11)": _mk(9, [1.0] * 9, forfeits=[0] * 8 + [3]),
    "gate anchor: 0.9 catch vs 0 rubber-stamp (4 vs 5)": _mk(9, [0.9] * 4 + [0.0] * 5),
    "near-binary: eight 1.0, one 0.9 gate-return": _mk(9, [1.0] * 8 + [0.891]),
    "abstain 0.4 vs attempt 1.0 (mixed)": _mk(9, [0.4, 0.4, 0.4, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
    "all wrong (0) but forfeit counts differ (filter test)": _mk(
        9, [0.0] * 9, forfeits=[0, 0, 0, 0, 1, 1, 2, 2, 3], outcomes=[0.0] * 9
    ),
}


def main() -> None:
    print(
        "=== decision 12: how a lone forfeit's step-advantage compares to the clinical signal ==="
    )
    g = SHAPES["all-correct, ONE forfeit (decision 12/11)"]
    for mode in ("mean_std", "mean", "mean_std_floor"):
        A = advantages(g, mode)
        forfeiter = A[-1]
        clean = A[0]
        print(
            f"  {mode:16s} forfeiter step-adv range [{min(forfeiter):+.2f},"
            f" {max(forfeiter):+.2f}]  clean traj adv {clean[0]:+.3f}"
        )
    print(
        "  (mean_std turns a single -0.1 in a tiny anchor into an order-1 "
        "advantage; mean and the std-floor damp it.)"
    )

    print("\n=== decision 12: gate-anchor contrast survives every mode ===")
    g = SHAPES["gate anchor: 0.9 catch vs 0 rubber-stamp (4 vs 5)"]
    for mode in ("mean_std", "mean", "mean_std_floor"):
        A = advantages(g, mode)
        catch = statistics.fmean(A[0])
        stamp = statistics.fmean(A[4])
        print(
            f"  {mode:16s} catch {catch:+.3f}  rubber-stamp {stamp:+.3f}  gap {catch - stamp:+.3f}"
        )

    print("\n=== decision 11: which groups each filter keeps ===")
    tot = out = 0
    kept_but_no_outcome = []
    for name, g in SHAPES.items():
        kt, ko = keep_total_variance(g), keep_outcome_variance(g)
        tot += int(kt)
        out += int(ko)
        flag = " <- kept by total-var, NO outcome contrast" if kt and not ko else ""
        print(f"  total={int(kt)} outcome={int(ko)}  {name}{flag}")
        if kt and not ko:
            kept_but_no_outcome.append(name)
    print(f"  total-variance keeps {tot}/{len(SHAPES)}; outcome-variance keeps {out}/{len(SHAPES)}")
    if kept_but_no_outcome:
        print("  groups the fork trains on but that teach ONLY 'do not forfeit':")
        for n in kept_but_no_outcome:
            print("    -", n)

    print("\n=== decision 12 summary knob: std_floor sweep on the forfeit shape ===")
    g = SHAPES["all-correct, ONE forfeit (decision 12/11)"]
    for f in (0.0, 0.05, 0.1, 0.25):
        A = advantages(g, "mean_std_floor", std_floor=f)
        print(
            f"  std_floor={f:<4}  max |step-adv| on the forfeiter {max(abs(x) for x in A[-1]):.2f}"
        )


if __name__ == "__main__":
    main()
