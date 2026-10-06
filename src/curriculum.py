"""2D ScalingInter curriculum controller — the offline-testable library."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path

KINDS = ("answer", "corrupt", "ook")
R_MAX = {"answer": 1.0, "corrupt": 0.9}
OOK_IR = (2, 0)


def load_schedule(path: Path) -> dict:
    global _EPS_FLOOR
    sched = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_schedule(sched)
    _EPS_FLOOR = bool(sched["sampler"].get("eps_floor", False))
    return sched


def validate_schedule(s: dict) -> None:
    ph = s["phases"]
    assert len(ph) == len(s["phase_fracs"]), "phase count mismatch"
    assert abs(sum(s["phase_fracs"]) - 1.0) < 1e-9, "phase_fracs must sum to 1"
    for a, b in zip(ph, ph[1:]):
        for k in ("max_asks", "max_workup_turns", "max_searches"):
            assert a[k] <= b[k], f"caps must be monotone nondecreasing ({k})"
    for i, p in enumerate(ph):
        assert p["max_asks"] >= 1, "max_asks ceiling must admit an ask"
        assert p["admits"] == i + 1, "admits must be 1..p (columns H1..Hp)"
    last, ev = ph[-1], s["eval_budgets"]
    for k in ("max_asks", "max_workup_turns", "max_searches"):
        assert last[k] == ev[k], "final phase MUST train at the frozen eval budgets"
    assert s["r_ook"]["end"] < 0.65, (
        "r_ook must stay below r_mismatch_partial (reward-design invariant)"
    )
    q = s["quotas"]
    assert 0 < q["ook"] < 1 and 0 < q["corrupt_start"] <= q["corrupt_end"] < 1
    assert q["ook"] + q["corrupt_end"] < 1, "answer quota would go negative"
    assert s["jitter"]["workup_width"] >= 0 and s["jitter"]["asks_width"] >= 0


def phase_bounds(sched: dict, total_steps: int) -> list[tuple[int, int]]:
    """[start, end) global-step range per phase, snapped to the chunk grid so
    no advantage batch ever spans a caps change.
    """
    chunk = sched["chunk_steps"]
    n_chunks = max(len(sched["phases"]), total_steps // chunk)
    cum, bounds, start = 0.0, [], 0
    for i, frac in enumerate(sched["phase_fracs"]):
        cum += frac
        end = (
            n_chunks
            if i == len(sched["phase_fracs"]) - 1
            else max(start // chunk + 1, round(cum * n_chunks))
        ) * chunk
        end = min(end, n_chunks * chunk)
        bounds.append((start, end))
        start = end
    return bounds


def phase_at(sched: dict, total_steps: int, step: int) -> int:
    """0-based phase index for a global step."""
    for i, (a, b) in enumerate(phase_bounds(sched, total_steps)):
        if a <= step < b:
            return i
    return len(sched["phases"]) - 1


def _ramp(step: int, total_steps: int, start: float, end: float, ramp_frac: float) -> float:
    horizon = max(1.0, total_steps * ramp_frac)
    return start + (end - start) * min(1.0, step / horizon)


def r_ook_at(sched: dict, total_steps: int, step: int) -> float:
    r = sched["r_ook"]
    return round(_ramp(step, total_steps, r["start"], r["end"], r["ramp_frac"]), 4)


def quotas_at(sched: dict, total_steps: int, step: int) -> dict[str, float]:
    q = sched["quotas"]
    corrupt = _ramp(step, total_steps, q["corrupt_start"], q["corrupt_end"], q["ramp_frac"])
    return {
        "ook": q["ook"],
        "corrupt": round(corrupt, 4),
        "answer": round(1.0 - q["ook"] - corrupt, 4),
    }


def _hint(*parts) -> int:
    return int.from_bytes(
        hashlib.sha1("|".join(str(p) for p in parts).encode()).digest()[:8], "big"
    )


def _rng(*parts) -> random.Random:
    return random.Random(_hint(*parts))


def ir_effective(task: dict) -> tuple[int, int]:
    if task["task_kind"] == "ook":
        return OOK_IR
    return int(task["ir_workup"]), int(task["ir_searches"])


def fits(caps: dict, ir_workup: int, ir_searches: int) -> bool:
    """Search consumes BOTH a workup turn and the search budget (trustmed
    run loop + validator), so the workup cap must cover their sum.
    """
    return (
        caps["max_workup_turns"] >= ir_workup + ir_searches and caps["max_searches"] >= ir_searches
    )


def horizon_class(sched: dict, task: dict) -> int:
    w, s = ir_effective(task)
    for i, caps in enumerate(sched["phases"]):
        if fits(caps, w, s):
            return i + 1
    return len(sched["phases"]) + 1


def jitter_caps(sched: dict, task: dict, step: int, seed: int, caps: dict) -> dict:
    """Per-task-VISIT budget jitter under the phase cap (anti-rigidity,
    DoctorAgent-RL). Identical across the visit's rollouts by construction
    (keyed on step, not rollout). Floor >= the task's own feasibility.
    """
    w, s = ir_effective(task)
    j = sched["jitter"]
    lo_w = min(
        caps["max_workup_turns"], max(caps["max_workup_turns"] - j["workup_width"], w + s, 2)
    )
    lo_a = max(caps["max_asks"] - j["asks_width"], 1)
    r = _rng(seed, "jit", task["pmcid"], task["task_kind"], step)
    return {
        "max_asks": r.randint(lo_a, caps["max_asks"]),
        "max_workup_turns": r.randint(lo_w, caps["max_workup_turns"]),
        "max_searches": caps["max_searches"],
        "search_enabled": bool(caps["search_enabled"]),
    }


def corrupt_seed_for(task: dict, step: int, seed: int) -> str:
    """Fresh corruption identity per visit: the env contract requires the
    seed to differ across training iterations so case identity never
    predicts the label (trustmed Config.corrupt_seed).
    """
    return f"c{_hint(seed, 'cseed', task['pmcid'], step):016x}"


def _uid(pmcid: str, kind: str) -> str:
    return f"{pmcid}:{kind}"


def init_state(pool_rows: list[dict], sched: dict, seed: int, total_steps: int) -> dict:
    tasks, overflow = {}, 0
    for row in pool_rows:
        kind = row.get("task_kind", "answer")
        assert kind in KINDS, kind
        h = int(row.get("horizon_class") or horizon_class(sched, {**row, "task_kind": kind}))
        if h > len(sched["phases"]):
            overflow += 1
            continue
        m0 = min(0.95, max(0.05, 1.0 - float(row.get("curriculum_score", 0.5))))
        nu = sched["sampler"]["nu"]
        tasks[_uid(row["pmcid"], kind)] = {
            "pmcid": row["pmcid"],
            "task_kind": kind,
            "difficulty": str(row.get("difficulty") or ""),
            "rung": int(row.get("spectrum_rank", 0)),
            "ir_workup": int(row.get("ir_workup", 2)),
            "ir_searches": int(row.get("ir_searches", 0)),
            "h": h,
            "m0": round(m0, 4),
            "S": round(nu * m0, 4),
            "F": round(nu * (1 - m0), 4),
            "ema_std": None,
            "last_visit": -1,
            "visits": 0,
            "window": [],
            "succ_at": {},
            "retired": False,
            "shelved": False,
            "ingested": [],
        }
    return {
        "seed": seed,
        "total_steps": total_steps,
        "overflow_excluded": overflow,
        "cursor_chunk": 0,
        "schedule": sched,
        "tasks": tasks,
    }


def ingest(state: dict, records: list[dict]) -> dict:
    """Fold slim episode records into the state. Records of one (task, step)
    form one VISIT (a GiGPO group). Idempotent per visit.
    """
    sched, total = state["schedule"], state["total_steps"]
    smp = sched["sampler"]
    by_visit: dict[tuple[str, int], list[dict]] = {}
    for r in records:
        uid = _uid(r["pmcid"], r.get("task_kind", "answer"))
        if uid in state["tasks"]:
            by_visit.setdefault((uid, int(r["global_step"])), []).append(r)
    stats = {
        "visits": 0,
        "skipped_dup": 0,
        "ratchets": 0,
        "retired": 0,
        "unretired": 0,
        "shelved": 0,
    }
    for uid, step in sorted(by_visit, key=lambda k: (k[1], k[0])):
        t = state["tasks"][uid]
        if step in t["ingested"]:
            stats["skipped_dup"] += 1
            continue
        group = by_visit[(uid, step)]
        rmax = r_ook_at(sched, total, step) if t["task_kind"] == "ook" else R_MAX[t["task_kind"]]
        rhat = [max(0.0, min(1.0, r["reward"] / rmax)) for r in group]
        mu = statistics.fmean(rhat)
        std = statistics.pstdev(rhat) if len(rhat) > 1 else 0.0
        succ = sum(1 for x in rhat if x >= smp["success_bar"])
        g = smp["gamma"]
        t["S"] = round(g * t["S"] + succ, 4)
        t["F"] = round(g * t["F"] + (len(rhat) - succ), 4)
        b = smp["ema_beta"]
        t["ema_std"] = round(std if t["ema_std"] is None else (1 - b) * t["ema_std"] + b * std, 4)
        t["window"] = (t["window"] + [[round(mu, 4), round(std, 4), len(rhat), step]])[-4:]
        t["last_visit"], t["visits"] = step, t["visits"] + 1
        t["ingested"] = sorted(set(t["ingested"] + [step]))[-64:]
        for r, x in zip(group, rhat):
            if x < smp["success_bar"]:
                continue
            w = int(r.get("workup_turns", 99))
            s = int(r.get("searches", 99))
            for pi, caps in enumerate(sched["phases"]):
                if caps["max_workup_turns"] >= w and caps["max_searches"] >= s:
                    key = str(pi)
                    t["succ_at"][key] = t["succ_at"].get(key, 0) + 1
                    if t["succ_at"][key] >= smp["ratchet_successes"] and pi + 1 < t["h"]:
                        t["h"] = pi + 1
                        stats["ratchets"] += 1
                    break
        ret, shv = smp["retire"], smp["shelve"]
        win = t["window"]
        if t["retired"]:
            if mu < ret["unretire_mu_lt"]:
                t["retired"] = False
                stats["unretired"] += 1
        elif len(win) >= ret["window"] and all(
            v[0] > ret["mu_gt"] and v[1] < ret["std_lt"] for v in win[-ret["window"] :]
        ):
            t["retired"] = True
            stats["retired"] += 1
        elif (
            len(win) >= shv["window"]
            and all(v[0] < shv["mu_lt"] and v[1] < shv["std_lt"] for v in win[-shv["window"] :])
            and t["rung"] >= _frontier_rung(state) + shv["rung_gap"]
        ):
            t["shelved"] = True
            stats["shelved"] += 1
        stats["visits"] += 1
    return stats


def mastery(t: dict) -> float:
    return t["S"] / max(1e-9, t["S"] + t["F"])


def _frontier_rung(state: dict) -> float:
    rungs = [
        t["rung"]
        for t in state["tasks"].values()
        if not t["retired"] and not t["shelved"] and 0.2 <= mastery(t) <= 0.8
    ]
    if not rungs:
        rungs = [t["rung"] for t in state["tasks"].values()]
    return statistics.median(rungs) if rungs else 13


def admitted(state: dict, phase_idx: int) -> list[str]:
    lim = state["schedule"]["phases"][phase_idx]["admits"]
    cap = int(state["schedule"]["sampler"].get("max_visits") or 0)
    return [
        uid
        for uid, t in state["tasks"].items()
        if t["h"] <= lim
        and not t["retired"]
        and not t["shelved"]
        and (cap <= 0 or int(t.get("visits", 0)) < cap)
    ]


def _priorities(state: dict, uids: list[str], step: int) -> dict[str, float]:
    """PLR: rank-based score prioritization + ADDITIVE staleness mix."""
    smp = state["schedule"]["sampler"]
    ts = state["tasks"]

    def score(uid):
        t = ts[uid]
        if t["ema_std"] is not None:
            return t["ema_std"]
        return math.sqrt(t["m0"] * (1 - t["m0"]))

    order = sorted(uids, key=lambda u: (-score(u), u))
    ps = {u: (1.0 / (i + 1)) ** (1.0 / smp["beta_r"]) for i, u in enumerate(order)}
    zs = sum(ps.values()) or 1.0
    pc = {u: float(step - ts[u]["last_visit"]) for u in uids}
    zc = sum(pc.values()) or 1.0
    rho = smp["rho"]
    return {u: (1 - rho) * ps[u] / zs + rho * pc[u] / zc for u in uids}


_EPS_FLOOR = False


def _weighted_draw(rng: random.Random, pri: dict[str, float], k: int, eps: float) -> list[str]:
    """k draws WITHOUT replacement from one stratum: an eps share uniform
    (exploration floor / capacity probes), the rest priority-weighted.
    """
    pool = dict(pri)
    picked: list[str] = []
    n_eps = int(round(eps * min(k, len(pool))))
    if eps > 0 and n_eps == 0 and min(k, len(pool)) >= 1 and _EPS_FLOOR:
        n_eps = 1
    for i in range(min(k, len(pool)) if pool else 0):
        uids = sorted(pool)
        if i < n_eps:
            u = uids[rng.randrange(len(uids))]
        else:
            weights = [pool[x] for x in uids]
            tot = sum(weights)
            if tot <= 0:
                u = uids[rng.randrange(len(uids))]
            else:
                x, acc, u = rng.random() * tot, 0.0, uids[-1]
                for cand, wgt in zip(uids, weights):
                    acc += wgt
                    if x <= acc:
                        u = cand
                        break
        picked.append(u)
        pool.pop(u)
    return picked


def emit_chunk(state: dict, chunk_idx: int) -> dict:
    """Task-visit plan for one chunk: kind quotas are exact at chunk level (largest remainder) and sliced into steps; draws are without replacement within a step and refilled across steps, so small admitted strata revisit across steps but never inside one batch."""
    sched, total, seed = state["schedule"], state["total_steps"], state["seed"]
    chunk = sched["chunk_steps"]
    per_step = sched["tasks_per_step"]
    start = chunk_idx * chunk
    p_idx = phase_at(sched, total, start)
    caps = sched["phases"][p_idx]
    quotas = quotas_at(sched, total, start)
    smp = sched["sampler"]

    adm = admitted(state, p_idx)
    by_kind = {k: [u for u in adm if state["tasks"][u]["task_kind"] == k] for k in KINDS}
    retired = sorted(
        u for u, t in state["tasks"].items() if t["retired"] and t["h"] <= caps["admits"]
    )

    n_total = chunk * per_step
    n_replay = min(int(round(smp["replay_frac"] * n_total)), len(retired))
    n_live = n_total - n_replay
    avail = {k: q for k, q in quotas.items() if by_kind[k]}
    z = sum(avail.values()) or 1.0
    raw = {k: n_live * q / z for k, q in avail.items()}
    counts = {k: int(raw[k]) for k in avail}
    for k in sorted(avail, key=lambda k: raw[k] - counts[k], reverse=True):
        if sum(counts.values()) < n_live:
            counts[k] += 1
    deviation = {k: round(counts.get(k, 0) / max(1, n_live) - quotas[k], 4) for k in KINDS}

    plan, kind_seq = [], []
    for k in KINDS:
        kind_seq += [k] * counts.get(k, 0)
    kind_seq += ["replay"] * n_replay
    _rng(seed, "shuffle", chunk_idx).shuffle(kind_seq)

    cap = int(smp.get("max_visits") or 0)
    planned: dict[str, int] = {}
    n_short = n_topup = 0

    def _row(uid: str, step: int, replay: bool) -> dict:
        t = state["tasks"][uid]
        kw = jitter_caps(sched, t, step, seed, caps)
        kw["corrupt_rate"] = 1.0 if t["task_kind"] == "corrupt" else 0.0
        kw["corrupt_seed"] = corrupt_seed_for(t, step, seed) if t["task_kind"] == "corrupt" else "0"
        kw["r_ook"] = r_ook_at(sched, total, step)
        kw["difficulty"] = t.get("difficulty", "")
        return {
            "step": step,
            "pmcid": t["pmcid"],
            "task_kind": t["task_kind"],
            "replay": replay,
            "env_kwargs": kw,
        }

    for si in range(chunk):
        step = start + si
        picked_this_step: set[str] = set()
        picked_pmcids: set[str] = set()

        def _open(uid: str) -> bool:
            t = state["tasks"][uid]
            return (
                uid not in picked_this_step
                and t["pmcid"] not in picked_pmcids
                and (cap <= 0 or t["visits"] + planned.get(uid, 0) < cap)
            )

        def _take(uid: str, replay: bool) -> None:
            picked_this_step.add(uid)
            picked_pmcids.add(state["tasks"][uid]["pmcid"])
            planned[uid] = planned.get(uid, 0) + 1
            plan.append(_row(uid, step, replay))

        step_kinds = kind_seq[si * per_step : (si + 1) * per_step]
        short = 0
        for k in KINDS + ("replay",):
            need = step_kinds.count(k)
            if not need:
                continue
            rng = _rng(seed, "draw", chunk_idx, si, k)
            if k == "replay":
                pool = [u for u in retired if _open(u)]
                sel = []
                while pool and len(sel) < need:
                    u = pool[rng.randrange(len(pool))]
                    sel.append(u)
                    pool = [
                        x
                        for x in pool
                        if x != u and state["tasks"][x]["pmcid"] != state["tasks"][u]["pmcid"]
                    ]
            else:
                cand = [u for u in by_kind[k] if _open(u)]
                sel = _weighted_draw(rng, _priorities(state, cand, step), need, smp["eps_uniform"])
            taken = 0
            for uid in sel:
                if _open(uid):
                    _take(uid, k == "replay")
                    taken += 1
            short += need - taken
        if short:
            rng = _rng(seed, "topup", chunk_idx, si)
            filled = 0
            while filled < short:
                cand = [u for kk in KINDS for u in by_kind[kk] if _open(u)]
                if not cand:
                    break
                sel = _weighted_draw(
                    rng, _priorities(state, cand, step), short - filled, smp["eps_uniform"]
                )
                got = 0
                for uid in sel:
                    if _open(uid):
                        _take(uid, False)
                        got += 1
                if not got:
                    break
                filled += got
            n_topup += filled
            n_short += short

    frontier = [mastery(state["tasks"][u]) for u in adm]
    gate = state["schedule"]["gate_measure_only"]
    gate_frac = (
        (sum(1 for m in frontier if m >= gate["mastery_bar"]) / len(frontier)) if frontier else 0.0
    )
    header = {
        "chunk": chunk_idx,
        "phase": p_idx + 1,
        "steps": [start, start + chunk],
        "caps": caps,
        "quotas": quotas,
        "r_ook": r_ook_at(sched, total, start),
        "admitted": len(adm),
        "replay": n_replay,
        "quota_deviation": deviation,
        "short": n_short,
        "topup": n_topup,
        "max_visits": cap,
        "gate_measure_only": {
            "frac_mastered": round(gate_frac, 4),
            "would_advance": gate_frac >= gate["frontier_frac"],
        },
    }
    return {"header": header, "plan": plan}


def report(state: dict) -> dict:
    cells: dict[str, dict] = {}
    for t in state["tasks"].values():
        key = f"r{t['rung']:02d}xH{t['h']}"
        c = cells.setdefault(key, {"n": 0, "visited": 0, "retired": 0, "shelved": 0, "m_sum": 0.0})
        c["n"] += 1
        c["visited"] += 1 if t["visits"] else 0
        c["retired"] += t["retired"]
        c["shelved"] += t["shelved"]
        c["m_sum"] += mastery(t)
    for c in cells.values():
        c["m_mean"] = round(c.pop("m_sum") / c["n"], 4)
    ts = state["tasks"].values()
    return {
        "tasks": len(state["tasks"]),
        "overflow_excluded": state["overflow_excluded"],
        "retired": sum(t["retired"] for t in ts),
        "shelved": sum(t["shelved"] for t in ts),
        "visited": sum(1 for t in ts if t["visits"]),
        "frontier_rung": _frontier_rung(state),
        "cells": dict(sorted(cells.items())),
    }


def _read_rows(path: Path) -> list[dict]:
    """A pool or record file as a list of dicts: parquet or JSONL."""
    path = Path(path)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist()
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("init")
    pi.add_argument("--pool", type=Path, required=True)
    pi.add_argument("--schedule", type=Path, required=True)
    pi.add_argument("--out", type=Path, required=True)
    pi.add_argument("--seed", type=int, required=True)
    pi.add_argument("--total-steps", type=int, required=True)
    pg = sub.add_parser("ingest")
    pg.add_argument("--state", type=Path, required=True)
    pg.add_argument("--records", type=Path, required=True)
    pe = sub.add_parser("emit")
    pe.add_argument("--state", type=Path, required=True)
    pe.add_argument("--chunk", type=int, required=True)
    pr = sub.add_parser("report")
    pr.add_argument("--state", type=Path, required=True)
    a = p.parse_args()

    if a.cmd == "init":
        sched = load_schedule(a.schedule)
        assert a.total_steps % sched["chunk_steps"] == 0, "total steps must be a chunk multiple"
        state = init_state(_read_rows(a.pool), sched, a.seed, a.total_steps)
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "state.json").write_text(json.dumps(state), encoding="utf-8")
        rep = report(state)
        print(json.dumps({k: rep[k] for k in ("tasks", "overflow_excluded", "frontier_rung")}))
        for pi_, caps in enumerate(sched["phases"]):
            adm = admitted(state, pi_)
            for k in KINDS:
                n = sum(1 for u in adm if state["tasks"][u]["task_kind"] == k)
                need = quotas_at(sched, a.total_steps, 0)[k] * sched["tasks_per_step"]
                flag = "OK" if n >= need else "THIN (will renormalize)"
                print(f"P{pi_ + 1} {k}: admitted {n} vs per-step need ~{need:.1f} {flag}")
    elif a.cmd == "ingest":
        state = json.loads(a.state.read_text(encoding="utf-8"))
        stats = ingest(state, _read_rows(a.records))
        a.state.write_text(json.dumps(state), encoding="utf-8")
        print(json.dumps(stats))
    elif a.cmd == "emit":
        state = json.loads(a.state.read_text(encoding="utf-8"))
        out = emit_chunk(state, a.chunk)
        path = a.state.parent / f"chunk_{a.chunk:03d}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(out["header"]) + "\n")
            for row in out["plan"]:
                f.write(json.dumps(row) + "\n")
        state["cursor_chunk"] = a.chunk + 1
        a.state.write_text(json.dumps(state), encoding="utf-8")
        print(json.dumps(out["header"]))
    elif a.cmd == "report":
        state = json.loads(a.state.read_text(encoding="utf-8"))
        rep = report(state)
        path = a.state.parent / f"report_{state['cursor_chunk']:03d}.json"
        path.write_text(json.dumps(rep, indent=1), encoding="utf-8")
        print(json.dumps({k: rep[k] for k in rep if k != "cells"}))


if __name__ == "__main__":
    main()
