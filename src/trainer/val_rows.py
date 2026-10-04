"""Validation rows for SpConsultValDataset -- torch-free so it can be unit-tested anywhere."""

from __future__ import annotations

import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def eval_kwargs(sched: dict) -> dict:
    ev = sched["eval_budgets"]
    return {
        "max_asks": int(ev["max_asks"]),
        "max_workup_turns": int(ev["max_workup_turns"]),
        "max_searches": int(ev["max_searches"]),
        "search_enabled": bool(ev.get("max_searches", 0) > 0),
        "r_ook": float(sched["r_ook"]["end"]),
    }


def load_val_rows(pool_path: Path, sched: dict, batch_size: int) -> list[dict]:
    """-> rows with `task_kind`, `pmcid`, `env_kwargs` (eval budgets + corruption for corrupt rows)."""
    rows = read_jsonl(pool_path)
    if not rows:
        raise ValueError(f"validation pool {pool_path} is empty")
    if batch_size <= 0 or len(rows) % int(batch_size) != 0:
        raise ValueError(
            f"validation pool {pool_path} has {len(rows)} rows, not a multiple of "
            f"val_batch_size={batch_size} (every val batch must be full)"
        )
    base = eval_kwargs(sched)
    out = []
    for i, r in enumerate(rows):
        kind = str(r.get("task_kind") or "answer")
        if kind not in ("answer", "corrupt", "ook"):
            raise ValueError(f"row {i} of {pool_path}: task_kind {kind!r}")
        kw = dict(
            base,
            pmcid=r["pmcid"],
            task_kind=kind,
            corrupt_rate=1.0 if kind == "corrupt" else 0.0,
            corrupt_seed=str(r.get("corrupt_seed") or f"val-{r['pmcid']}")
            if kind == "corrupt"
            else "0",
        )
        out.append({"pmcid": r["pmcid"], "task_kind": kind, "env_kwargs": kw})
    return out


def assert_disjoint(val_path: Path, train_path: Path) -> None:
    v = {r["pmcid"] for r in read_jsonl(val_path)}
    t = {r["pmcid"] for r in read_jsonl(train_path)}
    if v & t:
        raise ValueError(
            f"validation pool {val_path} shares {len(v & t)} cases with the train pool "
            f"{train_path}: {sorted(v & t)[:5]}"
        )
