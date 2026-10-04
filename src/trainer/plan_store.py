"""Persisted curriculum plans + resume reconciliation (torch-free; used by curriculum_sampler.py)."""

from __future__ import annotations

import json
import os
from pathlib import Path


def chunk_path(state_dir: Path, k: int) -> Path:
    return Path(state_dir) / f"chunk_{k:03d}.json"


def save_chunk(
    state_dir: Path, k: int, header: dict, plan: list[dict], contract: dict | None = None
) -> None:
    """`contract` = {pool_sha256, schedule_sha256, seed, total_steps, tasks_per_step, chunk_steps}: a persisted plan is only ever reloaded into the curriculum it was emitted for."""
    d = {"header": header, "plan": plan}
    if contract:
        d["contract"] = contract
    p = chunk_path(state_dir, k)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(d), encoding="utf-8")
    os.replace(tmp, p)


def load_chunk(state_dir: Path, k: int) -> dict | None:
    p = chunk_path(state_dir, k)
    if not p.is_file():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    if "plan" not in d:
        return None
    return d


def persisted_chunks(state_dir: Path) -> list[int]:
    out = []
    for p in Path(state_dir).glob("chunk_*.json"):
        try:
            k = int(p.stem.split("_")[-1])
        except ValueError:
            continue
        if load_chunk(state_dir, k) is not None:
            out.append(k)
    return sorted(out)


def chunks_needed(resume_step: int, chunk_steps: int, total_steps: int) -> int:
    """Chunks a run resumed at `resume_step` has already consumed: chunk k covers steps
    (k*chunk, (k+1)*chunk] in the trainer's 1-based step count, so the checkpoint after step N
    has consumed ceil(N / chunk) chunks.
    """
    if resume_step <= 0:
        return 0
    return min((resume_step + chunk_steps - 1) // chunk_steps, max(1, total_steps // chunk_steps))


def load_lineage(path: Path | None) -> list[tuple[int, int]]:
    """[(from_step, generation)] sorted; [(0, 0)] when no file (a never-relaunched run)."""
    if path is None or not Path(path).is_file():
        return [(0, 0)]
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    ents = sorted((int(e["from_step"]), int(e["generation"])) for e in d.get("lineage", []))
    return ents or [(0, 0)]


def active_generation(lineage: list[tuple[int, int]], step: int) -> int:
    gen = lineage[0][1]
    for from_step, g in lineage:
        if from_step <= step:
            gen = g
    return gen


def split_records(
    records: list[dict], resume_step: int, lineage: list[tuple[int, int]] | None = None
) -> tuple[list[dict], int]:
    """(records the resumed state may ingest, count skipped). A record is kept iff its curriculum
    step is < resume_step (checkpoint after batch N owns steps 0..N-1) AND its run_generation is
    the one active at that step on the lineage (records without the stamp count as generation 0).
    """
    lineage = lineage or [(0, 0)]
    keep, stale = [], 0
    for r in records:
        try:
            s = int(r.get("global_step", -1))
            g = int(r.get("run_generation", 0) or 0)
        except (TypeError, ValueError):
            s, g = -1, 0
        if s < resume_step and g == active_generation(lineage, s):
            keep.append(r)
        else:
            stale += 1
    return keep, stale


def resume_step_from_config(resume_mode: str, resume_from_path: str | None, ckpt_dir: str) -> int:
    """The trainer's resume step (fork ray_trainer._load_checkpoint): resume_path -> the named
    folder; auto -> the tracker file latest_checkpointed_iteration.txt (the fork's own source of
    truth), else 0 exactly like the trainer; disable -> 0.
    """
    mode = str(resume_mode or "disable")
    if mode == "disable":
        return 0
    if mode == "resume_path":
        if not resume_from_path or "global_step_" not in str(resume_from_path):
            raise ValueError(
                f"resume_mode=resume_path needs resume_from_path with global_step_N, got {resume_from_path!r}"
            )
        return int(str(resume_from_path).rstrip("/\\").split("global_step_")[-1])
    tracker = Path(ckpt_dir) / "latest_checkpointed_iteration.txt"
    if tracker.is_file():
        try:
            n = int(tracker.read_text().strip())
            if (Path(ckpt_dir) / f"global_step_{n}").exists():
                return n
        except ValueError:
            pass
    return 0


def resync(
    state: dict,
    state_dir: Path,
    records: list[dict],
    resume_step: int,
    sched: dict,
    lineage: list[tuple[int, int]] | None,
    ingest,
    emit,
    contract: dict | None = None,
) -> dict:
    """Rebuild a CurriculumPlan at `resume_step`: ingest the lineage's records (<= resume_step) and reload the persisted plans of the consumed chunks. `ingest(state, records)` / `emit` are curriculum.ingest / curriculum.emit_chunk (passed in so this stays torch- and import-free). Returns {"plan": rows, "chunks": n, "records": kept, "stale": skipped}."""
    chunk = sched["chunk_steps"]
    need = chunks_needed(int(resume_step), chunk, state["total_steps"])
    have = persisted_chunks(state_dir)
    missing = [k for k in range(need) if k not in have]
    if missing:
        raise RuntimeError(
            f"resume at step {resume_step} needs persisted plans for chunks "
            f"{list(range(need))}; missing {missing} under {state_dir}"
        )
    if contract:
        for k in range(need):
            c = load_chunk(state_dir, k).get("contract")
            if c != contract:
                raise RuntimeError(
                    f"persisted chunk {k} was emitted under another curriculum contract (or none): {c} != {contract}"
                )
    keep, stale = split_records(records, int(resume_step), lineage)
    if keep:
        ingest(state, keep)
    plan: list[dict] = []
    for k in range(need):
        plan.extend(load_chunk(state_dir, k)["plan"])
    return {"plan": plan, "chunks": need, "records": len(keep), "stale": stale}
