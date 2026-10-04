"""In-process dynamic curriculum for verl-agent: Sampler + plan-backed Dataset."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset, Sampler

LH_ROOT = Path(os.environ.get("SP_CONSULT_HOME") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(LH_ROOT))

import curriculum as cur
import trainer.plan_store as ps


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).open(encoding="utf-8") if l.strip()]


class CurriculumPlan:
    """Owns curriculum state and the materialized visit plan, grown chunk by chunk. `row(i)` for i in [0, total_steps*tasks_per_step); emitting chunk k first ingests every slim record the env package has appended since the last emit (records the DAPO filter later drops included: an all-same group is strong mastery evidence)."""

    def __init__(
        self,
        pool_path: str | Path,
        schedule_path: str | Path,
        records_path: str | Path,
        state_dir: str | Path,
        seed: int,
        total_steps: int,
    ):
        self.sched = cur.load_schedule(Path(schedule_path))
        self.records_path = Path(records_path)
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state = cur.init_state(_read_jsonl(Path(pool_path)), self.sched, seed, total_steps)
        self._records_seen = 0
        self._plan: list[dict] = []
        self._emitted_chunks = 0
        self.per_step = self.sched["tasks_per_step"]
        self.total_rows = total_steps * self.per_step
        import hashlib as _h

        _sha = lambda p: _h.sha256(Path(p).read_bytes()).hexdigest()
        self.contract = {
            "pool_sha256": _sha(pool_path),
            "schedule_sha256": _sha(schedule_path),
            "seed": int(seed),
            "total_steps": int(total_steps),
            "tasks_per_step": int(self.per_step),
            "chunk_steps": int(self.sched["chunk_steps"]),
        }

    def resync_to(self, global_step: int) -> None:
        """Resume at `global_step` (the trainer's checkpoint step): RELOAD the persisted plans
        of every chunk the run had consumed (byte-identical to what it trained on; FATAL if
        one is missing) and rebuild the curriculum state from the records of steps < global_step
        only -- records after it belong to a segment being rolled back and are never read.
        """
        recs = _read_jsonl(self.records_path) if self.records_path.exists() else []
        lineage = ps.load_lineage(self.state_dir.parent / "lineage.json")
        res = ps.resync(
            self.state,
            self.state_dir,
            recs,
            int(global_step),
            self.sched,
            lineage,
            cur.ingest,
            cur.emit_chunk,
            contract=self.contract,
        )
        self._records_seen = len(recs)
        self._plan.extend(res["plan"])
        self._emitted_chunks = res["chunks"]
        self.resumed = {
            "step": int(global_step),
            "lineage": lineage,
            **{k: res[k] for k in ("chunks", "records", "stale")},
        }
        print(
            f"[curriculum] resumed at step {global_step} (lineage {lineage}): {res['chunks']} chunks reloaded, "
            f"{res['records']} records ingested, {res['stale']} stale records ignored",
            flush=True,
        )

    def _ingest_new(self) -> None:
        if not self.records_path.exists():
            return
        recs = _read_jsonl(self.records_path)
        if len(recs) > self._records_seen:
            lineage = ps.load_lineage(self.state_dir.parent / "lineage.json")
            fresh = recs[self._records_seen :]
            keep, _ = ps.split_records(fresh, self.state["total_steps"] + 1, lineage)
            if keep:
                cur.ingest(self.state, keep)
            self._records_seen = len(recs)

    def _emit_next(self) -> None:
        k = self._emitted_chunks
        out = cur.emit_chunk(self.state, k)
        ps.save_chunk(self.state_dir, k, out["header"], out["plan"], self.contract)
        self._plan.extend(out["plan"])
        self._emitted_chunks += 1

    def row(self, i: int) -> dict:
        assert 0 <= i < self.total_rows, i
        step = i // self.per_step
        while len(self._plan) <= i:
            self._ingest_new()
            self._emit_next()
        r = self._plan[i]
        assert r["step"] == step, (r["step"], step, "plan/step misalignment")
        return r


class SpConsultRLDataset(Dataset):
    """Plan-backed dataset. Construct with an explicit CurriculumPlan; the fork's `create_rl_dataset` custom_cls hook passes (data_paths, tokenizer, processor, config) and a thin adapter builds the plan from config keys and calls this."""

    def __init__(self, plan: CurriculumPlan, tokenizer=None, max_prompt_length: int = 16):
        self.plan = plan
        self.tokenizer = tokenizer
        self.max_prompt_length = int(max_prompt_length)

    def __len__(self) -> int:
        return self.plan.total_rows

    def __getitem__(self, i: int) -> dict:
        r = self.plan.row(i)
        L = self.max_prompt_length
        pad = int(getattr(self.tokenizer, "pad_token_id", 0) or 0)
        marker = f"sp_consult:{r['task_kind']}"
        if self.tokenizer is not None:
            ids = list(self.tokenizer.encode(marker, add_special_tokens=False))
        else:
            ids = [1, 2, 3]
        ids = (ids or [1])[: max(1, L - 1)]
        k = len(ids)
        input_ids = torch.tensor([pad] * (L - k) + ids, dtype=torch.long)
        attention_mask = torch.tensor([0] * (L - k) + [1] * k, dtype=torch.long)
        position_ids = torch.clamp(torch.cumsum(attention_mask, dim=0) - 1, min=0).long()
        ekw = dict(
            r["env_kwargs"], pmcid=r["pmcid"], task_kind=r["task_kind"], global_step=r["step"]
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "raw_prompt_ids": list(ids),
            "raw_prompt": [{"role": "user", "content": marker}],
            "data_source": "sp_consult",
            "index": i,
            "pmcid": r["pmcid"],
            "task_kind": r["task_kind"],
            "global_step": r["step"],
            "replay": r.get("replay", False),
            "env_kwargs": ekw,
        }


class SpConsultValDataset(Dataset):
    """Static validation set: validation never iterates the adaptive training plan (pulling future rows would emit later chunks from early records and freeze the curriculum). All rows of a PMCID-disjoint validation pool, each with its true task_kind (corrupt rows live, fixed corrupt_seed) at the schedule's frozen eval budgets, r_ook at the ramp end; the row count must be a multiple of the val batch size (`n`); `set_step` stamps the trainer step so trajectories_val.jsonl records name the checkpoint that produced them. Same tensor placeholders as the train dataset."""

    def __init__(
        self, pool_path, schedule_path, n: int = 64, tokenizer=None, max_prompt_length: int = 16
    ):
        from trainer.val_rows import load_val_rows

        self.sched = cur.load_schedule(Path(schedule_path))
        self.batch_size = int(n)
        self.rows = load_val_rows(Path(pool_path), self.sched, self.batch_size)
        self.current_step = -1
        self.tokenizer = tokenizer
        self.max_prompt_length = int(max_prompt_length)

    def set_step(self, step: int) -> None:
        self.current_step = int(step)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        L = self.max_prompt_length
        pad = int(getattr(self.tokenizer, "pad_token_id", 0) or 0)
        marker = f"sp_consult_val:{r.get('task_kind', 'answer')}"
        ids = (
            list(self.tokenizer.encode(marker, add_special_tokens=False))
            if self.tokenizer is not None
            else [1, 2, 3]
        )[: max(1, L - 1)] or [1]
        k = len(ids)
        input_ids = torch.tensor([pad] * (L - k) + ids, dtype=torch.long)
        attention_mask = torch.tensor([0] * (L - k) + [1] * k, dtype=torch.long)
        position_ids = torch.clamp(torch.cumsum(attention_mask, dim=0) - 1, min=0).long()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "raw_prompt_ids": list(ids),
            "raw_prompt": [{"role": "user", "content": marker}],
            "data_source": "sp_consult",
            "index": i,
            "pmcid": r["pmcid"],
            "task_kind": r["task_kind"],
            "global_step": self.current_step,
            "replay": False,
            "env_kwargs": dict(r["env_kwargs"], global_step=self.current_step),
        }


class CurriculumSequentialSampler(Sampler):
    """Row i at position i: the plan already is the schedule. Its own class (vs torch SequentialSampler) so resume can start mid-run: `start_row = resumed_global_step * tasks_per_step`."""

    def __init__(self, plan: CurriculumPlan, start_row: int = 0):
        self.plan = plan
        self.start_row = int(start_row)

    def __iter__(self):
        return iter(range(self.start_row, self.plan.total_rows))

    def __len__(self) -> int:
        return self.plan.total_rows - self.start_row
