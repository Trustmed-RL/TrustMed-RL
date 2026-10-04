"""Phase scheduler — AgentGym-RL's `StepRoundsScheduler` pattern, made
stateless and multi-budget.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

LH_ROOT = Path(os.environ.get("TRUSTMED_HOME") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(LH_ROOT))

import curriculum as cur


class PhaseScheduler:
    def __init__(
        self, schedule_path: str | Path, total_steps: int, max_batches_per_order: int | None = None
    ):
        self.sched = cur.load_schedule(Path(schedule_path))
        assert total_steps % self.sched["chunk_steps"] == 0, (
            "total_training_steps must be a chunk multiple"
        )
        self.total_steps = int(total_steps)
        b = (
            max_batches_per_order
            if max_batches_per_order is not None
            else os.environ.get("MAX_BATCHES_PER_ORDER") or 1
        )
        self.max_batches_per_order = max(1, int(b))

    def phase_at(self, step: int) -> int:
        """1-based phase index."""
        return cur.phase_at(self.sched, self.total_steps, step) + 1

    def caps_at(self, step: int) -> dict:
        p = cur.phase_at(self.sched, self.total_steps, step)
        caps = dict(self.sched["phases"][p])
        caps["phase"] = p + 1
        caps["r_ook"] = cur.r_ook_at(self.sched, self.total_steps, step)
        return caps

    def loop_guard(self, step: int) -> int:
        c = self.caps_at(step)
        return c["max_asks"] + 1 + (1 + self.max_batches_per_order) * c["max_workup_turns"] + 2 + 2

    def chunk_of(self, step: int) -> int:
        return step // self.sched["chunk_steps"]

    def is_chunk_boundary(self, step: int) -> bool:
        return step % self.sched["chunk_steps"] == 0
