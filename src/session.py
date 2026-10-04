"""Stepwise trainer facade over environment.run_episode: reset()/step() for verl-agent collectors,
with parity by construction.
"""

from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Any, Callable

import rewards as gx
import outcomes as sr
from environment import Config, run_episode

_ABORT = object()


class _Abort(Exception):
    pass


def kwargs_to_config(
    kw: dict,
    *,
    profiles: Path,
    out: Path,
    search_url: str | None = None,
    gate_bank_dir: Path | None = None,
    images_dir: Path | None = None,
    post_baseline: bool = False,
    identity_anchor: bool = False,
    panel_policy: str | None = None,
    **extra,
) -> Config:
    """Curriculum env_kwargs -> per-episode Config (the pinned bridge)."""
    base = dict(
        profiles=profiles,
        out=out,
        rl_mode=True,
        allow_defer=True,
        patient_temperature=0.0,
        max_asks=int(kw["max_asks"]),
        max_workup_turns=int(kw["max_workup_turns"]),
        max_searches=int(kw["max_searches"]),
        search_url=(search_url if kw.get("search_enabled") else None),
        corrupt_rate=float(kw.get("corrupt_rate", 0.0)),
        corrupt_seed=str(kw.get("corrupt_seed", "0")),
        post_baseline=bool(post_baseline),
        identity_anchor=bool(identity_anchor),
    )
    if panel_policy is not None:
        base["panel_policy"] = str(panel_policy)
    if gate_bank_dir is not None:
        base["gate_bank_dir"] = gate_bank_dir
    if images_dir is not None:
        base["images_dir"] = images_dir
    base.update(extra)
    return Config(**base)


def lexical_outcome_reward(rec: dict, gold: str) -> float:
    """Judge-free proxy reward: strict normalized match on the final diagnosis. An ontology or lexicon overlay replaces it via TrustMedSession(score_fn=) without touching the session."""
    if rec["final"].get("kind") != "final_diagnosis":
        return 0.0
    return (
        1.0 if sr.normalize(rec["final"].get("diagnosis", "")) == sr.normalize(gold or "") else 0.0
    )


def session_gold(row: dict) -> str:
    """The gold the live dx channel grades against: `score_records.gold_of`
    (orpha_name, case-report `diagnosis` only as a fallback) - the SAME gold
    the offline lexical outcome and the judge panel use, so a live reward, an
    exported step and an SFT keep never disagree about the answer.
    """
    return sr.gold_of(row)


def _is_dx(rec: dict) -> bool:
    """The only episodes a dx scorer may price: `classify_outcome` says dx.
    `metrics.outcome == "final_diagnosis"` is NOT enough -- a synthesized
    format-failure terminal and an "undetermined" placeholder carry it too.
    """
    return sr.classify_outcome(rec)[0] == "dx"


def _final_answer(rec: dict) -> str:
    mt = rec.get("metrics") or {}
    return str(mt.get("final_answer") or (rec.get("final") or {}).get("diagnosis") or "")


def ontology_outcome_reward(rec: dict, gold: str) -> float:
    """The guarded ontology scorer as a gym score_fn (`judge=ontology`, the fallback/ablation):
    deterministic guarded ontology verdict, binary at the CORE bar.
    """
    if not _is_dx(rec):
        return 0.0
    d = sr.ontology_outcome_detail(gold, _final_answer(rec))
    mt = rec["metrics"]
    mt["dx_score"] = float(d["score"])
    mt["dx_scorer"] = sr.DX_SCORER_ONTOLOGY
    mt["dx_source"] = "ontology"
    mt["dx_guard_reason"] = d["reason"]
    return float(d["score"])


def panel_score_fn(panel):
    """The judge-in-the-loop score_fn (`judge=panel`): the frozen eval panel
    + guarded ontology floor through `live_panel.LivePanel`, binary at the
    bar. Everything the panel decided lands on the record (`metrics.panel`,
    the graded fused value, the scorer identity, the injection flags) so an
    exported step and a replay carry the same evidence the reward did.
    """

    def score_fn(rec: dict, gold: str) -> float:
        if not _is_dx(rec):
            return 0.0
        meta = rec.get("meta") or {}
        b = panel.score(
            gold,
            _final_answer(rec),
            ctx={"pmcid": meta.get("pmcid"), "rollout_idx": meta.get("rollout_idx")},
        )
        mt = rec["metrics"]
        mt["panel"] = {
            k: b.get(k)
            for k in (
                "judges",
                "n_judges",
                "verdict_source",
                "ontology",
                "guard_reason",
                "fallback_reason",
                "key",
                "bar",
            )
        }
        mt["fused_t_reward"] = b.get("fused_t_reward")
        mt["prefusion_t_reward"] = b.get("prefusion_t_reward")
        mt["dx_score"] = float(b["dx_score"])
        mt["dx_scorer"] = b["dx_scorer"]
        mt["dx_source"] = b["source"]
        mt["injection_flags"] = list(b.get("injection_flags") or [])
        mt["judge_latency_ms"] = b.get("latency_ms")
        mt["judge_cost_usd"] = b.get("cost_usd")
        return float(b["dx_score"])

    score_fn.panel = panel
    return score_fn


def score_fn_for(mode: str, panel=None):
    """`judge=strict|ontology|panel` -> the session score_fn."""
    if mode == "strict":
        return lexical_outcome_reward
    if mode == "ontology":
        return ontology_outcome_reward
    if mode == "panel":
        if panel is None:
            raise ValueError("judge=panel needs a judge.live.LivePanel")
        return panel_score_fn(panel)
    raise ValueError(f"unknown judge mode {mode!r}; use strict | ontology | panel")


def episode_won(metrics: dict, task_kind: str | None) -> bool:
    """Did the episode take the RIGHT action for its task (GiGPO finding 19)?
    `won = terminal_reward >= 0.5` marked every correct out-of-knowledge
    abstention (max 0.40) a failure. Outcome-aware: a correct diagnosis, a
    caught mismatch, or an honest abstention on an ook task.
    """
    kind = metrics.get("outcome_kind")
    if kind == "dx":
        v = metrics.get("dx_score")
        if v is None:
            v = metrics.get("outcome_reward")
        return float(v or 0.0) >= 1.0
    if kind in ("defer_mismatch", "defer_mismatch_claimed"):
        return float(metrics.get("terminal_reward") or 0.0) > 0.0
    if kind == "defer_ook":
        return task_kind == "ook"
    return False


def rewards_for_visit(base: dict | None, kw: dict) -> dict:
    """The terminal-reward matrix for ONE task-visit."""
    r = dict(base or {})
    if kw.get("r_ook") is not None:
        ook = float(kw["r_ook"])
        r["ook"] = ook
        r["blind_cap"] = min(float(r.get("blind_cap", gx.REWARDS["blind_cap"])), ook)
    return gx.validate_rewards(r, strict_cap=True)


class _QueueDoctor:
    """Stands in for LLMAgent inside the worker thread: hands each prompt to
    the driving process, blocks until step() supplies the raw action text.
    """

    def __init__(self, req_q: queue.Queue, act_q: queue.Queue, model: str):
        self.req_q, self.act_q, self.model = req_q, act_q, model

    def complete(self, messages, pmcid, phase, step=None):
        self.req_q.put(
            ("act", {"messages": messages, "phase": phase, "step": step, "pmcid": pmcid})
        )
        raw = self.act_q.get()
        if raw is _ABORT:
            raise _Abort()
        return raw


class TrustMedSession:
    """One episode as a reset()/step() session; rl_mode only (one call = one turn)."""

    def __init__(
        self,
        cfg: Config,
        row: dict,
        patient,
        swap_pool=(),
        score_fn: Callable[[dict, str], float] = lexical_outcome_reward,
        rewards: dict | None = None,
        model_name: str = "policy",
        task_kind: str = None,
    ):
        assert cfg.rl_mode, "TrustMedSession requires rl_mode (zero retries)"
        assert task_kind in ("answer", "corrupt", "ook"), (
            f"TrustMedSession needs task_kind in answer|corrupt|ook, got {task_kind!r}"
        )
        self.task_kind = task_kind
        self.cfg, self.row, self.patient = cfg, row, patient
        self.swap_pool = list(swap_pool)
        self.score_fn = score_fn
        self.rewards = gx.validate_rewards(dict(rewards or gx.REWARDS))
        self.model_name = model_name
        self._req_q: queue.Queue = queue.Queue()
        self._act_q: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._done = False

    def reset(self) -> tuple[dict, dict]:
        assert self._thread is None, "one session = one episode; make a new one"
        doctor = _QueueDoctor(self._req_q, self._act_q, self.model_name)

        def _run():
            try:
                rec = run_episode(
                    self.cfg,
                    self.row,
                    doctor,
                    self.patient,
                    self.swap_pool,
                    on_turn=lambda t: self._req_q.put(("turn", t)),
                )
                self._req_q.put(("done", rec))
            except _Abort:
                pass
            except Exception as e:
                self._req_q.put(("error", e))

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        kind, payload = self._next()
        assert kind == "act", f"episode ended before the first policy call: {kind}"
        return payload, {"pmcid": self.row.get("pmcid")}

    def price_terminal(self, rec: dict) -> float:
        """Price the terminal outcome with the task kind ON the record."""
        rec.setdefault("meta", {}).setdefault("task_kind", self.task_kind)
        return gx.terminal_reward(rec, self.rewards)

    def step(self, raw_action: str) -> tuple[dict | None, float, bool, dict]:
        assert self._thread and not self._done, "call reset() first"
        self._act_q.put(raw_action)
        kind, payload = self._next()
        assert kind == "turn", f"expected a turn after an action, got {kind}"
        turn = payload
        reward = float(turn.get("reward", 0.0))
        forfeit = reward < 0
        info: dict[str, Any] = {
            "anchor": turn.get("anchor"),
            "verb": turn.get("verb"),
            "phase": turn.get("phase"),
            "forfeit": forfeit,
        }
        if turn.get("error"):
            info["error"] = turn["error"]
        if forfeit and getattr(self.cfg, "local_penalty", False):
            reward = 0.0
        kind, payload = self._next()
        if kind == "act":
            return payload, reward, False, info
        assert kind == "done", kind
        rec = payload
        self._done = True
        gold = session_gold(self.row)
        rec["metrics"]["outcome_reward"] = self.score_fn(rec, gold)
        ok, sub = sr.classify_outcome(rec)
        rec["metrics"]["outcome_kind"] = ok
        if sub is not None:
            rec["metrics"]["outcome_subkind"] = sub
        term = self.price_terminal(rec)
        rec["metrics"]["reward_matrix"] = dict(self.rewards)
        rec["metrics"]["reward_contract"] = gx.contract_name(self.rewards)
        rec["metrics"]["terminal_reward"] = float(term)
        info.update(
            record=rec,
            outcome_kind=ok,
            terminal_reward=term,
            won=episode_won(rec["metrics"], self.task_kind),
            slim={
                "pmcid": rec["meta"]["pmcid"],
                "reward": round(reward + term, 4),
                "workup_turns": rec["metrics"]["workup_turns"],
                "searches": rec["metrics"]["actions_by_verb"].get("search", 0),
                "outcome_kind": ok,
                "dx_score": rec["metrics"].get("dx_score"),
                "dx_scorer": rec["metrics"].get("dx_scorer"),
                "dx_source": rec["metrics"].get("dx_source"),
                "fused_t_reward": rec["metrics"].get("fused_t_reward"),
                "n_judges": (rec["metrics"].get("panel") or {}).get("n_judges"),
                "judge_latency_ms": rec["metrics"].get("judge_latency_ms"),
                "flagged": bool(rec["metrics"].get("injection_flags")),
                "terminal_reward": float(term),
                "turns": len(rec.get("turns") or []),
                "invalid_turns": rec["metrics"].get("invalid_turns"),
                "normalized_turns": rec["metrics"].get("normalized_turns"),
                "rule_noops": rec["metrics"].get("rule_noops"),
                "format_forfeits": rec["metrics"].get("format_forfeits"),
                "coverage": round(gx.evidence_coverage(rec), 4),
                "corruption_armed": bool(rec["metrics"].get("corruption_armed")),
                "corruption_exposed": bool(rec["metrics"].get("corruption_exposed")),
                "gate_correct": rec["metrics"].get("gate_correct"),
                "gate_turns": rec["metrics"].get("gate_turns"),
                "delivery_batches": rec["metrics"].get("delivery_batches"),
                "pending_max": rec["metrics"].get("pending_max"),
                "verdict_source": (rec["metrics"].get("panel") or {}).get("verdict_source"),
                "judge_cost_usd": rec["metrics"].get("judge_cost_usd"),
            },
        )
        return None, reward + term, True, info

    def close(self) -> None:
        if self._thread and self._thread.is_alive():
            self._act_q.put(_ABORT)
            self._thread.join(timeout=5)
        self._thread, self._done = None, True

    def _next(self):
        kind, payload = self._req_q.get()
        if kind == "error":
            self._done = True
            raise payload
        return kind, payload
