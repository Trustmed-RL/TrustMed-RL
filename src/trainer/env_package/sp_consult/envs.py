"""sp_consult env package for verl-agent — search-env pattern, thread-backed."""

from __future__ import annotations

import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

LH_ROOT = Path(os.environ.get("SP_CONSULT_HOME") or Path(__file__).resolve().parents[3])
_LH = LH_ROOT
sys.path.insert(0, str(_LH))

import environment as m
import session as g
import judge.live as lp
import rewards as gx


def pick_donor_list(is_train: bool, donor_list, donor_list_val):
    """The donor file THIS env binds corruption swaps from:
    a val env with a held-out val donor file uses it; everything else -- train
    envs, and val envs of a pack without the split -- uses the shared file.
    """
    if not is_train and donor_list_val is not None:
        return donor_list_val
    return donor_list


class SpConsultMultiThreadEnv:
    """One object owning env_num*group_n live sessions."""

    def __init__(self, seed: int, env_num: int, group_n: int, is_train: bool, env_config):
        self.seed, self.is_train = seed, is_train
        self.env_num, self.group_n = env_num, group_n
        self.n = env_num * group_n
        ec = env_config
        self.profiles_path = Path(ec.profiles)
        self.rows = {r["pmcid"]: r for r in m.load_profiles(self.profiles_path)}
        self.out_root = Path(ec.out_dir)
        self.records_out = Path(ec.records_out)
        self.search_url = getattr(ec, "search_url", None) or None
        self.gate_bank_dir = Path(ec.gate_bank_dir) if getattr(ec, "gate_bank_dir", None) else None
        self.images_dir = Path(ec.images_dir) if getattr(ec, "images_dir", None) else None
        self.post_baseline = bool(getattr(ec, "post_baseline", False))
        self.identity_anchor = bool(getattr(ec, "identity_anchor", False))
        self.anchor_mode = getattr(ec, "anchor_mode", None) or None
        self.panel_assets = Path(ec.panel_assets) if getattr(ec, "panel_assets", None) else None
        self.panel_assets_dir = (
            Path(ec.panel_assets_dir) if getattr(ec, "panel_assets_dir", None) else None
        )
        self.donor_list = pick_donor_list(
            is_train,
            Path(ec.donor_list) if getattr(ec, "donor_list", None) else None,
            Path(ec.donor_list_val) if getattr(ec, "donor_list_val", None) else None,
        )
        self.donor_min_per_category = int(getattr(ec, "donor_min_per_category", 10) or 0)
        self.run_generation = int(getattr(ec, "run_generation", 0) or 0)
        self.panel_policy = getattr(ec, "panel_policy", None)
        self.patient_model = ec.patient_model
        self.patient_base_url = getattr(ec, "patient_base_url", None)
        self.invalid_penalty = float(getattr(ec, "invalid_penalty", -0.01) or -0.01)
        self.max_items_per_action = int(getattr(ec, "max_items_per_action", 0) or 0) or None
        self.deliver_mode = str(getattr(ec, "deliver_mode", "") or "") or None
        self.deliver_chunk = int(getattr(ec, "deliver_chunk", 0) or 0) or None
        self.max_batches_per_order = int(getattr(ec, "max_batches_per_order", 0) or 0) or None
        self.local_penalty = str(getattr(ec, "local_penalty", "0")).strip().lower() in (
            "1",
            "true",
            "yes",
        )
        self.conclude_mode = str(getattr(ec, "conclude_mode", "") or "") or None
        self.rule_violation = str(getattr(ec, "rule_violation", "") or "") or None
        self.patient_temperature = float(getattr(ec, "patient_temperature", 0.0) or 0.0)
        self._attempts: dict = {}
        self.rewards = dict(getattr(ec, "rewards", None) or {})
        for key in ("coverage_floor", "coverage_power"):
            v = getattr(ec, key, None)
            if v is not None:
                self.rewards[key] = float(v)
        bcm = getattr(ec, "blind_cap_mode", None)
        if bcm:
            self.rewards["blind_cap_mode"] = str(bcm)
        legacy = getattr(ec, "false_exit", None)
        for key in (
            "false_mismatch",
            "false_ook",
            "zero_workup_wrong",
            "format_failure",
            "waved_dx",
        ):
            v = getattr(ec, key, None)
            if v is None and key in ("false_mismatch", "false_ook"):
                v = legacy
            if v is not None:
                self.rewards[key] = float(v)
        g.rewards_for_visit(self.rewards, {})
        self.judge_mode = str(getattr(ec, "judge", "panel") or "panel")
        self.ingest_records = bool(is_train)
        if not is_train:
            self.records_out = self.records_out.with_name(
                self.records_out.stem + "_val" + self.records_out.suffix
            )
        self.panel = None
        if self.judge_mode == "panel":
            cache = Path(
                getattr(ec, "judge_cache", None) or self.records_out.parent / "panel_cache.jsonl"
            )
            raw = Path(
                getattr(ec, "judge_raw_out", None) or self.records_out.parent / "panel_raw.jsonl"
            )
            if not is_train:
                cache = cache.with_name(cache.stem + "_val" + cache.suffix)
                raw = raw.with_name(raw.stem + "_val" + raw.suffix)
            judges_cfg = getattr(ec, "judges", None)
            if isinstance(judges_cfg, str):
                judges = [j.strip() for j in judges_cfg.split(",") if j.strip()]
            elif judges_cfg:
                judges = [str(j) for j in judges_cfg]
            else:
                judges = list(lp.DEFAULT_JUDGES)
            pkw = dict(
                judges=judges,
                cache_path=cache,
                raw_out=raw,
                concurrency=int(getattr(ec, "judge_concurrency", 48) or 48),
                fallback_halt_share=float(getattr(ec, "judge_fallback_halt_share", 0.0) or 0.0),
                flag_halt_share=float(getattr(ec, "judge_flag_halt_share", 0.02) or 0.02),
                judge_parse_retries=int(getattr(ec, "judge_parse_retries", 1) or 1),
            )
            factory = getattr(ec, "panel_factory", None)
            self.panel = factory(**pkw) if factory else lp.LivePanel(**pkw)
        elif self.judge_mode not in ("ontology", "strict"):
            raise ValueError(
                f"env.sp_consult.judge must be panel | ontology | strict, got {self.judge_mode!r}"
            )
        self.score_fn = g.score_fn_for(self.judge_mode, self.panel)
        self.pool = ThreadPoolExecutor(max_workers=self.n)
        self.session_out = self.out_root / "shared"
        first = next(iter(self.rows))
        base_cfg = self.session_cfg(
            0,
            {
                "pmcid": first,
                "task_kind": "answer",
                "max_asks": 4,
                "max_workup_turns": 6,
                "max_searches": 0,
                "search_enabled": False,
            },
        )
        self.swap_pool = m.build_swap_pool(list(self.rows.values()), base_cfg)
        self._donor_census(base_cfg)
        self._patient_factory = self._make_patient
        self.records_full = self.out_root / (
            "trajectories.jsonl" if is_train else "trajectories_val.jsonl"
        )
        self.panel_steps = self.out_root / (
            "panel_steps.jsonl" if is_train else "panel_steps_val.jsonl"
        )
        self.sessions: list[g.SpConsultSession | None] = [None] * self.n
        self.meta: list[dict] = [{} for _ in range(self.n)]
        self.acc_reward = np.zeros(self.n, dtype=np.float64)
        self.aborted: dict[int, str] = {}
        self._rec_lock = threading.Lock()

    def _donor_census(self, base_cfg) -> None:
        """Count the servable donors per category at boot and refuse to train when a main category has none (the image corruption channel is the only one): a pack without a donor file arms corruptions that never bind. Written to episodes/swap_pool.json either way."""
        counts: dict[str, int] = {}
        for _, path, cat in self.swap_pool:
            if Path(path).is_file():
                counts[cat] = counts.get(cat, 0) + 1
        census = {
            "donor_list": str(self.donor_list) if self.donor_list else None,
            "pool_rows": len(self.swap_pool),
            "servable_by_category": counts,
            "min_per_category": self.donor_min_per_category,
            "is_train": self.is_train,
        }
        try:
            self.out_root.mkdir(parents=True, exist_ok=True)
            (
                self.out_root / ("swap_pool.json" if self.is_train else "swap_pool_val.json")
            ).write_text(json.dumps(census, indent=1), encoding="utf-8")
        except OSError:
            pass
        if self.donor_min_per_category > 0:
            short = [
                c
                for c in ("radiology", "pathology", "clinical_photo")
                if counts.get(c, 0) < self.donor_min_per_category
            ]
            if short:
                raise RuntimeError(
                    f"corruption donor pool unusable: servable donors {counts} -- "
                    f"categories under {self.donor_min_per_category}: {short}. Ship the "
                    f"donor crops (tools/build_pack.py -> donors_v2.jsonl + DONOR_LIST) or "
                    f"set +env.sp_consult.donor_min_per_category=0 to run without the "
                    f"image corruption channel being trainable."
                )

    def session_cfg(self, i: int, kw: dict):
        """The Config one session runs under: the visit's env_kwargs on the run-family constants, one shared out/state dir for every session."""
        extra = {}
        if self.panel_assets is not None:
            extra["panel_assets"] = self.panel_assets
        if self.panel_assets_dir is not None:
            extra["panel_assets_dir"] = self.panel_assets_dir
        if self.donor_list is not None:
            extra["donor_list"] = self.donor_list
        extra["invalid_penalty"] = self.invalid_penalty
        if self.anchor_mode:
            extra["anchor_mode"] = self.anchor_mode
        if self.max_items_per_action:
            extra["max_items_per_action"] = self.max_items_per_action
        if self.deliver_mode:
            extra["deliver_mode"] = self.deliver_mode
        if self.deliver_chunk:
            extra["deliver_chunk"] = self.deliver_chunk
        if self.max_batches_per_order:
            extra["max_batches_per_order"] = self.max_batches_per_order
        if self.local_penalty:
            extra["local_penalty"] = True
        if getattr(self, "conclude_mode", None):
            extra["conclude_mode"] = self.conclude_mode
        if getattr(self, "rule_violation", None):
            extra["rule_violation"] = self.rule_violation
        return g.kwargs_to_config(
            kw,
            profiles=self.profiles_path,
            out=self.session_out,
            search_url=self.search_url,
            gate_bank_dir=self.gate_bank_dir,
            images_dir=self.images_dir,
            post_baseline=self.post_baseline,
            identity_anchor=self.identity_anchor,
            panel_policy=self.panel_policy,
            patient_model=self.patient_model,
            patient_base_url=self.patient_base_url,
            **extra,
        )

    def _make_patient(self, cfg):
        return m.LLMAgent(
            cfg,
            "patient",
            cfg.patient_model,
            cfg.patient_base_url,
            cfg.patient_temperature,
            lambda _u: None,
            json_mode=True,
        )

    def prewarm_openings(self, kwargs: list[dict]) -> int:
        """Fill the frozen-opening cache once per unique (pmcid, corrupt_seed) before any session boots, mirroring the standalone collector's serial pre-warm (sp_consult.main). Booting the group cold is a first-writer race: nine rollouts find the cache empty, each asks the patient for an opening, and the anchor state s1 they were meant to share never forms."""
        uniq: dict[tuple[str, str], dict] = {}
        for kw in kwargs:
            uniq.setdefault((kw["pmcid"], str(kw.get("corrupt_seed", "0"))), kw)

        def one(item):
            (pmcid, _seed), kw = item
            cfg = self.session_cfg(0, kw)
            stats = {
                k: 0
                for k in (
                    "doctor_retries",
                    "doctor_fallbacks",
                    "patient_retries",
                    "patient_fallbacks",
                )
            }
            ep = m.parse_profile(self.rows[pmcid], cfg, self.swap_pool)
            m.load_or_make_opening(cfg, ep, self._patient_factory(cfg), [], stats)
            return 1

        return sum(self.pool.map(one, uniq.items()))

    def write_record(self, rec: dict) -> None:
        self.records_full.parent.mkdir(parents=True, exist_ok=True)
        with self._rec_lock:
            with self.records_full.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def reset(self, kwargs):
        """kwargs: list of env_kwargs dicts, ONE PER SESSION (the manager has
        already repeated each task row group_n times, collector-side).
        """
        if self.panel is not None:
            closed = self.panel.begin_step()
            if closed.get("scored"):
                self.panel_steps.parent.mkdir(parents=True, exist_ok=True)
                with self._rec_lock, self.panel_steps.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({**closed, "spent_usd": self.panel.spent_usd}) + "\n")
            if self.panel.halt_reason:
                raise lp.PanelHalt(self.panel.halt_reason)
        kwargs = list(kwargs) if kwargs is not None else []
        kwargs = list(kwargs) if kwargs is not None else []
        assert len(kwargs) == self.n, (len(kwargs), self.n)
        _step0 = int((kwargs[0] or {}).get("global_step", -1)) if kwargs else -1
        attempt = self.attempt_for(_step0) if self.is_train else 0
        self.prewarm_openings(kwargs)
        for s in self.sessions:
            if s is not None:
                s.close()
        self.acc_reward[:] = 0.0
        self.aborted.clear()

        def _boot(i):
            kw = kwargs[i]
            self.meta[i] = {
                "pmcid": kw["pmcid"],
                "task_kind": kw["task_kind"],
                "global_step": kw.get("global_step", -1),
                "difficulty": str(kw.get("difficulty") or ""),
                "attempt": attempt,
                "run_generation": self.run_generation,
            }
            cfg = self.session_cfg(i, kw)
            patient = self._patient_factory(cfg)
            sess = g.SpConsultSession(
                cfg,
                self.rows[kw["pmcid"]],
                patient,
                swap_pool=self.swap_pool,
                score_fn=self.score_fn,
                rewards=self.visit_rewards(kw),
                task_kind=kw.get("task_kind"),
            )
            self.sessions[i] = sess
            req, _ = sess.reset()
            return self._obs(req)

        obs = list(self.pool.map(_boot, range(self.n)))
        infos = [{"is_action_valid": True} for _ in range(self.n)]
        return obs, infos

    def step(self, actions):
        assert len(actions) == self.n

        def _one(i):
            sess = self.sessions[i]
            if sess is None or sess._done:
                return (
                    self._obs(None),
                    0.0,
                    True,
                    {
                        "is_action_valid": True,
                        "won": False,
                        "anchor_obs": f"done:{i}",
                        "verb": "done",
                    },
                )
            req, reward, done, info = sess.step(actions[i])
            self.acc_reward[i] += reward
            out_info = {
                "is_action_valid": not bool(info.get("forfeit")),
                "anchor_obs": info.get("anchor") or f"noanchor:{i}",
                "verb": info.get("verb"),
            }
            if done:
                rec = info["record"]
                rec.setdefault("meta", {}).update(
                    global_step=self.meta[i].get("global_step"),
                    task_kind=self.meta[i].get("task_kind"),
                    difficulty=self.meta[i].get("difficulty", ""),
                    attempt=self.meta[i].get("attempt", 0),
                    run_generation=self.meta[i].get("run_generation", 0),
                    rollout_idx=i % self.group_n,
                )
                self.write_record(rec)
                _mt = rec.get("metrics") or {}
                _dx_ok = (
                    str(_mt.get("outcome") or "") == "final_diagnosis"
                    and float(_mt.get("dx_score") or 0.0) > 0.0
                )
                try:
                    _br = gx.dx_credit_branch(rec, gx.matrix_for_record(rec))
                except Exception as _e:
                    _br = f"error:{type(_e).__name__}"
                _dx_boost = (
                    _br == "full"
                    and float(_mt.get("outcome_reward") or 0.0) >= 1.0
                    and str(self.meta[i].get("task_kind") or "") != "ook"
                )
                out_info.update(
                    won=bool(info["won"]),
                    dx_correct=bool(_dx_ok),
                    dx_boost=bool(_dx_boost),
                    dx_credit_branch=_br,
                    episode_record=rec,
                )
                slim = dict(
                    info["slim"],
                    **self.meta[i],
                    reward=round(
                        float(rec["metrics"].get("terminal_reward", 0.0) or 0.0)
                        + float(rec["metrics"].get("step_reward_sum", 0.0) or 0.0),
                        4,
                    ),
                    trainer_return=round(float(self.acc_reward[i]), 4),
                    rollout_idx=i % self.group_n,
                    episode_group_uid=gx.episode_group_uid(rec),
                )
                if self.ingest_records:
                    with self._rec_lock:
                        with self.records_out.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(slim) + "\n")
            return self._obs(req), float(reward), bool(done), out_info

        results = list(self.pool.map(_one, range(self.n)))
        obs = [r[0] for r in results]
        rewards = np.array([r[1] for r in results], dtype=np.float32)
        dones = np.array([r[2] for r in results], dtype=np.bool_)
        infos = [r[3] for r in results]
        return obs, rewards, dones, infos

    def attempt_for(self, step: int) -> int:
        """0-based retry index for one global_step: the k-th (re)generation of
        this step's batch (filter_groups regen). Records stamp it so export and
        the group uid tell retries apart (G4).
        """
        c = self._attempts.get(step, 0)
        self._attempts[step] = c + 1
        return c

    def success_evaluator(
        self,
        total_infos=None,
        total_batch_list=None,
        episode_rewards=None,
        episode_lengths=None,
        **kw,
    ):
        """info['won'] of each env's LAST step. The fork's collector passes
        `total_infos` ENV-MAJOR (total_infos[env] = [per-step infos], built by
        `total_infos[i].append(infos[i])`); the stock manager shape is
        TIME-major (total_infos[step] = [per-env infos]). Accept both (G1):
        env-major iff the outer length is self.n.
        """
        won = np.zeros(self.n, dtype=np.float64)
        if not total_infos:
            return {"success_rate": won}
        env_major = len(total_infos) == self.n
        for i in range(self.n):
            if env_major:
                seq = total_infos[i]
            else:
                seq = [step[i] for step in total_infos if i < len(step)]
            last = [inf for inf in seq if isinstance(inf, dict) and "won" in inf]
            won[i] = float(last[-1]["won"]) if last else 0.0
        return {"success_rate": won}

    def abort_one(self, i: int, reason: str = "prompt_overflow") -> dict:
        """End episode i now, without an action. The collector calls this when it could not build the env's prompt (length over data.max_prompt_length under truncation=error, e.g. a gate turn with six composite plates): the session thread gets _ABORT (SpConsultSession.close), no record is written (no terminal happened; the curriculum must not read the visit as evidence) and every later step() pads the env as done. Returns the identity the collector logs."""
        sess = self.sessions[i]
        if sess is not None and not sess._done:
            sess.close()
        self.aborted[int(i)] = str(reason)
        m = self.meta[i] if i < len(self.meta) else {}
        return {
            "reason": str(reason),
            "env": int(i),
            **{k: m.get(k) for k in ("pmcid", "task_kind", "global_step", "attempt", "difficulty")},
        }

    def close(self):
        for s in self.sessions:
            if s is not None:
                s.close()
        self.pool.shutdown(wait=False)

    def visit_rewards(self, kw: dict) -> dict:
        """The terminal matrix for one task-visit: the run-family base
        (coverage_floor, overrides) + the row's scheduled r_ook - so the LIVE
        episode is priced by the same ramp the curriculum header reports.
        """
        return g.rewards_for_visit(self.rewards, kw)

    @staticmethod
    def _obs(req):
        if req is None:
            return {"text": "", "multi_modal": None}
        msgs = req["messages"][-1]["content"]
        if isinstance(msgs, str):
            return {"text": msgs, "multi_modal": None}
        text = "".join(p["text"] for p in msgs if p.get("type") == "text")
        imgs = [p for p in msgs if p.get("type") == "image_url"]
        return {"text": text, "multi_modal": imgs or None}


def won_for(term: float, outcome_kind: str, task_kind: str | None) -> bool:
    """Did the episode take the RIGHT action for its task (fork success signal)?
    Outcome-aware (G7): a correct dx or a caught mismatch (terminal > 0) wins; an
    out-of-knowledge abstention wins ONLY on an ook task; everything else loses.
    Takes the terminal scalar (not dx_score) because info['won'] is computed from
    the terminal the collector already has.
    """
    t = float(term or 0.0)
    if outcome_kind == "dx":
        return t > 0.0
    if outcome_kind in ("defer_mismatch", "defer_mismatch_claimed"):
        return t > 0.0
    if outcome_kind == "defer_ook":
        return task_kind == "ook"
    return False


def build_sp_consult_envs(seed: int, env_num: int, group_n: int, is_train: bool, env_config):
    """Factory the fork's `make_envs` branch calls."""
    return SpConsultMultiThreadEnv(seed, env_num, group_n, is_train, env_config)
