#!/usr/bin/env python3
"""Apply the TrustMed environment patches to a verl-agent checkout (langfengQ/verl-agent, commit 20bd331)."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
LH = HERE.parent


def _sub(path: Path, find: str, repl: str, marker: str) -> str:
    txt = path.read_text(encoding="utf-8")
    if marker in txt:
        return f"  = already patched: {path.name}"
    if find not in txt:
        raise SystemExit(f"  ! anchor NOT FOUND in {path} — fork rev drift?\n    {find[:80]!r}")
    path.write_text(txt.replace(find, repl, 1), encoding="utf-8")
    return f"  + patched: {path.name}"


def copy_modules(fork: Path) -> list[str]:
    out = []
    dst = fork / "agent_system/environments/env_package/trustmed"
    dst.mkdir(parents=True, exist_ok=True)
    for f in (HERE / "env_package/trustmed").glob("*.py"):
        shutil.copy2(f, dst / f.name)
    out.append(f"  + copied env_package/trustmed/*.py -> {dst}")
    cur = fork / "agent_system/curriculum"
    cur.mkdir(parents=True, exist_ok=True)
    (cur / "__init__.py").write_text("", encoding="utf-8")
    for name in ("curriculum_sampler.py", "schedulers.py", "val_rows.py", "plan_store.py"):
        shutil.copy2(HERE / name, cur / name)
    out.append(
        f"  + copied curriculum_sampler.py, schedulers.py, val_rows.py, plan_store.py -> {cur}"
    )
    return out


ENV_MGR_FIND = '    elif "gym_cards" in config.env.env_name.lower():'
ENV_MGR_REPL = """    elif "trustmed" in config.env.env_name.lower():
        from agent_system.environments.env_package.trustmed import (
            build_trustmed_envs, trustmed_projection, TrustMedEnvironmentManager)
        _envs = build_trustmed_envs(seed=config.env.seed, env_num=config.data.train_batch_size,
                                      group_n=group_n, is_train=True, env_config=config.env.trustmed)
        _val_envs = build_trustmed_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size,
                                          group_n=1, is_train=False, env_config=config.env.trustmed)
        projection_f = partial(trustmed_projection)
        envs = TrustMedEnvironmentManager(_envs, projection_f, config)
        val_envs = TrustMedEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "gym_cards" in config.env.env_name.lower():"""

POP_FIND = "obs, infos = envs.reset(kwargs=gen_batch.non_tensor_batch.pop('env_kwargs', None))"
POP_REPL = (
    "obs, infos = envs.reset(kwargs=gen_batch.non_tensor_batch.get('env_kwargs', None))"
    "  # trustmed:filter-groups-get: get-not-pop (filter_groups retries reuse the batch)"
)

MMIMG_FIND = "            row_dict['multi_modal_data'] = {'image': [process_image(obs_image)]}"
MMIMG_REPL = (
    "            _imgs = obs_image if isinstance(obs_image, (list, tuple)) else [obs_image]\n"
    "            # a PIL image (the trustmed manager decodes data-URIs to PIL) passes\n"
    "            # through; process_image handles array/tensor obs from other envs\n"
    "            row_dict['multi_modal_data'] = {'image': [\n"
    "                _im if hasattr(_im, 'convert') else process_image(_im) for _im in _imgs]}"
)

COLL_FIND = """        else:
            position_ids = compute_position_id_with_mask(attention_mask)"""
COLL_REPL = """        elif getattr(self, 'processor', None) is not None and hasattr(self.processor, 'image_processor'):
            # trustmed:text-row-mm-shape: a text-only row in a MULTIMODAL run must match the image
            # rows' (1, 4, seq) position-id rank, else a mixed image/text batch
            # fails to collate. mrope dims == the text positions; empty image list.
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [text_position_ids.repeat(4, 1)]  # (1, 4, seq)
            row_dict.setdefault('multi_modal_data', {'image': []})
            row_dict['multi_modal_inputs'] = {}  # trustmed:text-row-empty-mm: keep the batch key non-ragged
            row_dict.setdefault('multi_modal_inputs', {})  # every row carries the key or collate desyncs
        else:
            position_ids = compute_position_id_with_mask(attention_mask)"""


DPA_FIND = """        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))"""
DPA_REPL = """        if use_dynamic_bsz and not has_multi_modal_inputs:  # trustmed:mm-chunk-indices: mm chunks never built indices
            indices = list(itertools.chain.from_iterable(indices))"""

RT_FIND = "                logger.log(data=metrics, step=self.global_steps)"
RT_REPL = """                logger.log(data=metrics, step=self.global_steps)
                try:  # trustmed:metrics-sidecar: metrics sidecar for offline curves
                    import json as _json, os as _os
                    _d = self.config.trainer.default_local_dir
                    _os.makedirs(_d, exist_ok=True)
                    with open(_os.path.join(_d, "metrics.jsonl"), "a") as _f:
                        _f.write(_json.dumps({"step": self.global_steps,
                            **{k: (float(v) if hasattr(v, "__float__") else str(v))
                               for k, v in metrics.items()}}) + chr(10))
                except Exception:
                    pass"""

RT10_FIND = """                            gigpo_enable_similarity= self.config.algorithm.gigpo.enable_similarity,
                            gigpo_similarity_thresh=self.config.algorithm.gigpo.similarity_thresh,
                        )"""
RT10_REPL = """                            gigpo_enable_similarity= self.config.algorithm.gigpo.enable_similarity,
                            gigpo_similarity_thresh=self.config.algorithm.gigpo.similarity_thresh,
                        )
                        try:  # trustmed:step-group-telemetry: advantages + config sidecars
                            import json as _json, os as _os
                            _d = self.config.trainer.default_local_dir
                            _os.makedirs(_d, exist_ok=True)
                            if self.global_steps <= 1:
                                from omegaconf import OmegaConf as _OC
                                with open(_os.path.join(_d, "config.json"), "w") as _f:
                                    _json.dump(_OC.to_container(self.config, resolve=True), _f, default=str)
                            _adv = batch.batch["advantages"]
                            _rm = batch.batch["response_mask"].bool()
                            _tlr = batch.batch["token_level_rewards"]
                            _uid = batch.non_tensor_batch.get("uid")
                            _tuid = batch.non_tensor_batch.get("traj_uid")
                            with open(_os.path.join(_d, "advantages.jsonl"), "a") as _f:
                                for _i in range(_adv.shape[0]):
                                    _m = _rm[_i]
                                    _n = int(_m.sum())
                                    _f.write(_json.dumps({
                                        "step": self.global_steps,
                                        "uid": str(_uid[_i]) if _uid is not None else None,
                                        "traj_uid": str(_tuid[_i]) if _tuid is not None else None,
                                        "adv_mean": (float(_adv[_i][_m].mean()) if _n else 0.0),
                                        "reward_sum": float(_tlr[_i].sum()),
                                        "resp_len": _n}) + chr(10))
                        except Exception:
                            pass"""

EPI_FIND = """            if multi_modal_inputs is not None:
                pixel_values = multi_modal_inputs['pixel_values']
                image_grid_thw = multi_modal_inputs['image_grid_thw']"""
EPI_REPL = """            if multi_modal_inputs:  # trustmed:text-row-empty-values: text rows carry {} (values unused below)
                pixel_values = multi_modal_inputs.get('pixel_values')
                image_grid_thw = multi_modal_inputs.get('image_grid_thw')"""

UTILS_FIND = """    if isinstance(image, torch.Tensor):
        image = torch_to_numpy(image)
    if image.max() < 1:
        image = image * 255.0
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    image = Image.fromarray(image)"""
UTILS_REPL = """    if isinstance(image, Image.Image):
        pass  # trustmed:pil-passthrough: PIL passthrough (trustmed serves PIL crops)
    else:
        if isinstance(image, torch.Tensor):
            image = torch_to_numpy(image)
        if image.max() < 1:
            image = image * 255.0
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        image = Image.fromarray(image)"""

MAINPPO2_FIND = """        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)"""
MAINPPO2_REPL = """        if "trustmed" in config.env.env_name.lower():
            # trustmed:curriculum-plan-dataset: the curriculum plan is the ONLY
            # dataset for trustmed -- never touch data.train_files (gsm8k
            # default), keep the fork sampler out of this branch, give
            # validation a STATIC dataset, and resync the plan on resume.
            import os as _os, sys as _sys
            _lh = _os.environ.get("TRUSTMED_HOME")
            if _lh and _lh not in _sys.path:
                _sys.path.insert(0, _lh)
            from agent_system.curriculum.curriculum_sampler import (
                CurriculumPlan, TrustMedRLDataset, TrustMedValDataset,
                CurriculumSequentialSampler)
            _plan = CurriculumPlan(pool_path=config.curriculum.pool,
                                   schedule_path=config.curriculum.schedule,
                                   records_path=config.env.trustmed.records_out,
                                   state_dir=config.curriculum.state_dir,
                                   seed=config.curriculum.seed,
                                   total_steps=config.trainer.total_training_steps)
            # the sampler resumes exactly where the TRAINER resumes (resume_from_path or the
            # newest local checkpoint -- ray_trainer._load_checkpoint), reloading persisted plans
            from trainer.plan_store import resume_step_from_config
            _resumed = resume_step_from_config(str(config.trainer.resume_mode),
                                               config.trainer.get("resume_from_path", None),
                                               str(config.trainer.default_local_dir))
            if _resumed > 0:
                _plan.resync_to(_resumed)
            train_dataset = TrustMedRLDataset(_plan, tokenizer=tokenizer,
                                               max_prompt_length=config.data.max_prompt_length)
            train_sampler = CurriculumSequentialSampler(_plan, start_row=_resumed * _plan.per_step)
            # validation comes from its OWN pool (PMCID-disjoint from the train
            # pool, true task kinds); the old first-n-rows-of-the-train-pool validator is
            # only reachable with curriculum.allow_val_overlap=true (smokes)
            _val_pool = config.curriculum.get("val_pool", None)
            if not _val_pool:
                if not bool(config.curriculum.get("allow_val_overlap", False)):
                    raise ValueError("curriculum.val_pool is required (a PMCID-disjoint validation pool); "
                                     "set curriculum.allow_val_overlap=true only for a smoke that has none")
                _val_pool = config.curriculum.pool
                print("WARNING: validation rows come from the TRAIN pool (allow_val_overlap)")
            else:
                from trainer.val_rows import assert_disjoint
                assert_disjoint(_val_pool, config.curriculum.pool)
            val_dataset = TrustMedValDataset(_val_pool, config.curriculum.schedule,
                                              n=int(config.data.val_batch_size or 8),
                                              tokenizer=tokenizer,
                                              max_prompt_length=config.data.max_prompt_length)
        else:
            train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
            val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
            train_sampler = create_rl_sampler(config.data, train_dataset)"""

MAINPPO_FIND = "            train_sampler = create_rl_sampler(config.data, train_dataset)"
MAINPPO_REPL = """        train_sampler = create_rl_sampler(config.data, train_dataset)
        if "trustmed" in config.env.env_name.lower():
            # trustmed:curriculum-plan-schedule: the curriculum plan owns the visit schedule; inject a
            # plan-backed dataset + sequential sampler (create_rl_dataset cannot
            # see config.curriculum). val reuses the plan (a smoke never
            # validates: TEST_FREQ high, val_before_train False).
            import os as _os, sys as _sys
            _lh = _os.environ.get("TRUSTMED_HOME")
            if _lh and _lh not in _sys.path:
                _sys.path.insert(0, _lh)
            from agent_system.curriculum.curriculum_sampler import (
                CurriculumPlan, TrustMedRLDataset, CurriculumSequentialSampler)
            _plan = CurriculumPlan(pool_path=config.curriculum.pool,
                                   schedule_path=config.curriculum.schedule,
                                   records_path=config.env.trustmed.records_out,
                                   state_dir=config.curriculum.state_dir,
                                   seed=config.curriculum.seed,
                                   total_steps=config.trainer.total_training_steps)
            train_dataset = TrustMedRLDataset(_plan, tokenizer=tokenizer,
                                               max_prompt_length=config.data.max_prompt_length)
            train_sampler = CurriculumSequentialSampler(_plan, start_row=0)
            val_dataset = train_dataset"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fork", required=True, type=Path, help="verl-agent clone root")
    a = ap.parse_args()
    fork = a.fork.resolve()
    if not (fork / "verl/trainer/main_ppo.py").exists():
        raise SystemExit(f"not a verl-agent clone: {fork}")

    print("== copy modules ==")
    for line in copy_modules(fork):
        print(line)

    print("== edit 1: env_manager trustmed branch ==")
    print(
        _sub(
            fork / "agent_system/environments/env_manager.py",
            ENV_MGR_FIND,
            ENV_MGR_REPL,
            'elif "trustmed" in config.env.env_name.lower():',
        )
    )

    rl = fork / "agent_system/multi_turn_rollout/rollout_loop.py"
    print("== edit 3: env_kwargs get-not-pop ==")
    print(_sub(rl, POP_FIND, POP_REPL, "trustmed:filter-groups-get: get-not-pop"))
    print("== edit 6a: multi-image ==")
    print(_sub(rl, MMIMG_FIND, MMIMG_REPL, "for _im in _imgs"))
    print("== edit 6b: mixed-row collation ==")
    print(_sub(rl, COLL_FIND, COLL_REPL, "trustmed:text-row-mm-shape: a text-only row"))

    print("== edit 9: dp_actor dynamic-bsz mm guard ==")
    dpa = fork / "verl/workers/actor/dp_actor.py"
    t = dpa.read_text(encoding="utf-8")
    if "trustmed:mm-chunk-indices" not in t:
        n = t.count(DPA_FIND)
        assert n >= 1, "dp_actor tail not found"
        dpa.write_text(t.replace(DPA_FIND, DPA_REPL), encoding="utf-8")
        print(f"  + patched: dp_actor.py ({n} sites)")
    else:
        print("  = already patched: dp_actor.py")
    print("== edit 8: metrics.jsonl sidecar ==")
    print(
        _sub(fork / "verl/trainer/ppo/ray_trainer.py", RT_FIND, RT_REPL, "trustmed:metrics-sidecar")
    )
    print("== edit 10: advantages + config sidecars ==")
    print(
        _sub(
            fork / "verl/trainer/ppo/ray_trainer.py",
            RT10_FIND,
            RT10_REPL,
            "trustmed:step-group-telemetry",
        )
    )
    print("== edit 7: reward manager text-row guard ==")
    print(
        _sub(
            fork / "agent_system/reward_manager/episode.py",
            EPI_FIND,
            EPI_REPL,
            "trustmed:text-row-empty-values",
        )
    )
    print("== edit 6c: process_image PIL passthrough ==")
    print(
        _sub(
            fork / "agent_system/multi_turn_rollout/utils.py",
            UTILS_FIND,
            UTILS_REPL,
            "trustmed:pil-passthrough",
        )
    )
    print("== edit 5a: skip default dataset for trustmed ==")
    print(
        _sub(
            fork / "verl/trainer/main_ppo.py",
            MAINPPO2_FIND,
            MAINPPO2_REPL,
            "trustmed:curriculum-plan-dataset",
        )
    )

    print("== done. PATCHES applied to", fork, "==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
