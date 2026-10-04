#!/usr/bin/env python3
"""Trainer-side patches for verl-agent (langfengQ/verl-agent, commit 20bd331); apply after fork_patches.py."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

NL = chr(10)

STATE_OF_DECISION_ANCHOR = "            if 'tool_calling' in infos[0]:"
STATE_OF_DECISION_INSERT = (
    "            # trustmed:state-of-decision: row t's action was decided in the state the env\n"
    "            # stamped in THIS step's info; the pre-step obs carried the\n"
    "            # PREVIOUS step's anchor (off-by-one that poisons A^S groups).\n"
    "            batch.non_tensor_batch['anchor_obs'] = np.array(\n"
    "                [((info.get('anchor_obs') if isinstance(info, dict) else None)\n"
    "                  or ('noanchor:%d' % _pi))\n"
    "                 for _pi, info in enumerate(infos)], dtype=object)\n\n"
)

TRAJECTORY_BASELINE_OLD = "                        compute_mean_std_cross_steps: bool = True,"
TRAJECTORY_BASELINE_NEW = (
    "                        compute_mean_std_cross_steps: bool = False,"
    "  # trustmed:trajectory-baseline: per-trajectory A^E baseline (paper Eq.3); True"
    " length-weights R_i by trajectory length"
)

STEP_GROUP_TELEMETRY_ANCHOR = (
    "    scores = episode_advantages + step_advantage_w * step_advantages\n"
    "    return scores, scores"
)
STEP_GROUP_TELEMETRY_NEW = (
    "    scores = episode_advantages + step_advantage_w * step_advantages\n"
    "    try:  # trustmed:step-group-telemetry: per-step group telemetry for collapse/hacking forensics\n"
    "        import os as _os, json as _json\n"
    '        _path = _os.environ.get("GIGPO_TELEMETRY")\n'
    "        if _path:\n"
    "            _mask = response_mask.sum(-1).clamp(min=1)\n"
    "            _row_ret = token_level_rewards.sum(-1).tolist()\n"
    "            _ae = (episode_advantages.sum(-1) / _mask).tolist()\n"
    "            _asv = (step_advantages.sum(-1) / _mask).tolist()\n"
    "            _groups = {}\n"
    "            for _i in range(len(_row_ret)):\n"
    '                _g = _groups.setdefault(str(index[_i]), {"traj": [], "ret": [], "ae": [], "as": []})\n'
    '                _g["traj"].append(str(traj_index[_i]))\n'
    '                _g["ret"].append(round(_row_ret[_i], 4))\n'
    '                _g["ae"].append(round(_ae[_i], 4))\n'
    '                _g["as"].append(round(_asv[_i], 4))\n'
    '                _g.setdefault("row", []).append(_i)  # trustmed:advantages-join-key: batch row = join key with advantages.jsonl\n'
    "            from collections import Counter as _C\n"
    '            _sg = _C(step_group_uids.tolist() if hasattr(step_group_uids, "tolist") else list(step_group_uids))\n'
    "            _sizes = _C(_sg.values())\n"
    "            _mb = response_mask.bool()\n"
    '            _line = {"n_rows": len(_row_ret), "groups": _groups,\n'
    '                     "step_groups": {"distinct": len(_sg),\n'
    '                                      "collided": sum(1 for _v in _sg.values() if _v >= 2),\n'
    '                                      "sizes": {str(_k): _v for _k, _v in sorted(_sizes.items())}},\n'
    '                     "batch": {"adv_mean": float(scores[_mb].mean()),\n'
    '                                "adv_std": float(scores[_mb].std()),\n'
    '                                "adv_max": float(scores[_mb].max()),\n'
    '                                "adv_min": float(scores[_mb].min()),\n'
    '                                "as_nonzero_rows": int(sum(1 for _x in _asv if abs(_x) > 1e-9))},\n'
    '                     "timing_s": ({"ae": round(_t22b - _t22a, 5), "grouping": round(_t22c - _t22b, 5), "as": round(_t22d - _t22c, 5)} if "_t22d" in dir() else {})}\n'
    '            with open(_path, "a") as _f:\n'
    "                _f.write(_json.dumps(_line) + chr(10))\n"
    "    except Exception:\n"
    "        pass\n"
    "    return scores, scores"
)

_P12C_LOOP_OLD = (
    "    seen_pairs = set()\n"
    "    with torch.no_grad():\n"
    "        bsz = scores.shape[0]\n"
    "        for i in range(bsz):\n"
    "            if (index[i], traj_index[i]) in seen_pairs:\n"
    "                continue\n"
    "            id2score[index[i]].append(scores[i])\n"
    "            if not compute_mean_std_cross_steps:\n"
    "                seen_pairs.add((index[i], traj_index[i]))\n"
)
_P12C_LOOP_NEW = (
    "    seen_pairs = set()\n"
    "    with torch.no_grad():\n"
    "        bsz = scores.shape[0]\n"
    "        _traj_best = {}  # trustmed:unpenalised-vote: one vote per trajectory = its UNPENALISED score\n"
    "        for i in range(bsz):\n"
    "            if compute_mean_std_cross_steps:\n"
    "                id2score[index[i]].append(scores[i])\n"
    "                continue\n"
    "            _k = (index[i], traj_index[i])\n"
    "            if _k not in _traj_best or scores[i] > _traj_best[_k]:\n"
    "                _traj_best[_k] = scores[i]\n"
    "        if not compute_mean_std_cross_steps:\n"
    "            for (_idx, _t), _v in _traj_best.items():\n"
    "                id2score[_idx].append(_v)\n"
)
_P12C_GRPO_PREFIX = (
    "            shape is (bs, response_length)\n"
    '    """\n'
    "    scores = token_level_rewards.sum(dim=-1)\n"
    "\n"
    "    id2score = defaultdict(list)\n"
    "    id2mean = {}\n"
    "    id2std = {}\n"
)
UNPENALISED_VOTE_GRPO_OLD = _P12C_GRPO_PREFIX + _P12C_LOOP_OLD
UNPENALISED_VOTE_GRPO_NEW = _P12C_GRPO_PREFIX + _P12C_LOOP_NEW
UNPENALISED_VOTE_GIGPO_OLD = _P12C_LOOP_OLD
UNPENALISED_VOTE_GIGPO_NEW = _P12C_LOOP_NEW

ADVANTAGES_JOIN_KEY_1_OLD = '                _g["as"].append(round(_asv[_i], 4))\n'
ADVANTAGES_JOIN_KEY_1_NEW = (
    '                _g["as"].append(round(_asv[_i], 4))\n'
    '                _g.setdefault("row", []).append(_i)  # trustmed:advantages-join-key: batch row = join key with advantages.jsonl\n'
)
ADVANTAGES_JOIN_KEY_2_OLD = '                            _pd = batch.non_tensor_batch.get("pad_dup")  # trustmed:ledger-pad-dup'
ADVANTAGES_JOIN_KEY_2_NEW = (
    '                            _pd = batch.non_tensor_batch.get("pad_dup")  # trustmed:ledger-pad-dup\n'
    '                            _va = batch.non_tensor_batch.get("is_action_valid")  # trustmed:advantages-join-key\n'
    '                            _sr = batch.batch.get("step_rewards") if hasattr(batch.batch, "get") else None'
)
ADVANTAGES_JOIN_KEY_3_OLD = '                                        "pad_dup": (bool(_pd[_i]) if _pd is not None else False)}) + chr(10))'
ADVANTAGES_JOIN_KEY_3_NEW = (
    '                                        "pad_dup": (bool(_pd[_i]) if _pd is not None else False),\n'
    '                                        "row": _i,\n'
    '                                        "valid": (bool(_va[_i]) if _va is not None else True),\n'
    '                                        "step_ret": (float(_sr[_i]) if _sr is not None else None)}) + chr(10))'
)

ADJUST_BATCH_PAD_OLD = "                    batch = adjust_batch(self.config, batch)"
ADJUST_BATCH_PAD_NEW = (
    "                    _pre_adjust_bs = len(batch)  # trustmed:adjust-batch-pad\n"
    "                    batch = adjust_batch(self.config, batch)\n"
    "                    # trustmed:adjust-batch-pad: adjust_batch pads by CONCATENATING random\n"
    "                    # duplicate rows at the END; stamp them so advantage\n"
    "                    # grouping can exclude them (the flag array permutes\n"
    "                    # together with the batch under balance_batch).\n"
    "                    import numpy as _np13\n"
    "                    _flags = _np13.zeros(len(batch), dtype=bool)\n"
    "                    if len(batch) > _pre_adjust_bs:\n"
    "                        _flags[_pre_adjust_bs:] = True\n"
    "                    batch.non_tensor_batch['pad_dup'] = _flags"
)

PAD_DUP_OUTSIDE_OLD = (
    "    elif adv_estimator == AdvantageEstimator.GiGPO:\n"
    "        advantages, returns = core_gigpo.compute_gigpo_outcome_advantage(\n"
    "            token_level_rewards=data.batch['token_level_rewards'], # for episode group reward computing\n"
    "            step_rewards=data.batch['step_rewards'], # for step group reward computing\n"
    "            response_mask=data.batch['response_mask'],\n"
    "            anchor_obs=data.non_tensor_batch['anchor_obs'],\n"
    "            index=data.non_tensor_batch['uid'],\n"
    "            traj_index=data.non_tensor_batch['traj_uid'],\n"
    "            step_advantage_w=step_advantage_w,\n"
    "            mode=gigpo_mode,\n"
    "            enable_similarity=gigpo_enable_similarity,\n"
    "            similarity_thresh=gigpo_similarity_thresh,\n"
    "            )\n"
    "        data.batch['advantages'] = advantages\n"
    "        data.batch['returns'] = returns"
)
PAD_DUP_OUTSIDE_NEW = (
    "    elif adv_estimator == AdvantageEstimator.GiGPO:\n"
    "        # trustmed:pad-dup-outside: adjust_batch duplicates (pad_dup) must not sit inside\n"
    "        # real episode/step groups (they would double-weight their source\n"
    "        # row's return in every baseline). Give them singleton identities,\n"
    "        # then zero their advantage so they contribute no gradient.\n"
    "        _uid = data.non_tensor_batch['uid']\n"
    "        _tid = data.non_tensor_batch['traj_uid']\n"
    "        _aob = data.non_tensor_batch['anchor_obs']\n"
    "        _pad = data.non_tensor_batch.get('pad_dup')\n"
    "        if _pad is not None and _pad.any():\n"
    "            import numpy as _np13\n"
    "            _uid = _np13.array([('%s#pad%d' % (u, i)) if _pad[i] else u\n"
    "                                for i, u in enumerate(_uid)], dtype=object)\n"
    "            _tid = _np13.array([('%s#pad%d' % (t, i)) if _pad[i] else t\n"
    "                                for i, t in enumerate(_tid)], dtype=object)\n"
    "            _aob = _np13.array([('pad:%d' % i) if _pad[i] else a\n"
    "                                for i, a in enumerate(_aob)], dtype=object)\n"
    "        advantages, returns = core_gigpo.compute_gigpo_outcome_advantage(\n"
    "            token_level_rewards=data.batch['token_level_rewards'], # for episode group reward computing\n"
    "            step_rewards=data.batch['step_rewards'], # for step group reward computing\n"
    "            response_mask=data.batch['response_mask'],\n"
    "            anchor_obs=_aob,\n"
    "            index=_uid,\n"
    "            traj_index=_tid,\n"
    "            step_advantage_w=step_advantage_w,\n"
    "            mode=gigpo_mode,\n"
    "            enable_similarity=gigpo_enable_similarity,\n"
    "            similarity_thresh=gigpo_similarity_thresh,\n"
    "            )\n"
    "        if _pad is not None and _pad.any():\n"
    "            import torch as _t13\n"
    "            _pm = _t13.from_numpy(_pad.astype('bool')).to(advantages.device)\n"
    "            advantages = advantages.masked_fill(_pm.unsqueeze(-1), 0.0)\n"
    "            returns = returns.masked_fill(_pm.unsqueeze(-1), 0.0)\n"
    "        data.batch['advantages'] = advantages\n"
    "        data.batch['returns'] = returns"
)

METRICS_SIDECAR_START = "                try:  # trustmed:metrics-sidecar"
METRICS_SIDECAR_END = "                except Exception:\n                    pass"
METRICS_SIDECAR_NEW = (
    "                try:  # trustmed:metrics-sidecar: metrics sidecar for offline curves (per-value safe)\n"
    "                    import json as _json, os as _os\n"
    "                    _d = self.config.trainer.default_local_dir\n"
    "                    _os.makedirs(_d, exist_ok=True)\n"
    '                    _row = {"step": int(self.global_steps)}\n'
    "                    for _k, _v in metrics.items():\n"
    "                        try:\n"
    "                            _row[_k] = float(_v)\n"
    "                        except Exception:\n"
    "                            try:\n"
    "                                _row[_k] = str(_v)[:200]\n"
    "                            except Exception:\n"
    "                                pass\n"
    '                    with open(_os.path.join(_d, "metrics.jsonl"), "a") as _f:\n'
    "                        _f.write(_json.dumps(_row) + chr(10))\n"
    "                except Exception:\n"
    "                    pass"
)
METRICS_SIDECAR_MARKER = (
    "trustmed:metrics-sidecar: metrics sidecar for offline curves (per-value safe)"
)


class Edit:
    def __init__(
        self, name: str, rel: str, marker: str, find: str, repl: str, optional: bool = False
    ):
        self.name, self.rel, self.marker, self.find, self.repl = name, rel, marker, find, repl
        self.optional = optional

    def status(self, fork: Path) -> tuple[str, int]:
        txt = (fork / self.rel).read_text(encoding="utf-8")
        if self.marker in txt:
            return "already", 0
        n = txt.count(self.find)
        if n == 0 and self.optional:
            return "skip", 0
        return "pending", n


LEDGER_PAD_DUP_1_OLD = '                            _tuid = batch.non_tensor_batch.get("traj_uid")'
LEDGER_PAD_DUP_1_NEW = (
    '                            _tuid = batch.non_tensor_batch.get("traj_uid")\n'
    '                            _pd = batch.non_tensor_batch.get("pad_dup")  # trustmed:ledger-pad-dup\n'
    '                            _va = batch.non_tensor_batch.get("is_action_valid")  # trustmed:advantages-join-key\n'
    '                            _sr = batch.batch.get("step_rewards") if hasattr(batch.batch, "get") else None'
)
LEDGER_PAD_DUP_2_OLD = '                                        "resp_len": _n}) + chr(10))'
LEDGER_PAD_DUP_2_NEW = (
    '                                        "resp_len": _n,\n'
    '                                        "pad_dup": (bool(_pd[_i]) if _pd is not None else False),\n'
    '                                        "row": _i,\n'
    '                                        "valid": (bool(_va[_i]) if _va is not None else True),\n'
    '                                        "step_ret": (float(_sr[_i]) if _sr is not None else None)}) + chr(10))'
)

TRAJECTORY_BASELINE_GRPO_GRPO_OLD = (
    "    compute_mean_std_cross_steps: bool = True,\n"
    "):\n"
    '    """\n'
    "    Compute advantage for GRPO, operating only on Outcome reward"
)
TRAJECTORY_BASELINE_GRPO_GRPO_NEW = (
    "    compute_mean_std_cross_steps: bool = False,  # trustmed:trajectory-baseline-grpo: per-trajectory baseline (paper Eq.3), same as GiGPO A^E\n"
    "):\n"
    '    """\n'
    "    Compute advantage for GRPO, operating only on Outcome reward"
)
TRAJECTORY_BASELINE_GRPO_GIGPO_OLD = (
    "                        compute_mean_std_cross_steps: bool = True,"
)
TRAJECTORY_BASELINE_GRPO_GIGPO_NEW = (
    "                        compute_mean_std_cross_steps: bool = False,"
    "  # trustmed:trajectory-baseline-grpo: per-trajectory A^E baseline (paper Eq.3), same as the GRPO estimator"
)

PAD_DUP_OUTSIDE_MASK_OLD = (
    "        # Call compute_grpo_outcome_advantage with parameters matching its definition\n"
    "        advantages, returns = core_algos.compute_grpo_outcome_advantage(\n"
    '            token_level_rewards=data.batch["token_level_rewards"],\n'
    "            response_mask=grpo_calculation_mask,\n"
    '            index=data.non_tensor_batch["uid"],\n'
    "            traj_index=data.non_tensor_batch['traj_uid'],\n"
    "            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,\n"
    "        )\n"
    '        data.batch["advantages"] = advantages\n'
    '        data.batch["returns"] = returns'
)
PAD_DUP_OUTSIDE_MASK_NEW = (
    "        # trustmed:pad-dup-outside-mask: adjust_batch duplicates (pad_dup) must not sit inside the\n"
    "        # GRPO group baseline nor train a second copy of their source row --\n"
    "        # same treatment as the GiGPO branch (trustmed:pad-dup-outside).\n"
    "        _uid = data.non_tensor_batch['uid']\n"
    "        _tid = data.non_tensor_batch['traj_uid']\n"
    "        _pad = data.non_tensor_batch.get('pad_dup')\n"
    "        if _pad is not None and _pad.any():\n"
    "            import numpy as _np13\n"
    "            _uid = _np13.array([('%s#pad%d' % (u, i)) if _pad[i] else u\n"
    "                                for i, u in enumerate(_uid)], dtype=object)\n"
    "            _tid = _np13.array([('%s#pad%d' % (t, i)) if _pad[i] else t\n"
    "                                for i, t in enumerate(_tid)], dtype=object)\n"
    "        # Call compute_grpo_outcome_advantage with parameters matching its definition\n"
    "        advantages, returns = core_algos.compute_grpo_outcome_advantage(\n"
    '            token_level_rewards=data.batch["token_level_rewards"],\n'
    "            response_mask=grpo_calculation_mask,\n"
    "            index=_uid,\n"
    "            traj_index=_tid,\n"
    "            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,\n"
    "        )\n"
    "        if _pad is not None and _pad.any():\n"
    "            import torch as _t13\n"
    "            _pm = _t13.from_numpy(_pad.astype('bool')).to(advantages.device)\n"
    "            advantages = advantages.masked_fill(_pm.unsqueeze(-1), 0.0)\n"
    "            returns = returns.masked_fill(_pm.unsqueeze(-1), 0.0)\n"
    '        data.batch["advantages"] = advantages\n'
    '        data.batch["returns"] = returns'
)

MICRO_BATCH_MEAN_OLD = (
    '                        metrics["actor/kl_loss"] = kl_loss.detach().item()\n'
    '                        metrics["actor/kl_coef"] = self.config.kl_loss_coef'
)
MICRO_BATCH_MEAN_NEW = (
    '                        append_to_dict(metrics, {"actor/kl_loss": kl_loss.detach().item(),  # trustmed:micro-batch-mean: mean over micro-batches, not the last one\n'
    '                                                 "actor/kl_coef": self.config.kl_loss_coef})'
)

MIXED_IMAGE_TEXT_ROWS_OLD = (
    '        if "multi_modal_inputs" in micro_batch:\n'
    '            for key in micro_batch["multi_modal_inputs"][0].keys():\n'
    '                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)'
)
MIXED_IMAGE_TEXT_ROWS_NEW = (
    '        if "multi_modal_inputs" in micro_batch:\n'
    "            # trustmed:mixed-image-text-rows: rows may MIX images and text (text rows carry {}, trustmed:text-row-empty-mm);\n"
    "            # gather the union of keys over the rows that have them, in row order.\n"
    '            _mm_rows = [r for r in micro_batch["multi_modal_inputs"] if r]\n'
    "            for key in sorted({k for r in _mm_rows for k in r}):\n"
    "                multi_modal_inputs[key] = torch.cat([r[key] for r in _mm_rows if key in r], dim=0)"
)

VAL_STEP_STAMP_ANCHOR = "        for test_data in self.val_dataloader:\n"
VAL_STEP_STAMP_INSERT = (
    "        # trustmed:val-step-stamp: stamp the trainer step into the static val dataset so every\n"
    "        # validation record names the checkpoint it evaluates\n"
    '        _val_ds = getattr(self.val_dataloader, "dataset", None)\n'
    '        if hasattr(_val_ds, "set_step"):\n'
    "            _val_ds.set_step(int(self.global_steps))\n"
)

CURRICULUM_SAMPLER_RESUME_OLD = (
    '        dataloader_local_path = os.path.join(global_step_folder, "data.pt")\n'
)
CURRICULUM_SAMPLER_RESUME_NEW = (
    '        dataloader_local_path = os.path.join(global_step_folder, "data.pt")\n'
    '        if getattr(self.config, "curriculum", None) is not None:   # trustmed:curriculum-sampler-resume: curriculum sampler resumes itself\n'
    '            print("trustmed:curriculum-sampler-resume: curriculum sampler installed; data.pt not restored")\n'
    '            dataloader_local_path = os.path.join(global_step_folder, "data.pt.ignored")\n'
)
CONFIGURED_LR_OLD = (
    "        self.checkpoint_manager.load_checkpoint(local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load)\n"
    "\n"
    "        if self._is_offload_param:\n"
    "            offload_fsdp_model_to_cpu(self.actor_module_fsdp)\n"
)
CONFIGURED_LR_NEW = (
    "        self.checkpoint_manager.load_checkpoint(local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load)\n"
    "        # trustmed:configured-lr: the configured lr wins over the lr stored in the optimizer state\n"
    "        _lr = float(self.config.actor.optim.lr)\n"
    "        for _g in self.actor_optimizer.param_groups:\n"
    '            _g["lr"] = _lr\n'
    '        if getattr(self, "actor_lr_scheduler", None) is not None and hasattr(self.actor_lr_scheduler, "base_lrs"):\n'
    "            self.actor_lr_scheduler.base_lrs = [_lr for _ in self.actor_lr_scheduler.base_lrs]\n"
    '        print(f"trustmed:configured-lr: optimizer lr set to {_lr} after checkpoint load")\n'
    "\n"
    "        if self._is_offload_param:\n"
    "            offload_fsdp_model_to_cpu(self.actor_module_fsdp)\n"
)

STAGE_TIMERS_OLD = (
    "    # Compute episode relative advantages (Eq. 3 in the paper)."
    + NL
    + "    episode_advantages = episode_norm_reward(token_level_rewards, response_mask, index, traj_index, epsilon, remove_std)"
)
STAGE_TIMERS_NEW = (
    "    # Compute episode relative advantages (Eq. 3 in the paper)."
    + NL
    + "    import time as _t22; _t22a = _t22.perf_counter()  # trustmed:stage-timers: stage timers"
    + NL
    + "    episode_advantages = episode_norm_reward(token_level_rewards, response_mask, index, traj_index, epsilon, remove_std)"
    + NL
    + "    _t22b = _t22.perf_counter()"
)
STAGE_TIMER_GROUPING_OLD = "    step_group_uids = build_step_group(anchor_obs, index, enable_similarity, similarity_thresh)"
STAGE_TIMER_GROUPING_NEW = (
    "    step_group_uids = build_step_group(anchor_obs, index, enable_similarity, similarity_thresh)"
    + NL
    + "    _t22c = _t22.perf_counter()  # trustmed:stage-timer-grouping"
)
STAGE_TIMER_STEP_OLD = "    step_advantages = step_norm_reward(step_rewards, response_mask, step_group_uids, epsilon, remove_std)"
STAGE_TIMER_STEP_NEW = (
    "    step_advantages = step_norm_reward(step_rewards, response_mask, step_group_uids, epsilon, remove_std)"
    + NL
    + "    _t22d = _t22.perf_counter()  # trustmed:stage-timer-step"
)

_P21_DIV = "scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)"


def _p21_new(tag):
    return (
        "scores[i] = (scores[i] - id2mean[index[i]]) / (torch.clamp(id2std[index[i]], "
        'min=float(__import__("os").environ.get("GIGPO_STD_FLOOR", "0") or 0)) + epsilon)'
        "  # " + tag
    )


STD_FLOOR_GRPO_GRPO_OLD = (
    "            if norm_adv_by_std_in_grpo:" + NL + "                " + _P21_DIV
)
STD_FLOOR_GRPO_GRPO_NEW = (
    "            if norm_adv_by_std_in_grpo:"
    + NL
    + "                "
    + _p21_new("trustmed:std-floor-grpo: std floor (GRPO)")
)
STD_FLOOR_GRPO_EP_TAIL = (
    NL
    + "        episode_advantages = scores.unsqueeze(-1).tile([1, response_length]) * response_mask"
)
STD_FLOOR_GRPO_ST_TAIL = (
    NL + "        step_advantages = scores.unsqueeze(-1).tile([1, response_length]) * response_mask"
)
STD_FLOOR_GRPO_GIGPO_EP_OLD = (
    "            else:" + NL + "                " + _P21_DIV + STD_FLOOR_GRPO_EP_TAIL
)
STD_FLOOR_GRPO_GIGPO_EP_NEW = (
    "            else:"
    + NL
    + "                "
    + _p21_new("trustmed:std-floor-episode: std floor (A^E)")
    + STD_FLOOR_GRPO_EP_TAIL
)
STD_FLOOR_GRPO_GIGPO_ST_OLD = (
    "            else:" + NL + "                " + _P21_DIV + STD_FLOOR_GRPO_ST_TAIL
)
STD_FLOOR_GRPO_GIGPO_ST_NEW = (
    "            else:"
    + NL
    + "                "
    + _p21_new("trustmed:std-floor-step: std floor (A^S)")
    + STD_FLOOR_GRPO_ST_TAIL
)

STAGE_TIMER_PRETIMING_OLD = '                                "as_nonzero_rows": int(sum(1 for _x in _asv if abs(_x) > 1e-9))}}'
STAGE_TIMER_PRETIMING_NEW = (
    '                                "as_nonzero_rows": int(sum(1 for _x in _asv if abs(_x) > 1e-9))},'
    + NL
    + '                     "timing_s": ({"ae": round(_t22b - _t22a, 5), "grouping": round(_t22c - _t22b, 5), "as": round(_t22d - _t22c, 5)} if "_t22d" in dir() else {})}'
)


DX_BOOST_A_OLD = (
    "            if 'is_action_valid' in infos[0]:\n"
    "                batch.non_tensor_batch['is_action_valid'] = np.array([info['is_action_valid'] for info in infos], dtype=bool)\n"
    "            else:\n"
    "                batch.non_tensor_batch['is_action_valid'] = np.ones(batch_size, dtype=bool)\n"
)
DX_BOOST_A_NEW = (
    DX_BOOST_A_OLD
    + "            # trustmed:dx-boost: terminal-only boost eligibility (full-credit correct dx); same broadcast\n"
    "            batch.non_tensor_batch['dx_boost'] = np.array(\n"
    "                [bool(info.get('dx_boost', False)) for info in infos], dtype=bool)\n"
)


def _p26_mask(tag: str) -> str:
    """The mask block, tagged per branch (b = GRPO, c = GiGPO) so that each Edit's marker is
    introduced by that Edit alone; with one shared tag the GiGPO mask read as already applied once
    the GRPO branch carried its copy.
    """
    return (
        "        # trustmed:dx-boost-"
        + tag
        + ": dx boost mask -- the env's terminal-row dx_boost OR-ed over the ORIGINAL\n"
        "        # traj_uid (rollout identity, never the case uid), minus forfeit rows and pad dups\n"
        "        import os as _os26\n"
        "        import numpy as _np26\n"
        "        _bs26 = float(_os26.environ.get('DX_ADV_SCALE', '1') or 1)\n"
        "        _bm26 = None\n"
        "        _dxb26 = data.non_tensor_batch.get('dx_boost')\n"
        "        if _bs26 != 1.0 and _dxb26 is not None:\n"
        "            _tj26 = _np26.asarray(data.non_tensor_batch['traj_uid'])\n"
        "            _won26 = set(_tj26[_np26.asarray(_dxb26, dtype=bool)].tolist())\n"
        "            _va26 = data.non_tensor_batch.get('is_action_valid')\n"
        "            _pd26 = data.non_tensor_batch.get('pad_dup')\n"
        "            _bm26 = _np26.array([(_t in _won26)\n"
        "                                 and (True if _va26 is None else bool(_va26[_i]))\n"
        "                                 and not (_pd26 is not None and bool(_pd26[_i]))\n"
        "                                 for _i, _t in enumerate(_tj26.tolist())], dtype=bool)\n"
    )


DX_BOOST_B_OLD = (
    "        # Call compute_grpo_outcome_advantage with parameters matching its definition\n"
    "        advantages, returns = core_algos.compute_grpo_outcome_advantage(\n"
    '            token_level_rewards=data.batch["token_level_rewards"],\n'
    "            response_mask=grpo_calculation_mask,\n"
)
DX_BOOST_B_NEW = (
    _p26_mask("b")
    + DX_BOOST_B_OLD
    + "            boost_mask=_bm26, boost_scale=_bs26,  # trustmed:dx-boost-b\n"
)

DX_BOOST_C_OLD = (
    "        advantages, returns = core_gigpo.compute_gigpo_outcome_advantage(\n"
    "            token_level_rewards=data.batch['token_level_rewards'], # for episode group reward computing\n"
)
DX_BOOST_C_NEW = (
    _p26_mask("c") + "        advantages, returns = core_gigpo.compute_gigpo_outcome_advantage(\n"
    "            token_level_rewards=data.batch['token_level_rewards'], # for episode group reward computing\n"
)
DX_BOOST_D_OLD = "            similarity_thresh=gigpo_similarity_thresh,\n            )\n"
DX_BOOST_D_NEW = (
    "            similarity_thresh=gigpo_similarity_thresh,\n"
    "            boost_mask=_bm26, boost_scale=_bs26,  # trustmed:dx-boost-c (kwargs)\n"
    "            )\n"
)

DX_BOOST_E_OLD = (
    "                                   similarity_thresh: float = 0.95,\n"
    "                                   ):\n"
    '    """\n'
    "    Compute the advantages for GiGPO (https://arxiv.org/abs/2505.10978).\n"
)
DX_BOOST_E_NEW = (
    "                                   similarity_thresh: float = 0.95,\n"
    "                                   boost_mask=None,  # trustmed:dx-boost: per-row bool, dx boost eligibility\n"
    "                                   boost_scale: float = 1.0,\n"
    "                                   ):\n"
    '    """\n'
    "    Compute the advantages for GiGPO (https://arxiv.org/abs/2505.10978).\n"
)
DX_BOOST_F_OLD = (
    "    # Compute joint advantages (Eq. 8 in the paper).\n"
    "    scores = episode_advantages + step_advantage_w * step_advantages\n"
)
DX_BOOST_F_NEW = (
    "    if boost_mask is not None and float(boost_scale) != 1.0:  # trustmed:dx-boost: dx boost, post-normalisation, A^E only\n"
    "        _bm26 = torch.as_tensor(np.asarray(boost_mask, dtype=bool), device=episode_advantages.device)\n"
    "        _n26 = response_mask.sum(-1)\n"
    "        _ae26 = episode_advantages.sum(-1) / _n26.clamp(min=1)\n"
    "        _jt26 = (episode_advantages + step_advantage_w * step_advantages).sum(-1) / _n26.clamp(min=1)\n"
    "        _bm26 = _bm26 & (_n26 > 0) & (_ae26 > 0) & (_jt26 > 0)\n"
    "        episode_advantages = torch.where(_bm26.unsqueeze(-1), episode_advantages * float(boost_scale), episode_advantages)\n"
    "    # Compute joint advantages (Eq. 8 in the paper).\n"
    "    scores = episode_advantages + step_advantage_w * step_advantages\n"
)

DX_BOOST_G_OLD = '):\n    """\n    Compute advantage for GRPO, operating only on Outcome reward\n'
DX_BOOST_G_NEW = (
    "    boost_mask=None,  # trustmed:dx-boost: per-row bool, dx boost eligibility\n"
    "    boost_scale: float = 1.0,\n" + DX_BOOST_G_OLD
)
DX_BOOST_H_OLD = (
    "            else:\n"
    "                scores[i] = scores[i] - id2mean[index[i]]\n"
    "        scores = scores.unsqueeze(-1) * response_mask\n"
    "\n"
    "    return scores, scores\n"
)
DX_BOOST_H_NEW = (
    "            else:\n"
    "                scores[i] = scores[i] - id2mean[index[i]]\n"
    "        if boost_mask is not None and float(boost_scale) != 1.0:  # trustmed:dx-boost: dx boost (GRPO arm: A = A^E)\n"
    "            _bm26 = torch.as_tensor(np.asarray(boost_mask, dtype=bool), device=scores.device)\n"
    "            _bm26 = _bm26 & (response_mask.sum(-1) > 0) & (scores > 0)\n"
    "            scores = torch.where(_bm26, scores * float(boost_scale), scores)\n"
    "        scores = scores.unsqueeze(-1) * response_mask\n"
    "\n"
    "    return scores, scores\n"
)

ACTOR_INSTRUMENTATION_OLD = (
    "            if 'is_action_valid' in infos[0]:\n"
    "                batch.non_tensor_batch['is_action_valid'] = np.array([info['is_action_valid'] for info in infos], dtype=bool)\n"
    "            else:\n"
    "                batch.non_tensor_batch['is_action_valid'] = np.ones(batch_size, dtype=bool)\n"
)
ACTOR_INSTRUMENTATION_NEW = (
    ACTOR_INSTRUMENTATION_OLD + "\n"
    "            # trustmed:terminal-flag-broadcast: terminal-only flag; ray_trainer broadcasts it over the trajectory\n"
    "            batch.non_tensor_batch['dx_correct'] = np.array(\n"
    "                [bool(info.get('dx_correct', False)) for info in infos], dtype=bool)\n"
)

DX_CORRECT_BROADCAST_OLD = (
    "                            gigpo_similarity_thresh=self.config.algorithm.gigpo.similarity_thresh,\n"
    "                        )\n"
)
DX_CORRECT_BROADCAST_NEW = (
    DX_CORRECT_BROADCAST_OLD + "\n"
    "                        # trustmed:dx-correct-broadcast: the env sets dx_correct on the terminal row only; every row of\n"
    "                        # that trajectory carries its tokens, so the flag is OR-ed over traj_uid and\n"
    "                        # handed to the actor as a tensor batch key (survives split/chunk/rearrange).\n"
    "                        _dxc = batch.non_tensor_batch.get('dx_correct')\n"
    "                        _dxt = batch.non_tensor_batch.get('traj_uid')\n"
    "                        if _dxc is not None and _dxt is not None:\n"
    "                            _dxt = np.asarray(_dxt)\n"
    "                            _won = set(_dxt[np.asarray(_dxc, dtype=bool)].tolist())\n"
    "                            batch.batch['dx_correct'] = torch.tensor(\n"
    "                                [1.0 if _u in _won else 0.0 for _u in _dxt.tolist()],\n"
    "                                dtype=torch.float32)\n"
)

ACTOR_FLAG_SELECT_OLD = "from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, compute_policy_loss_gspo, kl_penalty\n"
ACTOR_FLAG_SELECT_NEW = (
    ACTOR_FLAG_SELECT_OLD
    + "try:                                   # trustmed:terminal-flag-broadcast/25: trustmed actor instrumentation\n"
    "    from agent_system.environments.env_package.trustmed import actor_metrics as _sp_am\n"
    "except Exception as _e:                # never let instrumentation stop a run\n"
    "    print('WARN trustmed actor_metrics unavailable:', repr(_e)[:120])\n"
    "    _sp_am = None\n"
)

ACTOR_SELECT_KEYS_OLD = (
    '        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]\n'
    "        if multi_turn:\n"
)
ACTOR_SELECT_KEYS_NEW = (
    '        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]\n'
    '        if "dx_correct" in data.batch.keys():          # trustmed:actor-select-keys\n'
    '            select_keys.append("dx_correct")\n'
    "        if multi_turn:\n"
)

CLIP_SPLIT_BY_SIGN_OLD = (
    "                    data = {\n"
    '                        "actor/pg_loss": pg_loss.detach().item(),\n'
)
CLIP_SPLIT_BY_SIGN_NEW = (
    "                    # trustmed:clip-split-by-sign: clipping split by sign and by correct diagnosis\n"
    "                    _sp_extra = {}\n"
    "                    if _sp_am is not None:\n"
    "                        try:\n"
    '                            _sel = data["dx_correct"] if "dx_correct" in data.keys() else None\n'
    "                            _sp_extra = _sp_am.success_clip_metrics(\n"
    "                                old_log_prob=old_log_prob, log_prob=log_prob,\n"
    "                                advantages=advantages, response_mask=response_mask,\n"
    "                                clip_high=clip_ratio_high, clip_low=clip_ratio_low,\n"
    "                                row_select=_sel)\n"
    "                        except Exception as _e:\n"
    '                            print("WARN trustmed:clip-split-by-sign:", repr(_e)[:160]); _sp_extra = {}\n'
    + CLIP_SPLIT_BY_SIGN_OLD
)

ACTOR_METRIC_MERGE_OLD = (
    "                    }\n"
    "                    append_to_dict(metrics, data)\n"
    "\n"
    "                grad_norm = self._optimizer_step()\n"
    '                data = {"actor/grad_norm": grad_norm.detach().item()}\n'
    "                append_to_dict(metrics, data)\n"
)
ACTOR_METRIC_MERGE_NEW = (
    "                    }\n"
    "                    data.update(_sp_extra)                      # trustmed:actor-metric-merge\n"
    "                    append_to_dict(metrics, data)\n"
    "\n"
    "                grad_norm = self._optimizer_step()\n"
    '                data = {"actor/grad_norm": grad_norm.detach().item()}\n'
    '                data.update(getattr(self, "_sp_step_metrics", None) or {})   # trustmed:update-norm-probe\n'
    "                append_to_dict(metrics, data)\n"
)

UPDATE_NORM_PROBE_OLD = (
    "        # if grad_norm is not finite, skip the update\n"
    "        if not torch.isfinite(grad_norm):\n"
    '            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")\n'
    "            self.actor_optimizer.zero_grad()\n"
    "        else:\n"
    "            self.actor_optimizer.step()\n"
    "        return grad_norm\n"
)
UPDATE_NORM_PROBE_NEW = (
    "        # trustmed:update-norm-probe: pre-clip norm, the scale clipping applied, and the real parameter delta\n"
    "        self._sp_step_metrics = {}\n"
    "        if _sp_am is not None:\n"
    "            try:\n"
    "                self._sp_step_metrics = _sp_am.clip_metrics(grad_norm, self.config.grad_clip)\n"
    "                if getattr(self, '_sp_probe', None) is None:\n"
    "                    self._sp_probe = _sp_am.ParamDeltaProbe(self.actor_module.parameters())\n"
    "                self._sp_probe.snapshot()\n"
    "            except Exception as _e:\n"
    "                print('WARN trustmed:update-norm-probe snapshot:', repr(_e)[:160]); self._sp_probe = None\n"
    "\n"
    "        # if grad_norm is not finite, skip the update\n"
    "        if not torch.isfinite(grad_norm):\n"
    '            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")\n'
    "            self.actor_optimizer.zero_grad()\n"
    "            self._sp_step_metrics['actor/step_skipped_nonfinite'] = 1.0\n"
    "        else:\n"
    "            self.actor_optimizer.step()\n"
    "            self._sp_step_metrics['actor/step_skipped_nonfinite'] = 0.0\n"
    "            if getattr(self, '_sp_probe', None) is not None:\n"
    "                try:\n"
    "                    self._sp_step_metrics.update(self._sp_probe.delta())\n"
    "                except Exception as _e:\n"
    "                    print('WARN trustmed:update-norm-probe delta:', repr(_e)[:160])\n"
    "        return grad_norm\n"
)


EDITS = [
    Edit(
        "no data.pt under the curriculum sampler",
        "verl/trainer/ppo/ray_trainer.py",
        "trustmed:curriculum-sampler-resume",
        CURRICULUM_SAMPLER_RESUME_OLD,
        CURRICULUM_SAMPLER_RESUME_NEW,
    ),
    Edit(
        "stage timer A^E",
        "gigpo/core_gigpo.py",
        "trustmed:stage-timers: stage timers",
        STAGE_TIMERS_OLD,
        STAGE_TIMERS_NEW,
    ),
    Edit(
        "retrofit timing into old step-group-telemetry",
        "gigpo/core_gigpo.py",
        "timing_s",
        STAGE_TIMER_PRETIMING_OLD,
        STAGE_TIMER_PRETIMING_NEW,
        optional=True,
    ),
    Edit(
        "stage timer grouping",
        "gigpo/core_gigpo.py",
        "trustmed:stage-timer-grouping",
        STAGE_TIMER_GROUPING_OLD,
        STAGE_TIMER_GROUPING_NEW,
    ),
    Edit(
        "stage timer A^S",
        "gigpo/core_gigpo.py",
        "trustmed:stage-timer-step",
        STAGE_TIMER_STEP_OLD,
        STAGE_TIMER_STEP_NEW,
    ),
    Edit(
        "GRPO std floor",
        "verl/trainer/ppo/core_algos.py",
        "trustmed:std-floor-grpo: std floor (GRPO)",
        STD_FLOOR_GRPO_GRPO_OLD,
        STD_FLOOR_GRPO_GRPO_NEW,
    ),
    Edit(
        "GiGPO A^E std floor",
        "gigpo/core_gigpo.py",
        "trustmed:std-floor-episode",
        STD_FLOOR_GRPO_GIGPO_EP_OLD,
        STD_FLOOR_GRPO_GIGPO_EP_NEW,
    ),
    Edit(
        "GiGPO A^S std floor",
        "gigpo/core_gigpo.py",
        "trustmed:std-floor-step",
        STD_FLOOR_GRPO_GIGPO_ST_OLD,
        STD_FLOOR_GRPO_GIGPO_ST_NEW,
    ),
    Edit(
        "lr re-applied after load",
        "verl/workers/fsdp_workers.py",
        "trustmed:configured-lr",
        CONFIGURED_LR_OLD,
        CONFIGURED_LR_NEW,
    ),
    Edit(
        "val step stamp",
        "verl/trainer/ppo/ray_trainer.py",
        "trustmed:val-step-stamp",
        VAL_STEP_STAMP_ANCHOR,
        VAL_STEP_STAMP_INSERT + VAL_STEP_STAMP_ANCHOR,
    ),
    Edit(
        "anchor alignment",
        "agent_system/multi_turn_rollout/rollout_loop.py",
        "trustmed:state-of-decision",
        STATE_OF_DECISION_ANCHOR,
        STATE_OF_DECISION_INSERT + STATE_OF_DECISION_ANCHOR,
    ),
    Edit(
        "mixed-row mm gather",
        "verl/workers/actor/dp_actor.py",
        "trustmed:mixed-image-text-rows",
        MIXED_IMAGE_TEXT_ROWS_OLD,
        MIXED_IMAGE_TEXT_ROWS_NEW,
    ),
    Edit(
        "GRPO per-trajectory baseline",
        "verl/trainer/ppo/core_algos.py",
        "trustmed:trajectory-baseline-grpo",
        TRAJECTORY_BASELINE_GRPO_GRPO_OLD,
        TRAJECTORY_BASELINE_GRPO_GRPO_NEW,
    ),
    Edit(
        "GiGPO per-trajectory A^E",
        "gigpo/core_gigpo.py",
        "trustmed:trajectory-baseline-grpo",
        TRAJECTORY_BASELINE_GRPO_GIGPO_OLD,
        TRAJECTORY_BASELINE_GRPO_GIGPO_NEW,
    ),
    Edit(
        "GRPO pad-dup masking",
        "verl/trainer/ppo/ray_trainer.py",
        "trustmed:pad-dup-outside-mask",
        PAD_DUP_OUTSIDE_MASK_OLD,
        PAD_DUP_OUTSIDE_MASK_NEW,
    ),
    Edit(
        "kl_loss mean over micro-batches",
        "verl/workers/actor/dp_actor.py",
        "trustmed:micro-batch-mean",
        MICRO_BATCH_MEAN_OLD,
        MICRO_BATCH_MEAN_NEW,
    ),
    Edit(
        "group telemetry",
        "gigpo/core_gigpo.py",
        "trustmed:step-group-telemetry",
        STEP_GROUP_TELEMETRY_ANCHOR,
        STEP_GROUP_TELEMETRY_NEW,
    ),
    Edit(
        "stamp pad_dup",
        "verl/trainer/ppo/ray_trainer.py",
        "trustmed:adjust-batch-pad",
        ADJUST_BATCH_PAD_OLD,
        ADJUST_BATCH_PAD_NEW,
    ),
    Edit(
        "dups out of GiGPO groups",
        "verl/trainer/ppo/ray_trainer.py",
        "trustmed:pad-dup-outside",
        PAD_DUP_OUTSIDE_OLD,
        PAD_DUP_OUTSIDE_NEW,
    ),
    Edit(
        "ledger pad_dup (1/2)",
        "verl/trainer/ppo/ray_trainer.py",
        "trustmed:ledger-pad-dup",
        LEDGER_PAD_DUP_1_OLD,
        LEDGER_PAD_DUP_1_NEW,
    ),
    Edit(
        "ledger pad_dup (2/2)",
        "verl/trainer/ppo/ray_trainer.py",
        '"pad_dup": (bool(_pd[_i])',
        LEDGER_PAD_DUP_2_OLD,
        LEDGER_PAD_DUP_2_NEW,
    ),
    Edit(
        "GRPO unpenalised group vote",
        "verl/trainer/ppo/core_algos.py",
        "trustmed:unpenalised-vote",
        UNPENALISED_VOTE_GRPO_OLD,
        UNPENALISED_VOTE_GRPO_NEW,
    ),
    Edit(
        "GiGPO unpenalised group vote",
        "gigpo/core_gigpo.py",
        "trustmed:unpenalised-vote",
        UNPENALISED_VOTE_GIGPO_OLD,
        UNPENALISED_VOTE_GIGPO_NEW,
    ),
    Edit(
        "groups ledger row key",
        "gigpo/core_gigpo.py",
        "trustmed:advantages-join-key",
        ADVANTAGES_JOIN_KEY_1_OLD,
        ADVANTAGES_JOIN_KEY_1_NEW,
        optional=True,
    ),
    Edit(
        "advantages ledger valid (1/2)",
        "verl/trainer/ppo/ray_trainer.py",
        "trustmed:advantages-join-key",
        ADVANTAGES_JOIN_KEY_2_OLD,
        ADVANTAGES_JOIN_KEY_2_NEW,
        optional=True,
    ),
    Edit(
        "advantages ledger valid (2/2)",
        "verl/trainer/ppo/ray_trainer.py",
        '"valid": (bool(_va[_i])',
        ADVANTAGES_JOIN_KEY_3_OLD,
        ADVANTAGES_JOIN_KEY_3_NEW,
        optional=True,
    ),
    Edit(
        "rollout dx_correct",
        "agent_system/multi_turn_rollout/rollout_loop.py",
        "trustmed:terminal-flag-broadcast",
        ACTOR_INSTRUMENTATION_OLD,
        ACTOR_INSTRUMENTATION_NEW,
    ),
    Edit(
        "broadcast dx_correct",
        "verl/trainer/ppo/ray_trainer.py",
        "trustmed:dx-correct-broadcast",
        DX_CORRECT_BROADCAST_OLD,
        DX_CORRECT_BROADCAST_NEW,
    ),
    Edit(
        "dp_actor import",
        "verl/workers/actor/dp_actor.py",
        "trustmed:terminal-flag-broadcast/25",
        ACTOR_FLAG_SELECT_OLD,
        ACTOR_FLAG_SELECT_NEW,
    ),
    Edit(
        "dp_actor select_keys",
        "verl/workers/actor/dp_actor.py",
        "trustmed:actor-select-keys",
        ACTOR_SELECT_KEYS_OLD,
        ACTOR_SELECT_KEYS_NEW,
    ),
    Edit(
        "dp_actor clip metrics",
        "verl/workers/actor/dp_actor.py",
        "trustmed:clip-split-by-sign",
        CLIP_SPLIT_BY_SIGN_OLD,
        CLIP_SPLIT_BY_SIGN_NEW,
    ),
    Edit(
        "metric merge",
        "verl/workers/actor/dp_actor.py",
        "trustmed:actor-metric-merge",
        ACTOR_METRIC_MERGE_OLD,
        ACTOR_METRIC_MERGE_NEW,
    ),
    Edit(
        "optimizer step accounting",
        "verl/workers/actor/dp_actor.py",
        "trustmed:update-norm-probe",
        UPDATE_NORM_PROBE_OLD,
        UPDATE_NORM_PROBE_NEW,
    ),
    Edit(
        "rollout dx_boost",
        "agent_system/multi_turn_rollout/rollout_loop.py",
        "trustmed:dx-boost",
        DX_BOOST_A_OLD,
        DX_BOOST_A_NEW,
    ),
    Edit(
        "GRPO branch mask + kwargs",
        "verl/trainer/ppo/ray_trainer.py",
        "trustmed:dx-boost-b",
        DX_BOOST_B_OLD,
        DX_BOOST_B_NEW,
    ),
    Edit(
        "GiGPO branch mask",
        "verl/trainer/ppo/ray_trainer.py",
        "trustmed:dx-boost-c: dx boost mask",
        DX_BOOST_C_OLD,
        DX_BOOST_C_NEW,
    ),
    Edit(
        "GiGPO branch kwargs",
        "verl/trainer/ppo/ray_trainer.py",
        "trustmed:dx-boost-c (kwargs)",
        DX_BOOST_D_OLD,
        DX_BOOST_D_NEW,
    ),
    Edit(
        "core_gigpo signature",
        "gigpo/core_gigpo.py",
        "trustmed:dx-boost: per-row bool",
        DX_BOOST_E_OLD,
        DX_BOOST_E_NEW,
    ),
    Edit(
        "core_gigpo boost",
        "gigpo/core_gigpo.py",
        "trustmed:dx-boost: dx boost, post-normalisation",
        DX_BOOST_F_OLD,
        DX_BOOST_F_NEW,
    ),
    Edit(
        "core_algos signature",
        "verl/trainer/ppo/core_algos.py",
        "trustmed:dx-boost: per-row bool",
        DX_BOOST_G_OLD,
        DX_BOOST_G_NEW,
    ),
    Edit(
        "core_algos boost",
        "verl/trainer/ppo/core_algos.py",
        "trustmed:dx-boost: dx boost (GRPO arm",
        DX_BOOST_H_OLD,
        DX_BOOST_H_NEW,
    ),
]

PROMPT_OVERFLOW_A_OLD = (
    "        # Process each sample in parallel\n"
    "        for item in range(batch_size):\n"
    "            # Extract per-sample observations\n"
    "            processed = self.preprocess_single_sample(\n"
    "                item=item,\n"
    "                gen_batch=gen_batch,\n"
    "                obs=obs,\n"
    "            )\n"
    "            processed_samples.append(processed)\n"
)
PROMPT_OVERFLOW_A_NEW = (
    "        # Process each sample in parallel\n"
    "        self._overflow_items = []  # trustmed:prompt-overflow-abort: (item, seq_len) of rows whose prompt exceeds max_prompt_length\n"
    "        for item in range(batch_size):\n"
    "            # Extract per-sample observations\n"
    "            try:\n"
    "                processed = self.preprocess_single_sample(\n"
    "                    item=item,\n"
    "                    gen_batch=gen_batch,\n"
    "                    obs=obs,\n"
    "                )\n"
    "            except (NotImplementedError, RuntimeError) as _e:\n"
    "                # trustmed:prompt-overflow-abort: data.truncation=error refused THIS env's observation. Never\n"
    "                # truncate; keep one row per env (envs.step maps action i -> env i by index) with a\n"
    "                # short placeholder, and let the step loop abort the episode.\n"
    "                _msg = str(_e)\n"
    "                if 'larger than' not in _msg and 'longer than' not in _msg:\n"
    "                    raise\n"
    "                _len = next((int(t) for t in _msg.replace('=', ' ').split() if t.isdigit()), 0)\n"
    "                self._overflow_items.append((int(item), int(_len)))\n"
    "                processed = self.preprocess_single_sample(\n"
    "                    item=item, gen_batch=gen_batch, obs={'text': {item: 'noop'}})\n"
    "            processed_samples.append(processed)\n"
)
PROMPT_OVERFLOW_B_OLD = (
    "            active_masks = np.logical_not(is_done)\n"
    "\n"
    "            batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs)\n"
)
PROMPT_OVERFLOW_B_NEW = (
    "            active_masks = np.logical_not(is_done)\n"
    "\n"
    "            batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs)\n"
    "            # trustmed:prompt-overflow-abort: prompt overflow -> abort the episode, drop its trajectory. The\n"
    "            # placeholder row still goes through generation (one row per env keeps the env/row index\n"
    "            # alignment envs.step relies on); the env is closed BEFORE the step so it pads this and\n"
    "            # every later turn as done; every row the trajectory already produced is de-activated so\n"
    "            # gather_rollout_data drops the whole episode (no terminal happened).\n"
    "            _ovf = list(getattr(self, '_overflow_items', []))\n"
    "            _ovf_max = int(self.config.env.get('prompt_overflow_max_per_step', 3))\n"
    "            if len(_ovf) > _ovf_max:\n"
    "                raise RuntimeError(f'trustmed:prompt-overflow-abort: {len(_ovf)} prompt overflows in one turn (> prompt_overflow_max_per_step='\n"
    "                                   f'{_ovf_max}): {_ovf} -- a data or prompt-growth defect, not a rare big-plate case')\n"
    "            _ovf_events = self.__dict__.setdefault('overflow_events', [])\n"
    "            _ovf_uids = self.__dict__.setdefault('_aborted_traj_uids', set())\n"
    "            if _ovf and bool(self.config.algorithm.filter_groups.get('enable', False)):\n"
    "                # known limitation: dynamic sampling filters groups and counts the\n"
    "                # target batch BEFORE gather_rollout_data drops the aborted episode, so it still occupies a\n"
    "                # group slot there. FILTER_GROUPS is off in every arm; if it is ever on, expect this line.\n"
    "                print(f'[trustmed:prompt-overflow-abort] WARN filter_groups is enabled: {len(_ovf)} aborted episode(s) still count in the DAPO group filter', flush=True)\n"
    "            for _oi, _olen in _ovf:\n"
    "                _abort = getattr(envs, 'abort_one', None)\n"
    "                _rec = _abort(_oi, 'prompt_overflow') if _abort is not None else {}\n"
    "                active_masks[_oi] = False\n"
    "                is_done[_oi] = True\n"
    "                _ovf_uids.add(str(traj_uid[_oi]))\n"
    "                for _row in total_batch_list[_oi]:\n"
    "                    _row['active_masks'] = False\n"
    "                _ev = {'turn': int(_step), 'env': int(_oi), 'seq_len': int(_olen),\n"
    "                       'max_prompt_length': int(self.config.data.max_prompt_length),\n"
    "                       'traj_uid': str(traj_uid[_oi]), 'rows_dropped': int(len(total_batch_list[_oi])) + 1,\n"
    "                       **{k: v for k, v in (_rec or {}).items() if k != 'env'}}\n"
    "                _ovf_events.append(_ev)\n"
    "                print(f'[trustmed:prompt-overflow-abort] prompt overflow, episode aborted and dropped: {_ev}', flush=True)\n"
)
PROMPT_OVERFLOW_C_OLD = (
    "                        gen_batch_output = self.traj_collector.multi_turn_loop(\n"
    "                                                                gen_batch=gen_batch,\n"
    "                                                                actor_rollout_wg=self.actor_rollout_wg,\n"
    "                                                                envs=self.envs,\n"
    "                                                                is_train=True,\n"
    "                                                                )\n"
)
PROMPT_OVERFLOW_C_NEW = (
    PROMPT_OVERFLOW_C_OLD
    + "                        # trustmed:prompt-overflow-abort: prompt-overflow aborts of this step (episodes dropped, never truncated)\n"
    "                        _ovf_ev = list(getattr(self.traj_collector, 'overflow_events', None) or [])\n"
    "                        metrics['rollout/prompt_overflow_aborts'] = float(len(_ovf_ev))\n"
    "                        if _ovf_ev:\n"
    "                            print(f'[trustmed:prompt-overflow-abort] step {self.global_steps}: {len(_ovf_ev)} prompt-overflow abort(s)', flush=True)\n"
    "                            try:\n"
    "                                with open(os.path.join(self.config.trainer.default_local_dir, 'prompt_overflow.jsonl'), 'a', encoding='utf-8') as _f:\n"
    "                                    for _e in _ovf_ev:\n"
    "                                        _f.write(json.dumps({'step': int(self.global_steps), 'split': 'train', **_e}, default=str) + chr(10))\n"
    "                            except Exception as _ex:  # noqa: BLE001 -- forensics must never stop a run\n"
    "                                print(f'[trustmed:prompt-overflow-abort] could not write prompt_overflow.jsonl: {_ex}', flush=True)\n"
    "                        self.traj_collector.overflow_events = []\n"
    "                        self.traj_collector._aborted_traj_uids = set()\n"
)
PROMPT_OVERFLOW_D_OLD = (
    "        for k, v in success_rate.items():\n"
    "            metric_dict[f'val/{k}'] = v\n"
    "\n"
    "        return metric_dict\n"
)
PROMPT_OVERFLOW_D_NEW = (
    "        for k, v in success_rate.items():\n"
    "            metric_dict[f'val/{k}'] = v\n"
    "        # trustmed:prompt-overflow-abort: prompt-overflow aborts during validation (those episodes are dropped from the val set)\n"
    "        _val_ovf = list(getattr(self.traj_collector, 'overflow_events', None) or [])\n"
    "        metric_dict['val/prompt_overflow_aborts'] = float(len(_val_ovf))\n"
    "        if _val_ovf:\n"
    "            print(f'[trustmed:prompt-overflow-abort] validation: {len(_val_ovf)} prompt-overflow abort(s): {_val_ovf}', flush=True)\n"
    "            try:\n"
    "                with open(os.path.join(self.config.trainer.default_local_dir, 'prompt_overflow.jsonl'), 'a', encoding='utf-8') as _f:\n"
    "                    for _e in _val_ovf:\n"
    "                        _f.write(json.dumps({'step': int(self.global_steps), 'split': 'val', **_e}, default=str) + chr(10))\n"
    "            except Exception as _ex:  # noqa: BLE001\n"
    "                print(f'[trustmed:prompt-overflow-abort] could not write prompt_overflow.jsonl: {_ex}', flush=True)\n"
    "        self.traj_collector.overflow_events = []\n"
    "        self.traj_collector._aborted_traj_uids = set()\n"
    "\n"
    "        return metric_dict\n"
)
OVERFLOW_OUTSIDE_SUCCESS_OLD = (
    "        success_rate = {}\n"
    "        for key, value in success.items():\n"
    "            success_rate[key] = np.mean(value)\n"
)
OVERFLOW_OUTSIDE_SUCCESS_NEW = (
    "        success_rate = {}\n"
    "        # trustmed:overflow-outside-success: an aborted (prompt-overflow) episode has no outcome; keep it out of the success means\n"
    "        _ab = getattr(self, '_aborted_traj_uids', None) or set()\n"
    "        _keep = np.array([_b for _b in range(batch_size) if str(traj_uid[_b]) not in _ab], dtype=int)\n"
    "        for key, value in success.items():\n"
    "            _v = np.asarray(value)\n"
    "            success_rate[key] = (float(np.mean(_v[_keep])) if (_ab and len(_v) == batch_size and len(_keep))\n"
    "                                 else np.mean(value))\n"
)

EDITS += [
    Edit(
        "prompt overflow -> placeholder row",
        "agent_system/multi_turn_rollout/rollout_loop.py",
        "trustmed:prompt-overflow-abort: (item, seq_len)",
        PROMPT_OVERFLOW_A_OLD,
        PROMPT_OVERFLOW_A_NEW,
    ),
    Edit(
        "prompt overflow -> abort episode",
        "agent_system/multi_turn_rollout/rollout_loop.py",
        "trustmed:prompt-overflow-abort: prompt overflow -> abort",
        PROMPT_OVERFLOW_B_OLD,
        PROMPT_OVERFLOW_B_NEW,
    ),
    Edit(
        "train metric + ledger",
        "verl/trainer/ppo/ray_trainer.py",
        "trustmed:prompt-overflow-abort: prompt-overflow aborts of this step",
        PROMPT_OVERFLOW_C_OLD,
        PROMPT_OVERFLOW_C_NEW,
    ),
    Edit(
        "val metric + ledger",
        "verl/trainer/ppo/ray_trainer.py",
        "trustmed:prompt-overflow-abort: prompt-overflow aborts during validation",
        PROMPT_OVERFLOW_D_OLD,
        PROMPT_OVERFLOW_D_NEW,
    ),
    Edit(
        "aborted episodes out of success means",
        "agent_system/multi_turn_rollout/rollout_loop.py",
        "trustmed:overflow-outside-success: an aborted",
        OVERFLOW_OUTSIDE_SUCCESS_OLD,
        OVERFLOW_OUTSIDE_SUCCESS_NEW,
    ),
]


def metrics_sidecar_status(fork: Path) -> tuple[str, int]:
    txt = (fork / "verl/trainer/ppo/ray_trainer.py").read_text(encoding="utf-8")
    if METRICS_SIDECAR_MARKER in txt:
        return "already", 0
    n = txt.count(METRICS_SIDECAR_START)
    if n != 1:
        return "pending", n
    start = txt.find(METRICS_SIDECAR_START)
    end = txt.find(METRICS_SIDECAR_END, start)
    return "pending", (1 if end > start else 0)


def apply_metrics_sidecar(fork: Path) -> None:
    p = fork / "verl/trainer/ppo/ray_trainer.py"
    txt = p.read_text(encoding="utf-8")
    start = txt.find(METRICS_SIDECAR_START)
    end = txt.find(METRICS_SIDECAR_END, start) + len(METRICS_SIDECAR_END)
    p.write_text(txt[:start] + METRICS_SIDECAR_NEW + txt[end:], encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fork", required=True, type=Path)
    ap.add_argument("--check", action="store_true", help="report anchor counts, write nothing")
    a = ap.parse_args()
    fork = a.fork.resolve()
    if not (fork / "gigpo/core_gigpo.py").exists():
        raise SystemExit(f"not a verl-agent clone: {fork}")

    plan = []
    bad = []
    for e in EDITS:
        st, n = e.status(fork)
        plan.append((e, st, n))
        print(f"  {e.name:32s} {e.rel:52s} {st}{'' if st == 'already' else f' (anchor x{n})'}")
        if st == "pending" and n != 1:
            bad.append(e.name)
    st8, n8 = metrics_sidecar_status(fork)
    print(
        f"  {'metrics-sidecar-fix per-value serializer':32s} {'verl/trainer/ppo/ray_trainer.py':52s} {st8}{'' if st8 == 'already' else f' (anchor x{n8})'}"
    )
    if st8 == "pending" and n8 != 1:
        bad.append("metrics-sidecar-fix")
    if bad:
        raise SystemExit(f"  ! anchor count != 1 for {bad} -- fork rev drift? nothing written")
    if a.check:
        print("  check only: nothing written")
        return 0

    touched = set()
    for e, st, _ in plan:
        if st in ("already", "skip"):
            continue
        p = fork / e.rel
        txt = p.read_text(encoding="utf-8")
        assert txt.count(e.find) == 1, e.name
        p.write_text(txt.replace(e.find, e.repl, 1), encoding="utf-8")
        touched.add(p)
        print(f"  + patched: {e.name}")
    if st8 == "pending":
        apply_metrics_sidecar(fork)
        touched.add(fork / "verl/trainer/ppo/ray_trainer.py")
        print("  + patched: metrics-sidecar-fix per-value serializer")

    for p in sorted(touched):
        ast.parse(p.read_text(encoding="utf-8"))
        print(f"  ast ok: {p.relative_to(fork)}")
    print("== extra patches done ==" if touched else "== nothing to do (all already applied) ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
