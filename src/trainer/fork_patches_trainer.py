#!/usr/bin/env python3
"""GiGPO-side fork patches for langfengQ/verl-agent@20bd331 — apply AFTER patch_fork.py."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

NL = chr(10)

P11_ANCHOR = "            if 'tool_calling' in infos[0]:"
P11_INSERT = (
    "            # Patch 11: row t's action was decided in the state the env\n"
    "            # stamped in THIS step's info; the pre-step obs carried the\n"
    "            # PREVIOUS step's anchor (off-by-one that poisons A^S groups).\n"
    "            batch.non_tensor_batch['anchor_obs'] = np.array(\n"
    "                [((info.get('anchor_obs') if isinstance(info, dict) else None)\n"
    "                  or ('noanchor:%d' % _pi))\n"
    "                 for _pi, info in enumerate(infos)], dtype=object)\n\n"
)

P12_OLD = "                        compute_mean_std_cross_steps: bool = True,"
P12_NEW = (
    "                        compute_mean_std_cross_steps: bool = False,"
    "  # Patch 12: per-trajectory A^E baseline (paper Eq.3); True"
    " length-weights R_i by trajectory length"
)

P10_ANCHOR = (
    "    scores = episode_advantages + step_advantage_w * step_advantages\n"
    "    return scores, scores"
)
P10_NEW = (
    "    scores = episode_advantages + step_advantage_w * step_advantages\n"
    "    try:  # Patch 10: per-step group telemetry for collapse/hacking forensics\n"
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
    '                _g.setdefault("row", []).append(_i)  # Patch 10c: batch row = join key with advantages.jsonl\n'
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
    "        _traj_best = {}  # Patch 12c: one vote per trajectory = its UNPENALISED score\n"
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
P12C_GRPO_OLD = _P12C_GRPO_PREFIX + _P12C_LOOP_OLD
P12C_GRPO_NEW = _P12C_GRPO_PREFIX + _P12C_LOOP_NEW
P12C_GIGPO_OLD = _P12C_LOOP_OLD
P12C_GIGPO_NEW = _P12C_LOOP_NEW

P10C1_OLD = '                _g["as"].append(round(_asv[_i], 4))\n'
P10C1_NEW = (
    '                _g["as"].append(round(_asv[_i], 4))\n'
    '                _g.setdefault("row", []).append(_i)  # Patch 10c: batch row = join key with advantages.jsonl\n'
)
P10C2_OLD = '                            _pd = batch.non_tensor_batch.get("pad_dup")  # Patch 10b'
P10C2_NEW = (
    '                            _pd = batch.non_tensor_batch.get("pad_dup")  # Patch 10b\n'
    '                            _va = batch.non_tensor_batch.get("is_action_valid")  # Patch 10c\n'
    '                            _sr = batch.batch.get("step_rewards") if hasattr(batch.batch, "get") else None'
)
P10C3_OLD = '                                        "pad_dup": (bool(_pd[_i]) if _pd is not None else False)}) + chr(10))'
P10C3_NEW = (
    '                                        "pad_dup": (bool(_pd[_i]) if _pd is not None else False),\n'
    '                                        "row": _i,\n'
    '                                        "valid": (bool(_va[_i]) if _va is not None else True),\n'
    '                                        "step_ret": (float(_sr[_i]) if _sr is not None else None)}) + chr(10))'
)

P13A_OLD = "                    batch = adjust_batch(self.config, batch)"
P13A_NEW = (
    "                    _pre_adjust_bs = len(batch)  # Patch 13a\n"
    "                    batch = adjust_batch(self.config, batch)\n"
    "                    # Patch 13a: adjust_batch pads by CONCATENATING random\n"
    "                    # duplicate rows at the END; stamp them so advantage\n"
    "                    # grouping can exclude them (the flag array permutes\n"
    "                    # together with the batch under balance_batch).\n"
    "                    import numpy as _np13\n"
    "                    _flags = _np13.zeros(len(batch), dtype=bool)\n"
    "                    if len(batch) > _pre_adjust_bs:\n"
    "                        _flags[_pre_adjust_bs:] = True\n"
    "                    batch.non_tensor_batch['pad_dup'] = _flags"
)

P13B_OLD = (
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
P13B_NEW = (
    "    elif adv_estimator == AdvantageEstimator.GiGPO:\n"
    "        # Patch 13b: adjust_batch duplicates (pad_dup) must not sit inside\n"
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

P8_START = "                try:  # Patch 8"
P8_END = "                except Exception:\n                    pass"
P8_NEW = (
    "                try:  # Patch 8: metrics sidecar for offline curves (per-value safe)\n"
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
P8_MARKER = "Patch 8: metrics sidecar for offline curves (per-value safe)"


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


P10B1_OLD = '                            _tuid = batch.non_tensor_batch.get("traj_uid")'
P10B1_NEW = (
    '                            _tuid = batch.non_tensor_batch.get("traj_uid")\n'
    '                            _pd = batch.non_tensor_batch.get("pad_dup")  # Patch 10b\n'
    '                            _va = batch.non_tensor_batch.get("is_action_valid")  # Patch 10c\n'
    '                            _sr = batch.batch.get("step_rewards") if hasattr(batch.batch, "get") else None'
)
P10B2_OLD = '                                        "resp_len": _n}) + chr(10))'
P10B2_NEW = (
    '                                        "resp_len": _n,\n'
    '                                        "pad_dup": (bool(_pd[_i]) if _pd is not None else False),\n'
    '                                        "row": _i,\n'
    '                                        "valid": (bool(_va[_i]) if _va is not None else True),\n'
    '                                        "step_ret": (float(_sr[_i]) if _sr is not None else None)}) + chr(10))'
)

P12B_GRPO_OLD = (
    "    compute_mean_std_cross_steps: bool = True,\n"
    "):\n"
    '    """\n'
    "    Compute advantage for GRPO, operating only on Outcome reward"
)
P12B_GRPO_NEW = (
    "    compute_mean_std_cross_steps: bool = False,  # Patch 12b: per-trajectory baseline (paper Eq.3), same as GiGPO A^E\n"
    "):\n"
    '    """\n'
    "    Compute advantage for GRPO, operating only on Outcome reward"
)
P12B_GIGPO_OLD = "                        compute_mean_std_cross_steps: bool = True,"
P12B_GIGPO_NEW = (
    "                        compute_mean_std_cross_steps: bool = False,"
    "  # Patch 12b: per-trajectory A^E baseline (paper Eq.3), same as the GRPO estimator"
)

P13C_OLD = (
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
P13C_NEW = (
    "        # Patch 13c: adjust_batch duplicates (pad_dup) must not sit inside the\n"
    "        # GRPO group baseline nor train a second copy of their source row --\n"
    "        # same treatment as the GiGPO branch (Patch 13b).\n"
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

P8B_OLD = (
    '                        metrics["actor/kl_loss"] = kl_loss.detach().item()\n'
    '                        metrics["actor/kl_coef"] = self.config.kl_loss_coef'
)
P8B_NEW = (
    '                        append_to_dict(metrics, {"actor/kl_loss": kl_loss.detach().item(),  # Patch 8b: mean over micro-batches, not the last one\n'
    '                                                 "actor/kl_coef": self.config.kl_loss_coef})'
)

P14A_OLD = (
    '        if "multi_modal_inputs" in micro_batch:\n'
    '            for key in micro_batch["multi_modal_inputs"][0].keys():\n'
    '                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)'
)
P14A_NEW = (
    '        if "multi_modal_inputs" in micro_batch:\n'
    "            # Patch 14a: rows may MIX images and text (text rows carry {}, Patch 6b2);\n"
    "            # gather the union of keys over the rows that have them, in row order.\n"
    '            _mm_rows = [r for r in micro_batch["multi_modal_inputs"] if r]\n'
    "            for key in sorted({k for r in _mm_rows for k in r}):\n"
    "                multi_modal_inputs[key] = torch.cat([r[key] for r in _mm_rows if key in r], dim=0)"
)

P17_ANCHOR = "        for test_data in self.val_dataloader:\n"
P17_INSERT = (
    "        # Patch 17: stamp the trainer step into the static val dataset so every\n"
    "        # validation record names the checkpoint it evaluates\n"
    '        _val_ds = getattr(self.val_dataloader, "dataset", None)\n'
    '        if hasattr(_val_ds, "set_step"):\n'
    "            _val_ds.set_step(int(self.global_steps))\n"
)

P18_OLD = '        dataloader_local_path = os.path.join(global_step_folder, "data.pt")\n'
P18_NEW = (
    '        dataloader_local_path = os.path.join(global_step_folder, "data.pt")\n'
    '        if getattr(self.config, "curriculum", None) is not None:   # Patch 18: curriculum sampler resumes itself\n'
    '            print("Patch 18: curriculum sampler installed; data.pt not restored")\n'
    '            dataloader_local_path = os.path.join(global_step_folder, "data.pt.ignored")\n'
)
P20_OLD = (
    "        self.checkpoint_manager.load_checkpoint(local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load)\n"
    "\n"
    "        if self._is_offload_param:\n"
    "            offload_fsdp_model_to_cpu(self.actor_module_fsdp)\n"
)
P20_NEW = (
    "        self.checkpoint_manager.load_checkpoint(local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load)\n"
    "        # Patch 20: the configured lr wins over the lr stored in the optimizer state\n"
    "        _lr = float(self.config.actor.optim.lr)\n"
    "        for _g in self.actor_optimizer.param_groups:\n"
    '            _g["lr"] = _lr\n'
    '        if getattr(self, "actor_lr_scheduler", None) is not None and hasattr(self.actor_lr_scheduler, "base_lrs"):\n'
    "            self.actor_lr_scheduler.base_lrs = [_lr for _ in self.actor_lr_scheduler.base_lrs]\n"
    '        print(f"Patch 20: optimizer lr set to {_lr} after checkpoint load")\n'
    "\n"
    "        if self._is_offload_param:\n"
    "            offload_fsdp_model_to_cpu(self.actor_module_fsdp)\n"
)

P22_OLD = (
    "    # Compute episode relative advantages (Eq. 3 in the paper)."
    + NL
    + "    episode_advantages = episode_norm_reward(token_level_rewards, response_mask, index, traj_index, epsilon, remove_std)"
)
P22_NEW = (
    "    # Compute episode relative advantages (Eq. 3 in the paper)."
    + NL
    + "    import time as _t22; _t22a = _t22.perf_counter()  # Patch 22: stage timers"
    + NL
    + "    episode_advantages = episode_norm_reward(token_level_rewards, response_mask, index, traj_index, epsilon, remove_std)"
    + NL
    + "    _t22b = _t22.perf_counter()"
)
P22B_OLD = "    step_group_uids = build_step_group(anchor_obs, index, enable_similarity, similarity_thresh)"
P22B_NEW = (
    "    step_group_uids = build_step_group(anchor_obs, index, enable_similarity, similarity_thresh)"
    + NL
    + "    _t22c = _t22.perf_counter()  # Patch 22b"
)
P22C_OLD = "    step_advantages = step_norm_reward(step_rewards, response_mask, step_group_uids, epsilon, remove_std)"
P22C_NEW = (
    "    step_advantages = step_norm_reward(step_rewards, response_mask, step_group_uids, epsilon, remove_std)"
    + NL
    + "    _t22d = _t22.perf_counter()  # Patch 22c"
)

_P21_DIV = "scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)"


def _p21_new(tag):
    return (
        "scores[i] = (scores[i] - id2mean[index[i]]) / (torch.clamp(id2std[index[i]], "
        'min=float(__import__("os").environ.get("GIGPO_STD_FLOOR", "0") or 0)) + epsilon)'
        "  # " + tag
    )


P21_GRPO_OLD = "            if norm_adv_by_std_in_grpo:" + NL + "                " + _P21_DIV
P21_GRPO_NEW = (
    "            if norm_adv_by_std_in_grpo:"
    + NL
    + "                "
    + _p21_new("Patch 21: std floor (GRPO)")
)
P21_EP_TAIL = (
    NL
    + "        episode_advantages = scores.unsqueeze(-1).tile([1, response_length]) * response_mask"
)
P21_ST_TAIL = (
    NL + "        step_advantages = scores.unsqueeze(-1).tile([1, response_length]) * response_mask"
)
P21_GIGPO_EP_OLD = "            else:" + NL + "                " + _P21_DIV + P21_EP_TAIL
P21_GIGPO_EP_NEW = (
    "            else:"
    + NL
    + "                "
    + _p21_new("Patch 21a: std floor (A^E)")
    + P21_EP_TAIL
)
P21_GIGPO_ST_OLD = "            else:" + NL + "                " + _P21_DIV + P21_ST_TAIL
P21_GIGPO_ST_NEW = (
    "            else:"
    + NL
    + "                "
    + _p21_new("Patch 21b: std floor (A^S)")
    + P21_ST_TAIL
)

P22D_OLD = '                                "as_nonzero_rows": int(sum(1 for _x in _asv if abs(_x) > 1e-9))}}'
P22D_NEW = (
    '                                "as_nonzero_rows": int(sum(1 for _x in _asv if abs(_x) > 1e-9))},'
    + NL
    + '                     "timing_s": ({"ae": round(_t22b - _t22a, 5), "grouping": round(_t22c - _t22b, 5), "as": round(_t22d - _t22c, 5)} if "_t22d" in dir() else {})}'
)


P26A_OLD = (
    "            if 'is_action_valid' in infos[0]:\n"
    "                batch.non_tensor_batch['is_action_valid'] = np.array([info['is_action_valid'] for info in infos], dtype=bool)\n"
    "            else:\n"
    "                batch.non_tensor_batch['is_action_valid'] = np.ones(batch_size, dtype=bool)\n"
)
P26A_NEW = (
    P26A_OLD
    + "            # Patch 26: terminal-only boost eligibility (full-credit correct dx); same broadcast\n"
    "            batch.non_tensor_batch['dx_boost'] = np.array(\n"
    "                [bool(info.get('dx_boost', False)) for info in infos], dtype=bool)\n"
)


def _p26_mask(tag: str) -> str:
    """the mask block, tagged per branch (26b GRPO / 26c GiGPO) so that each Edit's marker is
    introduced by that Edit alone -- with one shared tag the GiGPO mask read as "already" once
    the GRPO branch carried its copy
    """
    return (
        "        # Patch "
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


P26B_OLD = (
    "        # Call compute_grpo_outcome_advantage with parameters matching its definition\n"
    "        advantages, returns = core_algos.compute_grpo_outcome_advantage(\n"
    '            token_level_rewards=data.batch["token_level_rewards"],\n'
    "            response_mask=grpo_calculation_mask,\n"
)
P26B_NEW = (
    _p26_mask("26b") + P26B_OLD + "            boost_mask=_bm26, boost_scale=_bs26,  # Patch 26b\n"
)

P26C_OLD = (
    "        advantages, returns = core_gigpo.compute_gigpo_outcome_advantage(\n"
    "            token_level_rewards=data.batch['token_level_rewards'], # for episode group reward computing\n"
)
P26C_NEW = (
    _p26_mask("26c") + "        advantages, returns = core_gigpo.compute_gigpo_outcome_advantage(\n"
    "            token_level_rewards=data.batch['token_level_rewards'], # for episode group reward computing\n"
)
P26D_OLD = "            similarity_thresh=gigpo_similarity_thresh,\n            )\n"
P26D_NEW = (
    "            similarity_thresh=gigpo_similarity_thresh,\n"
    "            boost_mask=_bm26, boost_scale=_bs26,  # Patch 26c (kwargs)\n"
    "            )\n"
)

P26E_OLD = (
    "                                   similarity_thresh: float = 0.95,\n"
    "                                   ):\n"
    '    """\n'
    "    Compute the advantages for GiGPO (https://arxiv.org/abs/2505.10978).\n"
)
P26E_NEW = (
    "                                   similarity_thresh: float = 0.95,\n"
    "                                   boost_mask=None,  # Patch 26: per-row bool, dx boost eligibility\n"
    "                                   boost_scale: float = 1.0,\n"
    "                                   ):\n"
    '    """\n'
    "    Compute the advantages for GiGPO (https://arxiv.org/abs/2505.10978).\n"
)
P26F_OLD = (
    "    # Compute joint advantages (Eq. 8 in the paper).\n"
    "    scores = episode_advantages + step_advantage_w * step_advantages\n"
)
P26F_NEW = (
    "    if boost_mask is not None and float(boost_scale) != 1.0:  # Patch 26: dx boost, post-normalisation, A^E only\n"
    "        _bm26 = torch.as_tensor(np.asarray(boost_mask, dtype=bool), device=episode_advantages.device)\n"
    "        _n26 = response_mask.sum(-1)\n"
    "        _ae26 = episode_advantages.sum(-1) / _n26.clamp(min=1)\n"
    "        _jt26 = (episode_advantages + step_advantage_w * step_advantages).sum(-1) / _n26.clamp(min=1)\n"
    "        _bm26 = _bm26 & (_n26 > 0) & (_ae26 > 0) & (_jt26 > 0)\n"
    "        episode_advantages = torch.where(_bm26.unsqueeze(-1), episode_advantages * float(boost_scale), episode_advantages)\n"
    "    # Compute joint advantages (Eq. 8 in the paper).\n"
    "    scores = episode_advantages + step_advantage_w * step_advantages\n"
)

P26G_OLD = '):\n    """\n    Compute advantage for GRPO, operating only on Outcome reward\n'
P26G_NEW = (
    "    boost_mask=None,  # Patch 26: per-row bool, dx boost eligibility\n"
    "    boost_scale: float = 1.0,\n" + P26G_OLD
)
P26H_OLD = (
    "            else:\n"
    "                scores[i] = scores[i] - id2mean[index[i]]\n"
    "        scores = scores.unsqueeze(-1) * response_mask\n"
    "\n"
    "    return scores, scores\n"
)
P26H_NEW = (
    "            else:\n"
    "                scores[i] = scores[i] - id2mean[index[i]]\n"
    "        if boost_mask is not None and float(boost_scale) != 1.0:  # Patch 26: dx boost (GRPO arm: A = A^E)\n"
    "            _bm26 = torch.as_tensor(np.asarray(boost_mask, dtype=bool), device=scores.device)\n"
    "            _bm26 = _bm26 & (response_mask.sum(-1) > 0) & (scores > 0)\n"
    "            scores = torch.where(_bm26, scores * float(boost_scale), scores)\n"
    "        scores = scores.unsqueeze(-1) * response_mask\n"
    "\n"
    "    return scores, scores\n"
)

P24A_OLD = (
    "            if 'is_action_valid' in infos[0]:\n"
    "                batch.non_tensor_batch['is_action_valid'] = np.array([info['is_action_valid'] for info in infos], dtype=bool)\n"
    "            else:\n"
    "                batch.non_tensor_batch['is_action_valid'] = np.ones(batch_size, dtype=bool)\n"
)
P24A_NEW = (
    P24A_OLD + "\n"
    "            # Patch 24: terminal-only flag; ray_trainer broadcasts it over the trajectory\n"
    "            batch.non_tensor_batch['dx_correct'] = np.array(\n"
    "                [bool(info.get('dx_correct', False)) for info in infos], dtype=bool)\n"
)

P24B_OLD = (
    "                            gigpo_similarity_thresh=self.config.algorithm.gigpo.similarity_thresh,\n"
    "                        )\n"
)
P24B_NEW = (
    P24B_OLD + "\n"
    "                        # Patch 24b: the env sets dx_correct on the terminal row only; every row of\n"
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

P24C_OLD = "from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, compute_policy_loss_gspo, kl_penalty\n"
P24C_NEW = (
    P24C_OLD
    + "try:                                   # Patch 24/25: sp_consult actor instrumentation\n"
    "    from agent_system.environments.env_package.sp_consult import actor_metrics as _sp_am\n"
    "except Exception as _e:                # never let instrumentation stop a run\n"
    "    print('WARN sp_consult actor_metrics unavailable:', repr(_e)[:120])\n"
    "    _sp_am = None\n"
)

P24D_OLD = (
    '        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]\n'
    "        if multi_turn:\n"
)
P24D_NEW = (
    '        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]\n'
    '        if "dx_correct" in data.batch.keys():          # Patch 24d\n'
    '            select_keys.append("dx_correct")\n'
    "        if multi_turn:\n"
)

P24E_OLD = (
    "                    data = {\n"
    '                        "actor/pg_loss": pg_loss.detach().item(),\n'
)
P24E_NEW = (
    "                    # Patch 24e: clipping split by sign and by correct diagnosis\n"
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
    '                            print("WARN Patch 24e:", repr(_e)[:160]); _sp_extra = {}\n'
    + P24E_OLD
)

P24F_OLD = (
    "                    }\n"
    "                    append_to_dict(metrics, data)\n"
    "\n"
    "                grad_norm = self._optimizer_step()\n"
    '                data = {"actor/grad_norm": grad_norm.detach().item()}\n'
    "                append_to_dict(metrics, data)\n"
)
P24F_NEW = (
    "                    }\n"
    "                    data.update(_sp_extra)                      # Patch 24f\n"
    "                    append_to_dict(metrics, data)\n"
    "\n"
    "                grad_norm = self._optimizer_step()\n"
    '                data = {"actor/grad_norm": grad_norm.detach().item()}\n'
    '                data.update(getattr(self, "_sp_step_metrics", None) or {})   # Patch 25\n'
    "                append_to_dict(metrics, data)\n"
)

P25_OLD = (
    "        # if grad_norm is not finite, skip the update\n"
    "        if not torch.isfinite(grad_norm):\n"
    '            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")\n'
    "            self.actor_optimizer.zero_grad()\n"
    "        else:\n"
    "            self.actor_optimizer.step()\n"
    "        return grad_norm\n"
)
P25_NEW = (
    "        # Patch 25: pre-clip norm, the scale clipping applied, and the real parameter delta\n"
    "        self._sp_step_metrics = {}\n"
    "        if _sp_am is not None:\n"
    "            try:\n"
    "                self._sp_step_metrics = _sp_am.clip_metrics(grad_norm, self.config.grad_clip)\n"
    "                if getattr(self, '_sp_probe', None) is None:\n"
    "                    self._sp_probe = _sp_am.ParamDeltaProbe(self.actor_module.parameters())\n"
    "                self._sp_probe.snapshot()\n"
    "            except Exception as _e:\n"
    "                print('WARN Patch 25 snapshot:', repr(_e)[:160]); self._sp_probe = None\n"
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
    "                    print('WARN Patch 25 delta:', repr(_e)[:160])\n"
    "        return grad_norm\n"
)


EDITS = [
    Edit(
        "P18 no data.pt under the curriculum sampler",
        "verl/trainer/ppo/ray_trainer.py",
        "Patch 18",
        P18_OLD,
        P18_NEW,
    ),
    Edit("P22 stage timer A^E", "gigpo/core_gigpo.py", "Patch 22: stage timers", P22_OLD, P22_NEW),
    Edit(
        "P22d retrofit timing into old P10",
        "gigpo/core_gigpo.py",
        "timing_s",
        P22D_OLD,
        P22D_NEW,
        optional=True,
    ),
    Edit("P22b stage timer grouping", "gigpo/core_gigpo.py", "Patch 22b", P22B_OLD, P22B_NEW),
    Edit("P22c stage timer A^S", "gigpo/core_gigpo.py", "Patch 22c", P22C_OLD, P22C_NEW),
    Edit(
        "P21 GRPO std floor",
        "verl/trainer/ppo/core_algos.py",
        "Patch 21: std floor (GRPO)",
        P21_GRPO_OLD,
        P21_GRPO_NEW,
    ),
    Edit(
        "P21a GiGPO A^E std floor",
        "gigpo/core_gigpo.py",
        "Patch 21a",
        P21_GIGPO_EP_OLD,
        P21_GIGPO_EP_NEW,
    ),
    Edit(
        "P21b GiGPO A^S std floor",
        "gigpo/core_gigpo.py",
        "Patch 21b",
        P21_GIGPO_ST_OLD,
        P21_GIGPO_ST_NEW,
    ),
    Edit(
        "P20 lr re-applied after load", "verl/workers/fsdp_workers.py", "Patch 20", P20_OLD, P20_NEW
    ),
    Edit(
        "P17 val step stamp",
        "verl/trainer/ppo/ray_trainer.py",
        "Patch 17",
        P17_ANCHOR,
        P17_INSERT + P17_ANCHOR,
    ),
    Edit(
        "P11 anchor alignment",
        "agent_system/multi_turn_rollout/rollout_loop.py",
        "Patch 11",
        P11_ANCHOR,
        P11_INSERT + P11_ANCHOR,
    ),
    Edit(
        "P14a mixed-row mm gather",
        "verl/workers/actor/dp_actor.py",
        "Patch 14a",
        P14A_OLD,
        P14A_NEW,
    ),
    Edit(
        "P12b GRPO per-trajectory baseline",
        "verl/trainer/ppo/core_algos.py",
        "Patch 12b",
        P12B_GRPO_OLD,
        P12B_GRPO_NEW,
    ),
    Edit(
        "P12b GiGPO per-trajectory A^E",
        "gigpo/core_gigpo.py",
        "Patch 12b",
        P12B_GIGPO_OLD,
        P12B_GIGPO_NEW,
    ),
    Edit(
        "P13c GRPO pad-dup masking",
        "verl/trainer/ppo/ray_trainer.py",
        "Patch 13c",
        P13C_OLD,
        P13C_NEW,
    ),
    Edit(
        "P8b kl_loss mean over micro-batches",
        "verl/workers/actor/dp_actor.py",
        "Patch 8b",
        P8B_OLD,
        P8B_NEW,
    ),
    Edit("P10 group telemetry", "gigpo/core_gigpo.py", "Patch 10", P10_ANCHOR, P10_NEW),
    Edit("P13a stamp pad_dup", "verl/trainer/ppo/ray_trainer.py", "Patch 13a", P13A_OLD, P13A_NEW),
    Edit(
        "P13b dups out of GiGPO groups",
        "verl/trainer/ppo/ray_trainer.py",
        "Patch 13b",
        P13B_OLD,
        P13B_NEW,
    ),
    Edit(
        "P10b ledger pad_dup (1/2)",
        "verl/trainer/ppo/ray_trainer.py",
        "Patch 10b",
        P10B1_OLD,
        P10B1_NEW,
    ),
    Edit(
        "P10b ledger pad_dup (2/2)",
        "verl/trainer/ppo/ray_trainer.py",
        '"pad_dup": (bool(_pd[_i])',
        P10B2_OLD,
        P10B2_NEW,
    ),
    Edit(
        "P12c GRPO unpenalised group vote",
        "verl/trainer/ppo/core_algos.py",
        "Patch 12c",
        P12C_GRPO_OLD,
        P12C_GRPO_NEW,
    ),
    Edit(
        "P12c GiGPO unpenalised group vote",
        "gigpo/core_gigpo.py",
        "Patch 12c",
        P12C_GIGPO_OLD,
        P12C_GIGPO_NEW,
    ),
    Edit(
        "P10c groups ledger row key",
        "gigpo/core_gigpo.py",
        "Patch 10c",
        P10C1_OLD,
        P10C1_NEW,
        optional=True,
    ),
    Edit(
        "P10c advantages ledger valid (1/2)",
        "verl/trainer/ppo/ray_trainer.py",
        "Patch 10c",
        P10C2_OLD,
        P10C2_NEW,
        optional=True,
    ),
    Edit(
        "P10c advantages ledger valid (2/2)",
        "verl/trainer/ppo/ray_trainer.py",
        '"valid": (bool(_va[_i])',
        P10C3_OLD,
        P10C3_NEW,
        optional=True,
    ),
    Edit(
        "P24a rollout dx_correct",
        "agent_system/multi_turn_rollout/rollout_loop.py",
        "Patch 24",
        P24A_OLD,
        P24A_NEW,
    ),
    Edit(
        "P24b broadcast dx_correct",
        "verl/trainer/ppo/ray_trainer.py",
        "Patch 24b",
        P24B_OLD,
        P24B_NEW,
    ),
    Edit(
        "P24c dp_actor import", "verl/workers/actor/dp_actor.py", "Patch 24/25", P24C_OLD, P24C_NEW
    ),
    Edit(
        "P24d dp_actor select_keys",
        "verl/workers/actor/dp_actor.py",
        "Patch 24d",
        P24D_OLD,
        P24D_NEW,
    ),
    Edit(
        "P24e dp_actor clip metrics",
        "verl/workers/actor/dp_actor.py",
        "Patch 24e",
        P24E_OLD,
        P24E_NEW,
    ),
    Edit(
        "P24f/P25 metric merge", "verl/workers/actor/dp_actor.py", "Patch 24f", P24F_OLD, P24F_NEW
    ),
    Edit(
        "P25 optimizer step accounting",
        "verl/workers/actor/dp_actor.py",
        "Patch 25",
        P25_OLD,
        P25_NEW,
    ),
    Edit(
        "P26a rollout dx_boost",
        "agent_system/multi_turn_rollout/rollout_loop.py",
        "Patch 26",
        P26A_OLD,
        P26A_NEW,
    ),
    Edit(
        "P26b GRPO branch mask + kwargs",
        "verl/trainer/ppo/ray_trainer.py",
        "Patch 26b",
        P26B_OLD,
        P26B_NEW,
    ),
    Edit(
        "P26c GiGPO branch mask",
        "verl/trainer/ppo/ray_trainer.py",
        "Patch 26c: dx boost mask",
        P26C_OLD,
        P26C_NEW,
    ),
    Edit(
        "P26c GiGPO branch kwargs",
        "verl/trainer/ppo/ray_trainer.py",
        "Patch 26c (kwargs)",
        P26D_OLD,
        P26D_NEW,
    ),
    Edit(
        "P26 core_gigpo signature",
        "gigpo/core_gigpo.py",
        "Patch 26: per-row bool",
        P26E_OLD,
        P26E_NEW,
    ),
    Edit(
        "P26 core_gigpo boost",
        "gigpo/core_gigpo.py",
        "Patch 26: dx boost, post-normalisation",
        P26F_OLD,
        P26F_NEW,
    ),
    Edit(
        "P26 core_algos signature",
        "verl/trainer/ppo/core_algos.py",
        "Patch 26: per-row bool",
        P26G_OLD,
        P26G_NEW,
    ),
    Edit(
        "P26 core_algos boost",
        "verl/trainer/ppo/core_algos.py",
        "Patch 26: dx boost (GRPO arm",
        P26H_OLD,
        P26H_NEW,
    ),
]

P28A_OLD = (
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
P28A_NEW = (
    "        # Process each sample in parallel\n"
    "        self._overflow_items = []  # Patch 28: (item, seq_len) of rows whose prompt exceeds max_prompt_length\n"
    "        for item in range(batch_size):\n"
    "            # Extract per-sample observations\n"
    "            try:\n"
    "                processed = self.preprocess_single_sample(\n"
    "                    item=item,\n"
    "                    gen_batch=gen_batch,\n"
    "                    obs=obs,\n"
    "                )\n"
    "            except (NotImplementedError, RuntimeError) as _e:\n"
    "                # Patch 28: data.truncation=error refused THIS env's observation. Never\n"
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
P28B_OLD = (
    "            active_masks = np.logical_not(is_done)\n"
    "\n"
    "            batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs)\n"
)
P28B_NEW = (
    "            active_masks = np.logical_not(is_done)\n"
    "\n"
    "            batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs)\n"
    "            # Patch 28: prompt overflow -> abort the episode, drop its trajectory. The\n"
    "            # placeholder row still goes through generation (one row per env keeps the env/row index\n"
    "            # alignment envs.step relies on); the env is closed BEFORE the step so it pads this and\n"
    "            # every later turn as done; every row the trajectory already produced is de-activated so\n"
    "            # gather_rollout_data drops the whole episode (no terminal happened).\n"
    "            _ovf = list(getattr(self, '_overflow_items', []))\n"
    "            _ovf_max = int(self.config.env.get('prompt_overflow_max_per_step', 3))\n"
    "            if len(_ovf) > _ovf_max:\n"
    "                raise RuntimeError(f'Patch 28: {len(_ovf)} prompt overflows in one turn (> prompt_overflow_max_per_step='\n"
    "                                   f'{_ovf_max}): {_ovf} -- a data or prompt-growth defect, not a rare big-plate case')\n"
    "            _ovf_events = self.__dict__.setdefault('overflow_events', [])\n"
    "            _ovf_uids = self.__dict__.setdefault('_aborted_traj_uids', set())\n"
    "            if _ovf and bool(self.config.algorithm.filter_groups.get('enable', False)):\n"
    "                # known limitation: dynamic sampling filters groups and counts the\n"
    "                # target batch BEFORE gather_rollout_data drops the aborted episode, so it still occupies a\n"
    "                # group slot there. FILTER_GROUPS is off in every arm; if it is ever on, expect this line.\n"
    "                print(f'[Patch 28] WARN filter_groups is enabled: {len(_ovf)} aborted episode(s) still count in the DAPO group filter', flush=True)\n"
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
    "                print(f'[Patch 28] prompt overflow, episode aborted and dropped: {_ev}', flush=True)\n"
)
P28C_OLD = (
    "                        gen_batch_output = self.traj_collector.multi_turn_loop(\n"
    "                                                                gen_batch=gen_batch,\n"
    "                                                                actor_rollout_wg=self.actor_rollout_wg,\n"
    "                                                                envs=self.envs,\n"
    "                                                                is_train=True,\n"
    "                                                                )\n"
)
P28C_NEW = (
    P28C_OLD
    + "                        # Patch 28: prompt-overflow aborts of this step (episodes dropped, never truncated)\n"
    "                        _ovf_ev = list(getattr(self.traj_collector, 'overflow_events', None) or [])\n"
    "                        metrics['rollout/prompt_overflow_aborts'] = float(len(_ovf_ev))\n"
    "                        if _ovf_ev:\n"
    "                            print(f'[Patch 28] step {self.global_steps}: {len(_ovf_ev)} prompt-overflow abort(s)', flush=True)\n"
    "                            try:\n"
    "                                with open(os.path.join(self.config.trainer.default_local_dir, 'prompt_overflow.jsonl'), 'a', encoding='utf-8') as _f:\n"
    "                                    for _e in _ovf_ev:\n"
    "                                        _f.write(json.dumps({'step': int(self.global_steps), 'split': 'train', **_e}, default=str) + chr(10))\n"
    "                            except Exception as _ex:  # noqa: BLE001 -- forensics must never stop a run\n"
    "                                print(f'[Patch 28] could not write prompt_overflow.jsonl: {_ex}', flush=True)\n"
    "                        self.traj_collector.overflow_events = []\n"
    "                        self.traj_collector._aborted_traj_uids = set()\n"
)
P28D_OLD = (
    "        for k, v in success_rate.items():\n"
    "            metric_dict[f'val/{k}'] = v\n"
    "\n"
    "        return metric_dict\n"
)
P28D_NEW = (
    "        for k, v in success_rate.items():\n"
    "            metric_dict[f'val/{k}'] = v\n"
    "        # Patch 28: prompt-overflow aborts during validation (those episodes are dropped from the val set)\n"
    "        _val_ovf = list(getattr(self.traj_collector, 'overflow_events', None) or [])\n"
    "        metric_dict['val/prompt_overflow_aborts'] = float(len(_val_ovf))\n"
    "        if _val_ovf:\n"
    "            print(f'[Patch 28] validation: {len(_val_ovf)} prompt-overflow abort(s): {_val_ovf}', flush=True)\n"
    "            try:\n"
    "                with open(os.path.join(self.config.trainer.default_local_dir, 'prompt_overflow.jsonl'), 'a', encoding='utf-8') as _f:\n"
    "                    for _e in _val_ovf:\n"
    "                        _f.write(json.dumps({'step': int(self.global_steps), 'split': 'val', **_e}, default=str) + chr(10))\n"
    "            except Exception as _ex:  # noqa: BLE001\n"
    "                print(f'[Patch 28] could not write prompt_overflow.jsonl: {_ex}', flush=True)\n"
    "        self.traj_collector.overflow_events = []\n"
    "        self.traj_collector._aborted_traj_uids = set()\n"
    "\n"
    "        return metric_dict\n"
)
P28E_OLD = (
    "        success_rate = {}\n"
    "        for key, value in success.items():\n"
    "            success_rate[key] = np.mean(value)\n"
)
P28E_NEW = (
    "        success_rate = {}\n"
    "        # Patch 28e: an aborted (prompt-overflow) episode has no outcome; keep it out of the success means\n"
    "        _ab = getattr(self, '_aborted_traj_uids', None) or set()\n"
    "        _keep = np.array([_b for _b in range(batch_size) if str(traj_uid[_b]) not in _ab], dtype=int)\n"
    "        for key, value in success.items():\n"
    "            _v = np.asarray(value)\n"
    "            success_rate[key] = (float(np.mean(_v[_keep])) if (_ab and len(_v) == batch_size and len(_keep))\n"
    "                                 else np.mean(value))\n"
)

EDITS += [
    Edit(
        "P28a prompt overflow -> placeholder row",
        "agent_system/multi_turn_rollout/rollout_loop.py",
        "Patch 28: (item, seq_len)",
        P28A_OLD,
        P28A_NEW,
    ),
    Edit(
        "P28b prompt overflow -> abort episode",
        "agent_system/multi_turn_rollout/rollout_loop.py",
        "Patch 28: prompt overflow -> abort",
        P28B_OLD,
        P28B_NEW,
    ),
    Edit(
        "P28c train metric + ledger",
        "verl/trainer/ppo/ray_trainer.py",
        "Patch 28: prompt-overflow aborts of this step",
        P28C_OLD,
        P28C_NEW,
    ),
    Edit(
        "P28d val metric + ledger",
        "verl/trainer/ppo/ray_trainer.py",
        "Patch 28: prompt-overflow aborts during validation",
        P28D_OLD,
        P28D_NEW,
    ),
    Edit(
        "P28e aborted episodes out of success means",
        "agent_system/multi_turn_rollout/rollout_loop.py",
        "Patch 28e: an aborted",
        P28E_OLD,
        P28E_NEW,
    ),
]


def p8_status(fork: Path) -> tuple[str, int]:
    txt = (fork / "verl/trainer/ppo/ray_trainer.py").read_text(encoding="utf-8")
    if P8_MARKER in txt:
        return "already", 0
    n = txt.count(P8_START)
    if n != 1:
        return "pending", n
    start = txt.find(P8_START)
    end = txt.find(P8_END, start)
    return "pending", (1 if end > start else 0)


def apply_p8(fork: Path) -> None:
    p = fork / "verl/trainer/ppo/ray_trainer.py"
    txt = p.read_text(encoding="utf-8")
    start = txt.find(P8_START)
    end = txt.find(P8_END, start) + len(P8_END)
    p.write_text(txt[:start] + P8_NEW + txt[end:], encoding="utf-8")


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
    st8, n8 = p8_status(fork)
    print(
        f"  {'P8fix per-value serializer':32s} {'verl/trainer/ppo/ray_trainer.py':52s} {st8}{'' if st8 == 'already' else f' (anchor x{n8})'}"
    )
    if st8 == "pending" and n8 != 1:
        bad.append("P8fix")
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
        apply_p8(fork)
        touched.add(fork / "verl/trainer/ppo/ray_trainer.py")
        print("  + patched: P8fix per-value serializer")

    for p in sorted(touched):
        ast.parse(p.read_text(encoding="utf-8"))
        print(f"  ast ok: {p.relative_to(fork)}")
    print("== extra patches done ==" if touched else "== nothing to do (all already applied) ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
