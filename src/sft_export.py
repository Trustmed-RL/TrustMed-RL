"""Export trajectories as SFT pairs (state -> reasoning+action)."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from environment import (
    AssetAliases,
    Config,
    DEFER_VERB,
    _as_list,
    _j,
    _patient_tag,
    apply_patient_turn,
    gate_asset_list,
    load_panel_assets,
    load_profiles,
    new_deck,
    parse_profile,
    render_history_prompt,
    render_workup_prompt,
    render_conclude_prompt,
    surface_defer_verb,
    unalias_map,
)
from prompts import GATE_TEMPLATE

_ID = r"[A-Za-z0-9_.-]+"
ASSET_RE = re.compile(rf"\[image ({_ID}) attached")
RETURN_RE = re.compile(rf"^(.+?) — (.*?): \[image ({_ID}) attached", re.MULTILINE)
RETURN_RE_V2 = re.compile(
    rf"^(.+?) — (.*?): \[(?:image ({_ID}) attached|same image as ({_ID}) above)", re.MULTILINE
)


def returned_lines(obs: str, unalias: dict[str, str] | None = None) -> list[tuple[str, str, str]]:
    """(ordered pair, study name, real asset id) per RECORD an observation
    handed over, in the env's own order.
    """
    rev = unalias or {}
    return [
        (cat, name, rev.get(a1 or a2, a1 or a2))
        for cat, name, a1, a2 in RETURN_RE_V2.findall(obs or "")
    ]


def alias_maps(rec: dict) -> tuple[dict[str, str], dict[str, str]]:
    """(real -> alias, alias -> real) for one trajectory record."""
    aliases = (rec.get("budgets") or {}).get("asset_aliases") or {}
    return dict(aliases), unalias_map(aliases)


def returned_records(obs: str, ep, unalias: dict[str, str] | None = None) -> list:
    """The records one order's observation handed over, in the env's own order."""
    out, taken = [], set()
    for cat, name, aid in returned_lines(obs, unalias):
        for strict in (True, False):
            hit = next(
                (
                    r
                    for r in ep.records
                    if r.rid not in taken
                    and r.asset_id == aid
                    and (
                        not strict
                        or (r.name == name and (f"{r.l1}/{r.category}" == cat or "/" not in cat))
                    )
                ),
                None,
            )
            if hit is not None:
                taken.add(hit.rid)
                out.append(hit)
                break
    return out


def asset_paths_for(rec: dict, row: dict, cfg: Config) -> dict[str, str]:
    """{image_id: the file the DOCTOR was served} for one trajectory record."""
    paths = {
        im["image_id"]: str(cfg.images_dir / im["local_path"])
        for im in _as_list(_j(row.get("images")) or [])
        if im.get("local_path")
    }
    if cfg.panel_assets is not None:
        pmap = load_panel_assets(cfg.panel_assets)["map"]
        for (image_id, _pid), pa in pmap.items():
            if image_id in paths:
                paths[pa["panel_asset_id"]] = str(cfg.panel_assets_dir / pa["crop_path"])
    corr = rec.get("corruption") or {}
    if corr.get("asset_id") and corr.get("swap_path"):
        paths[corr["asset_id"]] = corr["swap_path"]
    return paths


class _NamedAsset:
    """The two fields `gate_asset_list` reads, for the fallback path where the
    replay could not line the gate's ids up with the records the previous order
    returned (a record whose surface name drifted). Never a Record: the id and
    the profile's name are all that survived of it.
    """

    __slots__ = ("asset_id", "name")

    def __init__(self, asset_id: str, name: str):
        self.asset_id, self.name = asset_id, name


class ReplayEnv(AssetAliases):
    """The slice of Env the renderers read back (used / returned_assets /
    released / verified / asset_aliases), rebuilt from the record. Duck-typed on
    purpose: a real Env would re-execute the episode, and the trajectory already
    carries every delta these lines are made of.
    """

    __slots__ = (
        "cfg",
        "used",
        "returned_assets",
        "released",
        "verified",
        "asset_aliases",
        "alias_assign",
    )

    def __init__(self, cfg: Config, aliases: dict[str, str] | None = None):
        self.cfg = cfg
        self.used: set[str] = set()
        self.returned_assets: list = []
        self.released: set[int] = set()
        self.verified: set[str] = set()
        self.asset_aliases: dict[str, str] = dict(aliases or {})
        self.alias_assign = False


def export_episode(
    rec: dict, row: dict, cfg: Config, asset_paths: dict[str, str], include_invalid: bool = False
) -> list[dict]:
    """SFT pairs by default. include_invalid=True is the RL channel: forfeited
    turns become samples too, and every sample carries its reward and anchor —
    GiGPO needs one step per policy call, priced and grouped. The default path
    is untouched, so existing exports stay byte-identical.
    """
    ep = parse_profile(row, cfg)
    deck = new_deck(ep)
    convo: list[dict] = []
    wk_steps: list[dict] = []
    pending_imgs: list[str] = []
    last_returned: list = []
    asks_used = 0
    wk_used = 0
    searches_done = 0
    samples: list[dict] = []
    aliases, unalias = alias_maps(rec)
    renv = ReplayEnv(cfg, aliases)
    assets = {r.asset_id: r for r in ep.records if r.asset_id}

    opening = rec["opening"]
    apply_patient_turn(deck, ep, opening["used_fact_ids"], opening["unknown_topics"])
    convo.append({"role": "patient", "text": opening["answer"]})

    def response_of(turn: dict) -> str:
        think = turn.get("think", "")
        if turn["verb"] == "invalid" and turn.get("raw"):
            return turn["raw"]
        if turn["verb"] == "begin_workup":
            return f"<reasoning>{think}</reasoning>\n<action>begin_workup</action>"
        payload = turn["payload"]
        payload = ", ".join(payload) if isinstance(payload, list) else str(payload)
        verb = turn["verb"]
        if verb == DEFER_VERB:
            verb = surface_defer_verb()
        return f"<reasoning>{think}</reasoning>\n<action>{verb}: {payload}</action>"

    def emit(t: dict, phase: str, prompt: str, images: list[str]) -> None:
        s = {
            "pmcid": rec["meta"]["pmcid"],
            "turn": t["turn"],
            "phase": phase,
            "images": images,
            "prompt": prompt,
            "response": response_of(t),
        }
        if include_invalid:
            s["reward"] = t.get("reward", 0.0)
            s["anchor"] = t.get("anchor")
        samples.append(s)

    names = {a: r.name for a, r in assets.items()}
    last_order = ""
    for t in rec["turns"]:
        phase = t["phase"]
        if phase == "gate":
            g = t.get("gate", {})
            gassets = g.get("assets", [])
            batch = g.get("batch")
            gate_order = last_order or "(previous order)"
            if batch:
                if int(batch.get("k", 1)) >= 2:
                    by_id = {r.asset_id: r for r in ep.records if r.asset_id}
                    delivered = [by_id[a] for a, _n in batch.get("records", []) if a in by_id]
                    wk_steps.append(
                        {
                            "verb": "results_delivered",
                            "payload_text": ", ".join(
                                renv.alias(a) for a, _n in batch.get("records", [])
                            ),
                            "observation_text": batch.get("delivery_obs") or "",
                        }
                    )
                    renv.returned_assets.extend(delivered)
                    last_returned = delivered
                gate_order = ", ".join(batch.get("ordered") or []) or gate_order
                if int(batch.get("K", 1)) > 1:
                    gate_order += f" (batch {int(batch.get('k', 1))} of {int(batch['K'])})"
            per_record = [r.asset_id for r in last_returned] == list(gassets)
            asset_list = gate_asset_list(
                renv.alias,
                last_returned
                if per_record
                else [_NamedAsset(a, names.get(a, "?")) for a in gassets],
            )
            prompt = GATE_TEMPLATE.format(
                patient_tag=_patient_tag(rec["meta"]["pmcid"]),
                demographics=ep.demographics or "(not stated)",
                ordered=gate_order,
                asset_list=asset_list,
                defer_note=(
                    " — a mismatch immediately ends the case: it is escalated to a human clinician"
                ),
            )
            emit(
                t,
                "gate",
                prompt,
                list(dict.fromkeys(asset_paths[a] for a in gassets if a in asset_paths)),
            )
            if g.get("verdict") == "pass":
                renv.verified.update(gassets)
            axes = ",".join(g.get("axes") or [])
            wk_steps.append(
                {
                    "verb": "integrity_check",
                    "payload_text": ", ".join(renv.alias(a) for a in gassets),
                    "observation_text": f"verdict: {g.get('verdict', '')}"
                    + (
                        f" ({axes}: {g.get('reason') or ''})"
                        if g.get("verdict") == "mismatch"
                        else ""
                    ),
                }
            )
            pending_imgs, last_returned = [], []
            continue
        if phase == "history":
            prompt = render_history_prompt(
                cfg, ep, deck, convo, asks_used, wk_used=wk_used, searches=searches_done
            )
        elif phase == "conclude":
            prompt = render_conclude_prompt(
                cfg, ep, deck, wk_steps, searches_done=searches_done, env=renv, wk_used=wk_used
            )
        else:
            prompt = render_workup_prompt(
                cfg,
                ep,
                deck,
                wk_steps,
                cfg.max_workup_turns - wk_used,
                searches_done=searches_done,
                images_pending=len(pending_imgs),
                env=renv,
                wk_used=wk_used,
            )
        images = [asset_paths[a] for a in pending_imgs if a in asset_paths]
        if include_invalid or t["verb"] != "invalid":
            emit(t, phase, prompt, images)
        pending_imgs, last_returned = [], []

        if t["verb"] == "ask":
            asks_used += 1
            convo.append({"role": "doctor", "text": str(t["payload"])})
            p = t["patient"]
            apply_patient_turn(deck, ep, p["used_fact_ids"], p["unknown_topics"])
            convo.append({"role": "patient", "text": p["answer"]})
        elif t["verb"] in ("final_diagnosis", "defer_to_human"):
            break
        elif t["verb"] == "begin_workup":
            pass
        elif t["verb"] == "invalid":
            if phase == "conclude":
                wk_steps.append(
                    {
                        "verb": "invalid",
                        "payload_text": "",
                        "observation_text": t.get(
                            "observation", "Invalid action; the turn was forfeited."
                        ),
                    }
                )
            elif phase == "history" and asks_used < cfg.max_asks:
                asks_used += 1
                convo.append(
                    {
                        "role": "env",
                        "text": t.get("observation", "Invalid action; the turn was forfeited."),
                    }
                )
            else:
                wk_used += 1
                wk_steps.append(
                    {
                        "verb": "invalid",
                        "payload_text": "",
                        "observation_text": t.get(
                            "observation", "Invalid action; the turn was forfeited."
                        ),
                    }
                )
        else:
            wk_used += 1
            payload = t["payload"]
            if t["verb"] == "search":
                searches_done += 1
                renv.used.add(f"search:{str(payload).lower()}")
            if t["verb"] in ("physical_examination", "order_tests"):
                items = payload if isinstance(payload, list) else [str(payload)]
                last_order = ", ".join(items)
                renv.used.update(f"{t['verb']}:{i.lower()}" for i in items)
            rel = t.get("released", [])
            if rel and isinstance(rel[0], dict):
                renv.released.update(x["rid"] for x in rel)
            elif rel:
                ids = set(rel)
                renv.released.update(r.rid for r in renv.returned_assets if r.asset_id in ids)
            obs = t.get("observation", "")
            wk_steps.append(
                {
                    "verb": t["verb"],
                    "payload_text": ", ".join(payload)
                    if isinstance(payload, list)
                    else str(payload),
                    "observation_text": obs,
                }
            )
            pending_imgs = [unalias.get(a, a) for a in ASSET_RE.findall(obs)]
            last_returned = returned_records(obs, ep, unalias)
            renv.returned_assets.extend(last_returned)
    return samples


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trajectories", required=True, type=Path)
    ap.add_argument("--profiles", required=True, type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    recs = [
        json.loads(l) for l in a.trajectories.read_text(encoding="utf-8").splitlines() if l.strip()
    ]
    recs = [r for r in recs if r.get("schema") == "trustmed/consult"]
    rows = {r["pmcid"]: r for r in load_profiles(a.profiles)}

    out_path = a.out or a.trajectories.parent / "sft.jsonl"
    n_samples = 0
    n_img = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for rec in recs:
            pmcid = rec["meta"]["pmcid"]
            row = rows.get(pmcid)
            if row is None:
                print(f"warning: {pmcid} not in profiles, skipped")
                continue
            b = rec["budgets"]
            cfg = Config(
                profiles=a.profiles,
                out=out_path.parent,
                max_asks=b["max_asks"],
                min_asks=b["min_asks"],
                max_workup_turns=b["max_workup_turns"],
                max_items_per_action=b["max_items_per_action"],
                post_baseline=b.get("post_baseline", False),
                identity_anchor=b.get("identity_anchor", False),
                allow_defer=b.get("allow_defer", False),
                integrity_hint=b.get("integrity_hint", True),
                max_searches=b.get("max_searches", 3),
                min_workup_actions=b.get("min_workup_actions", 0),
                deliver_mode=b.get("deliver_mode", "legacy"),
                conclude_mode=b.get("conclude_mode", "conclude_retry"),
                deliver_chunk=b.get("deliver_chunk", 6),
                max_batches_per_order=b.get("max_batches_per_order", 1),
                allow_history_final=b.get("allow_history_final", True),
                search_url="x" if b.get("search_enabled") else None,
                **(
                    {
                        "panel_assets": Path(b["panel_assets"]),
                        "panel_assets_dir": Path(b["panel_assets_dir"]),
                        "panel_policy": (b.get("panel_policy") or "whole_figure"),
                    }
                    if b.get("panel_assets")
                    else {}
                ),
            )
            paths = asset_paths_for(rec, row, cfg)
            for s in export_episode(rec, row, cfg, paths):
                fh.write(json.dumps(s, ensure_ascii=False) + "\n")
                n_samples += 1
                n_img += len(s["images"])
    print(
        f"wrote {out_path}  ({n_samples} samples, {n_img} image attachments, {len(recs)} episodes)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
