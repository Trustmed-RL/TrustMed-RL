"""Export scored sp_consult/v3 trajectories as GiGPO step records."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path
from sft_export import asset_paths_for, export_episode
from outcomes import claim_specificity, classify_outcome, waved_through_corrupt
from environment import Config, load_profiles

_DEFAULTS = {f.name: f.default for f in dataclasses.fields(Config)}

REWARDS = {
    "mismatch": 0.9,
    "mismatch_partial": 0.65,
    "ook": 0.4,
    "blind_cap": 0.4,
    "coverage_floor": 0.10,
    "coverage_power": 0.5,
    "false_mismatch": -0.10,
    "false_ook": -0.05,
    "zero_workup_wrong": -0.05,
    "format_failure": -0.10,
    "waved_dx": -0.10,
}
BLIND_CAP_MODES = ("legacy", "incomplete_coverage")


def contract_name(r: dict) -> str:
    """The three prices that make a board a contract, stamped per record as
    `metrics.reward_contract`, e.g. `fm-0.10_fo-0.05_zw-0.05_ff-0.10_wv-0.10`.
    """
    return (
        f"fm{r['false_mismatch']:+.2f}_fo{r['false_ook']:+.2f}"
        f"_zw{r['zero_workup_wrong']:+.2f}"
        f"_ff{r['format_failure']:+.2f}_wv{r['waved_dx']:+.2f}"
    )


def validate_rewards(r: dict, strict_cap: bool = False) -> dict:
    """Complete `r` from the board and check the ordering it relies on:
    ook < mismatch_partial < mismatch <= 1.0 (abstaining never beats looking,
    naming the axes pays), 0 <= blind_cap <= mismatch_partial (and <= ook on
    the live path, where it follows the scheduled r_ook), -1 <= false_* <= 0
    (0 = no penalty), 0 <= coverage_floor < ook, coverage_power > 0.
    """
    r = {**REWARDS, **r}
    mode = r.get("blind_cap_mode")
    if mode is not None and mode not in BLIND_CAP_MODES:
        raise ValueError(f"blind_cap_mode must be one of {BLIND_CAP_MODES}, got {mode!r}")
    if not (r["ook"] < r["mismatch_partial"] < r["mismatch"] <= 1.0):
        raise ValueError(
            f"need ook < mismatch_partial < mismatch <= 1.0, got ook={r['ook']} "
            f"partial={r['mismatch_partial']} mismatch={r['mismatch']}"
        )
    if not 0.0 <= r["blind_cap"] <= r["mismatch_partial"]:
        raise ValueError(
            f"need 0 <= blind_cap <= mismatch_partial, got cap={r['blind_cap']} "
            f"partial={r['mismatch_partial']}"
        )
    if strict_cap and r["blind_cap"] > r["ook"]:
        raise ValueError(
            f"live rewards need blind_cap <= ook, got cap={r['blind_cap']} ook={r['ook']}"
        )
    for k in ("false_mismatch", "false_ook", "zero_workup_wrong", "format_failure", "waved_dx"):
        if not -1.0 <= float(r[k]) <= 0.0:
            raise ValueError(f"need -1 <= {k} <= 0, got {r[k]}")
    f = r.get("coverage_floor")
    if f is not None and not 0.0 <= f < r["ook"]:
        raise ValueError(f"need 0 <= coverage_floor < ook, got {f} (ook={r['ook']})")
    if not float(r["coverage_power"]) > 0.0:
        raise ValueError(f"need coverage_power > 0, got {r['coverage_power']}")
    return r


def restore_panel_policy(cfg: Config, budgets: dict) -> Config:
    """A record collected under crops_only must be rebuilt under crops_only:
    the served evidence (prompts, images, coverage) differs from whole-figure
    serving.
    """
    pp = (budgets or {}).get("panel_policy")
    return dataclasses.replace(cfg, panel_policy=str(pp)) if pp else cfg


def matrix_for_record(rec: dict, overrides: dict | None = None) -> dict:
    """The matrix a stored record is priced with: what the live env stamped at
    done (`metrics.reward_matrix`: that step's scheduled r_ook / blind_cap, the
    run's floor and prices), explicit overrides on top, the board for an
    unstamped record. A stamp from before the prices were split carried one
    `false_exit` for every false exit, or none at all (= 0); one from before
    `coverage_power` priced coverage linearly; one from before item C
    (first32, v6a) paid its zero-workup wrong diagnoses the free 0.
    """
    m = dict((rec.get("metrics") or {}).get("reward_matrix") or {})
    if m:
        legacy = m.pop("false_exit", 0.0)
        m.setdefault("false_mismatch", legacy)
        m.setdefault("false_ook", legacy)
        m.setdefault("coverage_power", 1.0)
        m.setdefault("zero_workup_wrong", 0.0)
        m.setdefault("format_failure", 0.0)
        m.setdefault("waved_dx", 0.0)
    m.update(overrides or {})
    return validate_rewards(m)


def replay_terminal(rec: dict, matrix: dict) -> float:
    """The terminal scalar for an offline replay of `rec` under `matrix`,
    re-priced through terminal_reward -- except an abstention record with no
    meta.task_kind (collected before the task kind was stamped), which keeps
    the scalar it earned live rather than being priced under a guessed kind.
    """
    mt = rec.get("metrics") or {}
    kind = mt.get("outcome_kind") or classify_outcome(rec)[0]
    if (
        kind == "defer_ook"
        and (rec.get("meta") or {}).get("task_kind") is None
        and mt.get("terminal_reward") is not None
    ):
        return float(mt["terminal_reward"])
    return terminal_reward(rec, matrix)


def cli_reward_overrides(a) -> dict:
    """Only the reward flags the caller actually passed (argparse defaults are
    None), so a live record's own matrix is never clobbered by CLI defaults.
    """
    pairs = (
        ("mismatch", a.r_mismatch),
        ("mismatch_partial", a.r_mismatch_partial),
        ("ook", a.r_ook),
        ("blind_cap", a.blind_dx_cap),
        ("coverage_floor", a.coverage_floor),
        ("coverage_power", getattr(a, "coverage_power", None)),
        ("false_mismatch", getattr(a, "false_mismatch", None)),
        ("false_ook", getattr(a, "false_ook", None)),
        ("zero_workup_wrong", getattr(a, "zero_workup_wrong", None)),
        ("format_failure", getattr(a, "format_failure", None)),
        ("waved_dx", getattr(a, "waved_dx", None)),
    )
    out = {k: float(v) for k, v in pairs if v is not None}
    mode = getattr(a, "blind_cap_mode", None)
    if mode is not None:
        out["blind_cap_mode"] = str(mode)
    return out


def evidence_coverage(rec: dict) -> float:
    """Share of the case's image-bearing evidence this episode actually READ:
    released readings over owed readings (`records_expected`, record-level);
    older records fall back to department coverage; a record with no
    expectation at all (text-only case, pre-coverage record) is 1.0 --
    unmeasured evidence is never punished.
    """
    mt = rec.get("metrics") or {}
    n = mt.get("records_expected")
    if n:
        return min(1.0, float(mt.get("reports_released", 0)) / float(n))
    expected = mt.get("consults_expected")
    if expected:
        done = set(mt.get("consults_done") or [])
        return len(done & set(expected)) / len(expected)
    return 1.0


def _fully_read_image_arm(rec: dict, r: dict) -> bool:
    """blind_cap_mode incomplete_coverage: an armed episode whose measured
    record-level coverage is exactly 1.0 met every served study and still
    never met the swap -- the env's dud, exempt from the cap.
    """
    mt = rec["metrics"]
    return (
        r.get("blind_cap_mode") == "incomplete_coverage"
        and int(mt.get("records_expected") or 0) > 0
        and evidence_coverage(rec) >= 1.0
    )


DX_CREDIT_BRANCHES = (
    "not_dx",
    "zero_workup",
    "voided_text",
    "voided_waved",
    "blind_cap",
    "wrong",
    "full",
)


def dx_credit_branch(rec: dict, r: dict) -> str:
    """Which branch of `terminal_reward` a diagnosis record takes, by name, using the same predicates in the same order so the trainer-side dx boost (DX_ADV_SCALE) can ask whether a diagnosis kept its full credit without re-deriving the board. Branches: not_dx, zero_workup, voided_text, voided_waved, blind_cap, wrong, full."""
    mt = rec["metrics"]
    kind = mt.get("outcome_kind")
    if kind is None:
        kind = classify_outcome(rec)[0]
    if kind != "dx":
        return "not_dx"
    corr = rec.get("corruption") or {}
    dx = float(mt.get("outcome_reward") or 0.0)
    if dx == 0.0 and int(mt.get("workup_turns") or 0) == 0:
        return "zero_workup"
    if corr.get("channel") == "text":
        return "voided_text"
    if mt.get("corruption_exposed") and waved_through_corrupt(rec):
        return "voided_waved"
    if mt.get("corruption_armed") and not corr and not _fully_read_image_arm(rec, r):
        return "blind_cap"
    return "wrong" if dx <= 0.0 else "full"


def terminal_reward(rec: dict, r: dict) -> float:
    """One bounded terminal scalar for an episode, by how it ended
    (`metrics.outcome_kind`; derived from the transcript when absent).
    """
    mt = rec["metrics"]
    kind, sub = mt.get("outcome_kind"), mt.get("outcome_subkind")
    if kind is None:
        kind, sub = classify_outcome(rec)
    corr = rec.get("corruption") or {}
    if kind == "dx":
        if float(mt.get("outcome_reward") or 0.0) == 0.0 and int(mt.get("workup_turns") or 0) == 0:
            return r["zero_workup_wrong"]
        if corr.get("channel") == "text":
            return 0.0
        if mt.get("corruption_exposed") and waved_through_corrupt(rec):
            return r["waved_dx"]
        dx = min(1.0, max(0.0, float(mt.get("outcome_reward", 0.0))))
        f = r.get("coverage_floor")
        if f is not None:
            dx *= f + (1.0 - f) * evidence_coverage(rec) ** float(r["coverage_power"])
        if mt.get("corruption_armed") and not corr and not _fully_read_image_arm(rec, r):
            return min(dx, r["blind_cap"])
        return dx
    if kind == "format_failure":
        return r["format_failure"]
    if kind == "defer_mismatch":
        return {"exact": r["mismatch"], "partial": r["mismatch_partial"]}.get(
            sub, r["false_mismatch"]
        )
    if kind == "defer_mismatch_claimed":
        if not corr:
            return r["false_mismatch"]
        spec = mt.get("claim_specificity") or claim_specificity(rec)
        return r["mismatch"] if spec == "exact" else r["mismatch_partial"]
    if kind == "defer_ook":
        task_kind = (rec.get("meta") or {}).get("task_kind")
        if task_kind is None:
            raise ValueError(f"{rec['meta']['pmcid']}: defer_ook pricing needs meta.task_kind")
        return r["ook"] if task_kind == "ook" else r["false_ook"]
    raise ValueError(f"{rec['meta']['pmcid']}: unknown outcome_kind {kind!r}")


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def episode_group_uid(rec: dict) -> str:
    """Identity of the GiGPO episode group: (case, corruption iteration, s1)."""
    meta = rec.get("meta") or {}
    visit = (
        f"|{meta.get('task_kind')}|{meta.get('global_step')}"
        if meta.get("global_step") is not None
        else ""
    )
    attempt = meta.get("attempt", 0) or 0
    if attempt:
        visit += f"|a{attempt}"
    gen = meta.get("run_generation", 0) or 0
    if gen:
        visit += f"|g{gen}"
    return _sha1(
        f"{rec['meta']['pmcid']}|"
        f"{rec['budgets'].get('corrupt_seed', '0')}|"
        f"{_sha1(rec['opening']['answer'])}{visit}"
    )


def assert_uniform_scorers(recs: list[dict], allow_mixed: bool = False) -> dict:
    """Every record of a GiGPO group must have been priced by the same dx scorer: a group mixing a panel-judged and an ontology-fallback member would teach the advantage a provider outage, not an answer. Legacy records without `dx_scorer` count as one scorer. Returns the mixed groups; raises unless allow_mixed."""
    by_uid: dict[str, set] = {}
    for r in recs:
        scorer = (r.get("metrics") or {}).get("dx_scorer")
        if scorer is None:
            continue
        by_uid.setdefault(episode_group_uid(r), set()).add(scorer)
    mixed = {u: sorted(map(str, s)) for u, s in by_uid.items() if len(s) > 1}
    if mixed and not allow_mixed:
        sample = next(iter(mixed.items()))
        raise ValueError(
            f"{len(mixed)} GiGPO group(s) carry mixed dx_scorer values "
            f"(e.g. {sample[0][:12]}: {sample[1]}); a fallback-priced member "
            "and a panel-priced member cannot share an advantage baseline. "
            "Drop or re-judge them, or pass --allow-mixed-scorers."
        )
    return mixed


def export_steps(
    rec: dict,
    row: dict,
    cfg: Config,
    asset_paths: dict[str, str],
    scheme: str = "legacy",
    rewards: dict | None = None,
    pen_local: bool | None = None,
) -> list[dict]:
    """One step record per policy turn, in rollout order."""
    if pen_local is None:
        pen_local = bool((rec.get("budgets") or {}).get("local_penalty", False))
    samples = {
        s["turn"]: s for s in export_episode(rec, row, cfg, asset_paths, include_invalid=True)
    }
    uid = episode_group_uid(rec)
    turns = rec["turns"]
    board = scheme != "legacy"
    outcome = (
        replay_terminal(rec, matrix_for_record(rec, rewards))
        if board
        else float(rec["metrics"].get("outcome_reward", 0.0))
    )
    steps: list[dict] = []
    for i, t in enumerate(turns):
        s = samples.get(t["turn"])
        if s is None:
            raise ValueError(
                f"{rec['meta']['pmcid']}: no replayed prompt for turn {t['turn']} ({t.get('verb')})"
            )
        r = t.get("reward", 0.0)
        if t.get("phase") == "gate" and not board:
            r += t["gate"]["reward"]
        done = i == len(turns) - 1
        if done:
            r += outcome
        forfeit = float(t.get("reward", 0.0)) < 0
        steps.append(
            {
                "pmcid": rec["meta"]["pmcid"],
                "episode_group_uid": uid,
                "step_idx": t["turn"],
                "phase": t["phase"],
                "anchor": t.get("anchor"),
                "prompt": s["prompt"],
                "images": s["images"],
                "action": t.get("raw") or s["response"],
                "reward": r,
                "done": done,
                "valid": not forfeit,
                "trainer_reward": (r - float(t.get("reward", 0.0)))
                if (pen_local and forfeit)
                else r,
                "pen_local": bool(pen_local),
            }
        )
    return steps


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--records", required=True, type=Path, help="trajectories.jsonl, already scored"
    )
    ap.add_argument("--profiles", required=True, type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument(
        "--images-root",
        type=Path,
        default=_DEFAULTS["images_dir"],
        help="image root the local_path columns hang off",
    )
    ap.add_argument(
        "--reward-scheme",
        choices=("legacy", "board", "v3"),
        default="legacy",
        help="legacy: turn rewards + gate engine reward + outcome. "
        "board: one terminal scalar by outcome_kind from the "
        "reward board, gate engine rewards recorded but not "
        "trained on (v3 = the board's old name)",
    )
    ap.add_argument(
        "--r-mismatch",
        type=float,
        default=None,
        help=f"terminal reward for a gate mismatch with exact axes (board {REWARDS['mismatch']})",
    )
    ap.add_argument(
        "--r-mismatch-partial",
        type=float,
        default=None,
        help=f"...same catch, wrong axis set (board {REWARDS['mismatch_partial']})",
    )
    ap.add_argument(
        "--r-ook",
        type=float,
        default=None,
        help="terminal reward for an out_of_knowledge abstention "
        f"(board {REWARDS['ook']}; a live record carries the "
        "scheduled r_ook it was priced with)",
    )
    ap.add_argument(
        "--blind-dx-cap",
        type=float,
        default=None,
        help="ceiling on a diagnosis made on a corrupt-ARMED case "
        "whose corruption never bound. Must be <= "
        f"r_mismatch_partial (board {REWARDS['blind_cap']})",
    )
    ap.add_argument(
        "--allow-mixed-scorers",
        action="store_true",
        help="export groups whose members were priced by "
        "different dx scorers (panel vs ontology fallback); "
        "refused by default",
    )
    ap.add_argument(
        "--false-mismatch",
        type=float,
        default=None,
        help="terminal for a gate mismatch on clean evidence or an "
        f"unfounded mismatch claim (board {REWARDS['false_mismatch']}; "
        "0 = no penalty)",
    )
    ap.add_argument(
        "--false-ook",
        type=float,
        default=None,
        help="terminal for an abstention on an answerable / corrupt "
        f"task (board {REWARDS['false_ook']}; 0 = no penalty)",
    )
    ap.add_argument(
        "--zero-workup-wrong",
        type=float,
        default=None,
        help="terminal for a wrong diagnosis made with no workup action "
        f"(board {REWARDS['zero_workup_wrong']}; 0 = the pre-item-C "
        "free 0)",
    )
    ap.add_argument(
        "--format-failure",
        type=float,
        default=None,
        help="terminal for an episode whose terminal the env synthesized "
        f"(no action ever parsed) (board {REWARDS['format_failure']}; "
        "0 = no penalty)",
    )
    ap.add_argument(
        "--waved-dx",
        type=float,
        default=None,
        help="terminal for a diagnosis made after the gate passed the batch "
        f"carrying the swapped study (board {REWARDS['waved_dx']}; "
        "0 = no penalty)",
    )
    ap.add_argument(
        "--coverage-floor",
        type=float,
        default=None,
        help="evidence-coverage multiplier on the dx channel: "
        "dx *= floor + (1-floor)*coverage**power. Must satisfy "
        f"0 <= floor < r_ook (board {REWARDS['coverage_floor']}). "
        "Omit = the record's own matrix, else the board",
    )
    ap.add_argument(
        "--coverage-power",
        type=float,
        default=None,
        help="exponent on coverage in that multiplier (board "
        f"{REWARDS['coverage_power']} = square root; 1 = linear; "
        "a stamp without the key replays linear)",
    )
    ap.add_argument(
        "--blind-cap-mode",
        choices=BLIND_CAP_MODES,
        default=None,
        help="blind_cap predicate version (PC2): "
        "'incomplete_coverage' exempts an armed-unbound "
        "episode whose measured image coverage is exactly "
        "1.0. Omit = each record's own recorded mode "
        "(absent = legacy), byte-stable",
    )
    a = ap.parse_args()

    rewards = cli_reward_overrides(a)
    if a.reward_scheme != "legacy":
        try:
            validate_rewards(rewards)
        except ValueError as e:
            raise SystemExit(str(e))

    recs = [json.loads(l) for l in a.records.read_text(encoding="utf-8").splitlines() if l.strip()]
    recs = [r for r in recs if r.get("schema") == "sp_consult/v3"]
    mixed = assert_uniform_scorers(recs, allow_mixed=a.allow_mixed_scorers)
    if mixed:
        print(
            f"warning: {len(mixed)} group(s) with mixed dx_scorer exported (--allow-mixed-scorers)"
        )
    rows = {r["pmcid"]: r for r in load_profiles(a.profiles)}

    out_path = a.out or a.records.parent / "steps.jsonl"
    n_steps = n_eps = 0
    degraded = {"no_anchor": 0, "no_raw": 0}
    skipped = {"no_profile": 0, "unscored": 0, "unreplayable": 0}
    with out_path.open("w", encoding="utf-8") as fh:
        for rec in recs:
            pmcid = rec["meta"]["pmcid"]
            row = rows.get(pmcid)
            if row is None:
                print(f"warning: {pmcid} not in profiles, skipped")
                skipped["no_profile"] += 1
                continue
            if "outcome_reward" not in rec["metrics"]:
                print(
                    f"warning: {pmcid} has no metrics.outcome_reward "
                    "(run score_records.py first), skipped"
                )
                skipped["unscored"] += 1
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
                deliver_chunk=b.get("deliver_chunk", 6),
                max_batches_per_order=b.get("max_batches_per_order", 1),
                allow_history_final=b.get("allow_history_final", True),
                search_url="x" if b.get("search_enabled") else None,
                images_dir=a.images_root,
                **(
                    {
                        "panel_assets": Path(b["panel_assets"]),
                        "panel_assets_dir": Path(b["panel_assets_dir"]),
                    }
                    if b.get("panel_assets")
                    else {}
                ),
            )
            cfg = restore_panel_policy(cfg, b)
            paths = asset_paths_for(rec, row, cfg)
            try:
                steps = export_steps(rec, row, cfg, paths, scheme=a.reward_scheme, rewards=rewards)
            except ValueError as e:
                print(f"warning: {e}, skipped")
                skipped["unreplayable"] += 1
                continue
            degraded["no_anchor"] += any(s["anchor"] is None for s in steps)
            degraded["no_raw"] += any("raw" not in t for t in rec["turns"])
            for s in steps:
                fh.write(json.dumps(s, ensure_ascii=False) + "\n")
            n_steps += len(steps)
            n_eps += 1
    print(
        f"wrote {out_path}  ({n_steps} steps, {n_eps} episodes, "
        f"{sum(skipped.values())} skipped {skipped})"
    )
    if a.reward_scheme != "legacy":
        eff = validate_rewards(rewards)
        print(
            f"reward board: mismatch {eff['mismatch']} / partial "
            f"{eff['mismatch_partial']} / ook {eff['ook']} / blind cap "
            f"{eff['blind_cap']} / coverage floor {eff['coverage_floor']} "
            f"power {eff['coverage_power']} / "
            f"false mismatch {eff['false_mismatch']} / false ook {eff['false_ook']} "
            f"(explicit overrides: {rewards or 'none'}; a record carrying "
            "metrics.reward_matrix replays its own live matrix); gate "
            "engine rewards recorded but excluded from step rewards"
        )
    if any(degraded.values()):
        print(
            f"note: {degraded['no_anchor']} episode(s) with unanchored "
            f"steps, {degraded['no_raw']} with reconstructed (not sampled) "
            "actions — re-collect under --rl-mode for full step-level credit"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
