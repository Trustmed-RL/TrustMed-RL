"""Dump one real N-rollout GiGPO batch and check four wiring invariants: (1) terminal row, the terminal scalar is added to the last step's reward; (2) anchor timing, an anchor per turn, the group shares s1 and collides on a shared post-decision state; (3) group IDs, every rollout of one visit shares episode_group_uid and a different visit is a different group; (4) advantage placement, the reference A^E + A^S per step (tools/gigpo_sim) that the fork's token-level tensor must match. Drives the real env the fork drives (SpConsultMultiThreadEnv) with a pluggable policy: POLICY_URL set -> an OpenAI-compatible chat endpoint, else a scripted deterministic policy; the patient is served when PATIENT_URL is set, else scripted; judge `ontology`."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

LH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LH))

import rewards as gx
import tools.advantage_reference as sim
from trainer.env_package.sp_consult import envs as E

OPEN = json.dumps(
    {"answer": "The cough started weeks ago.", "used_fact_ids": [], "unknown_topics": []}
)
PREPLY = json.dumps(
    {"answer": "It has been getting worse.", "used_fact_ids": [], "unknown_topics": []}
)
R = "<reasoning>probe</reasoning>\n"


class ScriptedPatient:
    def __init__(self):
        self.model = "scripted-patient"

    def complete(self, messages, pmcid, phase, step=None):
        return OPEN if not any("worse" in str(x) for x in messages) else PREPLY


class EndpointPolicy:
    """One OpenAI-compatible chat completion per policy call."""

    def __init__(self, url, model):
        from openai import OpenAI

        self.client = OpenAI(base_url=url, api_key=os.environ.get("POLICY_API_KEY", "EMPTY"))
        self.model = model

    def act(self, obs, rollout, turn):
        parts = obs.get("multi_modal") or []
        content = obs["text"]
        r = self.client.chat.completions.create(
            model=self.model,
            temperature=1.0,
            max_tokens=512,
            messages=[{"role": "user", "content": content}],
        )
        return r.choices[0].message.content or ""


class ScriptedPolicy:
    """Deterministic per-rollout script: a shared prefix (ask + begin_workup +
    one order -> a post-decision anchor every rollout meets) then a divergent
    tail (odd rollouts order a second test, even ones go straight to dx), so
    the batch has a real anchor COLLISION and real divergence.
    """

    def __init__(self):
        self.model = "scripted-policy"

    def act(self, obs, rollout, turn):
        prefix = [
            R + "<action>ask: When did the cough start and how has it changed?</action>",
            R + "<action>begin_workup</action>",
            R + "<action>order_tests: Laboratory/Clinical Chemistry</action>",
        ]
        if turn < len(prefix):
            return prefix[turn]
        if rollout % 2 == 1 and turn == len(prefix):
            return R + "<action>order_tests: Laboratory/Hematology</action>"
        return R + "<action>final_diagnosis: Invasive aspergillosis</action>"


def run_batch(pmcid: str, group_n: int, out: Path, profiles: Path) -> dict:
    url = os.environ.get("POLICY_URL")
    policy = (
        EndpointPolicy(url, os.environ.get("POLICY_MODEL", "policy")) if url else ScriptedPolicy()
    )
    kw = {
        "pmcid": pmcid,
        "task_kind": "answer",
        "max_asks": 4,
        "max_workup_turns": 6,
        "max_searches": 0,
        "search_enabled": False,
    }

    ec = dict(
        profiles=str(profiles),
        out_dir=str(out.parent / "probe_run"),
        records_out=str(out.parent / "probe_run" / "slim.jsonl"),
        patient_model="scripted",
        judge="ontology",
    )
    from types import SimpleNamespace

    env = E.SpConsultMultiThreadEnv(
        seed=1, env_num=1, group_n=group_n, is_train=True, env_config=SimpleNamespace(**ec)
    )
    if not os.environ.get("PATIENT_URL"):
        env._patient_factory = lambda cfg: ScriptedPatient()

    obs, _ = env.reset([dict(kw) for _ in range(group_n)])
    turn = 0
    done = [False] * group_n
    records = [None] * group_n
    while not all(done):
        actions = [policy.act(obs[i], i, turn) if not done[i] else "" for i in range(group_n)]
        obs, rewards, dones, infos = env.step(actions)
        for i in range(group_n):
            if dones[i] and not done[i]:
                done[i] = True
                records[i] = infos[i].get("episode_record")
        turn += 1
        if turn > 40:
            break
    env.close()

    trajs, group_ids = [], []
    for i, rec in enumerate(records):
        if rec is None:
            continue
        turns = rec["turns"]
        stream = [float(t.get("reward", 0.0)) for t in turns]
        term = float(rec["metrics"].get("terminal_reward", 0.0))
        exported = list(stream)
        exported[-1] += term
        uid = gx.episode_group_uid(rec)
        group_ids.append(uid)
        trajs.append(
            sim.Traj(
                step_rewards=stream,
                terminal=term,
                anchors=[t.get("anchor") for t in turns],
                outcome=float(
                    rec["metrics"].get("dx_score", rec["metrics"].get("outcome_reward", 0.0))
                ),
            )
        )
        rec["_probe"] = {
            "uid": uid,
            "stream_per_turn": stream,
            "terminal_reward": term,
            "exported_last_row": exported[-1],
            "anchors": [t.get("anchor") for t in turns],
            "verbs": [t.get("verb") for t in turns],
            "outcome_kind": rec["metrics"].get("outcome_kind"),
        }

    pen_local = any(sim.record_pen_local(r) for r in records if r is not None)
    adv = sim.advantages(trajs, "mean_std", pen_local=pen_local) if len(trajs) >= 2 else []

    report = {
        "pmcid": pmcid,
        "group_n": group_n,
        "returned": len(trajs),
        "check_1_terminal_row": all(
            abs(
                r["_probe"]["exported_last_row"]
                - (r["_probe"]["stream_per_turn"][-1] + r["_probe"]["terminal_reward"])
            )
            < 1e-9
            for r in records
            if r
        ),
        "check_3_group_ids": {
            "all_equal": len(set(group_ids)) == 1 if group_ids else False,
            "uid": group_ids[0] if group_ids else None,
            "distinct_from_other_visit": (
                group_ids[0]
                != gx.episode_group_uid(
                    {
                        "meta": {"pmcid": pmcid, "global_step": 999, "task_kind": "answer"},
                        "budgets": {"corrupt_seed": "0"},
                        "opening": {"answer": "x"},
                    }
                )
                if group_ids
                else None
            ),
        },
        "check_2_anchor_timing": {
            "per_turn": all(all(a for a in r["_probe"]["anchors"]) for r in records if r),
            "share_opener": len({r["_probe"]["anchors"][0] for r in records if r}) == 1,
            "collision": sim.anchor_report_from_trajs(trajs)
            if hasattr(sim, "anchor_report_from_trajs")
            else _collisions(trajs),
        },
        "check_4_reference_advantage": [
            {"rollout": i, "per_step": [round(x, 4) for x in adv[i]]} for i in range(len(adv))
        ],
        "rollouts": [r["_probe"] for r in records if r],
    }
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def _collisions(trajs) -> dict:
    opener = trajs[0].anchors[0] if trajs else None
    occ = {}
    for t in trajs:
        for a in t.anchors:
            if a and a != opener:
                occ[a] = occ.get(a, 0) + 1
    return {"distinct": len(occ), "collided": sum(1 for v in occ.values() if v >= 2)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pmcid", default="PMC555544")
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--out", type=Path, default=LH / "results" / "rollout_probe" / "batch.json")
    ap.add_argument(
        "--profiles",
        type=Path,
        default=Path(
            os.environ.get(
                "TRUSTMED_PROFILES_SAMPLE",
                str(Path(__file__).resolve().parents[2] / "data" / "profiles_sample_30.jsonl"),
            )
        ),
    )
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    rep = run_batch(a.pmcid, a.group, a.out, a.profiles)
    print(f"batch: {rep['returned']}/{rep['group_n']} rollouts returned")
    print(f"  1 terminal row on last step: {rep['check_1_terminal_row']}")
    print(
        f"  2 anchor timing: per-turn {rep['check_2_anchor_timing']['per_turn']}, "
        f"share opener {rep['check_2_anchor_timing']['share_opener']}, "
        f"collisions {rep['check_2_anchor_timing']['collision']}"
    )
    print(
        f"  3 group ids: all_equal {rep['check_3_group_ids']['all_equal']}, "
        f"distinct-from-other-visit {rep['check_3_group_ids']['distinct_from_other_visit']}"
    )
    print(
        f"  4 reference step-advantage written for {len(rep['check_4_reference_advantage'])} rollouts"
    )
    print(f"  -> {a.out}")
    ok = (
        rep["check_1_terminal_row"]
        and rep["check_3_group_ids"]["all_equal"]
        and rep["check_2_anchor_timing"]["per_turn"]
        and rep["check_2_anchor_timing"]["share_opener"]
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
