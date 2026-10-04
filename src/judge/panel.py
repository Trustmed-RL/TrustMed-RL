"""LLM judge panel + ontology fusion over environment records."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from judge.prompt import extract_json, judge_prompt
from outcomes import classify_outcome, gold_of, guarded_verdict
from environment import done_key, load_profiles

from judge.rubric import compute_diagnosis_reward
from judge.verdict_fusion import fuse_case

PANEL: dict[str, dict] = {
    "gpt-4.1-mini": {"provider": "openai", "model": "gpt-4.1-mini"},
}

HELD_OUT: dict[str, dict] = {
    "gpt-4o": {"provider": "openai", "model": "gpt-4o"},
}


PANEL_REASONING: dict[str, dict] = {
    "gpt-5-mini": {"provider": "openai_reasoning", "model": "gpt-5-mini", "parse": "normalized"},
}
ALL_JUDGES: dict[str, dict] = {**PANEL, **PANEL_REASONING}


def judge_spec(judge: str) -> dict:
    """The spec of any registered judge , or a clear
    refusal — never a silent default.
    """
    try:
        return ALL_JUDGES[judge]
    except KeyError:
        raise KeyError(f"unknown judge {judge!r}; known: {list(ALL_JUDGES)}") from None


JUDGE_MAX_TOKENS = 512
REASONING_MAX_COMPLETION_TOKENS = 4096
REASONING_LIVE_WORKERS = int(os.environ.get("TRUSTMED_REASONING_LIVE_WORKERS", "16"))
OPENAI_TEMPERATURE = 0.0
BATCH_POLL_SECONDS = 20.0
LIVE_WORKERS = 8
_CID_BAD = re.compile(r"[^A-Za-z0-9_-]")


_LEVEL_RANK = {"full": 0, "core": 1, "partial": 2, "none": 3}

RAW_REPLIES_NAME = "panel_raw.jsonl"


def _cid(key: str) -> str:
    """A done-key as a batch-API-legal custom_id ([A-Za-z0-9_-]{1,64})."""
    return _CID_BAD.sub("_", key)[:64]


def _norm_model(name: str) -> str:
    """`Qwen/Qwen3-VL-235B-A22B-Thinking` -> `qwen3-vl-235b-a22b-thinking`."""
    return str(name or "").split("/")[-1].strip().lower()


def assert_judge_not_generator(records: list[dict], judges: list[str]) -> None:
    """Refuse to grade a record whose own doctor sits on the panel - except
    the arms in SELF_GRADED_ARMS, which the full panel grades, own vote and all.
    """
    panel = {_norm_model(judge_spec(j)["model"]) for j in judges} | {_norm_model(j) for j in judges}
    doctors = {str((r.get("models") or {}).get("doctor") or "") for r in records}
    clash = sorted(d for d in doctors if _norm_model(d) in panel)
    if clash:
        raise SystemExit(
            f"judge == generator: {clash} produced these episodes and is on the "
            f"panel {judges}. Drop that judge (--judges) or judge a different "
            f"file; a model grading its own transcripts cannot set the SFT bar."
        )


def _norm_term(v) -> str:
    return " ".join(str(v or "").lower().split())


def _clip_matches(verdict: dict) -> dict:
    """A verdict whose `matches` really are a ONE-TO-ONE alignment."""
    gt = int(verdict.get("gt_count") or 0)
    pred = int(verdict.get("pred_count") or 0)
    rows = [m for m in (verdict.get("matches") or []) if isinstance(m, dict)]
    rows = sorted(
        rows, key=lambda m: _LEVEL_RANK.get(str(m.get("level") or "").lower(), len(_LEVEL_RANK))
    )
    kept: list[dict] = []
    seen_gt: set[str] = set()
    seen_pred: set[str] = set()
    for m in rows:
        g, p = _norm_term(m.get("ground_truth")), _norm_term(m.get("predicted"))
        if (g and g in seen_gt) or (p and p in seen_pred):
            continue
        if g:
            seen_gt.add(g)
        if p:
            seen_pred.add(p)
        kept.append(m)
    out = dict(verdict)
    out["matches"] = kept[: max(0, min(gt, pred))]
    return out


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S | re.I)
_THINK_OPEN = re.compile(r"<think>.*\Z", re.S | re.I)


def strip_reasoning(text: str | None) -> str | None:
    """Only `message.content` is ever parsed, never `reasoning_content`.
    Every closed <think>...</think> block is removed; an unclosed <think>
    drops the rest of the text. What remains goes to the production parser;
    no JSON there is a parse failure, never a fallback to the reasoning.
    """
    if text is None:
        return None
    t = _THINK_BLOCK.sub("", text)
    t = _THINK_OPEN.sub("", t)
    return t


_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S | re.I)
NO_MATCH_LEVELS = frozenset(
    {
        "none",
        "no match",
        "no_match",
        "nomatch",
        "unrelated",
        "different",
        "discrepant",
        "mismatch",
        "no-match",
        "not matched",
        "unmatched",
    }
)


def _first_balanced_object(t: str) -> str | None:
    """The first top-level {...} span with balanced braces (string-aware)."""
    i = t.find("{")
    if i < 0:
        return None
    depth, in_str, esc = 0, False, False
    for k in range(i, len(t)):
        ch = t[k]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return t[i : k + 1]
    return None


def normalize_reply(text: str | None) -> tuple[str | None, dict]:
    """Validation-side normalisation of a judge reply, applied identically to
    every judge BEFORE the production parser (`detail_from_reply`).
    """
    info = {"fenced": False, "dropped_no_match_rows": 0, "changed": False}
    if not text:
        return text, info
    t = strip_reasoning(text) or ""
    m = _FENCE_RE.search(t)
    span = m.group(1) if m else _first_balanced_object(t)
    info["fenced"] = bool(m)
    if span is None:
        return text, info
    try:
        obj = json.loads(span)
    except ValueError:
        return text, info
    if not isinstance(obj, dict):
        return text, info
    i, j = t.find("{"), t.rfind("}")
    try:
        prod_obj = json.loads(t[i : j + 1]) if 0 <= i < j else None
    except ValueError:
        prod_obj = None
    matches = obj.get("matches")
    if isinstance(matches, list):
        kept = []
        for row in matches:
            lv = str((row or {}).get("level", "")).strip().lower() if isinstance(row, dict) else ""
            if lv in NO_MATCH_LEVELS:
                info["dropped_no_match_rows"] += 1
                continue
            kept.append(row)
        obj["matches"] = kept
    out = json.dumps(obj, ensure_ascii=False)
    info["changed"] = prod_obj != obj
    return out, info


def detail_from_reply(raw: str | None, *, normalize: bool = False) -> dict | None:
    """Judge text -> the scorer's detail dict, or None when it never parsed."""
    if not raw:
        return None
    if normalize:
        raw, _ = normalize_reply(raw)
    try:
        verdict = _clip_matches(extract_json(raw))
        detail = compute_diagnosis_reward(verdict, return_details=True)
    except Exception:
        return None
    if not isinstance(detail, dict):
        return None
    detail = dict(detail)
    detail["matches"] = list(verdict.get("matches") or [])
    return detail


def _openai_client():
    from openai import OpenAI

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=key, timeout=600.0, max_retries=3)


def _openai_body(model: str, prompt: str) -> dict:
    """The chat-completions request body: pinned temperature and a visible-token ceiling."""
    return {
        "model": model,
        "max_tokens": JUDGE_MAX_TOKENS,
        "temperature": OPENAI_TEMPERATURE,
        "messages": [{"role": "user", "content": prompt}],
    }


def _openai_reasoning_body(model: str, prompt: str) -> dict:
    """gpt-5*: `max_completion_tokens` (reasoning + visible), NO temperature —
    a non-default value is a 400 on a reasoning model, and the Batch API
    validates the same body.
    """
    return {
        "model": model,
        "max_completion_tokens": REASONING_MAX_COMPLETION_TOKENS,
        "messages": [{"role": "user", "content": prompt}],
    }


def openai_batch(
    model: str,
    items: list[tuple[str, str]],
    poll_seconds: float = BATCH_POLL_SECONDS,
    body_of: Callable[[str, str], dict] = _openai_body,
) -> dict[str, str | None]:
    """items: (custom_id, prompt). Submits one Batch job, polls, returns replies."""
    import io

    client = _openai_client()
    lines = [
        json.dumps(
            {
                "custom_id": cid,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body_of(model, prompt),
            },
            ensure_ascii=False,
        )
        for cid, prompt in items
    ]
    buf = io.BytesIO(("\n".join(lines) + "\n").encode())
    buf.name = "panel_judge.jsonl"
    fh = client.files.create(file=buf, purpose="batch")
    job = client.batches.create(
        input_file_id=fh.id, endpoint="/v1/chat/completions", completion_window="24h"
    )
    print(f"  openai batch {job.id}: {len(items)} request(s)")
    while True:
        job = client.batches.retrieve(job.id)
        if job.status in ("completed", "failed", "expired", "cancelled"):
            break
        time.sleep(poll_seconds)
    out: dict[str, str | None] = {cid: None for cid, _ in items}
    if job.output_file_id:
        for line in client.files.content(job.output_file_id).text.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("error"):
                continue
            out[row["custom_id"]] = row["response"]["body"]["choices"][0]["message"]["content"]
    return out


def _live(
    call: Callable[[str], str], items: list[tuple[str, str]], workers: int = LIVE_WORKERS
) -> dict[str, str | None]:
    out: dict[str, str | None] = {}

    def one(item: tuple[str, str]) -> None:
        cid, prompt = item
        try:
            out[cid] = call(prompt)
        except Exception as e:
            print(f"  live judge failed on {cid}: {str(e)[:160]}")
            out[cid] = None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, items))
    return out


def openai_live(model: str, items) -> dict[str, str | None]:
    client = _openai_client()

    def call(prompt: str) -> str:
        r = client.chat.completions.create(**_openai_body(model, prompt))
        return r.choices[0].message.content or ""

    return _live(call, items)


def openai_reasoning_batch(
    model: str, items: list[tuple[str, str]], poll_seconds: float = BATCH_POLL_SECONDS
) -> dict[str, str | None]:
    return openai_batch(model, items, poll_seconds, body_of=_openai_reasoning_body)


def openai_reasoning_live(model: str, items) -> dict[str, str | None]:
    """gpt-5* live: only `message.content` is returned (the reasoning never
    reaches the parser); 16 workers — the visible reply is the same ~130-token
    JSON, the latency is the model's thinking.
    """
    client = _openai_client()

    def call(prompt: str) -> str:
        r = client.chat.completions.create(**_openai_reasoning_body(model, prompt))
        return r.choices[0].message.content or ""

    return _live(call, items, workers=REASONING_LIVE_WORKERS)


def run_panel(
    pairs: dict[str, tuple[str, str]],
    judges: list[str],
    *,
    mode: str = "batch",
    _replies: dict[str, dict[str, str | None]] | None = None,
) -> tuple[dict[str, dict[str, dict]], dict[str, dict], list[dict]]:
    """{key: (gt, pred)} -> ({key: {judge: detail}}, {judge: status}, raw rows)."""
    items = [(_cid(k), judge_prompt(gt, pred)) for k, (gt, pred) in pairs.items()]
    by_cid = {_cid(k): k for k in pairs}
    details: dict[str, dict[str, dict]] = {k: {} for k in pairs}
    status: dict[str, dict] = {}
    raw_rows: list[dict] = []

    for judge in judges:
        spec = judge_spec(judge)
        normalize = spec.get("parse") == "normalized"
        try:
            if _replies is not None:
                replies = _replies.get(judge) or {}
            elif spec["provider"] == "openai":
                replies = (
                    openai_batch(spec["model"], items)
                    if mode == "batch"
                    else openai_live(spec["model"], items)
                )
            elif spec["provider"] == "openai_reasoning":
                replies = (
                    openai_reasoning_batch(spec["model"], items)
                    if mode == "batch"
                    else openai_reasoning_live(spec["model"], items)
                )
            else:
                raise RuntimeError(f"unknown judge provider {spec['provider']!r}")
        except Exception as e:
            status[judge] = {
                "state": "unreachable",
                "requested": len(pairs),
                "parsed": 0,
                "parse_failed": 0,
                "missing": len(pairs),
                "error": str(e)[:200],
            }
            print(f"  {judge}: unreachable — {str(e)[:160]}")
            continue
        scored = parse_failed = 0
        for cid, raw in replies.items():
            key = by_cid.get(cid)
            if key is None:
                continue
            raw_rows.append({"judge": judge, "key": key, "custom_id": cid, "raw": raw})
            d = detail_from_reply(raw, normalize=normalize)
            if d is not None:
                details[key][judge] = d
                scored += 1
            else:
                parse_failed += 1
        status[judge] = {
            "state": "ok",
            "requested": len(pairs),
            "parsed": scored,
            "parse_failed": parse_failed,
            "missing": len(pairs) - scored - parse_failed,
        }
        print(
            f"  {judge}: {scored}/{len(pairs)} parsed"
            f" ({parse_failed} parse-fail, {status[judge]['missing']} missing)"
        )
    return details, status, raw_rows


def ontology_verdicts(
    pairs: dict[str, tuple[str, str]], check: Callable[[str, str], dict] | None = None
) -> dict[str, dict | None]:
    """MedGen/MONDO/ORDO on every (gt, pred), cached and keyless."""
    if check is None:
        require_ontology()
        from judge.ontology_check import default_check_pair

        check = default_check_pair()
    out: dict[str, dict | None] = {}
    cache: dict[tuple[str, str], dict | None] = {}
    for key, (gt, pred) in pairs.items():
        ck = (gt, pred)
        if ck not in cache:
            try:
                cache[ck] = check(gt, pred)
            except Exception as e:
                raise RuntimeError(f"ontology lookup failed for {key} {ck!r}: {e}") from e
        out[key] = cache[ck]
    return out


def ontology_db_path() -> Path:
    """Where the local backend will look: TRUSTMED_ONTOLOGY_DB, else the file beside judge/ontology_local.py."""
    forced = os.environ.get("TRUSTMED_ONTOLOGY_DB")
    if forced:
        return Path(forced)
    try:
        import judge.ontology_local as ontology_local

        return Path(ontology_local.DB_PATH)
    except Exception:
        return Path(__file__).resolve().parent / "ontology_local.db"


_PROBE_PAIR = ("Fabry disease", "Fabry disease")
_probe_verdict: str | None = None


def ontology_probe_verdict() -> str | None:
    """What the last `require_ontology` probe returned (None before any)."""
    return _probe_verdict


def require_ontology(db: Path | None = None) -> Path:
    """Preflight: refuse to run the panel without the ontology floor."""
    global _probe_verdict
    db = Path(db) if db is not None else ontology_db_path()
    if not db.exists():
        raise RuntimeError(
            f"ontology_local.db not found at {db}; the panel refuses to run "
            "without the ontology floor. Build it with "
            "`python build_ontology_db.py --download` in the eval dir, or "
            "point TRUSTMED_ONTOLOGY_DB at an existing database"
        )
    try:
        from judge.ontology_local import LocalOntology

        v = LocalOntology(db).check_pair(*_PROBE_PAIR)
    except Exception as e:
        raise RuntimeError(f"ontology backend at {db} failed its probe {_PROBE_PAIR}: {e}") from e
    verdict = str((v.get("verdict") if isinstance(v, dict) else v) or "")
    if not verdict or verdict.upper() == "UNAVAILABLE":
        raise RuntimeError(f"ontology backend at {db} returned no verdict for {_PROBE_PAIR}: {v!r}")
    _probe_verdict = verdict
    return db


def assert_panel_deep_enough(judges: list[str], min_judges: int) -> None:
    """Refuse, before the first paid call, a judge list shorter than the depth it has to reach."""
    if len(judges) < min_judges:
        raise SystemExit(
            f"panel {judges} has {len(judges)} judge(s), below min_judges={min_judges}: "
            "every episode would come back unjudged."
        )


TRUSTED_MATCHES = frozenset(
    {
        "label",
        "normalized_label",
        "exact_title",
        "exact_synonym",
        "exact_title_synonym",
        "title",
        "synonym",
        "exact_title_normalized",
        "exact_synonym_normalized",
        "normalized_title",
        "normalized_synonym",
    }
)
_DECIDING_RELATIONS = ("equivalent", "a_subtype_of_b", "b_subtype_of_a", "subtype_siblings")


def _norm_term(t: str) -> str:
    """Comparison key: punctuation-, dash- and possessive-insensitive."""
    import unicodedata

    t = unicodedata.normalize("NFKD", t or "")
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = t.strip().lower()
    for ch in "'’ʼ":
        t = t.replace(ch, "")
    for ch in ",.-/()‐‑‒–—―":
        t = t.replace(ch, " ")
    toks = [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in t.split()]
    return " ".join(toks)


def _node_label(node: dict | None) -> str:
    """The label a leg resolved to, whichever source spelled it."""
    if not isinstance(node, dict):
        return ""
    return str(node.get("label") or node.get("title") or "")


def _side_kind(node: dict | None, term: str) -> str:
    """How well one resolved concept stands for the term it came from."""
    if not isinstance(node, dict) or not node.get("match"):
        return "missing"
    label = _node_label(node)
    nt, nl = _norm_term(term), _norm_term(label)
    if str(node["match"]) in TRUSTED_MATCHES or nt == nl:
        return "exact"
    if nl and set(nl.split()) <= set(nt.split()):
        return "contained"
    return "wandered"


def _node_id(node: dict | None) -> str:
    """Whichever key this source calls its concept identifier."""
    if not isinstance(node, dict):
        return ""
    return str(node.get("id") or node.get("cui") or "")


def _leg_untrusted(leg: dict | None, terms: tuple[str, str]) -> list[str]:
    """Empty when this source's claim may carry a floor, else the reasons."""
    kinds = [
        (side, _side_kind((leg or {}).get(side), term)) for side, term in zip(("a", "b"), terms)
    ]
    ids = {side: _node_id((leg or {}).get(side)) for side, _ in kinds}
    kmap = dict(kinds)
    if ids["a"] and ids["a"] == ids["b"]:
        for side, other in (("a", "b"), ("b", "a")):
            if kmap[side] == "wandered" and kmap[other] == "exact":
                kmap[side] = "same_concept_as_anchor"
        kinds = [(s, kmap[s]) for s, _ in kinds]
    bad = [
        f"{side}:{k}:{_node_label((leg or {}).get(side))[:36]!r}"
        for side, k in kinds
        if k in ("wandered", "missing")
    ]
    if bad:
        return bad
    if not any(k == "exact" for _, k in kinds):
        return [f"{side}:{k}:{_node_label((leg or {}).get(side))[:36]!r}" for side, k in kinds] + [
            "no-exact-anchor"
        ]
    return []


def guard_verdict(verdict: dict | None, lookup=None) -> dict | None:
    """Neutralize a floor-raising verdict that rests on fuzzy resolution."""
    if not verdict:
        return verdict
    name = str(verdict.get("verdict") or "").upper()
    if name not in ("FULL", "CORE"):
        return verdict

    per_source = verdict.get("per_source") or {}
    head = verdict.get("head_check") or {}
    if verdict.get("direction") == "head_terms_match" and head:
        terms = (str(head.get("head_a") or ""), str(head.get("head_b") or ""))
        rels = head.get("head_relations") or {}
        deciding_srcs = [s for s, r in rels.items() if r in _DECIDING_RELATIONS]
        bad: list[str] = []
        resolved = 0
        if lookup is not None:
            for s in deciding_srcs:
                pair_bad: list[str] = []
                ok = True
                for term in terms:
                    node = (lookup(term) or {}).get(s)
                    if not node:
                        ok = False
                        continue
                    resolved += 1
                    pair_bad += _leg_untrusted({"a": node}, (term, term))
                if ok and not pair_bad:
                    trusted_src = True
                bad += pair_bad
        trusted = resolved > 0 and bool(locals().get("trusted_src"))
        matches = bad
    else:
        terms = (str(verdict.get("a") or ""), str(verdict.get("b") or ""))
        deciding = [
            leg
            for leg in per_source.values()
            if isinstance(leg, dict) and leg.get("relation") in _DECIDING_RELATIONS
        ]
        per_leg = [_leg_untrusted(leg, terms) for leg in deciding]
        trusted = any(not bad for bad in per_leg)
        matches = [m for bad in per_leg for m in bad]

    if trusted:
        return verdict
    out = dict(verdict)
    out["verdict"] = "UNKNOWN"
    out["implied_weight"] = 0.0
    out["guard"] = {
        "refused": name,
        "reason": "floor rests on a concept that is not the term",
        "untrusted": matches,
    }
    out["flags"] = list(verdict.get("flags") or []) + [
        f"GUARD: {name} refused — deciding legs resolved onto {sorted(set(matches)) or 'nothing'}"
    ]
    return out


def _head_lookup():
    """`{source: match_kind}` per term from the local database, or None."""
    try:
        from judge.ontology_local import LocalOntology

        lo = LocalOntology()
    except Exception:
        return None

    def lookup(term: str) -> dict:
        if not term:
            return {}
        out: dict[str, str] = {}
        try:
            mg = lo.medgen_concept(term)
            if mg and mg.get("match"):
                out["medgen_umls"] = mg["match"]
        except Exception:
            pass
        for onto in ("mondo", "ordo"):
            try:
                d = lo.ols_lookup(term, onto)
                if d and d.get("match"):
                    out[onto] = d["match"]
            except Exception:
                pass
        return out

    return lookup


def _floorable(verdict: dict | None) -> dict | None:
    """Only a verdict the fusion knows how to read is handed to it; an
    `unavailable` marker is kept in the record but never fused.
    """
    if not verdict or str(verdict.get("verdict") or "").upper() == "UNAVAILABLE":
        return None
    return verdict


_HEAD_LOOKUP = None


def ensure_head_lookup() -> None:
    global _HEAD_LOOKUP
    if _HEAD_LOOKUP is None:
        _HEAD_LOOKUP = _head_lookup()


def panel_block(judge_details: dict[str, dict], verdict: dict | None) -> dict:
    """One episode's `metrics.panel`: per-judge scores + the fused verdict."""
    fused = fuse_case(judge_details, _floorable(verdict))
    return {
        "judges": {
            name: {
                "t_reward": round(float(d.get("t_reward") or 0.0), 4),
                "full": int(d.get("full_count") or 0),
                "core": int(d.get("core_count") or 0),
                "partial": int(d.get("partial_count") or 0),
                "gt_count": int(d.get("gt_count") or 0),
                "pred_count": int(d.get("pred_count") or 0),
            }
            for name, d in judge_details.items()
        },
        "fused_t_reward": round(float(fused["median_t_reward"]), 4),
        "prefusion_t_reward": round(float(fused["median_t_reward_prefusion"]), 4),
        "verdict_source": fused["verdict_source"],
        "none_conflict": bool(fused["none_conflict"]),
        "n_judges": int(fused["n_judges"]),
        "judges_present": list(fused["judges_present"]),
        "ontology": verdict,
    }


def is_dx(rec: dict) -> bool:
    return classify_outcome(rec)[0] == "dx"


def _socket_belt() -> None:
    import socket

    socket.setdefaulttimeout(75)


def backfill(
    records: Path,
    profiles: Path,
    judges: list[str],
    *,
    mode: str = "batch",
    min_judges: int | None = None,
    limit: int | None = None,
    restrict: set[str] | None = None,
    raw_out: Path | None = None,
    _replies=None,
    _check=None,
) -> dict:
    """Score every dx episode in `records` and write the panel back into it."""
    _socket_belt()
    ensure_head_lookup()
    if _check is None:
        print("ontology:", require_ontology())
    recs = [json.loads(l) for l in records.open(encoding="utf-8") if l.strip()]
    truth = {r["pmcid"]: gold_of(r) for r in load_profiles(profiles)}
    min_judges = int(min_judges or len(judges))
    assert_judge_not_generator(recs, judges)
    assert_panel_deep_enough(judges, min_judges)

    pairs: dict[str, tuple[str, str]] = {}
    for rec in recs:
        key = done_key(rec)
        gt = truth.get(rec["meta"]["pmcid"])
        if restrict is not None and key not in restrict:
            continue
        if gt and is_dx(rec):
            pairs[key] = (gt, rec["metrics"].get("final_answer", ""))
    if limit:
        pairs = dict(list(pairs.items())[:limit])
    print(f"{len(pairs)} dx episode(s) of {len(recs)} -> panel {judges}")
    if not pairs:
        return {"records": len(recs), "judged": 0, "judges": {}}

    details, status, raw_rows = run_panel(pairs, judges, mode=mode, _replies=_replies)
    raw_path = raw_out or records.with_name(RAW_REPLIES_NAME)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("a", encoding="utf-8") as fh:
        for row in raw_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    import statistics as _stats

    def _prefusion(ds: dict | None) -> float | None:
        ts = [
            v.get("t_reward")
            for v in (ds or {}).values()
            if isinstance(v, dict) and v.get("t_reward") is not None
        ]
        return _stats.median(ts) if ts else None

    _need = {k: p for k, p in pairs.items() if (_prefusion(details.get(k)) or 0.0) < 0.85}
    if len(_need) < len(pairs):
        print(
            f"  ontology: {len(_need)}/{len(pairs)} pair(s) below the bar "
            "(the rest keep on the panel alone)"
        )
    verdicts = ontology_verdicts(_need, check=_check)

    blocks: dict[str, dict] = {}
    for key in pairs:
        if details[key]:
            gt, pred = pairs[key]
            verdict = verdicts.get(key)
            guard_reason = None
            if verdict is not None:
                allowed, d = guarded_verdict(gt, pred, for_floor=True)
                guard_reason = d.get("reason")
                if allowed is None:
                    verdict = None
            block = panel_block(details[key], verdict)
            block["unjudged"] = block["n_judges"] < min_judges
            block["guard_reason"] = guard_reason
            blocks[key] = block

    tmp = records.with_name(records.name + ".tmp")
    n_written = n_thin = 0
    with records.open(encoding="utf-8") as src, tmp.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            rec = json.loads(line)
            block = blocks.get(done_key(rec))
            if block is not None:
                rec["metrics"]["panel"] = block
                if block["unjudged"]:
                    rec["metrics"].pop("fused_t_reward", None)
                    n_thin += 1
                else:
                    rec["metrics"]["fused_t_reward"] = block["fused_t_reward"]
                    n_written += 1
            dst.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, records)

    kept = sum(1 for b in blocks.values() if not b["unjudged"] and b["fused_t_reward"] >= 0.85)
    return {
        "records": len(recs),
        "dx": len(pairs),
        "judged": n_written,
        "at_or_above_0.85": kept,
        "judges": status,
        "min_judges": min_judges,
        "thin_panel": n_thin,
        "raw_replies": str(raw_path),
        "unjudged_dx": len(pairs) - len(blocks) + n_thin,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--records", type=Path, required=True, help="trajectories.jsonl, rewritten in place"
    )
    ap.add_argument("--profiles", type=Path, required=True)
    ap.add_argument(
        "--judges",
        default=",".join(PANEL),
        help="comma list from " + ", ".join(ALL_JUDGES) + " (default: the frozen panel)",
    )
    ap.add_argument(
        "--mode",
        choices=("batch", "live"),
        default="batch",
        help="batch (default): the OpenAI Batch API, -50%%",
    )
    ap.add_argument(
        "--min-judges",
        type=int,
        default=None,
        help="judges that must parse for an episode to count (default: all requested)",
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None, help="also write the run summary here")
    ap.add_argument(
        "--raw-out",
        type=Path,
        default=None,
        help=f"raw judge replies (default: {RAW_REPLIES_NAME} beside --records)",
    )
    a = ap.parse_args()

    judges = [j.strip() for j in a.judges.split(",") if j.strip()]
    unknown = [j for j in judges if j not in ALL_JUDGES]
    if unknown:
        raise SystemExit(f"unknown judge(s) {unknown}; known judges are {list(ALL_JUDGES)}")
    counts = backfill(
        a.records,
        a.profiles,
        judges,
        mode=a.mode,
        min_judges=a.min_judges,
        limit=a.limit,
        raw_out=a.raw_out,
    )
    print(json.dumps(counts, indent=2))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(counts, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
