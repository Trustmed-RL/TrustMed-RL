"""The judge-in-the-loop diagnosis reward: a configurable subset of the
frozen eval panel + the GUARDED ontology floor, called per episode inside
the RL env.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import outcomes as sr
from judge.panel import (
    ALL_JUDGES,
    PANEL,
    REASONING_MAX_COMPLETION_TOKENS,
    detail_from_reply,
    judge_spec,
    panel_block,
    require_ontology,
)
from judge.prompt import judge_prompt

DX_SCORER_PANEL = "panel_live"
DEFAULT_JUDGES = ("gpt-4.1-mini",)
DX_SCORER_FALLBACK = "ontology_fallback"
DEFAULT_BAR = 0.85
PRICES_PER_M: dict[str, tuple[float, float]] = {
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-5-mini": (0.25, 2.00),
}
EST_INPUT_TOKENS, EST_OUTPUT_TOKENS = 790, 120
EST_OUTPUT_TOKENS_REASONING = 200
JUDGE_MAX_TOKENS = 512
RETRY_ATTEMPTS = 5

_JSON_SHAPE = re.compile(r"[{}]")
_RUBRIC_KEYWORD = re.compile(r"\b(gt_count|pred_count|matches|level)\b", re.I)
_ROLE_MARKER = re.compile(r"(?:^|\n|\s)(system|assistant|user)\s*:", re.I)
_INSTRUCTION = re.compile(
    r"\b(ignore|disregard|forget|override)\b.{0,40}\b(previous|prior|above|rubric|"
    r"instructions?|scor\w*|grad\w*)|\b(return|output|respond with|print|emit)\b"
    r".{0,30}\b(json|level|matches|full|core|score)\b|\bgrade (?:this|it) as\b|"
    r"\bscore\s*[:=]?\s*1(?:\.0)?\b|\byou are (?:a|an|the)\b|\bas an ai\b|"
    r"\bnew instructions?\b",
    re.I,
)
_LONG_CHARS = 600


class ReplayMiss(RuntimeError):
    """A replay-only panel was asked for a pair its cache never saw."""


class PanelHalt(RuntimeError):
    """Raised by the env at reset() when the panel asked for a halt."""


def injection_flags(pred: str) -> list[str]:
    """Shapes a policy could learn to steer the judges with. Recorded and
    counted, never applied to the number (eval sees the same text).
    """
    t = str(pred or "")
    flags = []
    if _JSON_SHAPE.search(t):
        flags.append("json_shape")
    if _RUBRIC_KEYWORD.search(t):
        flags.append("rubric_keyword")
    if _ROLE_MARKER.search(t):
        flags.append("role_marker")
    if _INSTRUCTION.search(t):
        flags.append("instruction")
    if len(t) > _LONG_CHARS:
        flags.append("long")
    return flags


def estimate_pair_cost(judges: list[str] | None = None) -> float:
    """USD for one pair through the priced judges at the assumed token
    counts -- the unit spend accounting is expressed in.
    """
    total = 0.0
    for j in judges or list(PANEL):
        spec = ALL_JUDGES.get(j, {})
        model = spec.get("model", j)
        pin, pout = PRICES_PER_M.get(model, (0.0, 0.0))
        out = (
            EST_OUTPUT_TOKENS_REASONING
            if spec.get("provider") == "openai_reasoning"
            else EST_OUTPUT_TOKENS
        )
        total += EST_INPUT_TOKENS * pin / 1e6 + out * pout / 1e6
    return total


def required_env(judges: list[str]) -> set[str]:
    """The API keys the requested judges need."""
    need = set()
    for j in judges:
        p = judge_spec(j)["provider"]
        if p in ("openai", "openai_reasoning"):
            need.add("OPENAI_API_KEY")
    return need


def check_reachable(judges: list[str]) -> None:
    """Refuse a judge set that cannot be called: an unknown judge, or one whose API key is not in the environment."""
    unknown = [j for j in judges if j not in ALL_JUDGES]
    if unknown:
        raise ValueError(f"unknown judge(s) {unknown}; known: {list(ALL_JUDGES)}")
    if not judges:
        raise ValueError("at least one judge is required")
    missing = sorted(k for k in required_env(judges) if not os.environ.get(k))
    if missing:
        raise RuntimeError(f"judge set {judges} needs {missing} in the environment")


def usage_cost(model: str, usage: dict | None) -> float:
    pin, pout = PRICES_PER_M.get(model, (0.0, 0.0))
    u = usage or {}
    return (
        float(u.get("input_tokens") or 0) * pin + float(u.get("output_tokens") or 0) * pout
    ) / 1e6


def rubric_sha() -> str:
    from judge.prompt import RL_RUBRIC_ADDENDUM
    from judge.rubric import _JUDGE_INSTRUCTIONS

    return hashlib.sha256((_JUDGE_INSTRUCTIONS + RL_RUBRIC_ADDENDUM).encode("utf-8")).hexdigest()[
        :16
    ]


_DB_DIGEST: str | None = None


def ontology_db_digest() -> str:
    """Identity of the ontology database the floor rests on: the `meta`
    table build_ontology_db.py writes (built_at + source sha256s).
    """
    global _DB_DIGEST
    if _DB_DIGEST is None:
        db = require_ontology()
        h = hashlib.sha256()
        try:
            conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            try:
                for k, v in conn.execute("SELECT key, value FROM meta ORDER BY key"):
                    h.update(f"{k}={v}".encode("utf-8"))
            finally:
                conn.close()
        except sqlite3.Error:
            h.update(f"{db}:{db.stat().st_size}".encode("utf-8"))
        _DB_DIGEST = h.hexdigest()[:16]
    return _DB_DIGEST


def _collapse(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


class LivePanel:
    def __init__(
        self,
        *,
        judges: list[str] | None = None,
        cache_path: Path | str | None = None,
        raw_out: Path | str | None = None,
        concurrency: int = 48,
        bar: float = DEFAULT_BAR,
        fallback_halt_share: float = 0.0,
        flag_halt_share: float = 0.02,
        replay_only: bool = False,
        retry_backoff: float = 1.0,
        min_judges: int | None = None,
        judge_parse_retries: int = 1,
        _call: Callable[[str, str], tuple[str, dict]] | None = None,
        _check: Callable[[str, str], dict] | None = None,
    ):
        self.judges = list(judges) if judges else list(DEFAULT_JUDGES)
        unknown = [j for j in self.judges if j not in ALL_JUDGES]
        if unknown:
            raise ValueError(f"unknown judge(s) {unknown}; known: {list(ALL_JUDGES)}")
        if not self.judges:
            raise ValueError("at least one judge is required")
        self.min_judges = int(min_judges or len(self.judges))
        self.bar = float(bar)
        self.fallback_halt_share = float(fallback_halt_share)
        self.flag_halt_share = float(flag_halt_share)
        self.replay_only = bool(replay_only)
        self.retry_backoff = float(retry_backoff)
        self.judge_parse_retries = max(0, int(judge_parse_retries))
        self._call = _call
        self._check = _check
        self.canned = _call is not None
        self.cache_path = Path(cache_path) if cache_path else None
        self.raw_out = Path(raw_out) if raw_out else None
        self._sem = {
            p: threading.BoundedSemaphore(max(1, int(concurrency)))
            for p in {judge_spec(j)["provider"] for j in self.judges}
        }
        self._lock = threading.Lock()
        self._file_lock = threading.Lock()
        self._inflight: dict[str, threading.Lock] = {}
        self._cache: dict[str, dict] = {}
        self._clients: dict[str, object] = {}
        self.spent_usd = 0.0
        self.halt_reason: str | None = None
        self._step = self._fresh_step()
        self.totals = self._fresh_step()
        if not self.canned and not self.replay_only:
            require_ontology()
            check_reachable(self.judges)
        self._identity = {
            "judges": [(j, judge_spec(j)["model"]) for j in self.judges],
            "bar": self.bar,
            "rubric": rubric_sha(),
            "ontology_db": "canned" if self._check is not None else ontology_db_digest(),
            "guard": sr.DX_SCORER_ONTOLOGY,
            "scorer": DX_SCORER_PANEL,
            "min_judges": self.min_judges,
            "floor_head_match": False,
        }
        self._load_cache()
        self._load_spend()
        if not self.canned and not self.replay_only:
            self.probe()

    @staticmethod
    def _fresh_step() -> dict:
        return {
            "scored": 0,
            "cache_hits": 0,
            "fallback": 0,
            "flagged": 0,
            "vetoed": 0,
            "cost_usd": 0.0,
            "judge_calls": 0,
            "judge_failures": 0,
        }

    def _bump(self, k: str, v: float = 1) -> None:
        with self._lock:
            self._step[k] += v
            self.totals[k] += v

    def step_stats(self) -> dict:
        with self._lock:
            return dict(self._step)

    def begin_step(self) -> dict:
        """Close the previous env step: evaluate its shares against the halt
        thresholds and start fresh counters. Returns the closed stats.
        """
        with self._lock:
            closed = dict(self._step)
            self._step = self._fresh_step()
        n = closed["scored"]
        if n:
            fb = closed["fallback"] / n
            fl = closed["flagged"] / n
            if fb > self.fallback_halt_share:
                self.halt_reason = (
                    f"fallback share {fb:.2f} > {self.fallback_halt_share} over {n} pairs"
                )
            elif fl > self.flag_halt_share:
                self.halt_reason = f"flag share {fl:.2f} > {self.flag_halt_share} over {n} pairs"
        return closed

    def key(self, gold: str, pred: str) -> str:
        body = {"gold": sr.normalize(gold), "pred": _collapse(pred), **self._identity}
        return hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()

    def _load_cache(self) -> None:
        if not self.cache_path or not self.cache_path.exists():
            return
        for line in self.cache_path.open(encoding="utf-8"):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            k = row.get("key")
            if k:
                self._cache[k] = row

    @property
    def _spend_path(self) -> Path | None:
        return (
            self.cache_path.with_name(self.cache_path.stem + ".spend.json")
            if self.cache_path
            else None
        )

    def _load_spend(self) -> None:
        """The run's spend so far survives a restart (trainer.resume_mode=auto
        restarts the process; spend is accounted per RUN, not per process).
        """
        sp = self._spend_path
        if sp and sp.exists():
            try:
                self.spent_usd = float(json.loads(sp.read_text(encoding="utf-8"))["spent_usd"])
            except (ValueError, KeyError, OSError):
                self.spent_usd = 0.0

    def _save_spend(self) -> None:
        sp = self._spend_path
        if sp:
            sp.parent.mkdir(parents=True, exist_ok=True)
            tmp = sp.with_suffix(".tmp")
            tmp.write_text(json.dumps({"spent_usd": self.spent_usd}), encoding="utf-8")
            tmp.replace(sp)

    def probe(self) -> bool:
        """One fixture pair through every requested judge: each must answer
        and parse. Bad credentials or a dead endpoint fail HERE, not after
        the first terminal episodes start falling back.
        """
        prompt = judge_prompt("Fabry disease", "Fabry disease")
        for j in self.judges:
            text, _, err = self._judge_one(j, prompt)
            if text is None or detail_from_reply(text, normalize=self._normalize(j)) is None:
                raise RuntimeError(f"judge {j} failed its probe: {err or 'reply did not parse'}")
        return True

    def _store(self, block: dict) -> None:
        with self._lock:
            self._cache[block["key"]] = block
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self._file_lock, self.cache_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(block, ensure_ascii=False) + "\n")

    def _log_raw(self, rows: list[dict]) -> None:
        if not self.raw_out or not rows:
            return
        self.raw_out.parent.mkdir(parents=True, exist_ok=True)
        with self._file_lock, self.raw_out.open("a", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    def _client(self, provider: str):
        c = self._clients.get(provider)
        if c is not None:
            return c
        with self._lock:
            c = self._clients.get(provider)
            if c is None:
                if provider in ("openai", "openai_reasoning"):
                    from openai import OpenAI

                    key = os.environ.get("OPENAI_API_KEY")
                    if not key:
                        raise RuntimeError("OPENAI_API_KEY is not set")
                    c = OpenAI(
                        api_key=key,
                        max_retries=3,
                        timeout=600.0 if provider == "openai_reasoning" else 120.0,
                    )
                else:
                    raise RuntimeError(f"unknown judge provider {provider!r}")
                self._clients[provider] = c
        return c

    @staticmethod
    def _normalize(judge: str) -> bool:
        """A `parse: "normalized"` judge is read by the parser ITS VALIDATION USED
        (`panel_judge.normalize_reply`: reasoning stripped, the fenced object taken,
        an explicit no-match row dropped). Other judges keep the production read.
        """
        return judge_spec(judge).get("parse") == "normalized"

    def _provider_call(self, judge: str, prompt: str) -> tuple[str, dict]:
        spec = judge_spec(judge)
        provider = spec["provider"]
        model = spec["model"]
        client = self._client(provider)
        if provider == "openai_reasoning":
            kwargs = dict(
                model=model,
                max_completion_tokens=REASONING_MAX_COMPLETION_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
        else:
            kwargs = dict(
                model=model,
                max_tokens=JUDGE_MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
            if provider == "openai":
                kwargs["temperature"] = 0.0
        r = client.chat.completions.create(**kwargs)
        u = getattr(r, "usage", None)
        usage = {
            "input_tokens": getattr(u, "prompt_tokens", 0),
            "output_tokens": getattr(u, "completion_tokens", 0),
        }
        return (r.choices[0].message.content or ""), usage

    def _judge_one(self, judge: str, prompt: str) -> tuple[str | None, dict | None, str | None]:
        call = self._call or self._provider_call
        provider = judge_spec(judge)["provider"]
        model = judge_spec(judge)["model"]
        err = None
        for attempt in range(RETRY_ATTEMPTS):
            try:
                with self._sem[provider]:
                    text, usage = call(judge, prompt)
                self._bump("judge_calls")
                cost = usage_cost(model, usage)
                if cost:
                    self._account(cost)
                return text, usage, None
            except Exception as e:
                err = f"{type(e).__name__}: {str(e)[:160]}"
                self._bump("judge_failures")
                if attempt + 1 < RETRY_ATTEMPTS and self.retry_backoff > 0:
                    time.sleep(self.retry_backoff * (2**attempt))
        return None, None, err

    def _account(self, cost: float) -> None:
        with self._lock:
            self.spent_usd += cost
            self._step["cost_usd"] += cost
            self.totals["cost_usd"] += cost
            self._save_spend()

    def score(self, gold: str, pred: str, ctx: dict | None = None) -> dict:
        t0 = time.perf_counter()
        pred_raw = str(pred or "")
        pred_c = _collapse(pred_raw)
        k = self.key(gold, pred_c)
        with self._lock:
            hit = self._cache.get(k)
        if hit is not None:
            return self._served(hit, "cache", t0)
        if self.replay_only:
            block = self._fallback(gold, pred_raw, {}, "replay_miss")
            block.update(
                injection_flags=injection_flags(pred_raw),
                cost_usd=0.0,
                key=k,
                gold=gold,
                pred=pred_raw,
                bar=self.bar,
                latency_ms=int((time.perf_counter() - t0) * 1000),
            )
            self._bump("scored")
            return block
        with self._lock:
            lock = self._inflight.setdefault(k, threading.Lock())
        with lock:
            with self._lock:
                hit = self._cache.get(k)
            if hit is not None:
                return self._served(hit, "cache", t0)
            block = self._score_miss(gold, pred_raw, k, ctx)
            block["latency_ms"] = int((time.perf_counter() - t0) * 1000)
            if block["source"] != "ontology_fallback":
                self._store(block)
        with self._lock:
            self._inflight.pop(k, None)
        return dict(block)

    def _served(self, block: dict, source: str, t0: float) -> dict:
        out = dict(block)
        out["source"] = source
        out["latency_ms"] = int((time.perf_counter() - t0) * 1000)
        self._bump("scored")
        self._bump("cache_hits")
        if out.get("injection_flags"):
            self._bump("flagged")
        return out

    _PARSE_REPAIR = (
        "\n\nReturn ONLY the JSON object specified above — no prose, "
        "no markdown fences, no extra keys."
    )

    def _judge_scored(
        self, judge: str, base_prompt: str, k: str, gold: str, pred: str, ctx: dict | None
    ):
        """One judge's scored detail, retrying a SUCCESSFUL-but-unparseable reply
        up to judge_parse_retries times with a JSON-repair nudge before giving up. Transport failures are already retried inside
        _judge_one and the nudge cannot fix them, so a None reply stops early.
        Returns (judge, detail|None, raw_rows, cost).
        """
        model = judge_spec(judge)["model"]
        rows: list[dict] = []
        cost = 0.0
        for attempt in range(self.judge_parse_retries + 1):
            prompt = base_prompt if attempt == 0 else base_prompt + self._PARSE_REPAIR
            text, usage, err = self._judge_one(judge, prompt)
            cost += usage_cost(model, usage)
            rows.append(
                {
                    "judge": judge,
                    "key": k,
                    "gold": gold,
                    "pred": pred,
                    "raw": text,
                    "error": err,
                    "usage": usage,
                    "parse_attempt": attempt,
                    "ctx": ctx or {},
                    "ts": time.time(),
                }
            )
            d = (
                detail_from_reply(text, normalize=self._normalize(judge))
                if text is not None
                else None
            )
            if d is not None:
                return judge, d, rows, cost
            if text is None:
                break
        return judge, None, rows, cost

    def _score_miss(self, gold: str, pred: str, k: str, ctx: dict | None) -> dict:
        flags = injection_flags(pred)
        details: dict[str, dict] = {}
        raw_rows: list[dict] = []
        cost = 0.0
        base_prompt = judge_prompt(gold, pred)
        with ThreadPoolExecutor(max_workers=len(self.judges)) as ex:
            results = list(
                ex.map(
                    lambda j: self._judge_scored(j, base_prompt, k, gold, pred, ctx), self.judges
                )
            )
        for judge, d, rows, jcost in results:
            cost += jcost
            raw_rows.extend(rows)
            if d is not None:
                details[judge] = d
        self._log_raw(raw_rows)

        if len(details) < self.min_judges:
            block = self._fallback(gold, pred, details, "thin_panel")
        else:
            prefusion = statistics.median(float(d["t_reward"]) for d in details.values())
            verdict, guard_reason = None, None
            if prefusion < self.bar:
                verdict, gd = sr.guarded_verdict(gold, pred, check=self._check, for_floor=True)
                guard_reason = gd.get("reason")
            b = panel_block(details, verdict)
            fused = float(b["fused_t_reward"])
            dx = 1.0 if fused >= self.bar else 0.0
            if dx >= self.bar:
                veto, vd = sr.ontology_veto(gold, pred, check=self._check)
                if veto:
                    dx, guard_reason = min(dx, 0.5), vd.get("reason")
                    self._bump("vetoed")
            block = {
                "dx_score": dx,
                "dx_scorer": DX_SCORER_PANEL,
                "source": "panel",
                "fused_t_reward": fused,
                "prefusion_t_reward": float(b["prefusion_t_reward"]),
                "judges": b["judges"],
                "n_judges": int(b["n_judges"]),
                "verdict_source": b["verdict_source"],
                "ontology": b.get("ontology"),
                "guard_reason": guard_reason,
                "fallback_reason": None,
            }
        block.update(
            injection_flags=flags,
            cost_usd=round(cost, 6),
            key=k,
            gold=gold,
            pred=pred,
            bar=self.bar,
        )
        self._bump("scored")
        if flags:
            self._bump("flagged")
        return block

    def _fallback(self, gold: str, pred: str, details: dict, why: str) -> dict:
        d = sr.ontology_outcome_detail(gold, pred, check=self._check)
        ts = [float(x["t_reward"]) for x in details.values()]
        self._bump("fallback")
        return {
            "dx_score": float(d["score"]),
            "dx_scorer": DX_SCORER_FALLBACK,
            "source": "ontology_fallback",
            "fused_t_reward": (statistics.median(ts) if ts else None),
            "prefusion_t_reward": (statistics.median(ts) if ts else None),
            "judges": {
                j: {
                    "t_reward": round(float(x.get("t_reward") or 0.0), 4),
                    "full": int(x.get("full_count") or 0),
                    "core": int(x.get("core_count") or 0),
                    "partial": int(x.get("partial_count") or 0),
                    "gt_count": int(x.get("gt_count") or 0),
                    "pred_count": int(x.get("pred_count") or 0),
                }
                for j, x in details.items()
            },
            "n_judges": len(details),
            "verdict_source": "ontology_fallback",
            "ontology": d.get("raw_verdict"),
            "guard_reason": d.get("reason"),
            "fallback_reason": why,
        }


_PANEL: LivePanel | None = None


def configure(panel: LivePanel) -> None:
    global _PANEL
    _PANEL = panel


def _default_panel() -> LivePanel:
    """Built from the environment: LIVE_PANEL_CACHE (the run's cache file),
    LIVE_PANEL_REPLAY_ONLY=1 (a miss raises instead of paying), LIVE_PANEL_RAW.
    """
    global _PANEL
    if _PANEL is None:
        cache = os.environ.get("LIVE_PANEL_CACHE")
        if not cache:
            raise RuntimeError(
                "live_panel offline scoring needs LIVE_PANEL_CACHE (the run's panel_cache.jsonl)"
            )
        judges = [
            j.strip() for j in os.environ.get("LIVE_PANEL_JUDGES", "").split(",") if j.strip()
        ] or None
        _PANEL = LivePanel(
            judges=judges,
            cache_path=cache,
            raw_out=os.environ.get("LIVE_PANEL_RAW"),
            replay_only=os.environ.get("LIVE_PANEL_REPLAY_ONLY") == "1",
        )
    return _PANEL


def score_full(gt: str, pred: str) -> dict:
    b = _default_panel().score(gt, pred)
    samples = [float(v["t_reward"]) for v in (b.get("judges") or {}).values()]
    return {
        "t_reward": float(
            b["fused_t_reward"] if b.get("fused_t_reward") is not None else b["dx_score"]
        ),
        "outcome_reward": float(b["dx_score"]),
        "samples": samples,
        "ontology": b.get("ontology"),
        "source": b.get("source"),
        "dx_scorer": b.get("dx_scorer"),
    }


def score(gt: str, pred: str) -> float:
    return float(score_full(gt, pred)["outcome_reward"])


score.full = score_full
