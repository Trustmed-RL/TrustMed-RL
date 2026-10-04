"""Deterministic ontology check for diagnosis pairs: is B the same disease as A,
its parent/subtype, a sibling, or unrelated?
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
OLS = "https://www.ebi.ac.uk/ols4/api"
MAX_DEPTH = 4
_LAST_CALL: dict[str, float] = {}
_MIN_GAP = {"eutils": 0.4, "ols": 0.15}

CACHE_PATH = Path(__file__).with_name(".ontology_cache.json")
_cache: dict[str, dict] = {}
_cache_dirty = False

WEIGHT = {"FULL": 1.0, "CORE": 0.85, "PARTIAL": 0.50, "NONE": 0.0, "UNKNOWN": None}


ALLOW_NETWORK = "TRUSTMED_ONTOLOGY_ALLOW_NETWORK"


def _get(url: str, family: str) -> dict | None:
    """GET -> parsed json, rate-limited per API family, cached forever."""
    global _cache_dirty
    if url in _cache:
        return _cache[url]
    if os.environ.get(ALLOW_NETWORK) != "1":
        raise RuntimeError(
            f"ontology network leg is disabled ({url[:60]}...). The local "
            "database is the only supported backend — use "
            "`ontology_check.default_check_pair()`. To re-derive the "
            f"historical parity file set {ALLOW_NETWORK}=1 deliberately."
        )
    wait = _MIN_GAP[family] - (time.monotonic() - _LAST_CALL.get(family, 0.0))
    if wait > 0:
        time.sleep(wait)
    for attempt in range(5):
        _LAST_CALL[family] = time.monotonic()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "trustmed-ontology-check"})
            with urllib.request.urlopen(req, timeout=30) as r:
                out = json.load(r)
            _cache[url] = out
            _cache_dirty = True
            return out
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError) as e:
            code = getattr(e, "code", None)
            if code == 404:
                _cache[url] = None
                _cache_dirty = True
                return None
            time.sleep(1.5 * (attempt + 1))
    print(f"  [warn] gave up on {url[:100]}", file=sys.stderr)
    return None


def _load_cache() -> None:
    global _cache
    if CACHE_PATH.exists():
        try:
            _cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except ValueError:
            _cache = {}


def _save_cache() -> None:
    if _cache_dirty:
        CACHE_PATH.write_text(json.dumps(_cache), encoding="utf-8")


def _medgen_search(query: str, retmax: int = 10) -> list[dict]:
    q = urllib.parse.quote(query)
    d = _get(f"{EUTILS}/esearch.fcgi?db=medgen&term={q}&retmode=json&retmax={retmax}", "eutils")
    ids = (d or {}).get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []
    s = _get(f"{EUTILS}/esummary.fcgi?db=medgen&id={','.join(ids)}&retmode=json", "eutils")
    return [s["result"][i] for i in ids if i in (s or {}).get("result", {})]


def medgen_concept(term: str) -> dict | None:
    """term -> {cui, title, match} via MedGen, on a fixed ladder:"""
    t = term.strip().replace('"', "")
    for cand, kind in ((t, "exact_title"), (_norm(t), "exact_title_normalized")):
        docs = _medgen_search(f'"{cand}"[XTIT]', retmax=2)
        if docs:
            return {"cui": docs[0].get("conceptid"), "title": docs[0].get("title"), "match": kind}
    docs = _medgen_search(f'"{t}"[TITL]', retmax=5)
    if len({x.get("conceptid") for x in docs}) == 1 and docs:
        return {
            "cui": docs[0].get("conceptid"),
            "title": docs[0].get("title"),
            "match": "synonym_title",
        }
    docs = _medgen_search(t, retmax=5)
    if docs:
        return {
            "cui": docs[0].get("conceptid"),
            "title": docs[0].get("title"),
            "match": "top_hit_fuzzy",
        }
    return None


def medgen_relation(a: str, b: str) -> dict:
    ca, cb = medgen_concept(a), medgen_concept(b)
    out = {"source": "medgen_umls", "a": ca, "b": cb}
    if not ca or not cb or not ca.get("cui") or not cb.get("cui"):
        out["relation"] = "unmapped"
    elif ca["cui"] == cb["cui"]:
        out["relation"] = "equivalent"
        out["note"] = "same CUI (UMLS synonym lumping; can over-call equivalence)"
    else:
        out["relation"] = "unrelated"
    return out


def _norm(s: str) -> str:
    """Possessive- and punctuation-insensitive form: ontology labels write
    'Cushing disease' where clinicians write "Cushing's Disease".
    """
    t = s.strip().lower().replace("’", "'")
    t = t.replace("'s ", " ").removesuffix("'s")
    for ch in ",.-/()":
        t = t.replace(ch, " ")
    return " ".join(t.split())


def ols_lookup(term: str, ontology: str) -> dict | None:
    """term -> {id, iri, label, match} in one ontology."""
    for exact in ("true", "false"):
        q = urllib.parse.urlencode({"q": term, "ontology": ontology, "rows": 10, "exact": exact})
        d = _get(f"{OLS}/search?{q}", "ols")
        docs = [x for x in (d or {}).get("response", {}).get("docs", []) if x.get("obo_id")]
        if not docs:
            continue
        lab = [x for x in docs if x.get("label", "").strip().lower() == term.strip().lower()]
        if lab:
            doc, match = lab[0], "label"
        else:
            nrm = [x for x in docs if _norm(x.get("label", "")) == _norm(term)]
            if nrm:
                doc, match = nrm[0], "normalized_label"
            elif exact == "true":
                doc, match = min(docs, key=lambda x: len(x.get("label", ""))), "exact_synonym"
            else:
                doc, match = docs[0], "fuzzy"
        return {"id": doc["obo_id"], "iri": doc["iri"], "label": doc["label"], "match": match}
    return None


def _parents(iri: str, ontology: str) -> list[dict]:
    enc = urllib.parse.quote(urllib.parse.quote(iri, safe=""), safe="")
    d = _get(f"{OLS}/ontologies/{ontology}/terms/{enc}/hierarchicalParents?size=100", "ols")
    return (d or {}).get("_embedded", {}).get("terms", []) if d else []


def ancestors(iri: str, ontology: str, max_depth: int = MAX_DEPTH) -> dict[str, int]:
    """{ancestor_iri: min_depth} by BFS up hierarchicalParents."""
    seen: dict[str, int] = {}
    frontier = [iri]
    for depth in range(1, max_depth + 1):
        nxt = []
        for node in frontier:
            for p in _parents(node, ontology):
                pi = p.get("iri")
                if pi and pi not in seen:
                    seen[pi] = depth
                    nxt.append(pi)
        frontier = nxt
        if not frontier:
            break
    return seen


def ols_relation(a: str, b: str, ontology: str) -> dict:
    ta, tb = ols_lookup(a, ontology), ols_lookup(b, ontology)
    out = {"source": ontology, "a": ta, "b": tb}
    if not ta or not tb:
        out["relation"] = "unmapped"
        return out
    if ta["id"] == tb["id"]:
        out["relation"] = "equivalent"
        return out
    anc_a = ancestors(ta["iri"], ontology)
    anc_b = ancestors(tb["iri"], ontology)
    if tb["iri"] in anc_a:
        out["relation"] = "a_subtype_of_b"
        out["depth"] = anc_a[tb["iri"]]
    elif ta["iri"] in anc_b:
        out["relation"] = "b_subtype_of_a"
        out["depth"] = anc_b[ta["iri"]]
    else:
        shared = set(anc_a) & set(anc_b)
        if any(anc_a[s] == 1 and anc_b[s] == 1 for s in shared):
            out["relation"] = "siblings"
        elif shared:
            s = min(shared, key=lambda x: anc_a[x] + anc_b[x])
            out["relation"] = "distant"
            out["common_ancestor_depths"] = [anc_a[s], anc_b[s]]
        else:
            out["relation"] = "unrelated"
    return out


_QUALIFIER_SPLITS = (
    " due to ",
    " secondary to ",
    " caused by ",
    " associated with ",
    " complicated by ",
    " with ",
    " presenting ",
    " as a ",
    " in the setting of ",
    " status post ",
    ",",
)


def _head(term: str) -> str:
    """The concept-bearing head of a compound clinical phrase: 'Median Arcuate
    Ligament Syndrome complicated by ...' -> 'Median Arcuate Ligament
    Syndrome'. Split points are qualifier connectives, first hit wins.
    """
    t = " " + term.strip() + " "
    cut = len(t)
    for sep in _QUALIFIER_SPLITS:
        i = t.lower().find(sep)
        if i > 0:
            cut = min(cut, i)
    return t[:cut].strip(" ,")


def combine_legs(a: str, b: str, legs: list[dict], head_check=None) -> dict:
    """The fixed combination rules over three resolved legs (module doc)."""
    rels = {l["source"]: l["relation"] for l in legs}
    hier = [l for l in legs if l["relation"] in ("a_subtype_of_b", "b_subtype_of_a")]
    equiv = [l for l in legs if l["relation"] == "equivalent"]
    tsib = [l for l in legs if l["relation"] == "subtype_siblings"]
    sib = [l for l in legs if l["relation"] == "siblings"]
    mapped = [l for l in legs if l["relation"] != "unmapped"]

    flags = []
    if hier:
        dirs = {l["relation"] for l in hier}
        verdict = "CORE"
        confidence = "high" if (len(hier) >= 2 and len(dirs) == 1) or not equiv else "medium"
        if len(dirs) > 1:
            flags.append("sources disagree on subtype direction")
            confidence = "low"
        if equiv:
            flags.append(
                f"{'/'.join(l['source'] for l in equiv)} lumps the two "
                f"terms into one concept while an ontology splits them "
                f"hierarchically -- the known UMLS over-call pattern"
            )
        detail = hier[0]
    elif tsib:
        verdict, confidence, detail = "CORE", "medium", tsib[0]
        flags.append(
            f"{tsib[0]['source']} types both terms as subtypes of one "
            f"disorder ({tsib[0].get('shared_parent_label') or tsib[0].get('shared_parent')})"
            f" -- core disease matches, subtype qualifier differs"
        )
        if equiv:
            flags.append(
                f"{'/'.join(l['source'] for l in equiv)} lumps the two "
                f"subtypes into one concept -- the known UMLS over-call pattern"
            )
    elif equiv:
        verdict = "FULL"
        confidence = "high" if len(equiv) >= 2 else "low"
        if confidence == "low":
            flags.append(f"equivalence claimed by {equiv[0]['source']} only")
        detail = equiv[0]
    elif len(sib) >= 2:
        verdict, confidence, detail = "PARTIAL", "medium", sib[0]
    elif mapped:
        verdict, confidence, detail = "NONE", "high" if len(mapped) >= 2 else "low", None
        if sib:
            flags.append(f"sibling claim by {sib[0]['source']} only -- ignored")
    else:
        verdict, confidence, detail = "UNKNOWN", "none", None

    if head_check is not None and verdict in ("UNKNOWN", "NONE"):
        ha, hb = _head(a), _head(b)
        if (ha.lower(), hb.lower()) != (a.strip().lower(), b.strip().lower()) and ha and hb:
            hres = head_check(ha, hb)
            if hres["verdict"] in ("FULL", "CORE"):
                orig = verdict
                verdict = "CORE"
                confidence = "medium" if hres["confidence"] == "high" else "low"
                flags = flags + [
                    f"full strings were {orig}; heads {ha!r} vs {hb!r} are "
                    f"{hres['verdict']} ({hres['confidence']}) -- core disease "
                    f"matches, qualifiers differ"
                ]
                return {
                    "a": a,
                    "b": b,
                    "verdict": verdict,
                    "implied_weight": WEIGHT[verdict],
                    "confidence": confidence,
                    "direction": "head_terms_match",
                    "depth": None,
                    "flags": flags,
                    "per_source": {l["source"]: l for l in legs},
                    "head_check": {
                        "head_a": ha,
                        "head_b": hb,
                        "head_verdict": hres["verdict"],
                        "head_relations": hres["relations"],
                    },
                    "relations": rels,
                }

    return {
        "a": a,
        "b": b,
        "verdict": verdict,
        "implied_weight": WEIGHT[verdict],
        "confidence": confidence,
        "direction": (detail or {}).get("relation"),
        "depth": (detail or {}).get("depth"),
        "flags": flags,
        "per_source": {l["source"]: l for l in legs},
        "relations": rels,
    }


def check_pair(a: str, b: str, _allow_head_fallback: bool = True) -> dict:
    legs = [medgen_relation(a, b), ols_relation(a, b, "mondo"), ols_relation(a, b, "ordo")]
    head = (
        (lambda ha, hb: check_pair(ha, hb, _allow_head_fallback=False))
        if _allow_head_fallback
        else None
    )
    return combine_legs(a, b, legs, head_check=head)


def default_check_pair():
    """The ONLY check_pair a consumer may call: the local database."""
    from judge.ontology_local import check_pair as local_check

    return local_check


def _iter_pairs(path: Path):
    """Yield (case_id, predicted, gold) from judge_disagreements.json, an
    eval results json, or a generic jsonl.
    """
    if path.suffix == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                yield (
                    r.get("case_id"),
                    r.get("predicted") or r.get("predicted_diagnosis"),
                    r.get("ground_truth") or r.get("ground_truth_diagnosis"),
                )
        return
    d = json.loads(path.read_text(encoding="utf-8"))
    rows = d.get("disagreements") or d.get("results") or (d if isinstance(d, list) else [])
    for r in rows:
        yield (
            r.get("case_id"),
            r.get("predicted") or r.get("predicted_diagnosis"),
            r.get("ground_truth") or r.get("ground_truth_diagnosis"),
        )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="one pair")
    c.add_argument("term_a")
    c.add_argument("term_b")
    b = sub.add_parser("batch", help="every (predicted, gold) pair in a file")
    b.add_argument("--in", dest="inp", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    _load_cache()
    try:
        if args.cmd == "check":
            res = check_pair(args.term_a, args.term_b)
            print(json.dumps(res, indent=1, ensure_ascii=False))
        else:
            seen: dict[tuple, dict] = {}
            out = []
            pairs = list(_iter_pairs(args.inp))
            for i, (cid, pred, gold) in enumerate(pairs, 1):
                if not pred or not gold:
                    continue
                key = (pred.strip().lower(), gold.strip().lower())
                if key not in seen:
                    seen[key] = check_pair(pred, gold)
                    print(
                        f"[{i}/{len(pairs)}] {pred[:38]!r} vs {gold[:38]!r} "
                        f"-> {seen[key]['verdict']} ({seen[key]['confidence']})",
                        flush=True,
                    )
                out.append({"case_id": cid, **seen[key]})
            counts: dict[str, int] = {}
            for r in out:
                counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
            args.out.write_text(
                json.dumps(
                    {
                        "what": "deterministic ontology adjudication (MedGen/UMLS + MONDO + ORDO)",
                        "rules": "hierarchy beats lumping; FULL needs equivalence with no "
                        "hierarchical dissent; see ontology_check.py docstring",
                        "n": len(out),
                        "verdict_counts": counts,
                        "results": out,
                    },
                    indent=1,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            print(f"\n{counts} -> {args.out}")
    finally:
        _save_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
