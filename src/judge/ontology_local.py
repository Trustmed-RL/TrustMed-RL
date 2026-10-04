"""ontology_check's legs against the LOCAL database — offline, keyless, fast."""

from __future__ import annotations

import os
import re
import sqlite3
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from judge.build_ontology_db import fold_token
from judge.ontology_check import MAX_DEPTH, _norm, combine_legs

HERE = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("TRUSTMED_ONTOLOGY_DB", str(HERE / "ontology_local.db")))

_TOKEN = re.compile(r"[a-z0-9]+")

RARITY_BAND = 2


def _cand(n: int) -> str:
    """Subquery: concepts carrying ALL n tokens (across any of their names) —
    the document-level AND both search APIs apply to a plain query.
    """
    marks = ",".join(f":t{i}" for i in range(n))
    return (
        f"SELECT cid FROM tokens WHERE src=:src AND tok IN ({marks}) "
        f"GROUP BY cid HAVING COUNT(DISTINCT tok)={n}"
    )


def _cand_args(src: str, toks: list[str]) -> dict:
    return {"src": src, **{f"t{i}": t for i, t in enumerate(toks)}}


class LocalOntology:
    CACHE_MAX = 50_000
    """All three legs over one SQLite file. Stateless queries + memo caches."""

    def __init__(self, db_path: Path | str = DB_PATH):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"{self.db_path} not found — set TRUSTMED_ONTOLOGY_DB to an ontology_local.db "
                "(the trustmed-full-pack-v4 dataset ships one with its sha256) or build it with "
                "build_ontology_db.py (sources download keyless)"
            )
        self._local = threading.local()
        self._cache_lock = threading.Lock()
        self._concept_cache: dict[tuple, dict | None] = {}
        self._pair_cache: dict[tuple, dict] = {}

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(
                f"file:{self.db_path}?mode=ro", uri=True, check_same_thread=False
            )
            self._local.conn = conn
        return conn

    def _one(self, sql: str, args) -> str | None:
        row = self._conn().execute(sql, args).fetchone()
        return row[0] if row else None

    def _label(self, src: str, cid: str) -> str | None:
        return self._one("SELECT label FROM concepts WHERE src=? AND cid=?", (src, cid))

    def _fuzzy_best(self, src: str, toks: list[str]) -> str | None:
        """The plain-search top hit: highest token coverage wins, ties break
        to the shortest LABEL, then id. Label length, not shortest-name-
        anywhere: a concept carrying a two-letter synonym would otherwise
        beat the generic concept on every one-token tie, which is how
        "cardiac amyloidosis" once resolved past plain "Amyloidosis".
        """
        if not toks:
            return None
        marks = ",".join(f":t{i}" for i in range(len(toks)))
        args = _cand_args(src, toks)
        df = (
            self._conn()
            .execute(
                f"SELECT tok, COUNT(*) c FROM tokens WHERE src=:src "
                f"AND tok IN ({marks}) GROUP BY tok ORDER BY c, tok",
                args,
            )
            .fetchall()
        )
        if not df:
            return None
        required = [t for t, c in df if c <= df[0][1] * RARITY_BAND]
        req_marks = ",".join(f":r{i}" for i in range(len(required)))
        args = {
            **args,
            "mincov": (len(toks) + 1) // 2,
            "nreq": len(required),
            **{f"r{i}": t for i, t in enumerate(required)},
        }
        return self._one(
            f"""WITH cand AS (
                    SELECT cid, COUNT(DISTINCT tok) AS cov FROM tokens
                    WHERE src=:src AND tok IN ({marks})
                    GROUP BY cid HAVING COUNT(DISTINCT tok) >= :mincov)
                SELECT cand.cid FROM cand
                JOIN concepts co ON co.src = :src AND co.cid = cand.cid
                WHERE cov = (SELECT MAX(cov) FROM cand)
                  AND (SELECT COUNT(DISTINCT t.tok) FROM tokens t
                       WHERE t.src = :src AND t.cid = cand.cid
                         AND t.tok IN ({req_marks})) = :nreq
                ORDER BY LENGTH(co.label), cand.cid
                LIMIT 1""",
            args,
        )

    def _def_phrase(self, src: str, toks: list[str]) -> str | None:
        """Exact-phrase containment in DEFINITION text, shortest label wins."""
        if not toks or sum(len(t) for t in toks) < 6:
            return None
        phrase = '"' + " ".join(toks) + '"'
        return self._one(
            "SELECT d.cid FROM defs d "
            "JOIN concepts co ON co.src = d.src AND co.cid = d.cid "
            "WHERE defs MATCH ? AND d.src = ? "
            "ORDER BY LENGTH(co.label), d.cid LIMIT 1",
            (phrase, src),
        )

    def _memo(self, cache: dict, key, compute):
        """Bounded, locked memo: compute outside the lock (SQLite work), store
        under it, evict the oldest entry past CACHE_MAX.
        """
        with self._cache_lock:
            if key in cache:
                return cache[key]
        value = compute()
        with self._cache_lock:
            if key not in cache:
                cache[key] = value
                while len(cache) > self.CACHE_MAX:
                    cache.pop(next(iter(cache)))
            return cache[key]

    def medgen_concept(self, term: str) -> dict | None:
        return self._memo(self._concept_cache, ("medgen", term), lambda: self._medgen_concept(term))

    def _medgen_concept(self, term: str) -> dict | None:
        t = term.strip().replace('"', "")
        toks = _TOKEN.findall(t.lower())
        if not toks:
            return None

        def hit(cui: str | None, match: str) -> dict | None:
            if cui is None:
                return None
            return {"cui": cui, "title": self._label("medgen", cui), "match": match}

        for col, val, kind in (
            ("name_lower", t.lower(), "exact_title"),
            ("name_norm", _norm(t), "exact_title_normalized"),
        ):
            got = hit(
                self._one(
                    f"SELECT cid FROM names WHERE src='medgen' AND {col}=? "
                    "AND kind='pref' ORDER BY cid LIMIT 1",
                    (val,),
                ),
                kind,
            )
            if got:
                return got
        folded = sorted({fold_token(t) for t in toks})
        args = {**_cand_args("medgen", folded), "phr": "% " + " ".join(toks) + " %"}
        phrased = [
            r[0]
            for r in self._conn().execute(
                f"SELECT DISTINCT cid FROM names WHERE src=:src "
                f"AND cid IN ({_cand(len(folded))}) AND kind != 'id' "
                "AND name_tok LIKE :phr LIMIT 2",
                args,
            )
        ]
        if len(phrased) == 1:
            return hit(phrased[0], "synonym_title")
        return hit(self._fuzzy_best("medgen", folded), "top_hit_fuzzy") or hit(
            self._def_phrase("medgen", toks), "def_mention"
        )

    def medgen_relation(self, a: str, b: str) -> dict:
        ca, cb = self.medgen_concept(a), self.medgen_concept(b)
        out = {"source": "medgen_umls", "a": ca, "b": cb}
        if not ca or not cb or not ca.get("cui") or not cb.get("cui"):
            out["relation"] = "unmapped"
        elif ca["cui"] == cb["cui"]:
            out["relation"] = "equivalent"
            out["note"] = "same CUI (UMLS synonym lumping; can over-call equivalence)"
        else:
            out["relation"] = "unrelated"
        return out

    def ols_lookup(self, term: str, ontology: str) -> dict | None:
        return self._memo(
            self._concept_cache, (ontology, term), lambda: self._ols_lookup(term, ontology)
        )

    def _ols_lookup(self, term: str, ontology: str) -> dict | None:
        """The online leg's candidate-choice ladder: exact label, normalized
        label, then — among exact-name hits — the shortest label, then the
        top fuzzy hit. The exact pass is any-name equality; the fuzzy pass is
        the all-token candidate set of the normalized term (which is how a
        possessive query still finds its apostrophe-free label).
        """
        tl = term.strip().lower()
        tn = _norm(term)
        if not tl:
            return None

        def doc(cid: str | None, match: str) -> dict | None:
            if cid is None:
                return None
            row = (
                self._conn()
                .execute("SELECT iri, label FROM concepts WHERE src=? AND cid=?", (ontology, cid))
                .fetchone()
            )
            return {
                "id": cid,
                "iri": row[0] if row else None,
                "label": row[1] if row else None,
                "match": match,
            }

        got = doc(
            self._one(
                "SELECT cid FROM concepts WHERE src=? AND label_lower=? ORDER BY cid LIMIT 1",
                (ontology, tl),
            ),
            "label",
        )
        if got:
            return got
        got = doc(
            self._one(
                "SELECT c.cid FROM concepts c WHERE c.src=? AND c.label_norm=? "
                "AND EXISTS(SELECT 1 FROM names n WHERE n.src=c.src "
                "AND n.cid=c.cid AND n.name_lower=?) ORDER BY c.cid LIMIT 1",
                (ontology, tn, tl),
            ),
            "normalized_label",
        )
        if got:
            return got
        got = doc(
            self._one(
                "SELECT c.cid FROM concepts c "
                "JOIN (SELECT DISTINCT cid FROM names WHERE src=? "
                "      AND name_lower=?) x ON c.src=? AND c.cid=x.cid "
                "ORDER BY LENGTH(c.label), c.label, c.cid LIMIT 1",
                (ontology, tl, ontology),
            ),
            "exact_synonym",
        )
        if got:
            return got
        toks = sorted({fold_token(t) for t in _TOKEN.findall(tn)})
        if not toks:
            return None
        got = doc(
            self._one(
                "SELECT cid FROM concepts WHERE src=? AND label_norm=? ORDER BY cid LIMIT 1",
                (ontology, tn),
            ),
            "normalized_label",
        )
        if got:
            return got
        return doc(self._fuzzy_best(ontology, toks), "fuzzy") or doc(
            self._def_phrase(ontology, _TOKEN.findall(tn)), "def_mention"
        )

    def ancestors(self, cid: str, ontology: str, max_depth: int = MAX_DEPTH) -> dict[str, int]:
        """{ancestor_cid: min_depth} by BFS up the parents table — the same
        bounded walk the online leg does over hierarchicalParents.
        """
        seen: dict[str, int] = {}
        frontier = [cid]
        conn = self._conn()
        for depth in range(1, max_depth + 1):
            if not frontier:
                break
            marks = ",".join("?" * len(frontier))
            rows = conn.execute(
                f"SELECT DISTINCT parent FROM parents WHERE src=? AND child IN ({marks})",
                [ontology] + frontier,
            ).fetchall()
            frontier = [p for (p,) in rows if p not in seen]
            for p in frontier:
                seen[p] = depth
        return seen

    ORDO_SUBTYPE = "ORDO:557494"
    ORDO_DISORDER = "ORDO:557493"

    def _ordo_parents(self, cid: str) -> frozenset:
        key = ("ordo_parents", cid)
        if key not in self._concept_cache:
            rows = (
                self._conn()
                .execute("SELECT parent FROM parents WHERE src='ordo' AND child=?", (cid,))
                .fetchall()
            )
            self._concept_cache[key] = frozenset(p for (p,) in rows)
        return self._concept_cache[key]

    def ordo_subtype_sibling_parent(self, a: str, b: str) -> str | None:
        """The disorder both ORDO concepts are subtypes OF, or None."""
        pa, pb = self._ordo_parents(a), self._ordo_parents(b)
        if self.ORDO_SUBTYPE not in pa or self.ORDO_SUBTYPE not in pb:
            return None
        for p in sorted(pa & pb):
            if p in (self.ORDO_SUBTYPE, self.ORDO_DISORDER):
                continue
            if self.ORDO_DISORDER in self._ordo_parents(p):
                return p
        return None

    def ols_relation(self, a: str, b: str, ontology: str) -> dict:
        ta, tb = self.ols_lookup(a, ontology), self.ols_lookup(b, ontology)
        out = {"source": ontology, "a": ta, "b": tb}
        if not ta or not tb:
            out["relation"] = "unmapped"
            return out
        if ta["id"] == tb["id"]:
            out["relation"] = "equivalent"
            return out
        anc_a = self.ancestors(ta["id"], ontology)
        anc_b = self.ancestors(tb["id"], ontology)
        if tb["id"] in anc_a:
            out["relation"] = "a_subtype_of_b"
            out["depth"] = anc_a[tb["id"]]
        elif ta["id"] in anc_b:
            out["relation"] = "b_subtype_of_a"
            out["depth"] = anc_b[ta["id"]]
        else:
            shared = set(anc_a) & set(anc_b)
            typed = (
                self.ordo_subtype_sibling_parent(ta["id"], tb["id"]) if ontology == "ordo" else None
            )
            if typed is not None:
                out["relation"] = "subtype_siblings"
                out["shared_parent"] = typed
                out["shared_parent_label"] = self._label("ordo", typed)
            elif any(anc_a[s] == 1 and anc_b[s] == 1 for s in shared):
                out["relation"] = "siblings"
            elif shared:
                s = min(shared, key=lambda x: anc_a[x] + anc_b[x])
                out["relation"] = "distant"
                out["common_ancestor_depths"] = [anc_a[s], anc_b[s]]
            else:
                out["relation"] = "unrelated"
        return out

    def check_pair(self, a: str, b: str, _allow_head_fallback: bool = True) -> dict:
        key = (a, b, _allow_head_fallback)

        def compute():
            legs = [
                self.medgen_relation(a, b),
                self.ols_relation(a, b, "mondo"),
                self.ols_relation(a, b, "ordo"),
            ]
            head = (
                (lambda ha, hb: self.check_pair(ha, hb, _allow_head_fallback=False))
                if _allow_head_fallback
                else None
            )
            out = combine_legs(a, b, legs, head_check=head)
            out["backend"] = "local"
            return out

        return self._memo(self._pair_cache, key, compute)


_ONT = LocalOntology(DB_PATH)
check_pair = _ONT.check_pair


if __name__ == "__main__":
    import json as _json

    args = sys.argv[1:]
    if len(args) != 2:
        raise SystemExit("usage: python ontology_local.py <term_a> <term_b>")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(_json.dumps(check_pair(args[0], args[1]), indent=1, ensure_ascii=False))
