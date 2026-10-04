"""Build the LOCAL ontology database `ontology_local.db` for ontology_check."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from judge.ontology_check import _norm

HERE = Path(__file__).resolve().parent
DEFAULT_DB = HERE / "ontology_local.db"
DEFAULT_SRC = HERE / "ontology_src"

SOURCES = {
    "NAMES.RRF.gz": "https://ftp.ncbi.nlm.nih.gov/pub/medgen/NAMES.RRF.gz",
    "MGCONSO.RRF.gz": "https://ftp.ncbi.nlm.nih.gov/pub/medgen/MGCONSO.RRF.gz",
    "MGDEF.RRF.gz": "https://ftp.ncbi.nlm.nih.gov/pub/medgen/MGDEF.RRF.gz",
    "mondo.obo": "https://purl.obolibrary.org/obo/mondo.obo",
    "ordo_orphanet.owl": (
        "https://www.orphadata.com/data/ontologies/ordo/last_version/ordo_orphanet.owl"
    ),
}

OWL = "{http://www.w3.org/2002/07/owl#}"
RDF = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"
RDFS = "{http://www.w3.org/2000/01/rdf-schema#}"
EFO = "{http://www.ebi.ac.uk/efo/}"
SKOS = "{http://www.w3.org/2004/02/skos/core#}"
ORDO_NS = "http://www.orpha.net/ORDO/"
PART_OF = "http://purl.obolibrary.org/obo/BFO_0000050"

_TOKEN = re.compile(r"[a-z0-9]+")


def fold_token(tok: str) -> str:
    """Naive singular fold, applied identically at build and query time so
    plural/singular queries meet the index the way eutils' automatic term
    mapping lets them meet the online one ("carcinomas" must still find
    "carcinoma"). Consistency is what makes the naivety safe.
    """
    if tok.endswith("s") and not tok.endswith("ss") and len(tok) > 3:
        return tok[:-1]
    return tok


def parse_mondo_obo(text: str):
    """[Term] blocks -> rows. Skips obsolete terms, keeps every synonym scope
    (OLS's `synonym` search field indexes exact/related/broad/narrow alike).
    """
    concepts, names, parents = [], [], []
    donations: list[tuple[str, str]] = []
    cur: dict | None = None

    def flush():
        if not cur or not cur.get("id"):
            return
        if cur.get("obsolete"):
            rep = cur.get("replaced_by")
            if rep:
                label = re.sub(r"^obsolete ", "", cur.get("name") or "")
                for s in [label] + cur.get("syn", []):
                    if s:
                        donations.append((rep, s))
            return
        cid = cur["id"]
        iri = "http://purl.obolibrary.org/obo/" + cid.replace(":", "_")
        label = cur.get("name") or ""
        concepts.append(("mondo", cid, label, iri))
        if label:
            names.append(("mondo", cid, label, "label"))
        for s in cur.get("syn", []):
            names.append(("mondo", cid, s, "synonym"))
        names.append(("mondo", cid, cid, "id"))
        names.append(("mondo", cid, cid.replace(":", "_"), "id"))
        if cur.get("def"):
            names.append(("mondo", cid, cur["def"], "def"))
        for p in cur.get("isa", []):
            parents.append(("mondo", cid, p))

    syn_re = re.compile(r'^synonym: "(.*)" [A-Z]+')
    for line in text.splitlines():
        if line.startswith("[Term]"):
            flush()
            cur = {"syn": [], "isa": []}
        elif line.startswith("[") and cur is not None:
            flush()
            cur = None
        elif cur is None:
            continue
        elif line.startswith("id: "):
            cur["id"] = line[4:].strip()
        elif line.startswith("name: "):
            cur["name"] = line[6:].strip()
        elif line.startswith("def: "):
            m = re.match(r'^def: "(.*)" \[', line)
            if m:
                cur["def"] = m.group(1).replace('\\"', '"')
        elif line.startswith("is_obsolete: true"):
            cur["obsolete"] = True
        elif line.startswith("replaced_by: "):
            cur["replaced_by"] = line[len("replaced_by: ") :].strip()
        elif line.startswith("synonym: "):
            m = syn_re.match(line)
            if m:
                cur["syn"].append(m.group(1).replace('\\"', '"'))
        elif line.startswith("is_a: "):
            cur["isa"].append(line[6:].split("!")[0].split("{")[0].strip())
        elif line.startswith("relationship: part_of "):
            cur["isa"].append(
                line[len("relationship: part_of ") :].split("!")[0].split("{")[0].strip()
            )
    flush()
    live = {c[1] for c in concepts}
    names += [("mondo", rep, s, "synonym") for rep, s in donations if rep in live]
    return concepts, names, parents


def _ordo_cid(iri: str) -> str:
    """http://www.orpha.net/ORDO/Orphanet_797 -> ORDO:797 (OLS's obo_id)."""
    return "ORDO:" + iri.rsplit("_", 1)[-1]


def parse_ordo_owl(path: Path):
    """owl:Class elements -> rows, via iterparse (the file is 52MB)."""
    concepts, names, parents = [], [], []
    ctx = ET.iterparse(str(path), events=("start", "end"))
    _, root = next(ctx)
    for event, el in ctx:
        if event != "end" or el.tag != OWL + "Class":
            continue
        about = el.get(RDF + "about") or ""
        if not about.startswith(ORDO_NS):
            el.clear()
            continue
        cid = _ordo_cid(about)
        label = None
        defn = None
        syns: list[str] = []
        ids: list[str] = [cid, about.rsplit("/", 1)[-1]]
        ps: list[str] = []
        for child in el:
            if child.tag == RDFS + "label":
                label = (child.text or "").strip()
            elif child.tag == EFO + "alternative_term":
                s = (child.text or "").strip()
                if s:
                    syns.append(s)
            elif child.tag == EFO + "definition":
                defn = (child.text or "").strip()
            elif child.tag == SKOS + "notation":
                s = (child.text or "").strip()
                if s:
                    ids.append(s)
            elif child.tag == RDFS + "subClassOf":
                res = child.get(RDF + "resource")
                if res and res.startswith(ORDO_NS):
                    ps.append(_ordo_cid(res))
                else:
                    for r in child.iter(OWL + "Restriction"):
                        on = r.find(OWL + "onProperty")
                        sv = r.find(OWL + "someValuesFrom")
                        if (
                            on is not None
                            and sv is not None
                            and on.get(RDF + "resource") == PART_OF
                        ):
                            tgt = sv.get(RDF + "resource") or ""
                            if tgt.startswith(ORDO_NS):
                                ps.append(_ordo_cid(tgt))
        if label:
            concepts.append(("ordo", cid, label, about))
            names.append(("ordo", cid, label, "label"))
            for s in syns:
                names.append(("ordo", cid, s, "synonym"))
            for s in ids:
                names.append(("ordo", cid, s, "id"))
            if defn:
                names.append(("ordo", cid, defn, "def"))
            for p in ps:
                parents.append(("ordo", cid, p))
        el.clear()
        root.clear()
    return concepts, names, parents


def parse_medgen(names_gz: Path, mgconso_gz: Path, mgdef_gz: Path):
    """NAMES.RRF (preferred titles == the [XTIT] index) + MGCONSO.RRF (every
    non-suppressed atom == the [TITL] title/synonym index) + MGDEF.RRF
    (definitions — eutils' plain search reaches the DEF field too).
    """
    concepts, names = [], []
    with gzip.open(names_gz, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("|")
            if len(parts) < 4 or parts[3] == "Y":
                continue
            cui, name = parts[0], parts[1]
            if cui and name:
                concepts.append(("medgen", cui, name, ""))
                names.append(("medgen", cui, name, "pref"))
                names.append(("medgen", cui, cui, "id"))
    with gzip.open(mgconso_gz, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("|")
            if len(parts) < 13 or parts[12] == "Y":
                continue
            cui, name = parts[0], parts[11]
            if cui and name:
                names.append(("medgen", cui, name, "atom"))
    with gzip.open(mgdef_gz, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("|")
            if len(parts) < 4 or parts[3] == "Y":
                continue
            cui, defn = parts[0], parts[1]
            if cui and defn:
                names.append(("medgen", cui, defn, "def"))
    return concepts, names, []


def build(db_path: Path, src_dir: Path) -> dict:
    t0 = time.monotonic()
    tmp = db_path.with_name(db_path.name + ".tmp")
    tmp.unlink(missing_ok=True)
    conn = sqlite3.connect(tmp)
    conn.executescript("""
        PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;
        CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE concepts(src TEXT, cid TEXT, label TEXT, iri TEXT,
                              label_lower TEXT, label_norm TEXT,
                              PRIMARY KEY(src, cid));
        CREATE TABLE names(src TEXT, cid TEXT, name TEXT, name_lower TEXT,
                           name_norm TEXT, name_tok TEXT, kind TEXT);
        CREATE TABLE parents(src TEXT, child TEXT, parent TEXT);
        CREATE TABLE tokens(src TEXT, tok TEXT, cid TEXT);
    """)

    counts: dict[str, int] = {}

    def load(tag, concepts, names, parents):
        have = {(c[0], c[1]) for c in concepts}
        extra = {}
        for src, cid, name, kind in names:
            if kind in ("def", "id"):
                continue
            if (src, cid) not in have and (src, cid) not in extra:
                extra[(src, cid)] = (src, cid, name, "")
        concepts = concepts + list(extra.values())

        conn.executemany(
            "INSERT OR IGNORE INTO concepts VALUES (?,?,?,?,?,?)",
            [
                (src, cid, label, iri, label.lower().strip(), _norm(label))
                for src, cid, label, iri in concepts
            ],
        )
        seen: set[tuple] = set()
        name_rows, tok_rows = [], []
        for src, cid, name, kind in names:
            low = name.lower().strip()
            if not low:
                continue
            key = (src, cid, low, kind)
            if key in seen:
                continue
            seen.add(key)
            toks = _TOKEN.findall(low)
            name_rows.append((src, cid, name, low, _norm(name), " " + " ".join(toks) + " ", kind))
            if kind == "def":
                continue
            for tok in set(toks):
                tok_rows.append((src, tok, cid))
                folded = fold_token(tok)
                if folded != tok:
                    tok_rows.append((src, folded, cid))
        conn.executemany("INSERT INTO names VALUES (?,?,?,?,?,?,?)", name_rows)
        conn.executemany("INSERT OR IGNORE INTO tokens VALUES (?,?,?)", sorted(set(tok_rows)))
        conn.executemany("INSERT INTO parents VALUES (?,?,?)", sorted(set(parents)))
        counts[tag] = {
            "concepts": len(concepts),
            "names": len(name_rows),
            "parents": len(set(parents)),
        }
        print(f"  {tag}: {counts[tag]}", flush=True)

    print("parsing mondo.obo ...", flush=True)
    load(
        "mondo",
        *parse_mondo_obo((src_dir / "mondo.obo").read_text(encoding="utf-8", errors="replace")),
    )
    print("parsing ordo_orphanet.owl ...", flush=True)
    load("ordo", *parse_ordo_owl(src_dir / "ordo_orphanet.owl"))
    print("parsing MedGen RRFs ...", flush=True)
    load(
        "medgen",
        *parse_medgen(
            src_dir / "NAMES.RRF.gz", src_dir / "MGCONSO.RRF.gz", src_dir / "MGDEF.RRF.gz"
        ),
    )

    print("indexing ...", flush=True)
    conn.executescript("""
        CREATE INDEX idx_names_lower ON names(src, name_lower);
        CREATE INDEX idx_names_norm  ON names(src, name_norm);
        CREATE INDEX idx_names_cid   ON names(src, cid);
        CREATE INDEX idx_conc_lower  ON concepts(src, label_lower);
        CREATE INDEX idx_conc_norm   ON concepts(src, label_norm);
        CREATE INDEX idx_parents     ON parents(src, child);
        CREATE UNIQUE INDEX idx_tokens ON tokens(src, tok, cid);
        -- the def-phrase rung: FTS5 phrase queries instead of a LIKE scan
        -- over ~10^5 definition rows (that scan put ~25ms on every term the
        -- name rungs could not resolve)
        CREATE VIRTUAL TABLE defs USING fts5(name_tok, src UNINDEXED,
                                             cid UNINDEXED);
        INSERT INTO defs(name_tok, src, cid)
            SELECT name_tok, src, cid FROM names WHERE kind='def';
        ANALYZE;
    """)

    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": {},
        "counts": counts,
    }
    for fname, url in SOURCES.items():
        p = src_dir / fname
        h = hashlib.sha256()
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        meta["sources"][fname] = {
            "url": url,
            "bytes": p.stat().st_size,
            "sha256": h.hexdigest(),
            "mtime": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(
                timespec="seconds"
            ),
        }
    conn.executemany("INSERT INTO meta VALUES (?,?)", [(k, json.dumps(v)) for k, v in meta.items()])
    conn.commit()
    conn.close()
    os.replace(tmp, db_path)
    meta["elapsed_s"] = round(time.monotonic() - t0, 1)
    meta["db"] = str(db_path)
    meta["db_bytes"] = db_path.stat().st_size
    return meta


def download(src_dir: Path) -> None:
    src_dir.mkdir(parents=True, exist_ok=True)
    for fname, url in SOURCES.items():
        dest = src_dir / fname
        print(f"downloading {url} ...", flush=True)
        req = urllib.request.Request(url, headers={"User-Agent": "trustmed-ontology-build"})
        with urllib.request.urlopen(req, timeout=600) as r, dest.open("wb") as out:
            while chunk := r.read(1 << 20):
                out.write(chunk)
        print(f"  {dest.name}: {dest.stat().st_size} bytes", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--src-dir", type=Path, default=DEFAULT_SRC)
    ap.add_argument(
        "--download", action="store_true", help="fetch/refresh the four source files first"
    )
    a = ap.parse_args()
    if a.download:
        download(a.src_dir)
    missing = [f for f in SOURCES if not (a.src_dir / f).exists()]
    if missing:
        raise SystemExit(f"missing source file(s) {missing} in {a.src_dir}; run with --download")
    meta = build(a.db, a.src_dir)
    print(
        json.dumps(
            {k: meta[k] for k in ("built_at", "elapsed_s", "db", "db_bytes", "counts")}, indent=1
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
