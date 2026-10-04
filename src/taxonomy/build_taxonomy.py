"""Build the OSCE action-space taxonomy from a profiles.parquet run."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict

import pyarrow.parquet as pq

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from taxonomy.lexicons import (
    ANATOMY,
    IMAGING_CATEGORY,
    IMAGING_CONCEPT,
    LAB_CATEGORY,
    LAB_CONCEPT,
    PATHOLOGY_CATEGORY,
    PATHOLOGY_CONCEPT,
    PE_FINE_TO_COARSE,
    PE_REGION,
    PE_SECTION_FINE,
    PHYSIO_CATEGORY,
    PHYSIO_CONCEPT,
    SPECIALTY_GROUP,
    VITAL_CONCEPT,
    apply_rules,
    icd10_chapter,
    normalise,
)

DEFAULT_PARQUET = "profiles.parquet"
DEFAULT_ORPHA = "orpha_meta_info.json"

TEST_SECTIONS = {
    "Laboratory_Test_Results": ("Laboratory", LAB_CATEGORY, LAB_CONCEPT),
    "Physiologic_Test_Results": ("Physiologic", PHYSIO_CATEGORY, PHYSIO_CONCEPT),
    "Imaging_Results": ("Imaging", IMAGING_CATEGORY, IMAGING_CONCEPT),
    "Pathology_Results": ("Pathology", PATHOLOGY_CATEGORY, PATHOLOGY_CONCEPT),
}
COLS = [
    "pmcid",
    "diagnosis",
    "orpha_name",
    "specialty",
    "Physical_Examination_Findings",
    "Post_Baseline_Findings",
    *TEST_SECTIONS,
]


def jload(v):
    if not isinstance(v, str) or not v.strip():
        return None
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return None


def okey(s: str) -> str:
    s = unicodedata.normalize("NFKD", (s or "")).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", s)).strip()


def sect_key(s: str) -> str:
    return re.sub(r"[^a-z_]", "", (s or "").strip().lower().replace(" ", "_"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=DEFAULT_PARQUET)
    ap.add_argument("--orpha", default=DEFAULT_ORPHA)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(os.path.dirname(args.parquet), "taxonomy")
    os.makedirs(out, exist_ok=True)

    orpha_by_name, orpha_by_syn = {}, {}
    with open(args.orpha, encoding="utf-8") as fh:
        for m in json.load(fh):
            orpha_by_name.setdefault(okey(m["Name"]), m)
            for s in m.get("Synonyms") or []:
                orpha_by_syn.setdefault(okey(s), m)

    def lookup_orpha(name):
        k = okey(name)
        return orpha_by_name.get(k) or orpha_by_syn.get(k)

    pe_rows = defaultdict(lambda: {"n": 0, "cases": set()})
    test_rows = defaultdict(lambda: {"n": 0, "cases": set()})
    dis_rows = defaultdict(lambda: {"n": 0})
    spec_pairs = Counter()

    per_case = defaultdict(Counter)
    pe_sections_per_case = []
    section_present = Counter()
    pb_phase = Counter()
    pb_section = Counter()

    n_rows = 0
    pf = pq.ParquetFile(args.parquet)
    for batch in pf.iter_batches(batch_size=2000, columns=COLS):
        for row in batch.to_pylist():
            n_rows += 1
            pmcid = row["pmcid"]

            diag = (row["diagnosis"] or "").strip()
            orn = (row["orpha_name"] or "").strip()
            specs = [s.strip().lower() for s in (row["specialty"] or []) if s]
            groups = sorted({apply_rules(SPECIALTY_GROUP, s, "Other / unspecified") for s in specs})
            for g in groups:
                spec_pairs[g] += 1
            d = dis_rows[(diag, orn)]
            d["n"] += 1
            d.setdefault("specialties", set()).update(specs)
            d.setdefault("groups", set()).update(groups)

            pe = jload(row["Physical_Examination_Findings"]) or {}
            if pe:
                section_present["Physical_Examination_Findings"] += 1
            n_sec = 0
            for raw_sec, items in pe.items():
                if not isinstance(items, list):
                    items = [items]
                fine = PE_SECTION_FINE.get(sect_key(raw_sec))
                if fine is None:
                    fine = "Unmapped"
                coarse = PE_FINE_TO_COARSE.get(fine, "Unmapped")
                n_sec += 1
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    loc = (it.get("location") or "").strip()
                    nloc = normalise(loc)
                    if not nloc:
                        concept = "Unspecified site"
                    elif fine == "Vital_Signs":
                        concept = apply_rules(VITAL_CONCEPT, nloc, "Vital signs (composite)")
                    else:
                        concept = apply_rules(PE_REGION, nloc, "Unmapped site")
                    k = (fine, coarse, concept, loc)
                    pe_rows[k]["n"] += 1
                    pe_rows[k]["cases"].add(pmcid)
                    per_case[pmcid]["PE"] += 1
            pe_sections_per_case.append(n_sec)

            for sec, (l1, cat_rules, con_rules) in TEST_SECTIONS.items():
                items = jload(row[sec]) or []
                if not isinstance(items, list):
                    continue
                if items:
                    section_present[sec] += 1
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    raw = (it.get("test_name") or "").strip()
                    nm = normalise(raw)
                    modality = (it.get("modality") or "").strip().lower()
                    probe = f"{nm} {modality.replace('_', ' ')}".strip()
                    cat = apply_rules(cat_rules, probe, "Unmapped")
                    con = apply_rules(con_rules, probe, "Unmapped")
                    anat = apply_rules(ANATOMY, nm, "") or ""
                    k = (l1, cat, con, anat, raw)
                    test_rows[k]["n"] += 1
                    test_rows[k]["cases"].add(pmcid)
                    per_case[pmcid][l1] += 1

            for it in jload(row["Post_Baseline_Findings"]) or []:
                if isinstance(it, dict):
                    pb_phase[(it.get("phase") or "").strip()] += 1
                    pb_section[(it.get("section") or "").strip()] += 1

    with open(os.path.join(out, "pe_items.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "L1_class",
                "L2_system_fine",
                "L2_system_prompt13",
                "L3_site_or_measure",
                "L4_surface_form",
                "n_mentions",
                "n_cases",
            ]
        )
        for (fine, coarse, con, raw), v in sorted(pe_rows.items(), key=lambda kv: -kv[1]["n"]):
            w.writerow(["Physical_Examination", fine, coarse, con, raw, v["n"], len(v["cases"])])

    with open(os.path.join(out, "test_items.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "L1_class",
                "L2_category",
                "L3_concept",
                "anatomy_or_specimen",
                "L4_surface_form",
                "n_mentions",
                "n_cases",
            ]
        )
        for (l1, cat, con, anat, raw), v in sorted(test_rows.items(), key=lambda kv: -kv[1]["n"]):
            w.writerow([l1, cat, con, anat, raw, v["n"], len(v["cases"])])

    icd_chap = Counter()
    icd_code = Counter()
    orpha_hit = Counter()
    with open(os.path.join(out, "disease_items.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "L4_diagnosis_surface",
                "L3_orpha_name",
                "orpha_code",
                "orpha_disorder_type",
                "icd10_code",
                "L1_icd10_chapter",
                "L1_icd10_chapter_title",
                "L2_specialty_groups",
                "raw_specialties",
                "n_cases",
            ]
        )
        for (diag, orn), v in sorted(dis_rows.items(), key=lambda kv: -kv[1]["n"]):
            m = lookup_orpha(orn)
            code = m["OrphaCode"] if m else ""
            dtype = m.get("DisorderType", "") if m else ""
            icd = ""
            if m:
                refs = [
                    r for r in (m.get("ExternalReferences") or []) if r.get("source") == "ICD-10"
                ]
                if refs:
                    icd = refs[0]["reference"]
            ch = icd10_chapter(icd)
            orpha_hit["matched" if m else "unmatched"] += v["n"]
            if ch:
                icd_chap[(ch[0], ch[1], ch[2])] += v["n"]
                icd_code[icd] += v["n"]
            else:
                icd_chap[("--", "--", "Unmapped (no ICD-10 crosswalk)")] += v["n"]
            w.writerow(
                [
                    diag,
                    orn,
                    code,
                    dtype,
                    icd,
                    ch[0] if ch else "",
                    ch[2] if ch else "",
                    "; ".join(sorted(v["groups"])),
                    "; ".join(sorted(v["specialties"])),
                    v["n"],
                ]
            )

    def rollup(rows, idx_l1, idx_l2, idx_l3, idx_raw):
        l2 = defaultdict(lambda: {"n": 0, "concepts": set(), "surfaces": set(), "cases": set()})
        l3 = defaultdict(lambda: {"n": 0, "surfaces": set(), "cases": set()})
        for k, v in rows.items():
            a = k[idx_l1] if idx_l1 is not None else "Physical_Examination"
            b, c, raw = k[idx_l2], k[idx_l3], k[idx_raw]
            l2[(a, b)]["n"] += v["n"]
            l2[(a, b)]["concepts"].add(c)
            l2[(a, b)]["surfaces"].add(raw)
            l2[(a, b)]["cases"] |= v["cases"]
            l3[(a, b, c)]["n"] += v["n"]
            l3[(a, b, c)]["surfaces"].add(raw)
            l3[(a, b, c)]["cases"] |= v["cases"]
        return l2, l3

    pe_l2, pe_l3 = rollup(pe_rows, None, 0, 2, 3)
    test_l2, test_l3 = rollup(test_rows, 0, 1, 2, 4)

    with open(os.path.join(out, "categories_L2.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "L1_class",
                "L2_category",
                "n_L3_concepts",
                "n_L4_surface_forms",
                "n_mentions",
                "n_cases",
                "pct_cases",
            ]
        )
        for (a, b), v in sorted(pe_l2.items(), key=lambda kv: -kv[1]["n"]):
            w.writerow(
                [
                    "Physical_Examination",
                    b,
                    len(v["concepts"]),
                    len(v["surfaces"]),
                    v["n"],
                    len(v["cases"]),
                    round(100 * len(v["cases"]) / n_rows, 1),
                ]
            )
        for (a, b), v in sorted(test_l2.items(), key=lambda kv: (kv[0][0], -kv[1]["n"])):
            w.writerow(
                [
                    a,
                    b,
                    len(v["concepts"]),
                    len(v["surfaces"]),
                    v["n"],
                    len(v["cases"]),
                    round(100 * len(v["cases"]) / n_rows, 1),
                ]
            )

    with open(os.path.join(out, "concepts_L3.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "L1_class",
                "L2_category",
                "L3_concept",
                "n_L4_surface_forms",
                "n_mentions",
                "n_cases",
                "pct_cases",
            ]
        )
        for (a, b, c), v in sorted(pe_l3.items(), key=lambda kv: -kv[1]["n"]):
            w.writerow(
                [
                    "Physical_Examination",
                    b,
                    c,
                    len(v["surfaces"]),
                    v["n"],
                    len(v["cases"]),
                    round(100 * len(v["cases"]) / n_rows, 1),
                ]
            )
        for (a, b, c), v in sorted(test_l3.items(), key=lambda kv: (kv[0][0], -kv[1]["n"])):
            w.writerow(
                [
                    a,
                    b,
                    c,
                    len(v["surfaces"]),
                    v["n"],
                    len(v["cases"]),
                    round(100 * len(v["cases"]) / n_rows, 1),
                ]
            )

    with open(os.path.join(out, "diseases_L1_L2.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["axis", "label", "n_cases", "pct_cases"])
        for (roman, rng, title), n in sorted(icd_chap.items(), key=lambda kv: -kv[1]):
            w.writerow([f"ICD-10 chapter {roman} ({rng})", title, n, round(100 * n / n_rows, 1)])
        for g, n in spec_pairs.most_common():
            w.writerow(["Clinical specialty", g, n, round(100 * n / n_rows, 1)])

    def pct(counter, key):
        return round(100 * counter.get(key, 0) / n_rows, 1)

    def mean(vals):
        return round(sum(vals) / len(vals), 2) if vals else 0.0

    def unmapped(rows, idx):
        tot = sum(v["n"] for v in rows.values())
        bad = sum(v["n"] for k, v in rows.items() if k[idx].startswith("Unmapped"))
        return round(100 * (tot - bad) / tot, 2) if tot else 0.0

    stats = {
        "n_cases": n_rows,
        "section_case_coverage_pct": {
            k: pct(section_present, k) for k in ["Physical_Examination_Findings", *TEST_SECTIONS]
        },
        "mentions": {
            "Physical_Examination": sum(v["n"] for v in pe_rows.values()),
            **{
                l1: sum(v["n"] for k, v in test_rows.items() if k[0] == l1)
                for l1 in ("Laboratory", "Physiologic", "Imaging", "Pathology")
            },
        },
        "unique_surface_forms": {
            "Physical_Examination_site": len({k[3] for k in pe_rows}),
            **{
                l1: len({k[4] for k in test_rows if k[0] == l1})
                for l1 in ("Laboratory", "Physiologic", "Imaging", "Pathology")
            },
        },
        "mean_per_case": {
            "Physical_Examination_findings": mean([per_case[c]["PE"] for c in per_case]),
            "Physical_Examination_systems": mean(pe_sections_per_case),
            **{
                l1: mean([per_case[c][l1] for c in per_case])
                for l1 in ("Laboratory", "Physiologic", "Imaging", "Pathology")
            },
        },
        "L3_mapping_coverage_pct_of_mentions": {
            "Physical_Examination": unmapped(pe_rows, 2),
            **{
                l1: round(
                    100
                    * (
                        1
                        - sum(
                            v["n"]
                            for k, v in test_rows.items()
                            if k[0] == l1 and k[2] == "Unmapped"
                        )
                        / max(1, sum(v["n"] for k, v in test_rows.items() if k[0] == l1))
                    ),
                    2,
                )
                for l1 in ("Laboratory", "Physiologic", "Imaging", "Pathology")
            },
        },
        "L2_mapping_coverage_pct_of_mentions": {
            "Physical_Examination": unmapped(pe_rows, 0),
            **{
                l1: round(
                    100
                    * (
                        1
                        - sum(
                            v["n"]
                            for k, v in test_rows.items()
                            if k[0] == l1 and k[1] == "Unmapped"
                        )
                        / max(1, sum(v["n"] for k, v in test_rows.items() if k[0] == l1))
                    ),
                    2,
                )
                for l1 in ("Laboratory", "Physiologic", "Imaging", "Pathology")
            },
        },
        "taxonomy_size": {
            "PE_L2_fine": len({k[0] for k in pe_rows}),
            "PE_L2_prompt13": len({k[1] for k in pe_rows}),
            "PE_L3": len({k[2] for k in pe_rows}),
            "test_L2": len({(k[0], k[1]) for k in test_rows}),
            "test_L3": len({(k[0], k[2]) for k in test_rows}),
            "anatomy_axis": len({k[3] for k in test_rows if k[3]}),
        },
        "diseases": {
            "unique_free_text_diagnosis": len({d for d, _ in dis_rows}),
            "unique_orpha_name": len({o for _, o in dis_rows}),
            "orpha_crosswalk_matched_cases": orpha_hit["matched"],
            "orpha_crosswalk_matched_pct": round(100 * orpha_hit["matched"] / n_rows, 1),
            "unique_icd10_codes": len(icd_code),
            "icd10_chapters_used": len([k for k in icd_chap if k[0] != "--"]),
            "icd10_mapped_pct_of_cases": round(
                100 * sum(n for k, n in icd_chap.items() if k[0] != "--") / n_rows, 1
            ),
            "raw_specialty_strings": len({s for v in dis_rows.values() for s in v["specialties"]}),
            "specialty_groups": len(spec_pairs),
        },
        "post_baseline": {
            "mentions": sum(pb_section.values()),
            "by_section": pb_section.most_common(),
            "by_phase": pb_phase.most_common(),
        },
    }
    with open(os.path.join(out, "stats.json"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)

    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print("\nwrote ->", out)


if __name__ == "__main__":
    main()
