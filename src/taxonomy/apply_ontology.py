"""Resolve every taxonomy label to a published terminology code and rewrite
the taxonomy outputs with normalised category and concept names.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from taxonomy.ontology import Resolver, SYSTEM_TITLE, SCT_OBSERVABLE
import taxonomy.ontology_map as M

DEFAULT_DIR = "taxonomy"

PIN = {
    "Physical_Examination": {
        k: (v if isinstance(v, tuple) else ("sct", v)) for k, v in M.PE_SITE_TO_EXAM.items()
    },
    "Laboratory": {k: (v if isinstance(v, tuple) else ("loinc", v)) for k, v in M.LAB_PIN.items()},
    "Imaging": M.IMAGING_PIN,
    "Pathology": M.PATHOLOGY_PIN,
    "Physiologic": M.PHYSIO_PIN,
}
QUERY = {
    "Physical_Examination": {},
    "Laboratory": M.LAB_QUERY,
    "Imaging": M.IMAGING_QUERY,
    "Pathology": M.PATHOLOGY_QUERY,
    "Physiologic": M.PHYSIO_QUERY,
}

LOCAL_ONLY = {"Unmapped", "Unmapped site", "Unspecified site", "Unspecified (patient-level label)"}


def read(p):
    with open(p, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write(p, header, rows):
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DEFAULT_DIR)
    a = ap.parse_args()
    D = a.dir
    R = Resolver()

    cons = read(os.path.join(D, "concepts_L3.csv"))
    concepts = sorted({(r["L1_class"], r["L3_concept"]) for r in cons})

    resolved = {}
    for cls, local in concepts:
        rec = {
            "system": None,
            "code": None,
            "display": None,
            "method": "unmapped",
            "loinc_class": None,
            "loinc_class_title": None,
            "note": None,
        }
        if local in LOCAL_ONLY:
            rec["note"] = "no terminology concept (residual bucket)"
            resolved[(cls, local)] = rec
            continue
        pin = PIN.get(cls, {}).get(local)
        if pin:
            system, code = pin
            look = R.lookup(system, code)
            if look["ok"]:
                rec.update(system=system, code=code, display=look["display"], method="pinned")
            else:
                rec["note"] = f"pinned code failed: {look['error']}"
        else:
            q = QUERY.get(cls, {}).get(local)
            if q and q.startswith("sct:"):
                term = q[4:]
                hits = R.sct_search(term, n=1)
                if not hits:
                    hits = R.sct_search(term, parent=SCT_OBSERVABLE, n=1)
                if hits:
                    rec.update(system="sct", code=hits[0][0], display=hits[0][1], method="search")
                    rec["note"] = f'SNOMED procedure search="{term}"'
                else:
                    rec["note"] = f'no SNOMED hit for "{term}"'
            elif q:
                hits = R.loinc_search(q, 1)
                if hits:
                    code, name = hits[0][0], hits[0][1]
                    rec.update(system="loinc", code=code, display=name, method="search")
                    rec["note"] = f'LOINC search="{q}"'
                else:
                    rec["note"] = f'no LOINC hit for query="{q}"'
            else:
                rec["note"] = "no assertion"
        if not rec["code"] and cls == "Physical_Examination":
            v = M.PE_SITE_TO_CATEGORY.get(local)
            if v:
                system, code = v if isinstance(v, tuple) else ("sct", v)
                look = R.lookup(system, code)
                if look["ok"]:
                    rec.update(
                        system=system,
                        code=code,
                        display=look["display"],
                        method="category_fallback",
                    )
                    rec["note"] = (
                        "no site-specific examination procedure; "
                        "rolled up to the category examination"
                    )
        if rec["system"] == "loinc" and rec["code"]:
            part, mnem = R.loinc_class(rec["code"])
            rec["loinc_class"], rec["loinc_class_title"] = part, mnem
        resolved[(cls, local)] = rec
    R.save()

    pe_cat_display = {}
    for v in set(M.PE_SITE_TO_CATEGORY.values()):
        system, code = v if isinstance(v, tuple) else ("sct", v)
        pe_cat_display[v] = (system, code, R.display(system, code) or code)
    pe_panel_display = {}
    for panel in set(M.PE_SITE_TO_PANEL.values()):
        pe_panel_display[panel] = R.display("loinc", panel) or panel
    R.save()

    def pe_category(site):
        v = M.PE_SITE_TO_CATEGORY.get(site)
        if not v:
            return ("", "", "Unclassified physical examination")
        return pe_cat_display[v]

    def pe_loinc_panel(site):
        panel = M.PE_SITE_TO_PANEL.get(site)
        return (panel or "", pe_panel_display.get(panel, "") if panel else "")

    rows = []
    for (cls, local), rec in sorted(resolved.items()):
        rows.append(
            [
                cls,
                local,
                SYSTEM_TITLE.get(rec["system"], ""),
                rec["system"] or "",
                rec["code"] or "",
                rec["display"] or "",
                rec["loinc_class_title"] or "",
                rec["method"],
                rec["note"] or "",
            ]
        )
    write(
        os.path.join(D, "concept_ontology.csv"),
        [
            "L1_class",
            "L3_concept_local",
            "code_system_title",
            "code_system",
            "code",
            "normalized_concept_name",
            "loinc_class",
            "match_method",
            "note",
        ],
        rows,
    )

    dcm_l2 = {}
    for local, mod in M.IMAGING_L2_MODALITY.items():
        look = R.lookup("dcm", mod)
        dcm_l2[local] = ("dcm", mod, look["display"]) if look["ok"] else ("", "", local)
    R.save()

    l2_pin = {}
    for k, (system, code) in M.L2_PIN.items():
        look = R.lookup(system, code)
        if look["ok"]:
            l2_pin[k] = (system, code, look["display"])
    R.save()

    out = []
    for r in cons:
        cls, l2, l3 = r["L1_class"], r["L2_category"], r["L3_concept"]
        rec = resolved[(cls, l3)]
        if cls == "Physical_Examination":
            csys, ccode, cname = pe_category(l3)
        elif cls == "Imaging" and l2 in dcm_l2:
            csys, ccode, cname = dcm_l2[l2]
        elif (cls, l2) in l2_pin:
            csys, ccode, cname = l2_pin[(cls, l2)]
        else:
            csys, ccode, cname = "", "", l2
        out.append(
            [
                cls,
                l2,
                cname,
                csys,
                ccode,
                l3,
                rec["display"] or l3,
                rec["system"] or "",
                rec["code"] or "",
                rec["loinc_class_title"] or "",
                rec["method"],
                r["n_L4_surface_forms"],
                r["n_mentions"],
                r["n_cases"],
                r["pct_cases"],
            ]
        )
    write(
        os.path.join(D, "concepts_L3_normalized.csv"),
        [
            "L1_class",
            "L2_category_local",
            "L2_category_normalized",
            "L2_code_system",
            "L2_code",
            "L3_concept_local",
            "L3_concept_normalized",
            "L3_code_system",
            "L3_code",
            "loinc_class",
            "match_method",
            "n_L4_surface_forms",
            "n_mentions",
            "n_cases",
            "pct_cases",
        ],
        out,
    )

    agg = {}
    for r in out:
        cls, cname, csys, ccode = r[0], r[2], r[3], r[4]
        k = (cls, cname, csys, ccode)
        d = agg.setdefault(k, {"concepts": set(), "surf": 0, "ment": 0})
        d["concepts"].add(r[6])
        d["surf"] += int(r[11])
        d["ment"] += int(r[12])
    write(
        os.path.join(D, "categories_L2_normalized.csv"),
        [
            "L1_class",
            "L2_category_normalized",
            "code_system",
            "code",
            "n_L3_concepts",
            "n_L4_surface_forms",
            "n_mentions",
        ],
        [
            [k[0], k[1], k[2], k[3], len(v["concepts"]), v["surf"], v["ment"]]
            for k, v in sorted(agg.items(), key=lambda kv: (kv[0][0], -kv[1]["ment"]))
        ],
    )

    ti = read(os.path.join(D, "test_items.csv"))
    write(
        os.path.join(D, "test_items_normalized.csv"),
        [
            "L1_class",
            "L2_category",
            "L3_concept_local",
            "L3_concept_normalized",
            "L3_code_system",
            "L3_code",
            "anatomy_or_specimen",
            "L4_surface_form",
            "n_mentions",
            "n_cases",
        ],
        [
            [
                r["L1_class"],
                r["L2_category"],
                r["L3_concept"],
                resolved[(r["L1_class"], r["L3_concept"])]["display"] or r["L3_concept"],
                resolved[(r["L1_class"], r["L3_concept"])]["system"] or "",
                resolved[(r["L1_class"], r["L3_concept"])]["code"] or "",
                r["anatomy_or_specimen"],
                r["L4_surface_form"],
                r["n_mentions"],
                r["n_cases"],
            ]
            for r in ti
        ],
    )

    pi = read(os.path.join(D, "pe_items.csv"))
    pe_rows = []
    for r in pi:
        site = r["L3_site_or_measure"]
        rec = resolved[("Physical_Examination", site)]
        csys, ccode, cname = pe_category(site)
        panel_code, panel_name = pe_loinc_panel(site)
        pe_rows.append(
            [
                cname,
                csys,
                ccode,
                site,
                rec["display"] or site,
                rec["system"] or "",
                rec["code"] or "",
                panel_code,
                panel_name,
                r["L2_system_fine"],
                r["L4_surface_form"],
                r["n_mentions"],
                r["n_cases"],
            ]
        )
    write(
        os.path.join(D, "pe_items_normalized.csv"),
        [
            "L2_category_normalized",
            "L2_code_system",
            "L2_code",
            "L3_site_local",
            "L3_site_normalized",
            "L3_code_system",
            "L3_code",
            "loinc_panel_code",
            "loinc_panel_name",
            "generator_section",
            "L4_surface_form",
            "n_mentions",
            "n_cases",
        ],
        pe_rows,
    )

    tot_ment = {}
    for r in cons:
        tot_ment[r["L1_class"]] = tot_ment.get(r["L1_class"], 0) + int(r["n_mentions"])
    mapped_ment = {}
    for r in out:
        if r[8]:
            mapped_ment[r[0]] = mapped_ment.get(r[0], 0) + int(r[12])
    by_method = {}
    by_system = {}
    for rec in resolved.values():
        by_method[rec["method"]] = by_method.get(rec["method"], 0) + 1
        if rec["system"]:
            by_system[rec["system"]] = by_system.get(rec["system"], 0) + 1
    stats = {
        "n_concepts": len(resolved),
        "by_match_method": by_method,
        "by_code_system": by_system,
        "coded_pct_of_mentions": {
            k: round(100 * mapped_ment.get(k, 0) / v, 1) for k, v in sorted(tot_ment.items())
        },
        "unresolved": sorted(
            f"{c} | {l}" for (c, l), r in resolved.items() if not r["code"] and l not in LOCAL_ONLY
        ),
    }
    with open(os.path.join(D, "ontology_stats.json"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
