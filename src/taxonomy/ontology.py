"""Terminology resolver."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

TX = "https://tx.fhir.org/r4/CodeSystem/$lookup"
TX_EXPAND = "https://tx.fhir.org/r4/ValueSet/$expand"
CTSS = "https://clinicaltables.nlm.nih.gov/api/loinc_items/v3/search"

SCT_PROCEDURE = "71388002"
SCT_OBSERVABLE = "363787002"

SYSTEM_URI = {
    "loinc": "http://loinc.org",
    "sct": "http://snomed.info/sct",
    "dcm": "http://dicom.nema.org/resources/ontology/DCM",
}
SYSTEM_TITLE = {
    "loinc": "LOINC",
    "sct": "SNOMED CT",
    "dcm": "DICOM Controlled Terminology",
}

CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ontology_cache.json")


class Resolver:
    def __init__(self, cache_path: str = CACHE_PATH, sleep: float = 0.05):
        self.cache_path = cache_path
        self.sleep = sleep
        self.cache = {}
        if os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as fh:
                self.cache = json.load(fh)
        self._dirty = False

    def save(self):
        if self._dirty:
            with open(self.cache_path, "w", encoding="utf-8") as fh:
                json.dump(self.cache, fh, indent=1, ensure_ascii=False, sort_keys=True)
            self._dirty = False

    def lookup(self, system: str, code: str) -> dict:
        """Return {'display':…, 'properties':{…}, 'ok':bool, 'error':str|None}."""
        key = f"{system}|{code}"
        if key in self.cache:
            return self.cache[key]
        q = urllib.parse.urlencode({"system": SYSTEM_URI[system], "code": code})
        req = urllib.request.Request(TX + "?" + q, headers={"Accept": "application/fhir+json"})
        out = {"display": None, "properties": {}, "ok": False, "error": None}
        try:
            with urllib.request.urlopen(req, timeout=45) as r:
                d = json.load(r)
            out["display"] = next(
                (p["valueString"] for p in d["parameter"] if p["name"] == "display"), None
            )
            for p in d["parameter"]:
                if p["name"] == "property":
                    kv = {
                        x["name"]: (
                            x.get("valueString") or x.get("valueCode") or x.get("valueBoolean")
                        )
                        for x in p["part"]
                    }
                    c = kv.get("code")
                    if c and c != "RELATEDNAMES2":
                        out["properties"][c] = kv.get("value")
            out["ok"] = out["display"] is not None
        except urllib.error.HTTPError as e:
            out["error"] = f"HTTP {e.code}"
        except Exception as e:
            out["error"] = str(e)[:120]
        self.cache[key] = out
        self._dirty = True
        time.sleep(self.sleep)
        return out

    def display(self, system: str, code: str) -> str | None:
        return self.lookup(system, code)["display"]

    def loinc_class(self, code: str) -> tuple[str | None, str | None]:
        """(class part code, class mnemonic) for a LOINC term."""
        part = self.lookup("loinc", code)["properties"].get("CLASS")
        if not part:
            return None, None
        return part, self.display("loinc", part)

    def sct_search(self, term: str, parent: str = SCT_PROCEDURE, n: int = 5):
        """Free-text search inside a SNOMED CT sub-hierarchy."""
        key = f"sctsearch|{parent}|{n}|{term}"
        if key in self.cache:
            return self.cache[key]
        q = urllib.parse.urlencode(
            {
                "url": f"http://snomed.info/sct?fhir_vs=isa/{parent}",
                "filter": term,
                "count": max(n, 20),
            }
        )
        req = urllib.request.Request(
            TX_EXPAND + "?" + q, headers={"Accept": "application/fhir+json"}
        )
        res = []
        try:
            with urllib.request.urlopen(req, timeout=45) as r:
                d = json.load(r)
            for c in d.get("expansion", {}).get("contains", []):
                res.append([c["code"], c["display"]])
            res.sort(key=lambda cd: (len(cd[1]), cd[1]))
            res = res[:n]
        except Exception:
            res = []
        self.cache[key] = res
        self._dirty = True
        time.sleep(self.sleep)
        return res

    def loinc_search(self, term: str, n: int = 8):
        key = f"search|{n}|{term}"
        if key in self.cache:
            return self.cache[key]
        q = urllib.parse.urlencode(
            {"terms": term, "maxList": n, "df": "LOINC_NUM,LONG_COMMON_NAME"}
        )
        try:
            with urllib.request.urlopen(CTSS + "?" + q, timeout=45) as r:
                res = json.load(r)[3] or []
        except Exception:
            res = []
        self.cache[key] = res
        self._dirty = True
        time.sleep(self.sleep)
        return res
