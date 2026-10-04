"""Two-phase consultation environment and rollout runner."""

from __future__ import annotations

import argparse
import base64
import dataclasses
import difflib
import hashlib
import json
import logging
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from openai import OpenAI

_HERE = Path(__file__).resolve().parent
load_dotenv(_HERE.parent / ".env")

from taxonomy.lexicons import (
    IMAGING_CATEGORY,
    LAB_CATEGORY,
    PATHOLOGY_CATEGORY,
    PHYSIO_CATEGORY,
    PE_SECTION_FINE,
    PE_REGION,
    VITAL_CONCEPT,
    apply_rules,
    normalise,
)
import taxonomy.ontology_map as _OM

from prompts import (
    ABSTENTION_GUIDANCE,
    ABSTENTION_LINES,
    ABSTENTION_PHASE_ERR,
    BEGIN_WORKUP_LINE,
    BRACKET_NOTE,
    CONCLUDE_MENU,
    CONCLUDE_TEMPLATE,
    DEFER_PREFIXES,
    DEFER_SLOT_ERR,
    DEFER_SLOT_PHRASES,
    FINAL_DX_LINE,
    FINAL_DX_NOTE_HISTORY,
    GATE_TEMPLATE,
    GATED_BEGIN_WORKUP_MIN_ASKS,
    GATED_DX_HISTORY_FINAL,
    GATED_DX_MIN_ASKS,
    GATED_DX_MIN_WORKUP,
    HISTORY_ASK_LINE,
    HISTORY_FINAL_PHASE_ERR,
    HISTORY_MENU,
    HISTORY_TEMPLATE,
    INTEGRITY_LINE,
    PATIENT_PROMPT,
    SEARCH_BRIEF,
    SEARCH_LINE,
    SINGLE_DX_ERR,
    SINGLE_DX_ERR_NO_DEFER,
    TERMINAL_BRACKET_ERR,
    WORKUP_MENU,
    WORKUP_RULES_BRIEF,
    WORKUP_RULES_FULL,
    WORKUP_TEMPLATE,
)
import prompts as _prompts_module

PROMPT_SHA = hashlib.sha1(
    Path(_prompts_module.__file__).read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")
).hexdigest()[:12]

LOG = logging.getLogger("trustmed")

ANCHOR_PHASE_BATCH = "phase_batch"
ANCHOR_MODES = (ANCHOR_PHASE_BATCH,)

DELIVER_LEGACY = "legacy"
DELIVER_CHUNKED = "chunked"
DELIVER_MODES = (DELIVER_LEGACY, DELIVER_CHUNKED)
CONCLUDE_MODES = ("conclude", "conclude_retry")
RULE_VIOLATION_MODES = ("forfeit", "noop")
REFUSAL_TEXT_MODES = ("terse", "full")
PENDING_LINE = "[image pending — delivered with the next integrity check]"
NOT_DELIVERED_LINE = "[image not delivered this order — re-order to view]"
ANCHOR_MODE_DEFAULT = ANCHOR_PHASE_BATCH


PANEL_POLICIES = ("whole_figure", "crops_only", "strict")

VICTIM_SERVING_KINDS = ("crop", "single_plate")
SERVED_PANEL_STATUSES = ("crop", "single_plate", "composite_plate")

PANEL_MANIFEST_NAME = "panel_assets.parquet"


def default_panel_root() -> Path:
    """The crop tree this checkout should resolve `crop_path` against."""
    env = os.environ.get("TRUSTMED_PANEL_DIR")
    here = Path(__file__).resolve().parent
    built = here.parent / "Image"
    for cand in ((Path(env),) if env else ()) + (built,):
        if (cand / PANEL_MANIFEST_NAME).is_file():
            return cand
    return built


DEFAULT_PANEL_DIR = default_panel_root()
DEFAULT_PANEL_ASSETS = DEFAULT_PANEL_DIR / PANEL_MANIFEST_NAME


@dataclass(frozen=True, slots=True)
class Config:
    profiles: Path
    out: Path
    doctor_model: str = "Qwen/Qwen3-VL-8B-Instruct"
    patient_model: str = "gpt-4.1-mini"
    patient_temperature: float = 0.4
    doctor_base_url: str | None = None
    patient_base_url: str | None = None
    service_tier: str | None = None
    doctor_temperature: float = 0.7
    min_asks: int = 0
    max_asks: int = 10
    max_workup_turns: int = 20
    max_items_per_action: int = 4
    max_images_per_turn: int = 6
    deliver_mode: str = DELIVER_LEGACY
    deliver_chunk: int = 6
    max_batches_per_order: int = 1
    max_output_tokens: int = 4_000
    images_dir: Path = Path(os.environ.get("TRUSTMED_IMAGES_DIR", "images"))
    panel_assets: Path | None = None
    panel_assets_dir: Path = DEFAULT_PANEL_DIR
    donor_list: Path | None = None
    panel_policy: str = "whole_figure"
    identity_anchor: bool = False
    post_baseline: bool = False
    search_url: str | None = None
    search_topk: int = 3
    max_searches: int = 3
    min_workup_actions: int = 0
    allow_history_final: bool = True
    corrupt_rate: float = 0.0
    corrupt_seed: str = "0"
    gate_bank_dir: Path | None = None
    corrupt_axis: str | None = None
    allow_defer: bool = False
    integrity_hint: bool = True
    limit: int | None = 3
    offset: int = 0
    pmcids: tuple[str, ...] = ()
    workers: int = 2
    timeout: float = 900.0
    max_attempts: int = 6
    validation_retries: int = 2
    rl_mode: bool = False
    anchor_mode: str = ANCHOR_MODE_DEFAULT
    invalid_penalty: float = -0.01
    rule_violation: str = "forfeit"
    refusal_text: str = "terse"
    conclude_mode: str = "conclude_retry"
    local_penalty: bool = False
    freeze_opening: bool = False
    reward_info_per_fact: float = 0.0
    rollouts: int = 1

    @property
    def policy_retries(self) -> int:
        return 0 if self.rl_mode else self.validation_retries

    @property
    def state(self) -> Path:
        return self.out / "state"


ACTOR_SECTIONS = (
    "Demographics",
    "Chief_Complaint",
    "History_of_Present_Illness",
    "Denied_Symptoms",
    "Past_Medical_History",
    "Medication_History",
    "Surgical_or_Procedural_History",
    "Family_History",
    "Social_History",
    "Review_of_Systems",
)
FACT_ATTRS = ("onset", "duration", "frequency", "severity")

PRICE = {
    "gpt-4o": (2.50, 1.25, 10.00),
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
    "gpt-5-mini": (0.25, 0.025, 2.00),
}

CAREGIVER = re.compile(
    r"\b(newborn|neonate|infant|toddler|"
    r"\d+\s*-?\s*(day|week|month)s?[- ]old)\b",
    re.I,
)
AGE_YEARS = re.compile(r"(\d+)\s*-?\s*year", re.I)


LAB74 = {
    "Clinical chemistry": "Clinical Chemistry",
    "Hematology": "Hematology",
    "Microbiology & infectious serology": "Microbiology",
    "Urinalysis & body fluids": "Urinalysis",
    "Immunology & autoimmune serology": "Immunology",
    "Endocrinology": "Endocrinology",
    "General / unspecified laboratory": "General Laboratory Testing",
    "Unmapped": "Unmapped Laboratory",
    "Molecular genetics & cytogenetics": "Molecular Genetics",
    "Tumor markers": "Tumor Markers",
    "Metabolic & biochemical genetics": "Metabolic Screening",
    "Coagulation": "Coagulation",
    "Flow cytometry / immunophenotyping": "Flow Cytometry",
    "Toxicology & drug monitoring": "Toxicology Screening",
}
IMG74 = {
    "Computed tomography": "Computed Tomography",
    "Magnetic resonance imaging": "Magnetic Resonance Imaging",
    "Radiography & fluoroscopy": "Computed Radiography",
    "Ultrasonography": "Ultrasound",
    "Echocardiography": "Ultrasound",
    "Endoscopy": "Endoscopy",
    "Nuclear medicine": "Nuclear Medicine",
    "Catheter angiography": "X-Ray Angiography",
    "Ophthalmic imaging": "Ophthalmic Photography",
    "Other imaging": "Other Imaging Modality",
    "Unmapped": "Unmapped Imaging",
    "Mammography": "Mammography",
    "Dermatologic imaging": "External-Camera Photography",
}
PATH74 = {
    "Histopathology": "Histopathology",
    "Immunohistochemistry": "Immunohistochemistry",
    "Gross pathology": "Gross Examination",
    "Cytopathology": "Cytopathology",
    "Hematopathology (bone marrow)": "Bone Marrow Examination",
    "Electron microscopy": "Electron Microscopy",
    "Immunofluorescence": "Direct Immunofluorescence",
    "Special stains & histochemistry": "Special Stains and Histochemistry",
    "Hematopathology (blood film)": "Peripheral Blood Smear",
    "Other pathology": "Surgical Pathology",
    "Molecular / in-situ pathology": "Molecular Genetics",
    "Other microscopy": "Microscopy",
    "Unmapped": "Surgical Pathology",
}
PHY74 = {
    "Cardiac electrophysiology & monitoring": "Cardiovascular Physiologic Testing",
    "Neurophysiology": "Neurophysiologic Testing",
    "Echocardiography & cardiac imaging": "Echocardiography",
    "Other physiologic testing": "Other Physiologic Testing",
    "Ophthalmic function testing": "Ophthalmic Examination and Testing",
    "Pulmonary function & exercise testing": "Pulmonary Function Testing",
    "Neurocognitive & psychometric assessment": "Psychological Testing",
    "Audiovestibular testing": "Audiometry",
    "Invasive hemodynamics & pressure monitoring": "Hemodynamic Monitoring",
    "Provocative / bedside diagnostic test": "Provocative and Bedside Diagnostic Testing",
    "Gastrointestinal function testing": "Gastrointestinal Function Testing",
    "Urodynamics": "Urodynamic Testing",
    "Bedside physiologic monitoring": "Bedside Physiologic Monitoring",
    "Functional & performance assessment": "Functional and Performance Assessment",
    "Unmapped": "Other Physiologic Testing",
}
PANEL74 = {
    "8716-3": "Vital Signs",
    "8699-1": "Eye Examination",
    "8713-0": "General Examination",
    "8703-1": "Extremity Examination",
    "8694-2": "Abdominal Examination",
    "8701-5": "Head Examination",
    "8705-6": "Neurologic Examination",
    "8704-9": "Oral, Oropharyngeal, and Dental Examination",
    "11441-3": "Musculoskeletal Examination",
    "11443-9": "Respiratory Examination",
    "8709-8": "Skin Examination",
    "11421-5": "Cardiovascular Examination",
    "8693-4": "Mental Status Examination",
    "11448-8": "Hematologic, Lymphatic, and Immunologic Examination",
    "11442-1": "Neck Examination",
    "8700-7": "Genitourinary Examination",
    "34565-2": "Vital Signs and Anthropometric Measurements",
    "32422-8": "Breast Examination",
    "8698-3": "Ear Examination",
    "8706-4": "Nasal Examination",
    "8708-0": "Rectal Examination",
}
PE_OTHER = "Other Physical Examination"

PE_CATEGORIES = tuple(dict.fromkeys(list(PANEL74.values()) + [PE_OTHER]))

PE_MENU_MAP: dict[str, tuple[str, ...]] = {
    "vital signs": ("Vital Signs", "Vital Signs and Anthropometric Measurements"),
    "general": ("General Examination",),
    "skin": ("Skin Examination",),
    "lymph nodes": ("Hematologic, Lymphatic, and Immunologic Examination",),
    "head": ("Head Examination",),
    "eyes": ("Eye Examination",),
    "ears": ("Ear Examination",),
    "nose": ("Nasal Examination",),
    "mouth and throat": ("Oral, Oropharyngeal, and Dental Examination",),
    "neck": ("Neck Examination",),
    "respiratory": ("Respiratory Examination",),
    "cardiovascular": ("Cardiovascular Examination",),
    "breast": ("Breast Examination",),
    "abdomen": ("Abdominal Examination",),
    "genitourinary": ("Genitourinary Examination",),
    "rectal": ("Rectal Examination",),
    "musculoskeletal": ("Musculoskeletal Examination",),
    "extremities": ("Extremity Examination",),
    "neurologic": ("Neurologic Examination",),
    "mental status": ("Mental Status Examination",),
    "other": ("Other Physical Examination",),
}
PE_ORDER_NAMES = tuple(PE_MENU_MAP)
PE_SYNONYM = {i: alias for alias, internals in PE_MENU_MAP.items() for i in internals}
PE_VOCAB = PE_ORDER_NAMES + tuple(PE_SYNONYM)
TEST_PAIRS_FULL: tuple[str, ...] = tuple(
    f"{l1}/{l2}"
    for l1, table in (
        ("Imaging", IMG74),
        ("Laboratory", LAB74),
        ("Pathology", PATH74),
        ("Physiologic", PHY74),
    )
    for l2 in dict.fromkeys(table.values())
)

TEST_SECTIONS = {
    "Laboratory_Test_Results": ("Laboratory", LAB_CATEGORY, LAB74),
    "Physiologic_Test_Results": ("Physiologic", PHYSIO_CATEGORY, PHY74),
    "Imaging_Results": ("Imaging", IMAGING_CATEGORY, IMG74),
    "Pathology_Results": ("Pathology", PATHOLOGY_CATEGORY, PATH74),
}

MENU2_REMAP: dict[tuple[str, str], tuple[str, str]] = {
    ("Imaging", "Mammography"): ("Imaging", "Other Imaging Modality"),
    ("Laboratory", "Toxicology Screening"): ("Laboratory", "General Laboratory Testing"),
    ("Pathology", "Microscopy"): ("Pathology", "Histopathology"),
    ("Physiologic", "Audiometry"): ("Physiologic", "Other Physiologic Testing"),
    ("Physiologic", "Hemodynamic Monitoring"): ("Physiologic", "Other Physiologic Testing"),
    ("Physiologic", "Provocative and Bedside Diagnostic Testing"): (
        "Physiologic",
        "Other Physiologic Testing",
    ),
    ("Physiologic", "Gastrointestinal Function Testing"): (
        "Physiologic",
        "Other Physiologic Testing",
    ),
    ("Physiologic", "Urodynamic Testing"): ("Physiologic", "Other Physiologic Testing"),
    ("Physiologic", "Functional and Performance Assessment"): (
        "Physiologic",
        "Other Physiologic Testing",
    ),
    ("Physiologic", "Bedside Physiologic Monitoring"): ("Physiologic", "Other Physiologic Testing"),
}


def served_pair(l1: str, category: str) -> tuple[str, str]:
    """The (l1, category) the environment SERVES a record under."""
    return MENU2_REMAP.get((l1, category), (l1, category))


TEST_PAIRS_MENU: tuple[str, ...] = tuple(
    dict.fromkeys("%s/%s" % served_pair(*p.split("/", 1)) for p in TEST_PAIRS_FULL)
)


SPECIALIST_OF: dict[tuple[str, str], str] = {
    ("Imaging", "Computed Tomography"): "radiology",
    ("Imaging", "Magnetic Resonance Imaging"): "radiology",
    ("Imaging", "Ultrasound"): "radiology",
    ("Imaging", "Computed Radiography"): "radiology",
    ("Imaging", "Nuclear Medicine"): "radiology",
    ("Imaging", "X-Ray Angiography"): "radiology",
    ("Imaging", "Mammography"): "radiology",
    ("Imaging", "Other Imaging Modality"): "radiology",
    ("Imaging", "Unmapped Imaging"): "radiology",
    ("Imaging", "Endoscopy"): "gastroenterology",
    ("Imaging", "Ophthalmic Photography"): "ophthalmology",
    ("Imaging", "External-Camera Photography"): "dermatology",
    ("Pathology", "Histopathology"): "pathology",
    ("Pathology", "Immunohistochemistry"): "pathology",
    ("Pathology", "Gross Examination"): "pathology",
    ("Pathology", "Cytopathology"): "pathology",
    ("Pathology", "Bone Marrow Examination"): "pathology",
    ("Pathology", "Electron Microscopy"): "pathology",
    ("Pathology", "Direct Immunofluorescence"): "pathology",
    ("Pathology", "Special Stains and Histochemistry"): "pathology",
    ("Pathology", "Peripheral Blood Smear"): "pathology",
    ("Pathology", "Surgical Pathology"): "pathology",
    ("Pathology", "Molecular Genetics"): "pathology",
    ("Pathology", "Microscopy"): "pathology",
    ("Laboratory", "Clinical Chemistry"): "laboratory_medicine",
    ("Laboratory", "Hematology"): "laboratory_medicine",
    ("Laboratory", "Microbiology"): "laboratory_medicine",
    ("Laboratory", "Urinalysis"): "laboratory_medicine",
    ("Laboratory", "Immunology"): "laboratory_medicine",
    ("Laboratory", "Endocrinology"): "laboratory_medicine",
    ("Laboratory", "General Laboratory Testing"): "laboratory_medicine",
    ("Laboratory", "Unmapped Laboratory"): "laboratory_medicine",
    ("Laboratory", "Molecular Genetics"): "laboratory_medicine",
    ("Laboratory", "Tumor Markers"): "laboratory_medicine",
    ("Laboratory", "Metabolic Screening"): "laboratory_medicine",
    ("Laboratory", "Coagulation"): "laboratory_medicine",
    ("Laboratory", "Flow Cytometry"): "laboratory_medicine",
    ("Laboratory", "Toxicology Screening"): "laboratory_medicine",
    ("Physiologic", "Cardiovascular Physiologic Testing"): "cardiology",
    ("Physiologic", "Echocardiography"): "cardiology",
    ("Physiologic", "Hemodynamic Monitoring"): "cardiology",
    ("Physiologic", "Bedside Physiologic Monitoring"): "cardiology",
    ("Physiologic", "Pulmonary Function Testing"): "cardiology",
    ("Physiologic", "Neurophysiologic Testing"): "neurology",
    ("Physiologic", "Psychological Testing"): "neurology",
    ("Physiologic", "Functional and Performance Assessment"): "neurology",
    ("Physiologic", "Audiometry"): "neurology",
    ("Physiologic", "Ophthalmic Examination and Testing"): "ophthalmology",
    ("Physiologic", "Gastrointestinal Function Testing"): "gastroenterology",
    ("Physiologic", "Urodynamic Testing"): "laboratory_medicine",
    ("Physiologic", "Provocative and Bedside Diagnostic Testing"): "laboratory_medicine",
    ("Physiologic", "Other Physiologic Testing"): "laboratory_medicine",
}

SPECIALISTS = (
    "radiology",
    "pathology",
    "cardiology",
    "neurology",
    "dermatology",
    "ophthalmology",
    "gastroenterology",
    "laboratory_medicine",
)


def specialist_of(l1: str, category: str) -> str:
    """The department that reads a study of this SERVED (l1, category) pair."""
    try:
        return SPECIALIST_OF[(l1, category)]
    except KeyError:
        raise KeyError(
            f"no specialist for ({l1!r}, {category!r}): SPECIALIST_OF must be "
            "total over every served (l1, category) pair — add the line to the "
            "table in environment.py"
        ) from None


def canon_dept(s: str) -> str:
    """ "Laboratory Medicine" / "  RADIOLOGY " -> the department key. Parses the
    action payload only; the routing itself is SPECIALIST_OF's job.
    """
    return re.sub(r"[\s\-]+", "_", s.lower().strip())


VERBS = (
    "ask",
    "begin_workup",
    "physical_examination",
    "order_tests",
    "consult_specialist",
    "search",
    "defer_to_human",
    "final_diagnosis",
)

DEFER_VERB = "defer_to_human"
ABSTENTION_VERB = "abstention"


def surface_defer_verb() -> str:
    """What the policy is told to type for the abstention, at this contract."""
    return ABSTENTION_VERB


NON_DIAGNOSES = frozenset(
    {
        "undetermined",
        "unknown",
        "unclear",
        "uncertain",
        "cannot determine",
        "not determined",
        "no diagnosis",
    }
)
_NO_DX_PUNCT = re.compile(r"[^a-z0-9\s]+")
_NO_DX_WS = re.compile(r"\s+")
_NO_DX_ENUM_RE = re.compile(r"^\s*(?:[(\[]?(?:\d{1,2}|[a-z])[.)\]:]\s+)+")
_NO_DX_HEDGE_RE = re.compile(
    r"^(?:the |a |an )?(?:(?:definitive|specific|final|precise|exact|single|confident|clear|safe) )?"
    r"(?:(?:diagnosis|dx) )?"
    r"(?:cannot|can not|could not|can t|unable|not\b|insufficient|impossible|more information|"
    r"none\b|n a\b|na$|tbd\b|to be determined|diagnostic uncertainty|"
    r"uncertaint(?:y|ies)\b|certainty\b|ambiguity\b|dilemma\b|unavailable\b|nil\b|"
    r"incomplete (?:history|information|info|data|workup|evaluation|assessment|exam|examination|testing|imaging|"
    r"records?|objective|laboratory|available|diagnostic)|inadequate\b|"
    r"lack(?:s|ing)? (?:of )?(?:sufficient |adequate |enough |the |a |any )?"
    r"(?:information|info|data|history|evidence|details?|objective|findings|clinical|diagnostic|specificity|certainty|definitive)|"
    r"missing (?:data|information|info|history|details?|key|crucial|objective|clinical|exam|findings)|"
    r"too (?:early|soon|little|limited|few|vague|nonspecific|non specific|broad|many)\b|without\b|awaiting\b|"
    r"defer(?:red|ring|ral)?(?: (?:to|until|pending|due|for|diagnosis|final|a|the|further|workup|evaluation|insufficient|not)\b|$)|"
    r"work ?ups?\b|please\b|proceed\b|until\b|postpone\w*\b|wait(?:ing)?\b|skip\b|"
    r"leave\b|plea\b|refer(?:ral)?(?: (?:for|to|needed|required|not|out|urgently|indicated|recommended|is|and)\b|$)|"
    r"ruled (?:out|indeterminate|insufficient|unlikely|inconclusive|uncertain|undetermined|negative)\b|"
    r"needs?\b|needed\b|requires?\b|required\b|requiring\b|further\b|"
    r"unlikely\b|improbable\b|"
    r"(?:(?:urgent|emergency|emergent|immediate|imminent|impending|rapid|ongoing|further|subsequent|initial|thorough|"
    r"investigational|exploratory|requested|targeted|focused|prompt|formal|full|complete|comprehensive|additional|more|"
    r"extensive|appropriate|specialist|inpatient|outpatient|objective|systematic|proper|detailed|dedicated|first|line|"
    r"guided|tissue|biopsy|imaging|laboratory|lab|ecg|eeg|echo|mri|ct|cardiac|renal|ocular|dental|oral|nasal|"
    r"\w+(?:ologic|ological|ology|graphic|scopic|al|ic|ary|ive)) ){0,3}"
    r"(?:diagnostic )?(?:work ?ups?|evaluation|assessment|investigations?|testing|examination|referral) "
    r"(?:is |are |urgently |immediately |strongly |now |still )?"
    r"(?:required|needed|indicated|pending|recommended|warranted|necessary|mandatory|essential|advised|suggested|first|"
    r"before|prior|rather|more|to (?:be|determine|define|distinguish|exclude|clarify|evaluate|identify|confirm|rule|assess|establish)|"
    r"for|of|and|planned|ongoing|incomplete|nondiagnostic|failure|begun|not)\b|"
    r"(?:multiple|several|many|broad|wide|numerous) (?:possibilities|possible|differentials?|diagnoses|conditions|etiologies|causes|candidates)|"
    r"no (?:\w+ )?(?:diagnosis|dx|cause|etiology|answer|conclusion|entity|condition|label|disease|"
    r"disorder|determination|findings?)|no definitive|no specific|no single|no clear|"
    r"(?:undetermined|unclear|uncertain|indeterminate|inconclusive|unspecified|undiagnosed|pending|unsure|incomplete|"
    r"undecided|unestablished|unconfirmed|unidentified|undefined|unavailable|unsatisfactory|unsolved|unanswered|"
    r"unsupported|unverified|undocumented|ambiguous|improbable|indefinite|nondiagnostic|non diagnostic|indiagnostic)"
    r"(?: (?:diagnosis|dx|diagnoses|etiology|etiologies|cause|causes|category|nature|source|origin|classification|"
    r"differentials?|possibilities|possibility|considerations?|concerns?|suspicion|suspect|suspected|"
    r"workup|work|evaluation|examination|exam|testing|tests?|imaging|labs?|biopsy|tissue|assessment|investigations?|"
    r"history|hpi|data|information|info|details?|patient|presentation|symptoms|features|findings|course|picture|"
    r"objective|answers?|conclusion|results?|review|questions?|context|setting|"
    r"at|without|from|based|until|due|given|because|pending|deferred|further|additional|more|insufficient|inadequate|"
    r"not|no|none|whether|between|as|so|but|than|then|thus|therefore|hence|if|unless|while|since|though|although|however|"
    r"the|a|an|this|that|these|those|for|of|in|with|by|on|to|about|regarding|per|via|among|vs|versus|"
    r"is|are|was|were|be|been|being|has|have|had|do|does|did|cannot|can|could|would|should|may|might|will|shall|must|"
    r"needs?|requires?|requiring|awaiting|awaits|suggests?|suggested|indicates?|indicated|warrants?|warranted|"
    r"remains?|remained|seems?|appears?|see|refer|obtain|proceed|treat|please|evaluate|examine|order|check|"
    r"most|likely|possibly|possible|probable|probably|presumed|presumptive|provisional|potentially|maybe|perhaps|"
    r"favou?rs?|favou?ring|favou?red|consider|considering|considered|leaning|leans|rule|ddx|"
    r"highly|highest|leading|top|main|key|principal|priority|next|immediate|immediately|urgent|urgently|"
    r"emergency|emergent|prompt|promptly|specifically|definitive|definitively|definite|precise|exact|confident|"
    r"confidently|clearly|safe|safely|reliable|reliably|certain|certainly|conclusive|conclusively|fully|"
    r"still|yet|currently|presently|now|today|here|there|inability|unable|impossible|difficult|"
    r"too|only|also|even|working|best)\b|$)|"
    r"pending (?:further|additional|results?|workup|evaluation|biopsy|imaging|histolog)|"
    r"difficult to (?:determine|diagnose|establish|say|make|assess|pinpoint))(?:\b|$)"
)
_NO_DX_UNKNOWN_RE = re.compile(
    r"^unknown\b(?!\s+primary\s+(?:carcinoma|cancer|tumou?r|malignancy|adenocarcinoma|melanoma|neoplasm|"
    r"sarcoma|lymphoma|squamous|neuroendocrine|metasta\w*|site|origin|\w+carcinoma|\w+oma\b))"
)
_NO_DX_SEP_RE = re.compile(
    r"^\W*(?:the |a |an )?(?:undetermined|unclear|uncertain|uncertaint(?:y|ies)|indeterminate|inconclusive|unknown|"
    r"unspecified|undiagnosed|none|n/?a|tbd|pending|not diagnostic|not sufficient|not enough|no diagnosis|"
    r"diagnostic uncertainty|unsure|undecided|unresolved|unidentified|unconfirmed|unestablished|undefined|deferred|"
    r"deferral|defer|awaiting|nondiagnostic|non-diagnostic|non diagnostic|indiagnostic|insufficient|incomplete|inadequate|"
    r"unavailable|unsatisfactory|unlikely|improbable|nil|no|missing|absent|postponed?|skip|wait|unsolved|unanswered|"
    r"unsupported|unscheduled|undetectable|ineffective|incapable|unreliable|unverified|undocumented|inapparent|ambiguous|"
    r"ambiguity|low[- ]yield|limitation|unguessed|unenthusiastic|warning|defeat|indefinite)"
    r"\s*(?:[\u2014\u2013:;,.?!(\[]|\s-+\s|--|$)"
)
_NO_DX_LEADIN_WORDS = (
    r"i|we|it|this|that|there|here|the|a|an|my|our|current|currently|present|presently|working|"
    r"final|definitive|definite|conclusive|confirmed|presumptive|specific|precise|exact|single|confident|clear|safe|"
    r"diagnosis|diagnoses|diagnose|diagnosed|diagnostic|dx|answer|answers|case|impression|assessment|conclusion|summary|"
    r"overall|etiology|aetiology|etiologic|etiological|causative|proximate|cause|causes|underlying|condition|"
    r"disease|disorder|process|entity|nature|origin|source|primary|unifying|leading|top|best|highest|priority|"
    r"chief|clinical|clinically|medically|medical|record|pathologic|pathological|informed|initial|upfront|"
    r"unequivocal|eponymous|most likely|likely|probable|probably|likelihood|available|information|info|data|"
    r"history|evidence|findings|facts|details|detail|analysis|criteria|status|stage|timing|sequence|category|"
    r"classification|picture|presentation|provided|given|limited|nonspecific|non specific|atypical|notable|"
    r"investigational|differential|differentials|hypotheses|hypothesis|problem|issue|question|questions|result|"
    r"results|resolution|decision|determination|distinction|critical|exclusion|onset|genes|incidence|interim|"
    r"type|form|kind|class|causality|diagnostically|"
    r"review|reviewed|reconstruction|consolidated|retrospective|succinct|electronic|protocol|"
    r"child|patient|infant|newborn|neonate|woman|man|boy|girl|"
    r"based on|based|on the basis of|basis|on|in|from|with|alone|only|also|and|due to|before|"
    r"at this point|at this stage|at this time|at present|as of now|for now|so far|to date|thus far|"
    r"is|are|am|m|be|been|remains|remain|remained|was|were|stays|still|have|has|had|do|does|did|would|will|shall|"
    r"should|must|therefore|thus|so|unfortunately|honestly|frankly|really|truly|simply|genuinely|absolutely|"
    r"permanently|temporarily|suggest|suggests|suggested|emergency|emergent|urgent|immediate"
)
_NO_DX_LEADIN_RE = re.compile(r"^(?:(?:" + _NO_DX_LEADIN_WORDS + r") )+")
_NO_DX_RAW_LEADIN_RE = re.compile(r"^(?:(?:" + _NO_DX_LEADIN_WORDS + r")[:\s]+)+")
NO_DX_ERR_TAG = "no diagnosis named"


def _no_dx_raw_primary(text: str) -> str:
    """The primary diagnosis (before any '|'), lower-cased, list enumerators
    stripped, punctuation kept (the separator rule reads it).
    """
    return _NO_DX_ENUM_RE.sub("", str(text or "").split("|", 1)[0].strip().lower()).strip()


def no_diagnosis_primary(text: str) -> str:
    """The primary diagnosis (before any '|'): lower-cased, list enumerators
    stripped, punctuation to space, whitespace collapsed (digits stay).
    """
    return _NO_DX_WS.sub(" ", _NO_DX_PUNCT.sub(" ", _no_dx_raw_primary(text))).strip()


def no_diagnosis_final(text: str) -> bool:
    """True when a final_diagnosis names no disease: one of NON_DIAGNOSES, or a
    text that opens with a no-diagnosis phrase ("Unknown - further evaluation
    required", "Insufficient information for a definitive diagnosis", "Cannot
    determine from available data. Probable ..."). A named disease followed by
    hedge words ("X (less likely Y)") is NOT flagged.
    """
    p = no_diagnosis_primary(text)
    if not p:
        return False
    if p in NON_DIAGNOSES:
        return True
    raw = _no_dx_raw_primary(text)
    if _NO_DX_SEP_RE.match(raw) or _NO_DX_SEP_RE.match(_NO_DX_RAW_LEADIN_RE.sub("", raw)):
        return True
    head = _NO_DX_LEADIN_RE.sub("", p)
    if head in NON_DIAGNOSES:
        return True
    return bool(_NO_DX_HEDGE_RE.match(head) or _NO_DX_UNKNOWN_RE.match(head))


def no_dx_contract(cfg: "Config") -> str:
    """What this Config stamps into budgets["no_dx_final"]: "reject", a final that names no
    disease is refused at emission.
    """
    return "reject"


def no_dx_resume_preflight(traj_path: "Path", cfg: "Config") -> None:
    """Refuse to append records under one no-diagnosis contract to a
    trajectories file collected under another: the arm's contract fingerprint
    (Eval/code/baselines/common.py) would drift and the scorer would reject the
    arm only after the tokens were spent.
    """
    if not traj_path.exists():
        return
    want = no_dx_contract(cfg)
    with traj_path.open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            have = (json.loads(line).get("budgets") or {}).get("no_dx_final", "accept")
            if have != want:
                raise SystemExit(
                    f"{traj_path}: record {n} was collected with "
                    f"no_dx_final={have!r}, this run stamps {want!r}; "
                    "finish them in a fresh --out (the arm's contract "
                    "fingerprint would drift)"
                )


_ASK_STOP = frozenset(
    "a an and any are as at be been being but can could describe did do does "
    "els else ever feel felt for from get had has have having how i if in is "
    "it its just like me more most notice noticed of on or other please "
    "recall recent recently share specifically tell that the them then there "
    "these they this those to us was we were what when where whether which "
    "with would you your".split()
)


def _ask_words(q: str) -> frozenset[str]:
    return frozenset(
        w for w in re.findall(r"[a-z]+", q.lower()) if len(w) > 2 and w not in _ASK_STOP
    )


def ask_overlap(q: str, previous: list[frozenset[str]]) -> float:
    """Overlap coefficient |A∩B| / min(|A|,|B|) against the closest earlier
    question — robust to a repeat being a shorter or synonym-padded rephrase.
    """
    words = _ask_words(q)
    if not words:
        return 0.0
    best = 0.0
    for prev in previous:
        if prev:
            best = max(best, len(words & prev) / min(len(words), len(prev)))
    return best


_ASK_TERMINATORS = ".?!"
_ASK_OPEN, _ASK_CLOSE = "([{", ")]}"

MAX_ASK_SENTENCES = 2
MAX_ASK_QUESTIONS = 2


def ask_shape(text: Any) -> tuple[int, int]:
    """(sentences, question marks) for one ask bundle — THE shape measurement."""
    s = str(text or "")
    depth = 0
    pieces: list[str] = []
    cur: list[str] = []
    for ch in s:
        if ch in _ASK_OPEN:
            depth += 1
        elif ch in _ASK_CLOSE:
            depth = max(0, depth - 1)
        elif ch in _ASK_TERMINATORS and depth == 0:
            pieces.append("".join(cur))
            cur = []
            continue
        cur.append(ch)
    pieces.append("".join(cur))
    return len([p for p in pieces if p.strip()]), s.count("?")


def _sect_key(s: str) -> str:
    return normalise(s).replace(" ", "_")


def classify_pe(section: str, location: str | None) -> str:
    fine = PE_SECTION_FINE.get(_sect_key(section))
    nloc = normalise((location or "").strip())
    if not nloc:
        site = "Unspecified site"
    elif fine == "Vital_Signs":
        site = apply_rules(VITAL_CONCEPT, nloc, "Vital signs (composite)")
    else:
        site = apply_rules(PE_REGION, nloc, "Unmapped site")
    return PANEL74.get(_OM.PE_SITE_TO_PANEL.get(site), PE_OTHER)


def classify_test(section: str, test_name: str | None, modality: str | None) -> str:
    _, rules, bridge = TEST_SECTIONS[section]
    nm = normalise((test_name or "").strip())
    probe = f"{nm} {(modality or '').strip().lower().replace('_', ' ')}".strip()
    local = apply_rules(rules, probe, "Unmapped")
    return bridge.get(local) or bridge["Unmapped"]


_PE_MENU = ", ".join(PE_ORDER_NAMES)


def test_menu() -> str:
    """The L1 categories block, in the pair order of the menu."""
    by_l1: dict[str, list[str]] = {}
    for pair in TEST_PAIRS_MENU:
        l1, _, l2 = pair.partition("/")
        by_l1.setdefault(l1, []).append(l2)
    return "\n".join(f"  {l1}: " + ", ".join(l2s) for l1, l2s in by_l1.items())


def dept_menu() -> str:
    return ", ".join(SPECIALISTS)


def defer_slot_error(payload: Any) -> str | None:
    """reject an abstention whose reason is the prompt's own slot."""
    slot_err = DEFER_SLOT_ERR
    text = str(payload or "").strip()
    reason = text
    for prefix in DEFER_PREFIXES:
        if text.lower().startswith(prefix):
            reason = text[len(prefix) :].lstrip(" \t:—–-").strip()
            break
    if "[" in text or "]" in text:
        return slot_err
    phrases = DEFER_SLOT_PHRASES
    normalised = " ".join(re.findall(r"[a-z]+", reason.lower()))
    if any(phrase in normalised for phrase in phrases):
        return slot_err
    return None


MISMATCH_DEFER = " — if the image does not match the patient or the requested study, defer_to_human"
MISMATCH_NOTE = " — note any mismatch in your reasoning and weigh that study accordingly"


GATE_REWARD = {
    "use_clean": 1.0,
    "defer_exact": 1.25,
    "defer_partial": 0.75,
    "miss": -2.0,
    "false_defer": -0.5,
}


def gate_axes_exact(axes: list[str], gt_axes: list[str]) -> bool:
    """Did the verdict name the right axis? Contract 3 says one is enough."""
    return bool(axes) and set(axes) <= set(gt_axes)


def gate_reward(gt_mismatch: bool, gt_axes: list[str], verdict: str, axes: list[str]) -> float:
    """Mirrors evidence_corruption.CorruptionEngine.reward (kept in sync; the
    contract-3 axis rule is this env's own — see `gate_axes_exact`).
    """
    if not gt_mismatch:
        return GATE_REWARD["use_clean"] if verdict == "pass" else GATE_REWARD["false_defer"]
    if verdict == "pass":
        return GATE_REWARD["miss"]
    return (
        GATE_REWARD["defer_exact"]
        if gate_axes_exact(axes, gt_axes)
        else GATE_REWARD["defer_partial"]
    )


GATE_AXES = ("modality", "anatomy")


CORRUPTION_AXES: dict[str, tuple[str, ...]] = {
    "modality_anatomy": ("modality", "anatomy"),
}


def corruption_axes(kind: str) -> list[str]:
    """The closed-vocabulary axes for a corruption kind. Unknown kinds fall
    back to the image default, which is what `gate_ground_truth` has always
    returned for a raw-pool swap.
    """
    return list(CORRUPTION_AXES.get(kind, ("modality", "anatomy")))


def parse_gate(raw: str) -> tuple[str, str, list[str], str, str | None]:
    """-> (think, verdict, axes, reason, err). Free-form tolerant, order fixed."""
    tm = re.search(r"<reasoning>(.*?)</reasoning>", raw, re.S | re.I)
    am = re.search(r"<action>(.*?)</action>", raw, re.S | re.I)
    if not am:
        return (tm.group(1).strip() if tm else ""), "", [], "", "missing <action> </action> block"
    think = tm.group(1).strip() if tm else ""
    body = re.sub(r"\s+", " ", am.group(1)).strip()
    if not body.lower().startswith("gate_verdict"):
        return think, "", [], "", "action must be 'gate_verdict: ...'"
    payload = body.partition(":")[2].strip()
    low = payload.lower()
    if low.startswith("pass"):
        return think, "pass", [], "", None
    if not low.startswith("mismatch"):
        return think, "", [], "", "verdict must be 'pass' or 'mismatch — ...'"
    axes = [a for a in GATE_AXES if re.search(rf"\b{a}\b", low)]
    rm = re.search(r"reason\s*:\s*(.*)$", payload, re.I)
    return think, "mismatch", axes, (rm.group(1).strip() if rm else ""), None


def _as_list(v: Any) -> list[dict]:
    if v is None:
        return []
    if isinstance(v, dict):
        return [v] if v else []
    return [x for x in v if isinstance(x, dict) and x]


def _j(v: Any) -> Any:
    return json.loads(v) if isinstance(v, str) and v.strip() else v


_AGE_PHRASE = (
    r"\d+\s*[-‐-― ]?\s*(?:year|month|week|day)s?"
    r"[-‐-― ]?\s*old"
)
_PERSON_NOUN = (
    r"(?:patient|man|men|woman|women|male|female|boy|girl|child|"
    r"children|infant|neonate|newborn|baby|toddler|adolescent|"
    r"teenager|sibling|brother|sister|twin|proband|volunteer)s?"
)
_PERSON_MODIFIERS = r"(?:[A-Za-z][A-Za-z'’-]*\s+){0,3}"
_SOURCE_DEMOGRAPHICS_RE = re.compile(
    r"\s*(?:[,;]\s*)?[(\[]?\s*"
    r"(?:\b(?:from|in|of|for|on)\s+(?:a|an|the|this|our|another)\s+)?"
    rf"{_AGE_PHRASE}\s+{_PERSON_MODIFIERS}{_PERSON_NOUN}\b\s*[)\]]?\.?",
    re.IGNORECASE,
)


def sanitize_record_name(name: str) -> str:
    """A served study name with the source article's patient phrase removed."""
    if not name:
        return name
    out = _SOURCE_DEMOGRAPHICS_RE.sub(" ", str(name))
    out = re.sub(r"\s+", " ", out).strip().strip(",;:").strip()
    return out if re.search(r"[A-Za-z]", out) else str(name)


def served_name(raw: str) -> tuple[str, str | None]:
    """(the name the doctor sees, the profile's own name when they differ)."""
    raw = raw or ""
    clean = sanitize_record_name(raw)
    return (clean, raw) if clean != raw else (raw, None)


@dataclass(slots=True)
class Record:
    """One examined finding / test result, classified into the 74-space."""

    l1: str
    category: str
    name: str
    findings: str
    asset_id: str | None = None
    asset_path: str | None = None
    asset_category: str | None = None
    serving_kind: str | None = None
    native_category: str | None = None
    original_name: str | None = None
    rid: int = -1
    panel_id: str | None = None
    panel_modality: str | None = None
    parent_image_id: str | None = None
    panel_status: str | None = None


@dataclass(slots=True)
class Episode:
    pmcid: str
    encounter_facts: list[dict]
    patient_facts: list[dict]
    fact_index: dict[str, dict]
    speaker_role: str
    demographics: str
    records: list[Record]
    corruption: dict | None = None
    corruption_armed: bool = False
    corrupt_bind_k: int = 0

    def __post_init__(self) -> None:
        for i, r in enumerate(self.records):
            r.rid = i


_PANEL_ASSETS_CACHE: dict[str, dict] = {}
_PANEL_MISSING_WARNED: set[str] = set()


def load_panel_assets(path: Path) -> dict:
    """-> {"map": {(image_id, panel_id): {panel_asset_id, crop_path}},
    "by_image": {image_id: [crop_path, ...]}, "sha1"}.
    """
    key = str(Path(path).resolve())
    hit = _PANEL_ASSETS_CACHE.get(key)
    if hit is not None:
        return hit
    import hashlib
    import pandas as pd

    raw = Path(path).read_bytes()
    df = pd.read_parquet(path)
    for col in ("image_id", "panel_id", "panel_asset_id", "crop_path"):
        if col not in df.columns:
            raise ValueError(
                f"panel manifest {path} has no {col!r} column (got {list(df.columns)})"
            )
    m: dict[tuple[str, str], dict] = {}
    by_image: dict[str, list[str]] = {}
    orphans: list[str] = []
    grouped = "group_asset_id" in df.columns
    for r in df.to_dict("records"):
        iid, pid = str(r["image_id"]), str(r["panel_id"])
        aid = str(r["panel_asset_id"])
        if aid != f"{iid}#{pid}":
            if len(orphans) < 5:
                orphans.append(f"{aid!r} (expected {iid}#{pid!r})")
            continue
        entry = {"panel_asset_id": aid, "crop_path": r["crop_path"], "image_id": iid}
        sk = r.get("serving_kind")
        if sk is not None and not isinstance(sk, float):
            entry["serving_kind"] = str(sk)
        gid, gpath = (
            (r.get("group_asset_id"), r.get("group_crop_path")) if grouped else (None, None)
        )
        if gid and not (isinstance(gid, float)):
            if not str(gid).startswith(f"{iid}#"):
                if len(orphans) < 5:
                    orphans.append(f"group {gid!r} (expected prefix {iid}#)")
                continue
            entry["group_asset_id"] = str(gid)
            entry["group_crop_path"] = str(gpath)
        served_rel = entry.get("group_crop_path", r["crop_path"])
        m[(iid, pid)] = entry
        if served_rel not in by_image.setdefault(iid, []):
            by_image[iid].append(served_rel)
    if orphans:
        raise ValueError(
            f"panel manifest {path}: {len(orphans)}+ crop(s) do not name their "
            f"parent image as '<image_id>#<panel_id>' — {'; '.join(orphans)}. "
            "Every served crop must be traceable to its figure; fix the "
            "manifest in code/panel_clean rather than serving it"
        )
    out = {"map": m, "by_image": by_image, "sha1": hashlib.sha1(raw).hexdigest()}
    _PANEL_ASSETS_CACHE[key] = out
    return out


def serving_kind_of(cfg: Config, image_id: str | None, panel_id) -> str | None:
    """The manifest `serving_kind` of this record's row, or None (no manifest,
    no row, or a legacy manifest without the column). None counts as
    victim-eligible — see VICTIM_SERVING_KINDS.
    """
    if cfg.panel_assets is None or not image_id or panel_id in (None, ""):
        return None
    pa = load_panel_assets(cfg.panel_assets)["map"].get((image_id, str(panel_id)))
    return pa.get("serving_kind") if pa else None


def panel_overlay(cfg: Config, image_id: str | None, panel_id) -> tuple[str, str] | None:
    """(asset_id, asset_path) for this record's OWN panel crop, or None."""
    aid, path, _ = resolve_panel_asset(cfg, image_id, panel_id)
    return (aid, path) if aid else None


def resolve_panel_asset(
    cfg: Config, image_id: str | None, panel_id
) -> tuple[str | None, str | None, str]:
    """-> (crop asset_id, crop path, status)."""
    if cfg.panel_assets is None:
        return None, None, "off"
    if not image_id or panel_id in (None, ""):
        return None, None, "no_panel_id"
    pa = load_panel_assets(cfg.panel_assets)["map"].get((image_id, str(panel_id)))
    if pa is None:
        return None, None, "unbound"
    if "group_asset_id" in pa:
        gp = cfg.panel_assets_dir / pa["group_crop_path"]
        if gp.exists():
            return pa["group_asset_id"], str(gp), "crop"
        if pa["group_crop_path"] not in _PANEL_MISSING_WARNED:
            _PANEL_MISSING_WARNED.add(pa["group_crop_path"])
            print(
                f"warning: group image missing from the tree, serving the "
                f"panel crop instead: {pa['group_crop_path']}"
            )
    p = cfg.panel_assets_dir / pa["crop_path"]
    if not p.exists():
        if pa["crop_path"] not in _PANEL_MISSING_WARNED:
            _PANEL_MISSING_WARNED.add(pa["crop_path"])
            print(f"warning: panel crop missing from the tree: {pa['crop_path']}")
        return None, None, "missing_file"
    return pa["panel_asset_id"], str(p), pa.get("serving_kind", "crop")


def serve_asset(
    cfg: Config, im: dict | None, panel_id
) -> tuple[str | None, str | None, str | None, str | None]:
    """-> (asset_id, asset_path, parent_image_id, panel_status) for one record."""
    if not im:
        return None, None, None, None
    parent = im["image_id"]
    whole = str(cfg.images_dir / im["local_path"])
    aid, path, status = resolve_panel_asset(cfg, parent, panel_id)
    if status in SERVED_PANEL_STATUSES:
        return aid, path, parent, status
    if status == "off" or cfg.panel_policy == "whole_figure":
        return parent, whole, parent, (None if status == "off" else status)
    return None, None, parent, status


def parse_profile(
    row: dict,
    cfg: Config,
    swap_pool: list[tuple[str, str, str]] | None = None,
    rng: random.Random | None = None,
) -> Episode:
    osce = _j(row["OSCE_Examination"])
    enc = [{"section": "Encounter_Context", **f} for f in _as_list(osce.get("Encounter_Context"))]
    facts: list[dict] = []
    for section in ACTOR_SECTIONS:
        for f in _as_list(osce["Patient_Actor"].get(section)):
            fact = {"section": section, "fact_id": f["fact_id"], "atomic_fact": f["atomic_fact"]}
            fact.update({a: f[a] for a in FACT_ATTRS if f.get(a)})
            facts.append(fact)
    demo = " ".join(f["atomic_fact"] for f in facts if f["section"] == "Demographics")
    m = AGE_YEARS.search(demo)
    child = (m and int(m.group(1)) < 12) or bool(CAREGIVER.search(demo))

    images = {im["image_id"]: im for im in _as_list(_j(row.get("images")) or [])}
    records: list[Record] = []
    pe = _j(row.get("Physical_Examination_Findings")) or {}
    for section, items in pe.items():
        for it in _as_list(items):
            pe_name, pe_orig = served_name(it.get("location") or section.replace("_", " "))
            records.append(
                Record(
                    l1="Physical_Examination",
                    category=classify_pe(section, it.get("location")),
                    name=pe_name,
                    original_name=pe_orig,
                    findings=it.get("findings", ""),
                )
            )
    for section, (l1, _, _) in TEST_SECTIONS.items():
        for it in _as_list(_j(row.get(section)) or []):
            im = images.get(it.get("image_id") or "")
            native = classify_test(section, it.get("test_name"), it.get("modality"))
            _, served = served_pair(l1, native)
            t_name, t_orig = served_name(it.get("test_name", ""))
            a_id, a_path, a_parent, a_status = serve_asset(cfg, im, it.get("panel_id"))
            records.append(
                Record(
                    l1=l1,
                    category=served,
                    name=t_name,
                    original_name=t_orig,
                    findings=it.get("findings", ""),
                    asset_id=a_id,
                    asset_path=a_path,
                    asset_category=im.get("category") if im else None,
                    serving_kind=(
                        serving_kind_of(cfg, im["image_id"], it.get("panel_id"))
                        if a_id and im
                        else None
                    ),
                    native_category=native,
                    panel_id=(
                        str(it["panel_id"]) if im and it.get("panel_id") not in (None, "") else None
                    ),
                    panel_modality=it.get("modality") if im else None,
                    parent_image_id=a_parent,
                    panel_status=a_status,
                )
            )

    if cfg.post_baseline:
        for it in _as_list(_j(row.get("Post_Baseline_Findings")) or []):
            section = str(it.get("section") or "")
            tp = str(it.get("timepoint") or "").strip()
            tag = f"(post-treatment — {tp})" if tp else "(post-treatment)"
            if section == "Physical_Examination_Findings":
                pe_name, pe_orig = served_name(it.get("test_name") or "Post-treatment examination")
                records.append(
                    Record(
                        l1="Physical_Examination",
                        category=classify_pe(section, it.get("test_name")),
                        name=f"{pe_name} {tag}",
                        original_name=pe_orig,
                        findings=it.get("findings", ""),
                    )
                )
                continue
            if section not in TEST_SECTIONS:
                continue
            l1 = TEST_SECTIONS[section][0]
            im = images.get(it.get("image_id") or "")
            native = classify_test(section, it.get("test_name"), it.get("modality"))
            _, served = served_pair(l1, native)
            t_name, t_orig = served_name(it.get("test_name", ""))
            a_id, a_path, a_parent, a_status = serve_asset(cfg, im, it.get("panel_id"))
            records.append(
                Record(
                    l1=l1,
                    category=served,
                    name=f"{t_name} {tag}",
                    original_name=t_orig,
                    findings=it.get("findings", ""),
                    asset_id=a_id,
                    asset_path=a_path,
                    asset_category=im.get("category") if im else None,
                    serving_kind=(
                        serving_kind_of(cfg, im["image_id"], it.get("panel_id"))
                        if a_id and im
                        else None
                    ),
                    native_category=native,
                    panel_id=(
                        str(it["panel_id"]) if im and it.get("panel_id") not in (None, "") else None
                    ),
                    panel_modality=it.get("modality") if im else None,
                    parent_image_id=a_parent,
                    panel_status=a_status,
                )
            )

    rng = rng or random.Random(f"corrupt::{row['pmcid']}::{cfg.corrupt_seed}")
    armed = bool(cfg.corrupt_rate > 0 and swap_pool and rng.random() < cfg.corrupt_rate)
    bind_k = (1 if rng.random() < 0.7 else 2) if armed else 0

    return Episode(
        pmcid=row["pmcid"],
        encounter_facts=enc,
        patient_facts=facts,
        fact_index={f["fact_id"]: f for f in enc + facts},
        speaker_role="caregiver" if child else "patient",
        demographics=demo,
        records=records,
        corruption=None,
        corruption_armed=armed,
        corrupt_bind_k=bind_k,
    )


def load_profiles(path: Path) -> list[dict]:
    if path.suffix == ".parquet":
        import pandas as pd

        return pd.read_parquet(path).to_dict("records")
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_donor_list(cfg: Config) -> list[tuple[str, str, str]]:
    """The file-backed donor pool (Config.donor_list): rows in FILE ORDER, each
    resolved against panel_assets_dir, kept only when the crop exists on disk.
    Order matters -- `_maybe_bind_corruption` draws with a seeded rng over the
    list, so a reordered file would bind different donors. A row without a
    category is refused: the bind-time filter keys on it.
    """
    path = Path(cfg.donor_list)
    if not path.is_file():
        raise FileNotFoundError(f"donor_list not found: {path}")
    pool: list[tuple[str, str, str]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if not d.get("category"):
                raise ValueError(f"donor row without category: {d}")
            p = cfg.panel_assets_dir / d["crop_path"]
            if p.is_file():
                pool.append((str(d["pmcid"]), str(p), str(d["category"])))
    return pool


def build_swap_pool(rows: list[dict], cfg: Config, cap: int = 400) -> list[tuple[str, str, str]]:
    """(pmcid, absolute image path, asset category) pool for corruption swaps."""
    if cfg.donor_list is not None:
        return load_donor_list(cfg)
    pool: list[tuple[str, str, str]] = []
    crops = (
        load_panel_assets(cfg.panel_assets)["by_image"] if cfg.panel_assets is not None else None
    )
    for row in rows[:cap]:
        for im in _as_list(_j(row.get("images")) or []):
            if not im.get("local_path"):
                continue
            cat = im.get("category", "")
            rels = crops.get(im.get("image_id") or "") if crops is not None else None
            if rels:
                pool.extend((row["pmcid"], str(cfg.panel_assets_dir / rel), cat) for rel in rels)
            elif crops is None or cfg.panel_policy == "whole_figure":
                pool.append((row["pmcid"], str(cfg.images_dir / im["local_path"]), cat))
    return pool


def bind_telemetry(ep: "Episode", env: "Env") -> dict:
    """The two bind facts a dud/exposure analysis needs, per episode: the drawn k
    and how many eligible-bearing (image) returns actually happened.
    Serialized because neither the slim feed nor the metrics blob carried
    them — the pilot-24 dud-rate question ("how many full-coverage armed
    episodes were k=2 single-batch?") was unanswerable from stored runs.
    """
    return {"corrupt_bind_k": int(ep.corrupt_bind_k), "image_injections": int(env.image_injections)}


def _panel_serving_counts(records: list["Record"]) -> dict:
    """{panel_status: n} over the records that carry a figure — the per-episode
    face of `panel_coverage`, written into metrics so a run's crop rate is
    recomputable from the trajectories alone, with no manifest in hand.
    """
    out: dict[str, int] = {}
    for r in records:
        if r.panel_status:
            key = r.serving_kind or r.panel_status
            out[key] = out.get(key, 0) + 1
    return out


def imaged_items(row: dict, cfg: Config):
    """Yield (image dict, panel_id) for every record of `row` that carries a
    picture — the same sections, in the same order, that `parse_profile` builds
    image-bearing records from, `post_baseline` included when it is on.
    """
    images = {im["image_id"]: im for im in _as_list(_j(row.get("images")) or [])}
    sections = list(TEST_SECTIONS)
    for section in sections:
        for it in _as_list(_j(row.get(section)) or []):
            im = images.get(it.get("image_id") or "")
            if im:
                yield im, it.get("panel_id")
    if cfg.post_baseline:
        for it in _as_list(_j(row.get("Post_Baseline_Findings")) or []):
            if str(it.get("section") or "") in TEST_SECTIONS:
                im = images.get(it.get("image_id") or "")
                if im:
                    yield im, it.get("panel_id")


def panel_coverage(rows: list[dict], cfg: Config) -> dict:
    """How much of `rows`' imaging the frozen crop tree actually covers."""
    per_status: dict[str, int] = {}
    unbound_figs: dict[str, None] = {}
    cases_full = cases_partial = cases_imaged = 0
    for row in rows:
        n = bad = 0
        for im, pid in imaged_items(row, cfg):
            aid, _, status = resolve_panel_asset(cfg, im["image_id"], pid)
            per_status[status] = per_status.get(status, 0) + 1
            n += 1
            if status not in SERVED_PANEL_STATUSES:
                bad += 1
                if len(unbound_figs) < 20:
                    unbound_figs[im["image_id"]] = None
        if n:
            cases_imaged += 1
            cases_full += not bad
            cases_partial += bool(bad)
    served = sum(per_status.get(k, 0) for k in SERVED_PANEL_STATUSES)
    total = sum(per_status.values())
    return {
        "policy": cfg.panel_policy,
        "donor_list": str(cfg.donor_list) if cfg.donor_list else None,
        "manifest": str(cfg.panel_assets) if cfg.panel_assets else None,
        "crops_dir": str(cfg.panel_assets_dir) if cfg.panel_assets else None,
        "imaged_records": total,
        "by_status": per_status,
        "crop_rate": round(served / total, 4) if total else None,
        "cases_imaged": cases_imaged,
        "cases_fully_cropped": cases_full,
        "cases_partially_cropped": cases_partial,
        "uncovered_figure_examples": list(unbound_figs),
    }


def panel_preflight(rows: list[dict], cfg: Config) -> dict:
    """Report crop coverage before the first episode; enforce it under strict."""
    cov = panel_coverage(rows, cfg)
    n = cov["imaged_records"]
    served = sum(cov["by_status"].get(k, 0) for k in SERVED_PANEL_STATUSES)
    LOG.info(
        "panel serving: policy=%s  crops=%d/%d imaged records (%.1f%%)  "
        "cases fully cropped %d/%d  manifest=%s",
        cfg.panel_policy,
        served,
        n,
        100 * (cov["crop_rate"] or 0),
        cov["cases_fully_cropped"],
        cov["cases_imaged"],
        cfg.panel_assets,
    )
    gaps = {k: v for k, v in cov["by_status"].items() if k not in SERVED_PANEL_STATUSES}
    if gaps and cfg.panel_policy == "crops_only":
        LOG.warning(
            "panel serving: %d imaged record(s) have NO crop %s — "
            "their study is returned as WRITTEN REPORT ONLY; no "
            "uncropped figure is served. Examples: %s",
            n - served,
            gaps,
            ", ".join(cov["uncovered_figure_examples"][:5]),
        )
    if gaps and cfg.panel_policy == "strict":
        raise SystemExit(
            f"--panel-policy strict: {n - served} of {n} imaged records have "
            f"no crop in {cfg.panel_assets_dir} {gaps}. Examples: "
            + ", ".join(cov["uncovered_figure_examples"][:5])
            + ". Deliver the missing crops (or run --panel-policy crops_only, "
            "which withholds those pictures and serves the written report)."
        )
    return cov


def _b64_image_part(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    data = base64.b64encode(p.read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}


class LLMAgent:
    def __init__(
        self,
        cfg: Config,
        name: str,
        model: str,
        base_url: str | None,
        temperature: float,
        usage_log: Callable[[dict], None],
        system: str | None = None,
        json_mode: bool = False,
        transport: Callable[[dict], Any] | None = None,
    ):
        self.cfg, self.name, self.model = cfg, name, model
        self.temperature, self.usage_log = temperature, usage_log
        self.system, self.json_mode = system, json_mode
        self.transport = transport
        import os

        self.client = OpenAI(
            base_url=base_url or None,
            timeout=cfg.timeout,
            max_retries=0,
            api_key=(os.environ.get("DOCTOR_API_KEY", "EMPTY") if base_url else None),
        )

    def _create(self, kwargs: dict) -> Any:
        return self.client.chat.completions.create(**kwargs)

    def complete(
        self, messages: list[dict], pmcid: str, phase: str, step: int | None = None
    ) -> str:
        """step: the env step this call belongs to (turn number). Every retry
        inside one step repeats it, so viewers/exporters pair calls with the
        turn they produced instead of guessing from call order; patient calls
        leave it None.
        """
        kwargs: dict[str, Any] = dict(
            model=self.model, messages=messages, max_completion_tokens=self.cfg.max_output_tokens
        )
        if not re.match(r"gpt-5|o\d", self.model):
            kwargs["temperature"] = self.temperature
        if self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self.cfg.service_tier:
            kwargs["service_tier"] = self.cfg.service_tier
        last: Exception | None = None
        for attempt in range(self.cfg.max_attempts):
            try:
                rsp = (self.transport or self._create)(kwargs)
                u = rsp.usage
                cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
                pin, pcache, pout = PRICE.get(self.model, (0.0, 0.0, 0.0))
                self.usage_log(
                    {
                        "ts": round(time.time(), 1),
                        "pmcid": pmcid,
                        "phase": phase,
                        "step": step,
                        "agent": self.name,
                        "model": self.model,
                        "input_tokens": u.prompt_tokens,
                        "cached_tokens": cached,
                        "output_tokens": u.completion_tokens,
                        "cost_usd": round(
                            (
                                (u.prompt_tokens - cached) * pin
                                + cached * pcache
                                + u.completion_tokens * pout
                            )
                            / 1e6,
                            6,
                        ),
                    }
                )
                return rsp.choices[0].message.content or ""
            except Exception as e:
                last = e
                LOG.warning(
                    "%s %s attempt %d/%d failed: %s",
                    pmcid,
                    self.name,
                    attempt + 1,
                    self.cfg.max_attempts,
                    e,
                )
                time.sleep(min(60.0, 2.0**attempt + random.random()))
        raise RuntimeError(f"{self.name} failed after {self.cfg.max_attempts} attempts: {last}")


THINK_RE = re.compile(
    r"<(?:think|thinking|reasoning)>(.*?)</(?:think|thinking|reasoning)>", re.S | re.I
)


def _canon(s: str) -> str:
    s = re.sub(r"[^a-z0-9/]+", " ", s.lower()).strip()
    return re.sub(r"\s*/\s*", "/", s)


def _subset_hits(item: str, vocab: tuple[str, ...]) -> list[str]:
    toks = set(re.split(r"[ /]+", _canon(item)))
    if not toks or toks <= {"examination", "testing", "other"}:
        return []
    return [v for v in vocab if toks <= set(re.split(r"[ /]+", _canon(v)))]


def resolve_item(item: str, vocab: tuple[str, ...]) -> str | None:
    """Exact (case/punct-insensitive) match, else unique token-subset match —
    tolerates 'Lymphatic Examination' for the long comma-joined menu names.
    """
    ci = _canon(item)
    for v in vocab:
        if _canon(v) == ci:
            return v
    hits = _subset_hits(item, vocab)
    return hits[0] if len(hits) == 1 else None


EMPTY_LIST_ERR = "the action needs a comma-separated category list"


def split_items(payload: str, vocab: tuple[str, ...]) -> tuple[list[str] | None, str | None]:
    """Comma-split, then greedily re-merge fragments so category names that
    themselves contain commas ('Oral, Oropharyngeal, and Dental Examination')
    survive; resolve every piece against the vocabulary.
    """
    frags = [x.strip() for x in payload.split(",") if x.strip()]
    if not frags:
        return None, EMPTY_LIST_ERR
    items: list[str] = []
    i = 0
    while i < len(frags):
        merged = None
        for j in range(min(len(frags), i + 4), i, -1):
            cand = resolve_item(", ".join(frags[i:j]), vocab)
            if cand:
                merged, i = cand, j
                break
        if merged is None:
            hits = _subset_hits(frags[i], vocab)
            if len(hits) > 1:
                return None, (
                    f"ambiguous category '{frags[i]}' — it matches "
                    f"{' and '.join(hits)}; write the full L1/L2 pair"
                )
            return None, (f"unknown category '{frags[i]}'; use exact names from the action menu")
        if merged not in items:
            items.append(merged)
    return items, None


ACTION_RE = re.compile(r"<action>(.*?)</action>", re.S | re.I)
LIST_VERBS = {"physical_examination", "order_tests"}

BARE_ABSTENTION_RE = re.compile(
    r"^(?:" + "|".join(re.escape(p) for p in DEFER_PREFIXES) + r")(?!\w)", re.I
)


OPEN_THINK_RE = re.compile(r"<(?:think|thinking|reasoning)>(.*?)(?=<action>|\Z)", re.S | re.I)

INVALID_PENALTY = -0.01


_FORMAT_ERR_HEADS = (
    "missing <reasoning>",
    "missing <action>",
    "unknown action ''",
    "unknown specialist ''",
    "the ask slot was echoed",
    "give 1-",
    "template echo",
)
_FORMAT_ERR_TAILS = ("needs a payload after the colon",)


def forfeit_class(err: str | None) -> str:
    """'format' for a refusal that stays a forfeit under every rule_violation
    mode, 'rule' for one the lenient env turns into a no-op turn.
    """
    e = err or ""
    if e.startswith(_FORMAT_ERR_HEADS) or e.endswith(_FORMAT_ERR_TAILS):
        return "format"
    if e in (
        DEFER_SLOT_ERR,
        DEFER_SLOT_ERR,
        TERMINAL_BRACKET_ERR,
        NO_DX_ERR_TAG,
        SINGLE_DX_ERR,
        SINGLE_DX_ERR_NO_DEFER,
        EMPTY_LIST_ERR,
    ):
        return "format"
    return "rule"


_TERSE_RULES: tuple[tuple[str, str], ...] = (
    ("nothing to execute", "nothing to do"),
    ("already ordered", "already done"),
    ("already examined", "already done"),
    ("at most", "over the per-action cap"),
    ("unknown specialist", "unknown specialist"),
    ("unknown action", "unknown action"),
    ("unknown category", "not on the menu"),
    ("unknown test pairs", "not on the menu"),
    ("unknown examination categories", "not on the menu"),
    ("ambiguous category", "ambiguous item"),
    ("the search budget", "search budget spent"),
    ("identical query", "already searched"),
    ("search is not available", "search unavailable"),
    ("no questions remain", "no questions left"),
    ("no workup actions remain", "no workup actions left"),
    ("the interview is over", "not available in this phase"),
    ("the workup has already begun", "not available in this phase"),
    ("workup actions are unavailable during the interview", "not available in this phase"),
    ("abstention belongs to the workup", "not available in this phase"),
    ("final_diagnosis belongs to the workup", "not available in this phase"),
    ("complete at least", "not allowed yet"),
    ("a diagnostic workup is required", "not allowed yet"),
    ("the history is too thin", "not allowed yet"),
    ("this substantially repeats", "repeated question"),
)
_NO_STUDY_RE = re.compile(r"^'(?P<dept>[^']+)' has no studies awaiting")
_NOT_IN_RUN_RE = re.compile(r"^(?P<verb>\S+) is not available in this run")


def terse_reason(reason: str | None) -> str:
    """The fact behind a validator refusal, one clause, nothing prescriptive."""
    r = (reason or "").strip()
    if not r:
        return "the action could not be executed"
    m_ = _NO_STUDY_RE.match(r)
    if m_:
        return f"no study is waiting for {m_.group('dept')}"
    m_ = _NOT_IN_RUN_RE.match(r)
    if m_:
        return f"{m_.group('verb')} is not available in this run"
    for head, terse in _TERSE_RULES:
        if r.startswith(head):
            return terse
    return (
        re.split(r"[;—]| - ", r, maxsplit=1)[0].strip()[:80] or "the action could not be executed"
    )


def noop_observation(reason: str | None, mode: str = "terse") -> str:
    """The observation a rule-refused action earns under rule_violation=noop:
    the turn is spent, nothing is charged. terse = the fact only; full = the validator's own words.
    """
    if mode == "full":
        r = (reason or "the action could not be executed").strip()
        return f"No action taken ({r[:300]})."
    return f"No action taken: {terse_reason(reason)}."


def forfeit_observation(reason: str | None) -> str:
    """The observation an invalid action earns: the validator's own words, so a
    zero-retry (rl_mode) rollout still teaches the format from the trajectory.
    """
    r = (reason or "unparseable output").strip()
    return f"Invalid action ({r[:300]}); the turn was forfeited."


ABSTENTION_HEAD_RE = re.compile(
    r"^(?:(?P<ook>out[ _\-]*of[ _\-]*knowledge)|(?P<mismatch>mismatch))(?!\w)", re.I
)
DASH_SEPARATOR_RE = re.compile(r"^(?P<head>[^\s:]+)\s+[—–-]\s+(?P<rest>.+)$")
GLUED_SEPARATOR_RE = re.compile(r"^(?P<head>[a-z_]+)[/=<(\[](?P<rest>.+?)[>)\]]?$")
NO_COLON_RE = re.compile(r"^(?P<head>[A-Za-z_\-]+)\s+(?P<rest>.+)$")
_QUOTE_PAIR_RE = re.compile(r"^([\"'`])(.*)\1$", re.S)
_VERB_TRAIL_PUNCT = ".,;!"
L1_NAMES = ("Imaging", "Laboratory", "Pathology", "Physiologic")
_L1_BY_CANON = {l1.lower(): l1 for l1 in L1_NAMES}
VERB_ALIASES = {
    "physical_exam": "physical_examination",
    "physical_ex_examination": "physical_examination",
    "exam": "physical_examination",
    "examination": "physical_examination",
    "examine": "physical_examination",
    "pe": "physical_examination",
    "question": "ask",
    "question_bundle": "ask",
    "next_question": "ask",
    "next_question_bundle": "ask",
    "ask_question": "ask",
    "ask_questions": "ask",
    "ask_one": "ask",
    "ask_one_bundle": "ask",
    "ask_followup": "ask",
    "ask_patient": "ask",
    "inquire": "ask",
    "consult": "consult_specialist",
    "consultation": "consult_specialist",
    "refer": "consult_specialist",
    "order": "order_tests",
    "order_test": "order_tests",
    "tests": "order_tests",
    "test": "order_tests",
    "order_labs": "order_tests",
    "labs": "order_tests",
    "diagnosis": "final_diagnosis",
    "diagnose": "final_diagnosis",
    "final": "final_diagnosis",
    "finalize_diagnosis": "final_diagnosis",
    "finalise_diagnosis": "final_diagnosis",
    "final_diagnose": "final_diagnosis",
    "final_diagnoses": "final_diagnosis",
    "finish_diagnosis": "final_diagnosis",
    "finalize": "final_diagnosis",
    "finalise": "final_diagnosis",
    "final_dx": "final_diagnosis",
    "dx": "final_diagnosis",
    "end_interview": "begin_workup",
    "start_workup": "begin_workup",
    "begin_work_up": "begin_workup",
    "workup": "begin_workup",
    "start_work_up": "begin_workup",
    "abstain": ABSTENTION_VERB,
    "abstention_out_of_knowledge": ABSTENTION_VERB,
    "defer_to_human": ABSTENTION_VERB,
    "defer": ABSTENTION_VERB,
    "lookup": "search",
    "search_literature": "search",
    "pubmed": "search",
}
_ASK_PREFIX_RE = re.compile(r"^(?:ask|question)(?:_[a-z0-9_]+)?$")
_CONSULT_PREFIX_RE = re.compile(r"^consult(?:_specialist)?_(?P<dept>[a-z_]+)$")
FUZZY_CUTOFF = 0.85
DEPT_ALIASES = {
    "radiologist": "radiology",
    "imaging": "radiology",
    "radiologic": "radiology",
    "radiological": "radiology",
    "pathologist": "pathology",
    "path": "pathology",
    "histopathology": "pathology",
    "surgical_pathology": "pathology",
    "cytopathology": "pathology",
    "hematopathology": "pathology",
    "cardiologist": "cardiology",
    "cards": "cardiology",
    "cardiac": "cardiology",
    "neurologist": "neurology",
    "neuro": "neurology",
    "neurologic": "neurology",
    "dermatologist": "dermatology",
    "derm": "dermatology",
    "dermatologic": "dermatology",
    "skin": "dermatology",
    "ophthalmologist": "ophthalmology",
    "ophtho": "ophthalmology",
    "eye": "ophthalmology",
    "ophthalmic": "ophthalmology",
    "gastroenterologist": "gastroenterology",
    "gi": "gastroenterology",
    "lab": "laboratory_medicine",
    "laboratory": "laboratory_medicine",
    "lab_medicine": "laboratory_medicine",
    "clinical_laboratory": "laboratory_medicine",
    "clinical_pathology": "laboratory_medicine",
    "hematology": "laboratory_medicine",
    "microbiology": "laboratory_medicine",
    "genetics": "laboratory_medicine",
}


def known_heads() -> tuple[str, ...]:
    """The verbs a policy may type at `contract`: VERBS, with the abstention
    under its SURFACE spelling from contract 3 (`abstention`).
    """
    return tuple(v for v in VERBS if v != DEFER_VERB) + (ABSTENTION_VERB,)


def _fuzzy(
    word: str, candidates: "list[str] | tuple[str, ...]", cutoff: float = FUZZY_CUTOFF
) -> str | None:
    """The unique close match of `word` among `candidates` (difflib ratio >=
    cutoff), None when none or several. Short words never fuzz: a 4-letter
    typo is indistinguishable from a different word.
    """
    if len(word) < 5:
        return None
    hits = difflib.get_close_matches(word, list(candidates), n=2, cutoff=cutoff)
    return hits[0] if len(hits) == 1 else None


@dataclass(frozen=True, slots=True)
class ParsedAction:
    """parse_doctor_ex's result. `normalized` is the format-validity ledger: the
    spelling / vocabulary rules that fired, in order, () when the action was
    typed exactly as the menu prints it. `dropped` lists the list items no rule
    could resolve (contract 3): what Env.validate refuses or, under the
    lenient env, drops from the executed list. Never read by a reward path.
    """

    think: str
    verb: str
    payload: Any
    error: str | None
    normalized: tuple[str, ...] = ()
    dropped: tuple[str, ...] = ()


def canonical_abstention_payload(payload: str) -> tuple[str, str | None]:
    """(payload with its head in the canonical spelling, tag). `Out of Knowledge
    — x` -> `out_of_knowledge — x`, every byte after the head preserved (the
    axis words a mismatch claim is graded on stay as typed). Tag None when the
    head is already canonical, `abstention_noprefix` when there is no head at
    all (measured, never rewritten: classify_outcome reads such a reason as the
    honest abstention, and changing that is a contract decision).
    """
    hm = ABSTENTION_HEAD_RE.match(payload)
    if not hm:
        return payload, "abstention_noprefix"
    canon = "out_of_knowledge" if hm.group("ook") else "mismatch"
    if payload[: hm.end()] == canon:
        return payload, None
    return canon + payload[hm.end() :], "abstention_spelling"


_ITEM_STOP = frozenset(
    {
        "exam",
        "examination",
        "examinations",
        "testing",
        "test",
        "tests",
        "assessment",
        "evaluation",
        "check",
        "of",
        "the",
        "a",
        "an",
        "focused",
        "full",
        "complete",
        "detailed",
        "thorough",
        "targeted",
        "bilateral",
    }
)
PE_ALIASES: dict[str, tuple[str, ...]] = {
    "chest": ("respiratory", "cardiovascular"),
    "cardiopulmonary": ("respiratory", "cardiovascular"),
    "cardiorespiratory": ("respiratory", "cardiovascular"),
    "heart and lungs": ("respiratory", "cardiovascular"),
    "chest wall": ("respiratory",),
    "thorax": ("respiratory",),
    "thoracic": ("respiratory",),
    "lungs": ("respiratory",),
    "lung": ("respiratory",),
    "pulmonary": ("respiratory",),
    "breathing": ("respiratory",),
    "respiration": ("respiratory",),
    "airway": ("respiratory",),
    "trunk": ("respiratory",),
    "heart": ("cardiovascular",),
    "cardiac": ("cardiovascular",),
    "vascular": ("cardiovascular",),
    "peripheral vascular": ("cardiovascular",),
    "cv": ("cardiovascular",),
    "pulses": ("cardiovascular",),
    "face": ("head",),
    "facial": ("head",),
    "cranial": ("head",),
    "skull": ("head",),
    "hair": ("skin",),
    "scalp": ("skin",),
    "hair and scalp": ("skin",),
    "scalp and hair": ("skin",),
    "nails": ("skin",),
    "nail": ("skin",),
    "derm": ("skin",),
    "dermatologic": ("skin",),
    "dermatological": ("skin",),
    "wound": ("skin",),
    "rash": ("skin",),
    "lesions": ("skin",),
    "lesion": ("skin",),
    "integumentary": ("skin",),
    "eyelids": ("eyes",),
    "eyelid": ("eyes",),
    "lid": ("eyes",),
    "lids": ("eyes",),
    "vision": ("eyes",),
    "visual": ("eyes",),
    "visual acuity": ("eyes",),
    "visual fields": ("eyes",),
    "ocular": ("eyes",),
    "ophthalmic": ("eyes",),
    "ophthalmologic": ("eyes",),
    "ophthalmology": ("eyes",),
    "fundus": ("eyes",),
    "pupils": ("eyes",),
    "eye": ("eyes",),
    "conjunctiva": ("eyes",),
    "retina": ("eyes",),
    "ear": ("ears",),
    "hearing": ("ears",),
    "auditory": ("ears",),
    "vestibular": ("ears",),
    "auricle": ("ears",),
    "otoscopy": ("ears",),
    "otologic": ("ears",),
    "nasal": ("nose",),
    "sinuses": ("nose",),
    "sinus": ("nose",),
    "vitals": ("vital signs",),
    "vitals signs": ("vital signs",),
    "vital sign": ("vital signs",),
    "vital": ("vital signs",),
    "vitalsigns": ("vital signs",),
    "anthropometrics": ("vital signs",),
    "anthropometric": ("vital signs",),
    "growth parameters": ("vital signs",),
    "temperature": ("vital signs",),
    "blood pressure": ("vital signs",),
    "breasts": ("breast",),
    "axilla": ("breast",),
    "lips": ("mouth and throat",),
    "teeth": ("mouth and throat",),
    "oral cavity": ("mouth and throat",),
    "oral": ("mouth and throat",),
    "tongue": ("mouth and throat",),
    "gums": ("mouth and throat",),
    "pharynx": ("mouth and throat",),
    "throat": ("mouth and throat",),
    "mouth": ("mouth and throat",),
    "dental": ("mouth and throat",),
    "dentition": ("mouth and throat",),
    "mucosal": ("mouth and throat",),
    "mucous membranes": ("mouth and throat",),
    "oropharynx": ("mouth and throat",),
    "oral and throat": ("mouth and throat",),
    "oral mucosa": ("mouth and throat",),
    "tonsils": ("mouth and throat",),
    "thyroid": ("neck",),
    "cervical": ("neck",),
    "cranial nerves": ("neurologic",),
    "reflexes": ("neurologic",),
    "gait": ("neurologic",),
    "motor": ("neurologic",),
    "sensory": ("neurologic",),
    "neuro": ("neurologic",),
    "neurological": ("neurologic",),
    "neurology": ("neurologic",),
    "nervous system": ("neurologic",),
    "brain": ("neurologic",),
    "neuromuscular": ("neurologic",),
    "speech": ("neurologic",),
    "coordination": ("neurologic",),
    "cerebellar": ("neurologic",),
    "strength": ("neurologic",),
    "mentation": ("mental status",),
    "mental": ("mental status",),
    "cognition": ("mental status",),
    "cognitive": ("mental status",),
    "psychiatric": ("mental status",),
    "psych": ("mental status",),
    "mood": ("mental status",),
    "orientation": ("mental status",),
    "feet": ("extremities",),
    "foot": ("extremities",),
    "hands": ("extremities",),
    "hand": ("extremities",),
    "limbs": ("extremities",),
    "limb": ("extremities",),
    "arms": ("extremities",),
    "arm": ("extremities",),
    "legs": ("extremities",),
    "leg": ("extremities",),
    "lower extremities": ("extremities",),
    "upper extremities": ("extremities",),
    "lower extremity": ("extremities",),
    "upper extremity": ("extremities",),
    "extremity": ("extremities",),
    "elbow": ("extremities",),
    "knee": ("extremities",),
    "ankle": ("extremities",),
    "wrist": ("extremities",),
    "shoulder": ("extremities",),
    "hip": ("extremities",),
    "fingers": ("extremities",),
    "toes": ("extremities",),
    "peripheral": ("extremities",),
    "edema": ("extremities",),
    "joints": ("musculoskeletal",),
    "joint": ("musculoskeletal",),
    "spine": ("musculoskeletal",),
    "back": ("musculoskeletal",),
    "msk": ("musculoskeletal",),
    "muscle": ("musculoskeletal",),
    "muscles": ("musculoskeletal",),
    "orthopedic": ("musculoskeletal",),
    "skeletal": ("musculoskeletal",),
    "bones": ("musculoskeletal",),
    "bone": ("musculoskeletal",),
    "cervical spine": ("musculoskeletal",),
    "lumbar spine": ("musculoskeletal",),
    "posture": ("musculoskeletal",),
    "pelvis": ("genitourinary",),
    "pelvic": ("genitourinary",),
    "scrotum": ("genitourinary",),
    "scrotal": ("genitourinary",),
    "genitalia": ("genitourinary",),
    "genital": ("genitourinary",),
    "genitals": ("genitourinary",),
    "testicles": ("genitourinary",),
    "testes": ("genitourinary",),
    "testis": ("genitourinary",),
    "gu": ("genitourinary",),
    "groin": ("genitourinary",),
    "inguinal": ("genitourinary",),
    "urinary": ("genitourinary",),
    "bladder": ("genitourinary",),
    "gynecologic": ("genitourinary",),
    "gynecological": ("genitourinary",),
    "vagina": ("genitourinary",),
    "vaginal": ("genitourinary",),
    "uterus": ("genitourinary",),
    "penis": ("genitourinary",),
    "urogenital": ("genitourinary",),
    "prostate": ("rectal",),
    "perineum": ("rectal",),
    "perineal": ("rectal",),
    "perianal": ("rectal",),
    "anus": ("rectal",),
    "anal": ("rectal",),
    "anorectal": ("rectal",),
    "rectum": ("rectal",),
    "belly": ("abdomen",),
    "abdominal": ("abdomen",),
    "liver": ("abdomen",),
    "spleen": ("abdomen",),
    "liver and spleen": ("abdomen",),
    "hepatosplenomegaly": ("abdomen",),
    "gastrointestinal": ("abdomen",),
    "gi": ("abdomen",),
    "bowel": ("abdomen",),
    "organomegaly": ("abdomen",),
    "lymph": ("lymph nodes",),
    "lymphatic": ("lymph nodes",),
    "nodes": ("lymph nodes",),
    "nodal": ("lymph nodes",),
    "lymphadenopathy": ("lymph nodes",),
    "regional lymph nodes": ("lymph nodes",),
    "cervical lymph nodes": ("lymph nodes",),
    "inguinal lymph nodes": ("lymph nodes",),
    "inguinal nodes": ("lymph nodes",),
    "axillary lymph nodes": ("lymph nodes",),
    "hematologic": ("lymph nodes",),
    "immunologic": ("lymph nodes",),
    "general appearance": ("general",),
    "appearance": ("general",),
    "overall": ("general",),
    "constitutional": ("general",),
    "endocrine": ("other",),
    "systemic": ("other",),
}
L2_SYNONYMS: dict[str, str] = {
    "ct": "computed tomography",
    "ct scan": "computed tomography",
    "cat scan": "computed tomography",
    "ct angiography": "computed tomography",
    "computed tomography angiography": "computed tomography",
    "computed angiography": "computed tomography",
    "cta": "computed tomography",
    "ct angiogram": "computed tomography",
    "spine computed tomography": "computed tomography",
    "computed tomography scan": "computed tomography",
    "mri": "magnetic resonance imaging",
    "mr": "magnetic resonance imaging",
    "mr imaging": "magnetic resonance imaging",
    "magnetic resonance": "magnetic resonance imaging",
    "mra": "magnetic resonance imaging",
    "mr angiography": "magnetic resonance imaging",
    "x ray": "computed radiography",
    "xray": "computed radiography",
    "x rays": "computed radiography",
    "radiograph": "computed radiography",
    "radiographs": "computed radiography",
    "radiography": "computed radiography",
    "plain film": "computed radiography",
    "plain films": "computed radiography",
    "plain radiograph": "computed radiography",
    "chest x ray": "computed radiography",
    "cxr": "computed radiography",
    "chest radiograph": "computed radiography",
    "us": "ultrasound",
    "sonography": "ultrasound",
    "ultrasonography": "ultrasound",
    "doppler": "ultrasound",
    "doppler ultrasound": "ultrasound",
    "echography": "ultrasound",
    "pet": "nuclear medicine",
    "pet ct": "nuclear medicine",
    "pet scan": "nuclear medicine",
    "scintigraphy": "nuclear medicine",
    "bone scan": "nuclear medicine",
    "spect": "nuclear medicine",
    "nuclear": "nuclear medicine",
    "radionuclide": "nuclear medicine",
    "angiography": "x ray angiography",
    "angiogram": "x ray angiography",
    "dsa": "x ray angiography",
    "catheter angiography": "x ray angiography",
    "fundus photography": "ophthalmic photography",
    "retinal photography": "ophthalmic photography",
    "oct": "ophthalmic photography",
    "optical coherence tomography": "ophthalmic photography",
    "fluorescein angiography": "ophthalmic photography",
    "clinical photography": "external camera photography",
    "photograph": "external camera photography",
    "photographs": "external camera photography",
    "photo": "external camera photography",
    "clinical photo": "external camera photography",
    "dermoscopy": "external camera photography",
    "endoscopic": "endoscopy",
    "colonoscopy": "endoscopy",
    "gastroscopy": "endoscopy",
    "bronchoscopy": "endoscopy",
    "laryngoscopy": "endoscopy",
    "cystoscopy": "endoscopy",
    "egd": "endoscopy",
    "mammography": "other imaging modality",
    "mammogram": "other imaging modality",
    "other imaging": "other imaging modality",
    "chemistry": "clinical chemistry",
    "biochemistry": "clinical chemistry",
    "electrolytes": "clinical chemistry",
    "metabolic panel": "clinical chemistry",
    "bmp": "clinical chemistry",
    "cmp": "clinical chemistry",
    "liver function": "clinical chemistry",
    "liver function tests": "clinical chemistry",
    "lfts": "clinical chemistry",
    "renal function": "clinical chemistry",
    "kidney function": "clinical chemistry",
    "blood chemistry": "clinical chemistry",
    "serum chemistry": "clinical chemistry",
    "chemistries": "clinical chemistry",
    "lipid panel": "clinical chemistry",
    "glucose": "clinical chemistry",
    "enzymology": "clinical chemistry",
    "enzymes": "clinical chemistry",
    "cbc": "hematology",
    "complete blood count": "hematology",
    "blood count": "hematology",
    "hematologic": "hematology",
    "haematology": "hematology",
    "hematological": "hematology",
    "blood counts": "hematology",
    "differential count": "hematology",
    "culture": "microbiology",
    "cultures": "microbiology",
    "microbiologic": "microbiology",
    "microbiological": "microbiology",
    "virology": "microbiology",
    "bacteriology": "microbiology",
    "mycology": "microbiology",
    "parasitology": "microbiology",
    "blood culture": "microbiology",
    "blood cultures": "microbiology",
    "infectious": "microbiology",
    "infectious disease testing": "microbiology",
    "serology": "microbiology",
    "serologies": "microbiology",
    "urine": "urinalysis",
    "urine analysis": "urinalysis",
    "ua": "urinalysis",
    "urine studies": "urinalysis",
    "urine test": "urinalysis",
    "urine tests": "urinalysis",
    "immunologic": "immunology",
    "immunological": "immunology",
    "autoantibodies": "immunology",
    "autoimmune": "immunology",
    "autoimmune panel": "immunology",
    "immunoglobulins": "immunology",
    "complement": "immunology",
    "allergy": "immunology",
    "allergy testing": "immunology",
    "endocrine": "endocrinology",
    "hormone": "endocrinology",
    "hormones": "endocrinology",
    "hormonal": "endocrinology",
    "thyroid function": "endocrinology",
    "thyroid function tests": "endocrinology",
    "thyroid": "endocrinology",
    "endocrine panel": "endocrinology",
    "hormonal assay": "endocrinology",
    "general laboratory": "general laboratory testing",
    "other laboratory": "general laboratory testing",
    "other laboratory testing": "general laboratory testing",
    "basic labs": "general laboratory testing",
    "routine labs": "general laboratory testing",
    "routine laboratory": "general laboratory testing",
    "laboratory": "general laboratory testing",
    "labs": "general laboratory testing",
    "laboratory medicine": "general laboratory testing",
    "general labs": "general laboratory testing",
    "baseline labs": "general laboratory testing",
    "inflammatory markers": "general laboratory testing",
    "inflammation": "general laboratory testing",
    "toxicology": "general laboratory testing",
    "toxicology screening": "general laboratory testing",
    "genetics": "molecular genetics",
    "genetic testing": "molecular genetics",
    "genetic": "molecular genetics",
    "dna": "molecular genetics",
    "sequencing": "molecular genetics",
    "molecular": "molecular genetics",
    "whole exome sequencing": "molecular genetics",
    "exome sequencing": "molecular genetics",
    "karyotype": "molecular genetics",
    "karyotyping": "molecular genetics",
    "genetic analysis": "molecular genetics",
    "gene panel": "molecular genetics",
    "genomic": "molecular genetics",
    "cytogenetics": "molecular genetics",
    "fish": "molecular genetics",
    "pcr": "molecular genetics",
    "tumor marker": "tumor markers",
    "tumour markers": "tumor markers",
    "tumour marker": "tumor markers",
    "oncologic markers": "tumor markers",
    "metabolic": "metabolic screening",
    "newborn screening": "metabolic screening",
    "metabolic screen": "metabolic screening",
    "inborn errors": "metabolic screening",
    "metabolic workup": "metabolic screening",
    "coag": "coagulation",
    "coagulation studies": "coagulation",
    "coagulation profile": "coagulation",
    "hemostasis": "coagulation",
    "hemostatic studies": "coagulation",
    "pt inr": "coagulation",
    "clotting": "coagulation",
    "clotting studies": "coagulation",
    "coagulation tests": "coagulation",
    "flow": "flow cytometry",
    "immunophenotyping": "flow cytometry",
    "facs": "flow cytometry",
    "histology": "histopathology",
    "biopsy": "histopathology",
    "histopathologic": "histopathology",
    "histopathological": "histopathology",
    "tissue biopsy": "histopathology",
    "skin biopsy": "histopathology",
    "endomyocardial biopsy": "histopathology",
    "microscopy": "histopathology",
    "light microscopy": "histopathology",
    "histologic": "histopathology",
    "biopsy histopathology": "histopathology",
    "ihc": "immunohistochemistry",
    "immunostaining": "immunohistochemistry",
    "immunostains": "immunohistochemistry",
    "immunohistochemical": "immunohistochemistry",
    "immunohistochemical staining": "immunohistochemistry",
    "gross": "gross examination",
    "macroscopic": "gross examination",
    "macroscopy": "gross examination",
    "cytology": "cytopathology",
    "fna": "cytopathology",
    "fine needle aspiration": "cytopathology",
    "cytologic": "cytopathology",
    "cytological": "cytopathology",
    "csf cytology": "cytopathology",
    "aspiration cytology": "cytopathology",
    "bone marrow": "bone marrow examination",
    "marrow biopsy": "bone marrow examination",
    "bone marrow biopsy": "bone marrow examination",
    "bone marrow aspirate": "bone marrow examination",
    "marrow": "bone marrow examination",
    "hematopathology": "bone marrow examination",
    "em": "electron microscopy",
    "ultrastructural": "electron microscopy",
    "ultrastructure": "electron microscopy",
    "dif": "direct immunofluorescence",
    "immunofluorescence": "direct immunofluorescence",
    "special stains": "special stains and histochemistry",
    "histochemistry": "special stains and histochemistry",
    "stains": "special stains and histochemistry",
    "special staining": "special stains and histochemistry",
    "blood smear": "peripheral blood smear",
    "peripheral smear": "peripheral blood smear",
    "smear": "peripheral blood smear",
    "blood film": "peripheral blood smear",
    "surgical": "surgical pathology",
    "resection": "surgical pathology",
    "excision": "surgical pathology",
    "excisional biopsy": "surgical pathology",
    "surgical specimen": "surgical pathology",
    "ecg": "cardiovascular physiologic testing",
    "ekg": "cardiovascular physiologic testing",
    "electrocardiography": "cardiovascular physiologic testing",
    "electrocardiogram": "cardiovascular physiologic testing",
    "holter": "cardiovascular physiologic testing",
    "cardiac testing": "cardiovascular physiologic testing",
    "stress test": "cardiovascular physiologic testing",
    "cardiovascular testing": "cardiovascular physiologic testing",
    "cardiovascular": "cardiovascular physiologic testing",
    "cardiac": "cardiovascular physiologic testing",
    "cardiology": "cardiovascular physiologic testing",
    "cardiopulmonary": "cardiovascular physiologic testing",
    "cardiopulmonary physiologic testing": "cardiovascular physiologic testing",
    "cardiopulmonary testing": "cardiovascular physiologic testing",
    "hemodynamic monitoring": "cardiovascular physiologic testing",
    "hemodynamic": "cardiovascular physiologic testing",
    "eeg": "neurophysiologic testing",
    "emg": "neurophysiologic testing",
    "ncs": "neurophysiologic testing",
    "nerve conduction": "neurophysiologic testing",
    "nerve conduction studies": "neurophysiologic testing",
    "electroencephalography": "neurophysiologic testing",
    "electroencephalogram": "neurophysiologic testing",
    "electromyography": "neurophysiologic testing",
    "neurophysiology": "neurophysiologic testing",
    "evoked potentials": "neurophysiologic testing",
    "neurologic testing": "neurophysiologic testing",
    "neurologic": "neurophysiologic testing",
    "neurology": "neurophysiologic testing",
    "neurophysiological": "neurophysiologic testing",
    "electrophysiology": "neurophysiologic testing",
    "electrophysiologic": "neurophysiologic testing",
    "echo": "echocardiography",
    "echocardiogram": "echocardiography",
    "echo cardiography": "echocardiography",
    "transthoracic echocardiography": "echocardiography",
    "tte": "echocardiography",
    "tee": "echocardiography",
    "pft": "pulmonary function testing",
    "pfts": "pulmonary function testing",
    "spirometry": "pulmonary function testing",
    "pulmonary function": "pulmonary function testing",
    "lung function": "pulmonary function testing",
    "pulmonary": "pulmonary function testing",
    "respiratory": "pulmonary function testing",
    "ophthalmic": "ophthalmic examination and testing",
    "ophthalmologic": "ophthalmic examination and testing",
    "visual field": "ophthalmic examination and testing",
    "visual fields": "ophthalmic examination and testing",
    "visual acuity": "ophthalmic examination and testing",
    "slit lamp": "ophthalmic examination and testing",
    "fundoscopy": "ophthalmic examination and testing",
    "eye examination": "ophthalmic examination and testing",
    "ophthalmology": "ophthalmic examination and testing",
    "ocular": "ophthalmic examination and testing",
    "ocular physiologic testing": "ophthalmic examination and testing",
    "psychological": "psychological testing",
    "neuropsychological": "psychological testing",
    "neuropsychological testing": "psychological testing",
    "cognitive testing": "psychological testing",
    "psychometric": "psychological testing",
    "psychiatric": "psychological testing",
    "psychiatric evaluation": "psychological testing",
    "other physiologic": "other physiologic testing",
    "physiologic": "other physiologic testing",
    "other physiotesting": "other physiologic testing",
    "audiometry": "other physiologic testing",
    "audiology": "other physiologic testing",
    "hearing test": "other physiologic testing",
    "auditory testing": "other physiologic testing",
    "vestibular testing": "other physiologic testing",
    "urodynamic testing": "other physiologic testing",
    "urodynamics": "other physiologic testing",
    "uroflow": "other physiologic testing",
    "gastrointestinal function testing": "other physiologic testing",
    "manometry": "other physiologic testing",
    "provocative testing": "other physiologic testing",
    "bedside testing": "other physiologic testing",
    "functional assessment": "other physiologic testing",
    "exercise testing": "other physiologic testing",
    "cardiopulmonary exercise testing": "other physiologic testing",
    "sleep study": "other physiologic testing",
    "polysomnography": "other physiologic testing",
    "tilt table": "other physiologic testing",
}
_PE_IN_ORDER_HEAD_RE = re.compile(
    r"^(?:physical[ _\-]*exam(?:ination)?s?|pe|exam(?:ination)?)\s*[:/\-—]\s*", re.I
)
PE_VIRTUAL_L1 = "Physical Examination"


def _item_words(item: str) -> str:
    return " ".join(t for t in re.split(r"[ /]+", _canon(item)) if t and t not in _ITEM_STOP)


def resolve_pe_item(item: str) -> tuple[list[str], str | None]:
    """A physical-examination item -> its order name(s) and the rule that
    resolved it (None = the menu spelling / the existing loose resolver).
    [] with a tag of None means: not resolvable.
    """
    hit = resolve_item(item, PE_VOCAB)
    if hit:
        return [PE_SYNONYM.get(hit, hit)], None
    words = _item_words(item)
    if not words:
        return [], None
    if words in PE_ALIASES:
        return list(PE_ALIASES[words]), "pe_alias"
    hits = {PE_SYNONYM.get(v, v) for v in _subset_hits(words, PE_VOCAB)}
    if len(hits) == 1:
        return [hits.pop()], "item_stopword"
    if " and " in _canon(item):
        parts = [p for p in re.split(r"\s+and\s+", _canon(item)) if p]
        if len(parts) >= 2:
            out: list[str] = []
            for p in parts:
                names, _ = resolve_pe_item(p)
                if not names:
                    out = []
                    break
                out.extend(n for n in names if n not in out)
            if out:
                return out, "item_split"
    cand = _fuzzy(words, list(PE_ORDER_NAMES) + list(PE_ALIASES))
    if cand:
        names = list(PE_ALIASES.get(cand, (cand,)))
        return names, "item_fuzzy"
    return [], None


def _l2_index() -> dict[str, list[str]]:
    idx: dict[str, list[str]] = {}
    for p in TEST_PAIRS_MENU:
        idx.setdefault(_canon(p).split("/")[-1], []).append(p)
    return idx


def resolve_test_item(item: str) -> tuple[list[str], str | None]:
    """An order_tests item -> menu pair(s) (or a PE_VIRTUAL_L1 pair for a physical
    examination typed inside an order) and the rule that resolved it.
    """
    pairs = TEST_PAIRS_MENU
    hit = resolve_item(item, pairs)
    if hit:
        return [hit], None
    ci = _canon(item)
    l1c, _, l2c = ci.rpartition("/")
    l1c, l2c = l1c.strip(), l2c.strip()
    stripped = _PE_IN_ORDER_HEAD_RE.sub("", item)
    if stripped != item or (
        l1c and l1c not in _L1_BY_CANON and ("physical" in l1c or "exam" in l1c)
    ):
        names, _ = resolve_pe_item(stripped if stripped != item else item.rpartition("/")[-1])
        if names:
            return [f"{PE_VIRTUAL_L1}/{n}" for n in names], "pe_in_order"
    if not l1c:
        names, _ = resolve_pe_item(item)
        if names and _item_words(item) in PE_ORDER_NAMES:
            return [f"{PE_VIRTUAL_L1}/{n}" for n in names], "pe_in_order"
    idx = _l2_index()
    l1 = _L1_BY_CANON.get(l1c)
    words = _item_words(l2c)
    target = L2_SYNONYMS.get(words) or L2_SYNONYMS.get(l2c)
    if target and target in idx:
        cands = idx[target]
        if l1:
            same = [p for p in cands if p.startswith(l1 + "/")]
            if same:
                return same, "item_l2_synonym"
        if len(cands) == 1:
            return cands, "item_l2_synonym"
    key = l2c if l2c in idx else (words if words in idx else None)
    if key:
        cands = idx[key]
        if len(cands) == 1:
            return cands, "item_l1_corrected"
        if not l1:
            return list(cands), "item_l2_both"
    if l1c:
        names, _ = resolve_pe_item(l2c)
        if names and _item_words(l2c) in PE_ORDER_NAMES:
            return [f"{PE_VIRTUAL_L1}/{n}" for n in names], "pe_in_order"
    cand = _fuzzy(words or l2c, list(idx))
    if cand:
        cands = idx[cand]
        if l1:
            same = [p for p in cands if p.startswith(l1 + "/")]
            if same:
                return same, "item_fuzzy"
        if len(cands) == 1:
            return cands, "item_fuzzy"
    return [], None


def split_items_ex(payload: str, verb: str) -> tuple[list[str], tuple[str, ...], tuple[str, ...]]:
    """Contract 3 list parsing: (items, tags, dropped). Comma-split with the
    same longest-first re-merge as split_items, then every fragment through the
    lenient resolver; fragments no rule resolves land in `dropped` (Env.validate
    refuses them, or the lenient env drops them and executes the rest).
    """
    frags = [x.strip() for x in payload.split(",") if x.strip()]
    items: list[str] = []
    tags: list[str] = []
    dropped: list[str] = []
    resolve = (
        (lambda it: resolve_pe_item(it))
        if verb == "physical_examination"
        else (lambda it: resolve_test_item(it))
    )
    vocab = PE_VOCAB if verb == "physical_examination" else TEST_PAIRS_MENU
    i = 0
    while i < len(frags):
        merged = None
        for j in range(min(len(frags), i + 4), i + 1, -1):
            hit = resolve_item(", ".join(frags[i:j]), vocab)
            if hit:
                name = PE_SYNONYM.get(hit, hit) if verb == "physical_examination" else hit
                merged, i = ([name], None), j
                break
        if merged is None:
            names, tag = resolve(frags[i])
            if names:
                merged, i = (names, tag), i + 1
        if merged is None:
            dropped.append(frags[i])
            i += 1
            continue
        names, tag = merged
        for n in names:
            if n not in items:
                items.append(n)
        if tag and tag not in tags:
            tags.append(tag)
    return items, tuple(tags), tuple(dropped)


def parse_doctor_ex(text: str) -> ParsedAction:
    """-> ParsedAction(think, verb, payload, error, normalized, dropped). payload:
    list for list verbs, str else.
    """
    tm, am = THINK_RE.search(text), ACTION_RE.search(text)
    if not tm:
        tm = OPEN_THINK_RE.search(text)
    if not tm or not tm.group(1).strip():
        return ParsedAction(
            "",
            "",
            None,
            ("missing <reasoning> </reasoning> block; it is mandatory before the action"),
        )
    if not am:
        return ParsedAction(tm.group(1).strip(), "", None, "missing <action> </action> block")
    think = tm.group(1).strip()
    body = re.sub(r"\s+", " ", am.group(1)).strip()
    verb_raw, _, payload = body.partition(":")
    verb_raw = verb_raw.strip()
    verb = verb_raw.lower().replace(" ", "_")
    payload = payload.strip()
    heads = known_heads()
    settle = heads + (DEFER_VERB,)
    tags: list[str] = []
    if verb not in settle:
        cand = verb.replace("-", "_")
        if not payload:
            cand = cand.rstrip(_VERB_TRAIL_PUNCT)
        if cand != verb and cand in settle:
            verb = cand
    if verb in settle and verb != verb_raw:
        tags.append("verb_spelling")
    if verb not in settle:
        stripped = verb.strip("<>/").rstrip("_/").strip("<>/_")
        if stripped in settle:
            verb = stripped
            tags.append("xml_tag")
    if verb not in settle and ABSTENTION_HEAD_RE.match(body):
        verb, payload = ABSTENTION_VERB, body
        tags.append("bare_abstention")
    if verb not in settle:
        dm = DASH_SEPARATOR_RE.match(body)
        if dm:
            head = dm.group("head").lower().replace("-", "_")
            head = VERB_ALIASES.get(head, head)
            if head in settle:
                verb, payload = head, dm.group("rest").strip()
                tags.append("dash_separator")
    if verb not in settle:
        alias = VERB_ALIASES.get(verb.replace("-", "_").rstrip(_VERB_TRAIL_PUNCT))
        if verb == "abstention_mismatch":
            verb, payload = ABSTENTION_VERB, ("mismatch — " + payload if payload else "mismatch")
            tags.append("verb_alias")
        elif alias:
            verb = alias
            tags.append("verb_alias")
        elif _ASK_PREFIX_RE.match(verb) and payload:
            verb = "ask"
            tags.append("verb_alias")
        elif _CONSULT_PREFIX_RE.match(verb):
            dept = _CONSULT_PREFIX_RE.match(verb).group("dept")
            verb, payload = "consult_specialist", (payload or dept)
            tags.append("verb_alias")
        elif verb in _L1_BY_CANON or (_fuzzy(verb, list(_L1_BY_CANON)) in _L1_BY_CANON):
            l1 = _L1_BY_CANON.get(verb) or _L1_BY_CANON[_fuzzy(verb, list(_L1_BY_CANON))]
            items = [x.strip() for x in payload.split(",") if x.strip()]
            payload = ", ".join(x if "/" in x else f"{l1}/{x}" for x in items)
            verb = "order_tests"
            tags.append("l1_as_verb")
        else:
            gm = GLUED_SEPARATOR_RE.match(verb)
            hm = gm.group("head") if gm else None
            if hm is not None:
                hm = VERB_ALIASES.get(hm, hm)
                rest = gm.group("rest").replace("_", " ").strip(" >)]")
                if payload:
                    rest = f"{rest} {payload}".strip()
                if hm in settle:
                    verb, payload = hm, rest
                    tags.append("verb_glued_separator")
                elif hm in _L1_BY_CANON:
                    verb, payload = "order_tests", f"{_L1_BY_CANON[hm]}/{rest}"
                    tags.append("l1_as_verb")
            if verb not in settle and ":" not in body:
                nm = NO_COLON_RE.match(body)
                if nm:
                    head = nm.group("head").lower().replace("-", "_")
                    head = VERB_ALIASES.get(head, head)
                    if head in settle:
                        verb, payload = head, nm.group("rest").strip()
                        tags.append("verb_no_colon")
            if verb not in settle:
                fz = _fuzzy(verb, list(settle) + list(VERB_ALIASES))
                if fz:
                    verb = VERB_ALIASES.get(fz, fz)
                    tags.append("verb_fuzzy")
    if verb == DEFER_VERB and "verb_alias" not in tags:
        tags.append("verb_alias")
    if verb == ABSTENTION_VERB:
        verb = DEFER_VERB
    if verb not in VERBS:
        admissible = [surface_defer_verb() if v == DEFER_VERB else v for v in VERBS]
        return ParsedAction(
            think,
            verb,
            None,
            (f"unknown action '{verb}'; admissible: " + ", ".join(admissible)),
            tuple(tags),
        )
    if verb == DEFER_VERB:
        qm = _QUOTE_PAIR_RE.match(payload)
        if qm and qm.group(1) not in qm.group(2):
            payload = qm.group(2).strip()
            tags.append("payload_quotes")
        payload, tag = canonical_abstention_payload(payload)
        if tag:
            tags.append(tag)
    if verb == "consult_specialist" and payload:
        dept = canon_dept(payload)
        if dept not in SPECIALISTS:
            alt = DEPT_ALIASES.get(dept) or _fuzzy(dept, list(SPECIALISTS))
            if alt:
                payload = alt
                tags.append("dept_alias")
    if verb in LIST_VERBS:
        if not payload:
            return ParsedAction(think, verb, [], None, tuple(tags))
        items, item_tags, dropped = split_items_ex(payload, verb)
        tags.extend(t for t in item_tags if t not in tags)
        return ParsedAction(think, verb, items, None, tuple(tags), dropped)
    if verb in ("ask", "search", "final_diagnosis") and not payload:
        return ParsedAction(
            think, verb, None, f"'{verb}' needs a payload after the colon", tuple(tags)
        )
    return ParsedAction(think, verb, payload, None, tuple(tags))


def parse_doctor(text: str) -> tuple[str, str, Any, str | None]:
    """-> (think, verb, payload, error): the 4-tuple view of parse_doctor_ex. A contract-3
    list with unresolvable items reports the first one the way the old parser
    did, so the probe in Eval/code/baselines/serve.py keeps its reading.
    """
    p = parse_doctor_ex(text)
    if p.error is None and p.dropped and not p.payload:
        return (
            p.think,
            p.verb,
            None,
            (f"unknown category '{p.dropped[0]}'; use exact names from the action menu"),
        )
    return p.think, p.verb, p.payload, p.error


def render_facts(deck: dict) -> str:
    if not deck["known_facts"]:
        return "(nothing yet)"
    out = []
    for f in deck["known_facts"]:
        attrs = "; ".join(f"{a}: {f[a]}" for a in FACT_ATTRS if f.get(a))
        sec = f["section"].replace("_", " ").lower()
        out.append(f"- ({sec}) {f['atomic_fact']}" + (f" [{attrs}]" if attrs else ""))
    return "\n".join(out)


def render_conversation(convo: list[dict]) -> str:
    """The transcript, one "Role: text" line per entry."""
    return "\n".join(f"{t['role'].capitalize()}: {t['text']}" for t in convo) or "(none)"


def render_workup_log(steps: list[dict]) -> str:
    """steps: prior workup turns, each {verb, payload_text, observation_text}."""
    if not steps:
        return "(no actions yet)"
    out = []
    for i, s in enumerate(steps, 1):
        out.append(f"{i}. {s['verb']}: {s['payload_text']}")
        for line in s["observation_text"].splitlines():
            out.append(f"   {line}")
    return "\n".join(out)


def render_used_line(env: "Env") -> str:
    """Inadmissible repeats, read straight off the env's no-repeat keys, so the
    line can never disagree with the validator that rejects the repeat.
    """
    pe = sorted(k.split(":", 1)[1] for k in env.used if k.startswith("physical_examination:"))
    ot = sorted(k.split(":", 1)[1] for k in env.used if k.startswith("order_tests:"))
    parts = ([f"physical_examination: {', '.join(pe)}"] if pe else []) + (
        [f"order_tests: {', '.join(ot)}"] if ot else []
    )
    return "; ".join(parts) or "(none yet)"


def render_awaiting_line(env: "Env") -> str:
    """Studies whose written interpretation a consult would release now — same
    gate the consult_specialist validator applies (v>=2: integrity check
    passed), so the line only ever offers admissible consults.
    """
    pend = [
        r for r in env.returned_assets if r.rid not in env.released and r.asset_id in env.verified
    ]
    if not pend:
        return "(none)"
    return "; ".join(
        f"{env.alias(r.asset_id)} ({r.name}) -> consult {specialist_of(r.l1, r.category)}"
        for r in pend
    )


def defer_module(cfg: Config) -> str:
    """The defer block, empty when the run forbids deferring at all. One
    accessor for both menus so the two abstention forms can never drift apart
    between the history phase and the workup phase — a doctor that learned
    `out_of_knowledge` while interviewing must still have it after the images
    land. v<=2 keeps the single frozen line; contract 3 names the axis.
    """
    if not cfg.allow_defer:
        return ""
    return ABSTENTION_LINES


def gate_asset_list(alias: Callable[[str | None], str], records: list["Record"]) -> str:
    """The gate prompt's `Returned:` line, for the records ONE batch handed over."""
    groups: dict[str, list[str]] = {}
    for r in records:
        names = groups.setdefault(r.asset_id or "", [])
        if r.name not in names:
            names.append(r.name)
    return ", ".join(f"{alias(aid)} ({'; '.join(names)})" for aid, names in groups.items())


def menu_trailer(cfg: Config) -> str:
    """Contract 3's shared menu footer: the bracket rule, plus — only when the
    run allows abstaining at all — when each abstention is the right action.
    """
    return f"{BRACKET_NOTE}\n\n{ABSTENTION_GUIDANCE}" if cfg.allow_defer else BRACKET_NOTE


def terminal_menu_lines(
    cfg: Config, phase: str, asks_used: int, wk_used: int, searches: int, must_end: bool
) -> tuple[str, str]:
    """(final_diagnosis line, begin_workup line) for the CURRENT gate state."""
    dx_line = FINAL_DX_LINE
    bw_line = BEGIN_WORKUP_LINE
    if phase == "history" and asks_used < cfg.min_asks:
        return (
            GATED_DX_MIN_ASKS.format(min_asks=cfg.min_asks, k=cfg.min_asks - asks_used),
            GATED_BEGIN_WORKUP_MIN_ASKS.format(min_asks=cfg.min_asks),
        )
    if phase == "history" and not cfg.allow_history_final:
        return GATED_DX_HISTORY_FINAL, bw_line
    if must_end:
        return dx_line, bw_line
    if wk_used < cfg.min_workup_actions and phase != "history":
        return GATED_DX_MIN_WORKUP.format(m=cfg.min_workup_actions - wk_used), bw_line
    return dx_line, bw_line


def history_menu(
    cfg: Config, speaker_role: str, asks_used: int = 0, wk_used: int = 0, searches: int = 0
) -> str:
    ask_line = (
        "" if asks_used >= cfg.max_asks else HISTORY_ASK_LINE.format(speaker_role=speaker_role)
    )
    final_line, begin_line = terminal_menu_lines(
        cfg, "history", asks_used, wk_used, searches, must_end=False
    )
    return HISTORY_MENU.format(
        ask_line=ask_line,
        final_line=final_line,
        final_note=(FINAL_DX_NOTE_HISTORY if final_line == FINAL_DX_LINE else ""),
        begin_line=begin_line,
    )


def search_module(cfg: Config, searches_done: int) -> str:
    """Full protocol until the first search; brief line afterwards."""
    if not cfg.search_url or searches_done >= cfg.max_searches:
        return ""
    if searches_done == 0:
        return SEARCH_LINE
    brief = SEARCH_BRIEF
    return brief.format(searches_left=cfg.max_searches - searches_done)


def workup_menu(
    cfg: Config, searches_done: int = 0, wk_used: int = 0, must_end: bool = False
) -> str:
    final_line, _ = terminal_menu_lines(
        cfg, "workup", cfg.min_asks, wk_used, searches_done, must_end
    )
    fields = dict(
        max_items=cfg.max_items_per_action,
        pe_menu=_PE_MENU,
        test_menu=test_menu(),
        dept_menu=dept_menu(),
        gate_clause=" and that passed the integrity check",
        search_module=search_module(cfg, searches_done),
        defer_line=defer_module(cfg),
        final_line=final_line,
    )
    return WORKUP_MENU.format(trailer=menu_trailer(cfg), **fields)


def workup_rules(cfg: Config, wk_used: int) -> str:
    """The rules block: full on the first workup turn, brief afterwards,
    plus INTEGRITY_LINE as its own trailing paragraph on BOTH.
    """
    base = WORKUP_RULES_FULL if wk_used == 0 else WORKUP_RULES_BRIEF
    return f"{base}\n\n{INTEGRITY_LINE}" if cfg.integrity_hint else base


def render_history_prompt(
    cfg: Config,
    ep: Episode,
    deck: dict,
    convo: list[dict],
    asks_used: int,
    wk_used: int = 0,
    searches: int = 0,
) -> str:
    rendered = HISTORY_TEMPLATE.format(
        speaker_role=ep.speaker_role,
        step_count=asks_used,
        demographics=ep.demographics or "(not recorded)",
        remaining_asks=max(0, cfg.max_asks - asks_used),
        max_asks=cfg.max_asks,
        known_facts=render_facts(deck),
        unknown_topics=", ".join(deck["unknown_topics"]) or "(none)",
        no_new_fact_streak=deck["no_new_fact_streak"],
        conversation=render_conversation(convo),
        action_menu=history_menu(
            cfg, ep.speaker_role, asks_used, wk_used=wk_used, searches=searches
        ),
    )
    if cfg.identity_anchor:
        rendered += identity_anchor_block(_patient_tag(ep.pmcid), ep.demographics)
    return rendered


def render_workup_prompt(
    cfg: Config,
    ep: Episode,
    deck: dict,
    steps: list[dict],
    remaining: int,
    searches_done: int = 0,
    images_pending: int = 0,
    env: "Env | None" = None,
    wk_used: int = 0,
) -> str:
    history = (
        render_facts(deck)
        if any(f["section"] == "Demographics" for f in deck["known_facts"])
        else (f"- demographics: {ep.demographics or '(not stated)'}\n" + render_facts(deck))
    )
    unknowns = ", ".join(deck["unknown_topics"]) or "(none)"
    history += f"\n- topics the patient could not answer: {unknowns}"
    rendered = WORKUP_TEMPLATE.format(
        rules_module=workup_rules(cfg, wk_used),
        action_menu=workup_menu(cfg, searches_done, wk_used=wk_used, must_end=remaining <= 0),
        history_facts=history,
        workup_log=render_workup_log(steps),
        used_line=render_used_line(env),
        awaiting_line=render_awaiting_line(env),
        remaining_workup_turns=remaining,
        integrity_module="",
    )
    if cfg.identity_anchor:
        rendered += identity_anchor_block(_patient_tag(ep.pmcid), ep.demographics)
    return rendered


def conclude_menu(cfg: Config, searches_done: int = 0, wk_used: int = 0) -> str:
    """The CONCLUDE-phase menu: the two terminals the validator accepts at
    must_end and nothing else. Built from the same accessors the workup menu
    uses (`terminal_menu_lines(must_end=True)` for the diagnosis line,
    `defer_module` for the abstention block, `menu_trailer` for the slot
    rule), so the wording of a terminal can never drift between the two menus.
    The begin_workup line the terminal accessor also returns is discarded: the
    interview is over.
    """
    final_line, _ = terminal_menu_lines(
        cfg, "workup", cfg.min_asks, wk_used, searches_done, must_end=True
    )
    return CONCLUDE_MENU.format(
        defer_line=defer_module(cfg), final_line=final_line, trailer=menu_trailer(cfg)
    )


def render_conclude_prompt(
    cfg: Config,
    ep: Episode,
    deck: dict,
    steps: list[dict],
    searches_done: int = 0,
    env: "Env | None" = None,
    wk_used: int = 0,
) -> str:
    """The prompt of a CONCLUDE turn: the workup
    prompt's evidence blocks (history facts, workup log, awaiting line) with
    the workup menu replaced by `conclude_menu`. No rules module, no used
    line, no integrity module: none of them describe an admissible action any
    more. `render_workup_prompt` is untouched, so every legacy record and every
    non-conclude turn renders byte-identically to before.
    """
    history = (
        render_facts(deck)
        if any(f["section"] == "Demographics" for f in deck["known_facts"])
        else (f"- demographics: {ep.demographics or '(not stated)'}\n" + render_facts(deck))
    )
    unknowns = ", ".join(deck["unknown_topics"]) or "(none)"
    history += f"\n- topics the patient could not answer: {unknowns}"
    rendered = CONCLUDE_TEMPLATE.format(
        action_menu=conclude_menu(cfg, searches_done, wk_used=wk_used),
        history_facts=history,
        workup_log=render_workup_log(steps),
        awaiting_line=render_awaiting_line(env),
    )
    if cfg.identity_anchor:
        rendered += identity_anchor_block(_patient_tag(ep.pmcid), ep.demographics)
    return rendered


def anchor_key(
    phase: str,
    deck: dict,
    env: "Env",
    asks_used: int,
    wk_used: int,
    pending_gate: dict | None,
    mode: str = ANCHOR_MODE_DEFAULT,
) -> str:
    """GiGPO anchor: sha1 of the SEMANTIC env state. Deliberately excludes
    transcript wording (patient replies are sampled) and action ORDER within
    the used-set — residual within-anchor observation variance is phrasing
    only, the fuzz GiGPO's similarity mode tolerates.
    """
    import hashlib

    if mode not in ANCHOR_MODES:
        raise ValueError(f"unknown anchor mode {mode!r}; use one of {ANCHOR_MODES}")
    if phase in ("workup", "gate", "conclude"):
        corr = getattr(env, "bound_corruption", None) or getattr(
            getattr(env, "ep", None), "corruption", None
        )
        if phase in ("workup", "conclude"):
            state = {
                "mode": mode,
                "phase": phase,
                "wk": wk_used,
                "any_search": env.searches > 0,
                "n_consults": len(env.released),
                "verified": sorted(env.verified),
                "unreliable": sorted(env.unreliable),
                "corr": corr,
            }
            if phase == "conclude":
                state["retry_used"] = bool(getattr(env, "conclude_forfeits", 0))
        else:
            ceiling = getattr(getattr(env, "cfg", None), "max_workup_turns", None)
            if ceiling is None:
                ceiling = 20
            left = ceiling - wk_used
            state = {
                "mode": mode,
                "phase": phase,
                "gate": (
                    sorted(r.asset_id for r in pending_gate["assets"]) if pending_gate else []
                ),
                "ordered": (list(pending_gate.get("ordered", [])) if pending_gate else []),
                "verified": sorted(env.verified),
                "unreliable": sorted(env.unreliable),
                "corr": corr,
                "wk_left": ("endgame" if left <= 1 else "mid" if left <= 5 else "high"),
            }
            state["more_batches"] = bool((pending_gate or {}).get("pending_count") or 0)
        return hashlib.sha1(json.dumps(state, sort_keys=True).encode()).hexdigest()
    state = {
        "mode": mode,
        "phase": phase,
        "asks": asks_used,
        "wk": wk_used,
        "retry_used": bool(getattr(env, "conclude_forfeits", 0)) if phase == "conclude" else False,
        "used": sorted(env.used),
        "released": sorted(env.released),
        "verified": sorted(env.verified),
        "unreliable": sorted(env.unreliable),
        "searches": env.searches,
        "corr": (
            getattr(env, "bound_corruption", None)
            or getattr(getattr(env, "ep", None), "corruption", None)
        ),
        "gate": (sorted(r.asset_id for r in pending_gate["assets"]) if pending_gate else []),
    }
    return hashlib.sha1(json.dumps(state, sort_keys=True).encode()).hexdigest()


def new_deck(ep: "Episode | None" = None) -> dict:
    """Demographics are observed at first sight, so they seed the deck as
    already-known facts (mirrored in export_sft replay and the viewers).
    """
    deck = {"known_facts": [], "unknown_topics": [], "no_new_fact_streak": 0}
    if ep is not None:
        deck["known_facts"].extend(
            dict(f) for f in ep.patient_facts if f["section"] == "Demographics"
        )
    return deck


def apply_patient_turn(
    deck: dict, ep: Episode, used_ids: list[str], unknown: list[str]
) -> tuple[list[str], list[str]]:
    known = {f["fact_id"] for f in deck["known_facts"]}
    new_ids = [i for i in used_ids if i not in known]
    deck["known_facts"].extend(dict(ep.fact_index[i]) for i in new_ids)
    new_unknown = [t for t in unknown if t and t not in deck["unknown_topics"]]
    deck["unknown_topics"].extend(new_unknown)
    deck["no_new_fact_streak"] = 0 if new_ids else deck["no_new_fact_streak"] + 1
    return new_ids, new_unknown


PATIENT_FALLBACK = {"answer": "I'm not sure about that.", "used_fact_ids": [], "unknown_topics": []}
JSON_RE = re.compile(r"\{.*\}", re.S)


def ask_patient(
    cfg: Config,
    ep: Episode,
    patient: LLMAgent,
    convo: list[dict],
    question: str | None,
    stats: dict,
) -> dict:
    payload = {
        "encounter_facts": ep.encounter_facts,
        "patient_actor_facts": ep.patient_facts,
        "conversation_history": [t for t in convo if t["role"] != "env"],
        "doctor_question": question,
        "speaker_role": ep.speaker_role,
    }
    messages = [
        {"role": "system", "content": PATIENT_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    for _ in range(cfg.validation_retries + 1):
        raw = patient.complete(messages, ep.pmcid, "history")
        obj: Any = None
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            m = JSON_RE.search(raw)
            if m:
                try:
                    obj = json.loads(m.group(0))
                except json.JSONDecodeError:
                    obj = None
        err = None
        if (
            not isinstance(obj, dict)
            or not isinstance(obj.get("answer"), str)
            or not obj["answer"].strip()
        ):
            err = "output must be a JSON object with a non-empty 'answer'"
        elif not isinstance(obj.get("used_fact_ids"), list):
            err = "'used_fact_ids' must be a list"
        else:
            bad = [i for i in obj["used_fact_ids"] if i not in ep.fact_index]
            if bad:
                err = f"unknown fact ids {bad}; use only supplied fact_ids"
            elif question is None and len(set(obj["used_fact_ids"])) > 2:
                err = (
                    "the opening statement may reveal at most TWO atomic "
                    "facts (Encounter_Context first, then one central "
                    "Chief_Complaint fact)"
                )
        if err is None:
            return {
                "answer": obj["answer"].strip(),
                "used_fact_ids": list(dict.fromkeys(obj["used_fact_ids"])),
                "unknown_topics": [t for t in obj.get("unknown_topics", []) if isinstance(t, str)],
            }
        stats["patient_retries"] += 1
        messages += [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": f"Rejected: {err}. Return corrected JSON only."},
        ]
    stats["patient_fallbacks"] += 1
    return dict(PATIENT_FALLBACK)


def load_or_make_opening(cfg: Config, ep: Episode, patient, convo, stats, ask=ask_patient) -> dict:
    """The opening statement, frozen per (pmcid, corrupt_seed) under rl_mode."""
    if not (cfg.freeze_opening or cfg.rl_mode):
        return ask(cfg, ep, patient, convo, None, stats)
    cache = cfg.state / "openings" / f"{ep.pmcid}_{cfg.corrupt_seed}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    opening = ask(cfg, ep, patient, convo, None, stats)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(opening, ensure_ascii=False), encoding="utf-8")
    return opening


def _patient_tag(pmcid: str) -> str:
    import hashlib as _hl

    return f"PT-{int(_hl.sha1(pmcid.encode()).hexdigest(), 16) % 90000 + 10000}"


def identity_anchor_block(tag: str, demographics: str) -> str:
    """The ground-truth identity block."""
    demo = demographics or "demographics not recorded"
    return (
        f"\n\nPATIENT IDENTITY (ground truth): {tag} ({demo}).\n"
        f"Begin EVERY <reasoning> block with its first line exactly:\n"
        f"Patient: {tag} ({demo})\n"
        "Verify every returned result — headers, images, reports — "
        "against this identity; a result naming a different patient is a "
        "mismatch."
    )


ASSET_ALIAS_PREFIX = "IMG-"


class AssetAliases:
    """De-identified `IMG-1..n` study ids."""

    cfg: "Config"
    asset_aliases: dict[str, str]
    alias_assign: bool

    def alias(self, asset_id: str | None) -> str:
        """The id the doctor sees for this asset (the real one below contract 3)."""
        if not asset_id:
            return ""
        a = self.asset_aliases.get(asset_id)
        if a is None:
            if not self.alias_assign:
                return asset_id
            a = f"{ASSET_ALIAS_PREFIX}{len(self.asset_aliases) + 1}"
            self.asset_aliases[asset_id] = a
        return a


def unalias_map(aliases: dict[str, str] | None) -> dict[str, str]:
    """{alias -> real asset id}, the inverse of a record's `asset_aliases`."""
    return {v: k for k, v in (aliases or {}).items()}


class Env(AssetAliases):
    """Deterministic workup environment for one episode."""

    def __init__(self, cfg: Config, ep: Episode, swap_pool: list = ()):
        self.patient_tag = _patient_tag(ep.pmcid)
        self.cfg, self.ep = cfg, ep
        self.asset_aliases: dict[str, str] = {}
        self.alias_assign = True
        self.swap_pool = swap_pool
        self.used: set[str] = set()
        self.returned_assets: list[Record] = []
        self.released: set[int] = set()
        self.verified: set[str] = set()
        self.unreliable: set[str] = set()
        self.assets_injected = 0
        self.image_injections = 0
        self.searches = 0
        self.consulted: set[str] = set()
        self.asked: list[frozenset[str]] = []
        self.bound_corruption: str | None = None
        self.delivery_queue: list[dict] = []
        self.delivery_batches = 0
        self.pending_max = 0
        dm = getattr(cfg, "deliver_mode", DELIVER_LEGACY)
        if dm not in DELIVER_MODES:
            raise ValueError(f"deliver_mode must be one of {DELIVER_MODES}, got {dm!r}")
        cm = getattr(cfg, "conclude_mode", "conclude_retry")
        if cm not in CONCLUDE_MODES:
            raise ValueError(f"conclude_mode must be one of {CONCLUDE_MODES}, got {cm!r}")
        rv = getattr(cfg, "rule_violation", "forfeit")
        if rv not in RULE_VIOLATION_MODES:
            raise ValueError(f"rule_violation must be one of {RULE_VIOLATION_MODES}, got {rv!r}")
        rt = getattr(cfg, "refusal_text", "terse")
        if rt not in REFUSAL_TEXT_MODES:
            raise ValueError(f"refusal_text must be one of {REFUSAL_TEXT_MODES}, got {rt!r}")
        if dm == DELIVER_CHUNKED and not (
            1 <= int(cfg.deliver_chunk) <= int(cfg.max_images_per_turn)
        ):
            raise ValueError(
                f"deliver_chunk must satisfy 1 <= deliver_chunk <= max_images_per_turn, "
                f"got {cfg.deliver_chunk} / {cfg.max_images_per_turn}"
            )
        if dm == DELIVER_CHUNKED and int(cfg.max_batches_per_order) < 1:
            raise ValueError("max_batches_per_order must be >= 1")

    def _chunked(self) -> bool:
        return getattr(self.cfg, "deliver_mode", DELIVER_LEGACY) == DELIVER_CHUNKED

    def _dq(self) -> list[dict]:
        q = getattr(self, "delivery_queue", None)
        if q is None:
            q = self.delivery_queue = []
            self.delivery_batches = 0
            self.pending_max = 0
        return q

    def note_ask(self, question: str) -> None:
        self.asked.append(_ask_words(question))

    def note_gate(self, assets: list["Record"], verdict: str) -> None:
        ids = {r.asset_id for r in assets if r.asset_id}
        (self.verified if verdict == "pass" else self.unreliable).update(ids)

    def _maybe_bind_corruption(self, batch: list["Record"]) -> None:
        """Bind this case's armed corruption to a study in `batch`."""
        if not (self.ep.corruption_armed and self.ep.corruption is None and batch):
            return
        eligible = [
            r for r in batch if r.serving_kind is None or r.serving_kind in VICTIM_SERVING_KINDS
        ]
        if not eligible:
            return
        self.image_injections += 1
        if self.image_injections < self.ep.corrupt_bind_k:
            return
        rng = random.Random(f"corrupt-bind::{self.ep.pmcid}::{self.cfg.corrupt_seed}")
        seen: set[str] = set()
        uniq: list[Record] = []
        for r in eligible:
            if r.asset_id not in seen:
                seen.add(r.asset_id)
                uniq.append(r)
        victim = rng.choice(uniq)
        self.bound_corruption = f"{getattr(victim, 'rid', None) or victim.asset_id}"
        corruption = None
        vocab = GATE_AXES
        want = self.cfg.corrupt_axis or None
        if self.cfg.gate_bank_dir:
            from gate_engine_bridge import pick_bank_swap

            corruption = pick_bank_swap(
                victim,
                self.ep.pmcid,
                rng,
                self.cfg.gate_bank_dir,
                want_axis=want,
                allowed_axes=vocab,
            )
            if corruption and not Path(corruption["swap_path"]).is_file():
                corruption = None
            if corruption:
                axes = [a for a in (corruption.get("axes") or []) if a in vocab]
                if axes:
                    corruption["axes"] = axes
                else:
                    LOG.warning(
                        "bank swap for %s has no closed-vocabulary axes (%s); declining",
                        self.ep.pmcid,
                        corruption.get("axes"),
                    )
                    corruption = None
        if corruption is None:
            kind = "modality_anatomy"
            pool = [p for p in self.swap_pool if p[2] != victim.asset_category]
            pool = [p for p in pool if Path(p[1]).is_file()]
            if not pool:
                return
            spmcid, spath, scat = rng.choice(pool)
            corruption = {
                "kind": kind,
                "axes": corruption_axes(kind),
                "asset_id": victim.asset_id,
                "original_path": victim.asset_path,
                "swap_pmcid": spmcid,
                "swap_path": spath,
                "swap_category": scat,
            }
        victim.asset_path = corruption["swap_path"]
        self.ep.corruption = corruption

    @property
    def ESCAPE(self) -> str:
        alt = f" or {surface_defer_verb()}" if self.cfg.allow_defer else ""
        return f"; if nothing informative remains, output final_diagnosis{alt}"

    def validate(
        self,
        verb: str,
        payload: Any,
        phase: str,
        asks_used: int,
        must_end: bool,
        wk_used: int = 0,
        dropped: tuple[str, ...] = (),
    ) -> str | None:
        cfg = self.cfg
        if (
            dropped
            and verb in LIST_VERBS
            and getattr(cfg, "rule_violation", "forfeit") == "forfeit"
        ):
            return f"unknown category '{dropped[0]}'; use exact names from the action menu"
        defer_name = surface_defer_verb()
        terminals = ("final_diagnosis", DEFER_VERB) if cfg.allow_defer else ("final_diagnosis",)
        terminal_names = [defer_name if t == DEFER_VERB else t for t in terminals]
        if verb == "defer_to_human" and not cfg.allow_defer:
            return (
                f"{defer_name} is not available in this run; continue the "
                "workup or output final_diagnosis"
            )
        if verb == "defer_to_human" and phase == "history":
            return ABSTENTION_PHASE_ERR
        if verb == "final_diagnosis" and phase == "history" and not cfg.allow_history_final:
            return HISTORY_FINAL_PHASE_ERR
        if verb == "defer_to_human":
            slot_err = defer_slot_error(payload)
            if slot_err:
                return slot_err
        if verb == "final_diagnosis" and "|" in str(payload or ""):
            return SINGLE_DX_ERR if cfg.allow_defer else SINGLE_DX_ERR_NO_DEFER
        if verb == "final_diagnosis" and re.search(r"[\[\]]", str(payload or "")):
            return TERMINAL_BRACKET_ERR
        if (
            verb == "final_diagnosis"
            and not must_end
            and wk_used < cfg.min_workup_actions
            and phase != "history"
        ):
            if phase == "history":
                return (
                    "a diagnostic workup is required before finalizing — "
                    f"begin_workup first (at least {cfg.min_workup_actions} "
                    "workup action(s) required)"
                )
            return (
                f"complete at least {cfg.min_workup_actions} workup "
                f"action(s) before finalizing "
                f"({cfg.min_workup_actions - wk_used} remaining)"
            )
        if must_end and verb not in terminals:
            return "no workup actions remain: you must output " + " or ".join(terminal_names)
        if verb == "ask":
            if phase != "history":
                return "the interview is over; 'ask' is no longer admissible"
            if asks_used >= cfg.max_asks:
                return "no questions remain; choose a different action"
            if cfg.rl_mode:
                _t = str(payload or "").strip()
                if _t.startswith("[") and _t.endswith("]"):
                    return "the ask slot was echoed as written; ask a real question"
            if (
                getattr(cfg, "rule_violation", "forfeit") != "noop"
                and ask_overlap(str(payload), self.asked) >= 0.5
            ):
                return (
                    "this substantially repeats an earlier question — "
                    'answers already given (including "I don\'t know") are '
                    "final; ask about different information or start the "
                    "workup"
                )
            return None
        if phase == "history" and asks_used < cfg.min_asks and verb != "defer_to_human":
            return (
                f"the history is too thin to close — ask at least "
                f"{cfg.min_asks - asks_used} more question(s) first"
            )
        if verb == "final_diagnosis" and no_diagnosis_final(payload):
            return NO_DX_ERR_TAG
        if verb == "begin_workup":
            if phase != "history":
                return (
                    "the workup has already begun; choose an examination, "
                    "test, consult, or terminal action"
                )
            return None
        if verb == "physical_examination":
            if not payload:
                return f"give 1-{cfg.max_items_per_action} examination categories from the menu"
            bad = [x for x in payload if x not in PE_MENU_MAP]
            if bad:
                return f"unknown examination categories {bad}; use exact menu names"
            if len(payload) > cfg.max_items_per_action:
                return f"at most {cfg.max_items_per_action} categories per action"
            dup = [
                x
                for x in payload
                if f"physical_examination:{x.lower()}" in self.used
                or f"order_tests:{PE_VIRTUAL_L1.lower()}/{x.lower()}" in self.used
            ]
            if dup:
                return f"already examined: {dup}; never repeat a category" + self.ESCAPE
        elif verb == "order_tests":
            if not payload:
                return (
                    f"give 1-{cfg.max_items_per_action} L1/L2 test pairs "
                    "from the menu, e.g. Laboratory/Hematology"
                )
            bad = [
                x
                for x in payload
                if x not in TEST_PAIRS_MENU
                and not (x.startswith(PE_VIRTUAL_L1 + "/") and x.partition("/")[2] in PE_MENU_MAP)
            ]
            if bad:
                return (
                    f"unknown test pairs {bad}; use exact L1/L2 pairs from the "
                    "menu, e.g. Laboratory/Hematology"
                )
            if len(payload) > cfg.max_items_per_action:
                return f"at most {cfg.max_items_per_action} pairs per action"
            dup = [
                x
                for x in payload
                if f"order_tests:{x.lower()}" in self.used
                or (
                    x.startswith(PE_VIRTUAL_L1 + "/")
                    and f"physical_examination:{x.partition('/')[2].lower()}" in self.used
                )
            ]
            if dup:
                return f"already ordered: {dup}; never repeat a pair" + self.ESCAPE
        elif verb == "consult_specialist":
            dept = canon_dept(str(payload))
            if dept not in SPECIALISTS:
                return (
                    f"unknown specialist '{payload}'; image-interpretation "
                    f"specialists: {dept_menu()}"
                )
            if not self._pending_for(dept):
                hint = (
                    "order the study first; it must pass the integrity "
                    "check before a specialist reads it"
                )
                return (f"'{dept}' has no studies awaiting interpretation — " + hint) + self.ESCAPE
        elif verb == "search":
            if not self.cfg.search_url:
                return "search is not available in this run"
            if self.searches >= self.cfg.max_searches:
                return f"the search budget ({self.cfg.max_searches}) is exhausted" + self.ESCAPE
            if f"search:{str(payload).lower()}" in self.used:
                return "identical query already searched; refine it or act on what you have"
        return None

    def prepare_list(
        self, verb: str, items: list[str], dropped: tuple[str, ...] = ()
    ) -> tuple[list[str], list[str]]:
        """rule_violation=noop: the admissible part of a list
        action and the notes that explain what was left out -- repeats are
        skipped, the per-action cap truncates, items the parser could not
        resolve are dropped. Pure: nothing is executed or recorded here.
        """
        notes: list[str] = []
        kept: list[str] = []
        pe = verb == "physical_examination"
        for x in items:
            if pe:
                dup = (
                    f"physical_examination:{x.lower()}" in self.used
                    or f"order_tests:{PE_VIRTUAL_L1.lower()}/{x.lower()}" in self.used
                )
            else:
                dup = f"order_tests:{x.lower()}" in self.used or (
                    x.startswith(PE_VIRTUAL_L1 + "/")
                    and f"physical_examination:{x.partition('/')[2].lower()}" in self.used
                )
            if dup:
                notes.append(f"{x}: already {'examined' if pe else 'ordered'}, skipped")
            elif x not in kept:
                kept.append(x)
        cap = int(self.cfg.max_items_per_action)
        if len(kept) > cap:
            notes.append(f"at most {cap} per action: {', '.join(kept[cap:])} not taken")
            kept = kept[:cap]
        for d in dropped:
            notes.append(f"{d}: not on the menu, skipped")
        return kept, notes

    def execute(self, verb: str, payload: Any) -> tuple[str, list[dict], dict]:
        if verb == "physical_examination":
            return self._exam_or_tests(
                "physical_examination",
                payload,
                lambda cat: [
                    r
                    for r in self.ep.records
                    if r.l1 == "Physical_Examination" and r.category in PE_MENU_MAP[cat]
                ],
            )
        if verb == "order_tests":

            def match(pair: str):
                l1, _, l2 = pair.partition("/")
                if l1 == PE_VIRTUAL_L1:
                    return [
                        r
                        for r in self.ep.records
                        if r.l1 == "Physical_Examination" and r.category in PE_MENU_MAP.get(l2, ())
                    ]
                return [r for r in self.ep.records if r.l1 == l1 and r.category == l2]

            out = self._exam_or_tests("order_tests", payload, match)
            for x in payload:
                if x.startswith(PE_VIRTUAL_L1 + "/"):
                    self.used.add(f"physical_examination:{x.partition('/')[2].lower()}")
            return out
        if verb == "consult_specialist":
            return self._consult(canon_dept(str(payload)))
        if verb == "search":
            return self._search(str(payload))
        return "", [], {}

    def _attach_plan(self, items: list[str], match: Callable[[str], list[Record]]) -> list[Record]:
        """One record per distinct asset this turn attaches,
        in service order, inside the per-turn image budget.
        """
        seen: set[str] = set()
        ordered: list[Record] = []
        for cat in items:
            for r in match(cat):
                if not (r.asset_path and Path(r.asset_path).exists()):
                    continue
                if r.asset_id in seen:
                    continue
                seen.add(r.asset_id)
                ordered.append(r)

        def eligible(r: Record) -> bool:
            return r.serving_kind is None or r.serving_kind in VICTIM_SERVING_KINDS

        out = [r for r in ordered if eligible(r)] + [r for r in ordered if not eligible(r)]
        if getattr(getattr(self, "cfg", None), "deliver_mode", DELIVER_LEGACY) == DELIVER_CHUNKED:
            return out
        return out[: self.cfg.max_images_per_turn]

    def _exam_or_tests(
        self, verb: str, items: list[str], match: Callable[[str], list[Record]]
    ) -> tuple[str, list[dict], dict]:
        parts: list[dict] = []
        n_img = 0
        gate_assets: list[Record] = []
        plan = self._attach_plan(items, match)
        plan_gated = any(
            r.serving_kind == "composite_plate" for cat in items for r in match(cat) if r.asset_path
        )
        plan_ids = {r.asset_id for r in plan}
        chunked = self._chunked()
        chunk = (
            int(getattr(self.cfg, "deliver_chunk", self.cfg.max_images_per_turn)) if chunked else 0
        )
        max_batches = int(getattr(self.cfg, "max_batches_per_order", 1)) if chunked else 1
        batch_of = {r.asset_id: i // chunk for i, r in enumerate(plan)} if chunked else {}
        queued: dict[int, list] = {}
        if self.ep.corruption_armed and self.ep.corruption is None:
            img_batch = plan
            self._maybe_bind_corruption(img_batch)
        lines: list[str] = []
        lines.append(
            f"Results for patient {self.patient_tag} "
            f"({self.ep.demographics or 'demographics not stated'}):"
        )
        attached: dict[str, str] = {}
        served: set[tuple[str, str, str]] = set()
        for cat in items:
            self.used.add(f"{verb}:{cat.lower()}")
            recs = match(cat)
            if not recs:
                lines.append(f"{cat}: Not available in record.")
                continue
            for r in recs:
                if r.asset_path:
                    key = (cat, r.name, r.asset_id or "")
                    if key in served:
                        continue
                    if r.asset_id in attached:
                        served.add(key)
                        self.returned_assets.append(r)
                        gate_assets.append(r)
                        lines.append(
                            f"{cat} — {r.name}: [same image as {attached[r.asset_id]} above]"
                        )
                        continue
                later = batch_of.get(r.asset_id) if (chunked and r.asset_path) else None
                if later is not None and later >= 1:
                    key = (cat, r.name, r.asset_id or "")
                    if key in served:
                        continue
                    served.add(key)
                    if later < max_batches:
                        queued.setdefault(later, []).append((cat, r))
                        lines.append(f"{cat} — {r.name}: {PENDING_LINE}")
                    else:
                        lines.append(f"{cat} — {r.name}: {NOT_DELIVERED_LINE}")
                    continue
                if (
                    r.asset_path
                    and n_img < self.cfg.max_images_per_turn
                    and (not plan_gated or r.asset_id in plan_ids)
                    and (not chunked or later == 0)
                ):
                    img = _b64_image_part(r.asset_path)
                    if img:
                        self.returned_assets.append(r)
                        gate_assets.append(r)
                        n_img += 1
                        alias = self.alias(r.asset_id)
                        served.add((cat, r.name, r.asset_id or ""))
                        if r.asset_id:
                            attached[r.asset_id] = alias
                        lines.append(
                            f"{cat} — {r.name}: [image "
                            f"{alias} attached — "
                            "integrity check next; once it passes, consult the "
                            "matching specialist for the written interpretation]"
                        )
                        parts.append(img)
                        continue
                lines.append(f"{cat} — {r.name}: {r.findings or '(no findings text)'}")
        self.assets_injected += n_img
        extras = {"images": n_img, "gate_assets": gate_assets, "ordered_items": list(items)}
        if chunked:
            q = self._dq()
            for b in sorted(queued):
                q.append(
                    {"batch": b + 1, "groups": queued[b], "ordered": list(items), "verb": verb}
                )
            n_pending = sum(len(g) for g in queued.values())
            self.pending_max = max(getattr(self, "pending_max", 0), n_pending)
            extras["batches_total"] = 1 + len(queued)
            extras["pending_count"] = n_pending
        if gate_assets:
            extras["experts_expected"] = sorted(
                {specialist_of(r.l1, r.category) for r in gate_assets}
            )
        return "\n".join(lines), parts, extras

    def attach_batch(self, entry: dict) -> tuple[str, list[dict], list["Record"]]:
        """Deliver one queued batch: the SAME bookkeeping as the order turn's attach branch -- alias
        assignment, returned_assets, gate_assets, assets_injected, one
        attachment per figure -- so a figure delivered on a later gate is
        consultable, coverable and exposable exactly like one delivered first.
        Findings text is never released here (a consult does that after the
        gate). Returns (observation_text, image_parts, gate_assets).
        """
        lines: list[str] = []
        parts: list[dict] = []
        gate_assets: list[Record] = []
        attached: dict[str, str] = {}
        n_img = 0
        for cat, r in entry["groups"]:
            if r.asset_id in attached:
                self.returned_assets.append(r)
                gate_assets.append(r)
                lines.append(f"{cat} — {r.name}: [same image as {attached[r.asset_id]} above]")
                continue
            img = _b64_image_part(r.asset_path)
            if not img:
                lines.append(f"{cat} — {r.name}: {r.findings or '(no findings text)'}")
                continue
            self.returned_assets.append(r)
            gate_assets.append(r)
            n_img += 1
            alias = self.alias(r.asset_id)
            if r.asset_id:
                attached[r.asset_id] = alias
            lines.append(
                f"{cat} — {r.name}: [image {alias} attached — "
                "integrity check next; once it passes, consult the "
                "matching specialist for the written interpretation]"
            )
            parts.append(img)
        self.assets_injected += n_img
        self.delivery_batches = getattr(self, "delivery_batches", 0) + 1
        return "\n".join(lines), parts, gate_assets

    def _pending_for(self, dept: str) -> list["Record"]:
        """The RECORDS this department would release now: returned, not yet
        released, gate-passed where the gate applies, and routed here by the
        ordered category. One consult = one department: a batch that spans
        several departments takes one consult turn each, and each turn releases
        exactly its own department's records.
        """
        return [
            r
            for r in self.returned_assets
            if specialist_of(r.l1, r.category) == dept
            and r.rid not in self.released
            and r.asset_id in self.verified
        ]

    def _consult(self, dept: str) -> tuple[str, list[dict], dict]:
        pending = self._pending_for(dept)
        if not pending:
            return (
                f"{dept}: no gate-passed studies awaiting interpretation.",
                [],
                {"released": []},
            )
        self.consulted.add(dept)
        lines = [f"{dept} interpretation (patient {self.patient_tag}):"]
        for r in pending:
            self.released.add(r.rid)
            tag = r.category
            lines.append(f"- {r.name} ({tag}): {r.findings or '(no findings text)'}")
        return (
            "\n".join(lines),
            [],
            {"released": [{"rid": r.rid, "name": r.name, "category": r.category} for r in pending]},
        )

    def _search(self, query: str) -> tuple[str, list[dict], dict]:
        self.used.add(f"search:{query.lower()}")
        self.searches += 1
        import urllib.request

        req_body = {
            "queries": [query],
            "topk": self.cfg.search_topk,
            "exclude": [{"pmcids": [self.ep.pmcid]}],
            "return_text": True,
        }
        req_body["rewrite"] = True
        body = json.dumps(req_body).encode()
        req = urllib.request.Request(
            self.cfg.search_url.rstrip("/") + "/retrieve",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                payload = json.loads(r.read().decode())
            res = (payload.get("result") or [{}])[0]
            text = (
                res.get("evidence_text")
                or f"<information>{json.dumps(res, ensure_ascii=False)[:4000]}</information>"
            )
            extras = {
                "search_excluded": res.get("excluded", 0),
                "search_errors": res.get("errors", []),
            }
        except Exception as e:
            text = f"<information>search failed: {str(e)[:200]}</information>"
            extras = {"search_errors": [str(e)[:200]]}
        return text, [], extras


def gate_ground_truth(corruption: dict | None, assets: list["Record"]) -> tuple[bool, list[str]]:
    """(gt_mismatch, gt_axes) for ONE integrity-gate turn."""
    corr = corruption or {}
    if not any(r.asset_id == corr.get("asset_id") for r in assets):
        return False, []
    return True, list(corr.get("axes") or corruption_axes(str(corr.get("kind"))))


def run_episode(
    cfg: Config,
    row: dict,
    doctor: LLMAgent,
    patient: LLMAgent,
    swap_pool: list[tuple[str, str, str]],
    on_turn=None,
) -> dict:
    ep = parse_profile(row, cfg, swap_pool)
    env = Env(cfg, ep, swap_pool)
    deck = new_deck(ep)
    convo: list[dict] = []
    stats = {
        k: 0 for k in ("doctor_retries", "doctor_fallbacks", "patient_retries", "patient_fallbacks")
    }
    t0 = time.time()

    opening = load_or_make_opening(cfg, ep, patient, convo, stats)
    apply_patient_turn(deck, ep, opening["used_fact_ids"], opening["unknown_topics"])
    convo.append({"role": "patient", "text": opening["answer"]})

    turns: list[dict] = []
    wk_steps: list[dict] = []
    pending_parts: list[dict] = []
    pending_gate: dict | None = None
    phase = "history"
    asks_used = 0
    wk_used = 0
    conclude_forfeits = 0
    final: dict | None = None

    while final is None:
        if pending_gate is not None:
            ga = pending_gate
            pending_gate = None
            g_anchor = anchor_key("gate", deck, env, asks_used, wk_used, ga, mode=cfg.anchor_mode)
            gt_mismatch, gt_axes = gate_ground_truth(ep.corruption, ga["assets"])
            g_k, g_K = tuple(ga.get("batch") or (1, 1))
            gprompt = GATE_TEMPLATE.format(
                patient_tag=env.patient_tag,
                demographics=ep.demographics or "(not stated)",
                ordered=", ".join(ga["ordered"]) + (f" (batch {g_k} of {g_K})" if g_K > 1 else ""),
                asset_list=gate_asset_list(env.alias, ga["assets"]),
            )
            gmsgs = [{"role": "user", "content": [{"type": "text", "text": gprompt}] + ga["parts"]}]
            gthink = gverdict = greason = ""
            gaxes: list[str] = []
            gerr: str | None = None
            for _ in range(cfg.policy_retries + 1):
                graw = doctor.complete(gmsgs, ep.pmcid, "gate", step=len(turns) + 1)
                gthink, gverdict, gaxes, greason, gerr = parse_gate(graw)
                if gerr is None:
                    break
                gmsgs = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": gprompt + f"\n\nYour previous reply was invalid: {gerr}. "
                                "Answer again.",
                            }
                        ]
                        + ga["parts"],
                    }
                ]
            if gerr is not None:
                gverdict, gaxes = "pass", []
            env.note_gate(ga["assets"], gverdict)
            grew = gate_reward(gt_mismatch, gt_axes, gverdict, gaxes)
            gturn = {
                "turn": len(turns) + 1,
                "phase": "gate",
                "anchor": g_anchor,
                "think": gthink,
                "verb": "gate_verdict",
                "payload": gverdict + (f" | axes: {','.join(gaxes)}" if gaxes else ""),
                "observation": "verdict recorded",
                "reward": cfg.invalid_penalty if gerr is not None else 0.0,
                "gate": {
                    "assets": [r.asset_id for r in ga["assets"]],
                    "verdict": gverdict,
                    "axes": gaxes,
                    "reason": greason,
                    "gt_mismatch": gt_mismatch,
                    "gt_axes": gt_axes,
                    "reward": grew,
                },
            }
            if getattr(cfg, "deliver_mode", DELIVER_LEGACY) == DELIVER_CHUNKED:
                gturn["gate"]["batch"] = {
                    "k": g_k,
                    "K": g_K,
                    "order_turn": ga.get("order_turn"),
                    "ordered": list(ga["ordered"]),
                    "records": [[r.asset_id, r.name] for r in ga["assets"]],
                    "pending_asset_ids": [r.asset_id for e in env._dq() for _c, r in e["groups"]],
                    "delivery_obs": ga.get("delivery_obs"),
                }
            if cfg.rl_mode:
                gturn["raw"] = graw
            turns.append(gturn)
            if on_turn:
                on_turn(gturn)
            wk_steps.append(
                {
                    "verb": "integrity_check",
                    "payload_text": ", ".join(env.alias(r.asset_id) for r in ga["assets"]),
                    "observation_text": f"verdict: {gverdict}"
                    + (f" ({','.join(gaxes)}: {greason})" if gverdict == "mismatch" else ""),
                }
            )
            if gverdict != "mismatch" and env._dq():
                entry = env._dq().pop(0)
                d_obs, d_parts, d_assets = env.attach_batch(entry)
                wk_steps.append(
                    {
                        "verb": "results_delivered",
                        "payload_text": ", ".join(env.alias(r.asset_id) for r in d_assets),
                        "observation_text": d_obs,
                    }
                )
                pending_gate = {
                    "assets": d_assets,
                    "parts": d_parts,
                    "ordered": entry["ordered"],
                    "batch": (entry["batch"], g_K),
                    "order_turn": ga.get("order_turn"),
                    "pending_count": sum(len(e["groups"]) for e in env._dq()),
                    "delivery_obs": d_obs,
                }
                if not d_assets:
                    pending_gate = None
                continue
            if gverdict == "mismatch":
                env._dq().clear()
                final = {
                    "kind": "defer_to_human",
                    "reason": f"integrity check failed on "
                    f"{','.join(gaxes) or 'unspecified axes'}: "
                    f"{greason or 'evidence-order mismatch'}",
                    "gate": True,
                }
                break
            continue
        remaining_wk = cfg.max_workup_turns - wk_used
        if phase == "workup" and remaining_wk <= 0:
            phase = "conclude"
        must_end = phase in ("workup", "conclude") and remaining_wk <= 0
        if phase == "history":
            prompt = render_history_prompt(
                cfg, ep, deck, convo, asks_used, wk_used=wk_used, searches=env.searches
            )
        elif phase == "conclude":
            prompt = render_conclude_prompt(
                cfg, ep, deck, wk_steps, searches_done=env.searches, env=env, wk_used=wk_used
            )
        else:
            prompt = render_workup_prompt(
                cfg,
                ep,
                deck,
                wk_steps,
                remaining_wk,
                searches_done=env.searches,
                images_pending=len(pending_parts),
                env=env,
                wk_used=wk_used,
            )
        env.conclude_forfeits = conclude_forfeits
        t_anchor = anchor_key(phase, deck, env, asks_used, wk_used, None, mode=cfg.anchor_mode)
        content: Any = (
            prompt if not pending_parts else ([{"type": "text", "text": prompt}] + pending_parts)
        )
        messages = [{"role": "user", "content": content}]

        think = verb = ""
        payload: Any = None
        ok = False
        last_err: str | None = None
        coerced_from: str | None = None
        norm: tuple[str, ...] = ()
        for _ in range(cfg.policy_retries + 1):
            raw = doctor.complete(messages, ep.pmcid, phase, step=len(turns) + 1)
            pa = parse_doctor_ex(raw)
            think, verb, payload, err = pa.think, pa.verb, pa.payload, pa.error
            norm = pa.normalized
            coerced_from = None
            if (
                err is None
                and phase == "history"
                and verb in ("physical_examination", "order_tests", "search")
            ):
                if cfg.rl_mode:
                    err = (
                        "workup actions are unavailable during the interview; "
                        "use begin_workup to open the workup, then order."
                    )
                else:
                    coerced_from = verb + (
                        f": {', '.join(payload) if isinstance(payload, list) else payload}"
                        if payload
                        else ""
                    )
                    verb, payload = "begin_workup", None
            if (
                err is None
                and cfg.identity_anchor
                and not cfg.rl_mode
                and not think.lstrip().startswith(f"Patient: {env.patient_tag}")
            ):
                err = (
                    f"begin your <reasoning> with exactly "
                    f"'Patient: {env.patient_tag} "
                    f"({env.ep.demographics or 'demographics not recorded'})'"
                    " and verify results against this identity"
                )
            notes: list[str] = []
            dropped_left = pa.dropped
            if (
                err is None
                and verb in LIST_VERBS
                and (payload or pa.dropped)
                and getattr(cfg, "rule_violation", "forfeit") == "noop"
                and phase != "history"
            ):
                payload, notes = env.prepare_list(verb, payload, pa.dropped)
                dropped_left = ()
                if not payload:
                    if not [x for x in pa.dropped if "[" not in x and "]" not in x] and pa.dropped:
                        err = "template echo: " + "; ".join(notes)
                    else:
                        err = "nothing to execute: " + "; ".join(notes)
            if err is None:
                err = env.validate(
                    verb, payload, phase, asks_used, must_end, wk_used, dropped=dropped_left
                )
            if err is None:
                ok = True
                break
            last_err = err
            stats["doctor_retries"] += 1
            messages += [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": f"Rejected: {err}. Output your <reasoning> and one "
                    "corrected <action> only.",
                },
            ]
        if not ok:
            stats["doctor_fallbacks"] += 1
            fclass = forfeit_class(last_err)
            penalized = fclass == "format" or getattr(cfg, "rule_violation", "forfeit") != "noop"
            if not must_end:
                obs = (
                    forfeit_observation(last_err)
                    if penalized
                    else noop_observation(last_err, getattr(cfg, "refusal_text", "terse"))
                )
                turn = {
                    "turn": len(turns) + 1,
                    "phase": phase,
                    "anchor": t_anchor,
                    "think": think or "",
                    "verb": "invalid",
                    "payload": str(payload) if payload not in (None, []) else "",
                    "fallback": True,
                    "error": last_err,
                    "reward": cfg.invalid_penalty if penalized else 0.0,
                    "observation": obs,
                    "forfeit_class": fclass,
                }
                if not penalized:
                    turn["noop"] = True
                if cfg.rl_mode:
                    turn["raw"] = raw
                if norm:
                    turn["normalized"] = list(norm)
                if coerced_from:
                    turn["coerced_from"] = coerced_from
                if phase == "history" and asks_used >= cfg.max_asks:
                    phase = "workup"
                if phase == "history":
                    asks_used += 1
                    turn["remaining"] = cfg.max_asks - asks_used
                    convo.append({"role": "env", "text": obs})
                else:
                    wk_used += 1
                    turn["remaining"] = cfg.max_workup_turns - wk_used
                    wk_steps.append(
                        {"verb": "invalid", "payload_text": "", "observation_text": obs}
                    )
                turns.append(turn)
                if on_turn:
                    on_turn(turn)
                pending_parts = []
                continue
            if (
                phase == "conclude"
                and cfg.conclude_mode == "conclude_retry"
                and conclude_forfeits == 0
            ):
                obs = (
                    forfeit_observation(last_err)
                    if penalized
                    else noop_observation(last_err, getattr(cfg, "refusal_text", "terse"))
                )
                turn = {
                    "turn": len(turns) + 1,
                    "phase": phase,
                    "anchor": t_anchor,
                    "think": think or "",
                    "verb": "invalid",
                    "payload": str(payload) if payload not in (None, []) else "",
                    "fallback": True,
                    "error": last_err,
                    "reward": cfg.invalid_penalty if penalized else 0.0,
                    "observation": obs,
                    "remaining": 0,
                    "conclude_retry": True,
                    "forfeit_class": fclass,
                }
                if not penalized:
                    turn["noop"] = True
                if cfg.rl_mode:
                    turn["raw"] = raw
                if norm:
                    turn["normalized"] = list(norm)
                conclude_forfeits += 1
                wk_steps.append({"verb": "invalid", "payload_text": "", "observation_text": obs})
                turns.append(turn)
                if on_turn:
                    on_turn(turn)
                pending_parts = []
                continue
            think, payload = think or "", "undetermined"
            verb = "final_diagnosis"

        turn: dict = {
            "turn": len(turns) + 1,
            "phase": phase,
            "anchor": t_anchor,
            "think": think,
            "verb": verb,
            "payload": payload if isinstance(payload, list) else str(payload),
            "fallback": not ok,
            "reward": 0.0 if ok else cfg.invalid_penalty,
        }
        if cfg.rl_mode:
            turn["raw"] = raw
        if not ok:
            turn["error"] = last_err
        if norm:
            turn["normalized"] = list(norm)
        if coerced_from:
            turn["coerced_from"] = coerced_from
        pending_parts = []

        if verb == "ask":
            asks_used += 1
            question = str(payload)
            repeat_of = None
            if getattr(cfg, "rule_violation", "forfeit") == "noop" and env.asked:
                scores = [ask_overlap(question, [ws]) for ws in env.asked]
                best = max(range(len(scores)), key=lambda i: scores[i])
                if scores[best] >= 0.5:
                    repeat_of = best
            env.note_ask(question)
            convo.append({"role": "doctor", "text": question})
            if repeat_of is not None:
                doc_pos = [i for i, e in enumerate(convo) if e.get("role") == "doctor"]
                prev = ""
                if repeat_of < len(doc_pos) - 1:
                    for e in convo[doc_pos[repeat_of] + 1 :]:
                        if e.get("role") == "patient":
                            prev = str(e.get("text") or "")
                            break
                reply = {
                    "answer": f"I already answered that: {prev}"
                    if prev
                    else "I already answered that.",
                    "used_fact_ids": [],
                    "unknown_topics": [],
                    "repeat_of": repeat_of,
                }
                turn["rule_noop"] = "repeat_question"
            else:
                reply = ask_patient(cfg, ep, patient, convo, question, stats)
            new_ids, new_unknown = apply_patient_turn(
                deck, ep, reply["used_fact_ids"], reply["unknown_topics"]
            )
            convo.append({"role": "patient", "text": reply["answer"]})
            turn["reward"] = round(cfg.reward_info_per_fact * len(new_ids), 4)
            turn.update(
                remaining=cfg.max_asks - asks_used,
                patient=reply,
                new_fact_ids=new_ids,
                new_unknown_topics=new_unknown,
            )
        elif verb in ("final_diagnosis", "defer_to_human"):
            text = str(payload)
            if verb == "final_diagnosis":
                primary, alts = text.strip(), []
                final = {
                    "kind": "final_diagnosis",
                    "diagnosis": primary,
                    "alternates": [a for a in alts if a],
                    "think": think,
                    "source": "model" if ok else "synthesized",
                }
            else:
                final = {
                    "kind": "defer_to_human",
                    "reason": text,
                    "think": think,
                    "source": "model",
                }
            turn.update(remaining=remaining_wk)
        else:
            if verb == "begin_workup":
                phase = "workup"
                turn.update(
                    observation="Interview closed. The examination and test "
                    "menus follow next turn.",
                    remaining=cfg.max_workup_turns - wk_used,
                )
                turns.append(turn)
                if on_turn:
                    on_turn(turn)
                continue
            if phase == "history":
                phase = "workup"
            wk_used += 1
            obs_text, parts, extras = env.execute(verb, payload)
            if notes:
                obs_text = obs_text + "\n" + "\n".join(f"[{n}]" for n in notes)
                turn["notes"] = list(notes)
            gate_assets = extras.pop("gate_assets", [])
            ordered_items = extras.pop("ordered_items", [])
            if gate_assets:
                pending_gate = {
                    "assets": gate_assets,
                    "parts": list(parts),
                    "ordered": ordered_items,
                    "batch": (1, int(extras.get("batches_total", 1))),
                    "order_turn": turn["turn"],
                    "pending_count": int(extras.get("pending_count", 0)),
                    "delivery_obs": None,
                }
                pending_parts = []
            else:
                pending_parts = parts
            turn.update(remaining=cfg.max_workup_turns - wk_used, observation=obs_text, **extras)
            wk_steps.append(
                {
                    "verb": verb,
                    "payload_text": ", ".join(payload)
                    if isinstance(payload, list)
                    else str(payload),
                    "observation_text": obs_text,
                }
            )
            while pending_gate is None and env._dq():
                entry = env._dq().pop(0)
                d_obs, d_parts, d_assets = env.attach_batch(entry)
                wk_steps.append(
                    {
                        "verb": "results_delivered",
                        "payload_text": ", ".join(env.alias(r.asset_id) for r in d_assets),
                        "observation_text": d_obs,
                    }
                )
                if d_assets:
                    pending_gate = {
                        "assets": d_assets,
                        "parts": d_parts,
                        "ordered": entry["ordered"],
                        "batch": (entry["batch"], int(extras.get("batches_total", 1))),
                        "order_turn": turn["turn"],
                        "pending_count": sum(len(e["groups"]) for e in env._dq()),
                        "delivery_obs": d_obs,
                    }
                    pending_parts = []
        turns.append(turn)
        if on_turn:
            on_turn(turn)

    by_verb: dict[str, int] = {}
    for t in turns:
        by_verb[t["verb"]] = by_verb.get(t["verb"], 0) + 1
    revealed, total = len(deck["known_facts"]), len(ep.fact_index)
    return {
        "schema": "trustmed/consult",
        "meta": {"pmcid": ep.pmcid},
        "speaker_role": ep.speaker_role,
        "models": {"doctor": doctor.model, "patient": patient.model},
        "budgets": {
            "max_asks": cfg.max_asks,
            "min_asks": cfg.min_asks,
            "allow_history_final": cfg.allow_history_final,
            "allow_defer": cfg.allow_defer,
            "integrity_hint": cfg.integrity_hint,
            "corrupt_seed": cfg.corrupt_seed,
            "corrupt_rate": cfg.corrupt_rate,
            "prompt_sha": PROMPT_SHA,
            "gate_bank_dir": (str(cfg.gate_bank_dir) if cfg.gate_bank_dir else None),
            "panel_assets": (str(cfg.panel_assets) if cfg.panel_assets else None),
            "panel_assets_dir": (str(cfg.panel_assets_dir) if cfg.panel_assets else None),
            "panel_assets_sha1": (
                load_panel_assets(cfg.panel_assets)["sha1"] if cfg.panel_assets else None
            ),
            "panel_policy": (cfg.panel_policy if cfg.panel_assets else None),
            "corrupt_axis": cfg.corrupt_axis,
            "post_baseline": cfg.post_baseline,
            "identity_anchor": cfg.identity_anchor,
            "max_searches": cfg.max_searches,
            "min_workup_actions": cfg.min_workup_actions,
            "max_workup_turns": cfg.max_workup_turns,
            "max_items_per_action": cfg.max_items_per_action,
            "anchor_mode": cfg.anchor_mode,
            "deliver_mode": cfg.deliver_mode,
            "deliver_chunk": cfg.deliver_chunk,
            "max_batches_per_order": cfg.max_batches_per_order,
            "local_penalty": bool(cfg.local_penalty),
            "conclude_mode": cfg.conclude_mode,
            "rule_violation": getattr(cfg, "rule_violation", "forfeit"),
            "refusal_text": getattr(cfg, "refusal_text", "terse"),
            **({"no_dx_final": "reject"} if no_dx_contract(cfg) == "reject" else {}),
            "search_enabled": bool(cfg.search_url),
            "asset_aliases": dict(env.asset_aliases),
        },
        "opening": opening,
        "turns": turns,
        "final": final,
        "deck_final": deck,
        "corruption": ep.corruption,
        "metrics": {
            "facts_total": total,
            "facts_revealed": revealed,
            "fact_recall": round(revealed / total, 4) if total else 0.0,
            "asks_used": asks_used,
            "workup_turns": wk_used,
            "conclude_turns": sum(1 for t in turns if t.get("phase") == "conclude"),
            "conclude_forfeits": conclude_forfeits,
            "no_dx_forfeits": sum(1 for t in turns if t.get("error") == NO_DX_ERR_TAG),
            "actions_by_verb": by_verb,
            "invalid_turns": sum(1 for t in turns if t["verb"] == "invalid"),
            "format_forfeits": sum(
                1 for t in turns if t["verb"] == "invalid" and float(t.get("reward", 0.0)) < 0
            ),
            "rule_noops": sum(1 for t in turns if t.get("noop")),
            "partial_lists": sum(1 for t in turns if t.get("notes")),
            "repeat_answers": sum(1 for t in turns if t.get("rule_noop") == "repeat_question"),
            "normalized_turns": sum(1 for t in turns if t.get("normalized")),
            "terminal_normalized": list(turns[-1].get("normalized") or []) if turns else [],
            "step_reward_sum": round(sum(t.get("reward", 0.0) for t in turns), 2),
            "assets_injected": env.assets_injected,
            "reports_released": len(env.released),
            "consults_expected": sorted(
                {specialist_of(r.l1, r.category) for r in ep.records if r.asset_id}
            ),
            "records_expected": sum(1 for r in ep.records if r.asset_id),
            "panel_serving": _panel_serving_counts(ep.records),
            **bind_telemetry(ep, env),
            "consults_done": sorted(env.consulted),
            "gate_turns": sum(1 for t in turns if t.get("phase") == "gate"),
            "delivery_batches": int(getattr(env, "delivery_batches", 0)),
            "pending_max": int(getattr(env, "pending_max", 0)),
            "gate_reward_sum": round(
                sum(t["gate"]["reward"] for t in turns if t.get("phase") == "gate"), 2
            ),
            "gate_correct": sum(
                1
                for t in turns
                if t.get("phase") == "gate"
                and ((t["gate"]["verdict"] == "mismatch") == t["gate"]["gt_mismatch"])
            ),
            "corruption_armed": ep.corruption_armed,
            "corruption_exposed": bool(ep.corruption)
            and any(r.asset_id == ep.corruption["asset_id"] for r in env.returned_assets),
            "outcome": final["kind"],
            "final_answer": final.get("diagnosis") or final.get("reason", ""),
            "think_coverage": round(sum(1 for t in turns if t["think"]) / max(1, len(turns)), 3),
            "wall_seconds": round(time.time() - t0, 1),
            **stats,
        },
    }


def summarize(traj_path: Path) -> dict:
    recs = [json.loads(l) for l in traj_path.open(encoding="utf-8")] if traj_path.exists() else []
    recs = [r for r in recs if r.get("schema") == "trustmed/consult"]
    if not recs:
        return {"episodes": 0}
    m = [r["metrics"] for r in recs]
    n = len(recs)
    verbs: dict[str, int] = {}
    for x in m:
        for k, v in x["actions_by_verb"].items():
            verbs[k] = verbs.get(k, 0) + v
    outcomes: dict[str, int] = {}
    for x in m:
        outcomes[x["outcome"]] = outcomes.get(x["outcome"], 0) + 1
    return {
        "episodes": n,
        "avg_fact_recall": round(sum(x["fact_recall"] for x in m) / n, 4),
        "avg_asks": round(sum(x["asks_used"] for x in m) / n, 2),
        "avg_workup_turns": round(sum(x["workup_turns"] for x in m) / n, 2),
        "actions_by_verb": verbs,
        "outcomes": outcomes,
        "corrupted_episodes": sum(1 for r in recs if r.get("corruption")),
        "avg_think_coverage": round(sum(x["think_coverage"] for x in m) / n, 3),
        "fallbacks": {
            "doctor": sum(x["doctor_fallbacks"] for x in m),
            "patient": sum(x["patient_fallbacks"] for x in m),
        },
    }


def done_key(rec: dict) -> str:
    """Resume identity of one written record: a case AND which rollout of it."""
    return f"{rec['meta']['pmcid']}#{rec['meta'].get('rollout_idx', 0)}"


def main() -> None:
    d = {f.name: f.default for f in dataclasses.fields(Config)}
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--profiles", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--doctor-model", default=d["doctor_model"])
    p.add_argument("--patient-model", default=d["patient_model"])
    p.add_argument("--doctor-base-url", default=d["doctor_base_url"])
    p.add_argument("--patient-base-url", default=d["patient_base_url"])
    p.add_argument("--service-tier", default=d["service_tier"])
    p.add_argument("--doctor-temperature", type=float, default=d["doctor_temperature"])
    p.add_argument("--max-asks", type=int, default=d["max_asks"])
    p.add_argument("--max-workup-turns", type=int, default=d["max_workup_turns"])
    p.add_argument(
        "--images-dir",
        type=Path,
        default=d["images_dir"],
        help="whole-figure tree. Under --panel-assets this is only "
        "reached by a figure the registry did not bind, and "
        "only when --panel-policy allows it",
    )
    p.add_argument(
        "--panel-assets",
        type=Path,
        default=d["panel_assets"],
        help=f"frozen panel manifest (default off; the built one is "
        f"{DEFAULT_PANEL_ASSETS}). With it set, every BOUND "
        f"record is served ITS OWN single-panel crop under its "
        f"own asset id '<image_id>#<panel_id>' — so the gate, "
        f"the alias, the attach dedupe and a corruption swap "
        f"all operate per panel. Needs --prompt-contract 3",
    )
    p.add_argument(
        "--panel-assets-dir",
        type=Path,
        default=d["panel_assets_dir"],
        help="crop tree the manifest's relative crop_path resolves "
        f"against (default {DEFAULT_PANEL_DIR})",
    )
    p.add_argument(
        "--panel-policy",
        choices=PANEL_POLICIES,
        default=d["panel_policy"],
        help="what to do with a record whose figure the registry "
        "did not bind. whole_figure (default): serve the "
        "composite plate, the historical fail-open. "
        "crops_only: never serve an uncropped plate — withhold "
        "the picture and release the study's written report "
        "instead. strict: refuse to run until every referenced "
        "panel has a crop",
    )
    p.add_argument(
        "--panel-coverage",
        action="store_true",
        help="report the crop coverage of the selected cases and "
        "exit, running no episodes. What tells you whether a "
        "crop delivery is complete enough for --panel-policy "
        "strict",
    )
    p.add_argument("--search-url", default=d["search_url"])
    p.add_argument(
        "--min-asks",
        type=int,
        default=d["min_asks"],
        help="history-floor ablation / legacy-contract knob "
        "(default 0 = no floor, the RL and eval contract; "
        "teacher SFT manifests pin 3)",
    )
    p.add_argument(
        "--max-searches",
        type=int,
        default=d["max_searches"],
        help="search-action ceiling per episode (default 3). A "
        "ScalingInter phase schedule passes the per-stage "
        "value; env-enforced, recorded in budgets",
    )
    p.add_argument(
        "--min-workup-actions",
        type=int,
        default=d["min_workup_actions"],
        help="workup actions required before final_diagnosis "
        "(0 = off, the default and the value every RL "
        "collection uses). A collection-time forcing function "
        "for teacher SFT: it gates the DIAGNOSIS only — never "
        "defer_to_human — and the turn cap always releases it",
    )
    p.add_argument("--corrupt-rate", type=float, default=d["corrupt_rate"])
    p.add_argument(
        "--corrupt-seed",
        default=d["corrupt_seed"],
        help="iteration token; vary per training iteration, keep fixed within a rollout group",
    )
    p.add_argument(
        "--no-integrity-hint",
        dest="integrity_hint",
        action="store_false",
        help='Ablation: drop the "verify every result belongs '
        'to THIS patient and THIS order" line from the workup '
        "rules, so a mismatch catch measures internalized "
        "vigilance instead of instruction-following",
    )
    p.add_argument("--gate-bank-dir", type=Path, default=d["gate_bank_dir"])
    p.add_argument(
        "--corrupt-axis",
        choices=GATE_AXES,
        default=d["corrupt_axis"],
        help="target ONE axis for this run's corruptions: the bank "
        "is restricted to cells that move it (modality / "
        "anatomy). Default: the unconstrained draw",
    )
    p.add_argument("--allow-defer", action="store_true")
    p.add_argument(
        "--no-history-final",
        dest="allow_history_final",
        action="store_false",
        help="contract 3: withdraw final_diagnosis from the HISTORY "
        "menu, so the episode cannot early-stop before the "
        "workup. What B5/B6 collect under — their defining "
        "event (an abstention worth making, a corruption worth "
        "catching) only exists after the workup opens",
    )
    p.add_argument(
        "--post-baseline",
        action="store_true",
        help="serve Post_Baseline_Findings as orderable records "
        "under their own sections, names tagged "
        "(post-treatment). Default off; recorded in budgets "
        "so replay follows the collection",
    )
    p.add_argument(
        "--rl-mode",
        action="store_true",
        help="rollout collection: no in-context retries for the "
        "policy; an invalid action forfeits the turn",
    )
    p.add_argument(
        "--anchor-mode",
        choices=list(ANCHOR_MODES),
        default=ANCHOR_MODE_DEFAULT,
        help="GiGPO step-anchor scheme; recorded in budgets as anchor_mode",
    )
    p.add_argument(
        "--conclude-mode",
        choices=list(CONCLUDE_MODES),
        default=d["conclude_mode"],
        help="the CONCLUDE turn once the workup budget is spent: "
        "conclude (terminals-only prompt; a non-terminal is "
        "synthesized at once) or conclude_retry (one more "
        "sample after such a forfeit, the default); recorded "
        "in budgets as conclude_mode",
    )
    p.add_argument(
        "--rule-violation",
        choices=list(RULE_VIOLATION_MODES),
        default=d["rule_violation"],
        help="what a WELL-FORMED action the rules refuse earns "
        "(GiGPO-style leniency): forfeit = the "
        "legacy env (turn lost, invalid_penalty); noop = the "
        "turn is spent, reward 0, the reason is the observation, "
        "a list executes its admissible part, a repeated "
        "question gets the earlier answer back; recorded in "
        "budgets as rule_violation",
    )
    p.add_argument(
        "--refusal-text",
        choices=list(REFUSAL_TEXT_MODES),
        default=d["refusal_text"],
        help="wording of a no-op turn's observation under "
        "--rule-violation noop: terse (the fact, one clause) or "
        "full (the validator's words); recorded in budgets as "
        "refusal_text",
    )
    p.add_argument(
        "--freeze-opening",
        action="store_true",
        help="cache the opening statement per (pmcid, corrupt-seed) "
        "so a GiGPO episode group shares one s1 (implied by "
        "--rl-mode)",
    )
    p.add_argument(
        "--rollouts",
        type=int,
        default=d["rollouts"],
        help="rollouts per case (a GiGPO episode group); >1 needs "
        "--freeze-opening or --rl-mode to share s1",
    )
    p.add_argument("--limit", type=int, default=d["limit"])
    p.add_argument("--offset", type=int, default=d["offset"])
    p.add_argument("--pmcids", nargs="*", default=[])
    p.add_argument("--workers", type=int, default=d["workers"])
    a = p.parse_args()
    cfg = Config(
        profiles=a.profiles,
        out=a.out,
        doctor_model=a.doctor_model,
        patient_model=a.patient_model,
        doctor_base_url=a.doctor_base_url or None,
        patient_base_url=a.patient_base_url or None,
        service_tier=a.service_tier or None,
        doctor_temperature=a.doctor_temperature,
        min_asks=a.min_asks,
        max_asks=a.max_asks,
        max_workup_turns=a.max_workup_turns,
        images_dir=a.images_dir,
        search_url=a.search_url or None,
        panel_assets=a.panel_assets,
        panel_policy=a.panel_policy,
        panel_assets_dir=a.panel_assets_dir,
        max_searches=a.max_searches,
        min_workup_actions=a.min_workup_actions,
        corrupt_rate=a.corrupt_rate,
        corrupt_seed=str(a.corrupt_seed),
        gate_bank_dir=a.gate_bank_dir,
        corrupt_axis=a.corrupt_axis,
        allow_history_final=a.allow_history_final,
        post_baseline=a.post_baseline,
        allow_defer=a.allow_defer,
        integrity_hint=a.integrity_hint,
        rl_mode=a.rl_mode,
        anchor_mode=a.anchor_mode,
        conclude_mode=a.conclude_mode,
        rule_violation=a.rule_violation,
        refusal_text=a.refusal_text,
        freeze_opening=a.freeze_opening,
        rollouts=a.rollouts,
        limit=a.limit,
        offset=a.offset,
        pmcids=tuple(a.pmcids),
        workers=a.workers,
    )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if cfg.panel_policy not in PANEL_POLICIES:
        raise SystemExit(f"--panel-policy {cfg.panel_policy!r} is not one of {PANEL_POLICIES}")
    if cfg.panel_assets is not None and not Path(cfg.panel_assets).is_file():
        raise SystemExit(f"--panel-assets {cfg.panel_assets} does not exist")
    cfg.state.mkdir(parents=True, exist_ok=True)
    traj_path = cfg.out / "trajectories.jsonl"
    usage_path = cfg.state / "usage.jsonl"

    done: set[str] = set()
    if traj_path.exists():
        no_dx_resume_preflight(traj_path, cfg)
        done = {done_key(json.loads(l)) for l in traj_path.open(encoding="utf-8")}

    rows = load_profiles(cfg.profiles)
    swap_pool = build_swap_pool(rows, cfg) if cfg.corrupt_rate > 0 else []
    if cfg.pmcids:
        rows = [r for r in rows if r["pmcid"] in cfg.pmcids]
    if a.panel_coverage:
        sel = rows[cfg.offset :][: cfg.limit] if cfg.limit is not None else rows[cfg.offset :]
        if cfg.panel_assets is None:
            raise SystemExit(
                "--panel-coverage needs --panel-assets "
                f"(the built manifest is {DEFAULT_PANEL_ASSETS})"
            )
        print(json.dumps(panel_coverage(sel, cfg), indent=2))
        return
    rows = [
        r
        for r in rows[cfg.offset :]
        if any(f"{r['pmcid']}#{i}" not in done for i in range(cfg.rollouts))
    ]
    if cfg.limit is not None:
        rows = rows[: cfg.limit]
    work = [(r, i) for r in rows for i in range(cfg.rollouts) if f"{r['pmcid']}#{i}" not in done]
    LOG.info("%d episodes to run (%d already done)", len(work), len(done))
    if cfg.panel_assets is not None:
        panel_preflight(rows, cfg)
    elif any(r.get("images") for r in rows[:50]):
        LOG.warning(
            "no --panel-assets: imaging is served as WHOLE FIGURES "
            "(composite plates). Pass --panel-assets %s "
            "--panel-policy crops_only to serve single-panel crops",
            DEFAULT_PANEL_ASSETS,
        )

    lock = threading.Lock()

    def append(path: Path, obj: dict) -> None:
        with lock, path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def log_usage(u: dict) -> None:
        append(usage_path, u)

    doctor = LLMAgent(
        cfg, "doctor", cfg.doctor_model, cfg.doctor_base_url, cfg.doctor_temperature, log_usage
    )
    patient = LLMAgent(
        cfg,
        "patient",
        cfg.patient_model,
        cfg.patient_base_url,
        cfg.patient_temperature,
        log_usage,
        json_mode=True,
    )

    if cfg.rollouts > 1 and not (cfg.freeze_opening or cfg.rl_mode):
        LOG.warning(
            "--rollouts %d without --freeze-opening/--rl-mode: each "
            "rollout draws its own opening statement, so the episode "
            "groups will not form",
            cfg.rollouts,
        )
    if (cfg.freeze_opening or cfg.rl_mode) and cfg.rollouts > 1:
        warm = {
            k: 0
            for k in ("doctor_retries", "doctor_fallbacks", "patient_retries", "patient_fallbacks")
        }
        for row in rows:
            load_or_make_opening(cfg, parse_profile(row, cfg, swap_pool), patient, [], warm)
        LOG.info(
            "pre-warmed %d frozen opening(s) (%d patient retries, %d fallbacks)",
            len(rows),
            warm["patient_retries"],
            warm["patient_fallbacks"],
        )

    def one(item: tuple[dict, int]) -> None:
        row, idx = item
        try:
            rec = run_episode(cfg, row, doctor, patient, swap_pool)
        except Exception:
            LOG.exception("%s failed", row.get("pmcid"))
            return
        rec["meta"]["rollout_idx"] = idx
        append(traj_path, rec)
        LOG.info(
            "%s done: recall=%.2f asks=%d workup=%d -> %s: %s",
            done_key(rec),
            rec["metrics"]["fact_recall"],
            rec["metrics"]["asks_used"],
            rec["metrics"]["workup_turns"],
            rec["metrics"]["outcome"],
            rec["metrics"]["final_answer"][:60],
        )

    with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
        list(ex.map(one, work))

    summary = summarize(traj_path)
    (cfg.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOG.info("summary: %s", json.dumps(summary))


if __name__ == "__main__":
    main()
