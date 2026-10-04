"""Deterministic lexicons for the OSCE action-space taxonomy."""

from __future__ import annotations

import re
import unicodedata


_BRITISH = [
    (r"haemat", "hemat"),
    (r"haemo", "hemo"),
    (r"haem", "hem"),
    (r"tumour", "tumor"),
    (r"oesophag", "esophag"),
    (r"oedema", "edema"),
    (r"anaemi", "anemi"),
    (r"leucocyt", "leukocyt"),
    (r"leukaemi", "leukemi"),
    (r"paediatr", "pediatr"),
    (r"coeliac", "celiac"),
    (r"caecal", "cecal"),
    (r"caecum", "cecum"),
    (r"foetal", "fetal"),
    (r"diarrhoea", "diarrhea"),
    (r"aetiolog", "etiolog"),
    (r"orthopaedic", "orthopedic"),
    (r"gynaecolog", "gynecolog"),
    (r"colour", "color"),
    (r"fibre", "fiber"),
    (r"centre", "center"),
    (r"litre", "liter"),
    (r"catheterisation", "catheterization"),
    (r"visualis", "visualiz"),
    (r"characteris", "characteriz"),
    (r"analys(is|es)\b", r"analys\1"),
    (r"ultrasonograph", "ultrasound"),
    (r"ultrasonic", "ultrasound"),
    (r"sonograph", "ultrasound"),
    (r"echograph", "ultrasound"),
]

_QUAL = r"""(?:initial|repeat|repeated|follow[-\s]?up|followup|admission|
on\ admission|at\ admission|at\ presentation|on\ presentation|presenting|
pre[-\s]?operative|preoperative|post[-\s]?operative|postoperative|
intra[-\s]?operative|intraoperative|peri[-\s]?operative|
baseline|subsequent|serial|second|third|fourth|fifth|further|additional|
urgent|emergent|routine|emergency|first|later|earlier|early|outside|
referral|prior|previous|current|new|recent|initial\ workup|
day\ \d+|week\ \d+|month\ \d+|year\ \d+|\d+[-\s]?(?:day|week|month|year)s?|
patient\ \d+|case\ \d+|subject\ \d+|individual\ \d+|proband|
sibling|mother|father|index\ patient|our\ patient|the\ patient's)"""

_WS = re.compile(r"\s+")
_EDGE = re.compile(r"^[\s\-–—:;,.•'\"]+|[\s\-–—:;,.•'\"]+$")
_PAREN = re.compile(r"\s*\([^)]*\)|\s*\[[^\]]*\]")
_QLEAD = re.compile(r"^(?:" + _QUAL + r")[\s,\-]+", re.I | re.X)
_QTRAIL = re.compile(r"[\s,\-]+(?:" + _QUAL + r")$", re.I | re.X)


def normalise(s: str) -> str:
    """Case/spelling/qualifier-invariant surface form. Deterministic."""
    s = unicodedata.normalize("NFKC", s or "")
    s = s.replace("’", "'").replace("–", "-").replace("—", "-").replace("×", "x")
    s = _PAREN.sub(" ", s)
    s = _WS.sub(" ", s).strip().lower()
    s = _EDGE.sub("", s)
    s = re.sub(r"(?<=[a-z0-9])-(?=[a-z0-9])", " ", s)
    for a, b in _BRITISH:
        s = re.sub(a, b, s)
    for _ in range(4):
        t = _EDGE.sub("", _QTRAIL.sub("", _QLEAD.sub("", s)))
        if t == s:
            break
        s = t
    return _WS.sub(" ", s).strip()


def _rx(pairs):
    """Compile an ordered (pattern, label) rule list."""
    return [(re.compile(p, re.I), lab) for p, lab in pairs]


def apply_rules(rules, text, default=None):
    for pat, lab in rules:
        if pat.search(text):
            return lab
    return default


PE_SECTION_FINE = {
    "vital_signs": "Vital_Signs",
    "general_examination": "General",
    "general_examination_findings": "General",
    "general_examination_additional": "General",
    "general_examination_puberty": "General",
    "general_examination_digital_rectal": "Rectal",
    "general_examination_anal_mucosa": "Rectal",
    "peripheral_examination": "General",
    "dermatologic_examination": "Dermatologic",
    "nail_examination": "Dermatologic",
    "head_and_neck_examination": "Head_and_Neck",
    "neck_examination": "Head_and_Neck",
    "otolaryngologic_examination": "Head_and_Neck",
    "otorhinolaryngologic_examination": "Head_and_Neck",
    "nasal_examination": "Head_and_Neck",
    "ear_examination": "Head_and_Neck",
    "otologic_examination": "Head_and_Neck",
    "temporomandibular_joint_examination": "Head_and_Neck",
    "oropharyngeal_examination": "Oropharyngeal",
    "oral_examination": "Oral_Dental",
    "dental_examination": "Oral_Dental",
    "cardiovascular_examination": "Cardiovascular",
    "vascular_examination": "Cardiovascular",
    "peripheral_vascular_examination": "Cardiovascular",
    "respiratory_examination": "Respiratory",
    "chest_examination": "Respiratory",
    "abdominal_examination": "Abdominal",
    "abdominal_examination_findings": "Abdominal",
    "gastrointestinal_examination": "Abdominal",
    "neurologic_examination": "Neurologic",
    "sensory_examination": "Neurologic",
    "ophthalmologic_examination": "Ophthalmologic",
    "musculoskeletal_examination": "Musculoskeletal",
    "extremity_examination": "Musculoskeletal",
    "hand_examination": "Musculoskeletal",
    "spine_examination": "Musculoskeletal",
    "rheumatologic_examination": "Musculoskeletal",
    "breast_examination": "Breast",
    "psychiatric_examination": "Psychiatric",
    "lymphatic_examination": "Lymphatic",
    "genitourinary_examination": "Genitourinary",
    "genital_examination": "Genitourinary",
    "testicular_examination": "Genitourinary",
    "gynecologic_examination": "Gynecologic_Obstetric",
    "gynecological_examination": "Gynecologic_Obstetric",
    "obstetric_examination": "Gynecologic_Obstetric",
    "pelvic_examination": "Gynecologic_Obstetric",
    "rectal_examination": "Rectal",
    "anorectal_examination": "Rectal",
    "perirectal_examination": "Rectal",
}

PE_FINE_TO_COARSE = {
    "Vital_Signs": "Vital_Signs",
    "General": "General_Examination",
    "Dermatologic": "Dermatologic_Examination",
    "Head_and_Neck": "Head_and_Neck_Examination",
    "Oral_Dental": "Oropharyngeal_Examination",
    "Oropharyngeal": "Oropharyngeal_Examination",
    "Cardiovascular": "Cardiovascular_Examination",
    "Respiratory": "Respiratory_Examination",
    "Abdominal": "Abdominal_Examination",
    "Rectal": "Abdominal_Examination",
    "Neurologic": "Neurologic_Examination",
    "Ophthalmologic": "Ophthalmologic_Examination",
    "Musculoskeletal": "Musculoskeletal_Examination",
    "Breast": "Breast_Examination",
    "Psychiatric": "Psychiatric_Examination",
    "Lymphatic": "General_Examination",
    "Genitourinary": "General_Examination",
    "Gynecologic_Obstetric": "General_Examination",
}

VITAL_CONCEPT = _rx(
    [
        (r"\bblood pressure\b|\bbp\b|systolic|diastolic|\bmap\b|mean arterial", "Blood pressure"),
        (r"heart rate|pulse rate|\bpulse\b|\bhr\b|tachycard|bradycard", "Heart rate / pulse"),
        (r"respirator(y|ion) rate|\brr\b|breathing rate|respirations", "Respiratory rate"),
        (r"temperature|\bfever\b|febrile|afebrile|pyrexia", "Body temperature"),
        (r"oxygen saturation|\bspo2\b|\bsao2\b|pulse ox|oxygen sat", "Oxygen saturation"),
        (r"\bbmi\b|body mass index", "Body mass index"),
        (r"\bweight\b|\bmass\b", "Body weight"),
        (r"\bheight\b|\bstature\b|\blength\b", "Body height / length"),
        (r"head circumference|occipitofrontal", "Head circumference"),
        (r"glasgow|\bgcs\b|consciousness level", "Glasgow coma scale"),
        (r"pain score|pain scale|\bvas\b|numeric rating", "Pain score"),
        (r"growth (chart|percentile)|percentile", "Growth percentile"),
        (r"capillary refill", "Capillary refill time"),
        (r"waist|hip circumference|circumference", "Body circumference"),
        (r"vital sign|hemodynamic|observation", "Vital signs (composite)"),
    ]
)

PE_REGION = _rx(
    [
        (
            r"^(?:patient|case|subject|individual|proband|sibling|family member)\s*\d*$|"
            r"^(?:patient|case|subject)\s*[a-z]$|^proband$|^mother$|^father$",
            "Unspecified (patient-level label)",
        ),
        (
            r"^vital signs?$|^vitals$|blood pressure|heart rate|\bpulse rate\b|"
            r"respiratory rate|\bbody temperature\b|oxygen saturation|\bspo2\b|"
            r"capillary refill|^temperature$|hemodynamic status|volume status|"
            r"hydration status|urine output|fluid status",
            "Vital signs measured at bedside",
        ),
        (
            r"body (measurement|weight|mass index|proportion|habitus|surface area|"
            r"composition)|anthropometr|^height$|^weight$|\bbmi\b|"
            r"head circumference|occipitofrontal|arm span|growth (parameter|percentile|chart)",
            "Anthropometry",
        ),
        (
            r"\bfundus|fundi\b|fundoscop|ophthalmoscop|retina|macula|"
            r"optic (disc|disk|nerve)|vitreous|choroid|posterior segment|posterior pole",
            "Eye - posterior segment",
        ),
        (
            r"\bcorne|conjunctiv|sclera|anterior (chamber|segment)|\biris\b|\blens\b|"
            r"slit.?lamp|eyelid|lacrimal|palpebral|gonioscop|\bpterygium\b",
            "Eye - anterior segment / adnexa",
        ),
        (r"\bpupil|light reflex|\brapd\b|accommodation", "Pupils"),
        (
            r"visual acuity|visual field|\bvision\b|color vision|ocular (motility|alignment|movement)|"
            r"extraocular|strabismus|nystagmus|\bsquint|intraocular pressure|tonometr|"
            r"refraction|cover test|\bdiplopia\b",
            "Vision / ocular motility",
        ),
        (r"\beyes?\b|\borbit|periorbital|proptosis|exophthalm|ocular", "Eye - general / orbit"),
        (r"cranial nerve", "Cranial nerves"),
        (
            r"mental (status|state)|orientation|cognition|cognitive|memory|attention|"
            r"\bmmse\b|\bpraxis\b|higher (mental|cortical) function|"
            r"speech|language|aphasi|dysarthr|bulbar|"
            r"consciousness|conscious level|\bgcs\b|glasgow",
            "Mental status / consciousness / speech",
        ),
        (
            r"motor (system|exam|strength|power|function)|muscle (strength|power|tone|bulk)|"
            r"\btone\b|\bpower\b|paresis|plegia|weakness",
            "Motor system",
        ),
        (r"reflex|plantar|babinski|clonus", "Reflexes"),
        (
            r"sensor(y|ium)|vibration|proprioception|pinprick|light touch|sensation",
            "Sensory system",
        ),
        (
            r"\bgait\b|\bstance\b|romberg|coordination|cerebell|ataxi|finger.?nose|"
            r"dysdiadocho|\bbalance\b|movement exam|involuntary movement",
            "Gait / coordination",
        ),
        (r"meningeal|neck stiffness|kernig|brudzinski|nuchal", "Meningeal signs"),
        (
            r"\bear\b|\bears\b|tympanic|auricle|otoscop|external auditory|hearing|mastoid",
            "Ear / hearing",
        ),
        (r"\bnose\b|\bnasal\b|nostril|septum|turbinate|rhino", "Nose / nasal cavity"),
        (
            r"oral (cavity|mucosa|hygiene)|\bmouth\b|\btongue\b|\bgingiv|\bteeth\b|"
            r"\btooth\b|dental|dentit|denture|occlusion|palate|buccal|lip[s]?\b|"
            r"mucosa of the mouth|periodont|\bcaries\b|\benamel\b",
            "Oral cavity / dentition",
        ),
        (
            r"orophar|phary|tonsil|uvula|\bthroat\b|larynx|laryng|vocal|\bvoice\b|hoarse",
            "Pharynx / larynx",
        ),
        (
            r"\bface\b|facial|\bjaw\b|mandib|maxill|temporomandibular|\bchin\b|malar|"
            r"\bcheek|forehead|\btemple\b|temporal region|glabella|\bnasolabial\b|"
            r"philtrum|eyebrow|\bparotid\b|submental|submandibular",
            "Face / jaw",
        ),
        (
            r"\bscalp\b|\bskull\b|cranium|\bhead\b|fontanel|craniofacial|occiput|\bvertex\b",
            "Head / scalp",
        ),
        (r"\bneck\b|cervical region|thyroid|\btrachea\b|carotid|salivary", "Neck / thyroid"),
        (
            r"\blung|pulmonar|\bchest\b|thora|breath sound|respirator|"
            r"ausculta.*(lung|chest)|percussion|pleura|\bairway\b",
            "Chest / lungs",
        ),
        (
            r"\bheart\b|cardiac|precordi|apex beat|apical impulse|\bmurmur\b|heart sound|"
            r"\bs1\b|\bs2\b|cardiovascular|sternal border|auscultation of the heart|"
            r"(aortic|mitral|tricuspid|pulmonary) area",
            "Heart / precordium",
        ),
        (r"jugular|\bjvp\b|central venous", "Jugular venous pressure"),
        (
            r"peripheral pulse|radial|femoral pulse|dorsalis pedis|posterior tibial|"
            r"\bartery\b|arterial|\bvein\b|venous|varicos|vascular|\bpulses\b",
            "Peripheral vasculature",
        ),
        (r"\bbreast|\bnipple|areola|axillary tail", "Breast"),
        (r"\bliver\b|hepat", "Liver"),
        (r"\bspleen\b|splen", "Spleen"),
        (r"\bkidney|renal (angle|area)|costovertebral", "Kidney / renal angle"),
        (
            r"\babdom|\bflank\b|epigastr|umbilic|iliac fossa|hypochond|\bgroin\b|"
            r"inguinal|\bhernia|quadrant|\bruq\b|\brlq\b|\bluq\b|\bllq\b|"
            r"\bbowel sounds\b|\bascites\b|gastrointestinal",
            "Abdomen",
        ),
        (r"rectal|rectum|\banus\b|anal|perianal|perineum|perineal", "Anorectal / perineum"),
        (
            r"\bpenis|penile|scrotum|scrotal|testis|testes|testicle|testicul|\bvulva|"
            r"vagina|cervix|uterus|adnexa|\bpelvis\b|\bpelvic\b|prostate|"
            r"external genital|genitalia",
            "External genitalia / pelvis",
        ),
        (r"\bbladder\b|suprapubic|urethra|urinar", "Bladder / urinary tract"),
        (
            r"\bnails?\b|toenail|fingernail|\bhair\b|alopecia|scalp lesion",
            "Skin appendages (hair / nails)",
        ),
        (
            r"\bskin\b|cutaneous|dermat|\brash\b|\blesion|papul|macul|nodul|plaque|"
            r"vesic|bull|ulcer|erythema|pigment|purpur|petechi|ecchymos|pruritus|"
            r"\bwound\b|\bscar\b|\btrunk\b|\bback of\b|mucous membrane|\bmucosa\b|"
            r"buttock|\bflexur",
            "Skin / cutaneous lesion",
        ),
        (r"lymph node|lymphaden|lymphatic|cervical node|axilla|inguinal node", "Lymph nodes"),
        (r"\bspine\b|spinal|vertebr|kyphos|scolios|\bback\b|lumbar region", "Spine / back"),
        (
            r"\bjoints?\b|arthr|\bknees?\b|\bhips?\b|\bshoulders?\b|\belbows?\b|"
            r"\bwrists?\b|\bankles?\b|metacarpo|interphalangeal|sacroiliac",
            "Joints",
        ),
        (r"\bmuscles?\b|myopath|\bcalf\b|\bthighs?\b|atrophy", "Muscles"),
        (
            r"\bhands?\b|\bfoot\b|\bfeet\b|\bfingers?\b|\bthumbs?\b|\btoes?\b|"
            r"\bdigits?\b|\blimbs?\b|"
            r"extremit|\barms?\b|\blegs?\b|upper limb|lower limb|\bpalms?\b|\bsoles?\b|"
            r"\bforearm|\bshin\b|\bcalves\b",
            "Limbs / extremities",
        ),
        (r"\bbones?\b|\bskeleton\b|\bskeletal|\bsternum\b|deformit|fractur", "Bones / skeletal"),
        (
            r"psychiatric|\bmood\b|affect|behavio|thought|hallucin|psycho",
            "Mental state / behaviour",
        ),
        (
            r"neurolog|neuromuscular|nervous system|\bcns\b|peripheral nerve",
            "Neurologic examination (composite)",
        ),
        (
            r"right side|left side|hemi(body|paresis|plegia)|ipsilateral|contralateral|"
            r"bilateral(?! .*\b(eye|hand|leg)\b)",
            "Lateralised body region",
        ),
        (
            r"\bgrowth\b|\bpuberty|pubertal|tanner|development|"
            r"secondary sex(ual)? characteristic",
            "Growth / development",
        ),
        (
            r"nutrition|cachexia|body habitus|\bbuild\b|\bfacies\b|dysmorph|"
            r"general (appearance|examination|condition|inspection)|\bgeneral\b|"
            r"systemic exam|other (examined )?system|physical exam|clinical exam|"
            r"whole body|overall|functional status|peripheral tissue|"
            r"\bbody\b|\bappearance\b|\bexamination\b",
            "General appearance / habitus",
        ),
    ]
)


LAB_CATEGORY = _rx(
    [
        (
            r"exome|genome sequencing|\bwes\b|\bwgs\b|sanger|\bmlpa\b|array.?cgh|"
            r"comparative genomic hybrid|karyotyp|cytogenet|chromosom|\bfish\b|"
            r"fluorescence in.?situ|microarray|snp array|gene panel|mutation|variant|"
            r"genetic test|genetic analys|molecular (genetic|test|analys|diagnos)|"
            r"\bgene\b|sequencing|pedigree|methylation|\bmlpa\b|copy number|"
            r"southern blot|linkage analys|allele|genotyp|\bdna\b|\brna seq",
            "Molecular genetics & cytogenetics",
        ),
        (
            r"flow cytometr|immunophenotyp|\bcd4\b.*count|cd4/cd8|lymphocyte subset",
            "Flow cytometry / immunophenotyping",
        ),
        (
            r"\bcultures?\b|gram stain|acid.?fast|\bafb\b|ziehl|\bsmear.*(malaria|parasit)|"
            r"microbiolog|sensitivit(y|ies) test|antibiogram|\bpcr\b.*(virus|viral|bacteri|"
            r"tubercul|sars|covid)|\bmantoux\b|tuberculin|\bigra\b|quantiferon|"
            r"\bhiv\b|hepatitis|\bhbsag\b|\banti.?hcv\b|\bvdrl\b|\brpr\b|syphilis|treponem|"
            r"serolog|\bwidal\b|\belisa\b.*(antigen|antibody).*(virus|bacteri)|"
            r"cryptococc|galactomannan|\bparasit|stool (exam|microscop|ova)|ova and parasite|"
            r"\bviral load\b|blood film.*malaria|leishman|brucell|leptospir|"
            r"toxoplasm|\bcmv\b|\bebv\b|\bhsv\b|parvovirus|dengue|"
            r"cytomegalovirus|epstein.?barr|herpes|varicella|measles|rubella|"
            r"human immunodeficiency virus|\blyme\b|borreli|rickettsi|"
            r"tuberculosis (test|screen|work)|malaria|helicobacter|\bh pylori\b|"
            r"(viral|virus|bacterial|infectious|infection)[a-z ]{0,14}"
            r"(marker|test|screen|serolog|studies|work|panel)|"
            r"fungal|mycolog|\bpcr\b|nucleic acid amplification|\bnaat\b|"
            r"\bantigen (test|detection|assay)|interferon.?gamma release",
            "Microbiology & infectious serology",
        ),
        (
            r"anti.?nuclear|\bana\b|anti.?ds.?dna|\bena\b|extractable nuclear|\banca\b|"
            r"rheumatoid factor|anti.?ccp|anti.?cyclic citrullinated|"
            r"complement|\bc3\b|\bc4\b|\bch50\b|"
            r"immunoglobulin|\bigg\d?\b|\biga\b|\bigm\b|\bige\b|cryoglobulin|"
            r"antiphospholipid|lupus anticoagulant|anti.?cardiolipin|beta.?2 glycoprotein|"
            r"coombs|direct antiglobulin|autoantib|autoimmune|"
            r"anti.?s[sm]\b|anti.?smith|anti.?ss.?[ab]|anti.?ro|anti.?la|anti.?rnp|"
            r"anti.?centromere|anti.?scl|anti.?jo|anti.?mitochondrial|anti.?smooth muscle|"
            r"anti.?tissue transglutaminase|anti.?endomysial|"
            r"anti.?glomerular basement|anti.?gbm|"
            r"anti.?streptolysin|antistreptolysin|\baso titer\b|"
            r"acetylcholine receptor antibod|anti.?musk|aquaporin|anti.?mog|"
            r"celiac serolog|allerg|specific ige|skin prick|lymphocyte transformation|"
            r"\bhla\b|human leukocyte antigen|interleukin|\bil.?\d\b|"
            r"tumor necrosis factor|cytokine|lysozyme|\bkl 6\b|"
            r"thyroid antibod|myeloma (screen|panel)|light chain|"
            r"immunolog|immune (workup|panel)|immunoelectrophor|immunofixation",
            "Immunology & autoimmune serology",
        ),
        (
            r"complete blood count|\bcbc\b|full blood count|\bfbc\b|hemogram|blood count|"
            r"blood cell count|blood picture|"
            r"hemoglobin|hematocrit|\bhct\b|\bhb\b|red (blood )?cell|\brbc\b|"
            r"erythrocyte|white (blood )?cell|leukocyte|\bwbc\b|neutrophil|"
            r"lymphocyte|eosinophil|basophil|monocyte|platelet|thrombocyt|"
            r"differential|reticulocyte|\bmcv\b|\bmch\b|\bmchc\b|\brdw\b|"
            r"mean (corpuscular|cell) (volume|hemoglobin)|packed cell volume|"
            r"red cell distribution|red (blood )?cell ind|"
            r"(erythrocyte )?sedimentation rate|\besr\b|blood (film|smear|picture|morpholog)|"
            r"peripheral smear|peripheral blood|bone marrow|"
            r"hemolysis|haptoglobin|osmotic fragility|"
            r"hemoglobin electrophoresis|\bg6pd\b|sickling|blood group|cross.?match|"
            r"hematolog",
            "Hematology",
        ),
        (
            r"prothrombin|\bpt\b/|\binr\b|international normalized|"
            r"partial thromboplastin|\baptt\b|\bptt\b|fibrinogen|d.?dimer|"
            r"coagulation|clotting|factor (v|viii|ix|x|xi|xiii|ii)\b|"
            r"antithrombin|protein [cs]\b|thrombophilia|thrombin time|"
            r"fibrin degradation|fibrin split|"
            r"von willebrand|bleeding time|thromboelasto|anti.?xa",
            "Coagulation",
        ),
        (
            r"thyroid (function|profile|hormone)|\btsh\b|thyroid.?stimulating hormone|"
            r"thyroxine|\bt3\b|\bt4\b|triiodo|"
            r"thyroglobulin|thyroid peroxidase|\btpo\b|calcitonin|"
            r"cortisol|\bacth\b|adrenocorticotrop|dexamethasone suppression|synacthen|"
            r"parathyroid hormone|parathormone|\bpth\b|vitamin d|25.?hydroxy|calcitriol|"
            r"growth hormone|\bigf.?1\b|insulin.?like growth factor|prolactin|"
            r"luteinizing hormone|\blh\b|follicle.?stimulating|\bfsh\b|"
            r"testosterone|estradiol|estrogen|progesterone|androgen|\bdhea\b|"
            r"dehydroepiandrosterone|erythropoietin|"
            r"insulin|c.?peptide|glucagon|gastrin|"
            r"aldosterone|\brenin\b|catecholamine|metanephrine|"
            r"vanillylmandelic|\bvma\b|chromogranin|"
            r"\bhba1c\b|hemoglobin a1c|glycated hemoglobin|glycosylated|"
            r"oral glucose tolerance|\bogtt\b|"
            r"hormon|endocrin|pituitary function|adrenal function",
            "Endocrinology",
        ),
        (
            r"alpha.?fetoprotein|\bafp\b|carcinoembryonic|\bcea\b|"
            r"\bca.?19.?9\b|\bca.?125\b|\bca.?15.?3\b|\bca.?72.?4\b|"
            r"(carbohydrate|cancer) antigen|"
            r"prostate.?specific antigen|\bpsa\b|beta.?hcg|human chorionic|"
            r"neuron.?specific enolase|\bnse\b|squamous cell carcinoma antigen|"
            r"tumor marker|\bcyfra\b|\bb2m\b|beta.?2 microglobulin",
            "Tumor markers",
        ),
        (
            r"urinalys|\burine\b|urinary (protein|sediment|electrolyte|osmolal|"
            r"catecholamine|cortisol|excretion|analys|studies|test)|"
            r"24.?hour urin|proteinuria|albumin.?creatinine ratio|bence.?jones|"
            r"cerebrospinal fluid|\bcsf\b|lumbar puncture|"
            r"pleural fluid|ascit(ic|es) fluid|peritoneal fluid|synovial fluid|"
            r"pericardial fluid|body fluid|\bstool\b|fecal|semen analys|seminal|"
            r"sputum (analys|examination|cytolog)|"
            r"bronchoalveolar lavage|\bbal\b fluid|aqueous humor|vitreous (tap|sample)",
            "Urinalysis & body fluids",
        ),
        (
            r"amino acid|organic acid|acylcarnitine|carnitine|"
            r"enzyme (activity|assay|level)|lysosomal|"
            r"newborn screen|tandem mass spec|\bms/ms\b|"
            r"very long chain fatty acid|\bvlcfa\b|phytanic|"
            r"mucopolysaccharid|glycosaminoglycan|oligosaccharide|"
            r"biotinidase|galactose.?1.?phosphate|homocystein|"
            r"copper|ceruloplasmin|porphyrin|porphobilinogen|alpha 1 antitrypsin|"
            r"lactate\b|lactic acid|\bpyruvate\b|ammonia|blood gas|blood ph\b|acid.?base|"
            r"metabolic (screen|workup|panel)|inborn error",
            "Metabolic & biochemical genetics",
        ),
        (
            r"drug (level|screen|assay)|toxicolog|therapeutic (drug )?monitor|"
            r"\bethanol\b|alcohol level|\blead\b level|heavy metal|"
            r"tacrolimus|cyclosporin|vancomycin|digoxin level|"
            r"methotrexate level|carbamazepine|valproate|phenytoin|lithium level|"
            r"urine drug",
            "Toxicology & drug monitoring",
        ),
        (
            r"creatinine|\bbun\b|blood urea|\burea\b|\begfr\b|glomerular filtration|"
            r"renal (function|profile|parameter)|"
            r"kidney function|electrolyte|sodium|potassium|chloride|bicarbonate|"
            r"anion gap|osmolality|osmolarity|"
            r"\bcalcium\b|phosph(ate|orus)|magnesium|uric acid|"
            r"(liver|hepatic) (function|profile|enzyme|test|biochemical)|\blft\b|"
            r"aminotransferase|transaminase|\bast\b|\balt\b|\bsgot\b|\bsgpt\b|"
            r"alkaline phosphatase|\balp\b|gamma.?glutamyl|\bggt\b|bilirubin|"
            r"albumin|(total |serum )?protein\b|globulin|protein electrophoresis|"
            r"lactate dehydrogenase|lactic dehydrogenase|\bldh\b|"
            r"creatine (phospho)?kinase|\bck\b|\bcpk\b|aldolase|muscle enzyme|"
            r"troponin|\bbnp\b|natriuretic|myoglobin|cardiac (enzyme|biomarker|marker)|"
            r"glucose|blood sugar|glycemi|"
            r"lipid|cholesterol|triglyceride|\bhdl\b|\bldl\b|"
            r"amylase|lipase|pancreatic enzyme|"
            r"angiotensin.?converting enzyme|\bace\b|"
            r"c.?reactive protein|\bcrp\b|procalcitonin|"
            r"inflammat(ory|ion) marker|acute phase reactant|"
            r"ferritin|\biron\b|transferrin|iron.?binding|\btibc\b|"
            r"folate|folic acid|vitamin b12|cobalamin|"
            r"biochemistr|biochemical|chemistr|serum |plasma |blood level",
            "Clinical chemistry",
        ),
        (
            r"laborator|blood (test|work|investigation|panel|profile|analys|parameter|"
            r"examination|studies)|"
            r"routine (test|investigation|examination)|screening|investigation|workup|"
            r"panel|profile|assay|parameter|\btest\b|\bstudies\b",
            "General / unspecified laboratory",
        ),
    ]
)

LAB_CONCEPT = _rx(
    [
        (r"blood culture", "Blood culture"),
        (r"urine culture", "Urine culture"),
        (r"cerebrospinal|\bcsf\b|lumbar puncture", "Cerebrospinal fluid analysis"),
        (
            r"24.?hour urin|proteinuria|albumin.?creatinine|urin(e|ary) protein|"
            r"urin(e|ary) electrophor|urine immunofixation|bence.?jones|"
            r"urin(e|ary) (electrolyte|osmolal|creatinine|catecholamine|"
            r"metanephrine|excretion|studies|test)|urine (free )?cortisol",
            "Quantitative urine studies",
        ),
        (r"urinalys|\burin(e|ary)\b", "Urinalysis"),
        (
            r"pleural (fluid|effusion)|ascit|peritoneal fluid|pericardial fluid|"
            r"synovial fluid|body fluid",
            "Serous / synovial fluid analysis",
        ),
        (r"\bstool\b|fecal|occult blood|calprotectin", "Stool studies"),
        (r"sputum|bronchoalveolar|\bbal\b", "Sputum / BAL studies"),
        (r"semen analys|seminal", "Semen analysis"),
        (
            r"complete blood count|\bcbc\b|full blood count|\bfbc\b|hemogram|"
            r"blood count|blood cell count",
            "Complete blood count",
        ),
        (r"hemoglobin(?!.*electrophor)(?!.*a1c)|\bhb\b(?!a1c)", "Hemoglobin"),
        (r"hematocrit|\bhct\b|packed cell volume", "Hematocrit"),
        (r"white (blood )?cell|leukocyte count|\bwbc\b|total leukocyte", "White blood cell count"),
        (
            r"differential|neutrophil|lymphocyte|eosinophil|basophil|monocyte",
            "Leukocyte differential",
        ),
        (r"platelet|thrombocyte", "Platelet count"),
        (r"red (blood )?cell count|\brbc count\b|erythrocyte count", "Red blood cell count"),
        (
            r"\bmcv\b|\bmch\b|\bmchc\b|\brdw\b|red (blood )?cell ind|"
            r"mean (corpuscular|cell) (volume|hemoglobin)|red cell distribution",
            "Red cell indices",
        ),
        (r"reticulocyte", "Reticulocyte count"),
        (r"(erythrocyte )?sedimentation rate|\besr\b", "Erythrocyte sedimentation rate"),
        (r"c.?reactive protein|\bcrp\b", "C-reactive protein"),
        (r"procalcitonin", "Procalcitonin"),
        (r"inflammat(ory|ion) marker|acute phase reactant", "Inflammatory markers (composite)"),
        (
            r"blood (film|smear|picture|morpholog)|peripheral (blood )?(smear|film|"
            r"picture|examination|analys|cell count|morpholog)|peripheral film|"
            r"red (blood )?cell morpholog",
            "Peripheral blood smear",
        ),
        (r"bone marrow", "Bone marrow examination"),
        (
            r"hemoglobin electrophoresis|sickling|\bg6pd\b|osmotic fragility|haptoglobin",
            "Hemolysis / hemoglobinopathy workup",
        ),
        (r"blood group|cross.?match|\bcoombs\b|antiglobulin", "Blood group / compatibility"),
        (r"prothrombin|\binr\b|international normalized", "Prothrombin time / INR"),
        (r"partial thromboplastin|\baptt\b|\bptt\b", "Activated partial thromboplastin time"),
        (r"fibrinogen", "Fibrinogen"),
        (r"d.?dimer", "D-dimer"),
        (
            r"coagulation|clotting|factor (v|viii|ix|x|xi|xiii)|von willebrand|"
            r"antithrombin|protein [cs]\b|thrombophilia|anti.?xa|bleeding time|"
            r"thrombin time|fibrin (degradation|split)",
            "Coagulation studies",
        ),
        (
            r"creatinine|\bbun\b|blood urea|\burea\b|\begfr\b|glomerular filtration|"
            r"renal (function|parameter)|kidney function",
            "Renal function (creatinine / urea)",
        ),
        (
            r"electrolyte|sodium|potassium|chloride|bicarbonate|anion gap|osmolal",
            "Serum electrolytes",
        ),
        (r"\bcalcium\b|phosph(ate|orus)|magnesium", "Calcium / phosphate / magnesium"),
        (r"uric acid|urate", "Uric acid"),
        (
            r"(protein|serum) electrophoresis|immunoelectrophor|immunofixation|"
            r"free light chain|paraprotein|bence.?jones",
            "Protein electrophoresis / immunofixation",
        ),
        (
            r"(liver|hepatic) (function|profile|enzyme|test|biochemical)|\blft\b|"
            r"aminotransferase|transaminase|\bast\b|\balt\b|\bsgot\b|\bsgpt\b|"
            r"alkaline phosphatase|\balp\b|gamma.?glutamyl|\bggt\b|bilirubin",
            "Liver function tests",
        ),
        (r"albumin|total proteins?|globulin|\bproteins?\b", "Albumin / total protein"),
        (r"lactate dehydrogenase|lactic dehydrogenase|\bldh\b", "Lactate dehydrogenase"),
        (r"creatine (phospho)?kinase|\bck\b|\bcpk\b|aldolase|muscle enzyme", "Creatine kinase"),
        (
            r"troponin|\bbnp\b|natriuretic|myoglobin|cardiac (enzyme|marker|biomarker)|"
            r"nt.?pro",
            "Cardiac biomarkers",
        ),
        (r"angiotensin.?converting enzyme|\bace\b", "Angiotensin-converting enzyme"),
        (r"\bhba1c\b|hemoglobin a1c|glycated|glycosylated", "Hemoglobin A1c"),
        (r"glucose tolerance|\bogtt\b", "Glucose tolerance test"),
        (r"glucose|blood sugar|glycemi", "Blood glucose"),
        (r"lipid|cholesterol|triglyceride|\bhdl\b|\bldl\b", "Lipid profile"),
        (r"amylase|lipase|pancreatic enzyme", "Pancreatic enzymes (amylase / lipase)"),
        (r"ferritin|\biron\b|transferrin|\btibc\b|iron.?binding", "Iron studies / ferritin"),
        (r"vitamin b12|cobalamin|\bfolate\b|folic acid", "Vitamin B12 / folate"),
        (r"blood gas|acid.?base|\babg\b|\bvbg\b", "Arterial / venous blood gas"),
        (r"\blactate\b|lactic acid|\bpyruvate\b|ammonia", "Lactate / pyruvate / ammonia"),
        (
            r"thyroid (function|profile|hormone)|\btsh\b|thyroid.?stimulating hormone|"
            r"thyroxine|\bt3\b|\bt4\b|triiodo",
            "Thyroid function tests",
        ),
        (r"thyroid peroxidase|thyroglobulin|\btpo\b|thyroid antibod", "Thyroid autoantibodies"),
        (r"calcitonin", "Calcitonin"),
        (
            r"cortisol|\bacth\b|adrenocorticotrop|dexamethasone|synacthen|adrenal function",
            "Adrenal axis (cortisol / ACTH)",
        ),
        (r"parathyroid hormone|parathormone|\bpth\b", "Parathyroid hormone"),
        (r"vitamin d|25.?hydroxy|calcitriol", "Vitamin D"),
        (r"growth hormone|\bigf.?1\b|insulin.?like growth factor", "Growth hormone / IGF-1"),
        (r"prolactin", "Prolactin"),
        (
            r"luteinizing hormone|\blh\b|follicle.?stimulating|\bfsh\b|"
            r"testosterone|estradiol|estrogen|progesterone|androgen|\bdhea\b|"
            r"dehydroepiandrosterone|anti.?mullerian|gonadal",
            "Gonadal / reproductive hormones",
        ),
        (r"aldosterone|\brenin\b", "Renin / aldosterone"),
        (
            r"catecholamine|metanephrine|vanillylmandelic|\bvma\b|chromogranin",
            "Catecholamines / metanephrines",
        ),
        (r"erythropoietin", "Erythropoietin"),
        (r"insulin|c.?peptide|glucagon|gastrin|somatostatin|\bvip\b", "Pancreatic / gut hormones"),
        (r"anti.?nuclear|\bana\b(?!lys)", "Antinuclear antibody"),
        (
            r"anti.?ds.?dna|\bena\b|extractable nuclear|anti.?s[sm]\b|anti.?smith|"
            r"anti.?ss.?[ab]|anti.?ro|anti.?la|anti.?rnp|anti.?scl|anti.?jo|"
            r"anti.?centromere|myositis (panel|antibod)",
            "Extractable nuclear antigen antibodies",
        ),
        (r"\banca\b|myeloperoxidase|proteinase 3|\bmpo\b|\bpr3\b", "ANCA"),
        (r"rheumatoid factor|anti.?ccp|citrullinated", "Rheumatoid factor / anti-CCP"),
        (r"complement|\bc3\b|\bc4\b|\bch50\b", "Complement levels"),
        (
            r"immunoglobulin|\bigg\d?\b|\biga\b|\bigm\b|\bige\b(?!.*specific)",
            "Serum immunoglobulins",
        ),
        (
            r"antiphospholipid|lupus anticoagulant|anti.?cardiolipin|glycoprotein i",
            "Antiphospholipid antibodies",
        ),
        (
            r"anti.?mitochondrial|anti.?smooth muscle|anti.?lkm|liver autoantib",
            "Liver autoantibodies",
        ),
        (r"transglutaminase|endomysial|celiac serolog|gliadin", "Celiac serology"),
        (
            r"acetylcholine receptor|anti.?musk|aquaporin|anti.?mog|"
            r"onconeural|paraneoplastic antibod|anti.?ganglioside|"
            r"anti.?glomerular basement|anti.?gbm",
            "Neurologic / renal autoantibodies",
        ),
        (r"anti.?streptolysin|antistreptolysin|\baso\b", "Antistreptolysin O titre"),
        (r"\bhla\b|human leukocyte antigen", "HLA typing"),
        (
            r"interleukin|\bil.?\d\b|tumor necrosis factor|cytokine|\bkl 6\b|lysozyme",
            "Cytokine / soluble marker assay",
        ),
        (r"allerg|specific ige|skin prick|patch test", "Allergy testing"),
        (r"cryoglobulin", "Cryoglobulins"),
        (r"autoantib|autoimmune|antibod", "Other autoantibodies"),
        (r"immunolog(ic|y)|immune (workup|panel|screen)", "Unspecified immunological workup"),
        (
            r"flow cytometr|immunophenotyp|lymphocyte subset|cd4",
            "Flow cytometry / immunophenotyping",
        ),
        (r"blood culture", "Blood culture"),
        (r"urine culture", "Urine culture"),
        (r"\bculture\b|sensitivit|antibiogram", "Other microbiological culture"),
        (
            r"gram stain|acid.?fast|\bafb\b|ziehl|india ink|microscopy for",
            "Stain microscopy (Gram / AFB)",
        ),
        (
            r"mantoux|tuberculin|\bigra\b|quantiferon|interferon.?gamma release|"
            r"tuberculosis (test|screen|work)",
            "Tuberculin skin test / IGRA",
        ),
        (r"\bhiv\b|human immunodeficiency virus", "HIV testing"),
        (r"hepatitis|\bhbsag\b|anti.?hcv", "Viral hepatitis serology"),
        (r"syphilis|\bvdrl\b|rapid plasma reagin|\brpr\b|treponem", "Syphilis serology"),
        (r"\bpcr\b|nucleic acid|\bnaat\b|viral load|\brt.?pcr\b", "Pathogen nucleic-acid testing"),
        (
            r"serolog|antibod(y|ies) (titer|against)|\belisa\b|agglutinat|\bwidal\b|"
            r"cytomegalovirus|epstein.?barr|herpes|varicella|measles|rubella|"
            r"\blyme\b|borreli|rickettsi|brucell|leptospir|toxoplasm|dengue|"
            r"(viral|virus|bacterial|infectious|infection)\s*"
            r"(marker|test|screen|studies|work|panel)",
            "Infectious serology",
        ),
        (r"parasit|malaria|leishman|stool (ova|exam)|ova and parasite|helminth", "Parasitology"),
        (r"fungal|mycolog|galactomannan|cryptococc|beta.?d.?glucan", "Fungal testing"),
        (
            r"exome|genome sequencing|\bwes\b|\bwgs\b|gene panel|targeted sequencing",
            "Exome / genome / panel sequencing",
        ),
        (r"sanger|sequenc|chromatogram|electropherogram", "Targeted gene sequencing"),
        (r"karyotyp|cytogenet|chromosom", "Karyotype / cytogenetics"),
        (r"\bfish\b|fluorescence in.?situ", "Fluorescence in situ hybridization"),
        (
            r"array.?cgh|comparative genomic|microarray|snp array|copy number",
            "Chromosomal microarray",
        ),
        (
            r"\bmlpa\b|southern blot|methylation|repeat expansion|triplet repeat",
            "Targeted molecular assay (MLPA / methylation)",
        ),
        (r"pedigree|family (tree|study)|segregation|linkage", "Pedigree / segregation analysis"),
        (r"mutation|variant|\bgene\b|genetic|molecular", "Other molecular genetic testing"),
        (r"alpha.?fetoprotein|\bafp\b", "Alpha-fetoprotein"),
        (r"carcinoembryonic|\bcea\b", "Carcinoembryonic antigen"),
        (
            r"\bca.?19.?9\b|\bca.?125\b|\bca.?15.?3\b|\bca.?72.?4\b|"
            r"(carbohydrate|cancer) antigen",
            "Cancer antigen (CA 19-9 / 125 / 15-3)",
        ),
        (r"prostate.?specific|\bpsa\b", "Prostate-specific antigen"),
        (r"chorionic gonadotropin|\bhcg\b", "Human chorionic gonadotropin"),
        (
            r"tumor marker|neuron.?specific enolase|\bnse\b|\bcyfra\b|"
            r"beta.?2 microglobulin|squamous cell carcinoma antigen",
            "Other tumor markers",
        ),
        (
            r"urinalys|urine (analys|routine|dipstick|examination|microscop|sediment|"
            r"cytolog|specific gravity|\bph\b|ketone|colour|color)|urinary sediment",
            "Urinalysis",
        ),
        (
            r"24.?hour urin|proteinuria|albumin.?creatinine|urin(e|ary) protein|"
            r"urin(e|ary) electrophor|urine immunofixation|bence.?jones|"
            r"urin(e|ary) (electrolyte|osmolal|creatinine|catecholamine|"
            r"metanephrine|excretion|studies|test)|urine (free )?cortisol",
            "Quantitative urine studies",
        ),
        (r"cerebrospinal|\bcsf\b|lumbar puncture", "Cerebrospinal fluid analysis"),
        (r"semen analys|seminal", "Semen analysis"),
        (
            r"pleural (fluid|effusion)|ascit|peritoneal fluid|pericardial fluid|"
            r"synovial fluid|body fluid|aspirate (analys|fluid)",
            "Serous / synovial fluid analysis",
        ),
        (r"fine.?needle aspiration|\bfna\b|cytolog", "Aspiration cytology"),
        (r"\bstool\b|fecal|occult blood|calprotectin", "Stool studies"),
        (r"sputum|bronchoalveolar|\bbal\b", "Sputum / BAL studies"),
        (
            r"amino acid|organic acid|acylcarnitine|carnitine|newborn screen|"
            r"tandem mass|homocystein|mucopolysacchar|glycosaminoglycan|oligosacchar|"
            r"very long chain|phytanic|porphyrin|porphobilinogen|biotinidase",
            "Metabolic screen (amino / organic acids)",
        ),
        (r"enzyme (activity|assay|level)|lysosomal|enzymatic", "Enzyme activity assay"),
        (
            r"copper|ceruloplasmin|\bzinc\b|selenium|heavy metal|\blead\b",
            "Trace metals / ceruloplasmin",
        ),
        (
            r"drug (level|screen|assay)|toxicolog|therapeutic (drug )?monitor|"
            r"tacrolimus|cyclosporin|vancomycin|digoxin|methotrexate level|"
            r"carbamazepine|valproate|phenytoin|lithium|ethanol|alcohol level",
            "Drug levels / toxicology",
        ),
        (r"hormon|endocrin", "Unspecified hormonal workup"),
        (r"western blot|immunoblot|elisa|bret|assay", "Research / specialised assay"),
        (
            r"biochemistr|biochemical|chemistr|metabolic panel",
            "Basic / comprehensive metabolic panel",
        ),
        (r"hematolog", "Unspecified hematology workup"),
        (
            r"laborator|blood (test|work|investigation|analys|parameter|examination|"
            r"studies)|routine|screening|"
            r"investigation|workup|panel|profile|parameter|\btest\b|study",
            "Unspecified laboratory workup",
        ),
    ]
)


IMAGING_CATEGORY = _rx(
    [
        (
            r"\bpet\b|positron emission|\bspect\b|scintigraph|\bbone scan\b|"
            r"radionuclide|radioisotope|nuclear (medicine|scan)|\bmibg\b|"
            r"\bdmsa\b|\bdtpa\b|\bmag3\b|gallium scan|octreotide scan|"
            r"thallium|sestamibi|lymphoscintigraph|\bhida\b|"
            r"\bfdg\b|\bdotatate\b|\bpsma\b",
            "Nuclear medicine",
        ),
        (
            r"dermoscop|dermatoscop|reflectance confocal|wood.?s lamp|"
            r"clinical photograph|trichoscop",
            "Dermatologic imaging",
        ),
        (
            r"\boct\b|optical coherence|fundus (photograph|autofluorescence|imag)|"
            r"fluorescein angiograph|indocyanine|\bicga\b|gonioscop|"
            r"ultrasound biomicroscop|corneal topograph|specular microscop|"
            r"retinal (imag|photograph)|slit.?lamp photograph|anterior segment photograph|"
            r"confocal microscop.*(cornea|eye)|corneal confocal|scheimpflug|"
            r"axial length|biometr|ocular ultrasound|\bb.?scan\b",
            "Ophthalmic imaging",
        ),
        (r"mammograph|mammogram|breast tomosynthesis|galactograph", "Mammography"),
        (
            r"colonoscop|sigmoidoscop|gastroscop|gastroduodenoscop|"
            r"esophagogastroduodenoscop|\begd\b|"
            r"upper (gi |gastrointestinal )?endoscop|enteroscop|capsule endoscop|"
            r"bronchoscop|bronchofibroscop|cystoscop|ureteroscop|hysteroscop|"
            r"laparoscop|thoracoscop|"
            r"mediastinoscop|arthroscop|laryngoscop|nasoendoscop|nasal endoscop|"
            r"nasofibroscop|fibroscop|endomicroscop|"
            r"rhinoscop|otoscop|proctoscop|anoscop|\bercp\b|cholangioscop|"
            r"endoscopic ultraso|\beus\b|endoscop",
            "Endoscopy",
        ),
        (
            r"echocardiograph|echocardiogram|\btte\b|\btee\b|"
            r"transthoracic echo|transesophageal echo|\becho\b",
            "Echocardiography",
        ),
        (
            r"coronary (angiograph|angiogram)|\bdsa\b|digital subtraction|"
            r"catheter(ization)? angiograph|cardiac catheter|heart catheter|"
            r"ventriculograph|ventriculogram|aortograph|arteriograph|"
            r"venograph|venogram|phlebograph|"
            r"lymphangiograph|intravascular ultrasound|\bivus\b|"
            r"angiogra(ph|m)(?!.*(ct|mr|fluorescein|indocyanine|oct))",
            "Catheter angiography",
        ),
        (
            r"\bct\b|computed tomograph|computerized tomograph|\bcta\b|\bhrct\b|"
            r"\bmsct\b|multislice|multidetector|"
            r"cone.?beam|\bcbct\b|tomosynthesis|myelograph.*ct|"
            r"\bpet.?ct\b|dual.?energy ct",
            "Computed tomography",
        ),
        (
            r"\bmri\b|magnetic resonance|\bmra\b|\bmrv\b|\bmrcp\b|\bmrs\b|"
            r"diffusion.?weighted|\bdwi\b|\bflair\b|\bt1\b|\bt2\b|"
            r"functional (mri|neuroimaging)|\bfmri\b|tractograph|"
            r"mr spectroscop|spectroscopy.*magnetic",
            "Magnetic resonance imaging",
        ),
        (r"ultrasound|sonogram|doppler|elastograph|\bfast scan\b", "Ultrasonography"),
        (
            r"x.?ray|radiograph|radiogram|radiolog|roentgen|"
            r"orthopantomogram|orthopantomograph|panoramic|\bopg\b|"
            r"cephalometr|cephalogram|skeletal survey|bone survey|bone age|"
            r"fluoroscop|barium|contrast (swallow|study|series)|"
            r"(upper|lower) gastrointestinal series|small bowel (series|follow)|"
            r"esophagograph|esophagogram|"
            r"\bivu\b|intravenous (uro|pyelo)gra(ph|m)|pyelogram|pyelograph|"
            r"urethrogra(ph|m)|cystourethrogram|"
            r"sialogra(ph|m)|fistulogra(ph|m)|dexa|bone densitometr|"
            r"myelogra(ph|m)|arthrogra(ph|m)|hysterosalpingogra(ph|m)|\bkub\b",
            "Radiography & fluoroscopy",
        ),
        (r"renogram|renograph|scintigra|uptake scan", "Nuclear medicine"),
        (
            r"thermograph|capillaroscop|confocal microscop|\bimaging\b|\bscan\b|"
            r"examination|evaluation|investigation",
            "Other imaging",
        ),
    ]
)

IMAGING_CONCEPT = _rx(
    [
        (r"\bpet.?ct\b|pet/ct|positron emission.*computed tomograph", "PET/CT"),
        (r"\bpet.?mri\b|pet/mr", "PET/MRI"),
        (r"\bpet\b|positron emission", "PET"),
        (r"\bspect.?ct\b|spect/ct", "SPECT/CT"),
        (r"\bspect\b", "SPECT"),
        (r"bone scan|bone scintigraph|skeletal scintigraph", "Bone scintigraphy"),
        (r"thyroid (scan|scintigraph|uptake)", "Thyroid scintigraphy"),
        (
            r"\bmibg\b|octreotide|\bdotatate\b|\bpsma\b|somatostatin receptor",
            "Receptor-targeted scintigraphy",
        ),
        (r"lymphoscintigraph|sentinel node (scan|scintigraph)", "Lymphoscintigraphy"),
        (
            r"renal (scan|scintigraph)|renogra(m|ph)|\bdmsa\b|\bdtpa\b|\bmag3\b",
            "Renal scintigraphy",
        ),
        (r"\bhida\b|hepatobiliary scintigraph", "Hepatobiliary scintigraphy"),
        (r"myocardial perfusion|thallium|sestamibi|\bmpi\b", "Myocardial perfusion scintigraphy"),
        (r"gallium|leukocyte scan|labelled leukocyte", "Inflammation scintigraphy"),
        (r"scintigraph|radionuclide|radioisotope|nuclear", "Other scintigraphy"),
        (r"\bcta\b|ct angiograph|computed tomograph(y|ic) angiograph", "CT angiography"),
        (r"\bhrct\b|high.?resolution (ct|computed)", "High-resolution CT"),
        (r"cone.?beam|\bcbct\b", "Cone-beam CT"),
        (r"ct myelograph", "CT myelography"),
        (
            r"\bct\b|computed tomograph|computerized tomograph|\bmsct\b|"
            r"multislice|multidetector",
            "CT",
        ),
        (r"\bmrcp\b|magnetic resonance cholangio", "MRCP"),
        (r"\bmra\b|magnetic resonance angiograph|mr angiograph", "MR angiography"),
        (r"\bmrv\b|magnetic resonance venograph|mr venograph", "MR venography"),
        (r"\bmrs\b|magnetic resonance spectroscop|mr spectroscop", "MR spectroscopy"),
        (r"\bfmri\b|functional magnetic resonance", "Functional MRI"),
        (r"diffusion.?weighted|\bdwi\b|diffusion tensor|tractograph", "Diffusion MRI"),
        (r"\bmri\b|magnetic resonance", "MRI"),
        (r"\btte\b|transthoracic echo", "Transthoracic echocardiography"),
        (r"\btee\b|transesophageal echo", "Transesophageal echocardiography"),
        (r"fetal echo", "Fetal echocardiography"),
        (r"stress echo|dobutamine echo", "Stress echocardiography"),
        (r"echocardiograph|echocardiogram|\becho\b", "Echocardiography"),
        (r"doppler|duplex", "Doppler ultrasound"),
        (r"endoscopic ultraso|\beus\b", "Endoscopic ultrasound"),
        (r"intravascular ultrasound|\bivus\b", "Intravascular ultrasound"),
        (r"elastograph", "Elastography"),
        (r"ultrasound|sonogram", "Ultrasound"),
        (r"colonoscop", "Colonoscopy"),
        (r"sigmoidoscop", "Sigmoidoscopy"),
        (
            r"gastroscop|gastroduodenoscop|esophagogastroduodenoscop|\begd\b|"
            r"upper (gi |gastrointestinal )?endoscop|upper endoscop",
            "Esophagogastroduodenoscopy",
        ),
        (r"capsule endoscop|enteroscop", "Capsule endoscopy / enteroscopy"),
        (r"\bercp\b|endoscopic retrograde", "ERCP"),
        (r"bronchoscop|bronchofibroscop", "Bronchoscopy"),
        (r"cystoscop|ureteroscop", "Cystoscopy / ureteroscopy"),
        (r"hysteroscop", "Hysteroscopy"),
        (r"laparoscop", "Laparoscopy"),
        (r"thoracoscop|mediastinoscop", "Thoracoscopy / mediastinoscopy"),
        (r"arthroscop", "Arthroscopy"),
        (
            r"laryngoscop|nasoendoscop|nasal endoscop|nasofibroscop|rhinoscop|otoscop",
            "ENT endoscopy",
        ),
        (r"proctoscop|anoscop", "Proctoscopy / anoscopy"),
        (r"endomicroscop|fibroscop|endoscop", "Other endoscopy"),
        (r"\boct.?a\b|oct angiograph|optical coherence tomography angiograph", "OCT angiography"),
        (r"fluorescein angiogra(ph|m)", "Fluorescein angiography"),
        (r"indocyanine", "Indocyanine green angiography"),
        (r"coronary angiogra(ph|m)", "Coronary angiography"),
        (r"\bdsa\b|digital subtraction", "Digital subtraction angiography"),
        (r"cardiac catheter|heart catheter", "Cardiac catheterization"),
        (r"ventriculogra(ph|m)", "Ventriculography"),
        (r"venogra(ph|m)|phlebograph", "Venography"),
        (r"lymphangiograph", "Lymphangiography"),
        (r"angiogra(ph|m)|arteriograph|aortograph", "Catheter angiography"),
        (r"\boct\b|optical coherence", "Optical coherence tomography"),
        (r"fundus autofluorescence", "Fundus autofluorescence"),
        (
            r"fundus (photograph|imag)|retinal photograph|fundoscop|ophthalmoscop",
            "Fundus photography",
        ),
        (r"ultrasound biomicroscop|\bb.?scan\b|ocular ultrasound", "Ocular ultrasound / UBM"),
        (
            r"corneal topograph|specular microscop|pentacam|scheimpflug|"
            r"axial length|biometr|anterior segment (oct|photograph)|"
            r"slit.?lamp photograph",
            "Anterior-segment imaging",
        ),
        (r"dermoscop|dermatoscop|trichoscop", "Dermoscopy"),
        (r"reflectance confocal|corneal confocal|confocal", "Confocal microscopy (in vivo)"),
        (r"wood.?s lamp", "Wood's lamp examination"),
        (r"clinical photograph|photograph", "Clinical photography"),
        (r"mammogra(ph|m)|tomosynthesis|galactograph", "Mammography"),
        (r"orthopantomogram|orthopantomograph|panoramic|\bopg\b", "Panoramic radiograph"),
        (r"cephalometr|cephalogram", "Cephalometric radiograph"),
        (r"skeletal survey|bone survey", "Skeletal survey"),
        (r"bone age", "Bone age assessment"),
        (
            r"barium|contrast (swallow|meal|enema|study|series)|"
            r"(upper|lower) gastrointestinal series|small bowel (series|follow)|"
            r"esophagogra(ph|m)",
            "Contrast fluoroscopy (barium)",
        ),
        (
            r"\bivu\b|intravenous (uro|pyelo)gra(ph|m)|pyelogra(m|ph)|"
            r"urethrogra(ph|m)|cystogra(ph|m)|cystourethrogram|\bmcug\b|\bvcug\b",
            "Urographic contrast study",
        ),
        (r"hysterosalpingogra(ph|m)", "Hysterosalpingography"),
        (r"sialogra(ph|m)|fistulogra(ph|m)|dacryocystograph", "Other contrast study"),
        (r"myelogra(ph|m)", "Myelography"),
        (r"arthrogra(ph|m)", "Arthrography"),
        (r"cholangiogra(ph|m)", "Cholangiography"),
        (r"dexa|\bdxa\b|bone densitometr|bone mineral density", "Bone densitometry"),
        (r"fluoroscop", "Fluoroscopy"),
        (r"x.?ray|radiogra(ph|m)|roentgen|\bkub\b|\bfilm\b|radiolog", "Radiography"),
        (r"thermograph|capillaroscop", "Other imaging modality"),
        (r"imaging|\bscan\b|examination|evaluation|investigation", "Unspecified imaging"),
    ]
)

ANATOMY = _rx(
    [
        (
            r"\bbrain\b|cerebr|cranial|intracranial|\bhead\b|skull|pituitar|sella|"
            r"posterior fossa|cerebell|temporal lobe|frontal lobe|parietal|occipital|"
            r"basal ganglia|corpus callosum|white matter|ventricul(ar|e)|hippocamp|"
            r"neuro(?:imaging|axis)|\bcns\b",
            "Brain / cranium",
        ),
        (
            r"\bspine\b|spinal|vertebr|cervical spine|lumbar|thoracic spine|sacr|"
            r"coccy|intervertebral|cauda equina|\bcord\b",
            "Spine / spinal cord",
        ),
        (
            r"\borbit|\beye\b|\beyes\b|ocular|retina|macula|choroid|optic nerve|"
            r"cornea|anterior segment|vitreous|lacrimal",
            "Orbit / eye",
        ),
        (r"temporal bone|\bear\b|mastoid|cochlea|internal auditory|petrous", "Temporal bone / ear"),
        (
            r"paranasal|\bsinus(es)?\b|maxillar|ethmoid|sphenoid|nasal cav|nasophar",
            "Paranasal sinuses / nasopharynx",
        ),
        (
            r"\bmandib|\bmaxill|\bdental\b|\bteeth\b|\btooth\b|jaw|temporomandibular|"
            r"periapical|alveolar|odontogenic|\boral\b",
            "Maxillofacial / dentition",
        ),
        (
            r"\bneck\b|cervical (region|soft tissue)|thyroid|parathyroid|parotid|"
            r"salivary|submandibular|larynx|laryng|pharyn|carotid",
            "Neck / head-and-neck soft tissue",
        ),
        (
            r"\bchest\b|thora(x|cic)(?! spine)|\blung|pulmonar|pleura|mediastin|"
            r"bronch|airway|\bhilar\b|diaphragm",
            "Chest / lungs",
        ),
        (
            r"\bheart\b|cardiac|myocard|pericard|coronary|aortic valve|mitral|"
            r"tricuspid|atri(um|al)|ventricle of the heart|\bcardio",
            "Heart",
        ),
        (
            r"\baorta\b|aortic|\bartery\b|arterial|\bvein\b|venous|vascular|vessel|"
            r"pulmonary artery|iliac|femoral artery|popliteal",
            "Blood vessels",
        ),
        (r"\bliver\b|hepat|biliar|gallbladder|\bbile duct|cholang|portal", "Liver / biliary tract"),
        (r"pancrea", "Pancreas"),
        (r"\bspleen\b|splen", "Spleen"),
        (r"\bkidney|renal|nephro|ureter|urinary tract", "Kidney / urinary tract"),
        (r"\bbladder\b|urethra|prostat", "Bladder / prostate"),
        (
            r"\bstomach\b|gastric|duoden|\bbowel\b|intestin|jejun|ileum|ileal|"
            r"colon|colonic|rectum|rectal|\bcecum\b|cecal|appendix|appendice|"
            r"esophag|anus|anal|gastrointestinal|\bgi tract\b|mesenter|omentum",
            "Gastrointestinal tract",
        ),
        (r"\babdom|peritone|retroperitone|\bflank\b", "Abdomen (general)"),
        (
            r"\bpelvi|uter|ovar|adnex|cervix|cervical (canal|os)|vagina|vulva|"
            r"endometri|fallopian|\bfetus\b|fetal|placenta|obstetric",
            "Pelvis / female reproductive",
        ),
        (r"testis|testicul|scrotal|scrotum|penis|penile|epididym", "Male reproductive"),
        (r"adrenal", "Adrenal glands"),
        (r"\bbreast|\bnipple|axilla", "Breast / axilla"),
        (r"bone marrow", "Bone marrow"),
        (r"lymph node|lymphaden|lymphatic|nodal", "Lymph nodes"),
        (r"\bskin\b|cutaneous|dermal|subcutaneous|\bscalp\b|\bnail\b", "Skin / soft tissue"),
        (r"\bmuscle|muscular|myopath", "Skeletal muscle"),
        (
            r"\bjoint|\bknee\b|\bhip\b|\bshoulder|\belbow\b|\bwrist\b|\bankle\b|"
            r"arthro|sacroiliac|\bfoot\b|\bhand\b|\bfinger|\btoe\b|carpal|tarsal",
            "Joints / extremities",
        ),
        (
            r"\bbone\b|osseous|skelet|\bfemur\b|\btibia\b|\bhumerus\b|\bfibula\b|"
            r"\bradius\b|\bulna\b|\bpelvic bone|\brib\b|\bclavicle\b|\bsternum\b",
            "Bone / skeleton",
        ),
        (r"whole body|total body|systemic|full body", "Whole body"),
    ]
)


PHYSIO_CATEGORY = _rx(
    [
        (
            r"echocardiogra|\btte\b|\btee\b|transthoracic echo|transesophageal echo|"
            r"\becho\b|cardiotocogra|\bctg\b",
            "Echocardiography & cardiac imaging",
        ),
        (
            r"electrocardio|\becg\b|\bekg\b|holter|telemetr|event monitor|"
            r"rhythm (strip|monitor|assessment|analys)|electrophysiolog(ic|y) stud|"
            r"electroanatomic|\bicd\b (interrogation|telemetry|electrogram)|"
            r"pacemaker (check|interrogation)|tilt table|"
            r"cardiac (function|evaluation|assessment|monitor)|"
            r"intracardiac electrogram",
            "Cardiac electrophysiology & monitoring",
        ),
        (
            r"cardiac catheter|right heart cath|left heart cath|hemodynamic|"
            r"\bffr\b|fractional flow|pressure (tracing|recording|gradient)|"
            r"swan.?ganz|cardiac output|shunt (study|fraction)|"
            r"central venous pressure|blood pressure monitoring|ankle.?brachial|"
            r"postural blood pressure",
            "Invasive hemodynamics & pressure monitoring",
        ),
        (
            r"electroencephalo|\beeg\b|electromyogra|\bemg\b|nerve conduction|"
            r"electrodiagnostic|\bf.?wave\b|neurophysiolog|"
            r"opening pressure|csf (tap|pressure)|"
            r"\bncs\b|evoked potential|\bvep\b|\bssep\b|\bbaep\b|\baber\b|"
            r"polysomnograph|sleep study|\bmslt\b|actigraph|blink reflex|"
            r"repetitive nerve stimulation|single.?fiber|transcranial magnetic|"
            r"electroneuro|magnetoencephalo|\bmeg\b|patch clamp|"
            r"voltage clamp|autonomic (testing|function)|sympathetic skin response|"
            r"tilt.?table",
            "Neurophysiology",
        ),
        (
            r"spirometr|(pulmonary|lung|respiratory) function|\bpft\b|\bdlco\b|"
            r"diffusing capacity|plethysmograph|bronchial (challenge|provocation)|"
            r"methacholine|6.?minute walk|six.?minute walk|"
            r"exercise (test|tolerance|capacity)|"
            r"cardiopulmonary exercise|\bcpet\b|peak (flow|expiratory)|"
            r"pulse oximetr|oxygen(ation)? (saturation|assessment)|blood gas|capnograph|"
            r"respiratory assessment|mechanical ventilation|"
            r"treadmill|stress test",
            "Pulmonary function & exercise testing",
        ),
        (
            r"visual acuity|visual field|perimetr|intraocular pressure|tonometr|"
            r"refraction|electroretinogra|\berg\b|electro.?oculogra|\beog\b|"
            r"color vision|contrast sensitivity|microperimetr|"
            r"schirmer|pachymetr|orthoptic|hess chart|"
            r"dark adaptation|amsler|stereopsis|ocular motility",
            "Ophthalmic function testing",
        ),
        (
            r"audiometr|audiogram|audiolog|tympanometr|otoacoustic|\boae\b|"
            r"auditory brainstem|hearing (test|assessment|threshold)|"
            r"speech (audiometry|discrimination)|vestibular|caloric|"
            r"videonystagmograph|video.?oculograph|\bvng\b|electronystagmograph|"
            r"posturograph",
            "Audiovestibular testing",
        ),
        (
            r"manometr|\bph\b (monitoring|study|metry)|impedance|gastric emptying|"
            r"breath test|motility|transit study|defecograph|"
            r"anorectal",
            "Gastrointestinal function testing",
        ),
        (r"urodynamic|uroflowmetr|cystometr|post.?void residual", "Urodynamics"),
        (
            r"neuropsycholog|cognitive (test|assessment|evaluation|screen)|"
            r"\bmmse\b|mini.?mental|\bmoca\b|montreal cognitive|"
            r"intelligence quotient|\biq\b|wechsler|"
            r"developmental (assessment|quotient|scale)|bayley|griffiths|"
            r"psychometric|neurobehavio|memory test|"
            r"psychiatric (rating|scale)|depression (score|scale|inventory)|"
            r"\bhamd\b|\bbdi\b|\bpanss\b|autism|\bados\b|\bwais\b|\bwisc\b|"
            r"speech (and language )?(assessment|evaluation)|"
            r"(scale|inventory|questionnaire|rating|battery|\btask\b|"
            r"screening test|test of)\b|"
            r"behavioral (task|test)",
            "Neurocognitive & psychometric assessment",
        ),
        (
            r"gait (analys|kinematic)|posturograph|dynamometr|muscle strength test|"
            r"range of motion|functional (capacity|assessment|scale)|"
            r"nutritional assessment",
            "Functional & performance assessment",
        ),
        (
            r"skin prick|patch test|allergy challenge|provocation test|"
            r"sweat (test|chloride)|tensilon|edrophonium|ice pack test|"
            r"neostigmine|stimulation test|pulp vitality|vitality test",
            "Provocative / bedside diagnostic test",
        ),
        (r"urine output|fluid balance|intake and output", "Bedside physiologic monitoring"),
        (r"", "Other physiologic testing"),
    ]
)

PHYSIO_CONCEPT = _rx(
    [
        (r"echocardiogra|\btte\b|\btee\b|\becho\b", "Echocardiography"),
        (r"cardiotocogra|\bctg\b", "Cardiotocography"),
        (
            r"holter|24.?hour (ecg|electrocardio|monitor)|ambulatory (ecg|monitor)|"
            r"event monitor|telemetr|loop recorder|rhythm monitor",
            "Ambulatory ECG monitoring",
        ),
        (
            r"\bicd\b|pacemaker|device (interrogation|check)|intracardiac electrogram|"
            r"electroanatomic|electrophysiolog(ic|y) stud",
            "Cardiac device / EP study",
        ),
        (r"electrocardio|\becg\b|\bekg\b|rhythm (strip|assessment|analys)", "Electrocardiogram"),
        (
            r"cardiac (function|evaluation|assessment|monitor)",
            "Cardiac functional assessment (unspecified)",
        ),
        (r"tilt.?table", "Head-up tilt table test"),
        (r"right heart cath|swan.?ganz|pulmonary artery catheter", "Right heart catheterization"),
        (r"\bffr\b|fractional flow", "Fractional flow reserve"),
        (r"ankle.?brachial", "Ankle-brachial index"),
        (
            r"blood pressure monitoring|postural blood pressure|central venous pressure",
            "Blood-pressure / CVP monitoring",
        ),
        (r"opening pressure|csf (tap|pressure)", "CSF opening pressure"),
        (
            r"cardiac catheter|hemodynamic|pressure (tracing|recording|gradient)|cardiac output",
            "Invasive hemodynamic study",
        ),
        (
            r"polysomnograph|sleep study|\bmslt\b|actigraph|actimetr",
            "Polysomnography / sleep study",
        ),
        (r"electroencephalo|\beeg\b|magnetoencephalo|\bmeg\b", "Electroencephalography"),
        (r"nerve conduction|\bncs\b|\bf.?wave\b", "Nerve conduction study"),
        (
            r"electromyogra|\bemg\b|single.?fiber|repetitive nerve stimulation|"
            r"electrodiagnostic",
            "Electromyography",
        ),
        (r"visual evoked|\bvep\b", "Visual evoked potentials"),
        (r"somatosensory evoked|\bssep\b", "Somatosensory evoked potentials"),
        (
            r"auditory brainstem|\bbaep\b|\babr\b|brainstem auditory",
            "Brainstem auditory evoked potentials",
        ),
        (r"evoked potential|blink reflex|transcranial magnetic", "Other evoked potentials"),
        (
            r"autonomic (testing|function)|sympathetic skin response|"
            r"valsalva|heart rate variability",
            "Autonomic function testing",
        ),
        (r"patch clamp|voltage clamp|electrophysiolog", "Cellular electrophysiology (research)"),
        (r"\bdlco\b|diffusing capacity", "Diffusing capacity (DLCO)"),
        (r"spirometr", "Spirometry"),
        (r"plethysmograph|lung volume", "Body plethysmography"),
        (
            r"methacholine|bronchial (challenge|provocation)|reversibility",
            "Bronchial challenge / reversibility",
        ),
        (
            r"(pulmonary|lung|respiratory) function|\bpft\b|respiratory assessment|"
            r"mechanical ventilation",
            "Pulmonary function test (composite)",
        ),
        (r"6.?minute walk|six.?minute walk", "Six-minute walk test"),
        (
            r"cardiopulmonary exercise|\bcpet\b|treadmill|exercise (stress|tolerance|test)|"
            r"stress test",
            "Exercise stress testing",
        ),
        (r"pulse oximetr|oxygen(ation)? (saturation|assessment)", "Pulse oximetry"),
        (r"blood gas|capnograph", "Blood gas / capnography"),
        (r"peak (flow|expiratory)", "Peak expiratory flow"),
        (r"visual acuity", "Visual acuity"),
        (r"visual field|perimetr", "Visual field / perimetry"),
        (r"intraocular pressure|tonometr", "Intraocular pressure"),
        (r"refraction|cycloplegic", "Refraction"),
        (r"electroretinogra|\berg\b", "Electroretinography"),
        (r"electro.?oculogra|\beog\b", "Electro-oculography"),
        (
            r"color vision|contrast sensitivity|dark adaptation|amsler|stereopsis",
            "Other psychophysical vision test",
        ),
        (
            r"ocular motility|orthoptic|hess chart|strabismus measurement",
            "Ocular motility / orthoptic assessment",
        ),
        (r"schirmer|pachymetr|tear (film|break)", "Ocular surface testing"),
        (r"tympanometr", "Tympanometry"),
        (r"otoacoustic|\boae\b", "Otoacoustic emissions"),
        (
            r"audiometr|audiogram|audiolog|hearing (test|assessment|threshold)|"
            r"speech (audiometry|discrimination)",
            "Audiometry",
        ),
        (
            r"vestibular|caloric|videonystagmograph|video.?oculograph|\bvng\b|"
            r"electronystagmograph",
            "Vestibular testing",
        ),
        (r"esophageal manometr|anorectal manometr|manometr", "Manometry"),
        (r"\bph\b (monitoring|study|metry)|impedance", "pH / impedance monitoring"),
        (r"gastric emptying|transit study|motility", "GI transit / motility study"),
        (r"breath test", "Breath test"),
        (r"urodynamic|uroflowmetr|cystometr|post.?void residual", "Urodynamic study"),
        (
            r"\bmmse\b|mini.?mental|\bmoca\b|montreal cognitive|cognitive (screen|test)",
            "Cognitive screening (MMSE / MoCA)",
        ),
        (r"intelligence quotient|\biq\b|wechsler", "Intelligence testing"),
        (r"developmental (assessment|quotient|scale)|bayley|griffiths", "Developmental assessment"),
        (
            r"neuropsycholog|psychometric|neurobehavio|memory test|battery|"
            r"\bwais\b|\bwisc\b",
            "Neuropsychological battery",
        ),
        (
            r"depression (score|scale|inventory)|\bhamd\b|\bbdi\b|\bpanss\b|"
            r"psychiatric (rating|scale)|autism|\bados\b|"
            r"scale|inventory|questionnaire|rating|\btask\b|screening test|test of",
            "Clinical rating scale / questionnaire",
        ),
        (r"speech (and language )?(assessment|evaluation)", "Speech & language assessment"),
        (r"gait (analys|kinematic)|posturograph", "Gait / posture analysis"),
        (
            r"dynamometr|muscle strength|range of motion|functional (capacity|scale)",
            "Musculoskeletal functional testing",
        ),
        (r"sweat (test|chloride)", "Sweat chloride test"),
        (
            r"tensilon|edrophonium|ice pack|neostigmine|stimulation test|"
            r"pulp vitality|vitality test",
            "Pharmacologic / provocative bedside test",
        ),
        (r"skin prick|patch test|challenge", "Allergy / provocation testing"),
        (r"urine output|fluid balance|intake and output", "Fluid-balance monitoring"),
        (r"", "Other physiologic test"),
    ]
)


PATHOLOGY_CATEGORY = _rx(
    [
        (
            r"immunohistochem|\bihc\b|immunostain|immunoperoxidase|immunolabel|"
            r"immunocytochem",
            "Immunohistochemistry",
        ),
        (r"immunofluorescen|\bdif\b|\bif microscopy\b", "Immunofluorescence"),
        (
            r"electron microscop|\bem\b ultrastructur|ultrastructur|"
            r"transmission electron|scanning electron",
            "Electron microscopy",
        ),
        (r"bone marrow|myelogram|trephine", "Hematopathology (bone marrow)"),
        (
            r"peripheral (blood )?(smear|film)|blood film|blood smear",
            "Hematopathology (blood film)",
        ),
        (
            r"cytolog|cytopatholog|fine.?needle aspiration|\bfna\b|\bfnac\b|"
            r"smear cytolog|pap smear|exfoliative|tzanck|"
            r"brush(ing)? cytolog|fluid cytolog",
            "Cytopathology",
        ),
        (
            r"gross (patholog|specimen|examination|appearance)|macroscop|"
            r"resection specimen|surgical specimen|excised (specimen|tumor)|"
            r"autopsy|post.?mortem",
            "Gross pathology",
        ),
        (
            r"\bfish\b|in.?situ hybridi|\bish\b|molecular patholog|"
            r"pcr on tissue|clonality|gene rearrangement|karyotyp",
            "Molecular / in-situ pathology",
        ),
        (
            r"special stain|periodic acid|\bpas\b|congo red|\bgms\b|grocott|"
            r"ziehl|acid.?fast|masson|trichrome|reticulin|"
            r"prussian blue|perls|oil red|alcian|giemsa|toluidine|"
            r"von kossa|silver stain|histochem",
            "Special stains & histochemistry",
        ),
        (
            r"polarized|polarizing|birefring|dark.?field|phase.?contrast|"
            r"confocal|fluorescence microscop",
            "Other microscopy",
        ),
        (
            r"histopatholog|histolog|biops|pathologic examination|"
            r"microscopic examination|surgical patholog|\bh&e\b|hematoxylin|"
            r"frozen section|light microscop|pathology|microscop",
            "Histopathology",
        ),
        (r"", "Other pathology"),
    ]
)

PATHOLOGY_CONCEPT = _rx(
    [
        (r"ki.?67|\bmib.?1\b", "IHC - proliferation index (Ki-67)"),
        (r"\bcd\d+\b|\bcd\s?\d+\b", "IHC - CD marker panel"),
        (r"cytokeratin|\bck\d|\bck\s?\d|\bema\b|epithelial membrane", "IHC - epithelial markers"),
        (r"\bs.?100\b|\bhmb.?45\b|melan.?a|sox10|melanocyt", "IHC - melanocytic markers"),
        (
            r"desmin|smooth muscle actin|\bsma\b|myogenin|myod1|caldesmon|"
            r"vimentin|muscle marker",
            "IHC - mesenchymal / muscle markers",
        ),
        (
            r"synaptophysin|chromogranin|\bnse\b|neuroendocrine marker|\bgfap\b|"
            r"neurofilament|\bnse\b",
            "IHC - neural / neuroendocrine markers",
        ),
        (
            r"estrogen receptor|progesterone receptor|\bher2\b|\ber\b/\bpr\b|"
            r"hormone receptor",
            "IHC - hormone receptors / HER2",
        ),
        (
            r"\bp53\b|\bp16\b|\bp63\b|\bbcl.?2\b|\balk\b|\bpd.?l1\b|"
            r"\bbraf\b|\bmlh1\b|\bmsh2\b|mismatch repair",
            "IHC - prognostic / predictive markers",
        ),
        (
            r"immunohistochem|\bihc\b|immunostain|immunoperoxidase|immunolabel|immunocytochem",
            "IHC - other / unspecified",
        ),
        (r"direct immunofluorescen|\bdif\b", "Direct immunofluorescence"),
        (r"indirect immunofluorescen", "Indirect immunofluorescence"),
        (r"immunofluorescen", "Immunofluorescence (unspecified)"),
        (r"transmission electron", "Transmission electron microscopy"),
        (r"scanning electron", "Scanning electron microscopy"),
        (r"electron microscop|ultrastructur", "Electron microscopy"),
        (r"bone marrow (aspirat|smear|cytolog)", "Bone marrow aspirate / smear"),
        (r"bone marrow (biops|trephine|histopatholog)|trephine", "Bone marrow trephine biopsy"),
        (r"bone marrow", "Bone marrow examination"),
        (r"peripheral (blood )?(smear|film)|blood film|blood smear", "Peripheral blood smear"),
        (r"fine.?needle aspiration|\bfna\b|\bfnac\b", "Fine-needle aspiration cytology"),
        (r"pap smear|cervical cytolog", "Cervical (Pap) cytology"),
        (
            r"(pleural|ascitic|peritoneal|pericardial|csf|urine|bronchial|"
            r"fluid) cytolog|effusion cytolog",
            "Body-fluid cytology",
        ),
        (
            r"tzanck|brush(ing)? cytolog|exfoliative|imprint|touch prep",
            "Other cytologic preparation",
        ),
        (r"cytolog|cytopatholog", "Cytology (unspecified)"),
        (r"autopsy|post.?mortem", "Autopsy examination"),
        (
            r"gross (patholog|specimen|examination|appearance)|macroscop|"
            r"resection specimen|surgical specimen|excised",
            "Gross / macroscopic pathology",
        ),
        (r"\bfish\b|fluorescence in.?situ", "FISH on tissue"),
        (r"in.?situ hybridi|\bish\b", "In-situ hybridization"),
        (
            r"clonality|gene rearrangement|molecular patholog|karyotyp|"
            r"mutation analysis on tissue",
            "Molecular pathology",
        ),
        (r"periodic acid|\bpas\b", "PAS stain"),
        (r"congo red|amyloid stain", "Congo red stain"),
        (r"\bgms\b|grocott|silver stain|methenamine", "Silver / fungal stain"),
        (r"ziehl|acid.?fast|\bafb\b", "Acid-fast stain"),
        (r"gram stain", "Gram stain"),
        (r"masson|trichrome|reticulin|van gieson", "Connective-tissue stain"),
        (r"prussian blue|perls|iron stain", "Iron stain"),
        (r"oil red|sudan|lipid stain", "Lipid stain"),
        (r"alcian|mucicarmine|mucin stain", "Mucin stain"),
        (
            r"giemsa|toluidine|von kossa|histochem|special stain|"
            r"enzyme histochem|\batpase\b|nadh|cox stain|succinate dehydrogenase",
            "Other special / histochemical stain",
        ),
        (
            r"polarized|polarizing|birefring|dark.?field|phase.?contrast",
            "Polarized / phase-contrast microscopy",
        ),
        (r"confocal|fluorescence microscop", "Fluorescence / confocal microscopy"),
        (r"light microscop", "Light microscopy"),
        (r"frozen section", "Frozen section"),
        (
            r"histopatholog|histolog|pathologic examination|microscopic examination|"
            r"surgical patholog|\bh&e\b|hematoxylin|biops|pathology|microscop",
            "Histopathology (H&E)",
        ),
        (r"", "Other pathology examination"),
    ]
)

PATH_SPECIMEN = ANATOMY


SPECIALTY_GROUP = _rx(
    [
        (r"neonatal|neonatolog", "Pediatrics & neonatology"),
        (
            r"pediatric|paediatric|adolescent|child|developmental.?behavio",
            "Pediatrics & neonatology",
        ),
        (r"neurosurg|neurointerventional|neuroradiolog", "Neurosurgery & neuro-interventional"),
        (
            r"neurolog|epilep|stroke|movement disorder|neuromuscular|"
            r"headache|neurophysiolog|neuro.?oncolog|neuroimmunolog|neurogenetic|"
            r"neuropsycholog|memory medicine|autonomic|spinal cord medicine|"
            r"cognitive neurolog|neuro.?otolog|neurotolog",
            "Neurology",
        ),
        (r"psychiatr|psychosomatic|addiction|neuropsychiatr", "Psychiatry"),
        (
            r"ophthalmolog|retina|cornea|glaucoma|uveitis|oculoplast|"
            r"strabismus|vitreoretinal|refractive surgery|ocular oncolog|orbit",
            "Ophthalmology",
        ),
        (
            r"otolaryngolog|otorhinolaryngolog|head and neck surgery|otolog|"
            r"audiolog|laryngolog|rhinolog",
            "Otolaryngology (ENT)",
        ),
        (r"dermatolog|dermatopatholog|wound care", "Dermatology"),
        (
            r"cardiolog|cardiac|cardiovascular|heart failure|electrophysiolog|"
            r"hypertension",
            "Cardiology & cardiac surgery",
        ),
        (
            r"pulmonolog|respirator|respirolog|thoracic medicine|sleep medicine|"
            r"cystic fibrosis|lung transplant|interventional pulmonolog|pulmonary medicine",
            "Pulmonology",
        ),
        (
            r"gastroenterolog|hepatolog|pancreatolog|colorectal|"
            r"upper gastrointestinal surgery|gastrointestinal surgery|"
            r"hepatobiliary|hepatopancreatobiliary|bariatric",
            "Gastroenterology & hepatology",
        ),
        (r"nephrolog|dialysis|transplant nephrolog", "Nephrology"),
        (r"urolog|andrology|genitourinary medicine|uro.?neurolog|sexual health", "Urology"),
        (
            r"endocrin|diabet|metabolic (medicine|bone)|thyroid surgery|"
            r"reproductive endocrin|clinical nutrition|nutrition",
            "Endocrinology & metabolism",
        ),
        (r"hematolog|haematolog|hematopatholog|transfusion", "Hematology"),
        (r"oncolog|cancer|radiation oncolog", "Oncology"),
        (r"rheumatolog|musculoskeletal medicine", "Rheumatology"),
        (
            r"orthop(a)?edic|spine surgery|spinal surgery|hand surgery|"
            r"sports medicine|podiatry|traumatolog|trauma surgery|chiropractic",
            "Orthopedics & trauma",
        ),
        (r"infectious|tropical medicine|travel medicine|hiv", "Infectious diseases"),
        (r"immunolog|allergy", "Allergy & clinical immunology"),
        (r"genetic|genomic", "Clinical genetics"),
        (
            r"obstetric|gynecolog|gynaecolog|maternal.?fetal|reproductive medicine|"
            r"infertility|urogynecolog",
            "Obstetrics & gynecology",
        ),
        (
            r"oral|dental|dentist|maxillofacial|orthodont|periodont|endodont|"
            r"prosthodont|stomatolog|craniofacial|orofacial|odontostomatolog|"
            r"dentofacial",
            "Oral & maxillofacial / dentistry",
        ),
        (r"radiolog|nuclear medicine|imaging", "Radiology & nuclear medicine"),
        (r"patholog|laboratory medicine", "Pathology & laboratory medicine"),
        (
            r"emergency|critical care|intensive care|acute (medicine|care)|"
            r"anesthes|anaesth|perioperative|pain medicine|resuscitat|"
            r"neurocritical|hyperbaric",
            "Emergency & critical care",
        ),
        (
            r"plastic|reconstructive|burn surgery|dermatologic surgery|"
            r"cosmetic",
            "Plastic & reconstructive surgery",
        ),
        (r"vascular", "Vascular medicine & surgery"),
        (r"transplant", "Transplant medicine"),
        (
            r"rehabilitation|physical medicine|occupational medicine|"
            r"speech.?language",
            "Rehabilitation & physical medicine",
        ),
        (r"geriatric", "Geriatrics"),
        (r"toxicolog", "Toxicology"),
        (r"surg", "General & other surgery"),
        (
            r"internal medicine|general (medicine|practice)|family medicine|"
            r"primary care|medicine",
            "Internal & general medicine",
        ),
    ]
)

ICD10_CHAPTERS = [
    ("I", "A00-B99", "Certain infectious and parasitic diseases"),
    ("II", "C00-D48", "Neoplasms"),
    ("III", "D50-D89", "Diseases of the blood and immune mechanism"),
    ("IV", "E00-E90", "Endocrine, nutritional and metabolic diseases"),
    ("V", "F00-F99", "Mental and behavioural disorders"),
    ("VI", "G00-G99", "Diseases of the nervous system"),
    ("VII", "H00-H59", "Diseases of the eye and adnexa"),
    ("VIII", "H60-H95", "Diseases of the ear and mastoid process"),
    ("IX", "I00-I99", "Diseases of the circulatory system"),
    ("X", "J00-J99", "Diseases of the respiratory system"),
    ("XI", "K00-K93", "Diseases of the digestive system"),
    ("XII", "L00-L99", "Diseases of the skin and subcutaneous tissue"),
    ("XIII", "M00-M99", "Diseases of the musculoskeletal system and connective tissue"),
    ("XIV", "N00-N99", "Diseases of the genitourinary system"),
    ("XV", "O00-O99", "Pregnancy, childbirth and the puerperium"),
    ("XVI", "P00-P96", "Certain conditions originating in the perinatal period"),
    ("XVII", "Q00-Q99", "Congenital malformations and chromosomal abnormalities"),
    ("XVIII", "R00-R99", "Symptoms, signs and abnormal clinical findings, NEC"),
    ("XIX", "S00-T98", "Injury, poisoning and external causes"),
    ("XX", "V01-Y98", "External causes of morbidity and mortality"),
    ("XXI", "Z00-Z99", "Factors influencing health status"),
    ("XXII", "U00-U99", "Codes for special purposes"),
]

_CODE_RX = re.compile(r"^([A-Z])(\d{2})")


def icd10_chapter(code: str):
    """Map an ICD-10 code (e.g. 'Q77.3', 'A16') to (roman, range, title)."""
    if not code:
        return None
    m = _CODE_RX.match(code.strip().upper())
    if not m:
        return None
    letter, num = m.group(1), int(m.group(2))
    for roman, rng, title in ICD10_CHAPTERS:
        lo, hi = rng.split("-")
        l0, n0 = lo[0], int(lo[1:])
        l1, n1 = hi[0], int(hi[1:])
        if l0 == l1:
            if letter == l0 and n0 <= num <= n1:
                return roman, rng, title
        else:
            if (letter == l0 and num >= n0) or (letter == l1 and num <= n1):
                return roman, rng, title
            if l0 < letter < l1:
                return roman, rng, title
    return None
