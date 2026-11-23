"""Read-only analysis of the search-string blocks (protocol v3.1) in titles, abstracts and author keywords of the 347 documents.
Answers the string-aligned research questions:
  PI2 (block 2): which technical and transversal competencies the literature names and how they are distributed (year, channel);
  PI3 (block 3): which instrument types co-occur with those competencies;
  PI4 (block 4): measurement-property vocabulary, and the share of documents citing the classical psychometric roots, by channel and period.
Term families come from the string blocks and the protocol definitions (technical: design, simulation, programming, technical problem
solving, prototyping; transversal: critical thinking, collaboration, communication, ethics, self-regulation, employability, self-efficacy).
It describes what the documents NAME; it does not code what each instrument measures nor appraise any measurement property.
Every record met the four blocks in its database, so presence of block terms is guaranteed; the informative part is the composition.
Reads G:/AUTOMATIZACIONES/biblio140302_11/data/enriched_articles.json without modifying the tool.
Run: python -X utf8 -I calculos_bloques_cadena.py   ->  calculos_bloques_cadena.json and calculos_bloques_cadena.md"""
import collections, json, pathlib, re
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
TOOL = pathlib.Path(r"G:\AUTOMATIZACIONES\biblio140302_11")
arts = json.load(open(TOOL / "data" / "enriched_articles.json", encoding="utf-8"))
arts = arts if isinstance(arts, list) else arts["articles"]
assert len(arts) == 347

def text(a):
    kw = a.get("author_keywords") or []
    kw = kw if isinstance(kw, list) else [kw]
    return " ".join([str(a.get("title") or ""), str(a.get("abstract") or ""), " ".join(map(str, kw))]).lower()

# ---- dictionaries (regex on lower-case title + abstract + author keywords)
COMP = {  # family: (group, regex)
    "design":                  ("technical",   r"\bengineering design\b|\bdesign (skill|competenc|abilit|self[- ]?efficac|thinking)"),
    "programming/computational": ("technical", r"\bprogramming\b|\bcoding (skill|abilit)|\bcomputational thinking"),
    "simulation/modelling":    ("technical",   r"\bsimulation (skill|competenc)|\bmodell?ing (skill|competenc)"),
    "problem solving":         ("technical",   r"\bproblem[- ]solving"),
    "prototyping":             ("technical",   r"\bprototyp"),
    "technical/engineering skills (generic)": ("technical", r"\btechnical (skill|competenc)|\bengineering (skill|competenc)"),
    "self-efficacy":           ("transversal", r"\bself[- ]?efficac"),
    "critical thinking":       ("transversal", r"\bcritical thinking"),
    "communication":           ("transversal", r"\bcommunication (skill|competenc|abilit)|\b(oral|written) communication|\bcommunicative competenc"),
    "teamwork/collaboration":  ("transversal", r"\bteam ?work|\bteam (skill|competenc|effectiveness)|\bcollaborat\w* (skill|competenc|problem)"),
    "ethics":                  ("transversal", r"\bethic"),
    "self-regulation":         ("transversal", r"\bself[- ]regulat|\bmetacogniti"),
    "employability/professional": ("transversal", r"\bemployabilit|\bprofessional (skill|competenc)"),
    "soft/generic/transversal (generic)": ("transversal", r"\bsoft skill|\btransversal (skill|competenc)|\bgeneric (skill|competenc)"),
    "creativity/innovation":   ("transversal", r"\bcreativ|\binnovati\w* (competenc|skill|behaviou?r|capabilit)"),
    "leadership":              ("transversal", r"\bleadership"),
    "entrepreneurship":        ("transversal", r"\bentrepreneur"),
}
INST = {  # instrument-type family: (group, regex)
    "questionnaire":   ("self-report",  r"\bquestionnaire"),
    "scale":           ("self-report",  r"(?<!large)(?<!small)(?<!large-)(?<!small-)\bscales?\b(?![- ]up)"),
    "inventory (non-concept)": ("self-report", r"(?<!concept )\binventor(y|ies)\b"),
    "survey":          ("self-report",  r"\bsurvey"),
    "self-report/self-assessment": ("self-report", r"\bself[- ]report|\bself[- ]assess|\blikert"),
    "rubric":          ("performance/observation", r"\brubric"),
    "concept inventory/test": ("performance/observation", r"\bconcept inventor|\bperformance[- ]based|\bperformance (task|assessment|test)|\bsituational judg"),
    "observation/portfolio": ("performance/observation", r"\bobservation (protocol|instrument)|\bportfolio"),
}
PROP = {  # block-4 vocabulary families
    "reliability (any)":       r"\breliab",
    "internal consistency/alpha": r"\binternal consistency|\bcronbach|\bcoefficient alpha|\bmcdonald|\bomega\b",
    "validity (any)":          r"\bvalidit",
    "validation (validat*)":   r"\bvalidat",
    "psychometric*":           r"\bpsychometric",
    "content validity/experts": r"\bcontent validit|\bexpert (panel|judg|review|validation)|\bdelphi",
    "factor analysis (EFA/CFA)": r"\bfactor analys|\bfactorial (structure|validit)|\bexploratory factor|\bconfirmatory factor|\befa\b|\bcfa\b",
    "structural equation modelling": r"\bstructural equation|\bpls[- ]sem\b|\bsem\b",
    "criterion/convergent/discriminant": r"\bcriterion validit|\bconvergent|\bdiscriminant|\bconcurrent validit|\bpredictive validit",
    "item response theory/Rasch": r"\bitem response|\birt\b|\brasch",
    "measurement invariance": r"\binvarian",
    "test-retest/inter-rater": r"\btest[- ]retest|\binter[- ]?rater|\bintraclass|\bicc\b|\bkappa\b",
}
ROOTS = {  # classical psychometric roots (cited base, Section 3.2), identified by title in the resolved references
    "Cronbach (1951)": "coefficient alpha and the internal structure",
    "Kaiser (1960)": "application of electronic computers to factor analysis",
    "Cattell (1966)": "scree test for the number of factors",
    "Fornell & Larcker (1981)": "evaluating structural equation models with unobservable",
    "Hu & Bentler (1999)": "cutoff criteria for fit",
}
PERIODS = [(2014, 2019), (2020, 2024), (2025, 2026)]
pl = lambda y: next(f"{a}-{b}" for a, b in PERIODS if a <= y <= b)
TYPE = {"Journal Article": "journal", "Conference Paper": "conference"}

docs = []
for a in arts:
    t = text(a)
    refs = " || ".join(str(r.get("title") or "").lower() for r in (a.get("references_resolved") or []))
    d = {"key": a["key"], "year": int(a["publication_year"]), "type": TYPE.get(a.get("item_type"), a.get("item_type")),
         "has_abstract": bool(str(a.get("abstract") or "").strip()), "has_refs": bool(a.get("references_resolved")),
         "comp": [f for f, (_, rx) in COMP.items() if re.search(rx, t)],
         "inst": [f for f, (_, rx) in INST.items() if re.search(rx, t)],
         "prop": [f for f, rx in PROP.items() if re.search(rx, t)],
         "roots": [k for k, s in ROOTS.items() if s in refs]}
    d["comp_groups"] = sorted({COMP[f][0] for f in d["comp"]})
    d["inst_groups"] = sorted({INST[f][0] for f in d["inst"]})
    d["period"] = pl(d["year"])
    docs.append(d)

rng = np.random.default_rng(42); B = 10000
def prop_ci(x):
    x = np.asarray(x, float)
    if len(x) == 0: return None
    s = rng.choice(x, (B, len(x))).mean(1)
    return [round(100 * x.mean(), 1), round(100 * np.percentile(s, 2.5), 1), round(100 * np.percentile(s, 97.5), 1), int(len(x))]
def diff_ci(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    s = rng.choice(x, (B, len(x))).mean(1) - rng.choice(y, (B, len(y))).mean(1)
    return [round(100 * (x.mean() - y.mean()), 1), round(100 * np.percentile(s, 2.5), 1), round(100 * np.percentile(s, 97.5), 1)]

def cg(d):  # competency group of a document
    g = d["comp_groups"]
    return "both" if g == ["technical", "transversal"] else (g[0] if g else "none")

R = {"n_docs": len(docs), "n_with_abstract": sum(d["has_abstract"] for d in docs), "n_with_refs": sum(d["has_refs"] for d in docs),
     "dictionaries": {"competencies": {k: v for k, v in COMP.items()}, "instruments": {k: v for k, v in INST.items()},
                      "properties": PROP, "roots": ROOTS}}
# PI2
R["PI2_families"] = {f: sum(f in d["comp"] for d in docs) for f in COMP}
R["PI2_groups"] = dict(collections.Counter(cg(d) for d in docs))
R["PI2_group_by_period"] = {p: dict(collections.Counter(cg(d) for d in docs if d["period"] == p)) for p in [f"{a}-{b}" for a, b in PERIODS]}
R["PI2_group_by_type"] = {t: dict(collections.Counter(cg(d) for d in docs if d["type"] == t)) for t in ("journal", "conference")}
R["PI2_families_by_period"] = {f: {p: sum(f in d["comp"] for d in docs if d["period"] == p) for p in R["PI2_group_by_period"]} for f in COMP}
R["PI2_transversal_share_journal_vs_conference"] = {
    "journal": prop_ci([("transversal" in d["comp_groups"]) for d in docs if d["type"] == "journal"]),
    "conference": prop_ci([("transversal" in d["comp_groups"]) for d in docs if d["type"] == "conference"])}
R["PI2_technical_share_journal_vs_conference"] = {
    "journal": prop_ci([("technical" in d["comp_groups"]) for d in docs if d["type"] == "journal"]),
    "conference": prop_ci([("technical" in d["comp_groups"]) for d in docs if d["type"] == "conference"])}
# PI3
R["PI3_inst_families"] = {f: sum(f in d["inst"] for d in docs) for f in INST}
R["PI3_inst_groups"] = dict(collections.Counter("+".join(d["inst_groups"]) or "none" for d in docs))
R["PI3_family_x_instgroup"] = {f: {g: sum(f in d["comp"] and g in d["inst_groups"] for d in docs) for g in ("self-report", "performance/observation")} | {"n": R["PI2_families"][f]} for f in COMP}
R["PI3_compgroup_x_instgroup"] = {c: {"n": sum(cg(d) == c for d in docs),
                                      "self-report": prop_ci([("self-report" in d["inst_groups"]) for d in docs if cg(d) == c]),
                                      "performance/observation": prop_ci([("performance/observation" in d["inst_groups"]) for d in docs if cg(d) == c])}
                                  for c in ("technical", "transversal", "both", "none")}
R["PI3_performance_tech_vs_transversal_only"] = diff_ci([("performance/observation" in d["inst_groups"]) for d in docs if cg(d) == "technical"],
                                                        [("performance/observation" in d["inst_groups"]) for d in docs if cg(d) == "transversal"])
R["PI3_instgroup_by_type"] = {t: {g: prop_ci([(g in d["inst_groups"]) for d in docs if d["type"] == t]) for g in ("self-report", "performance/observation")} for t in ("journal", "conference")}
# PI4
R["PI4_prop_families"] = {f: sum(f in d["prop"] for d in docs) for f in PROP}
R["PI4_prop_by_type"] = {f: {t: prop_ci([(f in d["prop"]) for d in docs if d["type"] == t]) for t in ("journal", "conference")} for f in PROP}
R["PI4_prop_by_period"] = {f: {p: sum(f in d["prop"] for d in docs if d["period"] == p) for p in R["PI2_group_by_period"]} for f in PROP}
R["PI4_period_sizes"] = dict(collections.Counter(d["period"] for d in docs))
R["PI4_no_block4_term"] = sum(not d["prop"] for d in docs)
wr = [d for d in docs if d["has_refs"]]
R["PI4_roots_each"] = {k: sum(k in d["roots"] for d in wr) for k in ROOTS}
R["PI4_cites_any_root"] = {"all": prop_ci([bool(d["roots"]) for d in wr]),
                           **{t: prop_ci([bool(d["roots"]) for d in wr if d["type"] == t]) for t in ("journal", "conference")},
                           **{p: prop_ci([bool(d["roots"]) for d in wr if d["period"] == p]) for p in R["PI2_group_by_period"]},
                           **{f"{t} {p}": prop_ci([bool(d["roots"]) for d in wr if d["type"] == t and d["period"] == p])
                              for t in ("journal", "conference") for p in R["PI2_group_by_period"]}}
R["PI4_roots_journal_minus_conference"] = diff_ci([bool(d["roots"]) for d in wr if d["type"] == "journal"], [bool(d["roots"]) for d in wr if d["type"] == "conference"])
R["PI4_roots_by_compgroup"] = {c: prop_ci([bool(d["roots"]) for d in wr if cg(d) == c]) for c in ("technical", "transversal", "both", "none")}
R["PI4_roots_by_instgroup"] = {g: prop_ci([bool(d["roots"]) for d in wr if g in d["inst_groups"]]) for g in ("self-report", "performance/observation")}
R["per_document"] = docs

(HERE / "calculos_bloques_cadena.json").write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding="utf-8")
# compact summary
L = ["# Blocks of the search string: summary (read-only; seed 42; 10,000 bootstrap replicates; values = % [95% CI], n)\n"]
for k in [k for k in R if k.startswith("PI")]:
    L.append(f"## {k}\n```\n{json.dumps(R[k], ensure_ascii=False)}\n```\n")
(HERE / "calculos_bloques_cadena.md").write_text("\n".join(L), encoding="utf-8")
print("ok")
