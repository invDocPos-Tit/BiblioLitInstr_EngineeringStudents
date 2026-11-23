"""Measurement-property families of the search-string block read against the sources of validity evidence of the
Standards for Educational and Psychological Testing (AERA, APA & NCME, 2014). Read-only regrouping of the per-document
families already computed in calculos_bloques_cadena.json (dictionaries in Supplementary material D1); no new term matching.
The dictionaries have no terms for response processes or consequences of testing, so those sources are not reported.
Run: python -X utf8 -I calculos_fuentes_validez.py  ->  calculos_fuentes_validez.json"""
import json, pathlib
HERE = pathlib.Path(__file__).resolve().parent
docs = json.load(open(HERE / "calculos_bloques_cadena.json", encoding="utf-8"))["per_document"]
SOURCES = {
    "test content": ["content validity/experts"],
    "internal structure": ["factor analysis (EFA/CFA)", "structural equation modelling", "measurement invariance", "item response theory/Rasch"],
    "relations to other variables": ["criterion/convergent/discriminant"],
    "reliability (specific terms)": ["internal consistency/alpha", "test-retest/inter-rater"],
    "generic terms only": None,   # documents whose only block-4 terms are reliability, validity, validation or psychometric
}
GENERIC = {"reliability (any)", "validity (any)", "validation (validat*)", "psychometric*"}
def has(d, s):
    fam = SOURCES[s]
    if fam is None: return bool(d["prop"]) and set(d["prop"]) <= GENERIC
    return any(f in d["prop"] for f in fam)
def group(d):
    g = d["comp_groups"]
    return "technical only" if g == ["technical"] else "transversal only" if g == ["transversal"] else "both" if set(g) == {"technical", "transversal"} else "none"
subsets = {"all": docs, "journal": [d for d in docs if d["type"] == "journal"], "conference": [d for d in docs if d["type"] != "journal"]}
for g in ("technical only", "transversal only"): subsets[g] = [d for d in docs if group(d) == g]
subsets["self-report terms"] = [d for d in docs if "self-report" in d["inst_groups"]]
subsets["performance or observation terms"] = [d for d in docs if "performance/observation" in d["inst_groups"]]
R = {"mapping": {k: v for k, v in SOURCES.items()}, "generic_families": sorted(GENERIC), "by_subset": {}}
for name, ds in subsets.items():
    R["by_subset"][name] = {"n": len(ds), **{s: [sum(has(d, s) for d in ds), round(100 * sum(has(d, s) for d in ds) / len(ds), 1)] for s in SOURCES},
                            "two or more specific sources": [sum(sum(has(d, s) for s in list(SOURCES)[:4]) >= 2 for d in ds),
                                                             round(100 * sum(sum(has(d, s) for s in list(SOURCES)[:4]) >= 2 for d in ds) / len(ds), 1)]}
(HERE / "calculos_fuentes_validez.json").write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding="utf-8")
for k, v in R["by_subset"].items(): print(k, v)
