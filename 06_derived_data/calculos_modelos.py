"""Explanatory layer of the bibliometric profile (read-only; does not run or modify the analysis tool).
Five association models aligned with the research questions (search-string blocks of protocol v3.1):
  M1 (RQ1)  annual production trend, negative binomial on yearly counts 2014-2025 (2026 is partial and excluded)
  M2 (RQ2)  logistic: document names technical competencies ~ year + channel
  M3 (RQ3)  logistic: document names performance/observation instrument terms ~ technical + transversal + channel + year
  M4 (RQ4)  logistic: document cites a classical psychometric root ~ channel + year + transversal + technical + self-report + log(1+references)
  M5 (RQ5)  negative binomial: citations ~ publication-year factor + channel + open access + multi-country + specific
            measurement-property terms + cites psychometric root + log(1+references)
Sources (all read-only):
  - G:/AUTOMATIZACIONES/biblio140302_11/output/reports/bibliometric_report.xlsx (sheet Articles: Key, Year, Document_Type,
    Citations, Open_Access)  -> citations, channel, year and open-access status as reported by the tool
  - G:/AUTOMATIZACIONES/biblio140302_11/data/modelo_corpus.json (articulos[key].paises)  -> country set used by the tool for MCP
  - G:/AUTOMATIZACIONES/biblio140302_11/data/enriched_articles.json (references_resolved, deduplicated per document-work)
  - ./calculos_bloques_cadena.json (per_document: competency, instrument and measurement-property families; psychometric roots)
Associations only; no causal interpretation. Run with a Python that has statsmodels:  py -3 -X utf8 -I calculos_modelos.py"""
import json, pathlib, re, warnings
import numpy as np, pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.simplefilter("ignore")
HERE = pathlib.Path(__file__).resolve().parent
TOOL = pathlib.Path(r"G:\AUTOMATIZACIONES\biblio140302_11")

x = pd.read_excel(TOOL / "output" / "reports" / "bibliometric_report.xlsx", sheet_name="Articles")[["Key", "Year", "Document_Type", "Citations", "Open_Access"]]
modelo = json.load(open(TOOL / "data" / "modelo_corpus.json", encoding="utf-8"))["articulos"]
arts = json.load(open(TOOL / "data" / "enriched_articles.json", encoding="utf-8"))
arts = arts if isinstance(arts, list) else arts["articles"]
B = json.load(open(HERE / "calculos_bloques_cadena.json", encoding="utf-8"))["per_document"]

def nref(a):
    seen = set()
    for r in a.get("references_resolved") or []:
        w = str(r.get("openalex_id") or r.get("id") or r.get("doi") or r.get("title") or "")
        w = w.split("/")[-1].lower()
        if w: seen.add(w)
    return len(seen)
REF = {a["key"]: nref(a) for a in arts}
ABW = {a["key"]: len(re.findall(r"\w+", str(a.get("abstract") or ""))) for a in arts}   # abstract length in words
SPECIFIC = ["reliability (any)", "internal consistency/alpha", "validity (any)", "psychometric*", "content validity/experts",
            "factor analysis (EFA/CFA)", "structural equation modelling", "criterion/convergent/discriminant",
            "item response theory/Rasch", "measurement invariance", "test-retest/inter-rater"]
rows = []
for d in B:
    k = d["key"]; xr = x[x.Key == k].iloc[0]
    rows.append(dict(key=k, year=int(xr.Year), journal=int(xr.Document_Type == "Journal Article"),
                     citations=int(xr.Citations), oa=int(str(xr.Open_Access).lower() not in ("closed", "nan", "", "false", "none")),
                     mcp=int(len(modelo.get(k, {}).get("paises") or []) >= 2),
                     technical=int("technical" in d["comp_groups"]), transversal=int("transversal" in d["comp_groups"]),
                     selfreport=int("self-report" in d["inst_groups"]), performance=int("performance/observation" in d["inst_groups"]),
                     specific=int(any(f in d["prop"] for f in SPECIFIC)), roots=int(bool(d["roots"])),
                     has_refs=int(d["has_refs"]), nrefs=REF.get(k, 0), abw=ABW.get(k, 0)))
df = pd.DataFrame(rows); assert len(df) == 347
df["yc"] = df.year - 2020                     # centred year
df["lref"] = np.log1p(df.nrefs); df["labw"] = np.log1p(df.abw)

def table(res, exp=True, drop=("Intercept",)):
    ci = res.conf_int(); out = {}
    for n in res.params.index:
        if n in drop or n.startswith("C(year") or n == "alpha": continue
        f = np.exp if exp else (lambda v: v)
        out[n] = {"estimate": round(float(f(res.params[n])), 3), "ci_low": round(float(f(ci.loc[n, 0])), 3),
                  "ci_high": round(float(f(ci.loc[n, 1])), 3), "p": round(float(res.pvalues[n]), 4)}
    return out

R = {"n": int(len(df)), "notes": "estimates are exponentiated (odds ratios for logistic, rate ratios for negative binomial) with 95% Wald intervals"}
# M1 annual trend 2014-2025
yc = df[df.year <= 2025].groupby("year").size().reindex(range(2014, 2026), fill_value=0).reset_index(name="n")
yc["t"] = yc.year - 2014
m1 = smf.negativebinomial("n ~ t", data=yc).fit(disp=0, maxiter=500)            # dispersion estimated
m1b = smf.negativebinomial("n ~ t", data=yc[yc.year <= 2024]).fit(disp=0, maxiter=500)
pred25 = float(np.exp(m1b.params["Intercept"] + m1b.params["t"] * 11))
R["M1_trend"] = {"yearly_counts": dict(zip(yc.year.astype(int).astype(str), yc.n.astype(int))),
                 "rate_ratio_per_year_2014_2025": table(m1)["t"], "rate_ratio_per_year_2014_2024": table(m1b)["t"],
                 "expected_2025_from_2014_2024_trend": round(pred25, 1), "observed_2025": int(yc[yc.year == 2025].n.iloc[0])}
# M2 technical ~ year + channel
m2 = smf.logit("technical ~ yc + journal", data=df).fit(disp=0); R["M2_technical"] = table(m2)
# M3 performance ~ technical + transversal + journal + year
m3 = smf.logit("performance ~ technical + transversal + journal + yc", data=df).fit(disp=0); R["M3_performance"] = table(m3)
# M4 roots ~ ... among documents with references
d4 = df[df.has_refs == 1]
m4 = smf.logit("roots ~ journal + yc + transversal + technical + selfreport + lref", data=d4).fit(disp=0)
m4i = smf.logit("roots ~ journal * yc + transversal + technical + selfreport + lref", data=d4).fit(disp=0)
m4u = smf.logit("roots ~ journal", data=d4).fit(disp=0)
R["M4_roots"] = {"n": int(len(d4)), "unadjusted_journal": table(m4u)["journal"], "main": table(m4),
                 "interaction_journal_x_year": table(m4i).get("journal:yc"),
                 "median_references": {"journal": float(d4[d4.journal == 1].nrefs.median()), "conference": float(d4[d4.journal == 0].nrefs.median())}}
# M4b specific measurement-property vocabulary ~ channel, adjusted for abstract length
m4bu = smf.logit("specific ~ journal", data=df).fit(disp=0)
m4b = smf.logit("specific ~ journal + labw + yc + transversal + technical", data=df).fit(disp=0)
R["M4b_specific_terms"] = {"unadjusted_journal": table(m4bu)["journal"], "adjusted": table(m4b),
                           "median_abstract_words": {"journal": float(df[df.journal == 1].abw.median()), "conference": float(df[df.journal == 0].abw.median())}}
# M5 citations (negative binomial with estimated dispersion), year as factor = exposure control
m5 = smf.negativebinomial("citations ~ C(year) + journal + oa + mcp + specific + roots + lref", data=df).fit(disp=0, maxiter=500)
R["M5_citations"] = {"n": int(len(df)), "alpha": round(float(m5.params["alpha"]), 3), "effects": table(m5)}
# full specification of each model for the supplementary table (n, events, reference categories, dispersion, all terms)
def detail(res, kind, outcome, data, events=None, ref="", note=""):
    ci = res.conf_int(); terms = []
    for n in res.params.index:
        if n == "alpha" or n.startswith("C(year"): continue
        terms.append({"term": n, "estimate": float(f"{np.exp(res.params[n]):.6g}"), "ci_low": float(f"{np.exp(ci.loc[n, 0]):.6g}"),
                      "ci_high": float(f"{np.exp(ci.loc[n, 1]):.6g}"), "p": round(float(res.pvalues[n]), 4)})
    d = {"type": kind, "outcome": outcome, "n": int(len(data)), "terms": terms, "reference": ref, "note": note,
         "log_likelihood": round(float(res.llf), 2), "aic": round(float(res.aic), 1)}
    if events is not None: d["events"] = int(events); d["mcfadden_r2"] = round(float(res.prsquared), 3)
    if "alpha" in res.params.index: d["dispersion_alpha"] = round(float(res.params["alpha"]), 3)
    return d
R["model_details"] = {
 "M1": detail(m1b, "negative binomial", "documents per year, 2014-2024", yc[yc.year <= 2024], ref="t = years since 2014",
              note=f"2025 expected from the 2014-2024 trend: {round(pred25, 1)}; observed: 76"),
 "M2": detail(m2, "logistic", "names technical competencies", df, events=df.technical.sum(), ref="conference paper; yc = year - 2020"),
 "M3": detail(m3, "logistic", "names performance or observation instrument terms", df, events=df.performance.sum(), ref="conference paper; not naming the competency group; yc = year - 2020"),
 "M4": detail(m4, "logistic", "cites at least one classical psychometric root", d4, events=d4.roots.sum(), ref="conference paper; not naming the group; lref = log(1 + resolved references)"),
 "M4b": detail(m4b, "logistic", "uses specific measurement-property terms", df, events=df.specific.sum(), ref="conference paper; labw = log(1 + abstract words)"),
 "M5": detail(m5, "negative binomial", "citations", df, ref="conference paper, closed access, single country; publication year as factor (reference 2014; coefficients not shown)"),
}
R["descriptives"] = {k: int(df[k].sum()) for k in ["journal", "oa", "mcp", "technical", "transversal", "selfreport", "performance", "specific", "roots"]}
# per-document analysis table (derived variables only; no abstracts or licensed record content), for the data repository
exp = df.copy()
fam = {d["key"]: d for d in B}
for col, src in (("competency_families", "comp"), ("instrument_families", "inst"), ("property_families", "prop"), ("psychometric_roots_cited", "roots")):
    exp[col] = exp.key.map(lambda k: "; ".join(fam[k][src]))
doi = dict(zip(x.Key, x.get("DOI", pd.Series(dtype=str)))) if "DOI" in x.columns else {}
exp.rename(columns={"journal": "journal_article", "oa": "open_access", "mcp": "multi_country", "selfreport": "self_report_terms",
                    "performance": "performance_terms", "specific": "specific_property_terms", "roots": "cites_psychometric_root",
                    "nrefs": "n_resolved_references_dedup", "abw": "abstract_words"}).drop(columns=["yc", "lref", "labw"]).to_csv(
    HERE / "document_variables_347.csv", index=False, encoding="utf-8")
(HERE / "calculos_modelos.json").write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in R.items() if k != "notes"}, ensure_ascii=False, indent=1))
