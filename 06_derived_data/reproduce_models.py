"""Refits the five association models of the article (M1-M5) from document_variables_347.csv alone, so that they can be
reproduced without the licensed bibliographic records. Requires pandas, numpy and statsmodels (tested with statsmodels 0.14.6).
Run from this folder:  python reproduce_models.py
The estimates should match calculos_modelos.json (odds ratios or rate ratios with 95% Wald intervals)."""
import json, pathlib, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf

warnings.simplefilter("ignore")
HERE = pathlib.Path(__file__).resolve().parent
df = pd.read_csv(HERE / "document_variables_347.csv")
df["journal"] = df.journal_article; df["oa"] = df.open_access; df["mcp"] = df.multi_country
df["selfreport"] = df.self_report_terms; df["performance"] = df.performance_terms
df["specific"] = df.specific_property_terms; df["roots"] = df.cites_psychometric_root
df["yc"] = df.year - 2020; df["lref"] = np.log1p(df.n_resolved_references_dedup); df["labw"] = np.log1p(df.abstract_words)

def table(res):
    ci = res.conf_int(); out = {}
    for n in res.params.index:
        if n == "Intercept" or n.startswith("C(year") or n == "alpha": continue
        out[n] = [round(float(np.exp(res.params[n])), 3), round(float(np.exp(ci.loc[n, 0])), 3), round(float(np.exp(ci.loc[n, 1])), 3)]
    return out

R = {}
yc = df[df.year <= 2024].groupby("year").size().reindex(range(2014, 2025), fill_value=0).reset_index(name="n"); yc["t"] = yc.year - 2014
R["M1 annual counts 2014-2024 (rate ratio per year)"] = table(smf.negativebinomial("n ~ t", data=yc).fit(disp=0, maxiter=500))
R["M2 technical competencies"] = table(smf.logit("technical ~ yc + journal", data=df).fit(disp=0))
R["M3 performance or observation terms"] = table(smf.logit("performance ~ technical + transversal + journal + yc", data=df).fit(disp=0))
d4 = df[df.has_refs == 1]
R["M4 cites a psychometric root"] = table(smf.logit("roots ~ journal + yc + transversal + technical + selfreport + lref", data=d4).fit(disp=0))
R["M4b specific measurement-property terms"] = table(smf.logit("specific ~ journal + labw + yc + transversal + technical", data=df).fit(disp=0))
R["M5 citations"] = table(smf.negativebinomial("citations ~ C(year) + journal + oa + mcp + specific + roots + lref", data=df).fit(disp=0, maxiter=500))
print(json.dumps(R, indent=1))
