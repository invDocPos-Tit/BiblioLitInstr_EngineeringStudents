"""Field-normalised impact by OpenAlex subject field (read-only; does not run or modify the analysis tool).
Source: output/leiden/mncs_detail.json, which gives for each document its OpenAlex field, its normalised score and the
normalisation used: 'openalex_world_fwci' (OpenAlex FWCI, world baseline; 256 documents) or 'cohort_field_year'
(normalisation within the corpus by field and year, applied by the tool to the 91 documents without an FWCI).
The article uses the FWCI subset as the main measure; the mixed MNCS (all 332 documents with a field) is kept only as a check.
Run: python -X utf8 -I calculos_fwci_campo.py  ->  calculos_fwci_campo.json"""
import collections, json, pathlib, statistics as st
HERE = pathlib.Path(__file__).resolve().parent
m = json.load(open(r"G:\AUTOMATIZACIONES\biblio140302_11\output\leiden\mncs_detail.json", encoding="utf-8"))
def summary(rows):
    by = collections.defaultdict(list)
    for x in rows: by[x["field"]].append(float(x["mncs"]))
    return {f: {"n": len(v), "median": round(st.median(v), 2), "mean": round(st.mean(v), 2),
                "percent_above_1": round(100 * sum(a > 1 for a in v) / len(v), 1)} for f, v in sorted(by.items(), key=lambda kv: -len(kv[1]))}
fw = [x for x in m if x["normalization_source"] == "openalex_world_fwci"]
co = [x for x in m if x["normalization_source"] != "openalex_world_fwci"]
allf = [x for x in m if x["field"] != "Unknown"]
v = lambda rows: [float(x["mncs"]) for x in rows]
R = {"fwci_openalex_by_field": summary(fw),
     "fwci_openalex_overall": {"n": len(fw), "median": round(st.median(v(fw)), 3), "mean": round(st.mean(v(fw)), 3)},
     "within_corpus_normalisation": {"n": len(co), "median": round(st.median(v(co)), 3), "mean": round(st.mean(v(co)), 3),
                                     "fields": dict(collections.Counter(x["field"] for x in co))},
     "mixed_mncs_check": {"n": len(allf), "median": round(st.median(v(allf)), 3), "mean": round(st.mean(v(allf)), 3),
                          "by_field": summary(allf)}}
assert len(fw) == 256 and len(co) == 91 and len(allf) == 332
(HERE / "calculos_fwci_campo.json").write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: R[k] for k in ("fwci_openalex_overall", "within_corpus_normalisation")}, indent=1))
print({f: R["fwci_openalex_by_field"][f] for f in list(R["fwci_openalex_by_field"])[:6]})
