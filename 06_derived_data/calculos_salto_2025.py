"""Composition of the 2025-2026 increase compared with 2014-2024 (read-only; does not run or modify the analysis tool).
Describes which sources, channels, source databases and competency families make up the documents of each period;
it does not identify causes.
Sources: output/reports/bibliometric_report.xlsx (Key, Year, Journal, Document_Type); data/enriched_articles.json (merged_from,
the databases each document was retrieved from, as counted by the tool: Scopus 251 in total); ./calculos_bloques_cadena.json
(competency families); output/impact_evolution/impact_evolution_report.md (OpenAlex topic life cycle).
Run: python -X utf8 -I calculos_salto_2025.py  ->  calculos_salto_2025.json"""
import collections, json, pathlib, re
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
TOOL = pathlib.Path(r"G:\AUTOMATIZACIONES\biblio140302_11")
x = pd.read_excel(TOOL / "output/reports/bibliometric_report.xlsx", sheet_name="Articles")[["Key", "Year", "Journal", "Document_Type"]]
arts = json.load(open(TOOL / "data/enriched_articles.json", encoding="utf-8")); arts = arts if isinstance(arts, list) else arts["articles"]
merged = {a["key"]: str(a.get("merged_from") or "") for a in arts}
B = {b["key"]: b for b in json.load(open(HERE / "calculos_bloques_cadena.json", encoding="utf-8"))["per_document"]}
x["period"] = x.Year.apply(lambda y: "2025-2026" if y >= 2025 else "2014-2024")
DBS = ["Scopus", "WoS", "IEEE", "EBSCO", "ERIC", "Dimensions"]
FAM = ["self-efficacy", "problem solving", "critical thinking", "creativity/innovation", "programming/computational", "ethics",
       "technical/engineering skills (generic)", "communication", "teamwork/collaboration"]
R = {}
for p, g in x.groupby("period"):
    n = len(g); vc = g.Journal.value_counts(); fams = collections.Counter(f for k in g.Key for f in B[k]["comp"])
    R[p] = {"documents": n, "journal_articles": int((g.Document_Type == "Journal Article").sum()),
            "conference_papers": int((g.Document_Type == "Conference Paper").sum()),
            "distinct_sources": int(g.Journal.nunique()), "sources_with_one_document": int((vc == 1).sum()),
            "asee_annual_conference": int(vc.get("ASEE Annual Conference and Exposition, Conference Proceedings", 0)),
            "retrieved_from": {db: int(sum(db in merged[k] for k in g.Key)) for db in DBS},
            "not_retrieved_from_scopus": int(sum("Scopus" not in merged[k] for k in g.Key)),
            "competency_families": {f: int(fams[f]) for f in FAM}}
    R[p]["percent"] = {"asee": round(100 * R[p]["asee_annual_conference"] / n, 1),
                       **{f"retrieved_{db}": round(100 * R[p]["retrieved_from"][db] / n, 1) for db in DBS},
                       **{f: round(100 * fams[f] / n, 1) for f in FAM}}
rep = (TOOL / "output/impact_evolution/impact_evolution_report.md").read_text(encoding="utf-8")
m = re.search(r"\| Artificial Intelligence in Education \| (\w+) \| (\d{4}) \| (\d{4}) \| (\d+) \|", rep)
R["openalex_topic_AI_in_Education"] = {"phase": m.group(1), "first_year": int(m.group(2)), "last_year": int(m.group(3)), "documents": int(m.group(4))}
assert R["2014-2024"]["documents"] + R["2025-2026"]["documents"] == 347
assert sum(R[p]["retrieved_from"]["Scopus"] for p in ("2014-2024", "2025-2026")) == 251
(HERE / "calculos_salto_2025.json").write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(R, ensure_ascii=False, indent=1))
