"""Read-only reproducible calculations for the EJEE English manuscript.
Reads G:/AUTOMATIZACIONES/biblio140302_11/data/ and output/ without modifying the tool.
Produces calculos_entrega_en.json documenting:
  1) Deduplicated reference counts and RPYS (1950–2026) per (document, work_id)
  2) Curated author-keyword counts from modelo_corpus.json
  3) Document-type split for 2014–2024 vs 2025–2026
  4) Self-efficacy query-sensitivity check (n = 95 vs n = 252)
  5) Top-5 bibliographic-coupling brokers by weighted betweenness
Run: python -X utf8 calculos_entrega_en.py"""
import collections
import csv
import json
import pathlib
import numpy as np
from scipy.stats import spearmanr

HERE = pathlib.Path(__file__).resolve().parent
TOOL = pathlib.Path(r"G:\AUTOMATIZACIONES\biblio140302_11")

arts = json.load(open(TOOL / "data" / "enriched_articles.json", encoding="utf-8"))
arts = arts if isinstance(arts, list) else arts["articles"]
modelo = json.load(open(TOOL / "data" / "modelo_corpus.json", encoding="utf-8"))
bridges = json.load(open(TOOL / "output" / "networks" / "bridge_articles_report.json", encoding="utf-8"))

# 1. Deduplicated references & RPYS (Marx et al., 2014) per (document, work_id)
raw_total = 0
dedup_total_all = 0
unique_wid_resolved = set()
unique_wid_rw = set()
by_year = collections.Counter()
works_by_year = collections.defaultdict(collections.Counter)
labels = {}
hu_docs = hu_dup_docs = hu_occ = 0

for a in arts:
    for r in a.get("referenced_works") or []:
        if isinstance(r, str) and r:
            unique_wid_rw.add(r.split("/")[-1] if "/" in r else r)
    seen_doc = set()
    hu_i = 0
    for r in a.get("references_resolved") or []:
        raw_total += 1
        oa_id = str(r.get("openalex_id") or "")
        wid = oa_id.split("/")[-1] if "/" in oa_id else oa_id
        fa = str(r.get("first_author") or "Unknown")
        yr = r.get("year")
        title = str(r.get("title") or "")
        dk = wid or f"{fa}_{yr}_{title[:60]}"
        if "Cutoff criteria for fit" in title:
            hu_i += 1
        if dk in seen_doc:
            continue
        seen_doc.add(dk)
        unique_wid_resolved.add(dk)
        dedup_total_all += 1
        if isinstance(yr, int) and 1950 <= yr <= 2026:
            by_year[yr] += 1
            works_by_year[yr][dk] += 1
            labels.setdefault(dk, {"first_author": fa, "year": yr, "title": title[:90]})
    if hu_i:
        hu_docs += 1
        hu_occ += hu_i
        if hu_i > 1:
            hu_dup_docs += 1

min_y, max_y = min(by_year), max(by_year)
series = []
for y in range(min_y, max_y + 1):
    c = by_year[y]
    win = [by_year[t] for t in range(max(min_y, y - 2), min(max_y, y + 2) + 1)]
    med = float(np.median(win))
    top_k, top_n = works_by_year[y].most_common(1)[0] if works_by_year[y] else ("", 0)
    conc = round(100.0 * top_n / c, 1) if c else 0.0
    series.append({
        "year": y,
        "citations": c,
        "median_5": med,
        "deviation": round(c - med, 1),
        "top_count": top_n,
        "concentration_pct": conc,
        "top_work": labels.get(top_k, {}),
        "incomplete_window": y >= 2024,
    })

# 1b. Concordance of deduplicated RPYS against bibliometrix (11_rpys_por_ano.csv)
bib_rpys_path = TOOL / "validacion_externa" / "bibliometrix" / "11_rpys_por_ano.csv"
bib_rows = list(csv.DictReader(open(bib_rpys_path, encoding="utf-8-sig")))
bib_map = {int(float(r["Year"])): float(r["Citations"]) for r in bib_rows}
ded_map = {r["year"]: r["citations"] for r in series}
common_years = sorted(set(bib_map) & set(ded_map))
rho_rpys, p_rpys = spearmanr([ded_map[y] for y in common_years], [bib_map[y] for y in common_years])
rpys_concordance = {
    "source_bibliometrix": str(bib_rpys_path),
    "common_years": len(common_years),
    "sum_dedup_common_years": int(sum(ded_map[y] for y in common_years)),
    "sum_bibliometrix_common_years": int(sum(bib_map[y] for y in common_years)),
    "spearman_rho": round(float(rho_rpys), 4),
    "p_value": float(p_rpys),
}

# 2. Curated canonical author keywords from modelo_corpus.json
kw_cnt = collections.Counter(t for a in modelo["articulos"].values() for t in a.get("palabras_clave", []))
docs_with_kw = sum(1 for a in modelo["articulos"].values() if a.get("palabras_clave"))

# 3. Period x document type split
split_25_26 = dict(collections.Counter(a["item_type"] for a in arts if a["publication_year"] >= 2025))
split_14_24 = dict(collections.Counter(a["item_type"] for a in arts if a["publication_year"] < 2025))

# 4. Self-efficacy query-sensitivity check
se_docs = [
    a for a in arts
    if "self-efficacy" in (a.get("title", "") + " " + a.get("abstract", "") + " " + " ".join(a.get("author_keywords") or [])).lower()
    or "self efficacy" in (a.get("title", "") + " " + a.get("abstract", "") + " " + " ".join(a.get("author_keywords") or [])).lower()
]
non_se_docs = [a for a in arts if a not in se_docs]
hu_se = sum(1 for a in se_docs if any("Cutoff criteria for fit" in str(r.get("title", "")) for r in (a.get("references_resolved") or [])))
hu_non_se = sum(1 for a in non_se_docs if any("Cutoff criteria for fit" in str(r.get("title", "")) for r in (a.get("references_resolved") or [])))

# 5. Top-5 brokers by weighted betweenness
top5_brokers = bridges["bridge_articles"][:5]
top5_meta = {a["key"]: {"authors": a.get("authors"), "year": a.get("publication_year"), "title": a.get("title"),
                        "journal": a.get("publication_title"), "volume": a.get("volume"), "issue": a.get("issue"),
                        "pages": a.get("pages"), "doi": a.get("doi")}
             for a in arts if a["key"] in {b["key"] for b in top5_brokers}}

out = {
    "references": {
        "raw_resolved_occurrences": raw_total,
        "dedup_citations_all_years": dedup_total_all,
        "dedup_citations_1950_2026": sum(by_year.values()),
        "unique_works_resolved": len(unique_wid_resolved),
        "unique_works_referenced_works": len(unique_wid_rw),
        "hu_bentler_docs": hu_docs,
        "hu_bentler_dup_docs": hu_dup_docs,
        "hu_bentler_raw_occ": hu_occ,
    },
    "rpys_dedup": {
        "year_range": [min_y, max_y],
        "series": series,
        "roots_ge_50pct": [r for r in series if r["deviation"] > 0 and r["concentration_pct"] >= 50.0 and not r["incomplete_window"] and r["citations"] >= 5],
        "concordance_bibliometrix": rpys_concordance,
    },
    "curated_keywords": {
        "docs_with_author_keywords": docs_with_kw,
        "unique_canonical_terms": len(kw_cnt),
        "top_20": kw_cnt.most_common(20),
    },
    "period_split": {"2014_2024": split_14_24, "2025_2026": split_25_26},
    "self_efficacy_sensitivity": {
        "n_se": len(se_docs),
        "pct_se": round(100.0 * len(se_docs) / len(arts), 1),
        "n_non_se": len(non_se_docs),
        "pct_non_se": round(100.0 * len(non_se_docs) / len(arts), 1),
        "non_se_conference_pct": round(100.0 * sum(1 for a in non_se_docs if a["item_type"] == "Conference Paper") / len(non_se_docs), 1),
        "non_se_2025_2026_pct": round(100.0 * sum(1 for a in non_se_docs if a["publication_year"] >= 2025) / len(non_se_docs), 1),
        "hu_bentler_in_se": hu_se,
        "hu_bentler_in_non_se": hu_non_se,
    },
    "top5_brokers": top5_brokers,
    "top5_brokers_meta": top5_meta,
}

out_path = HERE / "calculos_entrega_en.json"
out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("Saved:", out_path)
print("Refs:", out["references"])
print("Roots >=50%:", [(r["year"], r["citations"], r["deviation"], r["concentration_pct"], r["top_work"].get("first_author")) for r in out["rpys_dedup"]["roots_ge_50pct"]])
print("LeChasseur2025 meta:", top5_meta.get("LeChasseur2025"))
