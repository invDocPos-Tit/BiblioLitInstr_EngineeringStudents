"""Enriquecedor local desde exportaciones de Scopus y Web of Science en data/:
- data/CitationScopus.csv  (Scopus Citation Overview: ISSN, Journal, Volume, Issue, citas anuales 2001-2026)
- data/CitationWos.xlsx / CitationWos.csv (WoS Citation Report: DOI, Source Title, Volume, Issue, Pages, citas anuales 1900-2026)
- data/scopusReferences.ris (Referencias citadas exportadas de Scopus en formato RIS)
- data/scopusCitadopor.ris  (Articulos citantes 'Cited by' exportados de Scopus en formato RIS)
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from config import DATA_DIR
from src.utils import normalize_doi, normalize_journal_name

logger = logging.getLogger("bibliometrico.local_citations")


def _norm_title(s: str) -> str:
    if not s or pd.isna(s):
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def _safe_int(val) -> int:
    if val is None or pd.isna(val):
        return 0
    try:
        return int(float(str(val).strip().replace(",", "")))
    except (ValueError, TypeError):
        return 0


def _merge_db_tag(article: dict, db_name: str) -> None:
    """Agrega db_name (ej. 'Scopus' o 'WoS') a merged_from y source_db si estaban vacios."""
    if not article.get("source_db"):
        article["source_db"] = db_name
    mf = (article.get("merged_from") or "").strip()
    existing = {d.strip() for d in re.split(r"[+;,\s]+", mf) if d.strip()} if mf else set()
    if db_name not in existing:
        existing.add(db_name)
        article["merged_from"] = "+".join(sorted(existing))


def _merge_counts_by_year(article: dict, year_dict: dict[int, int]) -> None:
    """Fusiona conteos anuales de citas en article['openalex']['counts_by_year'] tomando el maximo por año."""
    if not year_dict:
        return
    oa = article.get("openalex")
    if not isinstance(oa, dict):
        oa = {}
        article["openalex"] = oa
    cby_list = oa.get("counts_by_year")
    if not isinstance(cby_list, list):
        cby_list = []

    by_yr: dict[int, int] = {}
    for item in cby_list:
        if isinstance(item, dict) and item.get("year"):
            by_yr[int(item["year"])] = max(int(item.get("cited_by_count", 0) or 0), 0)

    for yr, cnt in year_dict.items():
        if cnt > 0:
            by_yr[yr] = max(by_yr.get(yr, 0), cnt)

    oa["counts_by_year"] = [
        {"year": yr, "cited_by_count": cnt}
        for yr, cnt in sorted(by_yr.items(), reverse=True)
    ]


def enrich_from_scopus_citation_csv(articles: list[dict], csv_path: Path) -> int:
    """Enriquece articulos con data/CitationScopus.csv (Scopus Citation Overview)."""
    if not csv_path.exists():
        return 0

    try:
        # Detectar fila de encabezado ('Publication Year,Document Title,...')
        raw_lines = csv_path.read_text(encoding="utf-8", errors="replace").splitlines()
        header_idx = 6
        year_header_row: list[str] = []
        for idx, line in enumerate(raw_lines[:15]):
            if "Document Title" in line and "Publication Year" in line:
                header_idx = idx
                if idx > 0:
                    year_header_row = [c.strip() for c in raw_lines[idx - 1].split(",")]
                break

        df = pd.read_csv(csv_path, skiprows=header_idx, encoding="utf-8")
    except Exception as e:
        logger.warning(f"No se pudo leer {csv_path.name}: {e}")
        return 0

    # Mapear indices de columnas anuales usando la fila superior (ej. 2001..2026)
    col_to_year: dict[str, int] = {}
    cols = list(df.columns)
    for i, col_name in enumerate(cols):
        if i < len(year_header_row):
            yh = year_header_row[i]
            if re.match(r"^(19|20)\d{2}$", yh):
                col_to_year[col_name] = int(yh)
        elif re.match(r"^(19|20)\d{2}$", str(col_name).strip()):
            col_to_year[col_name] = int(str(col_name).strip())

    total_col = cols[-1] if cols else None
    by_title: dict[str, dict] = {}
    for _, s_row in df.iterrows():
        row = s_row.to_dict()
        t = _norm_title(row.get("Document Title", ""))
        if t and len(t) >= 15:
            by_title[t] = row

    matched = 0
    for a in articles:
        t = _norm_title(a.get("title", ""))
        row = by_title.get(t)
        if row is None:
            continue

        matched += 1
        _merge_db_tag(a, "Scopus")

        # Revista, ISSN, Volumen, Issue
        j_title = str(row.get("Journal Title", "") or "").strip()
        if j_title and j_title != "nan" and not a.get("publication_title"):
            a["publication_title"] = normalize_journal_name(j_title)

        issn_raw = str(row.get("ISSN", "") or "").strip()
        if issn_raw and issn_raw != "nan" and not a.get("issn"):
            clean_i = re.sub(r"[^0-9Xx]", "", issn_raw).upper()
            if len(clean_i) == 8:
                a["issn"] = f"{clean_i[:4]}-{clean_i[4:]}"
            else:
                a["issn"] = issn_raw

        vol = str(row.get("Volume", "") or "").strip()
        if vol and vol not in ("nan", "None") and not a.get("volume"):
            a["volume"] = vol.removesuffix(".0")

        iss = str(row.get("Issue", "") or "").strip()
        if iss and iss not in ("nan", "None") and not a.get("issue"):
            a["issue"] = iss.removesuffix(".0")

        # Citas anuales y totales
        yr_cites: dict[int, int] = {}
        for col_name, yr in col_to_year.items():
            c = _safe_int(row.get(col_name, 0))
            if c > 0:
                yr_cites[yr] = c

        tot_cites = _safe_int(row.get(total_col, 0)) if total_col else sum(yr_cites.values())
        a["cited_by_count"] = max(int(a.get("cited_by_count", 0) or 0), tot_cites)
        a["scopus_citations_by_year"] = yr_cites
        _merge_counts_by_year(a, yr_cites)

    logger.info(f"CitationScopus.csv integrado: {matched}/{len(articles)} articulos emparejados")
    return matched


def enrich_from_wos_citation_file(articles: list[dict], wos_path: Path) -> int:
    """Enriquece articulos con data/CitationWos.xlsx o data/CitationWos.csv."""
    if not wos_path.exists():
        return 0

    try:
        if wos_path.suffix.lower() in (".xlsx", ".xls"):
            df_probe = pd.read_excel(wos_path, nrows=15, header=None)
            header_idx = 10
            for idx, r in df_probe.iterrows():
                vals = [str(v).strip() for v in r.values if pd.notna(v)]
                if "Title" in vals and ("Source Title" in vals or "DOI" in vals):
                    header_idx = int(idx)
                    break
            df = pd.read_excel(wos_path, skiprows=header_idx)
        else:
            df = pd.read_csv(wos_path, skiprows=10, encoding="utf-8", errors="replace")
    except Exception as e:
        logger.warning(f"No se pudo leer {wos_path.name}: {e}")
        return 0

    year_cols = [c for c in df.columns if re.match(r"^(19|20)\d{2}$", str(c).strip())]
    by_doi: dict[str, dict] = {}
    by_title: dict[str, dict] = {}

    for _, s_row in df.iterrows():
        row = s_row.to_dict()
        d = normalize_doi(str(row.get("DOI", "") or "")) if pd.notna(row.get("DOI")) else ""
        if d:
            by_doi[d.lower()] = row
        t = _norm_title(row.get("Title", ""))
        if t and len(t) >= 15:
            by_title[t] = row

    matched = 0
    for a in articles:
        d = (a.get("doi") or "").strip().lower()
        t = _norm_title(a.get("title", ""))
        row = (by_doi.get(d) if d else None) or by_title.get(t)
        if row is None:
            continue

        matched += 1
        _merge_db_tag(a, "WoS")

        if not a.get("doi") and pd.notna(row.get("DOI")):
            w_doi = normalize_doi(str(row["DOI"]))
            if w_doi:
                a["doi"] = w_doi
                if not a.get("url"):
                    a["url"] = f"https://doi.org/{w_doi}"

        src_title = str(row.get("Source Title", "") or "").strip()
        if src_title and src_title != "nan" and not a.get("publication_title"):
            a["publication_title"] = normalize_journal_name(src_title)

        vol = str(row.get("Volume", "") or "").strip()
        if vol and vol not in ("nan", "None") and not a.get("volume"):
            a["volume"] = vol.removesuffix(".0")

        iss = str(row.get("Issue", "") or "").strip()
        if iss and iss not in ("nan", "None") and not a.get("issue"):
            a["issue"] = iss.removesuffix(".0")

        if not a.get("pages"):
            bp = str(row.get("Beginning Page", "") or "").strip().removesuffix(".0")
            ep = str(row.get("Ending Page", "") or "").strip().removesuffix(".0")
            if bp and bp != "nan":
                a["pages"] = f"{bp}-{ep}" if ep and ep != "nan" and ep != bp else bp

        tot_cites = _safe_int(row.get("Total Citations", 0))
        a["cited_by_count"] = max(int(a.get("cited_by_count", 0) or 0), tot_cites)

        yr_cites: dict[int, int] = {}
        for yc in year_cols:
            c = _safe_int(row.get(yc, 0))
            if c > 0:
                yr_cites[int(str(yc).strip())] = c
        a["wos_citations_by_year"] = yr_cites
        _merge_counts_by_year(a, yr_cites)

    logger.info(f"{wos_path.name} integrado: {matched}/{len(articles)} articulos emparejados")
    return matched


def parse_ris_file(ris_path: Path) -> list[dict]:
    """Parsea un archivo RIS de Scopus en una lista de diccionarios normalizados."""
    if not ris_path.exists():
        return []

    records: list[dict] = []
    cur: dict = {"authors": [], "keywords": [], "affiliations": []}

    with open(ris_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if line.startswith("ER  -"):
                if cur.get("title") or cur.get("doi"):
                    records.append(cur)
                cur = {"authors": [], "keywords": [], "affiliations": []}
                continue
            if len(line) < 6 or line[2:6] != "  - ":
                continue
            tag = line[:2]
            val = line[6:].strip()
            if not val:
                continue
            if tag == "TI":
                cur["title"] = val
            elif tag == "T2":
                cur["journal"] = val
            elif tag == "J2":
                cur["journal_abbrev"] = val
            elif tag == "PY":
                m = re.search(r"(18|19|20)\d{2}", val)
                if m:
                    cur["year"] = int(m.group(0))
            elif tag == "DO":
                cur["doi"] = normalize_doi(val)
            elif tag == "SN":
                m = re.search(r"(\d{4})-?(\d{3}[\dXx])", val)
                if m:
                    cur["issn"] = f"{m.group(1)}-{m.group(2).upper()}"
            elif tag == "VL":
                cur["volume"] = val
            elif tag == "IS":
                cur["issue"] = val
            elif tag == "SP":
                cur["sp"] = val
            elif tag == "EP":
                cur["ep"] = val
            elif tag == "PB":
                cur["publisher"] = val
            elif tag == "AB":
                cur["abstract"] = val
            elif tag == "AU":
                cur["authors"].append(val)
            elif tag == "KW":
                cur["keywords"].append(val)
            elif tag == "AD":
                cur["affiliations"].append(val)
            elif tag == "N1" and val.startswith("Cited By:"):
                cur["cited_by"] = _safe_int(val.split(":", 1)[1])

    return records


def enrich_from_scopus_ris_files(
    articles: list[dict],
    refs_ris_path: Path,
    citedby_ris_path: Path,
) -> dict[str, int]:
    """Integra scopusReferences.ris y scopusCitadopor.ris para completar metadatos de articulos y referencias."""
    stats = {"ref_records": 0, "citedby_records": 0, "refs_enriched": 0, "articles_enriched": 0}

    ref_records = parse_ris_file(refs_ris_path) if refs_ris_path.exists() else []
    citedby_records = parse_ris_file(citedby_ris_path) if citedby_ris_path.exists() else []
    stats["ref_records"] = len(ref_records)
    stats["citedby_records"] = len(citedby_records)

    if not ref_records and not citedby_records:
        return stats

    by_doi: dict[str, dict] = {}
    by_title: dict[str, dict] = {}
    for rec in ref_records + citedby_records:
        d = (rec.get("doi") or "").lower()
        if d:
            by_doi[d] = rec
        t = _norm_title(rec.get("title", ""))
        if t and len(t) >= 15:
            by_title[t] = rec

    for a in articles:
        # 1. Enriquecer el propio articulo si aparece en alguno de los RIS
        d = (a.get("doi") or "").lower()
        t = _norm_title(a.get("title", ""))
        hit = (by_doi.get(d) if d else None) or by_title.get(t)
        if not hit and not d and len(t) >= 20:
            from rapidfuzz import fuzz
            for rt, cand_rec in by_title.items():
                if abs(len(rt) - len(t)) <= 15 and fuzz.ratio(t, rt) >= 92:
                    hit = cand_rec
                    break
        if hit:
            stats["articles_enriched"] += 1
            if not a.get("doi") and hit.get("doi"):
                a["doi"] = hit["doi"]
            if not a.get("publication_title") and hit.get("journal"):
                a["publication_title"] = normalize_journal_name(hit["journal"])
            if not a.get("issn") and hit.get("issn"):
                a["issn"] = hit["issn"]
            if not a.get("publisher") and hit.get("publisher"):
                a["publisher"] = hit["publisher"]
            if not a.get("journal_abbrev") and hit.get("journal_abbrev"):
                a["journal_abbrev"] = hit["journal_abbrev"]
            if not a.get("manual_tags") and hit.get("keywords"):
                a["manual_tags"] = hit["keywords"][:25]
                a["all_tags"] = list(dict.fromkeys((a.get("all_tags") or []) + a["manual_tags"]))

        # 2. Enriquecer las referencias citadas (referenced_works_detailed) con scopusReferences.ris
        for ref in a.get("referenced_works_detailed") or []:
            if not isinstance(ref, dict):
                continue
            rd = normalize_doi(str(ref.get("doi", "") or "")).lower()
            rt = _norm_title(ref.get("title", "") or ref.get("article_title", ""))
            rhit = (by_doi.get(rd) if rd else None) or (by_title.get(rt) if rt else None)
            if rhit:
                stats["refs_enriched"] += 1
                if not ref.get("title") and rhit.get("title"):
                    ref["title"] = rhit["title"]
                if not ref.get("publication_year") and rhit.get("year"):
                    ref["publication_year"] = rhit["year"]
                    ref["year"] = rhit["year"]
                if not ref.get("first_author") and rhit.get("authors"):
                    ref["first_author"] = rhit["authors"][0]
                if not ref.get("journal") and rhit.get("journal"):
                    ref["journal"] = rhit["journal"]
                if rhit.get("cited_by") is not None:
                    ref["cited_by_count"] = max(int(ref.get("cited_by_count", 0) or 0), rhit["cited_by"])

    logger.info(
        f"RIS Scopus integrados: {stats['ref_records']} refs RIS, "
        f"{stats['citedby_records']} cited-by RIS -> "
        f"{stats['articles_enriched']} articulos y {stats['refs_enriched']} referencias enriquecidas"
    )
    return stats


def run_local_citation_enrichment(articles: list[dict], data_dir: Path = DATA_DIR) -> dict:
    """Detecta e integra automaticamente CitationScopus.csv, CitationWos.xlsx/csv y *.ris en data/."""
    scopus_csv = data_dir / "CitationScopus.csv"
    wos_xlsx = data_dir / "CitationWos.xlsx"
    wos_csv = data_dir / "CitationWos.csv"
    refs_ris = data_dir / "scopusReferences.ris"
    citedby_ris = data_dir / "scopusCitadopor.ris"

    m_scop = enrich_from_scopus_citation_csv(articles, scopus_csv) if scopus_csv.exists() else 0
    wos_file = wos_xlsx if wos_xlsx.exists() else wos_csv
    m_wos = enrich_from_wos_citation_file(articles, wos_file) if wos_file.exists() else 0
    ris_stats = enrich_from_scopus_ris_files(articles, refs_ris, citedby_ris)

    return {
        "scopus_citation_matched": m_scop,
        "wos_citation_matched": m_wos,
        **ris_stats,
    }
