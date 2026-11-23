"""Fase 1e: Enriquecimiento longitudinal via CSVs locales de SCImago (SJR, cuartiles, H-index, OA Diamond).

Carga automaticamente todos los archivos `scimago/scimagojr YYYY.csv` (ej. 2020–2025)
o `data/scimagojr.csv` como respaldo. Empareja cada articulo con el SJR y cuartil
del año exacto de su publicacion (con respaldo al año mas cercano disponible) y
construye la serie longitudinal `sjr_by_year` y `quartile_by_year`.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from config import SCIMAGO_DIR, SCIMAGO_CSV_PATH, ENRICHED_PATH
from src.utils import _journal_match_key, save_json

logger = logging.getLogger("bibliometrico.scimago")


class SCImagoEnricher:
    """Enriquecedor longitudinal de articulos via CSVs locales de SCImago."""

    def __init__(
        self,
        scimago_dir: Path = SCIMAGO_DIR,
        fallback_csv_path: Path = SCIMAGO_CSV_PATH,
    ):
        self.scimago_dir = scimago_dir
        self.csv_path = fallback_csv_path
        # Multi-year indices: year -> {issn_or_title_key: record}
        self._by_year_issn: dict[int, dict[str, dict]] = {}
        self._by_year_title: dict[int, dict[str, dict]] = {}
        self._years: list[int] = []
        self._loaded = False

    def _discover_csv_files(self) -> list[tuple[int, Path]]:
        """Descubre todos los CSVs de SCImago en scimago/ o el archivo unico de respaldo."""
        found: list[tuple[int, Path]] = []
        if self.scimago_dir and self.scimago_dir.exists():
            for p in sorted(self.scimago_dir.glob("*.csv")):
                m = re.search(r"(19|20)\d{2}", p.stem)
                yr = int(m.group(0)) if m else 2025
                found.append((yr, p))

        if not found and self.csv_path and self.csv_path.exists():
            m = re.search(r"(19|20)\d{2}", self.csv_path.stem)
            yr = int(m.group(0)) if m else 2025
            found.append((yr, self.csv_path))

        found.sort(key=lambda x: x[0])
        return found

    def _load_csv(self) -> bool:
        """Carga e indexa todos los CSVs anuales de SCImago."""
        if self._loaded:
            return True

        csv_files = self._discover_csv_files()
        if not csv_files:
            logger.warning(
                f"No se encontraron CSVs de SCImago en {self.scimago_dir} ni en {self.csv_path}."
            )
            return False

        for yr, path in csv_files:
            logger.info(f"Cargando SCImago {yr}: {path.name}")
            try:
                df = pd.read_csv(path, sep=";", encoding="utf-8", low_memory=False)
            except Exception:
                try:
                    df = pd.read_csv(path, sep=",", encoding="utf-8", low_memory=False)
                except Exception as e:
                    logger.error(f"No se pudo parsear {path}: {e}")
                    continue

            df.columns = [c.strip().lower() for c in df.columns]
            issn_map: dict[str, dict] = {}
            title_map: dict[str, dict] = {}

            for _, row in df.iterrows():
                record = self._extract_record(row, yr)
                if not record:
                    continue

                issn_raw = str(row.get("issn", ""))
                for issn in self._parse_issns(issn_raw):
                    issn_map[issn] = record

                title = record["title"]
                t_norm = title.lower()
                title_map[t_norm] = record
                t_key = _journal_match_key(title)
                if t_key:
                    title_map[t_key] = record

            self._by_year_issn[yr] = issn_map
            self._by_year_title[yr] = title_map

        self._years = sorted(self._by_year_issn.keys())
        self._loaded = bool(self._years)
        if self._loaded:
            latest = self._years[-1]
            logger.info(
                f"SCImago cargado para años {self._years} "
                f"(año {latest}: {len(self._by_year_issn[latest])} ISSNs, "
                f"{len(self._by_year_title[latest])} claves de titulo)"
            )
        return self._loaded

    @staticmethod
    def _parse_issns(issn_str: str) -> list[str]:
        """Extrae y normaliza todos los ISSNs (con y sin guion) de una cadena."""
        if not issn_str or issn_str == "nan":
            return []
        raw = re.findall(r"\d{4}-?\d{3}[\dXx]", str(issn_str))
        result = []
        for i in raw:
            clean = i.replace("-", "").upper()
            formatted = f"{clean[:4]}-{clean[4:]}"
            result.append(formatted)
            result.append(clean)
        return list(dict.fromkeys(result))

    @staticmethod
    def _parse_float(val) -> Optional[float]:
        """Parsea un valor numerico que puede usar coma como separador decimal."""
        if pd.isna(val):
            return None
        s = str(val).strip().replace(",", ".")
        if not s or s.lower() == "nan":
            return None
        try:
            return float(s)
        except ValueError:
            return None

    @staticmethod
    def _parse_int(val) -> Optional[int]:
        """Parsea un entero de forma segura."""
        if pd.isna(val):
            return None
        try:
            return int(float(str(val).strip().replace(",", ".")))
        except ValueError:
            return None

    def _extract_record(self, row: pd.Series, year: int) -> Optional[dict]:
        """Extrae un registro normalizado de una fila del CSV de SCImago."""
        title = str(row.get("title", "")).strip()
        if not title or title == "nan":
            return None

        sjr = self._parse_float(row.get("sjr", None))
        best_quartile = str(row.get("sjr best quartile", row.get("best quartile", ""))).strip()
        if best_quartile in ("nan", "-"):
            best_quartile = ""

        docs_year_col = f"total docs. ({year})"
        total_docs_year = self._parse_int(row.get(docs_year_col, None))
        total_cites_3y = self._parse_int(
            row.get("total citations (3years)", row.get("total cites (3years)", None))
        )
        cites_per_doc_2y = self._parse_float(
            row.get("citations / doc. (2years)", row.get("cites / doc. (2years)", None))
        )

        return {
            "scimago_year": year,
            "rank": self._parse_int(row.get("rank", None)),
            "sourceid": str(row.get("sourceid", "")).strip(),
            "title": title,
            "type": str(row.get("type", "")).strip(),
            "issn": str(row.get("issn", "")).strip(),
            "sjr": sjr,
            "sjr_best_quartile": best_quartile,
            "h_index": self._parse_int(row.get("h index", None)),
            "open_access": str(row.get("open access", "")).strip(),
            "open_access_diamond": str(row.get("open access diamond", "")).strip(),
            "total_docs_year": total_docs_year,
            "total_docs_3years": self._parse_int(row.get("total docs. (3years)", None)),
            "total_refs": self._parse_int(row.get("total refs.", None)),
            "total_cites_3years": total_cites_3y,
            "citable_docs_3years": self._parse_int(row.get("citable docs. (3years)", None)),
            "cites_per_doc_2years": cites_per_doc_2y,
            "ref_per_doc": self._parse_float(row.get("ref. / doc.", None)),
            "pct_female": self._parse_float(row.get("%female", None)),
            "overton": self._parse_int(row.get("overton", None)),
            "country": str(row.get("country", "")).strip(),
            "region": str(row.get("region", "")).strip(),
            "publisher": str(row.get("publisher", "")).strip(),
            "coverage": str(row.get("coverage", "")).strip(),
            "categories": str(row.get("categories", "")).strip(),
            "areas": str(row.get("areas", "")).strip(),
        }

    def _extract_article_issns(self, article: dict) -> list[str]:
        """Extrae todos los ISSNs candidatos desde todos los campos enriquecidos del articulo."""
        raw: list[str] = []
        for k in ("issn", "eissn"):
            if article.get(k):
                raw.append(str(article[k]))

        oa = article.get("openalex") or {}
        if isinstance(oa, dict):
            if oa.get("issn_l"):
                raw.append(str(oa["issn_l"]))
            for x in oa.get("issn") or []:
                raw.append(str(x))

        oam = (article.get("openalex_meta") or {}).get("source") or {}
        if isinstance(oam, dict):
            if oam.get("issn_l"):
                raw.append(str(oam["issn_l"]))
            for x in oam.get("issn") or []:
                raw.append(str(x))

        cr = article.get("crossref") or {}
        if isinstance(cr, dict):
            for x in cr.get("issn") or []:
                raw.append(str(x))

        result: list[str] = []
        for s in raw:
            result.extend(self._parse_issns(s))
        return list(dict.fromkeys(result))

    def _extract_article_titles(self, article: dict) -> list[str]:
        """Extrae variantes de nombre de revista del articulo."""
        titles: list[str] = []
        if article.get("publication_title"):
            titles.append(str(article["publication_title"]))
        oa = article.get("openalex") or {}
        if isinstance(oa, dict) and oa.get("source_display_name"):
            titles.append(str(oa["source_display_name"]))
        oam = (article.get("openalex_meta") or {}).get("source") or {}
        if isinstance(oam, dict) and oam.get("display_name"):
            titles.append(str(oam["display_name"]))
        cr = article.get("crossref") or {}
        if isinstance(cr, dict) and cr.get("container_title"):
            ct = cr["container_title"]
            if isinstance(ct, list) and ct:
                titles.append(str(ct[0]))
            elif isinstance(ct, str):
                titles.append(ct)
        return [t.strip() for t in titles if t and str(t).strip()]

    def _lookup_in_year(
        self, year: int, issns: list[str], titles: list[str]
    ) -> Optional[dict]:
        """Busca una revista en un año especifico por ISSN o por clave de titulo."""
        issn_map = self._by_year_issn.get(year, {})
        for issn in issns:
            if issn in issn_map:
                return issn_map[issn]

        title_map = self._by_year_title.get(year, {})
        for t in titles:
            t_low = t.lower()
            if t_low in title_map:
                return title_map[t_low]
            t_key = _journal_match_key(t)
            if t_key and t_key in title_map:
                return title_map[t_key]

        return None

    def lookup(
        self,
        issn: str,
        journal_title: str = "",
        pub_year: Optional[int] = None,
        extra_issns: Optional[list[str]] = None,
        extra_titles: Optional[list[str]] = None,
    ) -> Optional[dict]:
        """Busca una revista en SCImago priorizando el año de publicacion y construye su serie longitudinal."""
        if not self._load_csv():
            return None

        issns = self._parse_issns(issn)
        if extra_issns:
            for ei in extra_issns:
                for parsed in self._parse_issns(ei):
                    if parsed not in issns:
                        issns.append(parsed)

        titles = [journal_title.strip()] if journal_title else []
        if extra_titles:
            for et in extra_titles:
                if et and et.strip() not in titles:
                    titles.append(et.strip())

        min_yr, max_yr = self._years[0], self._years[-1]
        target_yr = max_yr
        if pub_year:
            try:
                target_yr = min(max(int(pub_year), min_yr), max_yr)
            except (ValueError, TypeError):
                target_yr = max_yr

        # Orden de busqueda: primero el año de publicacion (acotado a [min_yr, max_yr]),
        # luego por cercania temporal (prefiriendo años mas recientes en caso de empate)
        ordered_years = sorted(
            self._years,
            key=lambda y: (0 if y == target_yr else 1, abs(y - target_yr), -y),
        )

        matched_rec: Optional[dict] = None
        for yr in ordered_years:
            rec = self._lookup_in_year(yr, issns, titles)
            if rec:
                matched_rec = dict(rec)
                break

        if not matched_rec:
            return None

        # Si el registro coincidio por titulo, incorporar tambien sus ISSNs de SCImago
        # para reconstruir toda su serie longitudinal 2020-2025
        rec_issns = list(dict.fromkeys(issns + self._parse_issns(matched_rec.get("issn", ""))))
        rec_titles = list(dict.fromkeys(titles + [matched_rec.get("title", "")]))

        sjr_by_year: dict[str, float] = {}
        quartile_by_year: dict[str, str] = {}
        latest_rec: Optional[dict] = None

        for yr in self._years:
            yr_rec = self._lookup_in_year(yr, rec_issns, rec_titles)
            if yr_rec:
                latest_rec = yr_rec
                if yr_rec.get("sjr") is not None:
                    sjr_by_year[str(yr)] = yr_rec["sjr"]
                if yr_rec.get("sjr_best_quartile"):
                    quartile_by_year[str(yr)] = yr_rec["sjr_best_quartile"]

        # Si en el año exacto el SJR o el cuartil estaban vacios ('-'), usar el mas reciente disponible
        if matched_rec.get("sjr") is None and latest_rec and latest_rec.get("sjr") is not None:
            matched_rec["sjr"] = latest_rec["sjr"]
        if not matched_rec.get("sjr_best_quartile") and latest_rec and latest_rec.get("sjr_best_quartile"):
            matched_rec["sjr_best_quartile"] = latest_rec["sjr_best_quartile"]

        matched_rec["sjr_by_year"] = sjr_by_year
        matched_rec["quartile_by_year"] = quartile_by_year
        if latest_rec:
            matched_rec["sjr_latest"] = latest_rec.get("sjr")
            matched_rec["sjr_quartile_latest"] = latest_rec.get("sjr_best_quartile", "")
            matched_rec["latest_year"] = latest_rec.get("scimago_year")

        return matched_rec

    def enrich_article(self, article: dict) -> dict:
        """Enriquece un articulo con datos longitudinales de SCImago."""
        issns = self._extract_article_issns(article)
        titles = self._extract_article_titles(article)
        pub_year = article.get("publication_year")

        primary_issn = issns[0] if issns else article.get("issn", "")
        primary_title = titles[0] if titles else article.get("publication_title", "")

        record = self.lookup(
            issn=primary_issn,
            journal_title=primary_title,
            pub_year=pub_year,
            extra_issns=issns,
            extra_titles=titles,
        )

        if record:
            article["scimago"] = record
            article["sjr"] = record.get("sjr")
            article["sjr_quartile"] = record.get("sjr_best_quartile")
            # Completar ISSN del articulo si estaba vacio
            if not article.get("issn") and record.get("issn"):
                parsed = self._parse_issns(record["issn"])
                if parsed:
                    article["issn"] = parsed[0]

        return article


def run_phase1e(articles: list[dict]) -> list[dict]:
    """Ejecuta Fase 1e: Enriquecimiento SCImago longitudinal (CSVs locales)."""
    logger.info("=== FASE 1e: Enriquecimiento SCImago (Longitudinal 2020-2025) ===")
    if any("scopus_citations_by_year" not in a and "wos_citations_by_year" not in a for a in articles):
        try:
            from src.local_citation_enricher import run_local_citation_enrichment
            run_local_citation_enrichment(articles)
        except Exception:
            pass
    if any(not a.get("issn") or not a.get("publication_title") for a in articles):
        try:
            from src.metadata_completer import MetadataCompleter
            MetadataCompleter().stage_a(articles)
        except Exception:
            pass
    enricher = SCImagoEnricher()

    if not enricher._load_csv():
        logger.warning("SCImago enrichment skipped: CSVs not available")
        return articles

    enriched = 0
    for article in tqdm(articles, desc="SCImago enrichment"):
        enricher.enrich_article(article)
        if article.get("sjr") is not None or article.get("sjr_quartile"):
            enriched += 1

    logger.info(f"SCImago enriched: {enriched}/{len(articles)} ({enriched/max(len(articles),1)*100:.1f}%)")

    # Quartile distribution
    q_dist: dict[str, int] = {}
    for a in articles:
        q = a.get("sjr_quartile", "")
        if q:
            q_dist[q] = q_dist.get(q, 0) + 1
    if q_dist:
        logger.info(f"  Quartile distribution: {dict(sorted(q_dist.items()))}")

    save_json(articles, ENRICHED_PATH)
    logger.info(f"Checkpoint guardado: {ENRICHED_PATH}")

    return articles
