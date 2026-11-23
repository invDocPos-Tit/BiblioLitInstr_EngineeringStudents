"""Fase 1d: Enriquecimiento de metricas de revistas y autores via OpenAlex.

Endpoints:
  - /sources/{id}  -> h-index, i10, 2yr_mean_citedness, APC, is_oa, DOAJ
  - /authors/{id}  -> h-index, i10, career_age, total_works
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from tqdm import tqdm

from rapidfuzz import fuzz

from config import (
    OPENALEX_BASE_URL, OPENALEX_EMAIL, OPENALEX_API_KEY,
    OPENALEX_RATE_LIMIT, OPENALEX_META_CACHE_DIR, ENRICHED_PATH,
    OPENALEX_FUZZY_THRESHOLD,
)
from src.utils import (
    RateLimiter, DiskCache, create_session, safe_request, save_json,
    normalize_doi,
)

logger = logging.getLogger("bibliometrico.openalex_meta")

# Formato ISSN canonico: NNNN-NNNC (C = digito o X). Sirve para descartar
# ISBN-13 de actas de congreso (que vienen en el campo ISSN) antes de
# consultar el endpoint /sources/issn: de OpenAlex.
_ISSN_RE = re.compile(r"^\d{4}-\d{3}[\dXx]$")


class OpenAlexMetaEnricher:
    """Enriquece articulos con metricas de revistas (Sources) y autores (Authors)."""

    def __init__(self, rate_override: float | None = None):
        self.session = create_session()
        rate = OPENALEX_RATE_LIMIT * 5 if OPENALEX_API_KEY else OPENALEX_RATE_LIMIT
        if rate_override and rate_override > 0:
            rate = rate_override
        self.rate_limiter = RateLimiter(min(rate, 50))
        self.cache = DiskCache(OPENALEX_META_CACHE_DIR)
        self.base_url = OPENALEX_BASE_URL
        self.email = OPENALEX_EMAIL
        self.api_key = OPENALEX_API_KEY

        # In-memory dedup: ISSN -> metrics, author_id -> metrics
        self._source_cache: dict[str, dict] = {}
        self._author_cache: dict[str, dict] = {}

    def _build_params(self, extra: dict | None = None) -> dict:
        params = {}
        if self.api_key:
            params["api_key"] = self.api_key
        elif self.email:
            params["mailto"] = self.email
        if extra:
            params.update(extra)
        return params

    # ------------------------------------------------------------------
    # Source (Journal) metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_issn(raw: str) -> str:
        """Normalize an ISSN value: strip suffixes, keep only digits/X, add dash."""
        import re
        # Take first ISSN if multiple separated by ; or ,
        s = raw.strip().split(";")[0].strip().split(",")[0].strip()
        # Remove parenthesised annotations like (ISSN), (Print), (Online)
        s = re.sub(r"\s*\(.*?\)", "", s).strip()
        # Remove dashes for digit extraction
        digits = re.sub(r"[^0-9Xx]", "", s)
        if len(digits) == 8:
            return f"{digits[:4]}-{digits[4:]}"
        return s  # fallback: return cleaned string as-is

    def _fetch_source_by_issn(self, issn: str) -> Optional[dict]:
        """Fetch source record by ISSN."""
        if not issn or str(issn).strip() in ("", "nan"):
            return None

        # Clean ISSN
        issn_clean = self._clean_issn(str(issn))
        if not issn_clean:
            return None

        # Solo consultar si es un ISSN real (NNNN-NNNC). Valores como los
        # ISBN-13 de actas de congreso (prefijo 978/979, 13 digitos) NUNCA
        # resuelven en /sources/issn: y solo generan 404 ruidosos.
        if not _ISSN_RE.match(issn_clean):
            logger.debug(
                "Omitido (no es ISSN, p.ej. ISBN de actas): %r", issn_clean
            )
            self._source_cache[issn_clean] = {}
            return {}

        # In-memory
        if issn_clean in self._source_cache:
            return self._source_cache[issn_clean]

        # Disk cache
        cache_key = f"source:issn:{issn_clean}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            self._source_cache[issn_clean] = cached
            return cached

        url = f"{self.base_url}/sources/issn:{issn_clean}"
        result = safe_request(
            url, session=self.session,
            rate_limiter=self.rate_limiter,
            params=self._build_params(),
            # 404 = revista no indexada en OpenAlex: esperado y benigno.
            quiet_statuses=(404,),
        )

        if result and "id" in result:
            metrics = self._extract_source_metrics(result)
            self.cache.set(cache_key, metrics)
            self._source_cache[issn_clean] = metrics
            return metrics

        # Store empty to avoid re-fetching
        empty = {}
        self.cache.set(cache_key, empty)
        self._source_cache[issn_clean] = empty
        return empty

    def _fetch_source_by_name(self, name: str) -> Optional[dict]:
        """Fallback: busca la fuente por nombre (titulo de revista/actas).

        Para articulos sin ISSN valido (p.ej. actas con ISBN) o cuya revista
        no resuelve por ISSN. Hace ``/sources?search=`` y exige un match
        difuso >= OPENALEX_FUZZY_THRESHOLD contra display_name /
        abbreviated_title / alternate_titles para no traer una revista
        equivocada. Asi se recolecta lo que se pueda con el titulo.
        """
        if not name or str(name).strip().lower() in ("", "nan"):
            return None
        q = " ".join(str(name).strip().split())
        if len(q) < 4:
            return None

        mem_key = f"name:{q.lower()}"
        if mem_key in self._source_cache:
            return self._source_cache[mem_key]

        cache_key = f"source:name:{q.lower()}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            self._source_cache[mem_key] = cached
            return cached

        url = f"{self.base_url}/sources"
        result = safe_request(
            url, session=self.session,
            rate_limiter=self.rate_limiter,
            params=self._build_params({"search": q, "per_page": 10}),
        )

        best, best_score = None, 0
        for src in (result or {}).get("results", []):
            cands = [src.get("display_name", ""),
                     src.get("abbreviated_title", "")]
            cands += src.get("alternate_titles", []) or []
            for cand in cands:
                if not cand:
                    continue
                score = fuzz.token_set_ratio(q.lower(), str(cand).lower())
                if score > best_score:
                    best_score, best = score, src

        if best and best_score >= OPENALEX_FUZZY_THRESHOLD:
            metrics = self._extract_source_metrics(best)
            metrics["matched_by"] = "name"
            metrics["match_score"] = best_score
            self.cache.set(cache_key, metrics)
            self._source_cache[mem_key] = metrics
            logger.info(
                "Fuente por nombre (score=%d): %s -> %s",
                best_score, q[:50], best.get("display_name", "")[:50],
            )
            return metrics

        if best:
            logger.debug(
                "Sin match de fuente confiable (best=%d) para %r",
                best_score, q[:60],
            )
        empty = {}
        self.cache.set(cache_key, empty)
        self._source_cache[mem_key] = empty
        return empty

    def _fetch_full_source(self, source_id: str) -> Optional[dict]:
        """Baja el record completo de una source por su id OpenAlex."""
        if not source_id:
            return None
        sid = str(source_id).rstrip("/").split("/")[-1]
        if not sid:
            return None
        mem_key = f"sid:{sid}"
        if mem_key in self._source_cache:
            return self._source_cache[mem_key]
        cache_key = f"source:id:{sid}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            self._source_cache[mem_key] = cached
            return cached

        url = f"{self.base_url}/sources/{sid}"
        result = safe_request(
            url, session=self.session,
            rate_limiter=self.rate_limiter,
            params=self._build_params(),
            quiet_statuses=(404,),
        )
        if result and "id" in result:
            metrics = self._extract_source_metrics(result)
            self.cache.set(cache_key, metrics)
            self._source_cache[mem_key] = metrics
            return metrics
        empty = {}
        self.cache.set(cache_key, empty)
        self._source_cache[mem_key] = empty
        return empty

    def _fetch_source_by_doi(self, doi: str) -> Optional[dict]:
        """Fallback mas fiable: del DOI del articulo a su source OpenAlex.

        ``/works/doi:`` -> ``primary_location.source``. Es la forma mas
        confiable de casar la revista (mapeo exacto, sin heuristica de
        nombre). Si la source trae ``issn_l`` reusa la ruta por ISSN (cache
        compartida); si no, baja el record completo por el id de source.
        """
        if not doi:
            return None
        doi_n = normalize_doi(str(doi))
        if not doi_n:
            return None

        mem_key = f"doi:{doi_n}"
        if mem_key in self._source_cache:
            return self._source_cache[mem_key]
        cache_key = f"source:doi:{doi_n}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            self._source_cache[mem_key] = cached
            return cached

        url = f"{self.base_url}/works/https://doi.org/{doi_n}"
        work = safe_request(
            url, session=self.session,
            rate_limiter=self.rate_limiter,
            params=self._build_params({
                "select": "id,primary_location,locations",
            }),
            quiet_statuses=(404,),
        )

        source = None
        if isinstance(work, dict):
            source = (work.get("primary_location") or {}).get("source")
            if not source:
                for loc in (work.get("locations") or []):
                    if isinstance(loc, dict) and loc.get("source"):
                        source = loc["source"]
                        break

        metrics: dict = {}
        if isinstance(source, dict):
            issn_l = source.get("issn_l") or ""
            sid = source.get("id") or ""
            m = None
            if issn_l:
                m = self._fetch_source_by_issn(issn_l)
            if not m and sid:
                m = self._fetch_full_source(sid)
            if m:
                metrics = dict(m)
                metrics["matched_by"] = "doi"

        self.cache.set(cache_key, metrics)
        self._source_cache[mem_key] = metrics
        if metrics:
            logger.info(
                "Fuente por DOI: %s -> %s",
                doi_n[:40], metrics.get("display_name", "")[:50],
            )
        return metrics

    def _extract_source_metrics(self, source: dict) -> dict:
        """Extract journal-level metrics from OpenAlex source object."""
        summary = source.get("summary_stats", {})
        raw_apc = source.get("apc_usd")
        if isinstance(raw_apc, dict):
            apc_price = raw_apc.get("price", None)
            apc_currency = raw_apc.get("currency", None)
        elif isinstance(raw_apc, (int, float)):
            apc_price = raw_apc
            apc_currency = "USD"
        else:
            apc_price = None
            apc_currency = None
        return {
            "openalex_source_id": source.get("id", ""),
            "display_name": source.get("display_name", ""),
            "type": source.get("type", ""),
            "is_oa": source.get("is_oa", False),
            "is_in_doaj": source.get("is_in_doaj", False),
            "h_index": summary.get("h_index", None),
            "i10_index": summary.get("i10_index", None),
            "2yr_mean_citedness": summary.get("2yr_mean_citedness", None),
            "works_count": source.get("works_count", 0),
            "cited_by_count": source.get("cited_by_count", 0),
            "apc_usd": apc_price,
            "apc_currency": apc_currency,
            "country_code": source.get("country_code", ""),
            "publisher": source.get("host_organization_name", ""),
            "subjects": [
                t.get("display_name", "")
                for t in source.get("topics", [])[:5]
            ],
        }

    # ------------------------------------------------------------------
    # Author metrics
    # ------------------------------------------------------------------

    def _fetch_author(self, author_id: str) -> Optional[dict]:
        """Fetch author record by OpenAlex author ID."""
        if not author_id:
            return None

        # In-memory
        if author_id in self._author_cache:
            return self._author_cache[author_id]

        # Disk cache
        cache_key = f"author:{author_id}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            self._author_cache[author_id] = cached
            return cached

        url = f"{self.base_url}/authors/{author_id.split('/')[-1]}"
        result = safe_request(
            url, session=self.session,
            rate_limiter=self.rate_limiter,
            params=self._build_params(),
        )

        if result and "id" in result:
            metrics = self._extract_author_metrics(result)
            self.cache.set(cache_key, metrics)
            self._author_cache[author_id] = metrics
            return metrics

        empty = {}
        self.cache.set(cache_key, empty)
        self._author_cache[author_id] = empty
        return empty

    def _extract_author_metrics(self, author: dict) -> dict:
        """Extract author-level metrics."""
        # OpenAlex devuelve estos campos como null, no ausentes, para autores
        # sin metricas. `.get(clave, defecto)` NO protege de eso: la clave
        # existe y su valor es None. Corto la fase 1d en el registro 123 de
        # 172 de PRIMARIA_MAT_CONTEXT.
        summary = author.get("summary_stats") or {}
        counts_by_year = author.get("counts_by_year") or []
        years = [c.get("year", 0) for c in counts_by_year if c and c.get("year")]
        career_start = min(years) if years else None
        career_age = (2026 - career_start) if career_start else None

        return {
            "openalex_author_id": author.get("id", ""),
            "display_name": author.get("display_name", ""),
            "h_index": summary.get("h_index", None),
            "i10_index": summary.get("i10_index", None),
            "2yr_mean_citedness": summary.get("2yr_mean_citedness", None),
            "works_count": author.get("works_count", 0),
            "cited_by_count": author.get("cited_by_count", 0),
            "career_start": career_start,
            "career_age": career_age,
            "last_known_institutions": [
                {
                    "name": inst.get("display_name", ""),
                    "country": inst.get("country_code", ""),
                }
                for inst in (author.get("last_known_institutions") or [])[:3]
            ],
        }

    # ------------------------------------------------------------------
    # Enrich one article
    # ------------------------------------------------------------------

    def enrich_article(self, article: dict) -> dict:
        """Enrich a single article with journal and author metrics."""
        # Metricas de revista en cascada por fiabilidad:
        #   1) ISSN (exacto)  2) DOI -> source OpenAlex (mapeo exacto)
        #   3) titulo de revista/actas (match difuso) — recolecta lo que se
        # pueda cuando no hay ISSN valido / la revista no esta indexada.
        openalex_data = article.get("openalex", {})
        if not isinstance(openalex_data, dict):
            openalex_data = {}

        issn = article.get("issn", "") or openalex_data.get("issn_l", "")
        jm = self._fetch_source_by_issn(issn)
        if not jm and openalex_data.get("source_id"):
            jm = self._fetch_full_source(openalex_data["source_id"])
        if not jm:
            jm = self._fetch_source_by_doi(article.get("doi", "") or openalex_data.get("doi", ""))
        if not jm:
            jm = self._fetch_source_by_name(
                article.get("publication_title", "") or openalex_data.get("source_display_name", "")
            )
        if jm:
            article["journal_metrics"] = jm

        # Author metrics via OpenAlex authorships
        authorships = openalex_data.get("authorships", [])
        if not isinstance(authorships, list):
            authorships = []
        author_metrics = []
        for auth in authorships:
            if not isinstance(auth, dict):
                continue
            aid = auth.get("openalex_id", "")
            if not aid:
                continue
            am = self._fetch_author(aid)
            if am:
                author_metrics.append(am)

        if author_metrics:
            article["author_metrics"] = author_metrics

        return article


def run_phase1d(
    articles: list[dict],
    rate_override: float | None = None,
) -> list[dict]:
    """Ejecuta Fase 1d: Enriquecimiento OpenAlex Meta (Sources + Authors)."""
    logger.info("=== FASE 1d: Enriquecimiento OpenAlex Meta (Sources + Authors) ===")
    try:
        from src.metadata_completer import MetadataCompleter
        MetadataCompleter().stage_a(articles)
    except Exception:
        pass
    try:
        from src.local_citation_enricher import run_local_citation_enrichment
        run_local_citation_enrichment(articles)
    except Exception as e:
        logger.warning(f"Local citation enrichment warning in 1d: {e}")
    enricher = OpenAlexMetaEnricher(rate_override=rate_override)

    enriched_journals = 0
    enriched_authors = 0

    for article in tqdm(articles, desc="OpenAlex Meta enrichment"):
        enricher.enrich_article(article)
        if article.get("journal_metrics"):
            enriched_journals += 1
        if article.get("author_metrics"):
            enriched_authors += 1

    logger.info(f"Journal metrics: {enriched_journals}/{len(articles)}")
    logger.info(f"Author metrics: {enriched_authors}/{len(articles)}")
    logger.info(
        f"Unique sources fetched: {len(enricher._source_cache)}, "
        f"Unique authors fetched: {len(enricher._author_cache)}"
    )

    save_json(articles, ENRICHED_PATH)
    logger.info(f"Checkpoint guardado: {ENRICHED_PATH}")

    return articles
