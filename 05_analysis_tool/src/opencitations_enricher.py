"""Fase 1g: Enriquecimiento via API OpenCitations (grafo de citas abierto).

Tres endpoints:
- /index/v2/citations/doi:X  — quien cita este articulo (ya existente)
- /index/v2/references/doi:X — a quien cita este articulo (NUEVO)
- /meta/v1/metadata/doi:X    — metadatos del articulo (NUEVO)

Datos: fechas exactas de citacion, auto-citas de revista/autor, trayectoria
temporal de citas, citation velocity, referencias outbound, metadata gaps.
API gratis, CC0. 3 req/s (mas con token).
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from tqdm import tqdm

from config import (
    OPENCITATIONS_BASE_URL, OPENCITATIONS_META_URL,
    OPENCITATIONS_TOKEN, OPENCITATIONS_RATE_LIMIT,
    OPENCITATIONS_CACHE_DIR, ENRICHED_PATH,
)
from src.utils import (
    RateLimiter, DiskCache, create_session, safe_request, save_json,
)

logger = logging.getLogger("bibliometrico.opencitations")


class OpenCitationsEnricher:
    """Enriquecedor de articulos via OpenCitations API."""

    def __init__(self):
        self.session = create_session()
        self.rate_limiter = RateLimiter(OPENCITATIONS_RATE_LIMIT)
        self.cache = DiskCache(OPENCITATIONS_CACHE_DIR)
        self.base_url = OPENCITATIONS_BASE_URL
        self.meta_url = OPENCITATIONS_META_URL
        self.token = OPENCITATIONS_TOKEN

        if self.token:
            self.session.headers.update({
                "authorization": self.token,
            })

    # ------------------------------------------------------------------
    # Low-level fetchers
    # ------------------------------------------------------------------

    def _fetch_list(self, url: str, cache_key: str) -> list[dict]:
        """Generic fetcher for OpenCitations list-returning endpoints."""
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        result = safe_request(
            url, session=self.session,
            rate_limiter=self.rate_limiter,
            timeout=60,
        )

        if result is None:
            # API returns a list, not a dict — try direct parsing
            try:
                self.rate_limiter.wait()
                resp = self.session.get(url, timeout=60)
                if resp.status_code == 200:
                    result = resp.json()
            except Exception:
                pass

        if isinstance(result, list):
            self.cache.set(cache_key, result)
            return result

        self.cache.set(cache_key, [])
        return []

    def fetch_citations(self, doi: str) -> list[dict]:
        """Fetch all citations TO a DOI (who cites this article)."""
        if not doi:
            return []
        url = f"{self.base_url}/citations/doi:{doi}"
        return self._fetch_list(url, f"opencit:citations:{doi}")

    def fetch_references(self, doi: str) -> list[dict]:
        """Fetch all references FROM a DOI (what this article cites)."""
        if not doi:
            return []
        url = f"{self.base_url}/references/doi:{doi}"
        return self._fetch_list(url, f"opencit:references:{doi}")

    def fetch_metadata(self, doi: str) -> Optional[dict]:
        """Fetch metadata for a DOI from OpenCitations Meta API."""
        if not doi:
            return None

        cache_key = f"opencit:meta:{doi}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached if cached else None

        url = f"{self.meta_url}/metadata/doi:{doi}"
        try:
            self.rate_limiter.wait()
            resp = self.session.get(url, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and data:
                    result = data[0]  # Meta API returns a list with 1 element
                    self.cache.set(cache_key, result)
                    return result
        except Exception:
            pass

        self.cache.set(cache_key, {})
        return None

    # ------------------------------------------------------------------
    # Analysis: citations (inbound)
    # ------------------------------------------------------------------

    def _extract_citation_data(
        self, citations: list[dict], article: dict
    ) -> dict:
        """Analyze citations: self-citations, velocity, temporal trajectory."""
        total = len(citations)
        if total == 0:
            return {
                "total_citations": 0,
                "self_citations_author": 0,
                "self_citations_journal": 0,
                "self_citation_rate": 0.0,
                "citation_velocity": 0.0,
                "citations_by_year": {},
                "citation_dates": [],
            }

        # Article metadata for self-citation detection
        article_authors = set()
        for auth in article.get("authors", []):
            parts = auth.split(",")
            if parts:
                article_authors.add(parts[0].strip().lower())
        for auth in article.get("openalex", {}).get("authorships", []):
            name = auth.get("name", "")
            parts = name.split()
            if parts:
                article_authors.add(parts[-1].strip().lower())

        pub_year = article.get("publication_year")

        self_cite_author = 0
        self_cite_journal = 0
        citation_dates = []
        citations_by_year = {}

        for cit in citations:
            creation_date = cit.get("creation", "")
            journal_sc = cit.get("journal_sc", "no")
            author_sc = cit.get("author_sc", "no")

            if str(author_sc).lower() == "yes":
                self_cite_author += 1
            if str(journal_sc).lower() == "yes":
                self_cite_journal += 1

            if creation_date:
                try:
                    year = int(creation_date[:4])
                    citations_by_year[year] = citations_by_year.get(year, 0) + 1
                    citation_dates.append(creation_date)
                except (ValueError, IndexError):
                    pass

        self_citation_rate = self_cite_author / total if total > 0 else 0.0

        if pub_year and pub_year < 2100:
            current_year = datetime.now().year
            years_since = max(1, current_year - pub_year)
            citation_velocity = total / years_since
        else:
            citation_velocity = 0.0

        return {
            "total_citations": total,
            "self_citations_author": self_cite_author,
            "self_citations_journal": self_cite_journal,
            "self_citation_rate": round(self_citation_rate, 4),
            "citation_velocity": round(citation_velocity, 2),
            "citations_by_year": dict(sorted(citations_by_year.items())),
            "citation_dates": sorted(citation_dates)[:100],
        }

    # ------------------------------------------------------------------
    # Analysis: references (outbound)
    # ------------------------------------------------------------------

    def _extract_references_data(self, references: list[dict]) -> dict:
        """Extract outbound reference data from OpenCitations references endpoint."""
        total = len(references)
        if total == 0:
            return {"total_references": 0, "reference_dois": [], "references_by_year": {}}

        ref_dois = []
        refs_by_year = {}

        for ref in references:
            # 'cited' field contains the referenced DOI
            cited = ref.get("cited", "")
            if cited:
                # Extract DOI from format "doi:10.xxxx/yyyy" or plain
                doi = cited.replace("doi:", "").strip()
                if doi:
                    ref_dois.append(doi)

            creation = ref.get("creation", "")
            if creation:
                try:
                    year = int(creation[:4])
                    refs_by_year[year] = refs_by_year.get(year, 0) + 1
                except (ValueError, IndexError):
                    pass

        return {
            "total_references": total,
            "reference_dois": ref_dois,
            "references_by_year": dict(sorted(refs_by_year.items())),
        }

    # ------------------------------------------------------------------
    # Analysis: metadata gap filling
    # ------------------------------------------------------------------

    def _fill_metadata_gaps(self, article: dict, meta: dict) -> dict:
        """Fill missing article fields from OpenCitations Meta API.

        Only fills fields that are currently empty/missing.
        Returns dict of fields that were filled.
        """
        filled = {}

        # Volume
        if not article.get("volume") or article["volume"] in ("", "nan"):
            vol = meta.get("volume", "")
            if vol:
                article["volume"] = str(vol)
                filled["volume"] = vol

        # Issue
        if not article.get("issue") or article["issue"] in ("", "nan"):
            issue = meta.get("issue", "")
            if issue:
                article["issue"] = str(issue)
                filled["issue"] = issue

        # Pages
        if not article.get("pages") or article["pages"] in ("", "nan"):
            page = meta.get("page", "")
            if page:
                article["pages"] = str(page)
                filled["pages"] = page

        # Publisher
        if not article.get("publisher") or article["publisher"] in ("", "nan"):
            pub = meta.get("publisher", "")
            if pub:
                article["publisher"] = str(pub)
                filled["publisher"] = pub

        # Publication title (journal/venue)
        if not article.get("publication_title") or article["publication_title"] in ("", "nan"):
            source_title = meta.get("source_title", "")
            if source_title:
                article["publication_title"] = str(source_title)
                filled["publication_title"] = source_title

        # ISSN
        if not article.get("issn") or article["issn"] in ("", "nan"):
            source_id = meta.get("source_id", "")
            if source_id and "issn:" in source_id:
                # source_id format: "issn:1234-5678"
                import re
                issn_match = re.search(r"issn:(\d{4}-\d{3}[\dXx])", source_id)
                if issn_match:
                    article["issn"] = issn_match.group(1)
                    filled["issn"] = issn_match.group(1)

        # Publication year
        if not article.get("publication_year"):
            year_str = meta.get("year", "")
            if year_str:
                try:
                    article["publication_year"] = int(year_str)
                    filled["publication_year"] = int(year_str)
                except (ValueError, TypeError):
                    pass

        return filled

    # ------------------------------------------------------------------
    # Main enrichment per article
    # ------------------------------------------------------------------

    def enrich_article(self, article: dict) -> dict:
        """Enrich a single article with all OpenCitations data."""
        doi = article.get("doi", "")
        if not doi:
            return article

        # 1. Citations (inbound) — who cites this article
        citations = self.fetch_citations(doi)
        cit_data = self._extract_citation_data(citations, article)
        article["opencitations"] = cit_data
        article["self_citation_rate"] = cit_data["self_citation_rate"]
        article["citation_velocity"] = cit_data["citation_velocity"]

        # 2. References (outbound) — what this article cites
        references = self.fetch_references(doi)
        ref_data = self._extract_references_data(references)
        article["opencitations"]["references"] = ref_data

        # Merge reference DOIs into referenced_works if not already populated
        existing_refs = set(article.get("referenced_works", []))
        new_refs = [d for d in ref_data["reference_dois"] if d not in existing_refs]
        if new_refs:
            article["referenced_works"] = list(existing_refs) + new_refs

        # 3. Metadata gap-filling
        meta = self.fetch_metadata(doi)
        if meta:
            filled = self._fill_metadata_gaps(article, meta)
            article["opencitations"]["metadata_filled"] = filled

        return article


def run_phase1g(articles: list[dict]) -> list[dict]:
    """Ejecuta Fase 1g: Enriquecimiento OpenCitations."""
    logger.info("=" * 60)
    logger.info("FASE 1g: Enriquecimiento OpenCitations")
    logger.info("  Endpoints: citations (inbound) + references (outbound) + metadata")
    logger.info("=" * 60)
    enricher = OpenCitationsEnricher()

    enriched = 0
    total_cites = 0
    total_self = 0
    total_refs_added = 0
    meta_fields_filled = {}

    for article in tqdm(articles, desc="OpenCitations enrichment"):
        # Count existing refs before
        refs_before = len(article.get("referenced_works", []))

        enricher.enrich_article(article)

        oc = article.get("opencitations", {})
        if oc.get("total_citations", 0) > 0:
            enriched += 1
            total_cites += oc["total_citations"]
            total_self += oc.get("self_citations_author", 0)

        # Count new refs added
        refs_after = len(article.get("referenced_works", []))
        total_refs_added += (refs_after - refs_before)

        # Track metadata fills
        for field in oc.get("metadata_filled", {}):
            meta_fields_filled[field] = meta_fields_filled.get(field, 0) + 1

    # --- Summary ---
    logger.info(f"OpenCitations enriched: {enriched}/{len(articles)}")

    if enriched > 0:
        logger.info(f"  Citations (inbound): {total_cites} total")
        logger.info(
            f"  Author self-citation rate: "
            f"{total_self / total_cites * 100:.1f}% ({total_self}/{total_cites})"
        )

    # Citation velocity stats
    velocities = [
        a.get("citation_velocity", 0) or 0 for a in articles
        if a.get("citation_velocity") is not None
    ]
    if velocities:
        import statistics
        logger.info(
            f"  Citation velocity: mean={statistics.mean(velocities):.2f}, "
            f"median={statistics.median(velocities):.2f} cites/year"
        )

    # References stats
    if total_refs_added > 0:
        logger.info(f"  References (outbound): +{total_refs_added} DOIs added to referenced_works")

    # Metadata fill stats
    if meta_fields_filled:
        logger.info(f"  Metadata gaps filled via OpenCitations Meta:")
        for field, count in sorted(meta_fields_filled.items(), key=lambda x: -x[1]):
            logger.info(f"    {field}: +{count}")

    save_json(articles, ENRICHED_PATH)
    logger.info(f"Checkpoint guardado: {ENRICHED_PATH}")

    return articles
