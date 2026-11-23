"""Fase 1h: Enriquecimiento via API Unpaywall (OA detallado).

Datos: color OA (gold/green/hybrid/bronze), repositorios, licencias, URLs de PDF.
API gratis con email. 100k calls/dia, 10 req/s.
"""
from __future__ import annotations

import logging
from typing import Optional

from tqdm import tqdm

from config import (
    UNPAYWALL_BASE_URL, UNPAYWALL_EMAIL, UNPAYWALL_RATE_LIMIT,
    UNPAYWALL_CACHE_DIR, ENRICHED_PATH,
)
from src.utils import (
    RateLimiter, DiskCache, create_session, safe_request, save_json,
    get_canonical_oa_status,
)

logger = logging.getLogger("bibliometrico.unpaywall")


class UnpaywallEnricher:
    """Enriquecedor de articulos via Unpaywall API."""

    def __init__(self):
        self.session = create_session()
        self.rate_limiter = RateLimiter(UNPAYWALL_RATE_LIMIT)
        self.cache = DiskCache(UNPAYWALL_CACHE_DIR)
        self.base_url = UNPAYWALL_BASE_URL
        self.email = UNPAYWALL_EMAIL

        if not self.email:
            logger.warning(
                "UNPAYWALL_EMAIL not set. Set UNPAYWALL_EMAIL or OPENALEX_EMAIL "
                "env var for Unpaywall API access."
            )

    def fetch_by_doi(self, doi: str) -> Optional[dict]:
        """Fetch OA data for a DOI."""
        if not doi or not self.email:
            return None

        cache_key = f"unpaywall:{doi}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        url = f"{self.base_url}/{doi}"
        params = {"email": self.email}

        result = safe_request(
            url, session=self.session,
            rate_limiter=self.rate_limiter,
            params=params,
        )

        if result and "is_oa" in result:
            extracted = self._extract_data(result)
            self.cache.set(cache_key, extracted)
            return extracted

        # Cache misses too
        self.cache.set(cache_key, {})
        return {}

    def _extract_data(self, data: dict) -> dict:
        """Extract relevant fields from Unpaywall response."""
        best_oa = data.get("best_oa_location") or {}

        # All OA locations
        oa_locations = []
        for loc in data.get("oa_locations", []):
            oa_locations.append({
                "url": loc.get("url", ""),
                "url_for_pdf": loc.get("url_for_pdf", ""),
                "repository_institution": loc.get("repository_institution", ""),
                "host_type": loc.get("host_type", ""),
                "version": loc.get("version", ""),
                "license": loc.get("license", ""),
                "is_best": loc.get("is_best", False),
            })

        return {
            "is_oa": data.get("is_oa", False),
            "oa_status": data.get("oa_status", ""),  # gold, green, hybrid, bronze, closed
            "journal_is_oa": data.get("journal_is_oa", False),
            "journal_is_in_doaj": data.get("journal_is_in_doaj", False),
            "best_oa_url": best_oa.get("url", ""),
            "best_oa_pdf_url": best_oa.get("url_for_pdf", ""),
            "best_oa_host_type": best_oa.get("host_type", ""),
            "best_oa_version": best_oa.get("version", ""),
            "best_oa_license": best_oa.get("license", ""),
            "oa_locations": oa_locations,
            "oa_locations_count": len(oa_locations),
            "publisher": data.get("publisher", ""),
            "journal_issns": data.get("journal_issns", ""),
            "has_repository_copy": data.get("has_repository_copy", False),
        }

    def enrich_article(self, article: dict) -> dict:
        """Enrich a single article with Unpaywall data."""
        doi = article.get("doi", "")
        if not doi:
            return article

        data = self.fetch_by_doi(doi)
        if not data:
            return article

        # Store full Unpaywall response for downstream use
        article["unpaywall"] = data

        # Enrich the existing open_access field with detailed info
        existing_oa = article.get("open_access") or {}
        if isinstance(existing_oa, dict):
            existing_oa.update({
                "journal_is_oa": data.get("journal_is_oa", False),
                "journal_is_in_doaj": data.get("journal_is_in_doaj", False),
                "best_oa_url": data.get("best_oa_url", ""),
                "best_oa_pdf_url": data.get("best_oa_pdf_url", ""),
                "best_oa_license": data.get("best_oa_license", ""),
                "best_oa_version": data.get("best_oa_version", ""),
                "has_repository_copy": data.get("has_repository_copy", False),
                "oa_locations_count": data.get("oa_locations_count", 0),
            })
            canonical_status = get_canonical_oa_status(article)
            existing_oa["status"] = canonical_status
            existing_oa["oa_color"] = canonical_status
            article["open_access"] = existing_oa

        return article


def run_phase1h(articles: list[dict]) -> list[dict]:
    """Ejecuta Fase 1h: Enriquecimiento Unpaywall (OA detallado)."""
    logger.info("=== FASE 1h: Enriquecimiento Unpaywall ===")

    if not UNPAYWALL_EMAIL:
        logger.warning(
            "Unpaywall enrichment skipped: no email configured. "
            "Set UNPAYWALL_EMAIL or OPENALEX_EMAIL env var."
        )
        return articles

    enricher = UnpaywallEnricher()

    enriched = 0
    with_pdf = 0
    for article in tqdm(articles, desc="Unpaywall enrichment"):
        enricher.enrich_article(article)
        up = article.get("unpaywall", {})
        if up.get("is_oa"):
            enriched += 1
        if up.get("best_oa_pdf_url"):
            with_pdf += 1

    logger.info(f"Unpaywall OA: {enriched}/{len(articles)}")
    logger.info(f"With PDF URL: {with_pdf}")

    # OA color distribution
    color_dist = {}
    for a in articles:
        color = (a.get("unpaywall") or {}).get("oa_status", "")
        if color:
            color_dist[color] = color_dist.get(color, 0) + 1
    if color_dist:
        logger.info(f"  OA color distribution: {color_dist}")

    save_json(articles, ENRICHED_PATH)
    logger.info(f"Checkpoint guardado: {ENRICHED_PATH}")

    return articles
