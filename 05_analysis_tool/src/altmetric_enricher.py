"""Fase 1f: Enriquecimiento via API Altmetric (impacto social).

Datos: Attention Score, menciones en Twitter/noticias/politicas/Wikipedia/blogs,
lectores Mendeley.
API gratis con API key (registro en altmetric.com). ~1 req/s tier gratuito.
"""
from __future__ import annotations

import logging
from typing import Optional

from tqdm import tqdm

from config import (
    ALTMETRIC_BASE_URL, ALTMETRIC_API_KEY, ALTMETRIC_RATE_LIMIT,
    ALTMETRIC_CACHE_DIR, ENRICHED_PATH,
)
from src.utils import (
    RateLimiter, DiskCache, create_session, safe_request, save_json,
)

logger = logging.getLogger("bibliometrico.altmetric")


class AltmetricEnricher:
    """Enriquecedor de articulos via Altmetric API."""

    def __init__(self):
        self.session = create_session()
        self.rate_limiter = RateLimiter(ALTMETRIC_RATE_LIMIT)
        self.cache = DiskCache(ALTMETRIC_CACHE_DIR)
        self.base_url = ALTMETRIC_BASE_URL
        self.api_key = ALTMETRIC_API_KEY

    def fetch_by_doi(self, doi: str) -> Optional[dict]:
        """Fetch Altmetric data for a DOI."""
        if not doi:
            return None

        cache_key = f"altmetric:doi:{doi}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        url = f"{self.base_url}/doi/{doi}"
        params = {}
        if self.api_key:
            params["key"] = self.api_key

        result = safe_request(
            url, session=self.session,
            rate_limiter=self.rate_limiter,
            params=params,
        )

        if result and "altmetric_id" in result:
            extracted = self._extract_data(result)
            self.cache.set(cache_key, extracted)
            return extracted

        # Cache misses
        self.cache.set(cache_key, {})
        return {}

    def _extract_data(self, data: dict) -> dict:
        """Extract relevant fields from Altmetric response."""
        context = data.get("context", {})
        counts = {}

        # Social media and mention counts
        counts["twitter"] = data.get("cited_by_tweeters_count", 0)
        counts["facebook"] = data.get("cited_by_fbwalls_count", 0)
        counts["news"] = data.get("cited_by_msm_count", 0)
        counts["blogs"] = data.get("cited_by_feeds_count", 0)
        counts["wikipedia"] = data.get("cited_by_wikipedia_count", 0)
        counts["policy"] = data.get("cited_by_policies_count", 0)
        counts["patents"] = data.get("cited_by_patents_count", 0)
        counts["peer_review"] = data.get("cited_by_peer_review_sites_count", 0)
        counts["reddit"] = data.get("cited_by_rdts_count", 0)
        counts["video"] = data.get("cited_by_videos_count", 0)
        counts["mendeley"] = data.get("readers", {}).get("mendeley", 0)

        return {
            "altmetric_id": data.get("altmetric_id"),
            "score": data.get("score", 0),
            "score_1d": data.get("history", {}).get("1d", 0),
            "score_1w": data.get("history", {}).get("1w", 0),
            "score_1m": data.get("history", {}).get("1m", 0),
            "score_1y": data.get("history", {}).get("1y", 0),
            "counts": counts,
            "url": data.get("details_url", ""),
            "image_url": data.get("images", {}).get("small", ""),
            # Context: percentiles
            "context_all_pct": (context.get("all") or {}).get("pct", None),
            "context_journal_pct": (context.get("journal") or {}).get("pct", None),
            "context_similar_age_pct": (context.get("similar_age_3m") or {}).get("pct", None),
        }

    def enrich_article(self, article: dict) -> dict:
        """Enrich a single article with Altmetric data."""
        doi = article.get("doi", "")
        if not doi:
            return article

        data = self.fetch_by_doi(doi)
        if not data:
            return article

        article["altmetric"] = data
        article["altmetric_score"] = data.get("score")

        counts = data.get("counts", {})
        article["social_media_mentions"] = {
            "twitter": counts.get("twitter", 0),
            "facebook": counts.get("facebook", 0),
            "news": counts.get("news", 0),
            "blogs": counts.get("blogs", 0),
            "wikipedia": counts.get("wikipedia", 0),
            "reddit": counts.get("reddit", 0),
            "video": counts.get("video", 0),
            "mendeley": counts.get("mendeley", 0),
        }
        article["policy_citations"] = counts.get("policy", 0)

        return article


def run_phase1f(articles: list[dict]) -> list[dict]:
    """Ejecuta Fase 1f: Enriquecimiento Altmetric (impacto social)."""
    logger.info("=== FASE 1f: Enriquecimiento Altmetric ===")

    if not ALTMETRIC_API_KEY:
        logger.warning(
            "Altmetric enrichment skipped: ALTMETRIC_API_KEY not set. "
            "Register at https://www.altmetric.com/research-access/ "
            "and set ALTMETRIC_API_KEY env var."
        )
        return articles

    enricher = AltmetricEnricher()

    enriched = 0
    total_score = 0.0
    for article in tqdm(articles, desc="Altmetric enrichment"):
        enricher.enrich_article(article)
        score = article.get("altmetric_score")
        if score and score > 0:
            enriched += 1
            total_score += score

    logger.info(f"Altmetric enriched: {enriched}/{len(articles)}")
    if enriched > 0:
        logger.info(f"  Mean Altmetric score: {total_score / enriched:.1f}")

    # Top social impact
    scored = [(a, a.get("altmetric_score", 0) or 0) for a in articles]
    scored.sort(key=lambda x: x[1], reverse=True)
    top5 = scored[:5]
    if top5 and top5[0][1] > 0:
        logger.info("  Top 5 Altmetric scores:")
        for a, s in top5:
            if s > 0:
                logger.info(f"    [{a['key']}] {a['title'][:50]}... score={s}")

    save_json(articles, ENRICHED_PATH)
    logger.info(f"Checkpoint guardado: {ENRICHED_PATH}")

    return articles
