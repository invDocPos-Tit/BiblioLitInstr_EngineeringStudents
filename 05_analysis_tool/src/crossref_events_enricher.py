"""Fase 1l: Enriquecimiento via Crossref Event Data (impacto social abierto).

Crossref Event Data rastrea menciones de DOIs en fuentes externas:
- Wikipedia (en multiples idiomas)
- Twitter / X
- Reddit
- Hypothesis (anotaciones)
- Newsfeed (blogs, noticias)
- DataCite (datasets que citan el articulo)
- StackExchange
- Web (enlaces genericos)

API gratuita, sin API key requerida. Solo necesita un mailto para cortesia.
Endpoint: https://api.eventdata.crossref.org/v1/events?obj-id=DOI&mailto=EMAIL

Datos extraidos por articulo:
- event_total: conteo total de eventos
- event_by_source: desglose por fuente (wikipedia, twitter, reddit, etc.)
- event_first_date / event_last_date: rango temporal de menciones
- wikipedia_langs: idiomas de Wikipedia donde se menciona
- event_timeline: eventos por anio (para velocity)
"""
from __future__ import annotations

import logging
import os
from collections import Counter, defaultdict
from typing import Optional

from tqdm import tqdm

from config import DATA_DIR, ENRICHED_PATH
from src.utils import RateLimiter, DiskCache, create_session, save_json

logger = logging.getLogger("bibliometrico.crossref_events")

# Cache directory
EVENTS_CACHE_DIR = DATA_DIR / "crossref_events_cache"

# API config
EVENTS_API_BASE = "https://api.eventdata.crossref.org/v1/events"
EVENTS_RATE_LIMIT = 5.0  # requests per second (generous, API is lenient)
EVENTS_MAILTO = os.getenv("CROSSREF_MAILTO", os.getenv("UNPAYWALL_EMAIL", ""))


class CrossrefEventsEnricher:
    """Enriquecedor de articulos via Crossref Event Data API."""

    def __init__(self, rate_limit: float = EVENTS_RATE_LIMIT):
        self.session = create_session()
        self.rate_limiter = RateLimiter(rate_limit)
        self.cache = DiskCache(EVENTS_CACHE_DIR)

        # Use polite mailto
        self.mailto = EVENTS_MAILTO
        if not self.mailto:
            logger.warning(
                "No CROSSREF_MAILTO or UNPAYWALL_EMAIL set. "
                "Crossref Event Data requests will be anonymous (slower)."
            )

    def _fetch_events(self, doi: str) -> list[dict]:
        """Fetch all events for a DOI from Crossref Event Data."""
        cache_key = f"events_{doi.replace('/', '_')}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        self.rate_limiter.wait()

        params = {"obj-id": doi, "rows": 1000}
        if self.mailto:
            params["mailto"] = self.mailto

        try:
            resp = self.session.get(
                EVENTS_API_BASE,
                params=params,
                timeout=30,
            )
            if resp.status_code == 200:
                data = resp.json()
                events = data.get("message", {}).get("events", [])
                self.cache.set(cache_key, events)
                return events
            elif resp.status_code == 404:
                self.cache.set(cache_key, [])
                return []
            else:
                logger.debug(f"Event Data API {resp.status_code} for {doi}")
                return []
        except Exception as e:
            logger.debug(f"Event Data API error for {doi}: {e}")
            return []

    def enrich_article(self, article: dict) -> dict:
        """Enrich a single article with Crossref Event Data."""
        doi = article.get("doi", "")
        if not doi:
            return {}

        events = self._fetch_events(doi)
        if not events:
            return {"event_total": 0}

        # Aggregate by source
        by_source = Counter()
        by_year = Counter()
        wikipedia_langs = set()
        dates = []

        for evt in events:
            source = evt.get("source_id", "unknown")
            by_source[source] += 1

            occurred = evt.get("occurred_at", "")
            if occurred:
                dates.append(occurred[:10])
                year = occurred[:4]
                if year.isdigit():
                    by_year[int(year)] += 1

            # Extract Wikipedia language
            if source == "wikipedia":
                subj = evt.get("subj_id", "") or evt.get("subj", {}).get("pid", "")
                if ".wikipedia.org" in subj:
                    lang = subj.split(".wikipedia.org")[0].split("//")[-1]
                    if lang and len(lang) <= 5:
                        wikipedia_langs.add(lang)

        result = {
            "event_total": len(events),
            "event_by_source": dict(by_source.most_common()),
            "event_timeline": dict(sorted(by_year.items())),
        }

        if dates:
            result["event_first_date"] = min(dates)
            result["event_last_date"] = max(dates)

        if wikipedia_langs:
            result["wikipedia_langs"] = sorted(wikipedia_langs)
            result["wikipedia_mentions"] = by_source.get("wikipedia", 0)

        # Specific source counts for easy access
        for src in ["wikipedia", "twitter", "reddit", "hypothesis",
                     "newsfeed", "datacite", "stackexchange", "web"]:
            if by_source.get(src, 0) > 0:
                result[f"event_{src}"] = by_source[src]

        return result


def run_phase1l(
    articles: list[dict],
    rate_limit: float = EVENTS_RATE_LIMIT,
) -> list[dict]:
    """Run Phase 1l: Crossref Event Data enrichment.

    Args:
        articles: List of enriched article dicts.
        rate_limit: Requests per second (default: 5.0).

    Returns:
        Updated articles list.
    """
    logger.info("=" * 60)
    logger.info("Phase 1l: Crossref Event Data (social impact)")
    logger.info("=" * 60)

    EVENTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    enricher = CrossrefEventsEnricher(rate_limit=rate_limit)

    # Filter articles with DOI that haven't been enriched yet
    to_process = [
        a for a in articles
        if a.get("doi") and "event_total" not in a
    ]
    already = len(articles) - len(to_process)
    no_doi = sum(1 for a in articles if not a.get("doi"))

    logger.info(
        f"Articles: {len(articles)} total, {len(to_process)} to process, "
        f"{already} already enriched, {no_doi} without DOI"
    )

    if not to_process:
        logger.info("Nothing to process — all articles already enriched or without DOI")
        return articles

    n_enriched = 0
    n_with_events = 0
    total_events = 0

    for article in tqdm(to_process, desc="Crossref Events", unit="art"):
        result = enricher.enrich_article(article)
        if result:
            for k, v in result.items():
                article[k] = v
            n_enriched += 1
            if result.get("event_total", 0) > 0:
                n_with_events += 1
                total_events += result["event_total"]

    # Summary stats
    source_totals = Counter()
    for a in articles:
        for src, count in a.get("event_by_source", {}).items():
            source_totals[src] += count

    logger.info(f"Enriched: {n_enriched} articles processed")
    logger.info(f"With events: {n_with_events}/{n_enriched} ({total_events} total events)")
    if source_totals:
        logger.info(f"Events by source: {dict(source_totals.most_common(10))}")

    save_json(articles, ENRICHED_PATH)
    logger.info(f"Checkpoint saved: {ENRICHED_PATH}")

    return articles
