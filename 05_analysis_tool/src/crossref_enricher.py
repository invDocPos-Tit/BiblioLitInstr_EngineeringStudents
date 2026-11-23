"""Fase 1b: Enriquecimiento via API Crossref."""
from __future__ import annotations

import logging
from typing import Optional

from rapidfuzz import fuzz
from tqdm import tqdm

from config import (
    CROSSREF_BASE_URL, CROSSREF_EMAIL, CROSSREF_PLUS_API_TOKEN,
    CROSSREF_RATE_LIMIT, CROSSREF_CACHE_DIR, CROSSREF_FUZZY_THRESHOLD,
    ENRICHED_PATH,
)
from src.utils import (
    RateLimiter, DiskCache, create_session, safe_request,
    normalize_doi, save_json,
)

logger = logging.getLogger("bibliometrico.crossref")


class CrossrefEnricher:
    """Enriquecedor de articulos via Crossref API."""

    def __init__(self):
        self.session = create_session()
        self.rate_limiter = RateLimiter(CROSSREF_RATE_LIMIT)
        self.cache = DiskCache(CROSSREF_CACHE_DIR)
        self.base_url = CROSSREF_BASE_URL
        self.email = CROSSREF_EMAIL
        self.token = CROSSREF_PLUS_API_TOKEN

        # Polite pool: User-Agent con email
        user_agent = "BiblioMetrico/1.0 (mailto:{})".format(
            self.email or "your.email@example.org"
        )
        self.session.headers.update({"User-Agent": user_agent})

        # Crossref Plus API token (optional, for higher rate limits)
        if self.token:
            self.session.headers.update({
                "Crossref-Plus-API-Token": f"Bearer {self.token}"
            })
            logger.info("Using Crossref Plus API token")

    def search_by_doi(self, doi: str) -> Optional[dict]:
        """Busca un trabajo por DOI en Crossref."""
        if not doi:
            return None

        cache_key = f"crossref:doi:{doi}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            logger.debug(f"Cache hit: {doi}")
            return cached

        url = f"{self.base_url}/works/{doi}"
        result = safe_request(
            url, session=self.session,
            rate_limiter=self.rate_limiter,
        )

        if result and "message" in result:
            work = result["message"]
            self.cache.set(cache_key, work)
            return work
        return None

    def search_by_title(self, title: str) -> Optional[dict]:
        """Busca un trabajo por titulo con fuzzy matching."""
        if not title:
            return None

        cache_key = f"crossref:title:{title[:100]}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        url = f"{self.base_url}/works"
        params = {
            "query.bibliographic": title[:200],
            "rows": 5,
        }

        result = safe_request(
            url, session=self.session,
            rate_limiter=self.rate_limiter,
            params=params,
        )

        if not result or "message" not in result:
            return None

        items = result["message"].get("items", [])
        if not items:
            return None

        # Fuzzy match
        best_match = None
        best_score = 0
        for work in items:
            work_titles = work.get("title", [])
            if not work_titles:
                continue
            work_title = work_titles[0]
            score = fuzz.ratio(title.lower(), work_title.lower())
            if score > best_score:
                best_score = score
                best_match = work

        if best_match and best_score >= CROSSREF_FUZZY_THRESHOLD:
            self.cache.set(cache_key, best_match)
            logger.info(f"Title match (score={best_score}): {title[:60]}")
            return best_match

        logger.warning(f"No title match (best={best_score}): {title[:60]}")
        return None

    def extract_work_data(self, work: dict) -> dict:
        """Extrae datos relevantes de un resultado Crossref."""
        container = work.get("container-title") or []
        data = {
            "doi": normalize_doi(str(work.get("DOI", ""))) if work.get("DOI") else "",
            "cited_by_count": work.get("is-referenced-by-count", 0),
            "type": work.get("type", ""),
            "publisher": work.get("publisher", ""),
            "container_title": container[0] if isinstance(container, list) and container else str(container or ""),
            "issn": work.get("ISSN") or [],
            "volume": str(work.get("volume", "") or "").strip(),
            "issue": str(work.get("issue", "") or "").strip(),
            "page": str(work.get("page", "") or "").strip(),
        }

        # Funders
        funders = []
        for funder in work.get("funder", []):
            funder_info = {
                "name": funder.get("name", ""),
                "doi": funder.get("DOI", ""),
                "award": funder.get("award", []),
            }
            funders.append(funder_info)
        data["funders"] = funders

        # Licenses
        licenses = []
        for lic in work.get("license", []):
            licenses.append({
                "url": lic.get("URL", ""),
                "content_version": lic.get("content-version", ""),
                "delay_in_days": lic.get("delay-in-days", 0),
            })
        data["licenses"] = licenses

        # References with DOIs
        references = []
        for ref in work.get("reference", []):
            ref_info = {
                "key": ref.get("key", ""),
                "doi": ref.get("DOI", ""),
                "author": ref.get("author", ""),
                "year": ref.get("year", ""),
                "journal_title": ref.get("journal-title", ""),
                "article_title": ref.get("article-title", ""),
                "volume": ref.get("volume", ""),
                "first_page": ref.get("first-page", ""),
            }
            references.append(ref_info)
        data["references"] = references

        # Author ORCIDs
        author_orcids = []
        for author in work.get("author", []):
            orcid = author.get("ORCID", "")
            if orcid:
                # Normalize ORCID: extract just the ID
                orcid_id = orcid.replace("http://orcid.org/", "").replace(
                    "https://orcid.org/", ""
                )
                author_orcids.append({
                    "name": "{} {}".format(
                        author.get("given", ""), author.get("family", "")
                    ).strip(),
                    "orcid": orcid_id,
                    "authenticated": author.get("authenticated-orcid", False),
                })
        data["author_orcids"] = author_orcids

        # Subjects
        data["subjects"] = work.get("subject", [])

        # Clinical trial numbers
        data["clinical_trials"] = work.get("clinical-trial-number", [])

        # Abstract (Crossref provides JATS XML abstract for some works)
        abstract_raw = work.get("abstract", "")
        if abstract_raw:
            import re
            # Strip JATS XML tags: <jats:p>, <jats:italic>, etc.
            data["abstract"] = re.sub(r"<[^>]+>", "", abstract_raw).strip()
        else:
            data["abstract"] = ""

        # Subtitle
        subtitles = work.get("subtitle", [])
        data["subtitle"] = subtitles[0] if subtitles else ""

        # Short container title (journal abbreviation)
        short_titles = work.get("short-container-title", [])
        data["short_container_title"] = short_titles[0] if short_titles else ""

        # Alternative IDs (publisher internal IDs)
        data["alternative_ids"] = work.get("alternative-id", [])

        # Archive (CLOCKSS, LOCKSS, Portico preservation)
        data["archive"] = work.get("archive", [])

        # Relation (updates, corrections, etc.)
        data["relation"] = work.get("relation", {})

        # Content domain
        data["content_domain"] = work.get("content-domain", {}).get("domain", [])

        return data

    def enrich_article(self, article: dict) -> dict:
        """Enriquece un articulo individual con datos de Crossref."""
        doi = article.get("doi", "")
        title = article.get("title", "")

        # Buscar por DOI primero, luego por titulo
        work = self.search_by_doi(doi)
        if not work:
            logger.info(f"DOI not found in Crossref, trying title: {title[:60]}")
            work = self.search_by_title(title)

        if work:
            enrichment = self.extract_work_data(work)
            article["crossref"] = enrichment

            if enrichment.get("doi") and (not article.get("doi") or article["doi"] in ("", "nan")):
                article["doi"] = enrichment["doi"]
                if not article.get("url") or article["url"] in ("", "nan"):
                    article["url"] = f"https://doi.org/{enrichment['doi']}"
            if enrichment.get("container_title") and (not article.get("publication_title") or article["publication_title"] in ("", "nan")):
                article["publication_title"] = enrichment["container_title"]
            if enrichment.get("issn") and (not article.get("issn") or article["issn"] in ("", "nan")):
                article["issn"] = str(enrichment["issn"][0]).strip()
            if enrichment.get("publisher") and (not article.get("publisher") or article["publisher"] in ("", "nan")):
                article["publisher"] = enrichment["publisher"]
            if enrichment.get("volume") and (not article.get("volume") or article["volume"] in ("", "nan")):
                article["volume"] = enrichment["volume"]
            if enrichment.get("issue") and (not article.get("issue") or article["issue"] in ("", "nan")):
                article["issue"] = enrichment["issue"]
            if enrichment.get("page") and (not article.get("pages") or article["pages"] in ("", "nan")):
                article["pages"] = enrichment["page"]

            # Merge cited_by_count (max con existente)
            article["cited_by_count"] = max(
                article.get("cited_by_count", 0),
                enrichment["cited_by_count"],
            )

            # Merge funding
            if enrichment["funders"]:
                article["funding"] = enrichment["funders"]

            # Merge licenses
            if enrichment["licenses"]:
                article["licenses"] = enrichment["licenses"]

            # Merge author ORCIDs
            if enrichment["author_orcids"]:
                article["author_orcids"] = enrichment["author_orcids"]

            # Merge references (detailed with DOIs)
            if enrichment["references"]:
                article["referenced_works_detailed"] = enrichment["references"]

            # Merge subjects into concepts with source tracking
            existing_names = {
                c.get("name", "").lower() for c in article.get("concepts", [])
            }
            for subj in enrichment.get("subjects", []):
                if subj.lower() not in existing_names:
                    article.setdefault("concepts", []).append({
                        "name": subj,
                        "level": 0,
                        "score": 0,
                        "source": "crossref",
                    })
                    existing_names.add(subj.lower())

            # Clinical trials
            if enrichment["clinical_trials"]:
                article["clinical_trials"] = enrichment["clinical_trials"]

            # Abstract gap-fill (Crossref → article if empty)
            if enrichment.get("abstract") and (not article.get("abstract") or article["abstract"] in ("", "nan")):
                article["abstract"] = enrichment["abstract"]

            # Journal abbreviation gap-fill
            if enrichment.get("short_container_title") and (not article.get("journal_abbrev") or article["journal_abbrev"] in ("", "nan")):
                article["journal_abbrev"] = enrichment["short_container_title"]

            # Subtitle
            if enrichment.get("subtitle"):
                article["subtitle"] = enrichment["subtitle"]

            # Alternative IDs
            if enrichment.get("alternative_ids"):
                article["alternative_ids"] = enrichment["alternative_ids"]

            logger.info(
                f"  Crossref [{article['key']}]: "
                f"funders={len(enrichment['funders'])}, "
                f"orcids={len(enrichment['author_orcids'])}, "
                f"refs={len(enrichment['references'])}"
            )
        else:
            logger.warning(
                f"  No Crossref data: [{article['key']}] {title[:60]}"
            )

        return article


def run_phase1b(articles: list[dict]) -> list[dict]:
    """Ejecuta la Fase 1b: Enriquecimiento Crossref."""
    logger.info("=== FASE 1b: Enriquecimiento Crossref ===")
    enricher = CrossrefEnricher()

    enriched = 0
    for article in tqdm(articles, desc="Crossref enrichment"):
        enricher.enrich_article(article)
        if article.get("crossref", {}).get("doi"):
            enriched += 1

    logger.info(f"Enriquecidos via Crossref: {enriched}/{len(articles)}")

    # Guardar checkpoint
    save_json(articles, ENRICHED_PATH)
    logger.info(f"Checkpoint guardado: {ENRICHED_PATH}")

    # Estadisticas
    with_funders = sum(1 for a in articles if a.get("funding"))
    with_orcids = sum(1 for a in articles if a.get("author_orcids"))
    refs_with_doi = sum(
        1 for a in articles
        for r in a.get("referenced_works_detailed", [])
        if r.get("doi")
    )
    total_refs = sum(
        len(a.get("referenced_works_detailed", [])) for a in articles
    )

    logger.info(f"  Con funders: {with_funders}")
    logger.info(f"  Con ORCIDs: {with_orcids}")
    logger.info(f"  Referencias detalladas: {total_refs} (con DOI: {refs_with_doi})")

    return articles
