"""Fase 1: Enriquecimiento via API OpenAlex."""
from __future__ import annotations

import logging
from typing import Optional

from rapidfuzz import fuzz
from tqdm import tqdm

from config import (
    OPENALEX_BASE_URL, OPENALEX_EMAIL, OPENALEX_API_KEY,
    OPENALEX_RATE_LIMIT, OPENALEX_CACHE_DIR, OPENALEX_FUZZY_THRESHOLD,
    ENRICHED_PATH,
)
from src.utils import (
    RateLimiter, DiskCache, create_session, safe_request,
    normalize_doi, save_json, normalize_language,
)

logger = logging.getLogger("bibliometrico.openalex")


class OpenAlexEnricher:
    """Enriquecedor de articulos via OpenAlex API."""

    def __init__(self, rate_override: float | None = None):
        self.session = create_session()
        # With API key: higher rate limit (premium pool)
        rate = OPENALEX_RATE_LIMIT * 5 if OPENALEX_API_KEY else OPENALEX_RATE_LIMIT
        if rate_override and rate_override > 0:
            rate = rate_override
        self.rate_limiter = RateLimiter(min(rate, 50))
        self.cache = DiskCache(OPENALEX_CACHE_DIR)
        self.base_url = OPENALEX_BASE_URL
        self.email = OPENALEX_EMAIL
        self.api_key = OPENALEX_API_KEY
        if self.api_key:
            logger.info("Using OpenAlex API key (premium pool)")

    def _build_params(self, extra_params: dict = None) -> dict:
        params = {}
        if self.api_key:
            params["api_key"] = self.api_key
        elif self.email:
            params["mailto"] = self.email
        if extra_params:
            params.update(extra_params)
        return params

    def search_by_doi(self, doi: str) -> Optional[dict]:
        """Busca un trabajo por DOI en OpenAlex."""
        if not doi:
            return None

        cache_key = f"doi:{doi}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            logger.debug(f"Cache hit: {doi}")
            return cached

        url = f"{self.base_url}/works/https://doi.org/{doi}"
        result = safe_request(
            url, session=self.session,
            rate_limiter=self.rate_limiter,
            params=self._build_params(),
        )

        if result and "id" in result:
            self.cache.set(cache_key, result)
            return result
        return None

    def search_by_title(self, title: str) -> Optional[dict]:
        """Busca un trabajo por titulo con fuzzy matching."""
        if not title:
            return None

        cache_key = f"title:{title[:100]}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        url = f"{self.base_url}/works"
        params = self._build_params({
            "filter": f"title.search:{title[:200]}",
            "per_page": 5,
        })

        result = safe_request(
            url, session=self.session,
            rate_limiter=self.rate_limiter,
            params=params,
        )

        if not result or "results" not in result:
            return None

        # Fuzzy match
        best_match = None
        best_score = 0
        for work in result["results"]:
            work_title = work.get("title", "")
            if not work_title:
                continue
            score = fuzz.ratio(title.lower(), work_title.lower())
            if score > best_score:
                best_score = score
                best_match = work

        if best_match and best_score >= OPENALEX_FUZZY_THRESHOLD:
            self.cache.set(cache_key, best_match)
            logger.info(f"Title match (score={best_score}): {title[:60]}")
            return best_match

        logger.warning(f"No title match (best={best_score}): {title[:60]}")
        return None

    def extract_work_data(self, work: dict) -> dict:
        """Extrae datos relevantes de un resultado OpenAlex."""
        primary_loc = work.get("primary_location") or {}
        source_obj = primary_loc.get("source") or {}
        raw_doi = work.get("doi") or ""
        clean_doi = normalize_doi(str(raw_doi)) if raw_doi else ""
        data = {
            "openalex_id": work.get("id", ""),
            "doi": clean_doi,
            "cited_by_count": work.get("cited_by_count", 0),
            "referenced_works": work.get("referenced_works", []),
            "counts_by_year": work.get("counts_by_year", []),
            "type": work.get("type", ""),
            "is_oa": (work.get("open_access") or {}).get("is_oa", False),
            "oa_status": (work.get("open_access") or {}).get("oa_status", ""),
            "oa_url": (work.get("open_access") or {}).get("oa_url", ""),
            "source_id": source_obj.get("id", ""),
            "source_display_name": source_obj.get("display_name", ""),
            "issn_l": source_obj.get("issn_l", "") or "",
            "issn": source_obj.get("issn") or [],
            "publisher": source_obj.get("host_organization_name", "") or "",
            "journal_is_in_doaj": bool(source_obj.get("is_in_doaj", False)),
        }

        # Concepts/topics
        concepts = []
        for concept in work.get("concepts", []):
            concepts.append({
                "name": concept.get("display_name", ""),
                "level": concept.get("level", 0),
                "score": concept.get("score", 0),
            })
        data["concepts"] = concepts

        # Topics (CWTS Leiden hierarchy)
        topics = []
        for topic in work.get("topics", []):
            topics.append({
                "name": topic.get("display_name", ""),
                "display_name": topic.get("display_name", ""),
                "score": topic.get("score", 1.0),
                "subfield": topic.get("subfield", {}).get("display_name", ""),
                "field": topic.get("field", {}).get("display_name", ""),
                "domain": topic.get("domain", {}).get("display_name", ""),
            })
        data["topics"] = topics

        # Authorships con afiliaciones
        authorships = []
        for auth in work.get("authorships", []):
            author_info = {
                "name": auth.get("author", {}).get("display_name", ""),
                "openalex_id": auth.get("author", {}).get("id", ""),
                "orcid": auth.get("author", {}).get("orcid", "") or "",
                "position": auth.get("author_position", ""),
                "institutions": [],
                "countries": [],
            }
            for inst in auth.get("institutions", []):
                author_info["institutions"].append({
                    "name": inst.get("display_name", ""),
                    "country": inst.get("country_code", ""),
                    "type": inst.get("type", ""),
                })
                cc = inst.get("country_code", "")
                if cc and cc not in author_info["countries"]:
                    author_info["countries"].append(cc)
            authorships.append(author_info)
        data["authorships"] = authorships

        # --- Grants / Funding ---
        grants = []
        for grant in work.get("grants", []):
            grants.append({
                "funder_display_name": grant.get("funder_display_name", ""),
                "funder": grant.get("funder", ""),
                "award_id": grant.get("award_id", ""),
            })
        data["grants"] = grants

        # --- Sustainable Development Goals ---
        sdgs = []
        for sdg in work.get("sustainable_development_goals", []):
            sdgs.append({
                "id": sdg.get("id", ""),
                "display_name": sdg.get("display_name", ""),
                "score": sdg.get("score", 0),
            })
        data["sustainable_development_goals"] = sdgs

        # --- Keywords (author-provided) ---
        keywords = []
        for kw in work.get("keywords", []):
            keywords.append({
                "keyword": kw.get("keyword", "") or kw.get("display_name", ""),
                "score": kw.get("score", 0),
            })
        data["keywords"] = keywords

        # --- Biblio (volume, issue, pages) ---
        biblio = work.get("biblio", {}) or {}
        data["biblio"] = {
            "volume": biblio.get("volume", ""),
            "issue": biblio.get("issue", ""),
            "first_page": biblio.get("first_page", ""),
            "last_page": biblio.get("last_page", ""),
        }

        # --- Primary topic ---
        pt = work.get("primary_topic") or {}
        if pt:
            data["primary_topic"] = {
                "name": pt.get("display_name", ""),
                "subfield": pt.get("subfield", {}).get("display_name", ""),
                "field": pt.get("field", {}).get("display_name", ""),
                "domain": pt.get("domain", {}).get("display_name", ""),
            }

        # --- Citation percentile ---
        data["cited_by_percentile_year"] = work.get("cited_by_percentile_year", {})

        # --- Language ---
        data["language"] = work.get("language", "")

        # --- APC ---
        apc_list = work.get("apc_list") or {}
        if apc_list:
            data["apc_list"] = {
                "value": apc_list.get("value", 0),
                "currency": apc_list.get("currency", ""),
                "value_usd": apc_list.get("value_usd", 0),
            }
        apc_paid = work.get("apc_paid") or {}
        if apc_paid:
            data["apc_paid"] = {
                "value": apc_paid.get("value", 0),
                "currency": apc_paid.get("currency", ""),
                "value_usd": apc_paid.get("value_usd", 0),
            }

        # --- Corresponding author info ---
        data["corresponding_author_ids"] = work.get("corresponding_author_ids", [])
        data["corresponding_institution_ids"] = work.get("corresponding_institution_ids", [])

        return data

    def enrich_article(self, article: dict) -> dict:
        """Enriquece un articulo individual."""
        doi = article.get("doi", "")
        title = article.get("title", "")

        # Buscar por DOI primero, luego por titulo
        work = self.search_by_doi(doi)
        if not work:
            logger.info(f"DOI not found, trying title: {title[:60]}")
            work = self.search_by_title(title)

        if work:
            enrichment = self.extract_work_data(work)
            article["openalex"] = enrichment
            if enrichment.get("doi") and (not article.get("doi") or article["doi"] in ("", "nan")):
                article["doi"] = enrichment["doi"]
                if not article.get("url") or article["url"] in ("", "nan"):
                    article["url"] = f"https://doi.org/{enrichment['doi']}"
            if enrichment.get("source_display_name") and (not article.get("publication_title") or article["publication_title"] in ("", "nan")):
                article["publication_title"] = enrichment["source_display_name"]
            if not article.get("issn") or article["issn"] in ("", "nan"):
                cand_issn = enrichment.get("issn_l") or (enrichment.get("issn")[0] if enrichment.get("issn") else "")
                if cand_issn:
                    article["issn"] = str(cand_issn).strip()
            if enrichment.get("publisher") and (not article.get("publisher") or article["publisher"] in ("", "nan")):
                article["publisher"] = enrichment["publisher"]
            if enrichment.get("type") and (not article.get("item_type") or article["item_type"] in ("", "nan")):
                article["item_type"] = enrichment["type"]
            article["cited_by_count"] = max(
                article.get("cited_by_count", 0),
                enrichment["cited_by_count"],
            )
            if enrichment["referenced_works"]:
                article["referenced_works"] = enrichment["referenced_works"]
            article["concepts"] = enrichment["concepts"]
            article["topics"] = enrichment["topics"]
            article["open_access"] = {
                "is_oa": enrichment["is_oa"],
                "status": enrichment["oa_status"],
                "oa_url": enrichment.get("oa_url", ""),
                "journal_is_in_doaj": enrichment.get("journal_is_in_doaj", False),
            }

            # Extraer afiliaciones
            affiliations = []
            for auth in enrichment.get("authorships", []):
                for inst in auth.get("institutions", []):
                    affiliations.append(inst["name"])
            article["affiliations"] = list(set(affiliations))

            # Grants → merge into funding[]
            for grant in enrichment.get("grants", []):
                fname = grant.get("funder_display_name", "")
                if fname:
                    existing_names = {f.get("name", "").lower() for f in article.get("funding", [])}
                    if fname.lower() not in existing_names:
                        article.setdefault("funding", []).append({
                            "name": fname,
                            "award_id": grant.get("award_id", ""),
                            "source": "openalex",
                        })

            # SDGs
            if enrichment.get("sustainable_development_goals"):
                article["sustainable_development_goals"] = enrichment["sustainable_development_goals"]

            # Keywords from OpenAlex → merge into all_tags (and manual_tags if CSV had no keywords)
            had_manual_tags = bool(article.get("manual_tags"))
            for kw in enrichment.get("keywords", []):
                kw_text = kw.get("keyword", "")
                if kw_text:
                    if kw_text not in article.get("all_tags", []):
                        article.setdefault("all_tags", []).append(kw_text)
                    if not had_manual_tags and kw_text not in article.get("manual_tags", []):
                        article.setdefault("manual_tags", []).append(kw_text)

            # Biblio gap-fill (volume, issue, pages)
            bib = enrichment.get("biblio", {})
            if bib.get("volume") and (not article.get("volume") or article["volume"] in ("", "nan")):
                article["volume"] = bib["volume"]
            if bib.get("issue") and (not article.get("issue") or article["issue"] in ("", "nan")):
                article["issue"] = bib["issue"]
            if bib.get("first_page") and (not article.get("pages") or article["pages"] in ("", "nan")):
                pages = bib["first_page"]
                if bib.get("last_page"):
                    pages += f"-{bib['last_page']}"
                article["pages"] = pages

            # Language gap-fill (normalized to canonical English name)
            if enrichment.get("language") and (not article.get("language") or article["language"] in ("", "nan")):
                article["language"] = normalize_language(enrichment["language"])

            logger.info(
                f"  Enriched [{article['key']}]: "
                f"cites={enrichment['cited_by_count']}, "
                f"refs={len(enrichment['referenced_works'])}, "
                f"concepts={len(enrichment['concepts'])}, "
                f"grants={len(enrichment.get('grants', []))}, "
                f"sdgs={len(enrichment.get('sustainable_development_goals', []))}"
            )
        else:
            logger.warning(f"  No OpenAlex data: [{article['key']}] {title[:60]}")

        return article


def run_phase1(articles: list[dict], rate_override: float | None = None) -> list[dict]:
    """Ejecuta la Fase 1: Enriquecimiento OpenAlex."""
    logger.info("=== FASE 1: Enriquecimiento OpenAlex ===")
    enricher = OpenAlexEnricher(rate_override=rate_override)

    enriched = 0
    for article in tqdm(articles, desc="OpenAlex enrichment"):
        enricher.enrich_article(article)
        if article.get("openalex", {}).get("openalex_id"):
            enriched += 1

    logger.info(f"Enriquecidos: {enriched}/{len(articles)}")

    # Guardar checkpoint
    save_json(articles, ENRICHED_PATH)
    logger.info(f"Checkpoint guardado: {ENRICHED_PATH}")

    # Estadisticas de referencias
    total_refs = sum(len(a.get("referenced_works", [])) for a in articles)
    unique_refs = len(set(
        ref for a in articles for ref in a.get("referenced_works", [])
    ))
    logger.info(f"Total referencias: {total_refs}, Unicas: {unique_refs}")

    return articles
