"""Fase 1c: Enriquecimiento via Semantic Scholar API."""
from __future__ import annotations

import logging
from typing import Optional

from rapidfuzz import fuzz
from tqdm import tqdm

from config import (
    S2_BASE_URL, S2_API_KEY,
    S2_RATE_LIMIT, S2_CACHE_DIR, S2_FUZZY_THRESHOLD,
    ENRICHED_PATH,
)
from src.utils import (
    RateLimiter, DiskCache, create_session, safe_request,
    normalize_doi, save_json,
)

logger = logging.getLogger("bibliometrico.semantic_scholar")

# Fields to request from S2 API
# Note: citations.intents and citations.isInfluential were removed from the
# paper endpoint. Citation intents are fetched separately if needed.
S2_PAPER_FIELDS = ",".join([
    "paperId", "externalIds", "title", "abstract",
    "citationCount", "influentialCitationCount",
    "tldr", "embedding",
    "s2FieldsOfStudy",
    "citationStyles",
    "publicationVenue", "isOpenAccess", "openAccessPdf",
])


class SemanticScholarEnricher:
    """Enriquecedor de articulos via Semantic Scholar API."""

    def __init__(self):
        # Backoff mas amplio (1.5s, 3.0s, 6.0s) para respetar el limite estricto de 1 req/s en S2
        self.session = create_session(retries=4, backoff=1.5)
        self.rate_limiter = RateLimiter(S2_RATE_LIMIT)
        self.cache = DiskCache(S2_CACHE_DIR)
        self.base_url = S2_BASE_URL
        self.api_key = S2_API_KEY

        # API key header
        if self.api_key:
            self.session.headers.update({"x-api-key": self.api_key})
            logger.info("Using Semantic Scholar API key (strict 1 req/s cumulative)")

    def batch_prefetch_by_dois(self, dois: list[str]) -> int:
        """Pre-descarga hasta 500 DOIs por peticion via POST /paper/batch (1 sola llamada de 1s)."""
        missing = [
            d for d in dict.fromkeys(dois)
            if d and self.cache.get(f"s2:doi:{d}") is None
        ]
        if not missing:
            return 0

        fetched = 0
        url = f"{self.base_url}/paper/batch"
        params = {"fields": S2_PAPER_FIELDS}
        for i in range(0, len(missing), 400):
            chunk = missing[i:i + 400]
            self.rate_limiter.wait()
            try:
                resp = self.session.post(
                    url, params=params, json={"ids": [f"DOI:{d}" for d in chunk]}, timeout=60
                )
                self.rate_limiter.mark_done()
                if resp.status_code == 200:
                    items = resp.json()
                    if isinstance(items, list):
                        for d, item in zip(chunk, items):
                            if isinstance(item, dict) and item.get("paperId"):
                                self.cache.set(f"s2:doi:{d}", item)
                                fetched += 1
            except Exception as e:
                self.rate_limiter.mark_done()
                logger.warning(f"Batch S2 prefetch warning: {e}")
        if fetched:
            logger.info(f"S2 batch prefetch: {fetched}/{len(missing)} DOIs obtenidos en 1 peticion")
        return fetched

    def search_by_doi(self, doi: str) -> Optional[dict]:
        """Busca un paper por DOI en Semantic Scholar."""
        if not doi:
            return None

        cache_key = f"s2:doi:{doi}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            logger.debug(f"Cache hit: {doi}")
            return cached

        url = f"{self.base_url}/paper/DOI:{doi}"
        params = {"fields": S2_PAPER_FIELDS}

        result = safe_request(
            url, session=self.session,
            rate_limiter=self.rate_limiter,
            params=params,
        )

        if result and "paperId" in result:
            # Fetch citation intents from separate endpoint
            citations = self._fetch_citation_intents(result["paperId"])
            if citations:
                result["citations"] = citations
            self.cache.set(cache_key, result)
            return result
        return None

    def search_by_title(self, title: str) -> Optional[dict]:
        """Busca un paper por titulo con fuzzy matching."""
        if not title:
            return None

        cache_key = f"s2:title:{title[:100]}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        url = f"{self.base_url}/paper/search"
        params = {
            "query": title[:200],
            "limit": 5,
            "fields": S2_PAPER_FIELDS,
        }

        result = safe_request(
            url, session=self.session,
            rate_limiter=self.rate_limiter,
            params=params,
        )

        if not result or "data" not in result:
            return None

        papers = result["data"]
        if not papers:
            return None

        # Fuzzy match
        best_match = None
        best_score = 0
        for paper in papers:
            paper_title = paper.get("title", "")
            if not paper_title:
                continue
            score = fuzz.ratio(title.lower(), paper_title.lower())
            if score > best_score:
                best_score = score
                best_match = paper

        if best_match and best_score >= S2_FUZZY_THRESHOLD:
            logger.info(f"Title match (score={best_score}): {title[:60]}")
            citations = self._fetch_citation_intents(best_match["paperId"])
            if citations:
                best_match["citations"] = citations
            self.cache.set(cache_key, best_match)
            return best_match

        logger.warning(f"No title match (best={best_score}): {title[:60]}")
        return None

    def _fetch_full_paper(self, paper_id: str) -> Optional[dict]:
        """Obtiene datos completos de un paper por su paperId."""
        cache_key = f"s2:paper:{paper_id}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        url = f"{self.base_url}/paper/{paper_id}"
        params = {"fields": S2_PAPER_FIELDS}

        result = safe_request(
            url, session=self.session,
            rate_limiter=self.rate_limiter,
            params=params,
        )

        if result and "paperId" in result:
            # Fetch citation intents from separate endpoint
            citations = self._fetch_citation_intents(paper_id)
            if citations:
                result["citations"] = citations
            self.cache.set(cache_key, result)
            return result
        return None

    def _fetch_citation_intents(
        self, paper_id: str, limit: int = 500,
    ) -> list[dict]:
        """Fetches citation intents from the citations endpoint."""
        cache_key = f"s2:citations:{paper_id}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        url = f"{self.base_url}/paper/{paper_id}/citations"
        params = {
            "fields": "intents,isInfluential",
            "limit": limit,
        }

        result = safe_request(
            url, session=self.session,
            rate_limiter=self.rate_limiter,
            params=params,
        )

        if result and "data" in result:
            # Each item has {"citingPaper": {...}, "intents": [...], "isInfluential": bool}
            citations = []
            for item in result["data"]:
                citations.append({
                    "intents": item.get("intents", []),
                    "isInfluential": item.get("isInfluential", False),
                })
            self.cache.set(cache_key, citations)
            return citations
        return []

    def extract_work_data(self, paper: dict) -> dict:
        """Extrae datos relevantes de un resultado Semantic Scholar."""
        data = {
            "paper_id": paper.get("paperId", ""),
            "citation_count": paper.get("citationCount", 0),
            "influential_citation_count": paper.get(
                "influentialCitationCount", 0
            ),
        }

        # TLDR
        tldr_obj = paper.get("tldr")
        data["tldr"] = tldr_obj.get("text", "") if tldr_obj else ""

        # Embedding (SPECTER2)
        embedding_obj = paper.get("embedding")
        if embedding_obj and "vector" in embedding_obj:
            data["embedding"] = embedding_obj["vector"]
            data["embedding_model"] = embedding_obj.get("model", "")
        else:
            data["embedding"] = None
            data["embedding_model"] = ""

        # S2 Fields of Study
        s2_fields = []
        for field in paper.get("s2FieldsOfStudy", []) or []:
            s2_fields.append({
                "category": field.get("category", ""),
                "source": field.get("source", ""),
            })
        data["s2_fields_of_study"] = s2_fields

        # Citation intents (aggregate from citations)
        intents = {"background": 0, "methodology": 0, "result": 0}
        citations = paper.get("citations", []) or []
        influential_count = 0
        for cit in citations:
            if not cit:
                continue
            for intent in cit.get("intents", []) or []:
                if intent in intents:
                    intents[intent] += 1
            if cit.get("isInfluential"):
                influential_count += 1
        data["citation_intents"] = intents
        data["citations_influential_incoming"] = influential_count

        # External IDs (MAG, DBLP, ACL, PubMed, ArXiv, etc.)
        ext_ids = paper.get("externalIds") or {}
        data["external_ids"] = {
            "doi": ext_ids.get("DOI", ""),
            "mag": ext_ids.get("MAG", ""),
            "pmid": ext_ids.get("PubMed", ""),
            "pmc": ext_ids.get("PubMedCentral", ""),
            "dblp": ext_ids.get("DBLP", ""),
            "acl": ext_ids.get("ACL", ""),
            "arxiv": ext_ids.get("ArXiv", ""),
            "corpus_id": ext_ids.get("CorpusId", ""),
        }

        # Publication venue
        venue = paper.get("publicationVenue") or {}
        if venue:
            data["publication_venue"] = {
                "id": venue.get("id", ""),
                "name": venue.get("name", ""),
                "type": venue.get("type", ""),
                "alternate_names": venue.get("alternate_names", []),
                "issn": venue.get("issn", ""),
                "url": venue.get("url", ""),
            }

        # Open access
        data["is_open_access"] = paper.get("isOpenAccess", False)
        oa_pdf = paper.get("openAccessPdf") or {}
        data["open_access_pdf_url"] = oa_pdf.get("url", "")

        # Abstract (S2 provides plain text)
        data["abstract"] = paper.get("abstract", "")

        return data

    def enrich_article(self, article: dict) -> dict:
        """Enriquece un articulo individual con datos de Semantic Scholar."""
        doi = article.get("doi", "")
        title = article.get("title", "")

        # Buscar por DOI primero, luego por titulo
        paper = self.search_by_doi(doi)
        if not paper:
            logger.info(f"DOI not found in S2, trying title: {title[:60]}")
            paper = self.search_by_title(title)

        if paper:
            enrichment = self.extract_work_data(paper)
            article["semantic_scholar"] = enrichment

            # Merge cited_by_count (max con existente)
            article["cited_by_count"] = max(
                article.get("cited_by_count", 0),
                enrichment["citation_count"],
            )

            # Merge TLDR
            if enrichment["tldr"]:
                article["tldr"] = enrichment["tldr"]

            # Merge influential citation count
            article["influential_citation_count"] = enrichment[
                "influential_citation_count"
            ]

            # Merge citation intents
            if any(v > 0 for v in enrichment["citation_intents"].values()):
                article["citation_intents"] = enrichment["citation_intents"]

            # Merge S2 fields of study
            if enrichment["s2_fields_of_study"]:
                article["s2_fields_of_study"] = enrichment[
                    "s2_fields_of_study"
                ]

            # Merge embedding
            if enrichment["embedding"]:
                article["specter_embedding"] = enrichment["embedding"]

            # Merge fields of study into concepts with source tracking
            existing_names = {
                c.get("name", "").lower() for c in article.get("concepts", [])
            }
            for field in enrichment.get("s2_fields_of_study", []):
                cat = field.get("category", "")
                if cat and cat.lower() not in existing_names:
                    article.setdefault("concepts", []).append({
                        "name": cat,
                        "level": 0,
                        "score": 0,
                        "source": "semantic_scholar",
                    })
                    existing_names.add(cat.lower())

            # Merge external IDs (DOI, PMID, PMC, etc.)
            ext = enrichment.get("external_ids", {}) or {}
            if ext.get("DOI") and (not article.get("doi") or article["doi"] in ("", "nan")):
                article["doi"] = normalize_doi(str(ext["DOI"]))
            if ext.get("pmid") and (not article.get("pmid") or article["pmid"] in ("", "nan")):
                article["pmid"] = str(ext["pmid"])
            if ext.get("pmc") and (not article.get("pmc_id") or article["pmc_id"] in ("", "nan")):
                article["pmc_id"] = str(ext["pmc"])

            # Venue / Journal gap-fill
            venue = enrichment.get("publication_venue") or {}
            if isinstance(venue, dict):
                if venue.get("name") and (not article.get("publication_title") or article["publication_title"] in ("", "nan")):
                    article["publication_title"] = str(venue["name"]).strip()
                if venue.get("issn") and (not article.get("issn") or article["issn"] in ("", "nan")):
                    article["issn"] = str(venue["issn"]).strip()

            # Abstract gap-fill
            if enrichment.get("abstract") and (not article.get("abstract") or article["abstract"] in ("", "nan")):
                article["abstract"] = enrichment["abstract"]

            # OA PDF URL gap-fill (guard against article["open_access"] being None)
            oa_dict = article.get("open_access")
            if not isinstance(oa_dict, dict):
                oa_dict = {}
                article["open_access"] = oa_dict
            if enrichment.get("open_access_pdf_url") and not oa_dict.get("oa_url"):
                oa_dict["oa_url"] = enrichment["open_access_pdf_url"]

            logger.info(
                f"  S2 [{article['key']}]: "
                f"tldr={'yes' if enrichment['tldr'] else 'no'}, "
                f"embed={'yes' if enrichment['embedding'] else 'no'}, "
                f"intents={sum(enrichment['citation_intents'].values())}, "
                f"venue={'yes' if enrichment.get('publication_venue') else 'no'}"
            )
        else:
            logger.warning(
                f"  No S2 data: [{article['key']}] {title[:60]}"
            )

        return article


def run_phase1c(articles: list[dict]) -> list[dict]:
    """Ejecuta la Fase 1c: Enriquecimiento Semantic Scholar."""
    logger.info("=== FASE 1c: Enriquecimiento Semantic Scholar ===")
    if any(not a.get("doi") for a in articles):
        try:
            from src.metadata_completer import MetadataCompleter
            MetadataCompleter().stage_a(articles)
        except Exception:
            pass
    enricher = SemanticScholarEnricher()

    # Pre-descargar todos los DOIs conocidos en 1 sola llamada batch (POST /paper/batch)
    all_dois = [a.get("doi", "") for a in articles if a.get("doi")]
    if all_dois:
        enricher.batch_prefetch_by_dois(all_dois)

    enriched = 0
    for article in tqdm(articles, desc="Semantic Scholar enrichment"):
        enricher.enrich_article(article)
        if article.get("semantic_scholar", {}).get("paper_id"):
            enriched += 1

    logger.info(f"Enriquecidos via S2: {enriched}/{len(articles)}")

    # Guardar checkpoint
    save_json(articles, ENRICHED_PATH)
    logger.info(f"Checkpoint guardado: {ENRICHED_PATH}")

    # Estadisticas
    with_tldr = sum(1 for a in articles if a.get("tldr"))
    with_embed = sum(1 for a in articles if a.get("specter_embedding"))
    with_intents = sum(
        1 for a in articles
        if any(v > 0 for v in a.get("citation_intents", {}).values())
    )

    logger.info(f"  Con TLDR: {with_tldr}")
    logger.info(f"  Con embeddings: {with_embed}")
    logger.info(f"  Con citation intents: {with_intents}")

    return articles
