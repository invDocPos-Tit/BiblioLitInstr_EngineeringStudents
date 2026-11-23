"""Fase 1k: Metadata Gap-Filler (multi-fuente).

Consulta multiples fuentes para llenar campos vacios en articulos
ya enriquecidos por OpenAlex, Crossref, Semantic Scholar, etc.

Fuentes:
- Europe PMC: abstract, volume, issue, pages, language (ciencias de la salud)
- DOAJ: idioma a nivel de revista (OA journals)
- NLM Catalog (PubMed): abreviatura ISO de revistas biomedicas
- ISSN Portal: abreviatura ISO 4 oficial por ISSN
- langdetect: deteccion automatica de idioma por NLP sobre abstract/titulo

Estrategia:
- Solo procesa articulos con campos vacios despues de Phase 1j
- Nunca sobreescribe datos existentes
- Cache en data/gap_filler_cache/ (DiskCache)
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from rapidfuzz import fuzz

from config import (
    DOAJ_BASE_URL,
    DOAJ_RATE_LIMIT,
    ENRICHED_PATH,
    EUROPE_PMC_BASE_URL,
    EUROPE_PMC_RATE_LIMIT,
    GAP_FILLER_CACHE_DIR,
    GAP_FILLER_FUZZY_THRESHOLD,
    ISSN_PORTAL_BASE_URL,
    ISSN_RATE_LIMIT,
    NLM_CATALOG_BASE_URL,
    NLM_RATE_LIMIT,
)
from src.utils import (
    DiskCache, RateLimiter, create_session, save_json,
    normalize_language, canonicalize_corpus_journals,
)

logger = logging.getLogger("bibliometrico.metadata_gap_filler")

# Campos que intentamos llenar
GAP_FIELDS = ["abstract", "volume", "issue", "pages", "language", "journal_abbrev",
              "issn", "url"]


def _is_empty(val) -> bool:
    """True if value is effectively empty (consistent with metadata_completer)."""
    if val is None:
        return True
    s = str(val).strip()
    return s in ("", "nan", "None", "NaN", "none")


class MetadataGapFiller:
    """Gap-filler multi-fuente: Europe PMC, DOAJ, NLM Catalog, ISSN, langdetect."""

    def __init__(self):
        self.cache = DiskCache(GAP_FILLER_CACHE_DIR)
        self.session = create_session()

        # Rate limiters
        self._epmc_limiter = RateLimiter(EUROPE_PMC_RATE_LIMIT)
        self._doaj_limiter = RateLimiter(DOAJ_RATE_LIMIT)
        self._nlm_limiter = RateLimiter(NLM_RATE_LIMIT)
        self._issn_limiter = RateLimiter(ISSN_RATE_LIMIT)

        # Language detector (lazy)
        self._langdetect = None
        self._langdetect_available = False
        try:
            import langdetect as _ld
            self._langdetect = _ld
            self._langdetect_available = True
        except ImportError:
            try:
                import langid as _lid
                self._langdetect = _lid
                self._langdetect_available = True
            except ImportError:
                pass

        logger.info(
            "Gap-filler backends: Europe PMC, DOAJ, NLM Catalog, ISSN Portal%s",
            ", langdetect" if self._langdetect_available else "",
        )

    # ------------------------------------------------------------------
    # Gap detection
    # ------------------------------------------------------------------

    def _has_gaps(self, article: dict) -> bool:
        """Check if article has empty fields that gap-filler could fill."""
        for field in GAP_FIELDS:
            if _is_empty(article.get(field)):
                return True
        return False

    def _get_gap_fields(self, article: dict) -> list[str]:
        """Return list of field names that are empty/missing."""
        gaps = []
        for field in GAP_FIELDS:
            if _is_empty(article.get(field)):
                gaps.append(field)
        return gaps

    # ------------------------------------------------------------------
    # Europe PMC
    # ------------------------------------------------------------------

    def _search_europe_pmc(self, article: dict) -> Optional[dict]:
        """Search Europe PMC by DOI or title for metadata."""
        doi = article.get("doi", "")
        title = article.get("title", "")

        # Try DOI first
        if doi:
            cache_key = f"epmc:doi:{doi}"
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

            self._epmc_limiter.wait()
            try:
                resp = self.session.get(
                    f"{EUROPE_PMC_BASE_URL}/search",
                    params={
                        "query": f'DOI:"{doi}"',
                        "format": "json",
                        "resultType": "core",
                        "pageSize": 1,
                    },
                    timeout=20,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    results = data.get("resultList", {}).get("result", [])
                    if results:
                        self.cache.set(cache_key, results[0])
                        return results[0]
            except Exception as e:
                logger.debug("Europe PMC DOI search failed: %s", e)

        # Fallback to title search
        if title:
            cache_key = f"epmc:title:{title[:100]}"
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

            self._epmc_limiter.wait()
            try:
                # Escape special chars for Europe PMC query
                clean_title = re.sub(r'[^\w\s]', ' ', title)
                resp = self.session.get(
                    f"{EUROPE_PMC_BASE_URL}/search",
                    params={
                        "query": f'TITLE:"{clean_title[:200]}"',
                        "format": "json",
                        "resultType": "core",
                        "pageSize": 3,
                    },
                    timeout=20,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    results = data.get("resultList", {}).get("result", [])
                    # Fuzzy match title
                    for result in results:
                        r_title = result.get("title", "")
                        if fuzz.ratio(
                            title.lower(), r_title.lower()
                        ) >= GAP_FILLER_FUZZY_THRESHOLD:
                            self.cache.set(cache_key, result)
                            return result
            except Exception as e:
                logger.debug("Europe PMC title search failed: %s", e)

        return None

    def _extract_epmc_data(self, epmc: dict) -> dict:
        """Extract usable fields from Europe PMC result."""
        data: Dict[str, str] = {}

        abstract = epmc.get("abstractText", "")
        if abstract and len(abstract) > 50:
            data["abstract"] = abstract

        if epmc.get("language"):
            data["language"] = epmc["language"]

        journal_info = epmc.get("journalInfo", {})
        if isinstance(journal_info, dict):
            journal = journal_info.get("journal", {})
            if isinstance(journal, dict):
                if journal.get("medlineAbbreviation"):
                    data["journal_abbrev"] = journal["medlineAbbreviation"]
                if journal.get("isoabbreviation"):
                    data["journal_abbrev"] = journal["isoabbreviation"]
                # ISSN from journal
                if journal.get("issn"):
                    data["issn"] = str(journal["issn"])
                elif journal.get("essn"):
                    data["issn"] = str(journal["essn"])

            if journal_info.get("volume"):
                data["volume"] = str(journal_info["volume"])
            if journal_info.get("issue"):
                data["issue"] = str(journal_info["issue"])

        if epmc.get("pageInfo"):
            data["pages"] = str(epmc["pageInfo"])

        # DOI -> URL construction
        doi = epmc.get("doi", "")
        if doi and not _is_empty(doi):
            data["url"] = f"https://doi.org/{doi}"

        return data

    # ------------------------------------------------------------------
    # DOAJ (Directory of Open Access Journals)
    # ------------------------------------------------------------------

    def _search_doaj_journal(self, issn: str) -> Optional[dict]:
        """Search DOAJ for journal-level metadata by ISSN."""
        if not issn or len(issn) < 4:
            return None

        cache_key = f"doaj:issn:{issn}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        self._doaj_limiter.wait()

        try:
            resp = self.session.get(
                f"{DOAJ_BASE_URL}/search/journals/issn:{issn}",
                params={"pageSize": 1},
                timeout=20,
            )
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", [])
                if results:
                    self.cache.set(cache_key, results[0])
                    return results[0]
        except Exception as e:
            logger.debug("DOAJ search failed for ISSN %s: %s", issn, e)

        return None

    def _extract_doaj_data(self, doaj: dict) -> dict:
        """Extract language and ISSN info from DOAJ journal entry."""
        data: Dict[str, str] = {}

        bibjson = doaj.get("bibjson", {})
        if isinstance(bibjson, dict):
            # Language at journal level
            lang = bibjson.get("language", [])
            if isinstance(lang, list) and lang:
                data["language"] = lang[0]  # First language
            elif isinstance(lang, str) and lang:
                data["language"] = lang

            # ISSN
            identifiers = bibjson.get("identifier", [])
            if isinstance(identifiers, list):
                for ident in identifiers:
                    if isinstance(ident, dict) and ident.get("type") == "pissn":
                        data["issn"] = ident.get("id", "")
                        break

        return data

    # ------------------------------------------------------------------
    # NLM Catalog (PubMed/NCBI) — Journal abbreviations
    # ------------------------------------------------------------------

    def _search_nlm_journal(self, journal_name: str) -> Optional[str]:
        """Search NLM Catalog for journal ISO abbreviation."""
        if not journal_name or len(journal_name) < 3:
            return None

        cache_key = f"nlm:journal:{journal_name[:80]}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        self._nlm_limiter.wait()

        try:
            # Step 1: Search for journal in NLM Catalog
            resp = self.session.get(
                f"{NLM_CATALOG_BASE_URL}/esearch.fcgi",
                params={
                    "db": "nlmcatalog",
                    "term": f'"{journal_name}"[Title]',
                    "retmax": 3,
                    "retmode": "json",
                },
                timeout=20,
            )
            if resp.status_code != 200:
                return None

            search_data = resp.json()
            ids = (
                search_data.get("esearchresult", {}).get("idlist", [])
            )
            if not ids:
                return None

            # Step 2: Fetch details to get abbreviation
            self._nlm_limiter.wait()
            resp2 = self.session.get(
                f"{NLM_CATALOG_BASE_URL}/esummary.fcgi",
                params={
                    "db": "nlmcatalog",
                    "id": ",".join(ids[:3]),
                    "retmode": "json",
                },
                timeout=20,
            )
            if resp2.status_code != 200:
                return None

            summary = resp2.json()
            result_data = summary.get("result", {})

            for uid in ids[:3]:
                entry = result_data.get(uid, {})
                if not isinstance(entry, dict):
                    continue
                # medlineta = MedlineTA (NLM abbreviation)
                abbrev = entry.get("medlineta", "")
                if abbrev:
                    self.cache.set(cache_key, abbrev)
                    return abbrev

        except Exception as e:
            logger.debug("NLM Catalog search failed: %s", e)

        return None

    # ------------------------------------------------------------------
    # ISSN Portal — ISO 4 abbreviations
    # ------------------------------------------------------------------

    def _search_issn_portal(self, issn: str) -> Optional[str]:
        """Search ISSN Portal for ISO 4 journal abbreviation."""
        if not issn or len(issn) < 4:
            return None

        cache_key = f"issn:abbrev:{issn}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        self._issn_limiter.wait()

        try:
            resp = self.session.get(
                f"{ISSN_PORTAL_BASE_URL}/{issn}",
                params={"format": "json"},
                timeout=20,
                headers={"Accept": "application/json"},
            )
            if resp.status_code == 200:
                data = resp.json()
                # ISSN portal returns linked data; look for abbreviatedTitle
                # in @graph array
                graph = data.get("@graph", [])
                for item in graph:
                    if not isinstance(item, dict):
                        continue
                    abbrev = item.get("abbreviatedTitle")
                    if abbrev:
                        if isinstance(abbrev, list):
                            abbrev = abbrev[0]
                        self.cache.set(cache_key, str(abbrev))
                        return str(abbrev)
                    # Also check mainTitle for ISSN-L records
                    name = item.get("name")
                    if name and isinstance(name, str):
                        # Some entries have "KeyTitle" or abbreviated name
                        pass
        except Exception as e:
            logger.debug("ISSN Portal search failed for %s: %s", issn, e)

        return None

    # ------------------------------------------------------------------
    # Language detection via NLP
    # ------------------------------------------------------------------

    def _detect_language_nlp(self, text: str) -> Optional[str]:
        """Detect language from text using langdetect or langid."""
        if not self._langdetect_available or not text or len(text) < 30:
            return None

        try:
            if hasattr(self._langdetect, "detect"):
                # langdetect library
                lang = self._langdetect.detect(text)
                return lang
            elif hasattr(self._langdetect, "classify"):
                # langid library
                lang, confidence = self._langdetect.classify(text)
                if confidence > 0.5:
                    return lang
        except Exception:
            pass

        return None

    # ------------------------------------------------------------------
    # Gap filling orchestration
    # ------------------------------------------------------------------

    def _fill_gaps(self, article: dict, source_data: dict, source: str) -> dict:
        """Fill empty fields. Returns dict of filled fields."""
        filled: Dict[str, str] = {}

        for field in GAP_FIELDS:
            if not _is_empty(article.get(field)):
                continue  # Field already has data

            value = source_data.get(field)
            if value and not _is_empty(value):
                cleaned_val = str(value).strip()
                if field == "language":
                    cleaned_val = normalize_language(cleaned_val)
                if cleaned_val:
                    article[field] = cleaned_val
                    filled[field] = cleaned_val

        if filled:
            if "enrichment_sources" not in article:
                article["enrichment_sources"] = []
            if source not in article["enrichment_sources"]:
                article["enrichment_sources"].append(source)

        return filled

    def enrich_article(self, article: dict) -> dict:
        """Enrich a single article from multiple gap-filling sources.

        Returns dict of field_name -> new_value for fields that were filled.
        """
        all_filled: Dict[str, str] = {}
        remaining_gaps = set(self._get_gap_fields(article))

        if not remaining_gaps:
            return {}

        # --- Source 1: Europe PMC (abstract, volume, issue, pages, language, issn, url) ---
        metadata_gaps = remaining_gaps & {"abstract", "volume", "issue", "pages", "language", "journal_abbrev", "issn", "url"}
        if metadata_gaps:
            epmc_result = self._search_europe_pmc(article)
            if epmc_result:
                epmc_data = self._extract_epmc_data(epmc_result)
                filled = self._fill_gaps(article, epmc_data, "europe_pmc")
                all_filled.update(filled)
                remaining_gaps -= set(filled.keys())

        # --- Source 2: DOAJ (language + issn at journal level) ---
        if remaining_gaps & {"language", "issn"}:
            issn = article.get("issn", "")
            if issn and not _is_empty(issn):
                # Normalize ISSN
                if isinstance(issn, str):
                    issns = [i.strip() for i in issn.split(",")
                             if i.strip() and not _is_empty(i)]
                else:
                    issns = [str(issn)]
                for single_issn in issns[:2]:
                    doaj_result = self._search_doaj_journal(single_issn)
                    if doaj_result:
                        doaj_data = self._extract_doaj_data(doaj_result)
                        filled = self._fill_gaps(article, doaj_data, "doaj")
                        all_filled.update(filled)
                        remaining_gaps -= set(filled.keys())
                        break

        # --- Source 3: NLM Catalog (journal abbreviation) ---
        if "journal_abbrev" in remaining_gaps:
            journal_name = article.get("publication_title", "")
            if journal_name:
                abbrev = self._search_nlm_journal(journal_name)
                if abbrev:
                    filled = self._fill_gaps(
                        article, {"journal_abbrev": abbrev}, "nlm_catalog"
                    )
                    all_filled.update(filled)
                    remaining_gaps -= set(filled.keys())

        # --- Source 4: ISSN Portal (journal ISO 4 abbreviation) ---
        if "journal_abbrev" in remaining_gaps:
            issn = article.get("issn", "")
            if issn and not _is_empty(issn):
                issns = [i.strip() for i in str(issn).split(",")
                         if i.strip() and not _is_empty(i)]
                for single_issn in issns[:2]:
                    abbrev = self._search_issn_portal(single_issn)
                    if abbrev:
                        filled = self._fill_gaps(
                            article, {"journal_abbrev": abbrev}, "issn_portal"
                        )
                        all_filled.update(filled)
                        remaining_gaps -= set(filled.keys())
                        break

        # --- Source 5: langdetect NLP (language from abstract/title) ---
        if "language" in remaining_gaps:
            text = article.get("abstract", "")
            if _is_empty(text):
                text = article.get("title", "")
            if text and not _is_empty(text):
                lang = self._detect_language_nlp(text)
                if lang:
                    filled = self._fill_gaps(
                        article, {"language": lang}, "langdetect"
                    )
                    all_filled.update(filled)
                    remaining_gaps -= set(filled.keys())

        # --- Source 6: URL from DOI (simple construction) ---
        if "url" in remaining_gaps:
            doi = article.get("doi", "")
            if doi and not _is_empty(doi):
                url = f"https://doi.org/{doi}"
                filled = self._fill_gaps(
                    article, {"url": url}, "doi_url"
                )
                all_filled.update(filled)
                remaining_gaps -= set(filled.keys())

        if all_filled:
            logger.debug(
                "Gap-filler enriched [%s]: %s",
                article.get("key", "?"),
                ", ".join(
                    f"{k}={v[:30]}..." if len(v) > 30 else f"{k}={v}"
                    for k, v in all_filled.items()
                ),
            )

        return all_filled


def run_phase1k(articles: list[dict]) -> list[dict]:
    """Entry point for Phase 1k: Metadata Gap-Filler.

    Uses Europe PMC, DOAJ, NLM Catalog, ISSN Portal, and langdetect
    to fill empty fields left by previous enrichment phases.
    Never overwrites existing data.
    """
    logger.info("=" * 60)
    logger.info("Fase 1k: Metadata Gap-Filler (Europe PMC, DOAJ, NLM, langdetect)")
    logger.info("=" * 60)

    # Normalize existing language codes and journal names across the corpus
    for a in articles:
        if not _is_empty(a.get("language")):
            a["language"] = normalize_language(str(a["language"]))
    canonicalize_corpus_journals(articles)

    filler = MetadataGapFiller()

    # Filter articles with gaps
    articles_with_gaps = [
        (i, a) for i, a in enumerate(articles) if filler._has_gaps(a)
    ]

    if not articles_with_gaps:
        logger.info("No articles with metadata gaps found. Nothing to enrich.")
        return articles

    logger.info(
        "Articles with gaps: %d / %d", len(articles_with_gaps), len(articles)
    )

    # Pre-scan gap distribution
    gap_dist: Dict[str, int] = {f: 0 for f in GAP_FIELDS}
    for _, a in articles_with_gaps:
        for field in filler._get_gap_fields(a):
            gap_dist[field] += 1
    logger.info("Gap distribution before enrichment:")
    for field, count in gap_dist.items():
        if count > 0:
            logger.info("  %s: %d missing", field, count)

    # Stats
    total_filled: Dict[str, int] = {f: 0 for f in GAP_FIELDS}
    source_stats: Dict[str, int] = {}
    n_enriched = 0
    n_errors = 0

    for idx, (i, article) in enumerate(articles_with_gaps):
        try:
            filled = filler.enrich_article(article)
            if filled:
                n_enriched += 1
                for field in filled:
                    total_filled[field] = total_filled.get(field, 0) + 1
                # Track source
                for src in article.get("enrichment_sources", []):
                    if src in ("europe_pmc", "doaj", "nlm_catalog",
                               "issn_portal", "langdetect", "doi_url"):
                        source_stats[src] = source_stats.get(src, 0) + 1
        except Exception as e:
            n_errors += 1
            logger.warning(
                "Error enriching %s: %s", article.get("key", f"idx_{i}"), e
            )

        # Progress log every 25 articles
        if (idx + 1) % 25 == 0 or idx == len(articles_with_gaps) - 1:
            logger.info(
                "Progress: %d/%d processed (%d enriched)",
                idx + 1,
                len(articles_with_gaps),
                n_enriched,
            )

    # Summary
    logger.info("-" * 40)
    logger.info("Metadata gap-filler summary:")
    logger.info("  Articles processed: %d", len(articles_with_gaps))
    logger.info("  Articles enriched: %d", n_enriched)
    logger.info("  Errors: %d", n_errors)
    logger.info("  Fields filled:")
    for field, count in total_filled.items():
        if count > 0:
            logger.info("    %s: %d", field, count)
    logger.info("  By source:")
    for src, count in sorted(source_stats.items()):
        logger.info("    %s: %d articles", src, count)
    logger.info("-" * 40)

    # Save checkpoint
    save_json(articles, ENRICHED_PATH)
    logger.info("Checkpoint saved after Phase 1k")

    return articles
