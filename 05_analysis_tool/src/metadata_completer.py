"""Fase 1j: Metadata Completion — fill gaps from existing caches + minimal API.

Pipeline:
  Stage A — Cache extraction (0 API calls): OpenAlex + Crossref caches
  Stage B — Language API (~25 calls): OpenAlex /works/{id}?select=language
  Stage C — Journal abbreviation API (~100 calls): OpenAlex /sources/{id}
"""
from __future__ import annotations

import logging
from typing import Optional

from tqdm import tqdm

from config import (
    OPENALEX_BASE_URL, OPENALEX_EMAIL, OPENALEX_API_KEY,
    OPENALEX_RATE_LIMIT, OPENALEX_CACHE_DIR, OPENALEX_META_CACHE_DIR,
    CROSSREF_CACHE_DIR, S2_CACHE_DIR, ENRICHED_PATH,
)
from src.utils import (
    RateLimiter, DiskCache, create_session, safe_request,
    normalize_doi, save_json, normalize_language, canonicalize_corpus_journals,
)

logger = logging.getLogger("bibliometrico.metadata_completer")


def _is_empty(val) -> bool:
    """True if value is effectively empty."""
    if val is None:
        return True
    s = str(val).strip()
    return s in ("", "nan", "None", "NaN", "none")


def _map_language_code(code: str) -> str:
    """ISO 639-1/639-2 code or language name -> canonical English language name."""
    if not code or _is_empty(code):
        return ""
    return normalize_language(str(code))


def _format_pages(biblio: dict, crossref_page: str) -> str:
    """Build pages string from OpenAlex biblio or Crossref page field."""
    # Try OpenAlex biblio first
    fp = biblio.get("first_page", "") if biblio else ""
    lp = biblio.get("last_page", "") if biblio else ""
    if fp:
        fp = str(fp).strip()
        lp = str(lp).strip() if lp else ""
        if lp and lp != fp:
            return f"{fp}-{lp}"
        return fp

    # Fallback: Crossref page
    if crossref_page and not _is_empty(crossref_page):
        return str(crossref_page).strip()

    return ""


def _clean_issue(val) -> str:
    """Clean issue field — remove '.0' pandas artifact."""
    if _is_empty(val):
        return ""
    s = str(val).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def _clean_volume(val) -> str:
    """Clean volume field — remove '.0' pandas artifact."""
    if _is_empty(val):
        return ""
    s = str(val).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


class MetadataCompleter:
    """Fills metadata gaps from existing caches and minimal API calls."""

    def __init__(self, rate_override: float | None = None):
        self.oa_cache = DiskCache(OPENALEX_CACHE_DIR)
        self.oa_meta_cache = DiskCache(OPENALEX_META_CACHE_DIR)
        self.cr_cache = DiskCache(CROSSREF_CACHE_DIR)
        self.s2_cache = DiskCache(S2_CACHE_DIR)

        # API session (for stages B and C)
        self.session = create_session()
        rate = OPENALEX_RATE_LIMIT * 5 if OPENALEX_API_KEY else OPENALEX_RATE_LIMIT
        if rate_override and rate_override > 0:
            rate = rate_override
        self.rate_limiter = RateLimiter(min(rate, 50))
        self.base_url = OPENALEX_BASE_URL
        self.email = OPENALEX_EMAIL
        self.api_key = OPENALEX_API_KEY

        # Stats
        self.stats = {
            "language_cache": 0, "language_api": 0,
            "journal_abbrev_cache": 0, "journal_abbrev_api": 0,
            "pages_filled": 0, "volume_filled": 0, "issue_filled": 0,
            "url_constructed": 0, "issn_filled": 0,
        }

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
    # Cache lookup helpers
    # ------------------------------------------------------------------

    def _get_oa_cached_work(self, article: dict) -> Optional[dict]:
        """Get OpenAlex work data from cache (try work:{id}, doi:{doi}, title:{title[:100]})."""
        # Try work:{id}
        oa = article.get("openalex", {})
        oa_id = oa.get("openalex_id", "") if isinstance(oa, dict) else ""
        if oa_id:
            work_id = oa_id.split("/")[-1] if "/" in oa_id else oa_id
            cached = self.oa_cache.get(f"work:{work_id}")
            if cached and isinstance(cached, dict):
                return cached

        # Try doi:{doi}
        doi = article.get("doi", "") or (article.get("crossref") or {}).get("doi", "")
        if doi:
            cached = self.oa_cache.get(f"doi:{doi}")
            if cached and isinstance(cached, dict):
                return cached

        # Try title:{title[:100]}
        title = article.get("title", "")
        if title:
            cached = self.oa_cache.get(f"title:{title[:100]}")
            if cached and isinstance(cached, dict):
                return cached

        return None

    def _get_cr_cached_work(self, article: dict) -> Optional[dict]:
        """Get Crossref work data from cache (try crossref:doi:{doi} and crossref:title:{title[:100]})."""
        doi = article.get("doi", "") or (article.get("crossref") or {}).get("doi", "")
        if doi:
            cached = self.cr_cache.get(f"crossref:doi:{doi}")
            if cached and isinstance(cached, dict):
                return cached
        title = article.get("title", "")
        if title:
            cached = self.cr_cache.get(f"crossref:title:{title[:100]}")
            if cached and isinstance(cached, dict):
                return cached
        return None

    def _get_s2_cached_work(self, article: dict) -> Optional[dict]:
        """Get Semantic Scholar work data from cache (try s2:doi:{doi} and s2:title:{title[:100]})."""
        doi = article.get("doi", "")
        if doi:
            cached = self.s2_cache.get(f"s2:doi:{doi}")
            if cached and isinstance(cached, dict):
                return cached
        title = article.get("title", "")
        if title:
            cached = self.s2_cache.get(f"s2:title:{title[:100]}")
            if cached and isinstance(cached, dict):
                return cached
        return None

    # ------------------------------------------------------------------
    # Stage A: Cache extraction (0 API calls)
    # ------------------------------------------------------------------

    def stage_a(self, articles: list[dict]) -> None:
        """Extract metadata from existing caches without any API calls."""
        logger.info("Stage A: Extracting metadata from caches...")

        for article in tqdm(articles, desc="Stage A: cache extraction"):
            oa_work = self._get_oa_cached_work(article)
            cr_work = self._get_cr_cached_work(article)
            s2_work = self._get_s2_cached_work(article)

            # --- DOI ---
            if _is_empty(article.get("doi")):
                cand_doi = ""
                if oa_work and oa_work.get("doi"):
                    cand_doi = normalize_doi(str(oa_work["doi"]))
                if not cand_doi and cr_work and cr_work.get("DOI"):
                    cand_doi = normalize_doi(str(cr_work["DOI"]))
                if not cand_doi and isinstance(article.get("crossref"), dict) and article["crossref"].get("doi"):
                    cand_doi = normalize_doi(str(article["crossref"]["doi"]))
                if not cand_doi and s2_work:
                    s2_ext = s2_work.get("externalIds") or {}
                    if s2_ext.get("DOI"):
                        cand_doi = normalize_doi(str(s2_ext["DOI"]))
                if cand_doi:
                    article["doi"] = cand_doi

            # --- Publication Title (Journal / Venue) ---
            if _is_empty(article.get("publication_title")):
                j_title = ""
                if oa_work:
                    primary = oa_work.get("primary_location") or {}
                    source = primary.get("source") or {}
                    j_title = source.get("display_name", "") or ""
                if not j_title and cr_work:
                    ct = cr_work.get("container-title") or []
                    if isinstance(ct, list) and ct:
                        j_title = str(ct[0])
                    elif isinstance(ct, str):
                        j_title = ct
                if j_title and not _is_empty(j_title):
                    article["publication_title"] = str(j_title).strip()

            # --- Publisher & Item Type ---
            if _is_empty(article.get("publisher")):
                pub = ""
                if oa_work:
                    pub = ((oa_work.get("primary_location") or {}).get("source") or {}).get("host_organization_name", "") or ""
                if not pub and cr_work:
                    pub = cr_work.get("publisher", "") or ""
                if pub and not _is_empty(pub):
                    article["publisher"] = str(pub).strip()

            if _is_empty(article.get("item_type")):
                it = (oa_work.get("type") if oa_work else "") or (cr_work.get("type") if cr_work else "") or "Journal Article"
                article["item_type"] = str(it).strip()

            # --- Language ---
            if _is_empty(article.get("language")):
                lang_code = None
                if oa_work:
                    lang_code = oa_work.get("language")
                if not lang_code and cr_work:
                    lang_code = cr_work.get("language")
                if lang_code and not _is_empty(lang_code):
                    full_name = _map_language_code(lang_code)
                    if full_name:
                        article["language"] = full_name
                        self.stats["language_cache"] += 1

            # --- Volume ---
            if _is_empty(article.get("volume")):
                vol = None
                if oa_work:
                    biblio = oa_work.get("biblio", {}) or {}
                    vol = biblio.get("volume")
                if not vol and cr_work:
                    vol = cr_work.get("volume")
                if vol and not _is_empty(vol):
                    article["volume"] = _clean_volume(vol)
                    self.stats["volume_filled"] += 1

            # --- Issue ---
            if _is_empty(article.get("issue")):
                iss = None
                if oa_work:
                    biblio = oa_work.get("biblio", {}) or {}
                    iss = biblio.get("issue")
                if not iss and cr_work:
                    iss = cr_work.get("issue")
                if iss and not _is_empty(iss):
                    article["issue"] = _clean_issue(iss)
                    self.stats["issue_filled"] += 1

            # --- Pages ---
            if _is_empty(article.get("pages")):
                biblio = (oa_work.get("biblio", {}) or {}) if oa_work else {}
                cr_page = cr_work.get("page", "") if cr_work else ""
                pages = _format_pages(biblio, cr_page)
                if pages:
                    article["pages"] = pages
                    self.stats["pages_filled"] += 1

            # --- ISSN ---
            if _is_empty(article.get("issn")):
                issn = None
                if oa_work:
                    primary = oa_work.get("primary_location", {}) or {}
                    source = primary.get("source", {}) or {}
                    issn = source.get("issn_l")
                    if not issn:
                        issn_list = source.get("issn", [])
                        if issn_list and isinstance(issn_list, list):
                            issn = issn_list[0]
                if not issn and cr_work:
                    cr_issn = cr_work.get("ISSN", [])
                    if cr_issn and isinstance(cr_issn, list):
                        issn = cr_issn[0]
                if issn and not _is_empty(issn):
                    article["issn"] = str(issn).strip()
                    self.stats["issn_filled"] += 1

            # --- Journal abbreviation (from Crossref) ---
            if _is_empty(article.get("journal_abbrev")):
                abbrev = None
                if cr_work:
                    short = cr_work.get("short-container-title", [])
                    if short and isinstance(short, list) and short[0]:
                        abbrev = short[0]
                if abbrev and not _is_empty(abbrev):
                    article["journal_abbrev"] = abbrev
                    self.stats["journal_abbrev_cache"] += 1

            # --- URL from DOI ---
            if _is_empty(article.get("url")):
                doi = article.get("doi", "")
                if doi and not _is_empty(doi):
                    article["url"] = f"https://doi.org/{doi}"
                    self.stats["url_constructed"] += 1

        logger.info(
            f"Stage A complete: "
            f"language={self.stats['language_cache']}, "
            f"volume={self.stats['volume_filled']}, "
            f"issue={self.stats['issue_filled']}, "
            f"pages={self.stats['pages_filled']}, "
            f"issn={self.stats['issn_filled']}, "
            f"journal_abbrev={self.stats['journal_abbrev_cache']}, "
            f"url={self.stats['url_constructed']}"
        )

    # ------------------------------------------------------------------
    # Stage B: Language API (minimal calls)
    # ------------------------------------------------------------------

    def stage_b(self, articles: list[dict]) -> None:
        """Fetch language for articles still missing it after Stage A."""
        need_lang = [
            a for a in articles
            if _is_empty(a.get("language"))
            and a.get("openalex", {}).get("openalex_id")
        ]

        if not need_lang:
            logger.info("Stage B: All articles already have language, skipping.")
            return

        logger.info(f"Stage B: Fetching language for {len(need_lang)} articles via API...")

        for article in tqdm(need_lang, desc="Stage B: language API"):
            oa_id = article["openalex"]["openalex_id"]
            work_id = oa_id.split("/")[-1] if "/" in oa_id else oa_id

            url = f"{self.base_url}/works/{work_id}"
            result = safe_request(
                url, session=self.session,
                rate_limiter=self.rate_limiter,
                params=self._build_params({"select": "id,language"}),
            )

            if result and result.get("language"):
                lang = _map_language_code(result["language"])
                if lang:
                    article["language"] = lang
                    self.stats["language_api"] += 1

        logger.info(f"Stage B complete: {self.stats['language_api']} languages fetched via API")

    # ------------------------------------------------------------------
    # Stage C: Journal abbreviation API (unique sources)
    # ------------------------------------------------------------------

    def stage_c(self, articles: list[dict]) -> None:
        """Fetch journal abbreviations for articles still missing them."""
        # Collect unique source IDs that need abbreviation
        source_needs: dict[str, list[dict]] = {}  # source_id -> [articles]

        for article in articles:
            if not _is_empty(article.get("journal_abbrev")):
                continue

            # Get source ID from OpenAlex work data
            oa_work = self._get_oa_cached_work(article)
            if not oa_work:
                continue

            primary = oa_work.get("primary_location", {}) or {}
            source = primary.get("source", {}) or {}
            source_id = source.get("id", "")

            if not source_id:
                # Try locations
                for loc in oa_work.get("locations", []) or []:
                    src = (loc or {}).get("source", {}) or {}
                    sid = src.get("id", "")
                    if sid:
                        source_id = sid
                        break

            if source_id:
                short_id = source_id.split("/")[-1] if "/" in source_id else source_id
                source_needs.setdefault(short_id, []).append(article)

        if not source_needs:
            logger.info("Stage C: No articles need journal abbreviation, skipping.")
            return

        logger.info(f"Stage C: Fetching abbreviations for {len(source_needs)} unique sources...")

        for source_id, arts in tqdm(source_needs.items(), desc="Stage C: source API"):
            # Check meta cache first
            cache_key = f"source:{source_id}"
            cached = self.oa_meta_cache.get(cache_key)

            if cached is None:
                # Fetch from API
                url = f"{self.base_url}/sources/{source_id}"
                result = safe_request(
                    url, session=self.session,
                    rate_limiter=self.rate_limiter,
                    params=self._build_params({
                        "select": "id,display_name,abbreviated_title,alternate_titles"
                    }),
                )
                if result and "id" in result:
                    cached = result
                    self.oa_meta_cache.set(cache_key, cached)
                else:
                    self.oa_meta_cache.set(cache_key, {})
                    continue

            if not cached:
                continue

            # Determine best abbreviation
            abbrev = self._pick_abbreviation(cached)
            if not abbrev:
                continue

            for art in arts:
                if _is_empty(art.get("journal_abbrev")):
                    art["journal_abbrev"] = abbrev
                    self.stats["journal_abbrev_api"] += 1

        logger.info(
            f"Stage C complete: {self.stats['journal_abbrev_api']} "
            f"abbreviations set from {len(source_needs)} sources"
        )

    @staticmethod
    def _pick_abbreviation(source_data: dict) -> str:
        """Pick best abbreviation from source data.

        Fallback chain: abbreviated_title -> shortest alternate_title -> display_name (if short).
        """
        # Primary: abbreviated_title
        abbrev = source_data.get("abbreviated_title", "")
        if abbrev and not _is_empty(abbrev):
            return abbrev.strip()

        # Secondary: shortest alternate_title
        alts = source_data.get("alternate_titles", [])
        if alts and isinstance(alts, list):
            valid = [a for a in alts if a and not _is_empty(a)]
            if valid:
                shortest = min(valid, key=len)
                if len(shortest) <= 50:
                    return shortest.strip()

        # Tertiary: display_name if short enough
        name = source_data.get("display_name", "")
        if name and not _is_empty(name) and len(name) <= 30:
            return name.strip()

        return ""

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def log_final_stats(self, total: int) -> None:
        """Log summary of all metadata filled."""
        s = self.stats
        logger.info("=" * 60)
        logger.info("Phase 1j: Metadata Completion Summary")
        logger.info(f"  language:       +{s['language_cache']} from cache, "
                     f"+{s['language_api']} from API")
        logger.info(f"  journal_abbrev: +{s['journal_abbrev_cache']} from cache, "
                     f"+{s['journal_abbrev_api']} from API")
        logger.info(f"  pages:          +{s['pages_filled']} filled")
        logger.info(f"  volume:         +{s['volume_filled']} filled")
        logger.info(f"  issue:          +{s['issue_filled']} filled")
        logger.info(f"  issn:           +{s['issn_filled']} filled")
        logger.info(f"  url:            +{s['url_constructed']} constructed from DOI")

        # Remaining gaps
        remaining = {}
        for field in ("language", "journal_abbrev", "pages", "volume",
                      "issue", "issn", "url"):
            # We don't have access to articles here, so this is logged in run_phase1j
            pass
        logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_export_readiness(articles: list[dict]) -> dict:
    """Validate data completeness before Bibliometrix export."""
    total = len(articles)
    if total == 0:
        return {"warnings": ["No articles to validate"], "info": [], "summary": "0 articles"}

    # Count non-empty values for each field
    field_counts = {}
    fields_to_check = {
        # Critical fields
        "AU": lambda a: a.get("authors") and len(a["authors"]) > 0,
        "TI": lambda a: not _is_empty(a.get("title")),
        "SO": lambda a: not _is_empty(a.get("publication_title")),
        "PY": lambda a: not _is_empty(a.get("publication_year")),
        "DI": lambda a: not _is_empty(a.get("doi")),
        # Important fields
        "LA": lambda a: not _is_empty(a.get("language")),
        "CR": lambda a: bool(a.get("referenced_works")),
        "C1": lambda a: bool(a.get("openalex", {}).get("authorships")),
        "DE": lambda a: bool(a.get("manual_tags")),
        "ID": lambda a: bool(a.get("automatic_tags")) or bool(
            a.get("openalex", {}).get("concepts")),
        # Desirable fields
        "JI": lambda a: not _is_empty(a.get("journal_abbrev")),
        "VL": lambda a: not _is_empty(a.get("volume")),
        "IS": lambda a: not _is_empty(a.get("issue")),
        "PP": lambda a: not _is_empty(a.get("pages")),
        "SN": lambda a: not _is_empty(a.get("issn")),
        "URL": lambda a: not _is_empty(a.get("url")),
    }

    for field, check_fn in fields_to_check.items():
        field_counts[field] = sum(1 for a in articles if check_fn(a))

    # Classify warnings
    warnings = []
    info = []

    critical_fields = {"AU": "Authors", "TI": "Title", "SO": "Source",
                       "PY": "Year", "DI": "DOI"}
    important_fields = {"LA": "Language", "CR": "References", "C1": "Affiliations",
                        "DE": "Author Keywords", "ID": "Keywords Plus"}
    desirable_fields = {"JI": "Journal Abbrev", "VL": "Volume", "IS": "Issue",
                        "PP": "Pages", "SN": "ISSN", "URL": "URL"}

    for field, label in critical_fields.items():
        count = field_counts.get(field, 0)
        pct = count / total * 100
        missing = total - count
        if pct < 95:
            warnings.append(
                f"CRITICAL: {label} ({field}): {missing}/{total} missing ({100-pct:.0f}%)"
            )

    for field, label in important_fields.items():
        count = field_counts.get(field, 0)
        pct = count / total * 100
        missing = total - count
        if pct < 80:
            warnings.append(
                f"IMPORTANT: {label} ({field}): {missing}/{total} missing ({100-pct:.0f}%)"
            )

    for field, label in desirable_fields.items():
        count = field_counts.get(field, 0)
        pct = count / total * 100
        missing = total - count
        if pct < 50:
            info.append(
                f"DESIRABLE: {label} ({field}): {missing}/{total} missing ({100-pct:.0f}%)"
            )

    # Summary line
    n_warnings = len(warnings)
    n_info = len(info)
    summary = (
        f"{total} articles validated: "
        f"{n_warnings} warnings, {n_info} info notes"
    )

    # Log field completeness table
    logger.info("Export validation — field completeness:")
    all_fields = {**critical_fields, **important_fields, **desirable_fields}
    for field, label in all_fields.items():
        count = field_counts.get(field, 0)
        pct = count / total * 100
        bar = "#" * int(pct / 5) + "." * (20 - int(pct / 5))
        logger.info(f"  {field:4s} ({label:16s}): {count:4d}/{total} ({pct:5.1f}%) [{bar}]")

    return {"warnings": warnings, "info": info, "summary": summary}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_phase1j(articles: list[dict], rate_override: float | None = None) -> list[dict]:
    """Run Phase 1j: Metadata Completion from caches + minimal API."""
    logger.info("=" * 60)
    logger.info("=== FASE 1j: Metadata Completion ===")
    logger.info(f"Articles to process: {len(articles)}")

    for a in articles:
        if not _is_empty(a.get("language")):
            a["language"] = normalize_language(str(a["language"]))
    canonicalize_corpus_journals(articles)

    completer = MetadataCompleter(rate_override=rate_override)

    # Stage A: Cache extraction (0 API calls)
    completer.stage_a(articles)

    # Stage B: Language API (minimal calls)
    completer.stage_b(articles)

    # Stage C: Journal abbreviation API
    completer.stage_c(articles)

    # Final stats
    completer.log_final_stats(len(articles))

    # Log remaining gaps
    total = len(articles)
    for field_name in ("language", "journal_abbrev", "pages", "volume",
                       "issue", "issn", "url"):
        remaining = sum(1 for a in articles if _is_empty(a.get(field_name)))
        logger.info(f"  Remaining gaps — {field_name}: {remaining}/{total}")

    # Save checkpoint
    save_json(articles, ENRICHED_PATH)
    logger.info(f"Checkpoint saved: {ENRICHED_PATH}")

    return articles
