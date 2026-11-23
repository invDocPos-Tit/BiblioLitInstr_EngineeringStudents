"""Phase 1i: Enhanced reference resolution with concepts/topics.

Batch-fetches ALL unique referenced works from OpenAlex with enhanced
metadata (concepts, topics, keywords, cited_by_count, type) and attaches
``references_resolved`` to each article.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any, Optional

from tqdm import tqdm

from config import (
    ENRICHED_PATH,
    OPENALEX_API_KEY,
    OPENALEX_CACHE_DIR,
    OPENALEX_EMAIL,
    OPENALEX_RATE_LIMIT,
)
from src.utils import DiskCache, RateLimiter, create_session, save_json

logger = logging.getLogger("bibliometrico.reference_resolver")

# Enhanced select fields — superset of what network_analysis uses
ENHANCED_SELECT = (
    "id,doi,authorships,publication_year,title,concepts,topics,keywords,"
    "type,primary_location,biblio,cited_by_count"
)


# OpenAlex Work ID canonico: 'W' + digitos. Las entradas de
# referenced_works a veces vienen corruptas (fragmentos de DOI, ISBN, o
# cadenas tipo 'j.x.2021+openalex:W123+pmid:456'); el endpoint /works
# rechaza con 400 TODO el lote si una sola entrada no es un ID valido.
_WID_RE = re.compile(r"\bW\d{4,}\b")


def _canonical_wid(ref: str) -> str:
    """Extrae el OpenAlex Work ID canonico (``W\\d+``) de una referencia.

    Acepta ``https://openalex.org/W123``, ``W123``, ``openalex:W123`` e
    incluso cadenas corruptas que contengan un ``W\\d+`` embebido. Devuelve
    "" si no hay un Work ID resoluble (DOI/ISBN/basura) -> esa referencia
    no entra al lote (no se puede resolver por openalex_id).
    """
    if not ref or not isinstance(ref, str):
        return ""
    m = _WID_RE.search(ref)
    return m.group(0) if m else ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_first_author(work: dict) -> str:
    """Return first author family name or 'Unknown'."""
    authorships = work.get("authorships") or []
    if authorships:
        author = authorships[0].get("author", {})
        name = author.get("display_name", "")
        if name:
            return name.split()[-1] if " " in name else name
    return "Unknown"


def _normalize_concepts(work: dict) -> list[dict]:
    """Extract and normalize concepts from a work record."""
    concepts = []
    for c in work.get("concepts") or []:
        score = c.get("score", 0) or 0
        concepts.append({
            "name": c.get("display_name", ""),
            "level": c.get("level", 0),
            "score": round(float(score), 4),
        })
    return sorted(concepts, key=lambda x: x["score"], reverse=True)


def _normalize_topics(work: dict) -> list[dict]:
    """Extract and normalize topics from a work record."""
    topics = []
    for t in work.get("topics") or []:
        score = t.get("score", 1.0)
        entry: dict[str, Any] = {
            "name": t.get("display_name", "") or t.get("name", ""),
            "score": round(float(score if score is not None else 1.0), 4),
        }
        subfield = t.get("subfield", {})
        field = t.get("field", {})
        domain = t.get("domain", {})
        if isinstance(subfield, dict):
            entry["subfield"] = subfield.get("display_name", "")
        elif isinstance(subfield, str):
            entry["subfield"] = subfield
        if isinstance(field, dict):
            entry["field"] = field.get("display_name", "")
        elif isinstance(field, str):
            entry["field"] = field
        if isinstance(domain, dict):
            entry["domain"] = domain.get("display_name", "")
        elif isinstance(domain, str):
            entry["domain"] = domain
        topics.append(entry)
    return topics


def _build_resolved_entry(work: dict) -> dict:
    """Build a compact resolved reference entry from a full work record."""
    return {
        "openalex_id": work.get("id", ""),
        "year": work.get("publication_year"),
        "title": work.get("title", ""),
        "first_author": _extract_first_author(work),
        "concepts": _normalize_concepts(work),
        "topics": _normalize_topics(work),
        "keywords": [
            kw.get("keyword", "") if isinstance(kw, dict) else str(kw)
            for kw in (work.get("keywords") or [])
        ],
        "cited_by_count": work.get("cited_by_count", 0),
        "type": work.get("type", ""),
    }


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def collect_unique_refs(articles: list[dict]) -> set[str]:
    """Collect all unique OpenAlex work URLs from articles' referenced_works."""
    refs: set[str] = set()
    for a in articles:
        for ref in a.get("referenced_works") or []:
            if ref and isinstance(ref, str):
                refs.add(ref)
    return refs


def _needs_fetch(cached: Optional[dict]) -> bool:
    """Return True if cached entry is missing, lacks concepts, or lacks doi key."""
    if not cached or not isinstance(cached, dict):
        return True
    # Re-fetch if concepts missing or doi key absent (added to ENHANCED_SELECT)
    if cached.get("concepts") is not None and "doi" in cached:
        return False
    return True


def batch_resolve_references(
    ref_urls: list[str],
    cache: DiskCache,
    session,
    rate_limiter: RateLimiter,
    batch_size: int = 25,
    base_backoff: float = 1.0,
) -> dict[str, dict]:
    """Batch-fetch references from OpenAlex with enhanced select fields.

    Returns dict mapping work_id -> full API record (also cached to disk).
    """
    resolved: dict[str, dict] = {}
    urls_to_fetch: list[str] = []

    # Check cache first (with progress for large sets)
    total_refs = len(ref_urls)
    log_interval = max(1, total_refs // 5)  # Log ~5 times during check
    skipped_unresolvable = 0
    seen_wids: set[str] = set()
    for i, url in enumerate(ref_urls):
        work_id = _canonical_wid(url)
        if not work_id:
            # Fragmento de DOI / ISBN / cadena corrupta: no resoluble por
            # openalex_id. Se omite para no envenenar el lote (400).
            skipped_unresolvable += 1
            continue
        if work_id in seen_wids:
            continue
        seen_wids.add(work_id)
        cached = cache.get(f"work:{work_id}")
        if not _needs_fetch(cached):
            resolved[work_id] = cached
        else:
            urls_to_fetch.append(work_id)
        if (i + 1) % log_interval == 0:
            logger.info(
                f"  Cache check: {i + 1}/{total_refs} "
                f"({len(resolved)} cached, {len(urls_to_fetch)} pending)"
            )

    logger.info(
        f"Reference resolution: {len(resolved)} cached (with concepts), "
        f"{len(urls_to_fetch)} to fetch"
        + (f", {skipped_unresolvable} omitidas (sin Work ID OpenAlex)"
           if skipped_unresolvable else "")
    )

    if not urls_to_fetch:
        return resolved

    # Build base params
    base_params: dict[str, str] = {}
    if OPENALEX_API_KEY:
        base_params["api_key"] = OPENALEX_API_KEY
        logger.info("Using OpenAlex API key (premium pool)")
    elif OPENALEX_EMAIL:
        base_params["mailto"] = OPENALEX_EMAIL

    BATCH = max(1, min(50, batch_size))
    batches = [urls_to_fetch[i:i + BATCH]
               for i in range(0, len(urls_to_fetch), BATCH)]

    resolved_count = 0
    failed_batches = 0

    def _fetch_wids(wids: list[str]) -> int:
        """Fetch canonical Work IDs; cachea + almacena; devuelve cuantos.

        Lanza excepcion ante error HTTP para que el llamador reintente
        id-a-id (un ID corrupto/fabricado no debe tumbar el lote).
        """
        n = 0
        full_urls = [f"https://openalex.org/{w}" for w in wids]
        params = {
            **base_params,
            "filter": f"openalex_id:{'|'.join(full_urls)}",
            "select": ENHANCED_SELECT,
            "per_page": str(max(1, min(50, len(wids)))),
        }
        resp = session.get(
            "https://api.openalex.org/works", params=params, timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        for work in results:
            oa_id = work.get("id", "")
            wid = oa_id.split("/")[-1] if "/" in oa_id else oa_id
            cache.set(f"work:{wid}", work)
            resolved[wid] = work
            n += 1
        # Paginacion (raro con <=50 ids, pero por robustez)
        total_results = data.get("meta", {}).get("count", len(results))
        page = 2
        while len(results) < total_results:
            rate_limiter.wait()
            try:
                resp2 = session.get(
                    "https://api.openalex.org/works",
                    params={**params, "page": str(page)}, timeout=30,
                )
                resp2.raise_for_status()
                page_results = resp2.json().get("results", [])
                if not page_results:
                    break
                for work in page_results:
                    oa_id = work.get("id", "")
                    wid = oa_id.split("/")[-1] if "/" in oa_id else oa_id
                    cache.set(f"work:{wid}", work)
                    resolved[wid] = work
                    n += 1
                results.extend(page_results)
                page += 1
            except Exception:
                break
        return n

    for batch in tqdm(batches, desc="Resolving references (enhanced)"):
        rate_limiter.wait()
        try:
            resolved_count += _fetch_wids(batch)
        except Exception as e:
            # Reintento id-a-id: conserva las refs validas del lote aunque
            # una sea irresoluble (no deberia pasar tras sanitizar a W\\d+).
            recovered, bad = 0, 0
            if len(batch) > 1:
                for w in batch:
                    rate_limiter.wait()
                    try:
                        recovered += _fetch_wids([w])
                    except Exception:
                        bad += 1
            else:
                bad = 1
            resolved_count += recovered
            if bad:
                failed_batches += 1
                if failed_batches <= 3:
                    logger.warning(
                        f"Lote con {bad} ref(s) irresoluble(s); "
                        f"{recovered} recuperadas id-a-id ({e})"
                    )
                time.sleep(base_backoff * min(failed_batches, 5))

    logger.info(
        f"Resolved {resolved_count} new references "
        f"({failed_batches} failed batches)"
    )
    return resolved


def build_article_references_resolved(
    articles: list[dict],
    resolved: dict[str, dict],
) -> list[dict]:
    """Attach 'references_resolved' list to each article."""
    for a in articles:
        refs_resolved = []
        for ref_url in a.get("referenced_works") or []:
            work_id = _canonical_wid(ref_url)
            if not work_id:
                continue
            work = resolved.get(work_id)
            if work and isinstance(work, dict):
                refs_resolved.append(_build_resolved_entry(work))
        a["references_resolved"] = refs_resolved
    return articles


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_phase1i(
    articles: list[dict],
    rate_override: Optional[float] = None,
    batch_size: int = 25,
    base_backoff: float = 1.0,
) -> list[dict]:
    """Phase 1i: Enhanced reference resolution.

    1. Collect all unique referenced work URLs
    2. Check cache (skip if already has concepts)
    3. Batch-fetch from OpenAlex with enhanced select
    4. Attach references_resolved to each article
    5. Save checkpoint
    """
    logger.info("=" * 60)
    logger.info("Phase 1i: Enhanced Reference Resolution")
    logger.info("=" * 60)

    # 1. Collect unique refs
    ref_urls = collect_unique_refs(articles)
    logger.info(f"Unique referenced works: {len(ref_urls)}")

    if not ref_urls:
        logger.warning("No referenced works found in articles")
        return articles

    # Skip check: if all articles already have references_resolved, skip
    n_already = sum(1 for a in articles if a.get("references_resolved"))
    n_with_refs = sum(1 for a in articles if a.get("referenced_works"))
    if n_already >= n_with_refs and n_already > 0:
        logger.info(
            f"All {n_already} articles already have references_resolved — skipping"
        )
        return articles

    # 2-3. Cache + batch fetch
    cache = DiskCache(OPENALEX_CACHE_DIR)
    session = create_session()

    rate = OPENALEX_RATE_LIMIT
    if OPENALEX_API_KEY:
        rate = min(OPENALEX_RATE_LIMIT * 5, 50)
    if rate_override:
        rate = min(max(rate_override, 0.1), 50)
    rate_limiter = RateLimiter(rate)

    resolved = batch_resolve_references(
        list(ref_urls), cache, session, rate_limiter,
        batch_size=batch_size, base_backoff=base_backoff,
    )

    # 4. Attach to articles
    articles = build_article_references_resolved(articles, resolved)

    # Stats
    n_with_concepts = sum(
        1 for w in resolved.values()
        if w.get("concepts")
    )
    logger.info(
        f"References resolved: {len(resolved)}/{len(ref_urls)} "
        f"({n_with_concepts} with concepts)"
    )

    # 5. Save checkpoint
    save_json(articles, ENRICHED_PATH)
    logger.info("Phase 1i complete — references_resolved attached to articles")

    return articles
