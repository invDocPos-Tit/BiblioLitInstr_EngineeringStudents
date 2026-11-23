"""Phase 2g: Reference Spectroscopy — concept evolution through cited references.

Five analyses:
1. Enhanced RPYS (year histogram with concept-colored stacked bars + deviation)
2. Concept Evolution Heatmap (top concepts x reference years matrix)
3. Concept Lifecycle (emergence -> growth -> peak -> decline per concept)
4. Seminal Works (top-cited references per concept per decade)
5. Concept Co-occurrence (how concepts co-occur within same references over time)

Plus: spectroscopy_report.md and spectroscopy_data.json
"""
from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Optional

import numpy as np

from config import OPENALEX_CACHE_DIR, SPECTROSCOPY_DIR
from src.utils import DiskCache, save_json, is_noisy_legacy_concept

logger = logging.getLogger("bibliometrico.reference_spectroscopy")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MIN_CONCEPT_SCORE = 0.3
MIN_CONCEPT_FREQ = 10
TOP_N_CONCEPTS = 30
CONCEPT_LEVEL_MAX = 2
MIN_YEAR = 1950
MAX_YEAR = 2026
# RPYS estandar (Marx, Bornmann, Barth y Leydesdorff, 2014): desviacion respecto de la mediana
# de una ventana centrada de 5 anos (t-2 .. t+2). En los bordes se usa la parte disponible.
RPYS_VENTANA_MEDIANA = 5
# Los ultimos anos de referencias estan incompletos: esas obras aun no tuvieron tiempo de ser
# citadas. Se marcan (no se excluyen) para que no se lean como picos.
RPYS_ANOS_RECIENTES_INCOMPLETOS = 3
RPYS_N_PICOS_REPORTE = 15
RPYS_N_OBRAS_POR_PICO = 3


# ---------------------------------------------------------------------------
# Plotly save helper (same pattern used elsewhere)
# ---------------------------------------------------------------------------

def _save_plotly_figure(fig, output_dir: Path, name: str) -> None:
    """Save Plotly figure as HTML and optionally PNG."""
    from src.utils import safe_write_plotly_png

    output_dir.mkdir(parents=True, exist_ok=True)

    html_path = output_dir / f"{name}.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    logger.info(f"Saved: {html_path}")

    png_path = output_dir / f"{name}.png"
    safe_write_plotly_png(fig, png_path, width=1600, height=1000, scale=2, log=logger)


# ---------------------------------------------------------------------------
# Cache enrichment — fetch missing fields for cached references
# ---------------------------------------------------------------------------

def _enrich_cache_if_needed(articles: list[dict]) -> int:
    """Re-fetch OpenAlex data for references whose cache entry lacks
    cited_by_count or concepts (due to a restrictive `select` in Phase 3).

    Returns number of entries updated.
    """
    from config import OPENALEX_EMAIL, OPENALEX_API_KEY, OPENALEX_RATE_LIMIT
    from src.utils import RateLimiter, create_session

    cache = DiskCache(OPENALEX_CACHE_DIR)

    # Collect all ref work IDs
    all_refs: set[str] = set()
    for a in articles:
        for ref in a.get("referenced_works") or []:
            if ref and isinstance(ref, str):
                wid = ref.split("/")[-1] if "/" in ref else ref
                if wid.startswith("W") and wid[1:].isdigit():
                    all_refs.add(wid)

    # Find which ones need enrichment
    needs_update: list[str] = []
    for wid in all_refs:
        cached = cache.get(f"work:{wid}")
        if cached and isinstance(cached, dict):
            # Missing critical fields?
            if cached.get("cited_by_count") is None or not cached.get("concepts"):
                needs_update.append(wid)
        # Not cached at all — also fetch
        elif cached is None:
            needs_update.append(wid)

    if not needs_update:
        return 0

    logger.info(
        f"Enriching cache: {len(needs_update)}/{len(all_refs)} refs need "
        f"cited_by_count/concepts data"
    )

    session = create_session()
    base_params: dict = {}
    if OPENALEX_API_KEY:
        base_params["api_key"] = OPENALEX_API_KEY
    elif OPENALEX_EMAIL:
        base_params["mailto"] = OPENALEX_EMAIL

    rate = min(OPENALEX_RATE_LIMIT * (5 if OPENALEX_API_KEY else 1), 50)
    rate_limiter = RateLimiter(rate)

    BATCH_SIZE = 25
    batches = [needs_update[i:i + BATCH_SIZE]
               for i in range(0, len(needs_update), BATCH_SIZE)]

    updated = 0
    for batch in batches:
        rate_limiter.wait()
        full_urls = [f"https://openalex.org/{wid}" for wid in batch]
        filter_str = "|".join(full_urls)
        params = {
            **base_params,
            "filter": f"openalex_id:{filter_str}",
            "select": "id,doi,authorships,publication_year,title,cited_by_count,concepts,primary_location,biblio",
            "per_page": str(BATCH_SIZE),
        }
        try:
            resp = session.get(
                "https://api.openalex.org/works",
                params=params,
                timeout=30,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            for work in results:
                oa_id = work.get("id", "")
                wid = oa_id.split("/")[-1] if "/" in oa_id else oa_id
                cache.set(f"work:{wid}", work)
                updated += 1
        except Exception as e:
            logger.warning(f"Cache enrichment batch failed: {e}")

    logger.info(f"Cache enriched: {updated} refs updated with full data")
    return updated


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def _extract_concept_name_key(c: dict) -> str:
    """Normalize concept/topic name for grouping."""
    return (c.get("name") or c.get("display_name") or "").strip()


def _extract_topics_or_concepts(record: dict) -> list[dict]:
    """Extract CWTS Leiden topics first; fall back to filtered legacy concepts."""
    extracted: list[dict] = []
    seen_names: set[str] = set()

    # Priority 1: OpenAlex CWTS Leiden topics
    for t in record.get("topics") or []:
        name = _extract_concept_name_key(t)
        score = float(t.get("score", 0.85) or 0.85)
        if name and not is_noisy_legacy_concept(name) and name.lower() not in seen_names:
            seen_names.add(name.lower())
            extracted.append({
                "name": name,
                "level": 1,
                "score": score,
            })
        subfield = t.get("subfield", "")
        if isinstance(subfield, dict):
            subfield = subfield.get("display_name", "")
        subfield = (subfield or "").strip()
        if subfield and not is_noisy_legacy_concept(subfield) and subfield.lower() not in seen_names:
            seen_names.add(subfield.lower())
            extracted.append({
                "name": subfield,
                "level": 2,
                "score": round(score * 0.8, 4),
            })

    if extracted:
        return extracted

    # Priority 2: Legacy OpenAlex concepts (filtered against noisy generic tags)
    for c in record.get("concepts") or []:
        score = c.get("score", 0) or 0
        level = c.get("level", 0) or 0
        name = _extract_concept_name_key(c)
        if not name or is_noisy_legacy_concept(name):
            continue
        if float(score) < MIN_CONCEPT_SCORE:
            continue
        if int(level) > CONCEPT_LEVEL_MAX:
            continue
        if name.lower() not in seen_names:
            seen_names.add(name.lower())
            extracted.append({
                "name": name,
                "level": int(level),
                "score": float(score),
            })

    return extracted


def _build_ref_concept_data(articles: list[dict]) -> tuple[list[dict], dict]:
    """Extract reference-topic/concept pairs from references_resolved.

    Returns:
        records: list of dicts {year, title, first_author, cited_by_count,
                                concepts: [{name, level, score}], openalex_id}
        concept_freq: Counter of concept name -> total frequency
    """
    records: list[dict] = []
    concept_freq: Counter = Counter()
    seen_ids: set[str] = set()

    for a in articles:
        article_topics = _extract_topics_or_concepts({
            "topics": a.get("topics") or a.get("openalex", {}).get("topics", []),
            "concepts": a.get("concepts") or a.get("openalex", {}).get("concepts", []),
        })
        for ref in a.get("references_resolved") or []:
            oa_id = ref.get("openalex_id", "")
            dedup_key = oa_id or f"{ref.get('first_author', '')}_{ref.get('year', '')}_{str(ref.get('title', ''))[:60]}"
            if not dedup_key or dedup_key in seen_ids:
                continue
            seen_ids.add(dedup_key)

            year = ref.get("year")
            if not year or not isinstance(year, int):
                continue
            if year < MIN_YEAR or year > MAX_YEAR:
                continue

            concepts = _extract_topics_or_concepts(ref)
            if not concepts and article_topics:
                # External references (S2/Scopus/WoS) without OpenAlex topics inherit citing article's topics
                concepts = article_topics

            if not concepts:
                continue

            for c in concepts:
                concept_freq[c["name"]] += 1

            records.append({
                "year": year,
                "title": ref.get("title", ""),
                "first_author": ref.get("first_author", "Unknown"),
                "cited_by_count": ref.get("cited_by_count", 0) or 0,
                "concepts": concepts,
                "openalex_id": oa_id,
            })

    return records, concept_freq


def _ref_key(ref: dict) -> str:
    """Misma clave de obra que _build_ref_concept_data (OpenAlex ID o autor+ano+titulo)."""
    return ref.get("openalex_id", "") or (
        f"{ref.get('first_author', '')}_{ref.get('year', '')}_{str(ref.get('title', ''))[:60]}"
    )


def _count_reference_citations(articles: list[dict]) -> dict:
    """Cuenta CITAS (cada aparicion en la lista de referencias de un articulo), no obras distintas.

    RPYS mide cuantas veces el corpus cita trabajos de cada ano: una obra citada por 25
    articulos aporta 25. Contar obras distintas borra justamente los picos historicos.
    """
    by_year: Counter = Counter()
    by_work: Counter = Counter()
    works_by_year: dict[int, Counter] = defaultdict(Counter)
    labels: dict[str, dict] = {}
    for a in articles:
        for ref in a.get("references_resolved") or []:
            year = ref.get("year")
            if not isinstance(year, int) or year < MIN_YEAR or year > MAX_YEAR:
                continue
            key = _ref_key(ref)
            if not key:
                continue
            by_year[year] += 1
            by_work[key] += 1
            works_by_year[year][key] += 1
            labels.setdefault(key, {
                "first_author": ref.get("first_author", "Unknown"),
                "title": str(ref.get("title", "") or "")[:100],
                "year": year,
                "openalex_id": ref.get("openalex_id", ""),
            })
    return {"by_year": by_year, "by_work": by_work, "works_by_year": works_by_year, "labels": labels}


def _build_ref_concept_data_from_cache(
    articles: list[dict],
) -> tuple[list[dict], dict]:
    """Fallback: read references from OpenAlex cache when references_resolved
    is missing (Phase 1i was skipped)."""
    cache = DiskCache(OPENALEX_CACHE_DIR)
    records: list[dict] = []
    concept_freq: Counter = Counter()
    seen_ids: set[str] = set()

    all_refs: set[str] = set()
    for a in articles:
        for ref in a.get("referenced_works") or []:
            if ref and isinstance(ref, str):
                all_refs.add(ref)

    logger.info(f"Fallback: reading {len(all_refs)} refs from cache...")

    for ref_url in all_refs:
        work_id = ref_url.split("/")[-1] if "/" in ref_url else ref_url
        if work_id in seen_ids:
            continue
        seen_ids.add(work_id)

        cached = cache.get(f"work:{work_id}")
        if not cached or not isinstance(cached, dict):
            continue

        year = cached.get("publication_year")
        if not year or not isinstance(year, int):
            continue
        if year < MIN_YEAR or year > MAX_YEAR:
            continue

        concepts = _extract_topics_or_concepts(cached)
        if not concepts:
            continue

        for c in concepts:
            concept_freq[c["name"]] += 1

        # Extract first author
        authorships = cached.get("authorships") or []
        first_author = "Unknown"
        if authorships:
            author = authorships[0].get("author", {})
            disp = author.get("display_name", "")
            if disp:
                first_author = disp.split()[-1] if " " in disp else disp

        records.append({
            "year": year,
            "title": cached.get("title", ""),
            "first_author": first_author,
            "cited_by_count": cached.get("cited_by_count", 0) or 0,
            "concepts": concepts,
            "openalex_id": cached.get("id", ""),
        })

    return records, concept_freq


def _build_ref_data_from_article_proxy(
    articles: list[dict],
) -> tuple[list[dict], Counter]:
    """Fallback: use article-level concepts/keywords as thematic proxy for references.

    When references lack their own concept data, each reference inherits the
    thematic profile of its citing article. This is methodologically justified:
    a reference cited by an article on 'LSTM + forex prediction' participates
    in that thematic spectrum.

    Data sources (in priority order):
    1. Article concepts from OpenAlex (name + score)
    2. Author keywords (parsed from string)
    3. Keywords Plus / index terms

    References come from referenced_works_detailed (Crossref) or
    referenced_works_raw with year extraction.
    """
    records: list[dict] = []
    concept_freq: Counter = Counter()
    seen_refs: set[str] = set()

    # Build DOI→cited_by_count map from OpenAlex cache for citation enrichment
    cache = DiskCache(OPENALEX_CACHE_DIR)
    _doi_to_cites: dict[str, int] = {}
    _oa_id_to_cites: dict[str, int] = {}
    all_ref_urls: set[str] = set()
    for a in articles:
        for ref in a.get("referenced_works") or []:
            if ref and isinstance(ref, str):
                all_ref_urls.add(ref)
    for ref_url in all_ref_urls:
        work_id = ref_url.split("/")[-1] if "/" in ref_url else ref_url
        cached = cache.get(f"work:{work_id}")
        if cached and isinstance(cached, dict):
            cites = cached.get("cited_by_count", 0) or 0
            _oa_id_to_cites[ref_url] = cites
            ref_doi = cached.get("doi", "")
            if ref_doi:
                # Normalize DOI
                ref_doi = ref_doi.replace("https://doi.org/", "").lower().strip()
                _doi_to_cites[ref_doi] = cites

    for a in articles:
        # Build topic/concept list from article-level data (prioritize OpenAlex topics)
        article_concepts = _extract_topics_or_concepts({
            "topics": a.get("topics") or a.get("openalex", {}).get("topics", []),
            "concepts": a.get("concepts") or a.get("openalex", {}).get("concepts", []),
        })

        # Source 2: Author keywords (string or list)
        kw_raw = a.get("author_keywords", "") or ""
        if isinstance(kw_raw, str) and kw_raw.strip():
            for kw in kw_raw.split(";"):
                kw = kw.strip()
                if kw and len(kw) > 2:
                    article_concepts.append({
                        "name": kw,
                        "level": 2,
                        "score": 0.5,
                    })

        # Source 3: Keywords Plus / index terms
        kp_raw = a.get("keywords_plus", "") or a.get("index_keywords", "") or ""
        if isinstance(kp_raw, str) and kp_raw.strip():
            for kw in kp_raw.split(";"):
                kw = kw.strip()
                if kw and len(kw) > 2:
                    # Avoid duplicates with author keywords
                    if not any(c["name"].lower() == kw.lower() for c in article_concepts):
                        article_concepts.append({
                            "name": kw,
                            "level": 3,
                            "score": 0.3,
                        })

        if not article_concepts:
            continue

        # Extract references with year data
        refs = a.get("referenced_works_detailed") or []
        for ref in refs:
            # Build a unique key for dedup
            doi = ref.get("doi", "")
            ref_key = doi or f"{ref.get('author', '')}_{ref.get('year', '')}_{ref.get('first_page', '')}"
            if not ref_key or ref_key in seen_refs:
                continue

            # Extract year
            year = ref.get("year")
            if isinstance(year, str) and year.strip():
                try:
                    year = int(float(year))
                except (ValueError, TypeError):
                    continue
            if not year or not isinstance(year, (int, float)):
                continue
            year = int(year)
            if year < MIN_YEAR or year > MAX_YEAR:
                continue

            seen_refs.add(ref_key)

            # Each ref inherits citing article's concepts
            for c in article_concepts:
                concept_freq[c["name"]] += 1

            first_author = ref.get("author", "Unknown")
            if isinstance(first_author, str) and " and " in first_author:
                first_author = first_author.split(" and ")[0]

            # Try to recover cited_by_count from OpenAlex cache via DOI
            cites = 0
            if doi:
                norm_doi = doi.replace("https://doi.org/", "").lower().strip()
                cites = _doi_to_cites.get(norm_doi, 0)

            records.append({
                "year": year,
                "title": ref.get("article_title", "") or ref.get("title", ""),
                "first_author": first_author or "Unknown",
                "cited_by_count": cites,
                "concepts": article_concepts,
                "openalex_id": "",
                "proxy_source": "article_level",
            })

        # Also use referenced_works_raw if detailed refs are sparse
        if not refs:
            for raw_ref in (a.get("referenced_works_raw") or [])[:50]:
                if isinstance(raw_ref, str):
                    # Try to extract year from raw reference string
                    import re
                    year_match = re.search(r'\b(19[5-9]\d|20[0-2]\d)\b', raw_ref)
                    if not year_match:
                        continue
                    year = int(year_match.group(1))
                    if year < MIN_YEAR or year > MAX_YEAR:
                        continue

                    ref_key = raw_ref[:80]
                    if ref_key in seen_refs:
                        continue
                    seen_refs.add(ref_key)

                    for c in article_concepts:
                        concept_freq[c["name"]] += 1

                    records.append({
                        "year": year,
                        "title": raw_ref[:120],
                        "first_author": raw_ref.split(",")[0][:40] if "," in raw_ref else "Unknown",
                        "cited_by_count": 0,
                        "concepts": article_concepts,
                        "openalex_id": "",
                        "proxy_source": "raw_reference",
                    })

    logger.info(
        f"Article-proxy fallback: {len(records)} refs with "
        f"{len(concept_freq)} unique concepts from article-level data"
    )
    return records, concept_freq


def _get_top_concepts(concept_freq: Counter) -> list[str]:
    """Get top N concepts that meet the minimum frequency threshold."""
    filtered = {k: v for k, v in concept_freq.items() if v >= MIN_CONCEPT_FREQ}
    return [name for name, _ in Counter(filtered).most_common(TOP_N_CONCEPTS)]


# ---------------------------------------------------------------------------
# Analysis 1: Enhanced RPYS
# ---------------------------------------------------------------------------

def compute_enhanced_rpys(
    records: list[dict],
    top_concepts: list[str],
    output_dir: Path,
    citas: Optional[dict] = None,
) -> dict:
    """RPYS estandar (Marx, Bornmann, Barth y Leydesdorff, 2014).

    - N_t = numero de CITAS a referencias publicadas en el ano t (no obras distintas).
    - Desviacion D_t = N_t - mediana(N_{t-2} .. N_{t+2}); en los bordes, la ventana disponible.
    - Sin suavizado: los picos de RPYS son de un solo ano y un filtro los aplana.
    - Picos = anos con D_t > 0, ordenados por D_t, cada uno con las obras que lo explican.
    - Los ultimos RPYS_ANOS_RECIENTES_INCOMPLETOS anos se marcan como ventana incompleta.

    `citas` es la salida de _count_reference_citations. Si falta (corpus sin references_resolved),
    se usan las obras distintas de `records` y se declara en el resultado.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if citas and citas.get("by_year"):
        by_year = citas["by_year"]
        unidad = "citas"
    else:
        by_year = Counter(rec["year"] for rec in records)
        unidad = "obras distintas (sin references_resolved: no hay conteo de citas)"
        logger.warning("RPYS: sin conteo de citas; se usan obras distintas. Los picos historicos pueden no aparecer.")
    if not by_year:
        return {"status": "no_data"}

    distintas = Counter(rec["year"] for rec in records)
    min_yr, max_yr = min(by_year), max(by_year)
    all_years = list(range(min_yr, max_yr + 1))
    counts = np.array([by_year.get(y, 0) for y in all_years], dtype=float)
    mitad = RPYS_VENTANA_MEDIANA // 2
    mediana = np.array([
        np.median(counts[max(0, i - mitad): i + mitad + 1]) for i in range(len(all_years))
    ])
    deviation = counts - mediana
    corte_reciente = max_yr - RPYS_ANOS_RECIENTES_INCOMPLETOS + 1

    picos = []
    orden = sorted(
        (i for i in range(len(all_years)) if deviation[i] > 0),
        key=lambda i: (-deviation[i], all_years[i]),
    )
    for i in orden:
        y = all_years[i]
        obras = []
        if citas and y in citas.get("works_by_year", {}):
            for key, n in sorted(citas["works_by_year"][y].items(), key=lambda kv: (-kv[1], kv[0]))[:RPYS_N_OBRAS_POR_PICO]:
                lab = citas["labels"].get(key, {})
                obras.append({"first_author": lab.get("first_author", ""), "title": lab.get("title", ""),
                              "openalex_id": lab.get("openalex_id", ""), "citas_en_el_corpus": int(n)})
        principal = obras[0]["citas_en_el_corpus"] if obras else 0
        picos.append({
            "year": y, "citas": int(counts[i]), "mediana_5": float(mediana[i]),
            "desviacion": float(deviation[i]), "ventana_incompleta": y >= corte_reciente,
            # Proporcion de las citas del ano que reune la obra mas citada: alta = el pico lo
            # produce una obra (raiz historica); baja = pico de volumen, sin obra que lo explique.
            "concentracion_obra_principal": round(principal / counts[i], 3) if counts[i] else 0.0,
            "obras_que_lo_explican": obras,
        })

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=(
            f"Citas por año de publicación de la referencia (N = {int(counts.sum())} {unidad.split(' (')[0]})",
            f"Desviación respecto de la mediana de {RPYS_VENTANA_MEDIANA} años (t−2 … t+2)",
        ),
        vertical_spacing=0.08, row_heights=[0.6, 0.4],
    )
    year_labels = [str(y) for y in all_years]
    fig.add_trace(go.Bar(
        x=year_labels, y=counts.tolist(), name=unidad.split(" (")[0].capitalize(),
        marker_color=["#bab0ac" if y >= corte_reciente else "#4e79a7" for y in all_years],
        hovertemplate="Año %{x}: %{y} " + unidad.split(" (")[0] + "<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=year_labels, y=mediana.tolist(), mode="lines", name=f"Mediana {RPYS_VENTANA_MEDIANA} años",
        line=dict(color="black", width=1.5, dash="dot"),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=year_labels, y=[distintas.get(y, 0) for y in all_years], mode="lines",
        name="Obras distintas", line=dict(color="#f28e2b", width=1),
    ), row=1, col=1)
    dev_colors = [
        "#bab0ac" if y >= corte_reciente else ("#e15759" if d > 0 else "#76b7b2")
        for y, d in zip(all_years, deviation)
    ]
    fig.add_trace(go.Bar(
        x=year_labels, y=deviation.tolist(), marker_color=dev_colors, name="Desviación",
        hovertemplate="Año %{x}: desviación %{y:.0f}<extra></extra>", showlegend=False,
    ), row=2, col=1)
    for p in picos[:10]:
        if p["ventana_incompleta"] or not p["obras_que_lo_explican"]:
            continue
        o = p["obras_que_lo_explican"][0]
        fig.add_annotation(
            x=str(p["year"]), y=p["desviacion"], row=2, col=1, showarrow=True, arrowhead=1,
            text=f"{o['first_author']} {p['year']}", font=dict(size=9), ay=-25,
        )
    fig.update_layout(
        title="Reference Publication Year Spectroscopy (RPYS)",
        width=1800, height=1000, paper_bgcolor="white", showlegend=True,
        legend=dict(font=dict(size=10), orientation="h", y=-0.12),
        annotations=list(fig.layout.annotations) + [dict(
            text=f"En gris: últimos {RPYS_ANOS_RECIENTES_INCOMPLETOS} años (ventana de citación incompleta)",
            xref="paper", yref="paper", x=1, y=1.06, showarrow=False, font=dict(size=10, color="#666"),
        )],
    )
    fig.update_xaxes(title_text="Año de publicación de la referencia", row=2, col=1)
    fig.update_yaxes(title_text=unidad.split(" (")[0].capitalize(), row=1, col=1)
    fig.update_yaxes(title_text="Desviación", row=2, col=1)
    _save_plotly_figure(fig, output_dir, "rpys_enhanced")

    return {
        "metodo": "RPYS estandar (Marx et al., 2014): citas por ano, desviacion respecto de la "
                  f"mediana de {RPYS_VENTANA_MEDIANA} anos centrada, sin suavizado",
        "unidad": unidad,
        "total_refs": int(counts.sum()),
        "obras_distintas": int(sum(distintas.values())),
        "year_range": [min_yr, max_yr],
        "anos_ventana_incompleta": [y for y in all_years if y >= corte_reciente],
        "peak_years": [p["year"] for p in picos if not p["ventana_incompleta"]][:RPYS_N_PICOS_REPORTE],
        "picos": picos[: RPYS_N_PICOS_REPORTE + RPYS_ANOS_RECIENTES_INCOMPLETOS],
        "series": [
            {"year": y, "citas": int(c), "mediana_5": float(m), "desviacion": float(d),
             "obras_distintas": int(distintas.get(y, 0)), "ventana_incompleta": y >= corte_reciente}
            for y, c, m, d in zip(all_years, counts, mediana, deviation)
        ],
    }


# ---------------------------------------------------------------------------
# Analysis 2: Concept Evolution Heatmap
# ---------------------------------------------------------------------------

def compute_concept_evolution_heatmap(
    records: list[dict],
    top_concepts: list[str],
    output_dir: Path,
) -> dict:
    """Top N concepts x reference years frequency heatmap."""
    import plotly.graph_objects as go

    # Build matrix: concept x year
    year_concept: dict[str, Counter] = {cn: Counter() for cn in top_concepts}
    for rec in records:
        y = rec["year"]
        for c in rec["concepts"]:
            if c["name"] in year_concept:
                year_concept[c["name"]][y] += 1

    # Determine year range from data
    all_years_set: set[int] = set()
    for cn in top_concepts:
        all_years_set.update(year_concept[cn].keys())
    if not all_years_set:
        return {"status": "no_data"}

    min_yr = min(all_years_set)
    max_yr = max(all_years_set)
    all_years = list(range(min_yr, max_yr + 1))

    z_matrix = []
    for cn in top_concepts:
        row = [year_concept[cn].get(y, 0) for y in all_years]
        z_matrix.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=z_matrix,
        x=[str(y) for y in all_years],
        y=top_concepts,
        colorscale="YlOrRd",
        hovertemplate="Concept: %{y}<br>Year: %{x}<br>Frequency: %{z}<extra></extra>",
    ))

    fig.update_layout(
        title="Concept Evolution Heatmap (reference years)",
        xaxis_title="Reference Publication Year",
        yaxis_title="Concept",
        width=1800, height=max(600, len(top_concepts) * 25 + 200),
        paper_bgcolor="white",
        yaxis=dict(autorange="reversed"),
    )

    _save_plotly_figure(fig, output_dir, "concept_evolution_heatmap")

    return {
        "n_concepts": len(top_concepts),
        "year_range": [min_yr, max_yr],
    }


# ---------------------------------------------------------------------------
# Analysis 3: Concept Lifecycle
# ---------------------------------------------------------------------------

def compute_concept_lifecycle(
    records: list[dict],
    top_concepts: list[str],
    output_dir: Path,
) -> list[dict]:
    """Timeline per concept: emergence, growth, peak, decline phases."""
    import plotly.graph_objects as go

    # Build year series per concept
    concept_years: dict[str, Counter] = {cn: Counter() for cn in top_concepts}
    for rec in records:
        y = rec["year"]
        for c in rec["concepts"]:
            if c["name"] in concept_years:
                concept_years[c["name"]][y] += 1

    lifecycles = []
    fig = go.Figure()

    colors = [
        "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
        "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
        "#86bcb6", "#8cd17d", "#b6992d", "#499894", "#d37295",
    ]

    for i, cn in enumerate(top_concepts[:15]):  # limit to top 15 for readability
        yc = concept_years[cn]
        if not yc:
            continue

        years_sorted = sorted(yc.keys())
        counts = [yc[y] for y in years_sorted]

        # Determine lifecycle phases
        emergence_year = years_sorted[0]
        peak_year = years_sorted[counts.index(max(counts))]
        total = sum(counts)

        # Determine phase: cumulative 10% = emergence, 10-50% = growth, 50-90% = peak, >90% = decline
        cumsum = np.cumsum(counts)
        pct = cumsum / total if total > 0 else cumsum

        growth_start = emergence_year
        peak_start = emergence_year
        decline_start = years_sorted[-1]

        for j, p in enumerate(pct):
            if p >= 0.1 and growth_start == emergence_year and j > 0:
                growth_start = years_sorted[j]
            if p >= 0.5 and peak_start == emergence_year:
                peak_start = years_sorted[j]
            if p >= 0.9:
                decline_start = years_sorted[j]
                break

        if peak_year >= years_sorted[-1] - 2:
            stage_label = f"Active Frontier / Peak ({peak_year})"
        elif years_sorted[-1] >= decline_start and peak_year < decline_start - 2:
            stage_label = f"Consolidated / Post-Peak (peak {peak_year})"
        else:
            stage_label = f"Sustained Growth (peak {peak_year})"

        lifecycle = {
            "concept": cn,
            "stage": stage_label,
            "emergence_year": emergence_year,
            "growth_year": growth_start,
            "peak_year": peak_year,
            "decline_year": decline_start,
            "last_year": years_sorted[-1],
            "total_refs": total,
            "max_annual": max(counts),
        }
        lifecycles.append(lifecycle)

        # Plot line
        fig.add_trace(go.Scatter(
            x=years_sorted, y=counts,
            mode="lines+markers",
            name=cn,
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=4),
            hovertemplate=f"{cn}<br>Year: %{{x}}<br>Refs: %{{y}}<extra></extra>",
        ))

    fig.update_layout(
        title="Concept Lifecycle: Temporal Evolution of Key Concepts",
        xaxis_title="Reference Publication Year",
        yaxis_title="Number of References",
        width=1600, height=800, paper_bgcolor="white",
        showlegend=True,
        legend=dict(font=dict(size=9)),
    )

    _save_plotly_figure(fig, output_dir, "concept_lifecycle")

    return lifecycles


# ---------------------------------------------------------------------------
# Analysis 4: Seminal Works
# ---------------------------------------------------------------------------

def compute_seminal_works(
    records: list[dict],
    top_concepts: list[str],
    output_dir: Path,
) -> dict:
    """Obras mas citadas POR EL CORPUS (citas locales) por concepto y decada.

    Se ordena por `local_citations` (cuantos articulos del corpus citan la obra), como en RPYS
    y CiteSpace; las citas globales de OpenAlex (`cited_by_count`) solo desempatan. Antes se
    ordenaba por citas globales: una obra muy citada en el mundo pero citada una vez aqui
    aparecia como 'seminal'.
    """
    import plotly.graph_objects as go

    # Group records by concept and decade
    concept_decade_works: dict[str, dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for rec in records:
        y = rec["year"]
        decade = f"{(y // 10) * 10}s"
        for c in rec["concepts"]:
            if c["name"] in top_concepts:
                concept_decade_works[c["name"]][decade].append(rec)

    # Build table data: top 3 per concept per decade
    table_rows = []
    for cn in top_concepts:
        decades = concept_decade_works.get(cn, {})
        for decade in sorted(decades.keys()):
            works = decades[decade]
            works_sorted = sorted(
                works,
                key=lambda w: (-w.get("local_citations", 0), -(w.get("cited_by_count", 0) or 0), w.get("openalex_id", "")),
            )
            for w in works_sorted[:3]:
                table_rows.append({
                    "concept": cn,
                    "decade": decade,
                    "year": w["year"],
                    "first_author": w["first_author"],
                    "title": (w["title"] or "")[:80],
                    "local_citations": w.get("local_citations", 0),
                    "cited_by_count": w.get("cited_by_count", 0),
                    "openalex_id": w.get("openalex_id", ""),
                })

    if not table_rows:
        return {"status": "no_data", "rows": []}

    # Build Plotly table
    header_vals = ["Concept", "Decade", "Year", "Author", "Title", "Citas en el corpus", "Citas globales (OpenAlex)"]
    cell_vals = [
        [r["concept"] for r in table_rows],
        [r["decade"] for r in table_rows],
        [r["year"] for r in table_rows],
        [r["first_author"] for r in table_rows],
        [r["title"] for r in table_rows],
        [r["local_citations"] for r in table_rows],
        [r["cited_by_count"] for r in table_rows],
    ]

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=header_vals,
            fill_color="#4e79a7",
            font=dict(color="white", size=12),
            align="left",
        ),
        cells=dict(
            values=cell_vals,
            fill_color=[
                ["#f0f0f0" if i % 2 == 0 else "white" for i in range(len(table_rows))]
            ] * len(header_vals),
            font=dict(size=11),
            align="left",
            height=28,
        ),
    )])

    fig.update_layout(
        title="Seminal Works: References Most Cited by the Corpus, per Concept and Decade",
        width=1600, height=max(600, len(table_rows) * 28 + 200),
        paper_bgcolor="white",
    )

    _save_plotly_figure(fig, output_dir, "seminal_works_table")

    # Ranking global por citas locales (una fila por obra), independiente de conceptos y decadas
    unicos = {}
    for rec in records:
        k = rec.get("openalex_id") or f"{rec.get('first_author')}_{rec.get('year')}_{str(rec.get('title', ''))[:60]}"
        unicos.setdefault(k, rec)
    top_global = [
        {"year": r["year"], "first_author": r["first_author"], "title": (r["title"] or "")[:100],
         "local_citations": r.get("local_citations", 0), "cited_by_count": r.get("cited_by_count", 0),
         "openalex_id": r.get("openalex_id", "")}
        for r in sorted(unicos.values(),
                        key=lambda w: (-w.get("local_citations", 0), -(w.get("cited_by_count", 0) or 0), w.get("openalex_id", "")))[:20]
    ]

    return {
        "n_rows": len(table_rows),
        "n_concepts": len(set(r["concept"] for r in table_rows)),
        "orden": "citas locales (corpus); desempate por citas globales de OpenAlex",
        "rows": table_rows,
        "top_global": top_global,
    }


# ---------------------------------------------------------------------------
# Analysis 5: Concept Co-occurrence
# ---------------------------------------------------------------------------

def compute_concept_cooccurrence(
    records: list[dict],
    top_concepts: list[str],
    output_dir: Path,
) -> dict:
    """How concepts co-occur within same references over time."""
    import plotly.graph_objects as go

    # Build co-occurrence matrix
    top_set = set(top_concepts)
    cooccur: Counter = Counter()
    n_refs_with_multi = 0

    for rec in records:
        ref_concepts = [
            c["name"] for c in rec["concepts"] if c["name"] in top_set
        ]
        ref_concepts = list(set(ref_concepts))  # deduplicate within ref
        if len(ref_concepts) >= 2:
            n_refs_with_multi += 1
            for pair in combinations(sorted(ref_concepts), 2):
                cooccur[pair] += 1

    if not cooccur:
        return {"status": "no_data"}

    # Build matrix for top concepts
    n = len(top_concepts)
    matrix = np.zeros((n, n), dtype=int)
    idx = {cn: i for i, cn in enumerate(top_concepts)}

    for (a, b), count in cooccur.items():
        if a in idx and b in idx:
            matrix[idx[a]][idx[b]] = count
            matrix[idx[b]][idx[a]] = count

    # Diagonal = self-frequency (total refs per concept)
    concept_totals: Counter = Counter()
    for rec in records:
        for c in rec["concepts"]:
            if c["name"] in top_set:
                concept_totals[c["name"]] += 1
    for cn in top_concepts:
        if cn in idx:
            matrix[idx[cn]][idx[cn]] = concept_totals.get(cn, 0)

    fig = go.Figure(data=go.Heatmap(
        z=matrix.tolist(),
        x=top_concepts,
        y=top_concepts,
        colorscale="Blues",
        hovertemplate="Row: %{y}<br>Col: %{x}<br>Co-occurrences: %{z}<extra></extra>",
    ))

    fig.update_layout(
        title="Concept Co-occurrence in Referenced Works",
        width=1200, height=1000, paper_bgcolor="white",
        xaxis=dict(tickangle=45, tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=9), autorange="reversed"),
    )

    _save_plotly_figure(fig, output_dir, "concept_cooccurrence")

    # Top pairs
    top_pairs = cooccur.most_common(20)
    return {
        "n_refs_with_multiple_concepts": n_refs_with_multi,
        "n_unique_pairs": len(cooccur),
        "top_pairs": [
            {"concepts": list(pair), "count": count}
            for pair, count in top_pairs
        ],
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _generate_report(
    records: list[dict],
    top_concepts: list[str],
    concept_freq: Counter,
    rpys_result: dict,
    heatmap_result: dict,
    lifecycles: list[dict],
    seminal_result: dict,
    cooccur_result: dict,
    output_dir: Path,
) -> None:
    """Generate spectroscopy_report.md summary."""
    lines = [
        "# Reference Spectroscopy Report",
        "",
        "## Overview",
        f"- **References analyzed**: {len(records)}",
        f"- **Unique concepts extracted**: {len(concept_freq)}",
        f"- **Top concepts (freq >= {MIN_CONCEPT_FREQ})**: {len(top_concepts)}",
        f"- **Concept score threshold**: >= {MIN_CONCEPT_SCORE}",
        f"- **Concept level**: 0-{CONCEPT_LEVEL_MAX}",
        "",
    ]

    # RPYS summary
    lines.append("## 1. RPYS")
    if rpys_result.get("status") != "no_data":
        lines.append(f"- Método: {rpys_result.get('metodo', '')}")
        lines.append(f"- Rango de años de las referencias: {rpys_result.get('year_range', [])}")
        lines.append(f"- Total: {rpys_result.get('total_refs', 0)} {rpys_result.get('unidad', '')}; "
                     f"{rpys_result.get('obras_distintas', 0)} obras distintas")
        inc = rpys_result.get("anos_ventana_incompleta", [])
        if inc:
            lines.append(f"- Ventana de citación incompleta (no interpretar como picos): {', '.join(map(str, inc))}")
        picos = rpys_result.get("picos", [])
        if picos:
            lines.append("")
            lines.append("Concentración = proporción de las citas del año que reúne la obra principal. "
                         "Alta: el pico lo produce una obra (raíz histórica). Baja: pico de volumen.")
            lines.append("")
            lines.append("| Año | Citas | Mediana 5 años | Desviación | Concentración | Obra principal (citas en el corpus) |")
            lines.append("|-----|-------|----------------|------------|---------------|-------------------------------------|")
            for p in picos:
                o = (p.get("obras_que_lo_explican") or [{}])[0]
                marca = " *(ventana incompleta)*" if p.get("ventana_incompleta") else ""
                lines.append(
                    f"| {p['year']}{marca} | {p['citas']} | {p['mediana_5']:.0f} | +{p['desviacion']:.0f} | "
                    f"{p.get('concentracion_obra_principal', 0):.0%} | "
                    f"{o.get('first_author', '')} — {str(o.get('title', '')).replace('|', '/')[:70]} ({o.get('citas_en_el_corpus', 0)}) |"
                )
    else:
        lines.append("- *Insufficient data*")
    lines.append("")

    # Heatmap summary
    lines.append("## 2. Concept Evolution Heatmap")
    lines.append(f"- Concepts visualized: {heatmap_result.get('n_concepts', 0)}")
    lines.append(f"- Year range: {heatmap_result.get('year_range', [])}")
    lines.append("")

    # Lifecycle summary
    lines.append("## 3. Concept Lifecycle")
    lines.append("")
    if lifecycles:
        lines.append("| Concept | Emergence | Peak Year | Total Refs | Max Annual |")
        lines.append("|---------|-----------|-----------|------------|------------|")
        for lc in lifecycles[:20]:
            lines.append(
                f"| {lc['concept']} | {lc['emergence_year']} | "
                f"{lc['peak_year']} | {lc['total_refs']} | {lc['max_annual']} |"
            )
    lines.append("")

    # Seminal works summary
    lines.append("## 4. Seminal Works")
    n_sem = seminal_result.get("n_rows", 0)
    lines.append(f"- Entries in table: {n_sem}")
    lines.append(f"- Concepts covered: {seminal_result.get('n_concepts', 0)}")
    lines.append(f"- Orden: {seminal_result.get('orden', '')}")
    top = seminal_result.get("top_global", [])
    if top:
        lines.append("")
        lines.append("| Obra | Año | Citas en el corpus | Citas globales (OpenAlex) |")
        lines.append("|------|-----|--------------------|---------------------------|")
        for w in top[:15]:
            lines.append(f"| {w['first_author']} — {str(w['title']).replace('|', '/')[:70]} | {w['year']} | "
                         f"{w['local_citations']} | {w['cited_by_count']} |")
    lines.append("")

    # Co-occurrence summary
    lines.append("## 5. Concept Co-occurrence")
    if cooccur_result.get("status") != "no_data":
        lines.append(
            f"- References with multiple concepts: "
            f"{cooccur_result.get('n_refs_with_multiple_concepts', 0)}"
        )
        lines.append(
            f"- Unique concept pairs: {cooccur_result.get('n_unique_pairs', 0)}"
        )
        top_pairs = cooccur_result.get("top_pairs", [])
        if top_pairs:
            lines.append("")
            lines.append("### Top co-occurring concept pairs")
            lines.append("| Concept A | Concept B | Count |")
            lines.append("|-----------|-----------|-------|")
            for p in top_pairs[:15]:
                a, b = p["concepts"]
                lines.append(f"| {a} | {b} | {p['count']} |")
    else:
        lines.append("- *Insufficient data for co-occurrence analysis*")
    lines.append("")

    # Top concepts list
    lines.append("## Top Concepts by Frequency")
    lines.append("")
    lines.append("| Rank | Concept | Frequency |")
    lines.append("|------|---------|-----------|")
    for i, cn in enumerate(top_concepts, 1):
        lines.append(f"| {i} | {cn} | {concept_freq[cn]} |")
    lines.append("")

    report_path = output_dir / "spectroscopy_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Saved: {report_path}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_phase2g(articles: list[dict]) -> dict:
    """Phase 2g: Reference Spectroscopy.

    1. Extract reference-concept data (from references_resolved or cache fallback)
    2. Filter top concepts
    3. Run 5 analyses with visualizations
    4. Generate report + data JSON
    """
    logger.info("=" * 60)
    logger.info("Phase 2g: Reference Spectroscopy")
    logger.info("=" * 60)

    output_dir = SPECTROSCOPY_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Extract data — try references_resolved first, cache, then article-level proxy
    has_resolved = any(
        a.get("references_resolved") for a in articles
    )
    if has_resolved:
        logger.info("Using references_resolved from Phase 1i")
        records, concept_freq = _build_ref_concept_data(articles)
    else:
        # Enrich cache if refs are missing cited_by_count/concepts
        _enrich_cache_if_needed(articles)
        logger.info("references_resolved not found — falling back to cache")
        records, concept_freq = _build_ref_concept_data_from_cache(articles)

    # Fallback: use article-level concepts/keywords as proxy for reference themes
    if len(records) < 10:
        logger.info("Insufficient ref-level concepts — using article-level proxy")
        records, concept_freq = _build_ref_data_from_article_proxy(articles)

    logger.info(f"Extracted {len(records)} refs with concepts, "
                f"{len(concept_freq)} unique concepts")

    if len(records) < 10:
        logger.warning("Not enough reference data for spectroscopy analysis")
        return {"status": "insufficient_data", "n_refs_analyzed": len(records)}

    # 2. Top concepts
    top_concepts = _get_top_concepts(concept_freq)
    logger.info(f"Top concepts (freq >= {MIN_CONCEPT_FREQ}): {len(top_concepts)}")

    if not top_concepts:
        logger.warning("No concepts meet minimum frequency threshold")
        return {
            "status": "no_frequent_concepts",
            "n_refs_analyzed": len(records),
            "n_concepts": len(concept_freq),
        }

    # 3. Run analyses
    logger.info("Analysis 1/5: Enhanced RPYS...")
    citas = _count_reference_citations(articles) if has_resolved else None
    if citas:
        for rec in records:
            rec["local_citations"] = int(citas["by_work"].get(_ref_key(rec), 0))
        logger.info(
            f"RPYS: {sum(citas['by_year'].values())} citas a {len(citas['by_work'])} obras distintas"
        )
    rpys_result = compute_enhanced_rpys(records, top_concepts, output_dir, citas=citas)

    logger.info("Analysis 2/5: Concept Evolution Heatmap...")
    heatmap_result = compute_concept_evolution_heatmap(
        records, top_concepts, output_dir
    )

    logger.info("Analysis 3/5: Concept Lifecycle...")
    lifecycles = compute_concept_lifecycle(records, top_concepts, output_dir)

    logger.info("Analysis 4/5: Seminal Works...")
    seminal_result = compute_seminal_works(records, top_concepts, output_dir)

    logger.info("Analysis 5/5: Concept Co-occurrence...")
    cooccur_result = compute_concept_cooccurrence(
        records, top_concepts, output_dir
    )

    # 4. Report
    _generate_report(
        records, top_concepts, concept_freq,
        rpys_result, heatmap_result, lifecycles,
        seminal_result, cooccur_result, output_dir,
    )

    # 5. Data JSON
    data_export = {
        "n_refs_analyzed": len(records),
        "n_concepts": len(concept_freq),
        "n_top_concepts": len(top_concepts),
        "top_concepts": [
            {"name": cn, "frequency": concept_freq[cn]}
            for cn in top_concepts
        ],
        "rpys": rpys_result,
        "heatmap": {k: v for k, v in heatmap_result.items() if k != "status"},
        "lifecycles": lifecycles,
        "seminal_works": {
            "n_rows": seminal_result.get("n_rows", 0),
            "orden": seminal_result.get("orden", ""),
            "rows": seminal_result.get("rows", []),
            "top_global": seminal_result.get("top_global", []),
        },
        "cooccurrence": {
            k: v for k, v in cooccur_result.items() if k != "status"
        },
    }
    save_json(data_export, output_dir / "spectroscopy_data.json")

    result = {
        "status": "ok",
        "n_refs_analyzed": len(records),
        "n_concepts": len(top_concepts),
        "output_dir": str(output_dir),
    }
    logger.info(f"Phase 2g complete — {len(records)} refs, {len(top_concepts)} concepts")
    return result

run_phase2h = run_phase2g
