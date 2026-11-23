"""Phase 5: Impact Evolution Analysis.

Eight analyses:
1. FWCI (Field-Weighted Citation Impact) — distribution, by topic, temporal trend
2. Price Index (Knowledge Base Age) — age of cited references
3. Concept Evolution / Lifecycle — emerging/foundational/declining/active concepts
4. Topic Flow Sankey — subfield and domain flows across temporal periods
5. Beauty Coefficient — Sleeping Beauty detection via citation trajectory analysis
6. CSS Classification — Characteristic Scales and Scores (iterative truncation)
7. Outgrow Index (CR Index) — innovation disruption measurement
8. Citation Genealogy H-1 — backward citation genealogy (intellectual roots)

Plus: impact_evolution_report.md and impact_evolution_data.json
"""
from __future__ import annotations

import json
import logging
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

from config import (
    IMPACT_EVOLUTION_DIR,
    OPENALEX_CACHE_DIR,
)
from src.utils import DiskCache, normalize_doi, save_json, is_noisy_legacy_concept

logger = logging.getLogger("bibliometrico.impact_evolution")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MIN_YEAR = 1950
MAX_YEAR = 2026
TOP_N_TOPICS = 20
TOP_N_CONCEPTS = 25
MIN_CONCEPT_FREQ = 3


# ---------------------------------------------------------------------------
# Helpers
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


def _shannon_diversity(counts: list[int]) -> float:
    """Compute Shannon diversity index H' from a list of counts."""
    total = sum(counts)
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            h -= p * math.log(p)
    return round(h, 4)


def _compute_temporal_periods(years: list[int], n_periods: int = 3) -> list[tuple[int, int]]:
    """Auto-divide a list of years into n temporal periods.

    Returns list of (start_year, end_year) tuples.
    """
    if not years:
        return []
    min_y = min(years)
    max_y = max(years)
    span = max_y - min_y + 1
    if span < n_periods:
        return [(min_y, max_y)]

    step = span / n_periods
    periods = []
    for i in range(n_periods):
        start = min_y + int(i * step)
        end = min_y + int((i + 1) * step) - 1 if i < n_periods - 1 else max_y
        periods.append((start, end))
    return periods


def _lookup_fwci_from_cache(
    articles: list[dict],
) -> list[dict]:
    """Look up FWCI and citation percentile from enriched articles + OpenAlex cache.

    Priority:
    1. article["openalex"]["cited_by_percentile_year"] (from Phase 1a enrichment)
    2. Raw OpenAlex cache (fallback for FWCI not in enriched data)

    Returns list of dicts with keys: doi, key, year, fwci, percentile,
    primary_topic, cited_by_count, apc_paid_usd.
    """
    cache = DiskCache(OPENALEX_CACHE_DIR)
    results = []

    for a in articles:
        doi = a.get("doi", "")
        oa = a.get("openalex", {}) or {}
        year = a.get("publication_year")
        key = a.get("key", "")

        fwci = None
        percentile = None
        primary_topic = ""

        # --- Source 1: enriched article data ---
        perc_data = oa.get("cited_by_percentile_year", {})
        if isinstance(perc_data, dict):
            # OpenAlex returns {min: X, max: Y} for percentile
            percentile = perc_data.get("min") or perc_data.get("value")
        elif isinstance(perc_data, (int, float)):
            percentile = perc_data

        # Primary topic from enrichment
        pt = oa.get("primary_topic", {}) or {}
        if pt.get("name"):
            primary_topic = pt["name"]

        # --- Source 2: raw OpenAlex cache (for FWCI) ---
        if doi:
            ndoi = normalize_doi(doi)
            if ndoi:
                cached = cache.get(f"doi:{ndoi}")
                if cached and isinstance(cached, dict):
                    if fwci is None:
                        fwci = cached.get("fwci")
                    if percentile is None:
                        cnp = cached.get("citation_normalized_percentile")
                        if isinstance(cnp, dict):
                            percentile = cnp.get("value")
                        elif isinstance(cnp, (int, float)):
                            percentile = cnp
                    if not primary_topic:
                        topics = cached.get("topics") or []
                        if topics:
                            primary_topic = topics[0].get("display_name", "")
                    if year is None:
                        year = cached.get("publication_year")

        if fwci is None and percentile is None:
            continue

        # APC data (for cost-benefit analysis)
        apc = oa.get("apc_paid", {}) or {}
        apc_usd = apc.get("value_usd", 0) if isinstance(apc, dict) else 0

        results.append({
            "doi": normalize_doi(doi) if doi else "",
            "key": key,
            "year": year,
            "fwci": fwci,
            "percentile": percentile,
            "primary_topic": primary_topic,
            "cited_by_count": a.get("cited_by_count", 0),
            "apc_paid_usd": apc_usd,
        })

    return results


# ---------------------------------------------------------------------------
# Analysis 1: FWCI
# ---------------------------------------------------------------------------

def compute_fwci_analysis(
    articles: list[dict],
    output_dir: Path,
) -> dict:
    """FWCI distribution, by primary topic, and temporal trend."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    logger.info("Analysis 1/8: FWCI (Field-Weighted Citation Impact)...")

    records = _lookup_fwci_from_cache(articles)
    if not records:
        logger.warning("No FWCI data found in cache")
        return {"status": "no_data"}

    fwci_vals = [r["fwci"] for r in records if r["fwci"] is not None]
    pct_vals = [r["percentile"] for r in records if r["percentile"] is not None]

    result: dict[str, Any] = {
        "n_articles_with_fwci": len(fwci_vals),
        "n_articles_with_percentile": len(pct_vals),
    }

    # --- Chart 1: FWCI histogram ---
    if fwci_vals:
        fwci_arr = np.array(fwci_vals, dtype=float)
        result["fwci_mean"] = round(float(np.mean(fwci_arr)), 3)
        result["fwci_median"] = round(float(np.median(fwci_arr)), 3)
        result["fwci_std"] = round(float(np.std(fwci_arr)), 3)
        result["pct_above_1"] = round(
            float(np.sum(fwci_arr > 1.0) / len(fwci_arr) * 100), 1
        )
        result["pct_top_10"] = round(
            float(np.sum(fwci_arr >= np.percentile(fwci_arr, 90)) / len(fwci_arr) * 100), 1
        )
        result["pct_top_1"] = round(
            float(np.sum(fwci_arr >= np.percentile(fwci_arr, 99)) / len(fwci_arr) * 100), 1
        )

        fig1 = go.Figure()
        fig1.add_trace(go.Histogram(
            x=fwci_vals,
            nbinsx=40,
            marker_color="#4e79a7",
            hovertemplate="FWCI: %{x:.2f}<br>Count: %{y}<extra></extra>",
        ))
        fig1.add_vline(x=1.0, line_dash="dash", line_color="red",
                       annotation_text="World avg (1.0)")
        fig1.add_vline(x=result["fwci_median"], line_dash="dot",
                       line_color="green",
                       annotation_text=f"Median ({result['fwci_median']:.2f})")
        fig1.update_layout(
            title=f"FWCI Distribution (n={len(fwci_vals)}, mean={result['fwci_mean']:.2f})",
            xaxis_title="Field-Weighted Citation Impact",
            yaxis_title="Number of Articles",
            width=1200, height=600, paper_bgcolor="white",
        )
        _save_plotly_figure(fig1, output_dir, "fwci_histogram")
    else:
        logger.warning("No valid FWCI values for histogram")

    # --- Chart 2: FWCI by primary topic (bar) ---
    topic_fwci: dict[str, list[float]] = defaultdict(list)
    for r in records:
        if r["fwci"] is not None and r["primary_topic"]:
            topic_fwci[r["primary_topic"]].append(r["fwci"])

    if topic_fwci:
        # Sort by median FWCI, take top N
        topic_stats = []
        for topic, vals in topic_fwci.items():
            arr = np.array(vals, dtype=float)
            topic_stats.append({
                "topic": topic,
                "median": float(np.median(arr)),
                "mean": float(np.mean(arr)),
                "n": len(vals),
            })
        topic_stats.sort(key=lambda x: x["median"], reverse=True)
        top_topics = topic_stats[:TOP_N_TOPICS]
        result["fwci_by_topic"] = top_topics

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=[t["topic"] for t in top_topics],
            y=[t["median"] for t in top_topics],
            marker_color="#59a14f",
            text=[f"n={t['n']}" for t in top_topics],
            textposition="outside",
            hovertemplate="%{x}<br>Median FWCI: %{y:.2f}<extra></extra>",
        ))
        fig2.add_hline(y=1.0, line_dash="dash", line_color="red",
                       annotation_text="World avg")
        fig2.update_layout(
            title="Median FWCI by Primary Topic",
            xaxis_title="Primary Topic",
            yaxis_title="Median FWCI",
            width=1600, height=700, paper_bgcolor="white",
            xaxis=dict(tickangle=45, tickfont=dict(size=9)),
        )
        _save_plotly_figure(fig2, output_dir, "fwci_by_topic")

    # --- Chart 3: FWCI temporal trend ---
    year_fwci: dict[int, list[float]] = defaultdict(list)
    for r in records:
        if r["fwci"] is not None and r["year"] and isinstance(r["year"], int):
            year_fwci[r["year"]].append(r["fwci"])

    if year_fwci:
        years_sorted = sorted(year_fwci.keys())
        medians = [float(np.median(year_fwci[y])) for y in years_sorted]
        means = [float(np.mean(year_fwci[y])) for y in years_sorted]
        counts = [len(year_fwci[y]) for y in years_sorted]

        result["fwci_temporal"] = [
            {"year": y, "median": round(m, 3), "mean": round(mn, 3), "n": n}
            for y, m, mn, n in zip(years_sorted, medians, means, counts)
        ]

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=years_sorted, y=medians,
            mode="lines+markers", name="Median FWCI",
            line=dict(color="#4e79a7", width=2),
            hovertemplate="Year: %{x}<br>Median: %{y:.2f}<extra></extra>",
        ))
        fig3.add_trace(go.Scatter(
            x=years_sorted, y=means,
            mode="lines+markers", name="Mean FWCI",
            line=dict(color="#f28e2b", width=2, dash="dot"),
            hovertemplate="Year: %{x}<br>Mean: %{y:.2f}<extra></extra>",
        ))
        fig3.add_trace(go.Bar(
            x=years_sorted, y=counts,
            name="N articles", yaxis="y2",
            marker_color="rgba(200,200,200,0.5)",
            hovertemplate="Year: %{x}<br>N: %{y}<extra></extra>",
        ))
        fig3.add_hline(y=1.0, line_dash="dash", line_color="red")
        fig3.update_layout(
            title="FWCI Temporal Trend",
            xaxis_title="Publication Year",
            yaxis_title="FWCI",
            yaxis2=dict(title="N articles", overlaying="y", side="right"),
            width=1400, height=600, paper_bgcolor="white",
        )
        _save_plotly_figure(fig3, output_dir, "fwci_temporal")

    charts = sum(1 for name in ["fwci_histogram", "fwci_by_topic", "fwci_temporal"]
                 if (output_dir / f"{name}.html").exists())
    result["charts"] = charts
    return result


# ---------------------------------------------------------------------------
# Analysis 2: Price Index
# ---------------------------------------------------------------------------

def compute_price_index(
    articles: list[dict],
    output_dir: Path,
) -> dict:
    """Price Index and reference age analysis."""
    import plotly.graph_objects as go

    logger.info("Analysis 2/8: Price Index (Knowledge Base Age)...")

    records = []
    for a in articles:
        pub_year = a.get("publication_year")
        if not pub_year or not isinstance(pub_year, int):
            continue

        refs = a.get("references_resolved") or []
        if not refs:
            continue

        ref_years = []
        for ref in refs:
            ry = ref.get("year")
            if ry and isinstance(ry, int) and MIN_YEAR <= ry <= MAX_YEAR:
                ref_years.append(ry)

        if not ref_years:
            continue

        ages = [pub_year - ry for ry in ref_years]
        recent = sum(1 for ry in ref_years if pub_year - ry <= 5)
        price_index = round(recent / len(ref_years) * 100, 1) if ref_years else 0

        records.append({
            "doi": a.get("doi", ""),
            "year": pub_year,
            "n_refs": len(ref_years),
            "price_index": price_index,
            "median_age": float(np.median(ages)),
            "mean_age": round(float(np.mean(ages)), 1),
            "ref_years": ref_years,
        })

    if not records:
        logger.warning("No reference year data available for Price Index")
        return {"status": "no_data"}

    pi_vals = [r["price_index"] for r in records]
    age_vals = [r["median_age"] for r in records]

    result: dict[str, Any] = {
        "n_articles": len(records),
        "price_index_mean": round(float(np.mean(pi_vals)), 1),
        "price_index_median": round(float(np.median(pi_vals)), 1),
        "median_ref_age_mean": round(float(np.mean(age_vals)), 1),
        "median_ref_age_median": round(float(np.median(age_vals)), 1),
    }

    # --- Chart 1: Price Index distribution ---
    fig1 = go.Figure()
    fig1.add_trace(go.Histogram(
        x=pi_vals, nbinsx=30,
        marker_color="#e15759",
        hovertemplate="PI: %{x:.1f}%<br>Count: %{y}<extra></extra>",
    ))
    fig1.add_vline(x=result["price_index_median"], line_dash="dash",
                   line_color="blue",
                   annotation_text=f"Median ({result['price_index_median']:.1f}%)")
    fig1.update_layout(
        title=f"Price Index Distribution (n={len(records)})",
        xaxis_title="Price Index (% refs from last 5 years)",
        yaxis_title="Number of Articles",
        width=1200, height=600, paper_bgcolor="white",
    )
    _save_plotly_figure(fig1, output_dir, "price_index_distribution")

    # --- Chart 2: Reference age distribution (all refs pooled) ---
    all_ages = []
    for r in records:
        pub_y = r["year"]
        for ry in r["ref_years"]:
            all_ages.append(pub_y - ry)

    if all_ages:
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=all_ages, nbinsx=50,
            marker_color="#76b7b2",
            hovertemplate="Age: %{x} years<br>Count: %{y}<extra></extra>",
        ))
        median_all = float(np.median(all_ages))
        fig2.add_vline(x=median_all, line_dash="dash", line_color="red",
                       annotation_text=f"Median ({median_all:.0f} yrs)")
        fig2.update_layout(
            title=f"Reference Age Distribution (n={len(all_ages)} refs)",
            xaxis_title="Reference Age (years)",
            yaxis_title="Count",
            width=1200, height=600, paper_bgcolor="white",
        )
        _save_plotly_figure(fig2, output_dir, "reference_age_distribution")
        result["total_refs"] = len(all_ages)
        result["median_ref_age_overall"] = round(median_all, 1)

    # --- Chart 3: Price Index trend over time ---
    year_pi: dict[int, list[float]] = defaultdict(list)
    for r in records:
        year_pi[r["year"]].append(r["price_index"])

    if len(year_pi) >= 2:
        years_sorted = sorted(year_pi.keys())
        pi_medians = [float(np.median(year_pi[y])) for y in years_sorted]
        pi_counts = [len(year_pi[y]) for y in years_sorted]

        result["price_index_temporal"] = [
            {"year": y, "median_pi": round(m, 1), "n": n}
            for y, m, n in zip(years_sorted, pi_medians, pi_counts)
        ]

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=years_sorted, y=pi_medians,
            mode="lines+markers", name="Median PI",
            line=dict(color="#e15759", width=2),
            hovertemplate="Year: %{x}<br>Median PI: %{y:.1f}%<extra></extra>",
        ))
        fig3.add_trace(go.Bar(
            x=years_sorted, y=pi_counts,
            name="N articles", yaxis="y2",
            marker_color="rgba(200,200,200,0.5)",
        ))
        fig3.update_layout(
            title="Price Index Trend Over Time",
            xaxis_title="Publication Year",
            yaxis_title="Median Price Index (%)",
            yaxis2=dict(title="N articles", overlaying="y", side="right"),
            width=1400, height=600, paper_bgcolor="white",
        )
        _save_plotly_figure(fig3, output_dir, "price_index_trend")

    charts = sum(1 for name in ["price_index_distribution",
                                "reference_age_distribution", "price_index_trend"]
                 if (output_dir / f"{name}.html").exists())
    result["charts"] = charts
    return result


# ---------------------------------------------------------------------------
# Analysis 3: Concept Evolution / Lifecycle
# ---------------------------------------------------------------------------

def compute_concept_evolution(
    articles: list[dict],
    output_dir: Path,
) -> dict:
    """Concept evolution: frequency by year, lifecycle classification, diversity."""
    import plotly.graph_objects as go

    logger.info("Analysis 3/8: Concept Evolution / Lifecycle...")

    # Collect concepts from corpus (article-level) per year
    corpus_concepts_by_year: dict[int, Counter] = defaultdict(Counter)
    # Collect concepts from cited references per year
    refs_concepts_by_year: dict[int, Counter] = defaultdict(Counter)
    concept_freq: Counter = Counter()

    for a in articles:
        year = a.get("publication_year")
        if not year or not isinstance(year, int):
            continue

        # Corpus topics (CWTS Leiden OpenAlex topics — preferred over legacy concepts)
        topics = a.get("topics") or a.get("openalex", {}).get("topics", [])
        if topics:
            for t in topics:
                name = (t.get("display_name") or t.get("name") or "").strip()
                if name and not is_noisy_legacy_concept(name):
                    corpus_concepts_by_year[year][name] += 1
                    concept_freq[name] += 1
        else:
            # Fallback to legacy concepts filtered for noise
            for c in a.get("concepts") or []:
                name = (c.get("name") or c.get("display_name") or "").strip()
                if name and not is_noisy_legacy_concept(name):
                    corpus_concepts_by_year[year][name] += 1
                    concept_freq[name] += 1

        # OpenAlex keywords (author-provided)
        for kw_obj in a.get("openalex", {}).get("keywords", []):
            name = (kw_obj.get("keyword", "") if isinstance(kw_obj, dict) else "").strip()
            if name and not is_noisy_legacy_concept(name):
                corpus_concepts_by_year[year][name] += 1
                concept_freq[name] += 1

        # MeSH terms (controlled vocabulary)
        for mesh in a.get("mesh_terms", []):
            name = mesh.strip() if isinstance(mesh, str) else ""
            if name:
                corpus_concepts_by_year[year][name] += 1
                concept_freq[name] += 1

        # Primary topic subfield (mid-level classification)
        pt = a.get("openalex", {}).get("primary_topic", {}) or {}
        for field_name in ("subfield",):
            val = pt.get(field_name, "")
            if val and not is_noisy_legacy_concept(val):
                corpus_concepts_by_year[year][val] += 1
                concept_freq[val] += 1

        # Author keywords and keywords plus (article-level, semicolon-separated)
        for kw_field in ("author_keywords", "keywords_plus", "index_keywords"):
            kw_raw = a.get(kw_field, "") or ""
            if isinstance(kw_raw, str) and kw_raw.strip():
                for kw in kw_raw.split(";"):
                    kw = kw.strip()
                    if kw and len(kw) > 2:
                        corpus_concepts_by_year[year][kw] += 1
                        concept_freq[kw] += 1

        # Reference topics (preferred) or filtered legacy concepts
        for ref in a.get("references_resolved") or []:
            ref_topics = ref.get("topics") or []
            if ref_topics:
                for t in ref_topics:
                    name = (t.get("name") or t.get("display_name") or "").strip()
                    if name and not is_noisy_legacy_concept(name):
                        refs_concepts_by_year[year][name] += 1
            else:
                for c in ref.get("concepts") or []:
                    name = (c.get("name") or "").strip()
                    if name and not is_noisy_legacy_concept(name):
                        refs_concepts_by_year[year][name] += 1

    if not concept_freq:
        logger.warning("No concept data available")
        return {"status": "no_data"}

    # Top concepts by frequency
    top_concepts = [
        name for name, freq in concept_freq.most_common(TOP_N_CONCEPTS)
        if freq >= MIN_CONCEPT_FREQ
    ]

    if not top_concepts:
        logger.warning("No concepts meet minimum frequency threshold")
        return {"status": "no_frequent_concepts", "n_concepts": len(concept_freq)}

    result: dict[str, Any] = {
        "n_concepts_total": len(concept_freq),
        "n_top_concepts": len(top_concepts),
        "top_concepts": top_concepts,
    }

    # --- Build year x concept matrix for corpus ---
    all_years = sorted(corpus_concepts_by_year.keys())
    if not all_years:
        return {"status": "no_data"}

    # --- Chart 1: Concept x Year heatmap ---
    z_matrix = []
    for cn in top_concepts:
        row = [corpus_concepts_by_year.get(y, {}).get(cn, 0) for y in all_years]
        z_matrix.append(row)

    fig1 = go.Figure(data=go.Heatmap(
        z=z_matrix,
        x=[str(y) for y in all_years],
        y=top_concepts,
        colorscale="YlOrRd",
        hovertemplate="Concept: %{y}<br>Year: %{x}<br>Frequency: %{z}<extra></extra>",
    ))
    fig1.update_layout(
        title="Concept Evolution Heatmap (corpus articles)",
        xaxis_title="Publication Year",
        yaxis_title="Concept",
        width=1600, height=max(600, len(top_concepts) * 25 + 200),
        paper_bgcolor="white",
        yaxis=dict(autorange="reversed"),
    )
    _save_plotly_figure(fig1, output_dir, "concept_evolution_heatmap")

    # --- Chart 2: Lifecycle classification scatter ---
    lifecycles = []
    for cn in top_concepts:
        yearly_counts = {y: corpus_concepts_by_year.get(y, {}).get(cn, 0) for y in all_years}
        active_years = [y for y, c in yearly_counts.items() if c > 0]
        if not active_years:
            continue

        first_year = min(active_years)
        last_year = max(active_years)
        total = sum(yearly_counts.values())
        peak_year = max(active_years, key=lambda y: yearly_counts[y])

        # Classification
        recent_years = [y for y in active_years if y >= all_years[-1] - 3]
        early_years = [y for y in active_years if y <= all_years[0] + 3]
        recent_count = sum(yearly_counts.get(y, 0) for y in recent_years)
        early_count = sum(yearly_counts.get(y, 0) for y in early_years)
        recent_ratio = recent_count / total if total > 0 else 0
        early_ratio = early_count / total if total > 0 else 0

        if recent_ratio > 0.6 and first_year >= all_years[-1] - 5:
            phase = "emerging"
        elif early_ratio > 0.5 and last_year <= all_years[0] + 5:
            phase = "declining"
        elif len(active_years) >= len(all_years) * 0.7:
            phase = "foundational"
        else:
            phase = "active"

        lifecycles.append({
            "concept": cn,
            "first_year": first_year,
            "last_year": last_year,
            "peak_year": peak_year,
            "total": total,
            "phase": phase,
        })

    result["lifecycles"] = lifecycles

    if lifecycles:
        phase_colors = {
            "emerging": "#59a14f",
            "foundational": "#4e79a7",
            "declining": "#e15759",
            "active": "#f28e2b",
        }

        fig2 = go.Figure()
        for lc in lifecycles:
            fig2.add_trace(go.Scatter(
                x=[lc["first_year"]],
                y=[lc["total"]],
                mode="markers+text",
                marker=dict(
                    size=max(8, min(30, lc["total"])),
                    color=phase_colors.get(lc["phase"], "#999"),
                ),
                text=[lc["concept"]],
                textposition="top center",
                textfont=dict(size=9),
                name=f"{lc['concept']} ({lc['phase']})",
                hovertemplate=(
                    f"{lc['concept']}<br>"
                    f"Phase: {lc['phase']}<br>"
                    f"First: {lc['first_year']}, Peak: {lc['peak_year']}<br>"
                    f"Total: {lc['total']}<extra></extra>"
                ),
            ))

        fig2.update_layout(
            title="Concept Lifecycle Classification",
            xaxis_title="First Appearance Year",
            yaxis_title="Total Frequency",
            width=1400, height=700, paper_bgcolor="white",
            showlegend=False,
        )
        _save_plotly_figure(fig2, output_dir, "concept_lifecycle_scatter")

    # --- Chart 3: Shannon diversity over time ---
    diversity_by_year = []
    for y in all_years:
        counts = list(corpus_concepts_by_year.get(y, {}).values())
        if counts:
            h = _shannon_diversity(counts)
            n_concepts = len(counts)
            diversity_by_year.append({"year": y, "shannon": h, "n_concepts": n_concepts})

    result["diversity"] = diversity_by_year

    if len(diversity_by_year) >= 2:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=[d["year"] for d in diversity_by_year],
            y=[d["shannon"] for d in diversity_by_year],
            mode="lines+markers", name="Shannon H'",
            line=dict(color="#b07aa1", width=2),
            hovertemplate="Year: %{x}<br>H': %{y:.3f}<extra></extra>",
        ))
        fig3.add_trace(go.Bar(
            x=[d["year"] for d in diversity_by_year],
            y=[d["n_concepts"] for d in diversity_by_year],
            name="N concepts", yaxis="y2",
            marker_color="rgba(200,200,200,0.4)",
        ))
        fig3.update_layout(
            title="Conceptual Diversity Over Time (Shannon Index)",
            xaxis_title="Publication Year",
            yaxis_title="Shannon Diversity H'",
            yaxis2=dict(title="N distinct concepts", overlaying="y", side="right"),
            width=1400, height=600, paper_bgcolor="white",
        )
        _save_plotly_figure(fig3, output_dir, "concept_diversity_temporal")

    charts = sum(1 for name in ["concept_evolution_heatmap",
                                "concept_lifecycle_scatter",
                                "concept_diversity_temporal"]
                 if (output_dir / f"{name}.html").exists())
    result["charts"] = charts
    return result


# ---------------------------------------------------------------------------
# Analysis 4: Topic Flow Sankey
# ---------------------------------------------------------------------------

def compute_topic_flow_sankey(
    articles: list[dict],
    output_dir: Path,
) -> dict:
    """Topic flow Sankey diagrams at subfield and domain level."""
    import plotly.graph_objects as go

    logger.info("Analysis 4/8: Topic Flow Sankey...")

    # Collect topics per article with year
    article_topics = []
    for a in articles:
        year = a.get("publication_year")
        if not year or not isinstance(year, int):
            continue
        topics = a.get("topics") or a.get("openalex", {}).get("topics", []) or []
        if not topics:
            continue
        # Take primary topic (first)
        t = topics[0]
        subfield = (t.get("subfield") or "").strip()
        field = (t.get("field") or "").strip()
        domain = (t.get("domain") or "").strip()
        if subfield:
            article_topics.append({
                "year": year,
                "subfield": subfield,
                "field": field,
                "domain": domain,
            })

    if len(article_topics) < 5:
        logger.warning("Not enough topic data for Sankey")
        return {"status": "no_data"}

    years = [t["year"] for t in article_topics]
    periods = _compute_temporal_periods(years)

    if len(periods) < 2:
        logger.warning("Time span too narrow for Sankey periods")
        return {"status": "insufficient_timespan"}

    result: dict[str, Any] = {
        "n_articles_with_topics": len(article_topics),
        "periods": [{"start": p[0], "end": p[1]} for p in periods],
    }

    # --- Chart 1: Subfield-level Sankey ---
    _build_sankey(article_topics, periods, "subfield", output_dir,
                  "topic_flow_subfield", "Topic Flow: Subfield Level")

    # --- Chart 2: Domain-level Sankey ---
    _build_sankey(article_topics, periods, "domain", output_dir,
                  "topic_flow_domain", "Topic Flow: Domain Level")

    charts = sum(1 for name in ["topic_flow_subfield", "topic_flow_domain"]
                 if (output_dir / f"{name}.html").exists())
    result["charts"] = charts
    return result


def _build_sankey(
    article_topics: list[dict],
    periods: list[tuple[int, int]],
    level_key: str,
    output_dir: Path,
    filename: str,
    title: str,
) -> None:
    """Build and save a Sankey diagram for topic flow between periods."""
    import plotly.graph_objects as go

    # Count topic frequencies per period
    period_counts: list[Counter] = []
    for start, end in periods:
        c = Counter()
        for t in article_topics:
            if start <= t["year"] <= end:
                val = t.get(level_key, "")
                if val:
                    c[val] += 1
        period_counts.append(c)

    # Build node list: period_label:topic_name
    nodes: list[str] = []
    node_idx: dict[str, int] = {}

    for i, (start, end) in enumerate(periods):
        period_label = f"{start}-{end}"
        for topic in period_counts[i]:
            node_name = f"{period_label}: {topic}"
            if node_name not in node_idx:
                node_idx[node_name] = len(nodes)
                nodes.append(node_name)

    if not nodes:
        return

    # Build links between consecutive periods
    sources, targets, values, link_labels = [], [], [], []

    for i in range(len(periods) - 1):
        p1_label = f"{periods[i][0]}-{periods[i][1]}"
        p2_label = f"{periods[i+1][0]}-{periods[i+1][1]}"

        # Topics that appear in both periods
        common = set(period_counts[i].keys()) & set(period_counts[i+1].keys())
        all_topics = set(period_counts[i].keys()) | set(period_counts[i+1].keys())

        for topic in all_topics:
            src_key = f"{p1_label}: {topic}"
            tgt_key = f"{p2_label}: {topic}"

            if src_key not in node_idx or tgt_key not in node_idx:
                continue

            val = min(
                period_counts[i].get(topic, 0),
                period_counts[i+1].get(topic, 0),
            )
            if val > 0:
                sources.append(node_idx[src_key])
                targets.append(node_idx[tgt_key])
                values.append(val)
                link_labels.append(topic)

    if not sources:
        return

    # Colors
    palette = [
        "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
        "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
        "#86bcb6", "#8cd17d", "#b6992d", "#499894", "#d37295",
    ]
    # Assign colors by topic name
    unique_topics = sorted(set(link_labels))
    topic_color = {t: palette[i % len(palette)] for i, t in enumerate(unique_topics)}

    node_colors = []
    for n in nodes:
        topic = n.split(": ", 1)[-1] if ": " in n else n
        node_colors.append(topic_color.get(topic, "#999"))

    link_colors = [
        topic_color.get(lbl, "#999").replace(")", ",0.4)").replace("rgb", "rgba")
        if "rgb" in topic_color.get(lbl, "#999")
        else topic_color.get(lbl, "#999") + "88"
        for lbl in link_labels
    ]

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15, thickness=20,
            label=[n.split(": ", 1)[-1] if ": " in n else n for n in nodes],
            color=node_colors,
            customdata=nodes,
            hovertemplate="%{customdata}<br>Value: %{value}<extra></extra>",
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            label=link_labels,
            color=link_colors,
        ),
    )])

    # Add period labels as annotations
    n_periods = len(periods)
    for i, (start, end) in enumerate(periods):
        fig.add_annotation(
            x=i / max(1, n_periods - 1),
            y=1.08, xref="paper", yref="paper",
            text=f"<b>{start}-{end}</b>",
            showarrow=False, font=dict(size=13),
        )

    fig.update_layout(
        title=title,
        width=1400, height=800, paper_bgcolor="white",
    )
    _save_plotly_figure(fig, output_dir, filename)


# ---------------------------------------------------------------------------
# Analysis 5: Beauty Coefficient (Sleeping Beauty Detection)
# ---------------------------------------------------------------------------

def compute_beauty_coefficient(
    articles: list[dict],
    output_dir: Path,
) -> dict:
    """Detect 'Sleeping Beauties' via the Beauty Coefficient B(T).

    For each article with counts_by_year data, compares the actual citation
    trajectory c(t) against a linear recognition line y(t) from (0, c(0))
    to (t_m, c_m) where t_m is the year of maximum citations.

    B(T) = sum_{t=0}^{t_m} (y(t) - c(t)) / max(1, c(t))

    High positive B -> convex curve -> Sleeping Beauty.
    """
    import plotly.graph_objects as go

    logger.info("Analysis 5/8: Beauty Coefficient (Sleeping Beauty Detection)...")

    cache = DiskCache(OPENALEX_CACHE_DIR)
    records = []

    for a in articles:
        doi = a.get("doi", "")
        ndoi = normalize_doi(doi) if doi else ""
        pub_year = a.get("publication_year")
        if not pub_year or not isinstance(pub_year, int):
            continue

        # Get counts_by_year from OpenAlex cache
        cached = None
        if ndoi:
            cached = cache.get(f"doi:{ndoi}")
        if not cached:
            oa_id = a.get("openalex", {}).get("openalex_id", "")
            if oa_id:
                wid = oa_id.split("/")[-1] if "/" in oa_id else oa_id
                cached = cache.get(f"work:{wid}")
        if not cached or not isinstance(cached, dict):
            continue

        counts_by_year = cached.get("counts_by_year") or []
        if not counts_by_year or len(counts_by_year) < 3:
            continue

        # Build year -> citations mapping
        year_cites: dict[int, int] = {}
        for entry in counts_by_year:
            y = entry.get("year")
            c = entry.get("cited_by_count", 0)
            if y and isinstance(y, int):
                year_cites[y] = c

        if not year_cites:
            continue

        # Fill gaps: from publication year to max year in data
        min_y = pub_year
        max_y = max(year_cites.keys())
        if max_y <= min_y:
            continue

        # Build trajectory array c(t) for t=0..T
        T = max_y - min_y
        c_t = [year_cites.get(min_y + t, 0) for t in range(T + 1)]

        # Find t_m (year of max citations)
        c_m = max(c_t)
        if c_m == 0:
            continue
        t_m = c_t.index(c_m)
        if t_m == 0:
            continue  # No sleeping period

        # Build linear recognition line y(t)
        c_0 = c_t[0]
        slope = (c_m - c_0) / t_m

        # Calculate B(T)
        B = 0.0
        for t in range(t_m + 1):
            y_t = slope * t + c_0
            ct = c_t[t]
            B += (y_t - ct) / max(1, ct)

        records.append({
            "key": a.get("key", ""),
            "title": (a.get("title") or "")[:80],
            "year": pub_year,
            "beauty_coefficient": round(B, 2),
            "peak_year": min_y + t_m,
            "peak_citations": c_m,
            "cited_by_count": a.get("cited_by_count", 0),
            "sleeping_years": t_m,
        })

        # Store on article for cross-analysis
        a.setdefault("beauty_coefficient", round(B, 2))

    if not records:
        logger.warning("No counts_by_year data available for Beauty Coefficient")
        return {"status": "no_data"}

    result: dict[str, Any] = {
        "n_articles": len(records),
    }

    # Sort by B descending
    records.sort(key=lambda r: r["beauty_coefficient"], reverse=True)
    result["top_sleeping_beauties"] = records[:15]

    b_vals = [r["beauty_coefficient"] for r in records]
    result["b_mean"] = round(float(np.mean(b_vals)), 2)
    result["b_median"] = round(float(np.median(b_vals)), 2)
    result["b_max"] = round(float(max(b_vals)), 2)

    # --- Chart 1: Scatter (year vs B) ---
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=[r["year"] for r in records],
        y=[r["beauty_coefficient"] for r in records],
        mode="markers",
        marker=dict(
            size=[max(5, min(20, r["sleeping_years"])) for r in records],
            color=[r["cited_by_count"] for r in records],
            colorscale="Viridis",
            colorbar=dict(title="Total Cites"),
        ),
        text=[r["title"] for r in records],
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Year: %{x}<br>"
            "B(T): %{y:.2f}<br>"
            "<extra></extra>"
        ),
    ))
    fig1.update_layout(
        title=f"Beauty Coefficient — Sleeping Beauty Detection (n={len(records)})",
        xaxis_title="Publication Year",
        yaxis_title="Beauty Coefficient B(T)",
        width=1400, height=700, paper_bgcolor="white",
    )
    _save_plotly_figure(fig1, output_dir, "beauty_coefficient_scatter")

    # --- Chart 2: Histogram of B values ---
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(
        x=b_vals, nbinsx=30,
        marker_color="#b07aa1",
        hovertemplate="B(T): %{x:.1f}<br>Count: %{y}<extra></extra>",
    ))
    fig2.add_vline(x=result["b_median"], line_dash="dash", line_color="red",
                   annotation_text=f"Median ({result['b_median']:.1f})")
    fig2.update_layout(
        title="Beauty Coefficient Distribution",
        xaxis_title="Beauty Coefficient B(T)",
        yaxis_title="Number of Articles",
        width=1200, height=600, paper_bgcolor="white",
    )
    _save_plotly_figure(fig2, output_dir, "beauty_coefficient_histogram")

    charts = sum(1 for name in ["beauty_coefficient_scatter",
                                "beauty_coefficient_histogram"]
                 if (output_dir / f"{name}.html").exists())
    result["charts"] = charts
    return result


# ---------------------------------------------------------------------------
# Analysis 6: CSS Classification (Characteristic Scales and Scores)
# ---------------------------------------------------------------------------

def compute_css_classification(
    articles: list[dict],
    output_dir: Path,
) -> dict:
    """Classify articles using CSS (Characteristic Scales and Scores).

    Iterative truncation of citation distribution to find natural class boundaries:
    beta_1 = mean(all), beta_2 = mean(>beta_1), beta_3 = mean(>beta_2)

    Classes:
    - Poorly cited: < beta_1
    - Fairly cited: beta_1 to beta_2
    - Remarkably cited: beta_2 to beta_3
    - Outstandingly cited: >= beta_3
    """
    import plotly.graph_objects as go

    logger.info("Analysis 6/8: CSS Classification (Characteristic Scales)...")

    cites = []
    for a in articles:
        c = a.get("cited_by_count", 0) or 0
        cites.append({"key": a.get("key", ""), "title": (a.get("title") or "")[:80],
                       "year": a.get("publication_year", 0),
                       "cited_by_count": c})

    if not cites:
        return {"status": "no_data"}

    cite_vals = sorted([c["cited_by_count"] for c in cites], reverse=True)
    n = len(cite_vals)

    # Iterative truncation
    betas = []
    current_vals = cite_vals[:]
    for _ in range(10):  # Safety limit
        if not current_vals:
            break
        beta = sum(current_vals) / len(current_vals)
        betas.append(round(beta, 2))
        current_vals = [v for v in current_vals if v > beta]
        if len(current_vals) <= 1:
            break

    # We need at least beta_1, beta_2, beta_3
    while len(betas) < 3:
        betas.append(betas[-1] * 2 if betas else 0)

    beta_1, beta_2, beta_3 = betas[0], betas[1], betas[2]

    result: dict[str, Any] = {
        "n_articles": n,
        "beta_1": beta_1,
        "beta_2": beta_2,
        "beta_3": beta_3,
        "all_betas": betas,
    }

    # Classify articles
    class_labels = {
        "Poorly cited": 0,
        "Fairly cited": 0,
        "Remarkably cited": 0,
        "Outstandingly cited": 0,
    }
    for c in cites:
        v = c["cited_by_count"]
        if v >= beta_3:
            c["css_class"] = "Outstandingly cited"
        elif v >= beta_2:
            c["css_class"] = "Remarkably cited"
        elif v >= beta_1:
            c["css_class"] = "Fairly cited"
        else:
            c["css_class"] = "Poorly cited"
        class_labels[c["css_class"]] += 1

    result["class_distribution"] = class_labels

    # Store css_class on articles
    key_to_css = {c["key"]: c["css_class"] for c in cites}
    for a in articles:
        a["css_class"] = key_to_css.get(a.get("key", ""), "")

    # Top outstandingly cited
    outstanding = [c for c in cites if c["css_class"] == "Outstandingly cited"]
    outstanding.sort(key=lambda x: x["cited_by_count"], reverse=True)
    result["outstanding_articles"] = outstanding[:15]

    # --- Chart: CSS classification bar ---
    class_order = ["Poorly cited", "Fairly cited", "Remarkably cited", "Outstandingly cited"]
    class_colors = ["#e15759", "#f28e2b", "#59a14f", "#4e79a7"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=class_order,
        y=[class_labels[c] for c in class_order],
        marker_color=class_colors,
        text=[f"n={class_labels[c]}" for c in class_order],
        textposition="outside",
        hovertemplate="%{x}<br>Count: %{y}<extra></extra>",
    ))

    # Add threshold annotations
    fig.add_annotation(x=0.5, y=max(class_labels.values()) * 0.95,
                       text=f"β₁={beta_1:.1f}  β₂={beta_2:.1f}  β₃={beta_3:.1f}",
                       showarrow=False, font=dict(size=12))

    fig.update_layout(
        title=f"CSS Classification — Characteristic Citation Scales (n={n})",
        xaxis_title="CSS Class",
        yaxis_title="Number of Articles",
        width=1000, height=600, paper_bgcolor="white",
    )
    _save_plotly_figure(fig, output_dir, "css_classification")

    charts = 1 if (output_dir / "css_classification.html").exists() else 0
    result["charts"] = charts
    return result


# ---------------------------------------------------------------------------
# Analysis 7: Outgrow Index (CR Index)
# ---------------------------------------------------------------------------

def compute_outgrow_index(
    articles: list[dict],
    output_dir: Path,
) -> dict:
    """Compute the Outgrow Index (CR Index) for each article.

    CR(A) = 1 - R(A) / (TRef(A) + 1)

    Where R(A) is the rank of article A among its own references
    (sorted by cited_by_count descending), and TRef(A) is the total
    number of references. A value close to 1 means the article has
    surpassed almost all its bibliography.
    """
    import plotly.graph_objects as go

    logger.info("Analysis 7/8: Outgrow Index (CR Index)...")

    records = []
    for a in articles:
        a_cites = a.get("cited_by_count", 0) or 0
        refs = a.get("references_resolved") or []
        if not refs or len(refs) < 3:
            continue

        # Collect citation counts for references
        ref_cites = []
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            rc = ref.get("cited_by_count")
            if rc is not None and isinstance(rc, (int, float)):
                ref_cites.append(int(rc))

        if not ref_cites:
            continue

        # Extended set: refs + article itself
        extended = ref_cites + [a_cites]
        extended.sort(reverse=True)

        # Rank of article A (1-based)
        rank = extended.index(a_cites) + 1
        t_ref = len(ref_cites)

        cr = 1.0 - rank / (t_ref + 1)

        records.append({
            "key": a.get("key", ""),
            "title": (a.get("title") or "")[:80],
            "year": a.get("publication_year", 0),
            "cited_by_count": a_cites,
            "n_refs": t_ref,
            "rank_in_refs": rank,
            "outgrow_index": round(cr, 4),
        })

    if not records:
        logger.warning("No reference citation data for Outgrow Index")
        return {"status": "no_data"}

    records.sort(key=lambda r: r["outgrow_index"], reverse=True)

    cr_vals = [r["outgrow_index"] for r in records]
    result: dict[str, Any] = {
        "n_articles": len(records),
        "cr_mean": round(float(np.mean(cr_vals)), 4),
        "cr_median": round(float(np.median(cr_vals)), 4),
        "top_outgrowers": records[:15],
    }

    # --- Chart: Scatter (year vs CR) ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[r["year"] for r in records],
        y=[r["outgrow_index"] for r in records],
        mode="markers",
        marker=dict(
            size=[max(5, min(20, r["cited_by_count"] / max(1, max(cr_vals)) * 15))
                  for r in records],
            color=[r["cited_by_count"] for r in records],
            colorscale="Plasma",
            colorbar=dict(title="Total Cites"),
        ),
        text=[r["title"] for r in records],
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Year: %{x}<br>"
            "CR Index: %{y:.3f}<br>"
            "<extra></extra>"
        ),
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray",
                  annotation_text="CR=0.5 (median rank)")
    fig.update_layout(
        title=f"Outgrow Index (CR) — Innovation Disruption (n={len(records)})",
        xaxis_title="Publication Year",
        yaxis_title="Outgrow Index CR(A)",
        width=1400, height=700, paper_bgcolor="white",
    )
    _save_plotly_figure(fig, output_dir, "outgrow_index_scatter")

    charts = 1 if (output_dir / "outgrow_index_scatter.html").exists() else 0
    result["charts"] = charts
    return result


# ---------------------------------------------------------------------------
# Analysis 8: Citation Genealogy H-1 (Backward)
# ---------------------------------------------------------------------------

def compute_citation_genealogy_h1(
    articles: list[dict],
    output_dir: Path,
) -> dict:
    """Analyze backward citation genealogy (H-1): intellectual roots.

    Examines the references cited by corpus articles to understand:
    - Distribution of concepts/topics in cited literature
    - Temporal distribution of cited references
    - Core disciplines feeding the corpus
    """
    import plotly.graph_objects as go

    logger.info("Analysis 8/8: Citation Genealogy H-1 (Backward)...")

    concept_freq: Counter = Counter()
    topic_freq: Counter = Counter()
    ref_year_freq: Counter = Counter()
    n_refs_total = 0

    for a in articles:
        refs = a.get("references_resolved") or []
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            n_refs_total += 1

            # Year
            ry = ref.get("year")
            if ry and isinstance(ry, int) and MIN_YEAR <= ry <= MAX_YEAR:
                ref_year_freq[ry] += 1

            # Topics (CWTS Leiden OpenAlex topics — preferred over legacy concepts)
            ref_topics = ref.get("topics") or []
            if ref_topics:
                for t in ref_topics:
                    t_name = (t.get("name") or t.get("display_name") or "").strip()
                    if t_name and not is_noisy_legacy_concept(t_name):
                        concept_freq[t_name] += 1
                    subfield = (t.get("subfield") or "").strip()
                    if subfield and not is_noisy_legacy_concept(subfield):
                        topic_freq[subfield] += 1
            else:
                # Fallback to legacy concepts filtered for noise
                for c in ref.get("concepts") or []:
                    name = (c.get("name") or "").strip()
                    if name and not is_noisy_legacy_concept(name):
                        concept_freq[name] += 1

    if n_refs_total == 0:
        logger.warning("No resolved references for genealogy analysis")
        return {"status": "no_data"}

    result: dict[str, Any] = {
        "n_refs_total": n_refs_total,
        "n_unique_concepts": len(concept_freq),
        "n_unique_topics": len(topic_freq),
    }

    # --- Chart 1: Top concepts in cited literature ---
    top_concepts = concept_freq.most_common(TOP_N_CONCEPTS)
    result["top_concepts"] = [{"concept": c, "freq": f} for c, f in top_concepts]

    if top_concepts:
        fig1 = go.Figure()
        concepts_names = [c for c, _ in top_concepts]
        concepts_freqs = [f for _, f in top_concepts]
        concepts_names.reverse()
        concepts_freqs.reverse()

        fig1.add_trace(go.Bar(
            y=concepts_names,
            x=concepts_freqs,
            orientation="h",
            marker_color="#4e79a7",
            hovertemplate="%{y}<br>Frequency: %{x}<extra></extra>",
        ))
        fig1.update_layout(
            title=f"H-1 Genealogy: Intellectual Roots — Top Concepts (n={n_refs_total} refs)",
            xaxis_title="Frequency in Cited References",
            yaxis_title="",
            width=1200, height=max(500, len(top_concepts) * 25 + 200),
            paper_bgcolor="white",
            margin=dict(l=250),
        )
        _save_plotly_figure(fig1, output_dir, "genealogy_h1_concepts")

    # --- Chart 2: Temporal distribution of cited references ---
    if ref_year_freq:
        years_sorted = sorted(ref_year_freq.keys())
        counts = [ref_year_freq[y] for y in years_sorted]
        result["ref_year_range"] = f"{min(years_sorted)}-{max(years_sorted)}"
        result["ref_year_median"] = int(np.median(
            [y for y, c in ref_year_freq.items() for _ in range(c)]
        ))

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=years_sorted,
            y=counts,
            marker_color="#76b7b2",
            hovertemplate="Year: %{x}<br>Count: %{y}<extra></extra>",
        ))
        fig2.add_vline(x=result["ref_year_median"], line_dash="dash",
                       line_color="red",
                       annotation_text=f"Median ({result['ref_year_median']})")
        fig2.update_layout(
            title="H-1 Genealogy: Temporal Distribution of Cited References",
            xaxis_title="Publication Year of Cited Reference",
            yaxis_title="Number of Citations",
            width=1400, height=600, paper_bgcolor="white",
        )
        _save_plotly_figure(fig2, output_dir, "genealogy_h1_temporal")

    charts = sum(1 for name in ["genealogy_h1_concepts", "genealogy_h1_temporal"]
                 if (output_dir / f"{name}.html").exists())
    result["charts"] = charts
    return result


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _generate_report(
    fwci_result: dict,
    price_result: dict,
    concept_result: dict,
    sankey_result: dict,
    output_dir: Path,
    beauty_result: dict | None = None,
    css_result: dict | None = None,
    outgrow_result: dict | None = None,
    genealogy_result: dict | None = None,
    apc_result: dict | None = None,
) -> None:
    """Generate impact_evolution_report.md summary."""
    beauty_result = beauty_result or {}
    css_result = css_result or {}
    outgrow_result = outgrow_result or {}
    apc_result = apc_result or {}
    genealogy_result = genealogy_result or {}

    lines = [
        "# Impact Evolution Report (Phase 5)",
        "",
        "## Overview",
        "",
    ]

    # --- 1. FWCI ---
    lines.append("## 1. Field-Weighted Citation Impact (FWCI)")
    lines.append("")
    if fwci_result.get("status") != "no_data":
        lines.append(f"- **Articles with FWCI**: {fwci_result.get('n_articles_with_fwci', 0)}")
        lines.append(f"- **Mean FWCI**: {fwci_result.get('fwci_mean', 'N/A')}")
        lines.append(f"- **Median FWCI**: {fwci_result.get('fwci_median', 'N/A')}")
        lines.append(f"- **% above world average (>1.0)**: {fwci_result.get('pct_above_1', 'N/A')}%")
        lines.append(f"- **% in top 10%**: {fwci_result.get('pct_top_10', 'N/A')}%")
        lines.append(f"- **% in top 1%**: {fwci_result.get('pct_top_1', 'N/A')}%")

        top_by_topic = fwci_result.get("fwci_by_topic", [])
        if top_by_topic:
            lines.append("")
            lines.append("### FWCI by Primary Topic (top 10)")
            lines.append("")
            lines.append("| Topic | Median FWCI | Mean FWCI | N |")
            lines.append("|-------|-------------|-----------|---|")
            for t in top_by_topic[:10]:
                lines.append(
                    f"| {t['topic']} | {t['median']:.2f} | {t['mean']:.2f} | {t['n']} |"
                )
    else:
        lines.append("*No FWCI data available in OpenAlex cache.*")
    lines.append("")

    # --- 2. Price Index ---
    lines.append("## 2. Price Index (Knowledge Base Age)")
    lines.append("")
    if price_result.get("status") != "no_data":
        lines.append(f"- **Articles analyzed**: {price_result.get('n_articles', 0)}")
        lines.append(f"- **Mean Price Index**: {price_result.get('price_index_mean', 'N/A')}%")
        lines.append(f"- **Median Price Index**: {price_result.get('price_index_median', 'N/A')}%")
        lines.append(f"- **Median reference age (overall)**: {price_result.get('median_ref_age_overall', 'N/A')} years")
        lines.append(f"- **Total references analyzed**: {price_result.get('total_refs', 'N/A')}")
    else:
        lines.append("*No reference year data available (run Phase 1i first).*")
    lines.append("")

    # --- 3. Concept Evolution ---
    lines.append("## 3. Concept Evolution / Lifecycle")
    lines.append("")
    if concept_result.get("status") not in ("no_data", "no_frequent_concepts"):
        lines.append(f"- **Total unique concepts**: {concept_result.get('n_concepts_total', 0)}")
        lines.append(f"- **Top concepts analyzed**: {concept_result.get('n_top_concepts', 0)}")

        lifecycles = concept_result.get("lifecycles", [])
        if lifecycles:
            # Count by phase
            phase_counts = Counter(lc["phase"] for lc in lifecycles)
            for phase in ["emerging", "foundational", "active", "declining"]:
                lines.append(f"- **{phase.capitalize()}**: {phase_counts.get(phase, 0)}")

            lines.append("")
            lines.append("### Concept Lifecycle Details")
            lines.append("")
            lines.append("| Concept | Phase | First Year | Peak Year | Total |")
            lines.append("|---------|-------|------------|-----------|-------|")
            for lc in lifecycles:
                lines.append(
                    f"| {lc['concept']} | {lc['phase']} | "
                    f"{lc['first_year']} | {lc['peak_year']} | {lc['total']} |"
                )

        diversity = concept_result.get("diversity", [])
        if diversity:
            lines.append("")
            lines.append("### Shannon Diversity Index")
            lines.append("")
            latest = diversity[-1]
            earliest = diversity[0]
            lines.append(
                f"- Earliest ({earliest['year']}): H' = {earliest['shannon']:.3f} "
                f"({earliest['n_concepts']} concepts)"
            )
            lines.append(
                f"- Latest ({latest['year']}): H' = {latest['shannon']:.3f} "
                f"({latest['n_concepts']} concepts)"
            )
    else:
        lines.append("*No concept data available.*")
    lines.append("")

    # --- 4. Topic Flow ---
    lines.append("## 4. Topic Flow Sankey")
    lines.append("")
    if sankey_result.get("status") not in ("no_data", "insufficient_timespan"):
        lines.append(f"- **Articles with topics**: {sankey_result.get('n_articles_with_topics', 0)}")
        periods = sankey_result.get("periods", [])
        if periods:
            period_strs = [f"{p['start']}-{p['end']}" for p in periods]
            lines.append(f"- **Temporal periods**: {', '.join(period_strs)}")
    else:
        lines.append("*Insufficient topic data for Sankey analysis.*")
    lines.append("")

    # --- 5. Beauty Coefficient ---
    lines.append("## 5. Beauty Coefficient (Sleeping Beauty Detection)")
    lines.append("")
    if beauty_result.get("status") != "no_data" and beauty_result.get("n_articles"):
        lines.append(f"- **Articles analyzed**: {beauty_result.get('n_articles', 0)}")
        lines.append(f"- **Mean B(T)**: {beauty_result.get('b_mean', 'N/A')}")
        lines.append(f"- **Median B(T)**: {beauty_result.get('b_median', 'N/A')}")
        lines.append(f"- **Max B(T)**: {beauty_result.get('b_max', 'N/A')}")

        top_sb = beauty_result.get("top_sleeping_beauties", [])
        if top_sb:
            lines.append("")
            lines.append("### Top Sleeping Beauties")
            lines.append("")
            lines.append("| Article | Year | B(T) | Peak Year | Sleeping Years |")
            lines.append("|---------|------|------|-----------|----------------|")
            for sb in top_sb[:10]:
                lines.append(
                    f"| {sb['title']} | {sb['year']} | {sb['beauty_coefficient']:.1f} | "
                    f"{sb['peak_year']} | {sb['sleeping_years']} |"
                )
    else:
        lines.append("*No counts_by_year data available for Beauty Coefficient.*")
    lines.append("")

    # --- 6. CSS Classification ---
    lines.append("## 6. CSS Classification (Characteristic Scales)")
    lines.append("")
    if css_result.get("status") != "no_data" and css_result.get("n_articles"):
        lines.append(f"- **Articles classified**: {css_result.get('n_articles', 0)}")
        lines.append(f"- **Thresholds**: beta_1={css_result.get('beta_1', 'N/A')}, "
                      f"beta_2={css_result.get('beta_2', 'N/A')}, "
                      f"beta_3={css_result.get('beta_3', 'N/A')}")
        cd = css_result.get("class_distribution", {})
        for cls in ["Poorly cited", "Fairly cited", "Remarkably cited", "Outstandingly cited"]:
            lines.append(f"- **{cls}**: {cd.get(cls, 0)}")
    else:
        lines.append("*No citation data for CSS classification.*")
    lines.append("")

    # --- 7. Outgrow Index ---
    lines.append("## 7. Outgrow Index (CR Index)")
    lines.append("")
    if outgrow_result.get("status") != "no_data" and outgrow_result.get("n_articles"):
        lines.append(f"- **Articles analyzed**: {outgrow_result.get('n_articles', 0)}")
        lines.append(f"- **Mean CR**: {outgrow_result.get('cr_mean', 'N/A')}")
        lines.append(f"- **Median CR**: {outgrow_result.get('cr_median', 'N/A')}")

        top_og = outgrow_result.get("top_outgrowers", [])
        if top_og:
            lines.append("")
            lines.append("### Top Outgrowers (highest CR)")
            lines.append("")
            lines.append("| Article | Year | CR Index | Citations | Rank/Refs |")
            lines.append("|---------|------|----------|-----------|-----------|")
            for og in top_og[:10]:
                lines.append(
                    f"| {og['title']} | {og['year']} | {og['outgrow_index']:.3f} | "
                    f"{og['cited_by_count']} | {og['rank_in_refs']}/{og['n_refs']} |"
                )
    else:
        lines.append("*No reference citation data for Outgrow Index.*")
    lines.append("")

    # --- 8. Citation Genealogy H-1 ---
    lines.append("## 8. Citation Genealogy H-1 (Backward)")
    lines.append("")
    if genealogy_result.get("status") != "no_data" and genealogy_result.get("n_refs_total"):
        lines.append(f"- **Total references analyzed**: {genealogy_result.get('n_refs_total', 0)}")
        lines.append(f"- **Unique concepts in refs**: {genealogy_result.get('n_unique_concepts', 0)}")
        lines.append(f"- **Unique topics in refs**: {genealogy_result.get('n_unique_topics', 0)}")
        if genealogy_result.get("ref_year_range"):
            lines.append(f"- **Reference year range**: {genealogy_result['ref_year_range']}")
        if genealogy_result.get("ref_year_median"):
            lines.append(f"- **Median reference year**: {genealogy_result['ref_year_median']}")

        top_concepts = genealogy_result.get("top_concepts", [])
        if top_concepts:
            lines.append("")
            lines.append("### Top Concepts in Cited Literature")
            lines.append("")
            lines.append("| Concept | Frequency |")
            lines.append("|---------|-----------|")
            for tc in top_concepts[:15]:
                lines.append(f"| {tc['concept']} | {tc['freq']} |")
    else:
        lines.append("*No resolved references for genealogy analysis.*")
    lines.append("")

    # --- 9. APC Impact ---
    lines.append("## 9. APC Impact (Cost-Benefit)")
    lines.append("")
    if apc_result.get("n_apc", 0) >= 5:
        lines.append(f"- **Articles with APC data**: {apc_result.get('n_apc', 0)}")
        lines.append(f"- **Articles without APC**: {apc_result.get('n_non_apc', 0)}")
        lines.append(f"- **Mean APC**: ${apc_result.get('apc_mean_usd', 0):.0f} USD")
        lines.append(f"- **APC paid — mean citations**: {apc_result.get('apc_mean_citations', 0)}")
        lines.append(f"- **No APC — mean citations**: {apc_result.get('non_apc_mean_citations', 0)}")
        lines.append(f"- **APC paid — median citations**: {apc_result.get('apc_median_citations', 0)}")
        lines.append(f"- **No APC — median citations**: {apc_result.get('non_apc_median_citations', 0)}")
        if apc_result.get("cost_per_citation_usd"):
            lines.append(f"- **Cost per citation**: ${apc_result['cost_per_citation_usd']:.1f} USD")
    else:
        lines.append(f"*Insufficient APC data ({apc_result.get('n_apc', 0)} articles).*")
    lines.append("")

    # --- Output files ---
    lines.append("## Output Files")
    lines.append("")
    lines.append(f"All outputs in: `{output_dir}/`")
    lines.append("")

    report_path = output_dir / "impact_evolution_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Saved: {report_path}")


# ---------------------------------------------------------------------------
# Analysis 9: APC Impact (cost-benefit)
# ---------------------------------------------------------------------------

def compute_apc_impact(
    articles: list[dict],
    output_dir: Path,
) -> dict:
    """Compare citation impact of APC-paid (gold OA) vs non-APC articles."""
    logger.info("Analysis 9/9: APC Impact (cost-benefit)...")

    apc_articles = []  # articles with APC paid
    non_apc_articles = []

    for a in articles:
        apc = a.get("openalex", {}).get("apc_paid", {}) or {}
        apc_usd = apc.get("value_usd", 0) if isinstance(apc, dict) else 0
        cites = a.get("cited_by_count", 0) or 0
        year = a.get("publication_year")
        if not year:
            continue

        if apc_usd and apc_usd > 0:
            apc_articles.append({"cites": cites, "apc": apc_usd, "year": year})
        else:
            non_apc_articles.append({"cites": cites, "year": year})

    if len(apc_articles) < 5:
        logger.info(f"  Only {len(apc_articles)} articles with APC data — skipping")
        return {"status": "insufficient_data", "n_apc": len(apc_articles)}

    apc_cites = [a["cites"] for a in apc_articles]
    non_apc_cites = [a["cites"] for a in non_apc_articles]

    result = {
        "n_apc": len(apc_articles),
        "n_non_apc": len(non_apc_articles),
        "apc_mean_citations": round(sum(apc_cites) / max(len(apc_cites), 1), 2),
        "non_apc_mean_citations": round(sum(non_apc_cites) / max(len(non_apc_cites), 1), 2),
        "apc_median_citations": sorted(apc_cites)[len(apc_cites) // 2] if apc_cites else 0,
        "non_apc_median_citations": sorted(non_apc_cites)[len(non_apc_cites) // 2] if non_apc_cites else 0,
        "apc_mean_usd": round(sum(a["apc"] for a in apc_articles) / len(apc_articles), 0),
        "charts": 0,
    }

    # Cost per citation
    total_apc = sum(a["apc"] for a in apc_articles)
    total_apc_cites = sum(apc_cites)
    result["cost_per_citation_usd"] = round(total_apc / max(total_apc_cites, 1), 1)

    try:
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Box(
            y=apc_cites, name=f"APC Paid (n={len(apc_cites)})",
            marker_color="#e74c3c",
        ))
        fig.add_trace(go.Box(
            y=non_apc_cites, name=f"No APC (n={len(non_apc_cites)})",
            marker_color="#3498db",
        ))
        fig.update_layout(
            title="Citation Impact: APC Paid vs No APC",
            yaxis_title="Citations",
            showlegend=False,
        )
        _save_plotly_figure(fig, output_dir, "apc_impact_comparison")
        result["charts"] += 1
    except Exception as e:
        logger.debug(f"APC impact chart failed: {e}")

    return result


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_phase5(articles: list[dict]) -> dict:
    """Phase 5: Impact Evolution Analysis.

    1. FWCI analysis (from raw OpenAlex cache)
    2. Price Index (from references_resolved)
    3. Concept Evolution / Lifecycle (from article concepts + references)
    4. Topic Flow Sankey (from article topics)
    5. Beauty Coefficient (Sleeping Beauty detection)
    6. CSS Classification (Characteristic Scales and Scores)
    7. Outgrow Index (CR Index — innovation disruption)
    8. Citation Genealogy H-1 (backward — intellectual roots)
    """
    logger.info("=" * 60)
    logger.info("Phase 5: Impact Evolution Analysis")
    logger.info("=" * 60)

    output_dir = IMPACT_EVOLUTION_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    analyses_completed = 0
    total_charts = 0

    # Analysis 1: FWCI
    try:
        fwci_result = compute_fwci_analysis(articles, output_dir)
        if fwci_result.get("status") != "no_data":
            analyses_completed += 1
            total_charts += fwci_result.get("charts", 0)
    except Exception as e:
        logger.error(f"FWCI analysis failed: {e}")
        fwci_result = {"status": "error", "error": str(e)}

    # Analysis 2: Price Index
    try:
        price_result = compute_price_index(articles, output_dir)
        if price_result.get("status") != "no_data":
            analyses_completed += 1
            total_charts += price_result.get("charts", 0)
    except Exception as e:
        logger.error(f"Price Index analysis failed: {e}")
        price_result = {"status": "error", "error": str(e)}

    # Analysis 3: Concept Evolution
    try:
        concept_result = compute_concept_evolution(articles, output_dir)
        if concept_result.get("status") not in ("no_data", "no_frequent_concepts"):
            analyses_completed += 1
            total_charts += concept_result.get("charts", 0)
    except Exception as e:
        logger.error(f"Concept evolution analysis failed: {e}")
        concept_result = {"status": "error", "error": str(e)}

    # Analysis 4: Topic Flow Sankey
    try:
        sankey_result = compute_topic_flow_sankey(articles, output_dir)
        if sankey_result.get("status") not in ("no_data", "insufficient_timespan"):
            analyses_completed += 1
            total_charts += sankey_result.get("charts", 0)
    except Exception as e:
        logger.error(f"Topic flow Sankey failed: {e}")
        sankey_result = {"status": "error", "error": str(e)}

    # Analysis 5: Beauty Coefficient
    try:
        beauty_result = compute_beauty_coefficient(articles, output_dir)
        if beauty_result.get("status") != "no_data":
            analyses_completed += 1
            total_charts += beauty_result.get("charts", 0)
    except Exception as e:
        logger.error(f"Beauty Coefficient analysis failed: {e}")
        beauty_result = {"status": "error", "error": str(e)}

    # Analysis 6: CSS Classification
    try:
        css_result = compute_css_classification(articles, output_dir)
        if css_result.get("status") != "no_data":
            analyses_completed += 1
            total_charts += css_result.get("charts", 0)
    except Exception as e:
        logger.error(f"CSS Classification failed: {e}")
        css_result = {"status": "error", "error": str(e)}

    # Analysis 7: Outgrow Index
    try:
        outgrow_result = compute_outgrow_index(articles, output_dir)
        if outgrow_result.get("status") != "no_data":
            analyses_completed += 1
            total_charts += outgrow_result.get("charts", 0)
    except Exception as e:
        logger.error(f"Outgrow Index analysis failed: {e}")
        outgrow_result = {"status": "error", "error": str(e)}

    # Analysis 8: Citation Genealogy H-1
    try:
        genealogy_result = compute_citation_genealogy_h1(articles, output_dir)
        if genealogy_result.get("status") != "no_data":
            analyses_completed += 1
            total_charts += genealogy_result.get("charts", 0)
    except Exception as e:
        logger.error(f"Citation Genealogy H-1 failed: {e}")
        genealogy_result = {"status": "error", "error": str(e)}

    # Analysis 9: APC Impact
    try:
        apc_result = compute_apc_impact(articles, output_dir)
        if apc_result.get("status") not in ("insufficient_data",):
            analyses_completed += 1
            total_charts += apc_result.get("charts", 0)
    except Exception as e:
        logger.error(f"APC Impact analysis failed: {e}")
        apc_result = {"status": "error", "error": str(e)}

    # Generate report
    try:
        _generate_report(
            fwci_result, price_result, concept_result, sankey_result,
            output_dir,
            beauty_result=beauty_result,
            css_result=css_result,
            outgrow_result=outgrow_result,
            genealogy_result=genealogy_result,
            apc_result=apc_result,
        )
    except Exception as e:
        logger.error(f"Report generation failed: {e}")

    # Save data JSON
    data_export = {
        "fwci": {k: v for k, v in fwci_result.items()
                 if k not in ("charts",)},
        "price_index": {k: v for k, v in price_result.items()
                        if k not in ("charts", "ref_years")},
        "concept_evolution": {k: v for k, v in concept_result.items()
                              if k not in ("charts",)},
        "topic_flow": {k: v for k, v in sankey_result.items()
                       if k not in ("charts",)},
        "beauty_coefficient": {k: v for k, v in beauty_result.items()
                               if k not in ("charts",)},
        "css_classification": {k: v for k, v in css_result.items()
                               if k not in ("charts",)},
        "outgrow_index": {k: v for k, v in outgrow_result.items()
                          if k not in ("charts",)},
        "citation_genealogy_h1": {k: v for k, v in genealogy_result.items()
                                  if k not in ("charts",)},
        "apc_impact": {k: v for k, v in apc_result.items()
                       if k not in ("charts",)},
    }
    try:
        save_json(data_export, output_dir / "impact_evolution_data.json")
    except Exception as e:
        logger.error(f"Data JSON save failed: {e}")

    result = {
        "status": "ok",
        "analyses_completed": analyses_completed,
        "charts_saved": total_charts,
        "output_dir": str(output_dir),
    }
    logger.info(
        f"Phase 5 complete — {analyses_completed}/9 analyses, "
        f"{total_charts} charts saved"
    )
    return result

run_phase6 = run_phase5
