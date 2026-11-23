"""Motor de Estratificacion por Tipo de Documento.

Detecta automaticamente tipos de documento (item_type), y si hay suficientes
de cada tipo, ejecuta el pipeline de analisis por separado para cada estrato
mas un reporte comparativo.

Estructura de salida:
  output/figures/by_item_type/journalArticle/
  output/reports/by_item_type/journalArticle/
  output/reports/stratified_comparative_report.md
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import (
    OUTPUT_DIR, FIGURES_DIR, REPORTS_DIR, PLOTLY_DIR, NETWORKS_DIR,
    STRATIFIED_SUBDIR,
)

logger = logging.getLogger("bibliometrico.stratified")

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight"})


# ===========================================================================
# Configuration
# ===========================================================================

@dataclass
class StratumThresholds:
    """Minimum articles per stratum for each analysis type."""
    descriptive: int = 10
    abstract_analysis: int = 10
    bert: int = 20
    quality: int = 5
    networks: int = 20
    supplementary: int = 10

    @classmethod
    def from_min_size(cls, min_size: int) -> "StratumThresholds":
        """Create thresholds where everything uses min_size as minimum."""
        return cls(
            descriptive=min_size,
            abstract_analysis=min_size,
            bert=max(min_size, 20),
            quality=max(min_size // 2, 5),
            networks=max(min_size, 20),
            supplementary=min_size,
        )


# ===========================================================================
# Stratum Detection
# ===========================================================================

def detect_strata(
    articles: list[dict],
    min_size: int = 15,
    type_field: str = "item_type",
) -> tuple[dict[str, list[dict]], dict[str, int]]:
    """Count articles by type, return eligible strata and discarded counts.

    Args:
        articles: Full article list.
        min_size: Minimum articles for a stratum to be eligible.
        type_field: Field name for document type.

    Returns:
        (eligible_strata, discarded): eligible maps type->articles,
        discarded maps type->count.
    """
    # Canonical mapping: merge common variants into a single label
    _TYPE_CANONICAL: dict[str, str] = {
        "journalarticle": "journalArticle",
        "journal article": "journalArticle",
        "journal_article": "journalArticle",
        "article": "journalArticle",
        "conferencepaper": "conferencePaper",
        "conference paper": "conferencePaper",
        "conference_paper": "conferencePaper",
        "proceedings-article": "conferencePaper",
        "inproceedings": "conferencePaper",
        "booksection": "bookSection",
        "book section": "bookSection",
        "book_section": "bookSection",
        "book-chapter": "bookSection",
        "review-article": "review",
        "review article": "review",
    }

    type_counts = Counter()
    type_articles: dict[str, list[dict]] = {}

    for art in articles:
        doc_type = art.get(type_field, "unknown") or "unknown"
        # Normalize: strip, collapse whitespace, handle nan
        doc_type = doc_type.strip()
        if not doc_type or doc_type == "nan":
            doc_type = "unknown"
        # Canonicalize via case-insensitive lookup
        canon = _TYPE_CANONICAL.get(doc_type.lower().replace("_", "").replace("-", "").replace(" ", ""))
        if canon:
            doc_type = canon
        type_counts[doc_type] += 1
        type_articles.setdefault(doc_type, []).append(art)

    eligible = {}
    discarded = {}

    for doc_type, count in type_counts.most_common():
        if count >= min_size:
            eligible[doc_type] = type_articles[doc_type]
        else:
            discarded[doc_type] = count

    logger.info(f"Strata detected: {len(type_counts)} types total")
    logger.info(
        f"  Eligible (>= {min_size}): "
        + ", ".join(f"{t}({len(a)})" for t, a in eligible.items())
    )
    if discarded:
        logger.info(
            f"  Discarded: "
            + ", ".join(f"{t}({c})" for t, c in discarded.items())
        )

    return eligible, discarded


# ===========================================================================
# Per-Stratum Phase Wrappers
# ===========================================================================

def _make_stratum_dirs(stratum_name: str) -> dict[str, Path]:
    """Create output directories for a stratum."""
    dirs = {
        "figures": FIGURES_DIR / STRATIFIED_SUBDIR / stratum_name,
        "reports": REPORTS_DIR / STRATIFIED_SUBDIR / stratum_name,
        "plotly": PLOTLY_DIR / STRATIFIED_SUBDIR / stratum_name,
        "networks": NETWORKS_DIR / STRATIFIED_SUBDIR / stratum_name,
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def _run_stratum_phase2(
    articles: list[dict],
    stratum_name: str,
    thresholds: StratumThresholds,
) -> Optional[dict]:
    """Run descriptive analysis for a stratum."""
    if len(articles) < thresholds.descriptive:
        logger.info(f"  [{stratum_name}] Skipping Phase 2: {len(articles)} < {thresholds.descriptive}")
        return None

    dirs = _make_stratum_dirs(stratum_name)
    logger.info(f"  [{stratum_name}] Running Phase 2 ({len(articles)} articles)...")

    try:
        from src.descriptive_analysis import (
            lotka_analysis, bradford_analysis, temporal_trends,
            impact_analysis, keyword_analysis, affiliation_country_analysis,
            generate_descriptive_report,
        )

        results = {}
        results["lotka"] = lotka_analysis(articles, output_dir=dirs["figures"])
        results["bradford"] = bradford_analysis(articles, output_dir=dirs["figures"])
        results["temporal"] = temporal_trends(articles, output_dir=dirs["figures"])
        results["impact"] = impact_analysis(articles, output_dir=dirs["figures"])
        results["keywords"] = keyword_analysis(articles, output_dir=dirs["figures"])
        results["geography"] = affiliation_country_analysis(articles, output_dir=dirs["figures"])
        generate_descriptive_report(results, output_dir=dirs["reports"])

        logger.info(f"  [{stratum_name}] Phase 2 complete")
        return results

    except Exception as e:
        logger.error(f"  [{stratum_name}] Phase 2 failed: {e}")
        return None


def _run_stratum_phase2b(
    articles: list[dict],
    stratum_name: str,
    thresholds: StratumThresholds,
) -> Optional[dict]:
    """Run abstract analysis for a stratum."""
    if len(articles) < thresholds.abstract_analysis:
        return None

    dirs = _make_stratum_dirs(stratum_name)
    logger.info(f"  [{stratum_name}] Running Phase 2b ({len(articles)} articles)...")

    try:
        from src.abstract_analysis import (
            extract_methods_datasets, extract_performance_metrics,
            simple_topic_modeling, extract_gaps_from_abstracts,
            plot_topics, plot_topic_evolution, plot_methods_datasets,
            plot_performance_summary, plot_method_application_heatmap,
        )

        # Extract per-article data
        for a in articles:
            if "extracted_methods_datasets" not in a:
                a["extracted_methods_datasets"] = extract_methods_datasets(a.get("abstract", ""))
            if "performance_metrics" not in a:
                a["performance_metrics"] = extract_performance_metrics(a.get("abstract", ""))

        topic_data = simple_topic_modeling(articles, n_topics=min(5, max(2, len(articles) // 10)))
        gaps = extract_gaps_from_abstracts(articles)

        # Plots with redirected output
        plot_topics(topic_data, output_dir=dirs["figures"])
        plot_topic_evolution(articles, output_dir=dirs["figures"])
        plot_methods_datasets(articles, output_dir=dirs["figures"])
        plot_performance_summary(articles, output_dir=dirs["figures"])
        plot_method_application_heatmap(articles, output_dir=dirs["figures"])

        logger.info(f"  [{stratum_name}] Phase 2b complete")
        return {"topics": topic_data, "gaps": gaps}

    except Exception as e:
        logger.error(f"  [{stratum_name}] Phase 2b failed: {e}")
        return None


def _run_stratum_phase3(
    articles: list[dict],
    stratum_name: str,
    thresholds: StratumThresholds,
) -> Optional[dict]:
    """Run network analysis for a stratum (lightweight version)."""
    if len(articles) < thresholds.networks:
        return None

    dirs = _make_stratum_dirs(stratum_name)
    logger.info(f"  [{stratum_name}] Running Phase 3 networks ({len(articles)} articles)...")

    try:
        from src.network_analysis import (
            build_keyword_cooccurrence_network,
            build_coauthorship_network,
            detect_communities,
            export_network,
        )

        networks = {}

        # Keyword co-occurrence
        G_kw = build_keyword_cooccurrence_network(articles, min_cooccurrence=2)
        if G_kw.number_of_nodes() > 2:
            detect_communities(G_kw)
            export_network(G_kw, f"keyword_cooccurrence_{stratum_name}", output_dir=dirs["networks"])
            networks["keyword_cooccurrence"] = {
                "nodes": G_kw.number_of_nodes(),
                "edges": G_kw.number_of_edges(),
            }

        # Co-authorship
        G_ca = build_coauthorship_network(articles, level="authors")
        if G_ca.number_of_nodes() > 2:
            detect_communities(G_ca)
            export_network(G_ca, f"coauthorship_{stratum_name}", output_dir=dirs["networks"])
            networks["coauthorship"] = {
                "nodes": G_ca.number_of_nodes(),
                "edges": G_ca.number_of_edges(),
            }

        logger.info(f"  [{stratum_name}] Phase 3 complete: {len(networks)} networks")
        return networks

    except Exception as e:
        logger.error(f"  [{stratum_name}] Phase 3 failed: {e}")
        return None


def _collect_stratum_metrics(
    articles: list[dict],
    stratum_name: str,
    phase2_results: Optional[dict],
    phase2b_results: Optional[dict],
    phase3_results: Optional[dict],
) -> dict:
    """Collect key metrics for a stratum for the comparative report."""
    metrics = {
        "name": stratum_name,
        "count": len(articles),
    }

    # Basic metrics always available
    years = [a["publication_year"] for a in articles if a.get("publication_year")]
    cites = [a.get("cited_by_count", 0) or 0 for a in articles]

    metrics["year_range"] = f"{min(years)}-{max(years)}" if years else "N/A"
    metrics["mean_citations"] = round(sum(cites) / max(1, len(cites)), 2)
    metrics["median_citations"] = round(float(np.median(cites)), 2) if cites else 0
    metrics["total_citations"] = sum(cites)
    metrics["h_index"] = _calc_h_index(cites)

    # Authors
    all_authors = set()
    for a in articles:
        for auth in a.get("authors", []):
            all_authors.add(auth)
    metrics["unique_authors"] = len(all_authors)
    metrics["authors_per_article"] = round(
        sum(len(a.get("authors", [])) for a in articles) / max(1, len(articles)), 2
    )

    # Journals
    journals = set(a.get("publication_title", "") for a in articles if a.get("publication_title"))
    metrics["unique_journals"] = len(journals)

    # OA
    oa_count = sum(1 for a in articles if (a.get("open_access") or {}).get("is_oa"))
    metrics["oa_percentage"] = round(oa_count / max(1, len(articles)) * 100, 1)

    # SJR quartiles
    q_dist = Counter()
    for a in articles:
        q = a.get("sjr_quartile", "")
        if q:
            q_dist[q] += 1
    metrics["sjr_quartile_dist"] = dict(q_dist) if q_dist else {}

    # Keywords diversity
    all_kw = set()
    for a in articles:
        for t in a.get("all_tags", []):
            all_kw.add(t.lower().strip())
    metrics["unique_keywords"] = len(all_kw)

    # From phase2 results
    if phase2_results:
        lotka = phase2_results.get("lotka", {})
        metrics["top_author"] = lotka.get("top_authors", [("N/A", 0)])[0][0] if lotka.get("top_authors") else "N/A"
        bradford = phase2_results.get("bradford", {})
        metrics["core_journals"] = len(bradford.get("core_journals", [])) if isinstance(bradford.get("core_journals"), list) else 0

    # From phase2b
    if phase2b_results:
        metrics["topics_found"] = len(phase2b_results.get("topics", {}).get("topics", []))
        metrics["gaps_declared"] = len(phase2b_results.get("gaps", []))

    # From phase3
    if phase3_results:
        for net_name, net_info in phase3_results.items():
            if isinstance(net_info, dict):
                metrics[f"network_{net_name}_nodes"] = net_info.get("nodes", 0)
                metrics[f"network_{net_name}_edges"] = net_info.get("edges", 0)

    return metrics


def _calc_h_index(citations: list[int]) -> int:
    """Calculate h-index from a list of citation counts."""
    sorted_cites = sorted(citations, reverse=True)
    h = 0
    for i, c in enumerate(sorted_cites):
        if c >= i + 1:
            h = i + 1
        else:
            break
    return h


# ===========================================================================
# Comparative Report
# ===========================================================================

def generate_comparative_report(
    all_metrics: list[dict],
    discarded: dict[str, int],
    output_path: Path = None,
) -> Path:
    """Generate markdown report comparing metrics across strata."""
    if output_path is None:
        output_path = REPORTS_DIR / "stratified_comparative_report.md"

    lines = [
        "# Stratified Analysis: Comparative Report by Document Type",
        "",
        "---",
        "",
    ]

    # Summary table
    lines.append("## Overview")
    lines.append("")
    lines.append("| Stratum | Count | Year Range | Mean Cites | h-index | Unique Authors | Unique Journals | OA % |")
    lines.append("|---------|-------|------------|------------|---------|----------------|-----------------|------|")
    for m in all_metrics:
        lines.append(
            f"| {m['name']} | {m['count']} | {m['year_range']} | "
            f"{m['mean_citations']} | {m['h_index']} | "
            f"{m['unique_authors']} | {m['unique_journals']} | {m['oa_percentage']}% |"
        )
    lines.append("")

    # Discarded strata
    if discarded:
        lines.append("### Discarded Strata (below minimum size)")
        for name, count in discarded.items():
            lines.append(f"  - {name}: {count} articles")
        lines.append("")

    # Detailed comparison
    lines.append("## Detailed Comparison")
    lines.append("")

    for m in all_metrics:
        lines.append(f"### {m['name']} ({m['count']} articles)")
        lines.append("")
        lines.append(f"- **Year range**: {m['year_range']}")
        lines.append(f"- **Total citations**: {m['total_citations']}")
        lines.append(f"- **Mean citations**: {m['mean_citations']}")
        lines.append(f"- **Median citations**: {m['median_citations']}")
        lines.append(f"- **h-index**: {m['h_index']}")
        lines.append(f"- **Authors per article**: {m['authors_per_article']}")
        lines.append(f"- **Unique keywords**: {m['unique_keywords']}")

        if m.get("sjr_quartile_dist"):
            q_str = ", ".join(f"{k}: {v}" for k, v in sorted(m["sjr_quartile_dist"].items()))
            lines.append(f"- **SJR Quartiles**: {q_str}")

        if m.get("top_author", "N/A") != "N/A":
            lines.append(f"- **Top author**: {m['top_author']}")
        if m.get("topics_found"):
            lines.append(f"- **Topics discovered**: {m['topics_found']}")
        if m.get("gaps_declared"):
            lines.append(f"- **Declared gaps**: {m['gaps_declared']}")

        # Network info
        for key in sorted(m.keys()):
            if key.startswith("network_") and key.endswith("_nodes"):
                net_name = key.replace("network_", "").replace("_nodes", "")
                edges_key = f"network_{net_name}_edges"
                lines.append(
                    f"- **Network {net_name}**: "
                    f"{m[key]} nodes, {m.get(edges_key, 0)} edges"
                )

        lines.append("")

    # Visualizations generated
    lines.append("## Output Structure")
    lines.append("")
    lines.append("```")
    lines.append("output/")
    for m in all_metrics:
        name = m["name"]
        lines.append(f"  figures/{STRATIFIED_SUBDIR}/{name}/")
        lines.append(f"  reports/{STRATIFIED_SUBDIR}/{name}/")
    lines.append(f"  reports/stratified_comparative_report.md")
    lines.append("```")
    lines.append("")

    lines.append("---")
    lines.append("_Generated by Stratified Analysis Engine_")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Comparative report saved: {output_path}")
    return output_path


def _plot_comparative_charts(all_metrics: list[dict]) -> None:
    """Generate comparative visualizations across strata."""
    if len(all_metrics) < 2:
        return

    comp_dir = FIGURES_DIR / STRATIFIED_SUBDIR
    comp_dir.mkdir(parents=True, exist_ok=True)

    names = [m["name"] for m in all_metrics]
    x = range(len(names))

    # 1. Citation comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Mean citations
    vals = [m["mean_citations"] for m in all_metrics]
    axes[0].bar(x, vals, color=sns.color_palette("deep"))
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    axes[0].set_title("Mean Citations")
    axes[0].set_ylabel("Citations")

    # h-index
    vals = [m["h_index"] for m in all_metrics]
    axes[1].bar(x, vals, color=sns.color_palette("deep")[1:])
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    axes[1].set_title("h-index")

    # Article count
    vals = [m["count"] for m in all_metrics]
    axes[2].bar(x, vals, color=sns.color_palette("deep")[2:])
    axes[2].set_xticks(list(x))
    axes[2].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    axes[2].set_title("Article Count")

    plt.tight_layout()
    fig.savefig(comp_dir / "comparative_citations.png")
    plt.close(fig)
    logger.info("Saved: comparative_citations.png")

    # 2. OA percentage comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    vals = [m["oa_percentage"] for m in all_metrics]
    bars = ax.bar(x, vals, color=sns.color_palette("deep")[3:])
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_title("Open Access Percentage by Document Type")
    ax.set_ylabel("% Open Access")
    ax.set_ylim(0, 100)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val}%", ha="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(comp_dir / "comparative_oa.png")
    plt.close(fig)
    logger.info("Saved: comparative_oa.png")

    # 3. Authors & journals radar (if >= 3 strata)
    if len(all_metrics) >= 3:
        _plot_strata_radar(all_metrics, comp_dir / "comparative_radar.png")


def _plot_strata_radar(all_metrics: list[dict], output_path: Path) -> None:
    """Radar chart comparing strata across multiple dimensions."""
    categories = ["Count", "Mean Cites", "h-index", "Authors", "Journals", "Keywords"]
    raw_values = []
    for m in all_metrics:
        raw_values.append([
            m["count"],
            m["mean_citations"],
            m["h_index"],
            m["unique_authors"],
            m["unique_journals"],
            m["unique_keywords"],
        ])

    # Normalize to 0-1 per category
    arr = np.array(raw_values, dtype=float)
    maxes = arr.max(axis=0)
    maxes[maxes == 0] = 1
    norm = arr / maxes

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += [angles[0]]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = sns.color_palette("deep", len(all_metrics))

    for i, m in enumerate(all_metrics):
        values = norm[i].tolist() + [norm[i][0]]
        ax.fill(angles, values, alpha=0.1, color=colors[i])
        ax.plot(angles, values, "o-", linewidth=2, label=m["name"], color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_title("Strata Comparison Radar", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info(f"Saved: {output_path.name}")


# ===========================================================================
# Main Orchestrator
# ===========================================================================

def run_stratified_analysis(
    articles: list[dict],
    min_stratum_size: int = 15,
    type_field: str = "item_type",
) -> dict:
    """Run stratified analysis across document types.

    Args:
        articles: Full article list.
        min_stratum_size: Minimum articles for a stratum to be analyzed.
        type_field: Field name containing document type.

    Returns:
        dict with strata results and comparative metrics.
    """
    logger.info("=" * 60)
    logger.info("STRATIFIED ANALYSIS BY DOCUMENT TYPE")
    logger.info("=" * 60)

    # Detect strata
    eligible, discarded = detect_strata(articles, min_stratum_size, type_field)

    if len(eligible) < 2:
        logger.info(
            "Stratified analysis skipped: need at least 2 eligible strata "
            f"(found {len(eligible)})"
        )
        return {"skipped": True, "reason": "insufficient_strata", "eligible": len(eligible)}

    thresholds = StratumThresholds.from_min_size(min_stratum_size)
    all_metrics = []
    strata_results = {}

    for stratum_name, stratum_articles in eligible.items():
        logger.info(f"\n--- Stratum: {stratum_name} ({len(stratum_articles)} articles) ---")

        # Run phases
        p2 = _run_stratum_phase2(stratum_articles, stratum_name, thresholds)
        p2b = _run_stratum_phase2b(stratum_articles, stratum_name, thresholds)
        p3 = _run_stratum_phase3(stratum_articles, stratum_name, thresholds)

        # Collect metrics
        metrics = _collect_stratum_metrics(stratum_articles, stratum_name, p2, p2b, p3)
        all_metrics.append(metrics)

        strata_results[stratum_name] = {
            "count": len(stratum_articles),
            "phase2": p2 is not None,
            "phase2b": p2b is not None,
            "phase3": p3 is not None,
            "metrics": metrics,
        }

    # Generate comparative outputs
    _plot_comparative_charts(all_metrics)
    report_path = generate_comparative_report(all_metrics, discarded)

    logger.info("=" * 60)
    logger.info(
        f"Stratified analysis complete: {len(eligible)} strata analyzed, "
        f"{len(discarded)} discarded"
    )
    logger.info("=" * 60)

    return {
        "skipped": False,
        "strata_count": len(eligible),
        "discarded_count": len(discarded),
        "strata_results": strata_results,
        "report_path": str(report_path),
    }
