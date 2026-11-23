"""Fase 2: Analisis descriptivo - Lotka, Bradford, tendencias, impacto."""
from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import FIGURES_DIR, REPORTS_DIR
from src.utils import (
    normalize_language,
    normalize_journal_name,
    canonicalize_corpus_journals,
    get_canonical_oa_status,
    is_noisy_legacy_concept,
)

logger = logging.getLogger("bibliometrico.descriptive")

# Estilo global
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight"})


def lotka_analysis(articles: list[dict], output_dir: Path = FIGURES_DIR) -> dict:
    """Ley de Lotka: productividad de autores."""
    author_counts = Counter()
    for a in articles:
        for author in a.get("authors", []):
            author_counts[author] += 1

    # Distribucion: cuantos autores publicaron n articulos
    prod_dist = Counter(author_counts.values())
    total_authors = len(author_counts)

    # Top autores
    top_authors = author_counts.most_common(15)

    # Grafico
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Barplot top autores
    names = [a[0][:25] for a in top_authors]
    counts = [a[1] for a in top_authors]
    axes[0].barh(range(len(names)), counts, color=sns.color_palette("deep"))
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels(names, fontsize=8)
    axes[0].set_xlabel("Number of Articles")
    axes[0].set_title("Most Productive Authors")
    axes[0].invert_yaxis()

    # Distribucion de productividad
    n_pubs = sorted(prod_dist.keys())
    n_authors = [prod_dist[n] for n in n_pubs]
    axes[1].bar(n_pubs, n_authors, color=sns.color_palette("deep")[1])
    axes[1].set_xlabel("Number of Publications")
    axes[1].set_ylabel("Number of Authors")
    axes[1].set_title("Lotka's Law: Author Productivity Distribution")

    plt.tight_layout()
    fig.savefig(output_dir / "lotka_author_productivity.png")
    plt.close(fig)
    logger.info("Saved: lotka_author_productivity.png")

    return {
        "total_authors": total_authors,
        "top_authors": top_authors,
        "productivity_distribution": dict(prod_dist),
    }


def bradford_analysis(articles: list[dict], output_dir: Path = FIGURES_DIR) -> dict:
    """Ley de Bradford: revistas nucleo y metricas SCImago SJR."""
    journal_counts = Counter()
    journal_sjr_meta: dict[str, dict] = {}
    for a in articles:
        journal = a.get("publication_title", "Unknown")
        if journal and journal != "nan":
            journal_counts[journal] += 1
            if a.get("sjr") is not None or a.get("sjr_quartile"):
                sc = a.get("scimago") if isinstance(a.get("scimago"), dict) else {}
                journal_sjr_meta[journal] = {
                    "sjr": sc.get("sjr_latest") or a.get("sjr"),
                    "quartile": sc.get("sjr_quartile_latest") or a.get("sjr_quartile", ""),
                }

    # Ordenar por frecuencia descendente
    sorted_journals = journal_counts.most_common()
    total_articles = sum(c for _, c in sorted_journals)

    # Zonas de Bradford (dividir en 3 zonas con ~igual numero de articulos)
    cumulative = 0
    zones = {"core": [], "zone2": [], "zone3": []}
    third = total_articles / 3.0

    for journal, count in sorted_journals:
        cumulative += count
        if cumulative <= third:
            zones["core"].append((journal, count))
        elif cumulative <= 2 * third:
            zones["zone2"].append((journal, count))
        else:
            zones["zone3"].append((journal, count))

    # Si core esta vacio, poner al menos el primero
    if not zones["core"] and sorted_journals:
        zones["core"].append(sorted_journals[0])

    # Distribucion de cuartiles SJR y estadisticos SJR
    quartile_counts = Counter()
    sjrs: list[float] = []
    diamond_scimago = 0
    for a in articles:
        q = a.get("sjr_quartile") or ""
        if q in ("Q1", "Q2", "Q3", "Q4"):
            quartile_counts[q] += 1
        sjr_val = a.get("sjr")
        if isinstance(sjr_val, (int, float)) and sjr_val > 0:
            sjrs.append(float(sjr_val))
        sc = a.get("scimago")
        if isinstance(sc, dict) and str(sc.get("open_access_diamond", "")).lower() == "yes":
            diamond_scimago += 1

    # Grafico
    fig, ax = plt.subplots(figsize=(12, 6))
    journals = [j[:40] for j, _ in sorted_journals[:20]]
    counts = [c for _, c in sorted_journals[:20]]
    colors = []
    cum = 0
    for _, c in sorted_journals[:20]:
        cum += c
        if cum <= third:
            colors.append(sns.color_palette("deep")[0])
        elif cum <= 2 * third:
            colors.append(sns.color_palette("deep")[1])
        else:
            colors.append(sns.color_palette("deep")[2])

    ax.barh(range(len(journals)), counts, color=colors)
    ax.set_yticks(range(len(journals)))
    ax.set_yticklabels(journals, fontsize=8)
    ax.set_xlabel("Number of Articles")
    ax.set_title("Bradford's Law: Journal Distribution (Core=Blue, Zone2=Orange, Zone3=Green)")
    ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(output_dir / "bradford_journals.png")
    plt.close(fig)
    logger.info("Saved: bradford_journals.png")

    return {
        "total_journals": len(journal_counts),
        "core_journals": zones["core"],
        "zone2_journals": len(zones["zone2"]),
        "zone3_journals": len(zones["zone3"]),
        "journal_counts": dict(sorted_journals),
        "journal_sjr_meta": journal_sjr_meta,
        "sjr_quartile_counts": {q: quartile_counts.get(q, 0) for q in ("Q1", "Q2", "Q3", "Q4") if quartile_counts.get(q, 0) > 0},
        "sjr_matched_articles": len(sjrs),
        "sjr_mean": round(sum(sjrs) / len(sjrs), 3) if sjrs else 0.0,
        "sjr_median": round(float(np.median(sjrs)), 3) if sjrs else 0.0,
        "scimago_diamond_articles": diamond_scimago,
    }


def temporal_trends(articles: list[dict], output_dir: Path = FIGURES_DIR) -> dict:
    """Evolucion temporal de publicaciones."""
    years = [a["publication_year"] for a in articles if a.get("publication_year")]
    year_counts = Counter(years)

    if not year_counts:
        logger.warning("No publication years found")
        return {}

    min_year = min(year_counts.keys())
    max_year = max(year_counts.keys())
    all_years = list(range(min_year, max_year + 1))
    counts = [year_counts.get(y, 0) for y in all_years]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(all_years, counts, color=sns.color_palette("deep")[0], alpha=0.8)
    ax.plot(all_years, counts, "o-", color=sns.color_palette("deep")[3], linewidth=2)
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Publications")
    ax.set_title("Publication Trend Over Time")
    ax.set_xticks(all_years)
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    fig.savefig(output_dir / "temporal_trends.png")
    plt.close(fig)
    logger.info("Saved: temporal_trends.png")

    return {
        "year_counts": dict(year_counts),
        "min_year": min_year,
        "max_year": max_year,
        "total_years_span": max_year - min_year + 1,
    }


def impact_analysis(articles: list[dict], output_dir: Path = FIGURES_DIR) -> dict:
    """Metricas de impacto."""
    citations = [(a["key"], a.get("cited_by_count", 0), a["title"][:50]) for a in articles]
    citations.sort(key=lambda x: x[1], reverse=True)

    total_cites = sum(c for _, c, _ in citations)
    mean_cites = total_cites / len(citations) if citations else 0

    # h-index del corpus
    sorted_cites = sorted([c for _, c, _ in citations], reverse=True)
    h_index = 0
    for i, c in enumerate(sorted_cites, 1):
        if c >= i:
            h_index = i
        else:
            break

    # Grafico top cited
    fig, ax = plt.subplots(figsize=(12, 6))
    top_n = min(15, len(citations))
    labels = [f"[{k}] {t}" for k, _, t in citations[:top_n]]
    values = [c for _, c, _ in citations[:top_n]]

    ax.barh(range(top_n), values, color=sns.color_palette("deep")[2])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Citation Count")
    ax.set_title(f"Most Cited Articles (h-index={h_index})")
    ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(output_dir / "impact_citations.png")
    plt.close(fig)
    logger.info("Saved: impact_citations.png")

    return {
        "total_citations": total_cites,
        "mean_citations": round(mean_cites, 2),
        "h_index": h_index,
        "top_cited": citations[:10],
    }


def keyword_analysis(articles: list[dict], output_dir: Path = FIGURES_DIR) -> dict:
    """Frecuencia de keywords."""
    keyword_counts = Counter()
    for a in articles:
        for tag in a.get("manual_tags", []):
            keyword_counts[tag.lower()] += 1

    # También contar automatic tags por separado
    auto_counts = Counter()
    for a in articles:
        for tag in a.get("automatic_tags", []):
            auto_counts[tag.lower()] += 1

    # Grafico top keywords (manual)
    top_kw = keyword_counts.most_common(25)
    if top_kw:
        fig, ax = plt.subplots(figsize=(12, 7))
        words = [k[:35] for k, _ in top_kw]
        freqs = [c for _, c in top_kw]
        ax.barh(range(len(words)), freqs, color=sns.color_palette("deep")[4])
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words, fontsize=8)
        ax.set_xlabel("Frequency")
        ax.set_title("Top Author Keywords")
        ax.invert_yaxis()
        plt.tight_layout()
        fig.savefig(output_dir / "keyword_frequency.png")
        plt.close(fig)
        logger.info("Saved: keyword_frequency.png")

    return {
        "total_unique_keywords": len(keyword_counts),
        "top_keywords": top_kw[:20],
        "total_unique_auto_tags": len(auto_counts),
    }


def affiliation_country_analysis(articles: list[dict], output_dir: Path = FIGURES_DIR) -> dict:
    """Estadisticas de afiliaciones y paises."""
    country_counts = Counter()
    institution_counts = Counter()

    from src.modelo_corpus import article_countries, article_entry
    for a in articles:
        # Paises e instituciones POR ARTICULO, con la definicion unica del modelo del corpus
        # (antes se contaba una vez por cada afiliacion de cada autor).
        for c in article_countries(a):
            country_counts[c] += 1
        entrada = article_entry(a)
        insts = entrada["instituciones"] if entrada is not None else sorted({
            inst.get("name", "") for auth in (a.get("openalex") or {}).get("authorships", [])
            for inst in auth.get("institutions", []) if inst.get("name")})
        for name in insts:
            institution_counts[name] += 1

    # Grafico de paises
    if country_counts:
        top_countries = country_counts.most_common(15)
        fig, ax = plt.subplots(figsize=(10, 5))
        countries = [c for c, _ in top_countries]
        counts = [n for _, n in top_countries]
        ax.bar(countries, counts, color=sns.color_palette("deep")[5])
        ax.set_xlabel("Country Code")
        ax.set_ylabel("Documents")
        ax.set_title("Geographic Distribution of Research")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        fig.savefig(output_dir / "country_distribution.png")
        plt.close(fig)
        logger.info("Saved: country_distribution.png")

    return {
        "total_countries": len(country_counts),
        "country_distribution": dict(country_counts.most_common(20)),
        "total_institutions": len(institution_counts),
        "top_institutions": institution_counts.most_common(10),
    }


def country_quality_analysis(articles: list[dict], output_dir: Path = FIGURES_DIR) -> dict:
    """3-panel analysis: total cites, avg cites, h-index por pais (Fig 2a-c Cao et al.)."""
    # Agrupar articulos por pais
    country_data: dict[str, list[int]] = {}  # pais -> lista de citation counts
    for a in articles:
        openalex = a.get("openalex", {})
        article_countries = set()
        for auth in openalex.get("authorships", []):
            for inst in auth.get("institutions", []):
                country = inst.get("country", "")
                if country:
                    article_countries.add(country)
        tc = a.get("cited_by_count", 0)
        for c in article_countries:
            country_data.setdefault(c, []).append(tc)

    if not country_data:
        logger.warning("No country data for quality analysis")
        return {}

    # Calcular metricas por pais
    rows = []
    for country, cites_list in country_data.items():
        total_cites = sum(cites_list)
        n_articles = len(cites_list)
        avg_cites = total_cites / n_articles if n_articles else 0
        # h-index por pais
        sorted_c = sorted(cites_list, reverse=True)
        h = 0
        for i, c in enumerate(sorted_c, 1):
            if c >= i:
                h = i
            else:
                break
        rows.append({
            "country": country,
            "n_articles": n_articles,
            "total_cites": total_cites,
            "avg_cites": round(avg_cites, 2),
            "h_index": h,
        })

    df = pd.DataFrame(rows).sort_values("total_cites", ascending=False)
    top = df.head(15)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Total cited frequency
    axes[0].barh(range(len(top)), top["total_cites"].values, color=sns.color_palette("deep")[0])
    axes[0].set_yticks(range(len(top)))
    axes[0].set_yticklabels(top["country"].values, fontsize=8)
    axes[0].set_xlabel("Total Citations")
    axes[0].set_title("Total Cited Frequency by Country")
    axes[0].invert_yaxis()

    # Panel 2: Average citations per article
    top_avg = df.nlargest(15, "avg_cites")
    axes[1].barh(range(len(top_avg)), top_avg["avg_cites"].values, color=sns.color_palette("deep")[1])
    axes[1].set_yticks(range(len(top_avg)))
    axes[1].set_yticklabels(top_avg["country"].values, fontsize=8)
    axes[1].set_xlabel("Average Citations per Article")
    axes[1].set_title("Average Citations by Country")
    axes[1].invert_yaxis()

    # Panel 3: H-index
    top_h = df.nlargest(15, "h_index")
    axes[2].barh(range(len(top_h)), top_h["h_index"].values, color=sns.color_palette("deep")[2])
    axes[2].set_yticks(range(len(top_h)))
    axes[2].set_yticklabels(top_h["country"].values, fontsize=8)
    axes[2].set_xlabel("H-index")
    axes[2].set_title("H-index by Country")
    axes[2].invert_yaxis()

    plt.tight_layout()
    fig.savefig(output_dir / "country_quality.png")
    plt.close(fig)
    logger.info("Saved: country_quality.png")

    return {
        "top_by_total_cites": df.head(10).to_dict("records"),
        "top_by_avg_cites": df.nlargest(10, "avg_cites").to_dict("records"),
        "top_by_h_index": df.nlargest(10, "h_index").to_dict("records"),
    }


def rri_analysis(articles: list[dict], output_dir: Path = FIGURES_DIR) -> dict:
    """Relative Research Interest: dual-axis bars=N publications, line=RRI (Fig 1a Cao et al.)."""
    years = [a["publication_year"] for a in articles if a.get("publication_year")]
    if not years:
        logger.warning("No publication years for RRI analysis")
        return {}

    year_counts = Counter(years)
    total_corpus = len(articles)

    min_year = min(year_counts.keys())
    max_year = max(year_counts.keys())
    all_years = list(range(min_year, max_year + 1))
    counts = [year_counts.get(y, 0) for y in all_years]
    rri = [year_counts.get(y, 0) / total_corpus for y in all_years]

    fig, ax1 = plt.subplots(figsize=(12, 5))

    # Barras: N publicaciones
    ax1.bar(all_years, counts, color=sns.color_palette("deep")[0], alpha=0.7, label="Number of Articles")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Number of Articles", color=sns.color_palette("deep")[0])
    ax1.tick_params(axis="y", labelcolor=sns.color_palette("deep")[0])
    ax1.set_xticks(all_years)
    ax1.tick_params(axis="x", rotation=45)

    # Linea: RRI
    ax2 = ax1.twinx()
    ax2.plot(all_years, rri, "o-", color=sns.color_palette("deep")[3], linewidth=2.5,
             markersize=6, label="RRI")
    ax2.set_ylabel("Relative Research Interest (RRI)", color=sns.color_palette("deep")[3])
    ax2.tick_params(axis="y", labelcolor=sns.color_palette("deep")[3])

    # Leyenda combinada
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax1.set_title("Publication Trend and Relative Research Interest (RRI)")
    plt.tight_layout()
    fig.savefig(output_dir / "rri.png")
    plt.close(fig)
    logger.info("Saved: rri.png")

    return {
        "year_counts": dict(year_counts),
        "rri_by_year": {y: round(r, 4) for y, r in zip(all_years, rri)},
    }


def funding_analysis(articles: list[dict], output_dir: Path = FIGURES_DIR) -> dict:
    """Top funding agencies from all sources: OpenAlex grants, Crossref funders, CSV."""
    funder_counts = Counter()
    funded_articles = 0
    source_counts = Counter()

    for a in articles:
        article_funders = set()

        # Source 1: funding[] (aggregated from CSV, OpenAlex, Crossref)
        for f in a.get("funding", []):
            name = f.get("name", "").strip()
            if name and len(name) > 2:
                article_funders.add(name)
                source_counts[f.get("source", "unknown")] += 1

        # Source 2: OpenAlex grants (may have been merged already, but check)
        for grant in a.get("openalex", {}).get("grants", []):
            name = (grant.get("funder_display_name", "") or "").strip()
            if name and len(name) > 2:
                article_funders.add(name)

        for name in article_funders:
            funder_counts[name] += 1
        if article_funders:
            funded_articles += 1

    n_total = len(articles)
    funded_pct = round(funded_articles / max(n_total, 1) * 100, 1)

    if not funder_counts:
        logger.warning("No funding data found (openalex.grants + crossref + CSV)")
        return {"funded_articles": 0, "funded_pct": 0}

    top_funders = funder_counts.most_common(20)

    fig, ax = plt.subplots(figsize=(12, 7))
    names = [f[:55] for f, _ in top_funders[:15]]
    counts = [c for _, c in top_funders[:15]]
    bars = ax.barh(range(len(names)), counts, color=sns.color_palette("deep")[4])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Number of Funded Articles")
    ax.set_title(f"Top Funding Agencies ({funded_articles}/{n_total} articles funded, {funded_pct}%)")
    ax.invert_yaxis()
    # Add count labels
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                str(count), va="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_dir / "funding.png")
    plt.close(fig)
    logger.info("Saved: funding.png")

    return {
        "top_funders": top_funders,
        "funded_articles": funded_articles,
        "funded_pct": funded_pct,
        "total_unique_funders": len(funder_counts),
        "sources": dict(source_counts),
    }


def sdg_analysis(articles: list[dict], output_dir: Path = FIGURES_DIR) -> dict:
    """Sustainable Development Goals distribution from OpenAlex."""
    sdg_counts = Counter()
    sdg_scores = {}  # {sdg_name: [scores]}
    articles_with_sdg = 0

    for a in articles:
        sdgs = a.get("openalex", {}).get("sustainable_development_goals", [])
        if sdgs:
            articles_with_sdg += 1
        for sdg in sdgs:
            name = sdg.get("display_name", "")
            score = sdg.get("score", 0)
            if name:
                sdg_counts[name] += 1
                sdg_scores.setdefault(name, []).append(score)

    if not sdg_counts:
        logger.info("No SDG data found — skipping sdg_analysis")
        return {}

    top_sdgs = sdg_counts.most_common(17)  # Max 17 SDGs
    sdg_pct = round(articles_with_sdg / max(len(articles), 1) * 100, 1)

    # Bar chart: SDG frequency
    fig, ax = plt.subplots(figsize=(14, 7))
    names = [s[:50] for s, _ in top_sdgs]
    counts = [c for _, c in top_sdgs]
    colors = sns.color_palette("husl", len(names))
    bars = ax.barh(range(len(names)), counts, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Number of Articles")
    ax.set_title(f"Sustainable Development Goals ({articles_with_sdg}/{len(articles)} articles, {sdg_pct}%)")
    ax.invert_yaxis()
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                str(count), va="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_dir / "sdg_distribution.png")
    plt.close(fig)
    logger.info("Saved: sdg_distribution.png")

    # Mean score per SDG
    sdg_mean_scores = {
        name: round(sum(scores) / len(scores), 3)
        for name, scores in sdg_scores.items()
    }

    return {
        "top_sdgs": top_sdgs,
        "articles_with_sdg": articles_with_sdg,
        "sdg_pct": sdg_pct,
        "sdg_mean_scores": sdg_mean_scores,
    }


def primary_topic_analysis(articles: list[dict], output_dir: Path = FIGURES_DIR) -> dict:
    """Primary topic distribution from OpenAlex (domain > field > subfield > topic)."""
    domain_counts = Counter()
    field_counts = Counter()
    subfield_counts = Counter()
    topic_counts = Counter()
    articles_with_topic = 0

    for a in articles:
        pt = a.get("openalex", {}).get("primary_topic", {}) or {}
        if pt.get("name"):
            articles_with_topic += 1
            topic_counts[pt["name"]] += 1
            if pt.get("subfield"):
                subfield_counts[pt["subfield"]] += 1
            if pt.get("field"):
                field_counts[pt["field"]] += 1
            if pt.get("domain"):
                domain_counts[pt["domain"]] += 1

    if not topic_counts:
        logger.info("No primary_topic data found — skipping")
        return {}

    topic_pct = round(articles_with_topic / max(len(articles), 1) * 100, 1)

    # Chart: Top fields (mid-level granularity)
    top_fields = field_counts.most_common(15)
    if top_fields:
        fig, ax = plt.subplots(figsize=(12, 7))
        names = [f[:50] for f, _ in top_fields]
        counts = [c for _, c in top_fields]
        bars = ax.barh(range(len(names)), counts, color=sns.color_palette("Set2", len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("Number of Articles")
        ax.set_title(f"Research Fields (OpenAlex Primary Topic, {topic_pct}% coverage)")
        ax.invert_yaxis()
        for bar, count in zip(bars, counts):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    str(count), va="center", fontsize=8)
        plt.tight_layout()
        fig.savefig(output_dir / "primary_topic_fields.png")
        plt.close(fig)
        logger.info("Saved: primary_topic_fields.png")

    # Chart: Domain pie
    if domain_counts:
        fig, ax = plt.subplots(figsize=(8, 8))
        labels = [d for d, _ in domain_counts.most_common()]
        sizes = [c for _, c in domain_counts.most_common()]
        ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90,
               colors=sns.color_palette("pastel", len(labels)))
        ax.set_title("Research Domains (OpenAlex)")
        plt.tight_layout()
        fig.savefig(output_dir / "primary_topic_domains.png")
        plt.close(fig)
        logger.info("Saved: primary_topic_domains.png")

    return {
        "top_domains": domain_counts.most_common(),
        "top_fields": top_fields,
        "top_subfields": subfield_counts.most_common(20),
        "top_topics": topic_counts.most_common(20),
        "articles_with_topic": articles_with_topic,
        "topic_pct": topic_pct,
    }


def language_analysis(articles: list[dict], output_dir: Path = FIGURES_DIR) -> dict:
    """Publication language distribution."""
    lang_counts = Counter()
    for a in articles:
        lang = normalize_language(a.get("language", "") or "")
        if lang and lang not in ("", "nan"):
            a["language"] = lang
            lang_counts[lang] += 1

    if not lang_counts:
        return {}

    top_langs = lang_counts.most_common(15)

    fig, ax = plt.subplots(figsize=(10, 5))
    names = [l for l, _ in top_langs]
    counts = [c for _, c in top_langs]
    ax.bar(names, counts, color=sns.color_palette("deep")[2])
    ax.set_xlabel("Language")
    ax.set_ylabel("Articles")
    ax.set_title("Publication Language Distribution")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(output_dir / "language_distribution.png")
    plt.close(fig)
    logger.info("Saved: language_distribution.png")

    return {"top_languages": top_langs, "total_languages": len(lang_counts)}


def apc_analysis(articles: list[dict], output_dir: Path = FIGURES_DIR) -> dict:
    """Article Processing Charge analysis."""
    apc_values = []
    oa_types = Counter()

    for a in articles:
        status = get_canonical_oa_status(a)
        if status:
            oa_types[status] += 1

        apc = a.get("openalex", {}).get("apc_paid", {}) or {}
        val = apc.get("value_usd", 0)
        if val and val > 0:
            apc_values.append(val)

    result = {"oa_types": dict(oa_types)}

    if apc_values:
        result["apc_mean_usd"] = round(sum(apc_values) / len(apc_values), 0)
        result["apc_median_usd"] = round(sorted(apc_values)[len(apc_values) // 2], 0)
        result["apc_count"] = len(apc_values)
        result["apc_total_usd"] = round(sum(apc_values), 0)

    # OA status chart
    if oa_types:
        fig, ax = plt.subplots(figsize=(8, 6))
        labels = list(oa_types.keys())
        sizes = list(oa_types.values())
        colors = {"gold": "#FFD700", "green": "#2ECC71", "bronze": "#CD7F32",
                  "hybrid": "#3498DB", "closed": "#95A5A6", "diamond": "#E8DAEF"}
        chart_colors = [colors.get(l, "#BDC3C7") for l in labels]
        ax.pie(sizes, labels=[f"{l} ({c})" for l, c in zip(labels, sizes)],
               autopct="%1.1f%%", colors=chart_colors, startangle=90)
        ax.set_title("Open Access Status Distribution")
        plt.tight_layout()
        fig.savefig(output_dir / "oa_status.png")
        plt.close(fig)
        logger.info("Saved: oa_status.png")

    return result


def research_orientations(articles: list[dict], output_dir: Path = FIGURES_DIR) -> dict:
    """Top research topics/concepts from multiple sources with fallback chain."""
    concept_counts = Counter()
    for a in articles:
        openalex = a.get("openalex", {})
        # Source 1: Topics (CWTS Leiden OpenAlex hierarchy — preferred)
        topics = a.get("topics") or openalex.get("topics", [])
        has_topics = False
        for topic in topics:
            name = (topic.get("display_name") or topic.get("name") or "").strip()
            if name:
                concept_counts[name] += 1
                has_topics = True
        if has_topics:
            continue
        # Source 2: Legacy concepts (only as fallback when topics are unavailable, filtered for noise)
        for concept in openalex.get("concepts", []):
            level = concept.get("level", 99)
            score = concept.get("score", 0)
            name = (concept.get("display_name") or "").strip()
            if name and level <= 1 and score > 0.3 and not is_noisy_legacy_concept(name):
                concept_counts[name] += 1
        # Source 3: Article-level concepts (enriched, outside openalex dict)
        for concept in a.get("concepts", []):
            name = (concept.get("name") or concept.get("display_name") or "").strip()
            score = concept.get("score", 0) or 0
            level = concept.get("level", 99)
            if name and float(score) >= 0.3 and int(level) <= 1 and not is_noisy_legacy_concept(name):
                concept_counts[name] += 1

    # Fallback: author keywords + keywords plus when concepts are sparse
    if len(concept_counts) < 5:
        logger.info("Sparse concepts — enriching with author_keywords and keywords_plus")
        for a in articles:
            for kw_field in ("author_keywords", "keywords_plus", "index_keywords", "manual_tags"):
                kw_raw = a.get(kw_field, "") or ""
                if isinstance(kw_raw, str) and kw_raw.strip():
                    for kw in kw_raw.split(";"):
                        kw = kw.strip()
                        if kw and len(kw) > 2:
                            concept_counts[kw] += 1

    if not concept_counts:
        logger.warning("No topics/concepts found in OpenAlex data or keywords")
        return {}

    top_concepts = concept_counts.most_common(15)

    fig, ax = plt.subplots(figsize=(12, 6))
    names = [c[:45] for c, _ in top_concepts]
    counts = [n for _, n in top_concepts]
    ax.barh(range(len(names)), counts, color=sns.color_palette("deep")[5])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Frequency")
    ax.set_title("Research Orientations (Top Topics/Concepts)")
    ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(output_dir / "orientations.png")
    plt.close(fig)
    logger.info("Saved: orientations.png")

    return {"top_concepts": top_concepts}


def database_overlap_analysis(articles: list[dict], output_dir: Path = FIGURES_DIR) -> dict:
    """Analiza cobertura multi-base, solapamiento e Indice de Unicidad de Meyer desde Merged_From / Source_DB."""
    n = len(articles)
    if n == 0:
        return {}

    combo_counts = Counter()
    db_total_counts = Counter()
    db_exclusive_counts = Counter()
    db_meyer_weights: dict[str, float] = {}
    primary_source_counts = Counter()

    for a in articles:
        src_db = (a.get("source_db") or "").strip()
        if src_db and src_db != "nan":
            primary_source_counts[src_db] += 1

        mf = (a.get("merged_from") or src_db or "").strip()
        if not mf or mf == "nan":
            continue
        dbs = sorted({d.strip() for d in re.split(r"[+;,\s]+", mf) if d.strip()})
        if not dbs:
            continue

        combo_label = "+".join(dbs)
        combo_counts[combo_label] += 1
        k = len(dbs)
        for db in dbs:
            db_total_counts[db] += 1
            db_meyer_weights[db] = db_meyer_weights.get(db, 0.0) + (1.0 / k)
            if k == 1:
                db_exclusive_counts[db] += 1

    if not db_total_counts:
        return {}

    n_tagged = sum(combo_counts.values())
    n_untagged = max(0, n - n_tagged)
    total_raw_appearances = sum(db_total_counts.values())
    duplicates_merged = max(0, total_raw_appearances - n_tagged)
    redundancy_rate = round(duplicates_merged / total_raw_appearances * 100, 1) if total_raw_appearances > 0 else 0.0
    multi_db_articles = sum(cnt for combo, cnt in combo_counts.items() if "+" in combo)
    denom_tagged = max(n_tagged, 1)

    db_metrics = []
    for db, total_cnt in db_total_counts.most_common():
        excl_cnt = db_exclusive_counts.get(db, 0)
        meyer_idx = round(db_meyer_weights.get(db, 0.0) / total_cnt, 3) if total_cnt > 0 else 0.0
        rel_cov_corpus = round(total_cnt / n * 100, 1)
        rel_cov_tagged = round(total_cnt / denom_tagged * 100, 1)
        excl_pct = round(excl_cnt / n * 100, 1)
        excl_tagged_pct = round(excl_cnt / denom_tagged * 100, 1)
        db_metrics.append({
            "database": db,
            "total_records": total_cnt,
            "relative_coverage_pct": rel_cov_corpus,
            "relative_coverage_tagged_pct": rel_cov_tagged,
            "exclusive_records": excl_cnt,
            "exclusive_corpus_pct": excl_pct,
            "exclusive_tagged_pct": excl_tagged_pct,
            "meyer_uniqueness_index": meyer_idx,
        })

    # Grafico de cobertura total vs exclusiva por base de datos
    fig, ax = plt.subplots(figsize=(9, 4.5))
    dbs_sorted = [m["database"] for m in db_metrics]
    totals = [m["total_records"] for m in db_metrics]
    exclusives = [m["exclusive_records"] for m in db_metrics]
    shared = [t - e for t, e in zip(totals, exclusives)]

    x = np.arange(len(dbs_sorted))
    ax.bar(x, exclusives, label="Exclusive Records (Single DB)", color=sns.color_palette("deep")[0])
    ax.bar(x, shared, bottom=exclusives, label="Shared Records (Multi-DB Overlap)", color=sns.color_palette("deep")[1])
    ax.set_xticks(x)
    ax.set_xticklabels(dbs_sorted)
    ax.set_ylabel("Number of Consolidated Articles")
    ax.set_title(
        f"Multi-DB Overlap (N={n_tagged} DB-tagged / {n} total; {total_raw_appearances} DB hits, {duplicates_merged} duplicates merged)"
    )
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "database_overlap.png")
    plt.close(fig)
    logger.info("Saved: database_overlap.png")

    return {
        "consolidated_unique": n,
        "tagged_unique_articles": n_tagged,
        "untagged_articles": n_untagged,
        "total_raw_db_hits": total_raw_appearances,
        "inter_db_duplicates_merged": duplicates_merged,
        "redundancy_rate_pct": redundancy_rate,
        "multi_db_articles": multi_db_articles,
        "multi_db_pct": round(multi_db_articles / n * 100, 1),
        "multi_db_tagged_pct": round(multi_db_articles / denom_tagged * 100, 1),
        "database_metrics": db_metrics,
        "combination_counts": dict(combo_counts.most_common()),
        "primary_source_counts": dict(primary_source_counts.most_common()),
    }


def generate_descriptive_report(results: dict, output_dir: Path = REPORTS_DIR) -> None:
    """Genera reporte Markdown con resultados descriptivos."""
    lines = ["# Descriptive Bibliometric Analysis Report\n"]

    if "lotka" in results:
        r = results["lotka"]
        lines.append("## Author Productivity (Lotka's Law)")
        lines.append(f"- Total unique authors: {r['total_authors']}")
        lines.append("- Top 10 authors:")
        for name, count in r["top_authors"][:10]:
            lines.append(f"  - {name}: {count} articles")
        lines.append("")

    if "bradford" in results:
        r = results["bradford"]
        sjr_meta = r.get("journal_sjr_meta", {})
        lines.append("## Journal Distribution (Bradford's Law & SCImago SJR)")
        lines.append(f"- Total journals: {r['total_journals']}")
        if r.get("sjr_matched_articles"):
            lines.append(
                f"- Articles in SCImago-indexed journals: {r['sjr_matched_articles']} "
                f"(mean SJR={r.get('sjr_mean', 0):.3f}, median SJR={r.get('sjr_median', 0):.3f})"
            )
        if r.get("sjr_quartile_counts"):
            q_str = ", ".join(f"{q}: {cnt}" for q, cnt in r["sjr_quartile_counts"].items())
            lines.append(f"- SCImago Best Quartile distribution (publication year matched): {q_str}")
        if r.get("scimago_diamond_articles"):
            lines.append(f"- Articles in SCImago Diamond OA journals: {r['scimago_diamond_articles']}")
        lines.append("- Core journals:")
        for name, count in r["core_journals"]:
            jm = sjr_meta.get(name, {})
            if jm.get("sjr") or jm.get("quartile"):
                sjr_lbl = f" [SJR={jm['sjr']:.3f}, {jm['quartile']}]" if jm.get("sjr") else f" [{jm['quartile']}]"
            else:
                sjr_lbl = ""
            lines.append(f"  - {name}: {count} articles{sjr_lbl}")
        lines.append("")

    if "temporal" in results:
        r = results["temporal"]
        lines.append("## Temporal Trends")
        lines.append(f"- Year range: {r.get('min_year')}-{r.get('max_year')}")
        for year in sorted(r.get("year_counts", {}).keys()):
            lines.append(f"  - {year}: {r['year_counts'][year]} articles")
        lines.append("")

    if "impact" in results:
        r = results["impact"]
        lines.append("## Impact Metrics")
        lines.append(f"- Total citations: {r['total_citations']}")
        lines.append(f"- Mean citations: {r['mean_citations']}")
        lines.append(f"- h-index: {r['h_index']}")
        lines.append("")

    if "keywords" in results:
        r = results["keywords"]
        lines.append("## Keyword Analysis")
        lines.append(f"- Unique author keywords: {r['total_unique_keywords']}")
        lines.append("- Top keywords:")
        for kw, count in r.get("top_keywords", [])[:15]:
            lines.append(f"  - {kw}: {count}")
        lines.append("")

    if "geography" in results:
        r = results["geography"]
        lines.append("## Geographic Distribution")
        lines.append(f"- Countries represented: {r['total_countries']}")
        lines.append(f"- Institutions represented: {r['total_institutions']}")
        lines.append("")

    if "country_quality" in results:
        r = results["country_quality"]
        lines.append("## Country Quality Metrics")
        if r.get("top_by_total_cites"):
            lines.append("- Top by total citations:")
            for row in r["top_by_total_cites"][:5]:
                lines.append(f"  - {row['country']}: {row['total_cites']} cites, h={row['h_index']}")
        lines.append("")

    if "rri" in results:
        r = results["rri"]
        lines.append("## Relative Research Interest")
        for year, rri_val in sorted(r.get("rri_by_year", {}).items()):
            lines.append(f"  - {year}: RRI={rri_val}")
        lines.append("")

    if "funding" in results:
        r = results["funding"]
        lines.append("## Funding Analysis")
        lines.append(f"- Articles with funding: {r.get('funded_articles', 0)} ({r.get('funded_pct', 0)}%)")
        lines.append(f"- Unique funders: {r.get('total_unique_funders', 0)}")
        if r.get("sources"):
            lines.append(f"- Data sources: {', '.join(f'{k}({v})' for k, v in r['sources'].items())}")
        lines.append("- Top funders:")
        for name, count in r.get("top_funders", [])[:10]:
            lines.append(f"  - {name}: {count} articles")
        lines.append("")

    if "sdg" in results and results["sdg"]:
        r = results["sdg"]
        lines.append("## Sustainable Development Goals (SDGs)")
        lines.append(f"- Articles with SDG mapping: {r.get('articles_with_sdg', 0)} ({r.get('sdg_pct', 0)}%)")
        lines.append("- Top SDGs:")
        for name, count in r.get("top_sdgs", [])[:10]:
            score = r.get("sdg_mean_scores", {}).get(name, 0)
            lines.append(f"  - {name}: {count} articles (mean score={score})")
        lines.append("")

    if "primary_topic" in results and results["primary_topic"]:
        r = results["primary_topic"]
        lines.append("## Research Fields (OpenAlex Primary Topic)")
        lines.append(f"- Articles with topic: {r.get('articles_with_topic', 0)} ({r.get('topic_pct', 0)}%)")
        if r.get("top_domains"):
            lines.append("- Domains:")
            for name, count in r["top_domains"][:5]:
                lines.append(f"  - {name}: {count}")
        if r.get("top_fields"):
            lines.append("- Top fields:")
            for name, count in r["top_fields"][:10]:
                lines.append(f"  - {name}: {count}")
        lines.append("")

    if "language" in results and results["language"]:
        r = results["language"]
        lines.append("## Publication Languages")
        lines.append(f"- Total languages: {r.get('total_languages', 0)}")
        for lang, count in r.get("top_languages", [])[:10]:
            lines.append(f"  - {lang}: {count}")
        lines.append("")

    if "apc" in results and results["apc"]:
        r = results["apc"]
        lines.append("## Open Access & APC")
        if r.get("oa_types"):
            lines.append("- OA status distribution:")
            for status, count in sorted(r["oa_types"].items(), key=lambda x: -x[1]):
                lines.append(f"  - {status}: {count}")
        if r.get("apc_count"):
            lines.append(f"- Articles with APC data: {r['apc_count']}")
            lines.append(f"- Mean APC: ${r.get('apc_mean_usd', 0):.0f} USD")
            lines.append(f"- Median APC: ${r.get('apc_median_usd', 0):.0f} USD")
            lines.append(f"- Total APC spend: ${r.get('apc_total_usd', 0):.0f} USD")
        lines.append("")

    if "orientations" in results:
        r = results["orientations"]
        lines.append("## Research Orientations")
        for name, count in r.get("top_concepts", [])[:10]:
            lines.append(f"  - {name}: {count}")
        lines.append("")

    if "database_overlap" in results and results["database_overlap"]:
        r = results["database_overlap"]
        lines.append("## Multi-Database Coverage & Overlap (PRISMA 2020 & Meyer's Uniqueness Index)")
        n_tag = r.get("tagged_unique_articles", r["consolidated_unique"])
        n_untag = r.get("untagged_articles", 0)
        lines.append(
            f"- Consolidated unique articles: {r['consolidated_unique']} "
            f"({n_tag} with explicit Scopus/WoS citation-export provenance from {r['total_raw_db_hits']} raw database hits; "
            f"{r['inter_db_duplicates_merged']} inter-database duplicates merged, "
            f"redundancy rate={r['redundancy_rate_pct']}%; {n_untag} additional consolidated CSV records)"
        )
        lines.append(
            f"- Multi-database overlap: {r['multi_db_articles']} articles "
            f"({r.get('multi_db_tagged_pct', r['multi_db_pct'])}% of DB-tagged subset / {r['multi_db_pct']}% of total corpus) "
            "indexed in >= 2 databases"
        )
        lines.append("- Coverage and Meyer's Uniqueness Index by database:")
        for m in r.get("database_metrics", []):
            lines.append(
                f"  - {m['database']}: {m['total_records']} records "
                f"({m.get('relative_coverage_tagged_pct', m['relative_coverage_pct'])}% of DB-tagged / {m['relative_coverage_pct']}% of total corpus), "
                f"{m['exclusive_records']} exclusive ({m['exclusive_corpus_pct']}%), "
                f"Meyer's Index={m['meyer_uniqueness_index']:.3f}"
            )
        lines.append("- Database intersection breakdown (`Merged_From`):")
        for combo, cnt in list(r.get("combination_counts", {}).items())[:15]:
            lines.append(f"  - {combo}: {cnt} articles")
        lines.append("")

    report_path = output_dir / "descriptive_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Report saved: {report_path}")


def run_phase2(articles: list[dict]) -> dict:
    """Ejecuta la Fase 2 completa."""
    logger.info("=== FASE 2: Analisis Descriptivo ===")

    # Ensure canonical journal names and language names before descriptive statistics
    canonicalize_corpus_journals(articles)
    for a in articles:
        if a.get("language"):
            a["language"] = normalize_language(str(a["language"]))

    results = {}
    results["database_overlap"] = database_overlap_analysis(articles)
    results["lotka"] = lotka_analysis(articles)
    results["bradford"] = bradford_analysis(articles)
    results["temporal"] = temporal_trends(articles)
    results["impact"] = impact_analysis(articles)
    results["keywords"] = keyword_analysis(articles)
    results["geography"] = affiliation_country_analysis(articles)
    results["country_quality"] = country_quality_analysis(articles)
    results["rri"] = rri_analysis(articles)
    results["funding"] = funding_analysis(articles)
    results["sdg"] = sdg_analysis(articles)
    results["primary_topic"] = primary_topic_analysis(articles)
    results["language"] = language_analysis(articles)
    results["apc"] = apc_analysis(articles)
    results["orientations"] = research_orientations(articles)

    generate_descriptive_report(results)
    logger.info("Fase 2 completada.")
    return results
