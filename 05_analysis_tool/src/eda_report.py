"""EDA Report: comprehensive exploratory data analysis metrics for the corpus."""
from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

from config import REPORTS_DIR

logger = logging.getLogger("bibliometrico.eda")


def compute_eda_metrics(articles: list[dict]) -> dict:
    """Compute all 24 EDA metrics in a single pass over articles.

    Returns:
        Dict with all computed metrics.
    """
    total_docs = len(articles)
    if total_docs == 0:
        logger.warning("No articles for EDA report")
        return {}

    author_counter = Counter()
    institution_counter = Counter()
    country_counter = Counter()
    source_counter = Counter()
    language_counter = Counter()
    author_kw_counter = Counter()
    plus_kw_counter = Counter()
    year_counter = Counter()
    all_references = set()

    total_citations = 0
    author_citations: dict[str, int] = {}
    institution_citations: dict[str, int] = {}
    source_citations: dict[str, int] = {}

    single_author_docs = 0
    multi_author_docs = 0

    # Per-article citation list for h-index
    citation_values = []

    for a in articles:
        # Authors
        authors = a.get("authors", [])
        n_authors = len(authors)
        if n_authors <= 1:
            single_author_docs += 1
        else:
            multi_author_docs += 1

        for auth in authors:
            author_counter[auth] += 1

        # Institutions and countries from OpenAlex
        openalex = a.get("openalex", {})
        article_institutions = set()
        article_countries = set()
        for authorship in openalex.get("authorships", []):
            for inst in authorship.get("institutions", []):
                name = inst.get("name", "")
                if name:
                    article_institutions.add(name)
                country = inst.get("country", "")
                if country:
                    article_countries.add(country)
            for c in authorship.get("countries", []):
                if c:
                    article_countries.add(c)

        for inst in article_institutions:
            institution_counter[inst] += 1
        for c in article_countries:
            country_counter[c] += 1

        # Source / journal
        source = a.get("publication_title", "")
        if source and source != "nan":
            source_counter[source] += 1

        # Language
        lang = a.get("language", "")
        if lang and lang != "nan":
            language_counter[lang] += 1

        # Keywords
        for kw in a.get("manual_tags", []):
            author_kw_counter[kw.lower().strip()] += 1
        for kw in a.get("automatic_tags", []):
            plus_kw_counter[kw.lower().strip()] += 1

        # Year
        year = a.get("publication_year")
        if year:
            year_counter[year] += 1

        # References
        for ref in a.get("referenced_works", []):
            all_references.add(ref)

        # Citations
        tc = a.get("cited_by_count", 0)
        total_citations += tc
        citation_values.append(tc)

        for auth in authors:
            author_citations[auth] = author_citations.get(auth, 0) + tc
        for inst in article_institutions:
            institution_citations[inst] = institution_citations.get(inst, 0) + tc
        if source and source != "nan":
            source_citations[source] = source_citations.get(source, 0) + tc

    # Derived metrics
    total_authors = len(author_counter)
    total_institutions = len(institution_counter)
    total_countries = len(country_counter)
    total_sources = len(source_counter)
    total_languages = len(language_counter)
    total_author_keywords = len(author_kw_counter)
    total_plus_keywords = len(plus_kw_counter)
    total_references = len(all_references)

    years = sorted(year_counter.keys()) if year_counter else []
    publication_interval = f"{min(years)}-{max(years)}" if years else "N/A"
    n_years = len(year_counter)

    avg_docs_per_author = round(total_docs / max(total_authors, 1), 2)
    avg_docs_per_institution = round(total_docs / max(total_institutions, 1), 2)
    avg_docs_per_source = round(total_docs / max(total_sources, 1), 2)
    avg_docs_per_year = round(total_docs / max(n_years, 1), 2)

    # Collaboration index = total authorships / total multi-author docs
    total_authorships = sum(len(a.get("authors", [])) for a in articles)
    collaboration_index = round(
        total_authorships / max(multi_author_docs, 1), 2
    )

    # H-index of the corpus
    sorted_cites = sorted(citation_values, reverse=True)
    h_index = 0
    for i, c in enumerate(sorted_cites, 1):
        if c >= i:
            h_index = i
        else:
            break

    avg_citations_per_doc = round(total_citations / max(total_docs, 1), 2)
    avg_citations_per_author = round(
        total_citations / max(total_authors, 1), 2
    )
    avg_citations_per_institution = round(
        total_citations / max(total_institutions, 1), 2
    )
    avg_citations_per_source = round(
        total_citations / max(total_sources, 1), 2
    )

    metrics = {
        "publication_interval": publication_interval,
        "total_documents": total_docs,
        "total_authors": total_authors,
        "total_institutions": total_institutions,
        "total_countries": total_countries,
        "total_sources": total_sources,
        "total_references": total_references,
        "total_languages": total_languages,
        "total_author_keywords": total_author_keywords,
        "total_keywords_plus": total_plus_keywords,
        "avg_docs_per_author": avg_docs_per_author,
        "avg_docs_per_institution": avg_docs_per_institution,
        "avg_docs_per_source": avg_docs_per_source,
        "avg_docs_per_year": avg_docs_per_year,
        "single_author_docs": single_author_docs,
        "multi_author_docs": multi_author_docs,
        "collaboration_index": collaboration_index,
        "max_h_index": h_index,
        "total_citations": total_citations,
        "avg_citations_per_document": avg_citations_per_doc,
        "avg_citations_per_author": avg_citations_per_author,
        "avg_citations_per_institution": avg_citations_per_institution,
        "avg_citations_per_source": avg_citations_per_source,
        "total_authorships": total_authorships,
    }

    return metrics


def generate_eda_report(metrics: dict, output_dir: Path = REPORTS_DIR) -> Path:
    """Generate a markdown table EDA report.

    Returns:
        Path to the generated report file.
    """
    if not metrics:
        logger.warning("No metrics to report")
        return output_dir / "eda_report.md"

    labels = {
        "publication_interval": "Publication Interval",
        "total_documents": "Total Documents",
        "total_authors": "Total Authors",
        "total_institutions": "Total Institutions",
        "total_countries": "Total Countries",
        "total_sources": "Total Sources / Journals",
        "total_references": "Total Unique References",
        "total_languages": "Total Languages",
        "total_author_keywords": "Unique Author Keywords",
        "total_keywords_plus": "Unique Keywords Plus",
        "avg_docs_per_author": "Avg Documents per Author",
        "avg_docs_per_institution": "Avg Documents per Institution",
        "avg_docs_per_source": "Avg Documents per Source",
        "avg_docs_per_year": "Avg Documents per Year",
        "single_author_docs": "Single-Author Documents",
        "multi_author_docs": "Multi-Author Documents",
        "collaboration_index": "Collaboration Index",
        "total_authorships": "Total Author Appearances",
        "max_h_index": "Corpus H-index",
        "total_citations": "Total Citations",
        "avg_citations_per_document": "Avg Citations per Document",
        "avg_citations_per_author": "Avg Citations per Author",
        "avg_citations_per_institution": "Avg Citations per Institution",
        "avg_citations_per_source": "Avg Citations per Source",
    }

    lines = [
        "# Exploratory Data Analysis (EDA) Report",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ]
    for key, label in labels.items():
        value = metrics.get(key, "N/A")
        lines.append(f"| {label} | {value} |")

    lines.append("")

    report_path = output_dir / "eda_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"EDA report saved: {report_path}")
    return report_path


def run_eda_report(articles: list[dict]) -> dict:
    """Entry point: compute EDA metrics and generate report.

    Returns:
        Dict of computed metrics.
    """
    logger.info("=== EDA Report ===")
    metrics = compute_eda_metrics(articles)
    generate_eda_report(metrics)
    logger.info("EDA report completed.")
    return metrics
