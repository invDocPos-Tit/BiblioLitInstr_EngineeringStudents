"""
Exportador de Corpus Open Access para Analisis Sistematico
===========================================================
Filtra articulos OA del corpus enriquecido y genera un listado curado
con metadatos para identificar PDFs disponibles para revision sistematica.

Outputs en output/oa_corpus/:
- oa_corpus_listing.csv: Listado completo de articulos OA
- oa_corpus_report.md: Reporte ejecutivo en Markdown
- oa_corpus_stats.json: Estadisticas machine-readable
- oa_distribution_by_color.png: Pie chart de colores OA
- oa_distribution_by_year.png: Barras apiladas por year y color OA
- oa_distribution_by_type.png: Barras agrupadas OA vs no-OA por item_type

VERSION: 1.0
"""
from __future__ import annotations

import csv
import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import OA_CORPUS_DIR, CSV_ENCODING
from src.utils import get_canonical_oa_status

logger = logging.getLogger("bibliometrico.oa_corpus_export")


def ensure_oa_dir():
    """Crear directorio de OA corpus si no existe."""
    OA_CORPUS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Filtrado de articulos OA
# ---------------------------------------------------------------------------

def filter_oa_articles(articles: list[dict]) -> list[dict]:
    """
    Filtrar articulos donde open_access.is_oa == True.
    Ordena por cited_by_count descendente.

    Returns:
        Lista de articulos OA ordenados por citaciones.
    """
    oa = []
    for article in articles:
        oa_data = article.get('open_access', {})
        if isinstance(oa_data, dict) and oa_data.get('is_oa'):
            oa.append(article)

    oa.sort(key=lambda a: a.get('cited_by_count', 0) or 0, reverse=True)
    return oa


def _get_oa_color(article: dict) -> str:
    """Extraer color OA normalizado (prioriza diamond de OpenAlex)."""
    status = get_canonical_oa_status(article)
    return status if status != "closed" else "unknown"


def _get_authors_str(article: dict) -> str:
    """Formatear autores como string."""
    authors = article.get('authors')
    if isinstance(authors, list):
        return '; '.join(
            a.get('name', '') if isinstance(a, dict) else str(a)
            for a in authors[:5]
        )
    if isinstance(authors, str):
        return authors
    return ''


# ---------------------------------------------------------------------------
# Exportacion CSV
# ---------------------------------------------------------------------------

def export_oa_corpus_csv(
    oa_articles: list[dict],
    output_path: Path = None,
) -> Path:
    """
    Exportar CSV con columnas clave para analisis sistematico.
    Usa utf-8-sig para compatibilidad Windows/Excel.
    """
    if output_path is None:
        output_path = OA_CORPUS_DIR / "oa_corpus_listing.csv"

    columns = [
        'key', 'title', 'authors', 'year', 'doi', 'journal', 'item_type',
        'oa_status', 'oa_color', 'best_oa_pdf_url', 'best_oa_url',
        'license', 'version', 'has_repository_copy', 'oa_locations_count',
        'journal_is_oa', 'journal_is_in_doaj',
        'cited_by_count', 'abstract_quality_score', 'abstract_quality_level',
    ]

    with open(output_path, 'w', newline='', encoding=CSV_ENCODING) as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()

        for article in oa_articles:
            oa = article.get('open_access', {})
            quality = article.get('abstract_quality', {})

            row = {
                'key': article.get('key', ''),
                'title': article.get('title', ''),
                'authors': _get_authors_str(article),
                'year': article.get('publication_year', ''),
                'doi': article.get('doi', ''),
                'journal': article.get('publication_title', ''),
                'item_type': article.get('item_type', ''),
                'oa_status': oa.get('status', ''),
                'oa_color': oa.get('oa_color', ''),
                'best_oa_pdf_url': oa.get('best_oa_pdf_url', ''),
                'best_oa_url': oa.get('best_oa_url', ''),
                'license': oa.get('best_oa_license', ''),
                'version': oa.get('best_oa_version', ''),
                'has_repository_copy': oa.get('has_repository_copy', ''),
                'oa_locations_count': oa.get('oa_locations_count', 0),
                'journal_is_oa': oa.get('journal_is_oa', ''),
                'journal_is_in_doaj': oa.get('journal_is_in_doaj', ''),
                'cited_by_count': article.get('cited_by_count', 0) or 0,
                'abstract_quality_score': quality.get('overall_score', '') if quality else '',
                'abstract_quality_level': quality.get('overall_quality', '') if quality else '',
            }
            writer.writerow(row)

    logger.info(f"  Generated: {output_path} ({len(oa_articles)} rows)")
    return output_path


# ---------------------------------------------------------------------------
# Estadisticas del corpus OA
# ---------------------------------------------------------------------------

def compute_oa_corpus_stats(
    oa_articles: list[dict],
    total_articles: int,
) -> dict:
    """
    Calcular estadisticas del corpus OA.
    """
    n_oa = len(oa_articles)

    # Distribucion por color
    color_counts = Counter(_get_oa_color(a) for a in oa_articles)

    # Distribucion por year
    year_color: dict[str, Counter] = {}
    for a in oa_articles:
        y = str(a.get('publication_year', 'unknown'))
        year_color.setdefault(y, Counter())[_get_oa_color(a)] += 1

    # Distribucion por item_type
    type_counts = Counter(a.get('item_type', 'unknown') for a in oa_articles)

    # Licencias
    license_counts = Counter(
        a.get('open_access', {}).get('best_oa_license', 'unknown') or 'unknown'
        for a in oa_articles
    )

    # PDF disponibilidad
    with_pdf = sum(1 for a in oa_articles if a.get('open_access', {}).get('best_oa_pdf_url'))

    # Repository copies
    with_repo = sum(1 for a in oa_articles if a.get('open_access', {}).get('has_repository_copy'))

    # Citaciones
    citations = [a.get('cited_by_count', 0) or 0 for a in oa_articles]

    stats = {
        'total_corpus': total_articles,
        'total_oa': n_oa,
        'pct_oa': round(n_oa / max(total_articles, 1) * 100, 2),
        'with_pdf_url': with_pdf,
        'pct_with_pdf': round(with_pdf / max(n_oa, 1) * 100, 2),
        'with_repository_copy': with_repo,
        'by_color': dict(color_counts.most_common()),
        'by_year': {y: dict(c) for y, c in sorted(year_color.items())},
        'by_item_type': dict(type_counts.most_common()),
        'by_license': dict(license_counts.most_common()),
        'citation_stats': {
            'mean': round(np.mean(citations), 2) if citations else 0,
            'median': round(float(np.median(citations)), 2) if citations else 0,
            'max': max(citations) if citations else 0,
            'total': sum(citations),
        },
    }

    return stats


# ---------------------------------------------------------------------------
# Visualizaciones
# ---------------------------------------------------------------------------

# Colores canonicos para OA
OA_COLORS = {
    'diamond': '#E8DAEF',
    'gold': '#FFD700',
    'green': '#4CAF50',
    'hybrid': '#FF9800',
    'bronze': '#CD7F32',
    'unknown': '#9E9E9E',
}


def generate_oa_color_distribution_plot(
    oa_articles: list[dict],
    output_path: Path = None,
) -> Optional[Path]:
    """Pie chart de colores OA."""
    if output_path is None:
        output_path = OA_CORPUS_DIR / "oa_distribution_by_color.png"

    color_counts = Counter(_get_oa_color(a) for a in oa_articles)
    if not color_counts:
        return None

    labels = list(color_counts.keys())
    sizes = list(color_counts.values())
    colors = [OA_COLORS.get(c, '#9E9E9E') for c in labels]
    explode = [0.05 if s == max(sizes) else 0 for s in sizes]

    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(
        sizes, explode=explode, labels=[f"{l.capitalize()}\n(n={s})" for l, s in zip(labels, sizes)],
        colors=colors, autopct='%1.1f%%', shadow=True, startangle=90,
        textprops={'fontsize': 10},
    )
    for t in autotexts:
        t.set_fontsize(10)
        t.set_fontweight('bold')

    ax.set_title('Open Access Color Distribution', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"  Generated: {output_path}")
    return output_path


def generate_oa_by_year_plot(
    oa_articles: list[dict],
    output_path: Path = None,
) -> Optional[Path]:
    """Barras apiladas por year y color OA."""
    if output_path is None:
        output_path = OA_CORPUS_DIR / "oa_distribution_by_year.png"

    year_color: dict[str, Counter] = {}
    for a in oa_articles:
        y = str(a.get('publication_year', 'unknown'))
        if y == 'unknown':
            continue
        year_color.setdefault(y, Counter())[_get_oa_color(a)] += 1

    if len(year_color) < 2:
        logger.info("  Skipping OA-by-year plot (< 2 years with data)")
        return None

    years = sorted(year_color.keys())
    all_colors = sorted({c for yc in year_color.values() for c in yc})

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(years))
    bottoms = np.zeros(len(years))

    for color in all_colors:
        counts = [year_color[y].get(color, 0) for y in years]
        ax.bar(x, counts, bottom=bottoms,
               color=OA_COLORS.get(color, '#9E9E9E'),
               edgecolor='black', label=color.capitalize())
        bottoms += np.array(counts)

    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('Number of OA Articles', fontsize=11)
    ax.set_title('Open Access Articles by Year and Color', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(years, rotation=45, fontsize=10)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"  Generated: {output_path}")
    return output_path


def generate_oa_by_type_plot(
    oa_articles: list[dict],
    total_articles: list[dict],
    output_path: Path = None,
) -> Optional[Path]:
    """Barras agrupadas OA vs no-OA por item_type."""
    if output_path is None:
        output_path = OA_CORPUS_DIR / "oa_distribution_by_type.png"

    # Count total by type
    total_by_type = Counter(a.get('item_type', 'unknown') for a in total_articles)
    oa_by_type = Counter(a.get('item_type', 'unknown') for a in oa_articles)

    types = sorted(total_by_type.keys())
    if not types:
        return None

    oa_counts = [oa_by_type.get(t, 0) for t in types]
    non_oa_counts = [total_by_type[t] - oa_by_type.get(t, 0) for t in types]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(types))
    width = 0.35

    ax.bar(x - width / 2, oa_counts, width, color='#4CAF50', edgecolor='black', label='Open Access')
    ax.bar(x + width / 2, non_oa_counts, width, color='#F44336', edgecolor='black', alpha=0.7, label='Non-OA')

    # Add counts on bars
    for i, (oa_c, noa_c) in enumerate(zip(oa_counts, non_oa_counts)):
        total = oa_c + noa_c
        pct = oa_c / max(total, 1) * 100
        ax.text(i, max(oa_c, noa_c) + 2, f'{pct:.0f}% OA', ha='center', fontsize=9, fontweight='bold')

    ax.set_xlabel('Document Type', fontsize=11)
    ax.set_ylabel('Number of Articles', fontsize=11)
    ax.set_title('Open Access vs Non-OA by Document Type', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(types, fontsize=10)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"  Generated: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Reporte Markdown
# ---------------------------------------------------------------------------

def generate_oa_corpus_report(
    oa_articles: list[dict],
    stats: dict,
    output_path: Path = None,
) -> Path:
    """Reporte MD: resumen ejecutivo, tablas, top citados, listado con PDF URLs."""
    if output_path is None:
        output_path = OA_CORPUS_DIR / "oa_corpus_report.md"

    lines = [
        "# Open Access Corpus Report for Systematic Analysis",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        f"- **Total corpus:** {stats['total_corpus']} articles",
        f"- **Open Access articles:** {stats['total_oa']} ({stats['pct_oa']}%)",
        f"- **With PDF URL:** {stats['with_pdf_url']} ({stats['pct_with_pdf']}% of OA)",
        f"- **With repository copy:** {stats['with_repository_copy']}",
        "",
        "---",
        "",
        "## Distribution by OA Color",
        "",
        "| Color | Count | % of OA |",
        "|-------|-------|---------|",
    ]

    for color, count in sorted(stats['by_color'].items(), key=lambda x: x[1], reverse=True):
        pct = count / max(stats['total_oa'], 1) * 100
        lines.append(f"| {color.capitalize()} | {count} | {pct:.1f}% |")

    lines.extend(["", "---", "", "## Distribution by Document Type", "",
                   "| Type | OA Count | Total | OA % |",
                   "|------|----------|-------|------|"])

    # We need total by type from stats
    total_by_type = {}
    for a in oa_articles:
        t = a.get('item_type', 'unknown')
        total_by_type[t] = total_by_type.get(t, 0)
    # Use the stats by_item_type for OA counts, total_corpus info isn't per-type here
    for itype, count in sorted(stats['by_item_type'].items(), key=lambda x: x[1], reverse=True):
        lines.append(f"| {itype} | {count} | - | - |")

    lines.extend(["", "---", "", "## Distribution by License", "",
                   "| License | Count | % of OA |",
                   "|---------|-------|---------|"])

    for lic, count in sorted(stats['by_license'].items(), key=lambda x: x[1], reverse=True):
        pct = count / max(stats['total_oa'], 1) * 100
        lines.append(f"| {lic} | {count} | {pct:.1f}% |")

    lines.extend(["", "---", "", "## Citation Statistics (OA Articles)", "",
                   f"- **Mean citations:** {stats['citation_stats']['mean']}",
                   f"- **Median citations:** {stats['citation_stats']['median']}",
                   f"- **Max citations:** {stats['citation_stats']['max']}",
                   f"- **Total citations:** {stats['citation_stats']['total']}",
                   ""])

    # Top cited OA articles
    lines.extend(["---", "", "## Top 20 Most Cited OA Articles", "",
                   "| # | Title | Year | Citations | OA Color | PDF |",
                   "|---|-------|------|-----------|----------|-----|"])

    for i, a in enumerate(oa_articles[:20], 1):
        title = (a.get('title', '')[:60] + '...') if len(a.get('title', '')) > 60 else a.get('title', '')
        year = a.get('publication_year', '')
        cites = a.get('cited_by_count', 0) or 0
        color = _get_oa_color(a)
        has_pdf = 'Yes' if a.get('open_access', {}).get('best_oa_pdf_url') else 'No'
        lines.append(f"| {i} | {title} | {year} | {cites} | {color} | {has_pdf} |")

    lines.extend([
        "",
        "---",
        "",
        "## Files Generated",
        "",
        "- `oa_corpus_listing.csv` - Full listing (open in Excel)",
        "- `oa_corpus_report.md` - This report",
        "- `oa_corpus_stats.json` - Machine-readable statistics",
        "- `oa_distribution_by_color.png` - OA color pie chart",
        "- `oa_distribution_by_year.png` - OA by year stacked bars",
        "- `oa_distribution_by_type.png` - OA vs non-OA by document type",
    ])

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    logger.info(f"  Generated: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

def export_oa_stats_json(
    stats: dict,
    output_path: Path = None,
) -> Path:
    """JSON con estadisticas machine-readable."""
    if output_path is None:
        output_path = OA_CORPUS_DIR / "oa_corpus_stats.json"

    result = {
        'generated_at': datetime.now().isoformat(),
        **stats,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info(f"  Generated: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Orquestador principal
# ---------------------------------------------------------------------------

def run_oa_corpus_export(articles: list[dict]) -> dict:
    """
    Orquestador principal: filtra articulos OA y genera todos los outputs.

    Args:
        articles: Lista completa de articulos enriquecidos.

    Returns:
        dict con rutas de archivos generados y estadisticas.
    """
    logger.info("=" * 60)
    logger.info("Phase 4b: Open Access Corpus Export")
    logger.info("=" * 60)

    ensure_oa_dir()

    # Filtrar OA
    oa_articles = filter_oa_articles(articles)
    logger.info(f"  OA articles: {len(oa_articles)} / {len(articles)}")

    if not oa_articles:
        logger.warning("  No OA articles found in corpus")
        return {'error': 'No OA articles found'}

    # Estadisticas
    stats = compute_oa_corpus_stats(oa_articles, len(articles))

    # Generar outputs
    files = {}

    files['csv'] = export_oa_corpus_csv(oa_articles)
    files['report'] = generate_oa_corpus_report(oa_articles, stats)
    files['stats_json'] = export_oa_stats_json(stats)
    files['color_plot'] = generate_oa_color_distribution_plot(oa_articles)
    files['year_plot'] = generate_oa_by_year_plot(oa_articles)
    files['type_plot'] = generate_oa_by_type_plot(oa_articles, articles)

    # Filtrar None
    files = {k: str(v) for k, v in files.items() if v is not None}

    logger.info(f"  Generated {len(files)} files in {OA_CORPUS_DIR}")

    return {
        'files': files,
        'total_oa': len(oa_articles),
        'pct_oa': stats['pct_oa'],
        'with_pdf': stats['with_pdf_url'],
    }
