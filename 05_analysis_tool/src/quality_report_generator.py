"""
Generador de Reportes de Calidad de Abstracts
==============================================
Genera reportes visuales y estadisticos sobre la calidad de abstracts
evaluados en la Fase 2d.

Outputs:
- abstract_quality_report.md: Reporte detallado en Markdown
- quality_distribution.png: Histograma de distribucion de scores
- quality_by_dimension.png: Comparacion de dimensiones de calidad
- filtered_corpus_stats.json: Estadisticas del corpus filtrado

VERSION: 1.0
"""
from __future__ import annotations

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

from config import OUTPUT_DIR, FIGURES_DIR, REPORTS_DIR

logger = logging.getLogger("bibliometrico.quality_report")

# Directorio para reportes de calidad
QUALITY_DIR = Path(OUTPUT_DIR) / "quality"


def ensure_quality_dir():
    """Crear directorio de calidad si no existe."""
    QUALITY_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Extraccion de metricas de calidad
# ---------------------------------------------------------------------------

def extract_quality_metrics(articles: list[dict]) -> dict:
    """
    Extraer metricas de calidad de los articulos.

    Args:
        articles: Lista de articulos con campo 'abstract_quality'

    Returns:
        dict con metricas agregadas
    """
    metrics = {
        'total': len(articles),
        'evaluated': 0,
        'scores': {
            'overall': [],
            'completeness': [],
            'structure': [],
            'informativeness': [],
            'readability': [],
        },
        'quality_levels': {'high': 0, 'medium': 0, 'low': 0},
        'flags': Counter(),
        'by_year': {},
        'by_source': {},
    }

    for article in articles:
        quality = article.get('abstract_quality')
        if not quality:
            continue

        metrics['evaluated'] += 1

        # Scores
        metrics['scores']['overall'].append(quality.get('overall_score', 0))
        metrics['scores']['completeness'].append(quality.get('completeness_score', 0))
        metrics['scores']['structure'].append(quality.get('structure_score', 0))
        metrics['scores']['informativeness'].append(quality.get('informativeness_score', 0))
        metrics['scores']['readability'].append(quality.get('readability_score', 0))

        # Niveles de calidad
        level = quality.get('overall_quality', 'low')
        metrics['quality_levels'][level] = metrics['quality_levels'].get(level, 0) + 1

        # Flags
        for flag in quality.get('quality_flags', []):
            metrics['flags'][flag] += 1

        # Por año
        year = article.get('year') or article.get('PY')
        if year:
            year = str(year)
            if year not in metrics['by_year']:
                metrics['by_year'][year] = []
            metrics['by_year'][year].append(quality.get('overall_score', 0))

        # Por fuente/revista
        source = article.get('source') or article.get('SO') or article.get('journal')
        if source:
            if source not in metrics['by_source']:
                metrics['by_source'][source] = []
            metrics['by_source'][source].append(quality.get('overall_score', 0))

    return metrics


def compute_statistics(scores: list[float]) -> dict:
    """Calcular estadisticas descriptivas."""
    if not scores:
        return {'mean': 0, 'median': 0, 'std': 0, 'min': 0, 'max': 0, 'n': 0}

    return {
        'mean': round(np.mean(scores), 2),
        'median': round(np.median(scores), 2),
        'std': round(np.std(scores), 2),
        'min': round(min(scores), 2),
        'max': round(max(scores), 2),
        'n': len(scores),
    }


# ---------------------------------------------------------------------------
# Generacion de figuras
# ---------------------------------------------------------------------------

def generate_quality_distribution_plot(
    metrics: dict,
    output_path: Path = None,
) -> Path:
    """
    Generar histograma de distribucion de scores de calidad.

    Args:
        metrics: Metricas extraidas
        output_path: Ruta de salida (opcional)

    Returns:
        Path del archivo generado
    """
    if output_path is None:
        output_path = QUALITY_DIR / "quality_distribution.png"

    scores = metrics['scores']['overall']
    if not scores:
        logger.warning("No quality scores to plot")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histograma de scores
    ax1 = axes[0]
    ax1.hist(scores, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(np.mean(scores), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(scores):.1f}')
    ax1.axvline(np.median(scores), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(scores):.1f}')
    ax1.set_xlabel('Quality Score', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Distribution of Abstract Quality Scores', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.set_xlim(0, 100)

    # Pie chart de niveles
    ax2 = axes[1]
    levels = metrics['quality_levels']
    sizes = [levels.get('high', 0), levels.get('medium', 0), levels.get('low', 0)]
    labels = ['High', 'Medium', 'Low']
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    explode = (0.05, 0, 0)

    # Filtrar ceros
    non_zero = [(s, l, c, e) for s, l, c, e in zip(sizes, labels, colors, explode) if s > 0]
    if non_zero:
        sizes, labels, colors, explode = zip(*non_zero)
        ax2.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax2.set_title('Quality Level Distribution', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"  Generated: {output_path}")
    return output_path


def generate_dimension_comparison_plot(
    metrics: dict,
    output_path: Path = None,
) -> Path:
    """
    Generar comparacion de dimensiones de calidad.

    Args:
        metrics: Metricas extraidas
        output_path: Ruta de salida

    Returns:
        Path del archivo generado
    """
    if output_path is None:
        output_path = QUALITY_DIR / "quality_by_dimension.png"

    dimensions = ['completeness', 'structure', 'informativeness', 'readability']
    means = []
    stds = []

    for dim in dimensions:
        scores = metrics['scores'].get(dim, [])
        if scores:
            means.append(np.mean(scores))
            stds.append(np.std(scores))
        else:
            means.append(0)
            stds.append(0)

    if not any(means):
        logger.warning("No dimension scores to plot")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(dimensions))
    width = 0.6
    colors = ['#3498db', '#9b59b6', '#1abc9c', '#e67e22']

    bars = ax.bar(x, means, width, yerr=stds, capsize=5, color=colors, edgecolor='black', alpha=0.8)

    ax.set_ylabel('Score (0-100)', fontsize=11)
    ax.set_xlabel('Quality Dimension', fontsize=11)
    ax.set_title('Abstract Quality by Dimension', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in dimensions], fontsize=10)
    ax.set_ylim(0, 100)

    # Agregar valores sobre las barras
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax.annotate(f'{mean:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"  Generated: {output_path}")
    return output_path


def generate_flags_plot(
    metrics: dict,
    output_path: Path = None,
    top_n: int = 10,
) -> Path:
    """
    Generar grafico de flags de calidad mas comunes.

    Args:
        metrics: Metricas extraidas
        output_path: Ruta de salida
        top_n: Numero de flags a mostrar

    Returns:
        Path del archivo generado
    """
    if output_path is None:
        output_path = QUALITY_DIR / "quality_flags.png"

    flags = metrics['flags']
    if not flags:
        logger.warning("No quality flags to plot")
        return None

    # Top N flags
    top_flags = flags.most_common(top_n)
    if not top_flags:
        return None

    labels, counts = zip(*top_flags)

    # Formatear labels
    labels = [l.replace('_', ' ').title() for l in labels]

    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = np.arange(len(labels))
    colors = plt.cm.Reds(np.linspace(0.3, 0.8, len(labels)))

    bars = ax.barh(y_pos, counts, color=colors, edgecolor='black', alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Number of Abstracts', fontsize=11)
    ax.set_title('Most Common Quality Issues', fontsize=12, fontweight='bold')

    # Agregar valores
    for bar, count in zip(bars, counts):
        width = bar.get_width()
        ax.annotate(f'{count}',
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(3, 0),
                    textcoords="offset points",
                    ha='left', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"  Generated: {output_path}")
    return output_path


def generate_quality_by_year_plot(
    metrics: dict,
    output_path: Path = None,
) -> Path:
    """
    Generar grafico de calidad por año.

    Args:
        metrics: Metricas extraidas
        output_path: Ruta de salida

    Returns:
        Path del archivo generado
    """
    if output_path is None:
        output_path = QUALITY_DIR / "quality_by_year.png"

    by_year = metrics.get('by_year', {})
    if not by_year:
        logger.warning("No year data to plot")
        return None

    # Ordenar por año
    years = sorted(by_year.keys())
    means = [np.mean(by_year[y]) for y in years]
    counts = [len(by_year[y]) for y in years]

    if len(years) < 2:
        return None

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color1 = '#3498db'
    ax1.set_xlabel('Year', fontsize=11)
    ax1.set_ylabel('Mean Quality Score', color=color1, fontsize=11)
    line1 = ax1.plot(years, means, 'o-', color=color1, linewidth=2, markersize=8, label='Mean Score')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0, 100)

    # Segundo eje para conteo
    ax2 = ax1.twinx()
    color2 = '#e74c3c'
    ax2.set_ylabel('Number of Articles', color=color2, fontsize=11)
    bars = ax2.bar(years, counts, alpha=0.3, color=color2, label='Article Count')
    ax2.tick_params(axis='y', labelcolor=color2)

    ax1.set_title('Abstract Quality Trend by Year', fontsize=12, fontweight='bold')

    # Leyenda combinada
    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1 + [bars], ['Mean Score', 'Article Count'], loc='upper left')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"  Generated: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Generacion de reportes
# ---------------------------------------------------------------------------

def generate_markdown_report(
    articles: list[dict],
    metrics: dict,
    threshold: float = 50.0,
    output_path: Path = None,
) -> Path:
    """
    Generar reporte en formato Markdown.

    Args:
        articles: Lista de articulos
        metrics: Metricas extraidas
        threshold: Umbral de calidad usado
        output_path: Ruta de salida

    Returns:
        Path del archivo generado
    """
    if output_path is None:
        output_path = QUALITY_DIR / "abstract_quality_report.md"

    stats_overall = compute_statistics(metrics['scores']['overall'])
    stats_completeness = compute_statistics(metrics['scores']['completeness'])
    stats_structure = compute_statistics(metrics['scores']['structure'])
    stats_informativeness = compute_statistics(metrics['scores']['informativeness'])
    stats_readability = compute_statistics(metrics['scores']['readability'])

    levels = metrics['quality_levels']
    above_threshold = sum(1 for s in metrics['scores']['overall'] if s >= threshold)

    lines = [
        "# Abstract Quality Assessment Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Threshold:** {threshold}",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        f"- **Total articles:** {metrics['total']}",
        f"- **Evaluated:** {metrics['evaluated']}",
        f"- **Above threshold ({threshold}):** {above_threshold} ({above_threshold/max(metrics['evaluated'],1)*100:.1f}%)",
        "",
        "### Quality Distribution",
        "",
        f"- **High quality:** {levels.get('high', 0)} ({levels.get('high', 0)/max(metrics['evaluated'],1)*100:.1f}%)",
        f"- **Medium quality:** {levels.get('medium', 0)} ({levels.get('medium', 0)/max(metrics['evaluated'],1)*100:.1f}%)",
        f"- **Low quality:** {levels.get('low', 0)} ({levels.get('low', 0)/max(metrics['evaluated'],1)*100:.1f}%)",
        "",
        "---",
        "",
        "## Quality Scores by Dimension",
        "",
        "| Dimension | Mean | Median | Std | Min | Max |",
        "|-----------|------|--------|-----|-----|-----|",
        f"| **Overall** | {stats_overall['mean']} | {stats_overall['median']} | {stats_overall['std']} | {stats_overall['min']} | {stats_overall['max']} |",
        f"| Completeness | {stats_completeness['mean']} | {stats_completeness['median']} | {stats_completeness['std']} | {stats_completeness['min']} | {stats_completeness['max']} |",
        f"| Structure | {stats_structure['mean']} | {stats_structure['median']} | {stats_structure['std']} | {stats_structure['min']} | {stats_structure['max']} |",
        f"| Informativeness | {stats_informativeness['mean']} | {stats_informativeness['median']} | {stats_informativeness['std']} | {stats_informativeness['min']} | {stats_informativeness['max']} |",
        f"| Readability | {stats_readability['mean']} | {stats_readability['median']} | {stats_readability['std']} | {stats_readability['min']} | {stats_readability['max']} |",
        "",
        "---",
        "",
        "## Common Quality Issues",
        "",
    ]

    if metrics['flags']:
        lines.append("| Issue | Count | % of Evaluated |")
        lines.append("|-------|-------|----------------|")
        for flag, count in metrics['flags'].most_common(10):
            pct = count / max(metrics['evaluated'], 1) * 100
            flag_label = flag.replace('_', ' ').title()
            lines.append(f"| {flag_label} | {count} | {pct:.1f}% |")
    else:
        lines.append("No quality issues detected.")

    lines.extend([
        "",
        "---",
        "",
        "## Methodology",
        "",
        "Quality assessment evaluates abstracts across four dimensions:",
        "",
        "1. **Completeness (35% weight):** Presence of objective, methods, results, conclusion",
        "2. **Structure (25% weight):** Word count, sentence count, formatting",
        "3. **Informativeness (25% weight):** Technical density, lexical diversity, quantitative data",
        "4. **Readability (15% weight):** Flesch Reading Ease adapted for scientific text",
        "",
        "### Quality Levels",
        "",
        "- **High:** Overall score >= 70",
        "- **Medium:** Overall score 45-69",
        "- **Low:** Overall score < 45",
        "",
        "---",
        "",
        "## Files Generated",
        "",
        "- `abstract_quality_report.md` - This report",
        "- `quality_distribution.png` - Score distribution histogram",
        "- `quality_by_dimension.png` - Dimension comparison chart",
        "- `quality_flags.png` - Common issues chart",
        "- `quality_by_year.png` - Temporal trend (if year data available)",
        "- `filtered_corpus_stats.json` - Machine-readable statistics",
    ])

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    logger.info(f"  Generated: {output_path}")
    return output_path


def generate_json_stats(
    articles: list[dict],
    metrics: dict,
    threshold: float = 50.0,
    output_path: Path = None,
) -> Path:
    """
    Generar estadisticas en formato JSON.

    Args:
        articles: Lista de articulos
        metrics: Metricas extraidas
        threshold: Umbral de calidad
        output_path: Ruta de salida

    Returns:
        Path del archivo generado
    """
    if output_path is None:
        output_path = QUALITY_DIR / "filtered_corpus_stats.json"

    above_threshold = sum(1 for s in metrics['scores']['overall'] if s >= threshold)

    stats = {
        'generated_at': datetime.now().isoformat(),
        'threshold': threshold,
        'summary': {
            'total_articles': metrics['total'],
            'evaluated': metrics['evaluated'],
            'above_threshold': above_threshold,
            'below_threshold': metrics['evaluated'] - above_threshold,
            'pct_above_threshold': round(above_threshold / max(metrics['evaluated'], 1) * 100, 2),
        },
        'quality_levels': metrics['quality_levels'],
        'statistics': {
            'overall': compute_statistics(metrics['scores']['overall']),
            'completeness': compute_statistics(metrics['scores']['completeness']),
            'structure': compute_statistics(metrics['scores']['structure']),
            'informativeness': compute_statistics(metrics['scores']['informativeness']),
            'readability': compute_statistics(metrics['scores']['readability']),
        },
        'quality_flags': dict(metrics['flags'].most_common(20)),
        'by_year': {
            year: {
                'count': len(scores),
                'mean': round(np.mean(scores), 2) if scores else 0,
            }
            for year, scores in sorted(metrics.get('by_year', {}).items())
        },
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    logger.info(f"  Generated: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Funcion principal
# ---------------------------------------------------------------------------

def generate_quality_report(
    articles: list[dict],
    threshold: float = 50.0,
) -> dict:
    """
    Generar reporte completo de calidad de abstracts.

    Args:
        articles: Lista de articulos con campo 'abstract_quality'
        threshold: Umbral de calidad utilizado

    Returns:
        dict con rutas de archivos generados
    """
    logger.info("=" * 60)
    logger.info("Generating Abstract Quality Report")
    logger.info("=" * 60)

    ensure_quality_dir()

    # Extraer metricas
    metrics = extract_quality_metrics(articles)

    if metrics['evaluated'] == 0:
        logger.warning("No articles with quality assessment found")
        return {'error': 'No quality data available'}

    logger.info(f"  Evaluated articles: {metrics['evaluated']}")

    # Generar archivos
    files = {}

    # Figuras
    files['distribution_plot'] = generate_quality_distribution_plot(metrics)
    files['dimension_plot'] = generate_dimension_comparison_plot(metrics)
    files['flags_plot'] = generate_flags_plot(metrics)
    files['year_plot'] = generate_quality_by_year_plot(metrics)

    # Reportes
    files['markdown_report'] = generate_markdown_report(articles, metrics, threshold)
    files['json_stats'] = generate_json_stats(articles, metrics, threshold)

    # Filtrar None
    files = {k: str(v) for k, v in files.items() if v is not None}

    logger.info(f"  Generated {len(files)} files in {QUALITY_DIR}")

    # Quality by item type
    try:
        by_type_files = run_quality_by_type_report(articles, threshold)
        files.update(by_type_files)
    except Exception as e:
        logger.warning(f"Quality by type report failed: {e}")

    return files


def run_quality_report(articles: list[dict], threshold: float = 50.0) -> dict:
    """
    Entry point para integracion con pipeline.

    Args:
        articles: Lista de articulos
        threshold: Umbral de calidad

    Returns:
        dict con rutas de archivos generados
    """
    return generate_quality_report(articles, threshold)


# ---------------------------------------------------------------------------
# Calidad estratificada por item_type
# ---------------------------------------------------------------------------

def extract_quality_metrics_by_type(articles: list[dict]) -> dict[str, dict]:
    """
    Agrupar articulos por item_type y extraer metricas de calidad por grupo.

    Returns:
        dict mapping item_type -> metrics (same structure as extract_quality_metrics)
    """
    groups: dict[str, list[dict]] = {}
    for article in articles:
        itype = article.get('item_type', 'unknown')
        groups.setdefault(itype, []).append(article)

    metrics_by_type = {}
    for itype, group in sorted(groups.items()):
        m = extract_quality_metrics(group)
        if m['evaluated'] > 0:
            metrics_by_type[itype] = m

    return metrics_by_type


def generate_quality_dimensions_by_type_plot(
    metrics_by_type: dict[str, dict],
    output_path: Path = None,
) -> Optional[Path]:
    """Barras agrupadas: dimensiones de calidad por tipo de documento (con error bars)."""
    if output_path is None:
        output_path = QUALITY_DIR / "quality_dimensions_by_type.png"

    types = list(metrics_by_type.keys())
    if len(types) < 2:
        logger.info("  Skipping dimensions-by-type plot (< 2 types with data)")
        return None

    dimensions = ['completeness', 'structure', 'informativeness', 'readability']
    dim_labels = [d.capitalize() for d in dimensions]

    x = np.arange(len(dimensions))
    n_types = len(types)
    width = 0.8 / n_types
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, itype in enumerate(types):
        m = metrics_by_type[itype]
        means = [np.mean(m['scores'].get(d, [0])) for d in dimensions]
        stds = [np.std(m['scores'].get(d, [0])) for d in dimensions]
        offset = (i - n_types / 2 + 0.5) * width
        bars = ax.bar(x + offset, means, width, yerr=stds, capsize=3,
                      label=f"{itype} (n={m['evaluated']})",
                      color=colors[i % len(colors)], edgecolor='black', alpha=0.8)

    ax.set_ylabel('Score (0-100)', fontsize=11)
    ax.set_xlabel('Quality Dimension', fontsize=11)
    ax.set_title('Abstract Quality Dimensions by Document Type', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(dim_labels, fontsize=10)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"  Generated: {output_path}")
    return output_path


def generate_quality_levels_by_type_plot(
    metrics_by_type: dict[str, dict],
    output_path: Path = None,
) -> Optional[Path]:
    """Barras apiladas: proporcion high/medium/low por tipo de documento."""
    if output_path is None:
        output_path = QUALITY_DIR / "quality_levels_by_type.png"

    types = list(metrics_by_type.keys())
    if len(types) < 2:
        logger.info("  Skipping levels-by-type plot (< 2 types with data)")
        return None

    high_pcts, med_pcts, low_pcts = [], [], []
    labels = []
    for itype in types:
        m = metrics_by_type[itype]
        total = m['evaluated']
        if total == 0:
            continue
        labels.append(f"{itype}\n(n={total})")
        high_pcts.append(m['quality_levels'].get('high', 0) / total * 100)
        med_pcts.append(m['quality_levels'].get('medium', 0) / total * 100)
        low_pcts.append(m['quality_levels'].get('low', 0) / total * 100)

    if not labels:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels))

    ax.bar(x, high_pcts, color='#2ecc71', edgecolor='black', label='High')
    ax.bar(x, med_pcts, bottom=high_pcts, color='#f39c12', edgecolor='black', label='Medium')
    bottoms = [h + m for h, m in zip(high_pcts, med_pcts)]
    ax.bar(x, low_pcts, bottom=bottoms, color='#e74c3c', edgecolor='black', label='Low')

    ax.set_ylabel('Percentage (%)', fontsize=11)
    ax.set_xlabel('Document Type', fontsize=11)
    ax.set_title('Quality Level Distribution by Document Type', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"  Generated: {output_path}")
    return output_path


def generate_quality_flags_heatmap(
    metrics_by_type: dict[str, dict],
    output_path: Path = None,
    top_n: int = 10,
) -> Optional[Path]:
    """Heatmap: flags (filas) x tipos (columnas), celdas = % articulos con ese flag."""
    if output_path is None:
        output_path = QUALITY_DIR / "quality_flags_by_type.png"

    types = list(metrics_by_type.keys())
    if len(types) < 2:
        logger.info("  Skipping flags heatmap (< 2 types with data)")
        return None

    # Collect all flags and pick top N globally
    global_flags: Counter = Counter()
    for m in metrics_by_type.values():
        global_flags.update(m['flags'])

    top_flags = [f for f, _ in global_flags.most_common(top_n)]
    if not top_flags:
        return None

    # Build matrix: rows=flags, cols=types
    matrix = np.zeros((len(top_flags), len(types)))
    for j, itype in enumerate(types):
        m = metrics_by_type[itype]
        total = max(m['evaluated'], 1)
        for i, flag in enumerate(top_flags):
            matrix[i, j] = m['flags'].get(flag, 0) / total * 100

    fig, ax = plt.subplots(figsize=(max(8, len(types) * 2.5), max(6, len(top_flags) * 0.6)))

    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(np.arange(len(types)))
    ax.set_xticklabels([f"{t}\n(n={metrics_by_type[t]['evaluated']})" for t in types], fontsize=10)
    ax.set_yticks(np.arange(len(top_flags)))
    ax.set_yticklabels([f.replace('_', ' ').title() for f in top_flags], fontsize=10)

    # Annotate cells
    for i in range(len(top_flags)):
        for j in range(len(types)):
            val = matrix[i, j]
            color = 'white' if val > 50 else 'black'
            ax.text(j, i, f'{val:.0f}%', ha='center', va='center', fontsize=9, color=color)

    ax.set_title('Quality Flags by Document Type (%)', fontsize=12, fontweight='bold')
    fig.colorbar(im, ax=ax, label='% of articles', shrink=0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"  Generated: {output_path}")
    return output_path


def generate_quality_boxplot_by_type(
    metrics_by_type: dict[str, dict],
    output_path: Path = None,
) -> Optional[Path]:
    """Box plot: distribucion de score overall por tipo de documento."""
    if output_path is None:
        output_path = QUALITY_DIR / "quality_scores_boxplot_by_type.png"

    types = list(metrics_by_type.keys())
    if len(types) < 2:
        logger.info("  Skipping boxplot-by-type (< 2 types with data)")
        return None

    data = []
    labels = []
    for itype in types:
        scores = metrics_by_type[itype]['scores']['overall']
        if scores:
            data.append(scores)
            labels.append(f"{itype}\n(n={len(scores)})")

    if len(data) < 2:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    try:
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, notch=True)
    except TypeError:
        bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=True)
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']
    for patch, color in zip(bp['boxes'], colors[:len(data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('Overall Quality Score', fontsize=11)
    ax.set_xlabel('Document Type', fontsize=11)
    ax.set_title('Quality Score Distribution by Document Type', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 100)

    # Add mean markers
    for i, scores in enumerate(data):
        ax.plot(i + 1, np.mean(scores), 'D', color='black', markersize=6, label='Mean' if i == 0 else '')

    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"  Generated: {output_path}")
    return output_path


def generate_quality_by_type_markdown(
    metrics_by_type: dict[str, dict],
    threshold: float = 50.0,
    output_path: Path = None,
) -> Optional[Path]:
    """Reporte MD con tabla comparativa y detalle por tipo."""
    if output_path is None:
        output_path = QUALITY_DIR / "quality_by_item_type.md"

    types = list(metrics_by_type.keys())
    if not types:
        return None

    lines = [
        "# Abstract Quality by Document Type",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Threshold:** {threshold}",
        "",
        "---",
        "",
        "## Comparative Summary",
        "",
        "| Document Type | N Evaluated | Mean Score | Median | Std | High % | Medium % | Low % |",
        "|---------------|-------------|------------|--------|-----|--------|----------|-------|",
    ]

    for itype in types:
        m = metrics_by_type[itype]
        stats = compute_statistics(m['scores']['overall'])
        total = max(m['evaluated'], 1)
        high_pct = m['quality_levels'].get('high', 0) / total * 100
        med_pct = m['quality_levels'].get('medium', 0) / total * 100
        low_pct = m['quality_levels'].get('low', 0) / total * 100
        lines.append(
            f"| {itype} | {m['evaluated']} | {stats['mean']} | {stats['median']} "
            f"| {stats['std']} | {high_pct:.1f}% | {med_pct:.1f}% | {low_pct:.1f}% |"
        )

    lines.extend(["", "---", ""])

    # Detail per type
    for itype in types:
        m = metrics_by_type[itype]
        lines.append(f"## {itype} (n={m['evaluated']})")
        lines.append("")

        # Dimensions table
        lines.append("| Dimension | Mean | Median | Std | Min | Max |")
        lines.append("|-----------|------|--------|-----|-----|-----|")
        for dim in ['overall', 'completeness', 'structure', 'informativeness', 'readability']:
            s = compute_statistics(m['scores'].get(dim, []))
            label = f"**{dim.capitalize()}**" if dim == 'overall' else dim.capitalize()
            lines.append(f"| {label} | {s['mean']} | {s['median']} | {s['std']} | {s['min']} | {s['max']} |")
        lines.append("")

        # Above threshold
        above = sum(1 for s in m['scores']['overall'] if s >= threshold)
        lines.append(f"- **Above threshold ({threshold}):** {above}/{m['evaluated']} ({above/max(m['evaluated'],1)*100:.1f}%)")
        lines.append("")

        # Top flags
        if m['flags']:
            lines.append("**Top quality issues:**")
            lines.append("")
            for flag, count in m['flags'].most_common(5):
                pct = count / max(m['evaluated'], 1) * 100
                lines.append(f"- {flag.replace('_', ' ').title()}: {count} ({pct:.1f}%)")
            lines.append("")

        lines.append("---")
        lines.append("")

    # Visualizations reference
    lines.extend([
        "## Visualizations",
        "",
        "- `quality_dimensions_by_type.png` - Dimension comparison by type",
        "- `quality_levels_by_type.png` - Quality level proportions by type",
        "- `quality_flags_by_type.png` - Flags heatmap across types",
        "- `quality_scores_boxplot_by_type.png` - Score distribution boxplots",
    ])

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    logger.info(f"  Generated: {output_path}")
    return output_path


def generate_quality_by_type_json(
    metrics_by_type: dict[str, dict],
    threshold: float = 50.0,
    output_path: Path = None,
) -> Optional[Path]:
    """Stats JSON estratificadas por tipo de documento."""
    if output_path is None:
        output_path = QUALITY_DIR / "quality_by_item_type_stats.json"

    if not metrics_by_type:
        return None

    result = {
        'generated_at': datetime.now().isoformat(),
        'threshold': threshold,
        'by_type': {},
    }

    for itype, m in metrics_by_type.items():
        above = sum(1 for s in m['scores']['overall'] if s >= threshold)
        result['by_type'][itype] = {
            'evaluated': m['evaluated'],
            'total_in_group': m['total'],
            'above_threshold': above,
            'pct_above_threshold': round(above / max(m['evaluated'], 1) * 100, 2),
            'quality_levels': dict(m['quality_levels']),
            'statistics': {
                dim: compute_statistics(m['scores'].get(dim, []))
                for dim in ['overall', 'completeness', 'structure', 'informativeness', 'readability']
            },
            'top_flags': dict(m['flags'].most_common(10)),
        }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info(f"  Generated: {output_path}")
    return output_path


def run_quality_by_type_report(
    articles: list[dict],
    threshold: float = 50.0,
) -> dict:
    """
    Orquestador: genera reportes de calidad estratificados por item_type.

    Returns:
        dict con rutas de archivos generados
    """
    logger.info("-" * 40)
    logger.info("Generating Quality by Item Type Report")
    logger.info("-" * 40)

    ensure_quality_dir()

    metrics_by_type = extract_quality_metrics_by_type(articles)

    if len(metrics_by_type) < 2:
        logger.info("  Less than 2 document types with quality data; skipping by-type report")
        return {}

    files = {}

    files['dimensions_by_type'] = generate_quality_dimensions_by_type_plot(metrics_by_type)
    files['levels_by_type'] = generate_quality_levels_by_type_plot(metrics_by_type)
    files['flags_heatmap'] = generate_quality_flags_heatmap(metrics_by_type)
    files['boxplot_by_type'] = generate_quality_boxplot_by_type(metrics_by_type)
    files['markdown_by_type'] = generate_quality_by_type_markdown(metrics_by_type, threshold)
    files['json_by_type'] = generate_quality_by_type_json(metrics_by_type, threshold)

    files = {k: str(v) for k, v in files.items() if v is not None}

    logger.info(f"  Generated {len(files)} quality-by-type files")
    return files
