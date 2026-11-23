"""Fase 2f: Estado del Arte + Descubrimiento de Vacios de Conocimiento.

Layer A (Algoritmico — sin LLM):
  - Descubrimiento de dimensiones del corpus
  - Matrices de cobertura cruzada
  - Deteccion de vacios estructurales, declarados y emergentes
  - Evolucion temporal de gaps
  - Fronteras de investigacion
  - Paisaje de rendimiento
  - Taxonomia unificada de gaps

Layer B (LLM-enhanced — opcional):
  - Narrativa SOTA
  - Interpretacion contextual de gaps
  - Agenda de investigacion priorizada
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import SOTA_DIR, FIGURES_DIR, REPORTS_DIR, PLOTLY_DIR
from src.utils import save_json, is_noisy_legacy_concept, canonicalize_keyword, is_keyword_stopword, singularize_term_agnostic

logger = logging.getLogger("bibliometrico.sota_gap")

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight"})


# ===========================================================================
# Layer A: Algorithmic analysis
# ===========================================================================

# ---------------------------------------------------------------------------
# A1: Extract Research Dimensions
# ---------------------------------------------------------------------------

# Metodos: sin lista de metodos ni de tecnicas; ver src/metodos_declarados.py (comun con la Fase 2b).
from src.metodos_declarados import article_methods, build_method_vocabulary, declared_methods


def _canonicalize_sota_term(term: str) -> str:
    """Normaliza con el vocabulario comun de palabras clave del pipeline (reglas aprendidas del corpus)."""
    from src.utils import _VOCABULARIO
    return canonicalize_keyword(term, corpus_map=_VOCABULARIO.get("mapa"))

# Captura GENERICA y agnostica de datasets nombrados ("dataset/benchmark/
# corpus/database of X"). Se eliminaron los nombres especificos foraneos
# (ImageNet/MNIST/Kaggle/WHO...) por no ser agnosticos del dominio.
_DATASET_PATTERNS = [
    r"(?i)\b(dataset|benchmark|corpus|database)\s+(?:of\s+)?([A-Z][\w-]+)",
]

# El eje "applications" (areas/dominios de aplicacion) NO se hardcodea: antes
# habia una taxonomia multi-dominio foranea (healthcare/finance/education/...)
# que forzaba salida fuera de tema. Ahora se deriva del corpus (conceptos
# OpenAlex) en extract_research_dimensions -> agnostico entre revisiones.
# Los metodos tampoco usan lista: se leen de lo que el corpus declara (ver src/metodos_declarados.py).


def _compile_patterns(patterns: list[str]) -> list[re.Pattern]:
    return [re.compile(p) for p in patterns]


_COMPILED_DATASETS = _compile_patterns(_DATASET_PATTERNS)


# Vocabulario de metodos del corpus (lo llena extract_research_dimensions en su primera pasada)
_CORPUS_METHODS: set[str] = set()


def _is_method_term(term: str) -> bool:
    """True si el termino (ya canonicalizado) aparece como metodo declarado en los textos del corpus."""
    t = (term or "").strip()
    return bool(t) and t in _CORPUS_METHODS


def _extract_methods(text: str) -> list[str]:
    """Metodos declarados en el texto que pertenecen al vocabulario de metodos del corpus."""
    found = declared_methods(text)
    if _CORPUS_METHODS:
        found = [m for m in found if m in _CORPUS_METHODS]
    return found


def _extract_from_text(text: str, compiled_patterns: list[re.Pattern]) -> list[str]:
    """Extract all matches from text using compiled patterns."""
    found = []
    for pat in compiled_patterns:
        for match in pat.finditer(text):
            term = match.group(1) if match.lastindex else match.group(0)
            canon = _canonicalize_sota_term(term)
            if canon:
                found.append(canon)
    return found


def _extract_article_dimension_sets(art: dict) -> dict[str, set[str]]:
    """Extrae conjuntos canonicalizados por dimension para un unico articulo (1 conteo maximo por articulo)."""
    text = f"{art.get('title', '')} {art.get('abstract', '')}"
    m_set = set(article_methods(art, _CORPUS_METHODS))   # misma definicion que la 2b
    d_set = set(_extract_from_text(text, _COMPILED_DATASETS))
    a_set: set[str] = set()
    c_set: set[str] = set()

    topics = art.get("topics") or art.get("openalex", {}).get("topics", [])
    if topics:
        for t in topics:
            raw_name = (t.get("display_name") or t.get("name") or "").lower().strip()
            name = _canonicalize_sota_term(raw_name)
            if name and not is_noisy_legacy_concept(name):
                c_set.add(name)
                a_set.add(name)
    else:
        for c in art.get("concepts", []):
            raw_name = c.get("name", "").lower().strip()
            name = _canonicalize_sota_term(raw_name)
            if name and not is_noisy_legacy_concept(name):
                c_set.add(name)
                a_set.add(name)

    for tag in art.get("all_tags", []):
        tag_l = _canonicalize_sota_term(str(tag))
        if tag_l and not is_noisy_legacy_concept(tag_l):
            c_set.add(tag_l)
            # Los metodos NO se toman de all_tags (incluye palabras automaticas): la definicion unica es
            # article_methods (titulo, resumen y palabras clave de autor), igual que en la Fase 2b.

    bert_kws = art.get("bert_keywords", [])
    if isinstance(bert_kws, list):
        for kw in bert_kws:
            if isinstance(kw, str):
                kw_l = _canonicalize_sota_term(kw)
                if kw_l and not is_noisy_legacy_concept(kw_l):
                    c_set.add(kw_l)

    return {
        "methods": m_set,
        "datasets": d_set,
        "applications": a_set,
        "concepts": c_set,
        "all": m_set | d_set | a_set | c_set,
    }


def extract_research_dimensions(articles: list[dict]) -> dict:
    """Discover research dimensions from the corpus.

    Sources: abstracts, titles, manual/auto tags, OpenAlex topics/concepts,
    and BERT keywords.

    Returns dict with keys: methods, datasets, applications, concepts.
    """
    methods = Counter()
    datasets = Counter()
    applications = Counter()
    concepts = Counter()
    min_freq = max(2, len(articles) // 50)

    # Primera pasada: vocabulario de metodos DECLARADOS en los textos del corpus (comun con la 2b)
    from src.modelo_corpus import get_model
    _CORPUS_METHODS.clear()
    _CORPUS_METHODS.update(get_model(articles)["metodos"]["vocabulario"])   # vocabulario del modelo del corpus

    for art in articles:
        dim_sets = _extract_article_dimension_sets(art)
        for m in dim_sets["methods"]:
            methods[m] += 1
        for d in dim_sets["datasets"]:
            datasets[d] += 1
        for a in dim_sets["applications"]:
            applications[a] += 1
        for c in dim_sets["concepts"]:
            concepts[c] += 1

    # Filter by minimum frequency
    methods = Counter({k: v for k, v in methods.items() if v >= min_freq})
    datasets = Counter({k: v for k, v in datasets.items() if v >= min_freq})
    applications = Counter({k: v for k, v in applications.items() if v >= min_freq})
    concepts = Counter({k: v for k, v in concepts.items() if v >= min_freq})

    logger.info(
        f"Dimensions found: {len(methods)} methods, {len(datasets)} datasets, "
        f"{len(applications)} applications, {len(concepts)} concepts"
    )

    return {
        "methods": methods,
        "datasets": datasets,
        "applications": applications,
        "concepts": concepts,
    }


# ---------------------------------------------------------------------------
# A2: Build Coverage Matrix
# ---------------------------------------------------------------------------

def _article_mentions_item(art: dict, item: str, art_terms: Optional[set[str]] = None) -> bool:
    """Verifica si un articulo menciona un termino canonico (en sus conjuntos extraidos o en texto libre)."""
    canon = _canonicalize_sota_term(item)
    if art_terms is None:
        art_terms = _extract_article_dimension_sets(art)["all"]
    if canon in art_terms or item in art_terms:
        return True
    text = f"{art.get('title', '')} {art.get('abstract', '')} {' '.join(art.get('all_tags', []))}".lower()
    return canon in text or item in text


def build_coverage_matrix(
    articles: list[dict],
    dim_a: Counter,
    dim_b: Counter,
    dim_a_name: str = "dim_a",
    dim_b_name: str = "dim_b",
    top_n: int = 20,
) -> pd.DataFrame:
    """Build cross-tabulation matrix between two dimensions.

    Returns a DataFrame where cell (i,j) = count of articles mentioning both.
    """
    top_a = [k for k, _ in dim_a.most_common(top_n)]
    top_b = [k for k, _ in dim_b.most_common(top_n)]

    matrix = pd.DataFrame(0, index=top_a, columns=top_b)

    for art in articles:
        art_terms = _extract_article_dimension_sets(art)["all"]
        art_a = [t for t in top_a if _article_mentions_item(art, t, art_terms)]
        art_b = [t for t in top_b if _article_mentions_item(art, t, art_terms)]

        for a in art_a:
            for b in art_b:
                if a != b:
                    matrix.loc[a, b] += 1

    matrix.index.name = dim_a_name
    matrix.columns.name = dim_b_name
    return matrix


# ---------------------------------------------------------------------------
# A3: Detect Structural Gaps
# ---------------------------------------------------------------------------

def detect_structural_gaps(
    matrix: pd.DataFrame,
    min_individual_freq: int = 3,
    dim_a_counts: Optional[Counter] = None,
    dim_b_counts: Optional[Counter] = None,
) -> list[dict]:
    """Find empty cells in coverage matrix where both items are individually frequent.

    A structural gap = combination (A, B) where:
      - A appears in >= min_individual_freq articles
      - B appears in >= min_individual_freq articles
      - A+B co-occurrence == 0 (and A != B)
    """
    row_sums = matrix.sum(axis=1)
    col_sums = matrix.sum(axis=0)

    gaps = []
    for row_item in matrix.index:
        freq_a = int(dim_a_counts[row_item]) if (dim_a_counts and row_item in dim_a_counts) else int(row_sums[row_item])
        if freq_a < min_individual_freq or row_sums[row_item] == 0:
            continue
        for col_item in matrix.columns:
            if _canonicalize_sota_term(row_item) == _canonicalize_sota_term(col_item):
                continue
            freq_b = int(dim_b_counts[col_item]) if (dim_b_counts and col_item in dim_b_counts) else int(col_sums[col_item])
            if freq_b < min_individual_freq or col_sums[col_item] == 0:
                continue
            if matrix.loc[row_item, col_item] == 0:
                gaps.append({
                    "type": "structural",
                    "dimension_a": matrix.index.name or "dim_a",
                    "item_a": row_item,
                    "item_a_freq": freq_a,
                    "dimension_b": matrix.columns.name or "dim_b",
                    "item_b": col_item,
                    "item_b_freq": freq_b,
                    "priority": float(freq_a * freq_b),
                })

    gaps.sort(key=lambda g: g["priority"], reverse=True)
    logger.info(f"Structural gaps found: {len(gaps)}")
    return gaps


# ---------------------------------------------------------------------------
# A4: Analyze Declared Gaps
# ---------------------------------------------------------------------------

# Patrones de vacios declarados: ver src/vacios_declarados.py (comun con la Fase 2b).


def analyze_declared_gaps(articles: list[dict]) -> list[dict]:
    """Vacios declarados: detector unico compartido con la Fase 2b (src/vacios_declarados.py)."""
    from src.vacios_declarados import extract_declared_gaps
    declared_gaps = [
        {"type": "declared", "source": "abstract", "article_key": g["key"], "year": g["year"],
         "sentence": g["sentence"], "journal": g["journal"], "cue": g["cue"]}
        for g in extract_declared_gaps(articles)
    ]
    logger.info(f"Declared gaps extracted: {len(declared_gaps)}")
    return declared_gaps


# ---------------------------------------------------------------------------
# A5: Temporal Gap Evolution
# ---------------------------------------------------------------------------

def analyze_temporal_gap_evolution(
    articles: list[dict],
    dimensions: dict,
    n_periods: int = 3,
) -> dict:
    """Divide corpus into temporal periods and track dimension coverage shifts."""
    years = [a["publication_year"] for a in articles if a.get("publication_year")]
    if not years:
        return {}

    min_year, max_year = min(years), max(years)
    span = max_year - min_year + 1
    period_size = max(1, span // n_periods)

    periods = []
    for i in range(n_periods):
        start = min_year + i * period_size
        end = start + period_size - 1 if i < n_periods - 1 else max_year
        period_articles = [
            a for a in articles
            if a.get("publication_year") and start <= a["publication_year"] <= end
        ]
        periods.append({
            "start": start,
            "end": end,
            "n_articles": len(period_articles),
            "articles": period_articles,
        })

    # Track method and application coverage per period
    evolution = {}
    for dim_name in ["methods", "applications"]:
        dim_counter = dimensions.get(dim_name, Counter())
        top_items = [k for k, _ in dim_counter.most_common(15)]
        if not top_items:
            continue

        period_data = {}
        for period in periods:
            label = f"{period['start']}-{period['end']}"
            counts = Counter()
            for art in period["articles"]:
                art_terms = _extract_article_dimension_sets(art)["all"]
                for item in top_items:
                    if _article_mentions_item(art, item, art_terms):
                        counts[item] += 1
            # Normalize by period size
            n = max(1, period["n_articles"])
            period_data[label] = {item: counts.get(item, 0) / n for item in top_items}

        evolution[dim_name] = period_data

    logger.info(f"Temporal evolution tracked across {n_periods} periods")
    return evolution


# ---------------------------------------------------------------------------
# A6: Identify Research Frontiers
# ---------------------------------------------------------------------------

def identify_research_frontiers(
    articles: list[dict],
    dimensions: dict,
    recent_years: int = 3,
    growth_threshold: float = 1.5,
) -> list[dict]:
    """Identify items with accelerating growth in recent period (deduplicated across dimensions)."""
    years = [a["publication_year"] for a in articles if a.get("publication_year")]
    if not years:
        return []

    max_year = max(years)
    cutoff = max_year - recent_years

    early = [a for a in articles if a.get("publication_year") and a["publication_year"] <= cutoff]
    recent = [a for a in articles if a.get("publication_year") and a["publication_year"] > cutoff]

    if not early or not recent:
        return []

    early_terms = [(a, _extract_article_dimension_sets(a)["all"]) for a in early]
    recent_terms = [(a, _extract_article_dimension_sets(a)["all"]) for a in recent]

    frontiers = []
    seen_canonical: set[str] = set()

    for dim_name in ["methods", "applications", "concepts"]:
        dim_counter = dimensions.get(dim_name, Counter())
        top_items = [k for k, _ in dim_counter.most_common(30)]

        for item in top_items:
            canon_item = _canonicalize_sota_term(item)
            if not canon_item or canon_item in seen_canonical:
                continue

            early_count = sum(
                1 for a, terms in early_terms
                if _article_mentions_item(a, canon_item, terms)
            )
            recent_count = sum(
                1 for a, terms in recent_terms
                if _article_mentions_item(a, canon_item, terms)
            )

            # Normalize by period size
            early_rate = early_count / max(1, len(early))
            recent_rate = recent_count / max(1, len(recent))

            if early_rate > 0:
                growth = recent_rate / early_rate
            elif recent_count > 0:
                growth = float("inf")
            else:
                growth = 0

            if growth >= growth_threshold and recent_count >= 3:
                seen_canonical.add(canon_item)
                # Para ranking determinista cuando early_count == 0, usar pseudoconteo de Laplace (0.5)
                smooth_early_rate = max(early_count, 0.5) / max(1, len(early))
                smooth_growth = recent_rate / smooth_early_rate
                frontiers.append({
                    "dimension": dim_name,
                    "item": canon_item,
                    "early_count": early_count,
                    "recent_count": recent_count,
                    "early_rate": round(early_rate, 4),
                    "recent_rate": round(recent_rate, 4),
                    "growth_ratio": round(growth, 2) if growth != float("inf") else "new",
                    "_rank_score": recent_count * math.log1p(smooth_growth),
                })

    frontiers.sort(
        key=lambda f: (f.get("_rank_score", 0), f["recent_count"]),
        reverse=True,
    )
    for f in frontiers:
        f.pop("_rank_score", None)
    logger.info(f"Research frontiers identified: {len(frontiers)}")
    return frontiers


# ---------------------------------------------------------------------------
# A7: Detect Emergent Gaps
# ---------------------------------------------------------------------------

def detect_emergent_gaps(
    frontiers: list[dict],
    structural_gaps: list[dict],
) -> list[dict]:
    """Find frontier topics that lack coverage in existing methods/data."""
    frontier_items = {f["item"] for f in frontiers}
    gap_items = set()
    for g in structural_gaps:
        gap_items.add(g["item_a"])
        gap_items.add(g["item_b"])

    emergent = []
    for f in frontiers:
        item = f["item"]
        # Check if this frontier appears in structural gaps
        related_gaps = [
            g for g in structural_gaps
            if g["item_a"] == item or g["item_b"] == item
        ]
        if related_gaps:
            emergent.append({
                "type": "emergent",
                "frontier_item": item,
                "frontier_dimension": f["dimension"],
                "growth_ratio": f["growth_ratio"],
                "recent_count": f["recent_count"],
                "uncovered_combinations": len(related_gaps),
                "example_gaps": [
                    f"{g['item_a']} x {g['item_b']}"
                    for g in related_gaps[:5]
                ],
            })

    emergent.sort(key=lambda e: e["uncovered_combinations"], reverse=True)
    logger.info(f"Emergent gaps detected: {len(emergent)}")
    return emergent


# ---------------------------------------------------------------------------
# A8: Performance Landscape
# ---------------------------------------------------------------------------

_PERF_PATTERNS = [
    (re.compile(r"(?i)(accuracy|acc)\s*(?:of\s*)?(\d+\.?\d*)\s*%"), "accuracy"),
    (re.compile(r"(?i)(f1[- ]?score?|f1)\s*(?:of\s*)?(\d+\.?\d*)"), "f1_score"),
    (re.compile(r"(?i)(auc|auroc)\s*(?:of\s*)?(\d+\.?\d*)"), "auc"),
    (re.compile(r"(?i)(precision)\s*(?:of\s*)?(\d+\.?\d*)"), "precision"),
    (re.compile(r"(?i)(recall|sensitivity)\s*(?:of\s*)?(\d+\.?\d*)"), "recall"),
]


def build_performance_landscape(articles: list[dict]) -> list[dict]:
    """Extract performance metrics mapped to methods and years."""
    results = []

    for art in articles:
        abstract = art.get("abstract", "")
        if not abstract or abstract == "nan":
            continue

        for pat, metric_name in _PERF_PATTERNS:
            for match in pat.finditer(abstract):
                try:
                    value = float(match.group(2))
                    # Sanity check
                    if 0 <= value <= 100:
                        results.append({
                            "article_key": art.get("key", ""),
                            "year": art.get("publication_year"),
                            "metric": metric_name,
                            "value": value,
                            "title": art.get("title", "")[:80],
                        })
                except (ValueError, IndexError):
                    pass

    logger.info(f"Performance metrics extracted: {len(results)}")
    return results


# ---------------------------------------------------------------------------
# A9: Build Gap Taxonomy
# ---------------------------------------------------------------------------

def build_gap_taxonomy(
    structural_gaps: list[dict],
    declared_gaps: list[dict],
    emergent_gaps: list[dict],
) -> dict:
    """Build a unified taxonomy: structural / declared / emergent."""
    return {
        "structural": {
            "count": len(structural_gaps),
            "description": "Combinations of frequent items that never co-occur",
            "top_gaps": structural_gaps[:20],
        },
        "declared": {
            "count": len(declared_gaps),
            "description": "Gaps explicitly stated by authors in abstracts",
            "top_gaps": declared_gaps[:20],
        },
        "emergent": {
            "count": len(emergent_gaps),
            "description": "Growing topics lacking coverage in existing methods/data",
            "top_gaps": emergent_gaps[:20],
        },
        "total_gaps": len(structural_gaps) + len(declared_gaps) + len(emergent_gaps),
    }


# ===========================================================================
# Visualizations
# ===========================================================================

def _plot_coverage_heatmap(
    matrix: pd.DataFrame,
    title: str,
    output_path: Path,
) -> None:
    """Plot a coverage heatmap."""
    if matrix.empty:
        return

    fig_height = max(6, len(matrix.index) * 0.4)
    fig_width = max(8, len(matrix.columns) * 0.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    sns.heatmap(
        matrix, annot=True, fmt="d", cmap="YlOrRd",
        linewidths=0.5, ax=ax, cbar_kws={"label": "Co-occurrence count"},
    )
    ax.set_title(title, fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info(f"Saved: {output_path.name}")


def _wrap_label(text: str, width: int = 20, max_lines: int = 3) -> str:
    """Envuelve etiquetas largas en hasta max_lines lineas legibles sin truncar abruptamente."""
    import textwrap
    s = str(text).strip()
    if len(s) <= width:
        return s
    lines = textwrap.wrap(s, width=width)
    if len(lines) > max_lines:
        return "\n".join(lines[: max_lines - 1]) + "\n" + lines[max_lines - 1][: max(3, width - 3)] + "..."
    return "\n".join(lines)


def _plot_gap_priority_scatter(
    structural_gaps: list[dict],
    output_path: Path,
) -> None:
    """Scatter plot de prioridad de vacios estructurales en escala log-log con curvas de isoprioridad."""
    if not structural_gaps:
        return

    from matplotlib.ticker import ScalarFormatter, NullFormatter
    import textwrap

    # Deduplicar pares (item_a, item_b) preservando el de mayor prioridad
    deduped = []
    seen_pairs: set[tuple[str, str]] = set()
    for g in structural_gaps:
        pair = (_canonicalize_sota_term(g["item_a"]), _canonicalize_sota_term(g["item_b"]))
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            deduped.append(g)

    top = deduped[:50]
    if not top:
        return

    rng = np.random.RandomState(42)
    x_raw = np.array([max(1.0, float(g["item_a_freq"])) for g in top])
    y_raw = np.array([max(1.0, float(g["item_b_freq"])) for g in top])
    # Leve jitter multiplicativo determinista (2.5%) para separar puntos con coordenadas identicas en log-log
    x = x_raw * rng.uniform(0.985, 1.015, size=len(top))
    y = y_raw * rng.uniform(0.985, 1.015, size=len(top))
    priorities = x_raw * y_raw
    log_prio = np.log10(np.maximum(priorities, 1.0))

    sizes = 60 + 180 * (log_prio - log_prio.min()) / max(0.01, log_prio.max() - log_prio.min())

    fig, ax = plt.subplots(figsize=(11.5, 8.5))
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Limites holgados en espacio logaritmico
    x_min, x_max = max(1.5, x_raw.min() * 0.75), x_raw.max() * 1.45
    y_min, y_max = max(1.5, y_raw.min() * 0.75), y_raw.max() * 1.45
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Curvas de isoprioridad (P = f_A * f_B => rectas diagonales en escala log-log)
    xg = np.logspace(np.log10(x_min), np.log10(x_max), 200)
    p_min, p_max = priorities.min(), priorities.max()
    iso_levels = [lvl for lvl in (50, 100, 250, 500, 1000, 2500, 5000, 10000) if p_min * 0.6 <= lvl <= p_max * 1.4]
    for lvl in iso_levels:
        yg = lvl / xg
        mask = (yg >= y_min) & (yg <= y_max)
        if np.any(mask):
            ax.plot(xg[mask], yg[mask], "--", color="#9aa0a6", linewidth=0.9, alpha=0.55, zorder=1)
            idx_lbl = np.where(mask)[0][len(np.where(mask)[0]) // 2]
            ax.text(
                xg[idx_lbl], yg[idx_lbl], f"P={lvl:,}",
                fontsize=7.5, color="#5f6368", alpha=0.85,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.75),
                zorder=2,
            )

    scatter = ax.scatter(
        x, y, s=sizes, c=log_prio, cmap="viridis",
        alpha=0.85, edgecolors="white", linewidth=0.9, zorder=3,
    )
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label(r"Gap Priority $\log_{10}(f_A \times f_B)$", fontsize=10)

    # Ticks enteros legibles en ambos ejes logaritmicos
    tick_candidates = [2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500]
    ax.set_xticks([t for t in tick_candidates if x_min <= t <= x_max])
    ax.set_yticks([t for t in tick_candidates if y_min <= t <= y_max])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", linestyle="-", alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", alpha=0.15)

    dim_b_types = sorted({g.get("dimension_b", "Concepts/Applications") for g in top})
    dim_b_label = " / ".join(dim_b_types)
    ax.set_xlabel("Method Frequency (articles, log scale)", fontsize=11, fontweight="bold")
    ax.set_ylabel(f"{dim_b_label} Frequency (articles, log scale)", fontsize=11, fontweight="bold")
    ax.set_title(
        "Structural Gaps: Priority by Component Frequency (Log-Log Scale)",
        fontsize=13, fontweight="bold", pad=12,
    )

    # Etiquetar los top 12 vacios con adjustText para evitar solapamientos
    texts = []
    n_labels = min(12, len(top))
    for i in range(n_labels):
        g = top[i]
        a_short = textwrap.shorten(g["item_a"], width=26, placeholder="...")
        b_short = textwrap.shorten(g["item_b"], width=28, placeholder="...")
        lbl = f"{a_short}\n× {b_short}"
        txt = ax.text(
            x[i], y[i], lbl,
            fontsize=7.5, color="#1f2937",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#cbd5e1", alpha=0.88, lw=0.6),
            zorder=4,
        )
        texts.append(txt)

    try:
        from adjustText import adjust_text
        adjust_text(
            texts, x=x, y=y, ax=ax,
            arrowprops=dict(arrowstyle="-", color="#64748b", lw=0.7, alpha=0.75),
            expand_points=(1.35, 1.45),
            expand_text=(1.15, 1.25),
        )
    except Exception:
        pass

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    logger.info(f"Saved: {output_path.name}")


def _plot_frontier_radar(
    frontiers: list[dict],
    output_path: Path,
    top_n: int = 10,
) -> None:
    """Radar chart de fronteras de investigacion deduplicadas por etiqueta canonica."""
    if not frontiers:
        return

    # Garantizar unicidad absoluta de etiquetas canonicas en el radar
    unique_top = []
    seen_labels: set[str] = set()
    for f in frontiers:
        canon_lbl = _canonicalize_sota_term(f["item"])
        if canon_lbl and canon_lbl not in seen_labels:
            seen_labels.add(canon_lbl)
            f_copy = dict(f)
            f_copy["item"] = canon_lbl
            unique_top.append(f_copy)
        if len(unique_top) >= top_n:
            break

    labels = [_wrap_label(f["item"], width=22, max_lines=3) for f in unique_top]
    values = [f["recent_count"] for f in unique_top]

    N = len(labels)
    if N < 3:
        return

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values_plot = values + [values[0]]
    angles += [angles[0]]

    fig, ax = plt.subplots(figsize=(10, 9.5), subplot_kw=dict(polar=True))
    ax.fill(angles, values_plot, alpha=0.25, color="steelblue")
    ax.plot(angles, values_plot, "o-", linewidth=2.2, color="steelblue", markersize=6.5)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])

    max_val = max(values) if values else 1
    r_limit = max_val * 1.15
    ax.set_ylim(0, r_limit)

    # Colocar etiquetas externas con alineacion dependiente del angulo para no invadir el circulo
    label_r = r_limit * 1.06
    for ang, lbl, val, f_item in zip(angles[:-1], labels, values, unique_top):
        cos_a = math.cos(ang)
        sin_a = math.sin(ang)
        if abs(cos_a) < 0.18:
            ha = "center"
        elif cos_a > 0:
            ha = "left"
        else:
            ha = "right"
        if abs(sin_a) < 0.18:
            va = "center"
        elif sin_a > 0:
            va = "bottom"
        else:
            va = "top"

        growth_str = f"{f_item['growth_ratio']}x" if isinstance(f_item.get("growth_ratio"), (int, float)) else "new"
        ax.text(
            ang, label_r, f"{lbl}\n(n={val}, {growth_str})",
            fontsize=8.5, fontweight="bold", color="#1f2937",
            ha=ha, va=va,
        )

    ax.set_title("Research Frontiers (Recent Period)", fontsize=13, fontweight="bold", pad=42)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)
    logger.info(f"Saved: {output_path.name}")


def _plot_temporal_evolution(
    evolution: dict,
    output_path: Path,
) -> None:
    """Stacked area chart of temporal evolution."""
    for dim_name, period_data in evolution.items():
        if not period_data:
            continue

        periods = list(period_data.keys())
        # Get top items that appear in at least 2 periods
        all_items = set()
        for pd_items in period_data.values():
            all_items.update(pd_items.keys())

        active_items = []
        for item in all_items:
            n_periods_present = sum(
                1 for p in periods if period_data[p].get(item, 0) > 0
            )
            if n_periods_present >= 1:
                active_items.append(item)

        if not active_items:
            continue

        # Sort by total occurrence
        active_items.sort(
            key=lambda i: sum(period_data[p].get(i, 0) for p in periods),
            reverse=True,
        )
        active_items = active_items[:12]

        data = {item: [period_data[p].get(item, 0) for p in periods] for item in active_items}
        df = pd.DataFrame(data, index=periods)

        fig, ax = plt.subplots(figsize=(10, 6))
        df.plot.area(ax=ax, alpha=0.7, stacked=True)
        ax.set_title(f"Temporal Evolution: {dim_name.title()}")
        ax.set_xlabel("Period")
        ax.set_ylabel("Normalized Frequency")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
        plt.tight_layout()

        suffix = f"_{dim_name}"
        out = output_path.parent / f"{output_path.stem}{suffix}{output_path.suffix}"
        fig.savefig(out)
        plt.close(fig)
        logger.info(f"Saved: {out.name}")


def _generate_plotly_heatmap(
    matrix: pd.DataFrame,
    title: str,
    output_path: Path,
) -> None:
    """Generate interactive Plotly heatmap."""
    try:
        import plotly.graph_objects as go

        fig = go.Figure(data=go.Heatmap(
            z=matrix.values,
            x=matrix.columns.tolist(),
            y=matrix.index.tolist(),
            colorscale="YlOrRd",
            hovertemplate="%{y} x %{x}: %{z}<extra></extra>",
        ))
        fig.update_layout(
            title=title,
            xaxis_title=matrix.columns.name or "",
            yaxis_title=matrix.index.name or "",
            height=max(500, len(matrix.index) * 30),
            width=max(700, len(matrix.columns) * 40),
        )
        fig.write_html(str(output_path))
        logger.info(f"Saved: {output_path.name}")
    except ImportError:
        logger.debug("Plotly not available for interactive heatmap")


# ===========================================================================
# Report Generation
# ===========================================================================

def generate_sota_report(
    dimensions: dict,
    matrices: dict[str, pd.DataFrame],
    taxonomy: dict,
    frontiers: list[dict],
    evolution: dict,
    performance: list[dict],
    llm_narrative: Optional[str] = None,
    llm_gap_interpretation: Optional[str] = None,
    llm_agenda: Optional[str] = None,
    output_path: Path = None,
) -> Path:
    """Generate comprehensive SOTA/Gap markdown report."""
    if output_path is None:
        output_path = REPORTS_DIR / "sota_gap_report.md"

    lines = [
        "# State of the Art & Knowledge Gap Analysis",
        "",
        "---",
        "",
    ]

    # Section 1: Dimensions panorama
    lines.append("## 1. Research Dimensions Panorama")
    lines.append("")
    for dim_name in ["methods", "applications", "datasets", "concepts"]:
        counter = dimensions.get(dim_name, Counter())
        if counter:
            lines.append(f"### {dim_name.title()} ({len(counter)} unique)")
            for item, count in counter.most_common(15):
                lines.append(f"  - **{item}**: {count} articles")
            lines.append("")

    # Section 2: Coverage matrices
    lines.append("## 2. Coverage Analysis")
    lines.append("")
    for name, matrix in matrices.items():
        if matrix is not None and not matrix.empty:
            lines.append(f"### {name}")
            lines.append(f"- Matrix size: {matrix.shape[0]} x {matrix.shape[1]}")
            total_cells = matrix.shape[0] * matrix.shape[1]
            empty_cells = (matrix == 0).sum().sum()
            coverage = (1 - empty_cells / max(1, total_cells)) * 100
            lines.append(f"- Coverage: {coverage:.1f}% ({total_cells - empty_cells}/{total_cells} cells)")
            lines.append(f"- See heatmap: `sota_analysis/coverage_{name.lower().replace(' ', '_')}.png`")
            lines.append("")

    # Section 3: Gap taxonomy
    lines.append("## 3. Knowledge Gap Taxonomy")
    lines.append("")
    for gap_type in ["structural", "declared", "emergent"]:
        data = taxonomy.get(gap_type, {})
        lines.append(f"### {gap_type.title()} Gaps ({data.get('count', 0)})")
        lines.append(f"_{data.get('description', '')}_")
        lines.append("")
        for g in data.get("top_gaps", [])[:10]:
            if gap_type == "structural":
                lines.append(
                    f"  - **{g['item_a']}** x **{g['item_b']}** "
                    f"(freq: {g['item_a_freq']} / {g['item_b_freq']})"
                )
            elif gap_type == "declared":
                lines.append(
                    f"  - [{g.get('year', '?')}] {g.get('sentence', '')[:120]}..."
                )
            elif gap_type == "emergent":
                lines.append(
                    f"  - **{g.get('frontier_item', '')}** "
                    f"(growth: {g.get('growth_ratio', '?')}x, "
                    f"{g.get('uncovered_combinations', 0)} uncovered combos)"
                )
        lines.append("")

    lines.append(f"**Total gaps identified: {taxonomy.get('total_gaps', 0)}**")
    lines.append("")

    # Section 4: Frontiers
    lines.append("## 4. Research Frontiers")
    lines.append("")
    if frontiers:
        lines.append("| Item | Dimension | Growth | Recent Count |")
        lines.append("|------|-----------|--------|-------------|")
        for f in frontiers[:15]:
            lines.append(
                f"| {f['item']} | {f['dimension']} | "
                f"{f['growth_ratio']}x | {f['recent_count']} |"
            )
        lines.append("")
    else:
        lines.append("_No significant growth patterns detected._")
        lines.append("")

    # Section 5: Performance landscape
    lines.append("## 5. Performance Landscape")
    lines.append("")
    if performance:
        perf_df = pd.DataFrame(performance)
        for metric in perf_df["metric"].unique():
            metric_data = perf_df[perf_df["metric"] == metric]
            vals = metric_data["value"]
            lines.append(
                f"- **{metric}**: n={len(vals)}, "
                f"mean={vals.mean():.1f}, max={vals.max():.1f}, "
                f"min={vals.min():.1f}"
            )
        lines.append("")
    else:
        lines.append("_No quantitative performance metrics detected._")
        lines.append("")

    # Footer
    lines.append("---")
    lines.append("_Generated by SOTA/Gap Analysis Module (Phase 2f)_")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"SOTA report saved: {output_path}")
    return output_path


# ===========================================================================
# Main orchestrator
# ===========================================================================

def run_phase2f_sota(articles: list[dict]) -> dict:
    """Execute Phase 2f: SOTA + Gap Analysis (100% algorithmic).

    Args:
        articles: List of enriched article dicts.

    Returns:
        dict with all analysis results.
    """
    logger.info("=== FASE 2f: Estado del Arte + Vacios de Conocimiento ===")

    # Create output directories
    sota_fig_dir = SOTA_DIR / "figures"
    sota_plotly_dir = SOTA_DIR / "plotly"
    sota_fig_dir.mkdir(parents=True, exist_ok=True)
    sota_plotly_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # A1: Extract dimensions
    logger.info("A1: Extracting research dimensions...")
    dimensions = extract_research_dimensions(articles)
    results["dimensions"] = {k: dict(v.most_common(30)) for k, v in dimensions.items()}

    # A2: Build coverage matrices
    logger.info("A2: Building coverage matrices...")
    matrices = {}

    if dimensions["methods"] and dimensions["applications"]:
        m = build_coverage_matrix(
            articles, dimensions["methods"], dimensions["applications"],
            "Methods", "Applications",
        )
        matrices["Methods x Applications"] = m
        _plot_coverage_heatmap(
            m, "Coverage: Methods x Applications",
            sota_fig_dir / "coverage_methods_applications.png",
        )
        _generate_plotly_heatmap(
            m, "Coverage: Methods x Applications",
            sota_plotly_dir / "coverage_methods_applications.html",
        )

    if dimensions["methods"] and dimensions["datasets"]:
        m = build_coverage_matrix(
            articles, dimensions["methods"], dimensions["datasets"],
            "Methods", "Datasets",
        )
        matrices["Methods x Datasets"] = m
        _plot_coverage_heatmap(
            m, "Coverage: Methods x Datasets",
            sota_fig_dir / "coverage_methods_datasets.png",
        )

    if dimensions["methods"] and dimensions["concepts"]:
        m = build_coverage_matrix(
            articles, dimensions["methods"], dimensions["concepts"],
            "Methods", "Concepts", top_n=15,
        )
        matrices["Methods x Concepts"] = m
        _plot_coverage_heatmap(
            m, "Coverage: Methods x Concepts",
            sota_fig_dir / "coverage_methods_concepts.png",
        )

    results["matrices_shapes"] = {k: list(v.shape) for k, v in matrices.items()}

    # A3: Structural gaps
    logger.info("A3: Detecting structural gaps...")
    raw_structural_gaps = []
    for name, matrix in matrices.items():
        dim_a_key = (matrix.index.name or "").lower()
        dim_b_key = (matrix.columns.name or "").lower()
        gaps = detect_structural_gaps(
            matrix,
            dim_a_counts=dimensions.get(dim_a_key),
            dim_b_counts=dimensions.get(dim_b_key),
        )
        raw_structural_gaps.extend(gaps)
    raw_structural_gaps.sort(key=lambda g: g["priority"], reverse=True)
    all_structural_gaps = []
    seen_gap_pairs: set[tuple[str, str]] = set()
    for g in raw_structural_gaps:
        pair_key = (_canonicalize_sota_term(g["item_a"]), _canonicalize_sota_term(g["item_b"]))
        if pair_key not in seen_gap_pairs:
            seen_gap_pairs.add(pair_key)
            all_structural_gaps.append(g)
    results["structural_gaps_count"] = len(all_structural_gaps)

    if all_structural_gaps:
        _plot_gap_priority_scatter(
            all_structural_gaps,
            sota_fig_dir / "gap_priority_scatter.png",
        )

    # A4: Declared gaps
    logger.info("A4: Analyzing declared gaps...")
    declared_gaps = analyze_declared_gaps(articles)
    results["declared_gaps_count"] = len(declared_gaps)

    # A5: Temporal evolution
    logger.info("A5: Analyzing temporal gap evolution...")
    evolution = analyze_temporal_gap_evolution(articles, dimensions)
    results["temporal_periods"] = len(evolution.get("methods", {}))

    if evolution:
        _plot_temporal_evolution(
            evolution,
            sota_fig_dir / "temporal_evolution.png",
        )

    # A6: Research frontiers
    logger.info("A6: Identifying research frontiers...")
    frontiers = identify_research_frontiers(articles, dimensions)
    results["frontiers_count"] = len(frontiers)

    if frontiers:
        _plot_frontier_radar(
            frontiers,
            sota_fig_dir / "frontier_radar.png",
        )

    # A7: Emergent gaps
    logger.info("A7: Detecting emergent gaps...")
    emergent_gaps = detect_emergent_gaps(frontiers, all_structural_gaps)
    results["emergent_gaps_count"] = len(emergent_gaps)

    # A8: Performance landscape
    logger.info("A8: Building performance landscape...")
    performance = build_performance_landscape(articles)
    results["performance_metrics_count"] = len(performance)

    # A9: Gap taxonomy
    logger.info("A9: Building gap taxonomy...")
    taxonomy = build_gap_taxonomy(all_structural_gaps, declared_gaps, emergent_gaps)
    results["taxonomy"] = {
        k: {"count": v["count"], "description": v["description"]}
        for k, v in taxonomy.items()
        if isinstance(v, dict) and "count" in v
    }

    # Generate report
    report_path = generate_sota_report(
        dimensions=dimensions,
        matrices=matrices,
        taxonomy=taxonomy,
        frontiers=frontiers,
        evolution=evolution,
        performance=performance,
        output_path=SOTA_DIR / "sota_gap_report.md",
    )
    results["report_path"] = str(report_path)

    # --- Save structured JSON for Knowledge Map consumption ---
    data_export = {
        "structural_gaps": [
            {
                "item_a": g.get("row", g.get("item_a", "")),
                "item_b": g.get("col", g.get("item_b", "")),
                "dimension_a": g.get("dimension_a", g.get("row_dimension", "")),
                "dimension_b": g.get("dimension_b", g.get("col_dimension", "")),
                "freq_a": g.get("row_freq", g.get("freq_a", 0)),
                "freq_b": g.get("col_freq", g.get("freq_b", 0)),
                "priority": g.get("priority", 0),
            }
            for g in all_structural_gaps
        ],
        "declared_gaps": [
            {
                "sentence": dg.get("sentence", ""),
                "article_key": dg.get("article_key", dg.get("key", "")),
                "year": dg.get("year", ""),
            }
            for dg in declared_gaps
        ],
        "frontiers": [
            {
                "item": fr.get("item", ""),
                "type": fr.get("type", ""),
                "growth_ratio": fr.get("growth_ratio", 0),
                "recent_count": fr.get("recent_count", 0),
                "uncovered_combinations": fr.get("uncovered_combinations", 0),
            }
            for fr in frontiers
        ],
        "emergent_gaps": [
            {
                "item": eg.get("item", ""),
                "growth_ratio": eg.get("growth_ratio", 0),
                "uncovered_combinations": eg.get("uncovered_combinations", 0),
            }
            for eg in emergent_gaps
        ],
        "taxonomy": {
            k: {
                "count": v.get("count", 0),
                "description": v.get("description", ""),
                "top_gaps": v.get("top_gaps", [])[:10],
            }
            for k, v in taxonomy.items()
            if isinstance(v, dict) and "count" in v
        },
        "evolution_periods": len(evolution.get("methods", {})),
        "performance_metrics_count": len(performance),
    }
    try:
        save_json(data_export, SOTA_DIR / "sota_results.json")
        logger.info(f"SOTA data saved: {len(all_structural_gaps)} gaps, "
                     f"{len(frontiers)} frontiers → sota_results.json")
    except Exception as e:
        logger.warning(f"SOTA JSON save failed: {e}")

    logger.info(
        f"Phase 2f complete: {taxonomy.get('total_gaps', 0)} gaps, "
        f"{len(frontiers)} frontiers, {len(performance)} metrics"
    )

    return results

run_phase2g_sota = run_phase2f_sota
