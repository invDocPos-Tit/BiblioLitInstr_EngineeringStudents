"""Fase 2e: Indicadores bibliometricos suplementarios (agnostico).

Incluye:
  - Conceptual Structure Map (Correspondence Analysis / MDS sobre keywords)
  - Trend Topics (keywords x year scatter)
  - Produccion acumulativa (top autores y fuentes)
  - Author Dominance Factor
  - Citation Half-life
  - CAGR (Compound Annual Growth Rate)
"""
from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import FIGURES_DIR, PLOTLY_DIR, REPORTS_DIR

logger = logging.getLogger("bibliometrico.supplementary")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_keywords(article: dict, articles: Optional[list[dict]] = None) -> list[str]:
    """Palabras clave normalizadas con el vocabulario comun del pipeline (src.utils.article_keywords)."""
    from src.utils import article_keywords
    return article_keywords(article, articles)


def compute_conceptual_structure(
    articles: list[dict],
    top_n_keywords: int = 50,
    method: str = "ca",
) -> dict:
    """Calcula estructura conceptual via Correspondence Analysis o MDS sobre keywords.

    Construye la matriz binaria keywords x articles a partir de manual_tags
    (con fallback a automatic_tags).

    Args:
        articles: Lista de articulos.
        top_n_keywords: Numero de keywords mas frecuentes a incluir.
        method: 'ca' (Correspondence Analysis via prince/SVD) o 'mds'.

    Returns:
        dict con 'coords' (keyword -> (x, y)), 'clusters', 'explained_var'.
    """
    kw_freq: Counter = Counter()
    for a in articles:
        for kw in _extract_keywords(a, articles):
            kw_freq[kw] += 1

    row_labels = [kw for kw, _ in kw_freq.most_common(top_n_keywords)]
    row_freqs = {kw: kw_freq[kw] for kw in row_labels}

    n_kw = len(row_labels)
    n_art = len(articles)
    kw_idx = {kw: i for i, kw in enumerate(row_labels)}

    matrix = np.zeros((n_kw, n_art), dtype=float)
    for j, a in enumerate(articles):
        for kw in _extract_keywords(a, articles):
            if kw in kw_idx:
                matrix[kw_idx[kw], j] = 1.0

    logger.info(f"Conceptual structure: using {n_kw} keywords")

    top_kws = row_labels  # unified name for downstream code

    if len(top_kws) < 5:
        logger.warning("Too few items for conceptual structure")
        return {"coords": {}, "clusters": {}, "explained_var": []}

    # Eliminar rows sin apariciones (seguridad)
    row_sums = matrix.sum(axis=1)
    valid = row_sums > 0
    matrix = matrix[valid]
    valid_kws = [kw for kw, v in zip(top_kws, valid) if v]
    n_kw = len(valid_kws)

    if n_kw < 5:
        logger.warning("Too few valid keywords after filtering")
        return {"coords": {}, "clusters": {}, "explained_var": []}

    coords = {}
    explained_var = []

    # Filter out zero-sum columns (articles with no top keywords)
    col_sums = matrix.sum(axis=0)
    valid_cols = col_sums > 0
    matrix = matrix[:, valid_cols]
    n_art_valid = matrix.shape[1]

    # Build DataFrame for prince / fallback
    df = pd.DataFrame(matrix, index=valid_kws,
                      columns=[f"art_{j}" for j in range(n_art_valid)])

    if method == "ca":
        import prince
        ca = prince.CA(n_components=2)
        ca = ca.fit(df)

        row_coords = ca.row_coordinates(df)
        explained_var = list(ca.percentage_of_variance_)[:2]

        for kw in valid_kws:
            x = float(row_coords.loc[kw, 0])
            y = float(row_coords.loc[kw, 1]) if row_coords.shape[1] > 1 else 0.0
            coords[kw] = (x, y)

        logger.info(
            f"CA (prince): Dim1={explained_var[0]:.1f}%, "
            f"Dim2={explained_var[1]:.1f}% explained"
        )

    if method == "mds" or (method == "ca" and not coords):
        # MDS fallback: distance = 1 - Jaccard on co-occurrence
        from sklearn.manifold import MDS
        from scipy.spatial.distance import pdist, squareform

        dist = pdist(matrix, metric="jaccard")
        dist_matrix = squareform(dist)
        # Replace NaN (from zero vectors) with max distance
        dist_matrix = np.nan_to_num(dist_matrix, nan=1.0)

        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42,
                   n_init=4, max_iter=300, normalized_stress="auto")
        embedding = mds.fit_transform(dist_matrix)

        for i, kw in enumerate(valid_kws):
            coords[kw] = (float(embedding[i, 0]), float(embedding[i, 1]))
        explained_var = []  # MDS doesn't have explained variance

    # Cluster keywords using k-means on 2D coords
    clusters = {}
    if coords:
        try:
            from sklearn.cluster import KMeans
            coord_array = np.array([coords[kw] for kw in valid_kws])
            n_clusters = min(5, max(2, n_kw // 8))
            km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = km.fit_predict(coord_array)
            for i, kw in enumerate(valid_kws):
                clusters[kw] = int(labels[i])
        except ImportError:
            pass

    logger.info(
        f"Conceptual structure ({method.upper()}): "
        f"{len(coords)} keywords mapped to 2D"
    )

    return {
        "coords": coords,
        "clusters": clusters,
        "explained_var": explained_var,
        "method": method,
        "frequencies": {kw: row_freqs.get(kw, 0) for kw in valid_kws},
        "source": "keywords",
    }


def plot_conceptual_structure_static(
    cs_data: dict,
    output_dir: Path = FIGURES_DIR,
) -> Optional[Path]:
    """Genera mapa de estructura conceptual estatico (matplotlib)."""
    coords = cs_data.get("coords", {})
    clusters = cs_data.get("clusters", {})
    freqs = cs_data.get("frequencies", {})
    explained = cs_data.get("explained_var", [])
    method = cs_data.get("method", "ca")

    if not coords:
        return None

    fig, ax = plt.subplots(figsize=(14, 10))

    color_palette = plt.cm.Set2(np.linspace(0, 1, 8))
    cluster_ids = sorted(set(clusters.values())) if clusters else [0]

    for kw, (x, y) in coords.items():
        cid = clusters.get(kw, 0)
        freq = freqs.get(kw, 1)
        size = max(30, min(300, freq * 15))
        color = color_palette[cid % len(color_palette)]

        ax.scatter(x, y, s=size, color=color, alpha=0.6,
                   edgecolors="black", linewidth=0.3, zorder=3)
        ax.annotate(kw, (x, y), textcoords="offset points",
                    xytext=(4, 4), fontsize=6, alpha=0.85, zorder=4)

    # Axes labels
    if explained and len(explained) >= 2:
        ax.set_xlabel(f"Dim 1 ({explained[0]:.1f}%)", fontsize=11)
        ax.set_ylabel(f"Dim 2 ({explained[1]:.1f}%)", fontsize=11)
    else:
        ax.set_xlabel("Dimension 1", fontsize=11)
        ax.set_ylabel("Dimension 2", fontsize=11)

    method_name = "Correspondence Analysis" if method == "ca" else "MDS"
    ax.set_title(f"Conceptual Structure Map ({method_name})",
                 fontsize=14, fontweight="bold")
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.3)
    ax.axvline(0, color="gray", linewidth=0.5, alpha=0.3)

    # Cluster legend
    import matplotlib.patches as mpatches
    patches = []
    for cid in cluster_ids:
        color = color_palette[cid % len(color_palette)]
        cluster_kws = [kw for kw, c in clusters.items() if c == cid]
        top = sorted(cluster_kws, key=lambda k: freqs.get(k, 0), reverse=True)[:2]
        label = f"Cluster {cid + 1}: {', '.join(top)}"
        patches.append(mpatches.Patch(color=color, alpha=0.6, label=label))
    if patches:
        ax.legend(handles=patches, loc="best", fontsize=7, framealpha=0.8)

    plt.tight_layout()
    out_path = output_dir / "conceptual_structure_map.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info(f"Conceptual structure map saved: {out_path}")
    return out_path


def plot_conceptual_structure_plotly(
    cs_data: dict,
    output_dir: Path = PLOTLY_DIR,
) -> Optional[Path]:
    """Genera mapa de estructura conceptual interactivo (Plotly)."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        logger.warning("plotly not installed, skipping conceptual structure plotly")
        return None

    coords = cs_data.get("coords", {})
    clusters = cs_data.get("clusters", {})
    freqs = cs_data.get("frequencies", {})
    explained = cs_data.get("explained_var", [])
    method = cs_data.get("method", "ca")

    if not coords:
        return None

    color_palette = [
        "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854",
        "#ffd92f", "#e5c494", "#b3b3b3",
    ]

    cluster_ids = sorted(set(clusters.values())) if clusters else [0]

    fig = go.Figure()

    for cid in cluster_ids:
        kws_in_cluster = [kw for kw, c in clusters.items() if c == cid]
        xs = [coords[kw][0] for kw in kws_in_cluster]
        ys = [coords[kw][1] for kw in kws_in_cluster]
        sizes = [max(8, min(40, freqs.get(kw, 1) * 3)) for kw in kws_in_cluster]
        hover = [
            f"<b>{kw}</b><br>Freq: {freqs.get(kw, 0)}<br>Cluster: {cid + 1}"
            for kw in kws_in_cluster
        ]

        top_kws = sorted(kws_in_cluster,
                         key=lambda k: freqs.get(k, 0), reverse=True)[:2]
        cluster_label = f"Cluster {cid + 1}: {', '.join(top_kws)}"

        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers+text",
            marker=dict(
                size=sizes,
                color=color_palette[cid % len(color_palette)],
                opacity=0.7,
                line=dict(width=0.5, color="black"),
            ),
            text=kws_in_cluster,
            textposition="top center",
            textfont=dict(size=8),
            hovertext=hover,
            hoverinfo="text",
            name=cluster_label,
        ))

    method_name = "Correspondence Analysis" if method == "ca" else "MDS"
    x_label = f"Dim 1 ({explained[0]:.1f}%)" if explained else "Dimension 1"
    y_label = f"Dim 2 ({explained[1]:.1f}%)" if len(explained) > 1 else "Dimension 2"

    fig.update_layout(
        title=f"Conceptual Structure Map ({method_name})",
        xaxis_title=x_label,
        yaxis_title=y_label,
        template="plotly_white",
        width=1000, height=750,
    )

    out_path = output_dir / "conceptual_structure_map.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    logger.info(f"Conceptual structure map (Plotly) saved: {out_path}")
    return out_path


# ===========================================================================
# 2. TREND TOPICS (keywords x year)
# ===========================================================================

def compute_trend_topics(
    articles: list[dict],
    top_n_per_year: int = 5,
    min_freq: int = 2,
) -> dict:
    """Calcula keywords mas frecuentes por ano.

    Returns:
        dict con 'trend_data' lista de {year, keyword, freq, cumulative},
        'years', 'top_keywords'.
    """
    # keyword freq by year
    kw_year: dict[int, Counter] = defaultdict(Counter)
    kw_total: Counter = Counter()

    for a in articles:
        year = a.get("publication_year", 0)
        if not year:
            continue
        for kw in _extract_keywords(a, articles):
            kw_year[year][kw] += 1
            kw_total[kw] += 1

    years = sorted(kw_year.keys())
    if not years:
        return {"trend_data": [], "years": [], "top_keywords": []}

    # Top keywords per year
    trend_data = []
    all_top_kws = set()

    for year in years:
        top = kw_year[year].most_common(top_n_per_year)
        for kw, freq in top:
            if freq >= min_freq:
                trend_data.append({
                    "year": year,
                    "keyword": kw,
                    "freq": freq,
                })
                all_top_kws.add(kw)

    # Also compute global top keywords with yearly trajectory
    global_top = [kw for kw, _ in kw_total.most_common(top_n_per_year * 3)]
    trajectory = {}
    for kw in global_top:
        traj = []
        for year in years:
            traj.append(kw_year[year].get(kw, 0))
        trajectory[kw] = traj

    logger.info(f"Trend topics: {len(trend_data)} entries across {len(years)} years")

    # Metodo estandar de trend topics (Aria y Cuccurullo, bibliometrix::fieldByYear): cada palabra
    # se ubica en el ano MEDIANO de los documentos que la usan (con Q1 y Q3 como recorrido); por cada
    # ano mediano se toman las n palabras mas frecuentes. Lo que se grafica es esto. La lista anterior
    # ('trend_data', las n mas frecuentes de cada ano) se conserva solo como dato: repite el termino
    # general del campo casi todos los anos y no muestra tendencia.
    anos_por_kw: dict[str, list[int]] = defaultdict(list)
    for y, c in kw_year.items():
        for kw, n in c.items():
            anos_por_kw[kw].extend([int(y)] * int(n))
    por_mediana: dict[int, list[dict]] = defaultdict(list)
    for kw, ys in anos_por_kw.items():
        if len(ys) < min_freq:
            continue
        q1, med, q3 = (int(round(float(v))) for v in np.percentile(ys, [25, 50, 75]))
        por_mediana[med].append({"keyword": kw, "freq": len(ys), "year_q1": q1, "year_med": med, "year_q3": q3})
    median_year_topics = []
    for med in sorted(por_mediana):
        sel = sorted(por_mediana[med], key=lambda d: (-d["freq"], d["keyword"]))[:top_n_per_year]
        median_year_topics.extend(sel)

    return {
        "metodo": "ano mediano de uso por palabra (Q1-Q3), top n por ano mediano (como bibliometrix::fieldByYear)",
        "median_year_topics": median_year_topics,
        "trend_data": trend_data,
        "years": years,
        "top_keywords": sorted(all_top_kws),
        "trajectory": trajectory,
        "global_top": global_top,
        "kw_year": {y: dict(c) for y, c in kw_year.items()},
    }


def plot_trend_topics_static(
    trend_data: dict,
    output_dir: Path = FIGURES_DIR,
) -> Optional[Path]:
    """Trend topics (matplotlib): por palabra, recorrido Q1-Q3 del ano de uso y punto en el ano mediano."""
    data = trend_data.get("median_year_topics", [])
    if not data:
        return None
    data = sorted(data, key=lambda d: (d["year_med"], d["freq"], d["keyword"]))
    fig, ax = plt.subplots(figsize=(12, max(6, len(data) * 0.28)))
    max_f = max(d["freq"] for d in data)
    for i, d in enumerate(data):
        ax.plot([d["year_q1"], d["year_q3"]], [i, i], color="#9ecae1", linewidth=2, zorder=2)
        ax.scatter(d["year_med"], i, s=max(30, 400 * d["freq"] / max_f), color="#2171b5",
                   edgecolors="black", linewidth=0.3, zorder=3)
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels([f"{d['keyword']} ({d['freq']})" for d in data], fontsize=7)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_title("Trend Topics (median year of use, Q1-Q3; size = frequency)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = output_dir / "trend_topics.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info(f"Trend topics saved: {out_path}")
    return out_path


def plot_trend_topics_plotly(
    trend_data: dict,
    output_dir: Path = PLOTLY_DIR,
) -> Optional[Path]:
    """Trend topics interactivo (Plotly): mismo metodo que la figura estatica (ano mediano, Q1-Q3)."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        logger.warning("plotly not installed, skipping trend topics plotly")
        return None

    data = trend_data.get("median_year_topics", [])
    if not data:
        return None
    data = sorted(data, key=lambda d: (d["year_med"], d["freq"], d["keyword"]))
    labels = [d["keyword"] for d in data]
    max_f = max(d["freq"] for d in data)
    fig = go.Figure()
    for d in data:
        fig.add_trace(go.Scatter(x=[d["year_q1"], d["year_q3"]], y=[d["keyword"], d["keyword"]], mode="lines",
                                 line=dict(color="#9ecae1", width=3), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(
        x=[d["year_med"] for d in data], y=labels, mode="markers",
        marker=dict(size=[max(8, 30 * d["freq"] / max_f) for d in data], color="#2171b5",
                    line=dict(width=0.5, color="black")),
        hovertext=[f"<b>{d['keyword']}</b><br>Frecuencia: {d['freq']}<br>Año mediano: {d['year_med']}"
                   f"<br>Q1-Q3: {d['year_q1']}-{d['year_q3']}" for d in data],
        hoverinfo="text", showlegend=False,
    ))
    fig.update_layout(
        title="Trend Topics (median year of use, Q1-Q3; size = frequency)",
        xaxis_title="Year", yaxis_title="",
        yaxis=dict(categoryorder="array", categoryarray=labels),
        template="plotly_white", width=1000, height=max(500, len(labels) * 22),
    )
    out_path = output_dir / "trend_topics.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    logger.info(f"Trend topics (Plotly) saved: {out_path}")
    return out_path


# ===========================================================================
# 3. PRODUCCION ACUMULATIVA
# ===========================================================================

def compute_cumulative_production(
    articles: list[dict],
    top_n_authors: int = 10,
    top_n_sources: int = 10,
) -> dict:
    """Calcula produccion acumulativa de top autores y fuentes por ano.

    Returns:
        dict con 'authors' y 'sources', cada uno con {name: {year: cumulative}}.
    """
    # Count per year
    author_year: dict[str, Counter] = defaultdict(Counter)
    source_year: dict[str, Counter] = defaultdict(Counter)

    author_total: Counter = Counter()
    source_total: Counter = Counter()

    for a in articles:
        year = a.get("publication_year", 0)
        if not year:
            continue

        for auth in a.get("authors", []):
            name = auth.strip()
            if name:
                author_year[name][year] += 1
                author_total[name] += 1

        source = a.get("publication_title", "").strip()
        if source:
            source_year[source][year] += 1
            source_total[source] += 1

    years = sorted(set(
        y for a in articles if (y := a.get("publication_year", 0))
    ))

    # Top authors/sources
    top_authors = [name for name, _ in author_total.most_common(top_n_authors)]
    top_sources = [name for name, _ in source_total.most_common(top_n_sources)]

    # Build cumulative series
    def _cumulative(entity_year: dict, entities: list, years_list: list) -> dict:
        result = {}
        for entity in entities:
            cum = 0
            series = {}
            for y in years_list:
                cum += entity_year[entity].get(y, 0)
                series[y] = cum
            result[entity] = series
        return result

    authors_cum = _cumulative(author_year, top_authors, years)
    sources_cum = _cumulative(source_year, top_sources, years)

    logger.info(
        f"Cumulative production: {len(top_authors)} authors, "
        f"{len(top_sources)} sources"
    )

    return {
        "authors": authors_cum,
        "sources": sources_cum,
        "years": years,
    }


def plot_cumulative_production_static(
    cum_data: dict,
    output_dir: Path = FIGURES_DIR,
) -> Optional[Path]:
    """Genera graficos de produccion acumulativa (matplotlib)."""
    years = cum_data.get("years", [])
    if not years:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Authors
    ax = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, (name, series) in enumerate(cum_data.get("authors", {}).items()):
        ys = [series.get(y, 0) for y in years]
        short_name = name[:25] if len(name) > 25 else name
        ax.plot(years, ys, marker="o", markersize=3, linewidth=1.5,
                color=colors[i % len(colors)], label=short_name)
    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative Documents")
    ax.set_title("Top Authors — Cumulative Production", fontweight="bold")
    ax.legend(fontsize=6, loc="upper left", framealpha=0.8)

    # Sources
    ax = axes[1]
    for i, (name, series) in enumerate(cum_data.get("sources", {}).items()):
        ys = [series.get(y, 0) for y in years]
        short_name = name[:30] if len(name) > 30 else name
        ax.plot(years, ys, marker="s", markersize=3, linewidth=1.5,
                color=colors[i % len(colors)], label=short_name)
    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative Documents")
    ax.set_title("Top Sources — Cumulative Production", fontweight="bold")
    ax.legend(fontsize=5, loc="upper left", framealpha=0.8)

    plt.tight_layout()
    out_path = output_dir / "cumulative_production.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info(f"Cumulative production saved: {out_path}")
    return out_path


def plot_cumulative_production_plotly(
    cum_data: dict,
    output_dir: Path = PLOTLY_DIR,
) -> Optional[Path]:
    """Genera produccion acumulativa interactiva (Plotly)."""
    try:
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
    except ImportError:
        logger.warning("plotly not installed, skipping cumulative production plotly")
        return None

    years = cum_data.get("years", [])
    if not years:
        return None

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Top Authors", "Top Sources"),
    )

    colors = [
        "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
        "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
    ]

    for i, (name, series) in enumerate(cum_data.get("authors", {}).items()):
        ys = [series.get(y, 0) for y in years]
        fig.add_trace(go.Scatter(
            x=years, y=ys, mode="lines+markers",
            name=name[:25], line=dict(color=colors[i % len(colors)]),
            marker=dict(size=4),
            legendgroup="authors",
        ), row=1, col=1)

    for i, (name, series) in enumerate(cum_data.get("sources", {}).items()):
        ys = [series.get(y, 0) for y in years]
        fig.add_trace(go.Scatter(
            x=years, y=ys, mode="lines+markers",
            name=name[:30], line=dict(color=colors[i % len(colors)], dash="dot"),
            marker=dict(size=4, symbol="square"),
            legendgroup="sources",
        ), row=1, col=2)

    fig.update_layout(
        title="Cumulative Production Over Time",
        template="plotly_white",
        width=1200, height=550,
    )
    fig.update_xaxes(title_text="Year")
    fig.update_yaxes(title_text="Cumulative Documents")

    out_path = output_dir / "cumulative_production.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    logger.info(f"Cumulative production (Plotly) saved: {out_path}")
    return out_path


# ===========================================================================
# 4. AUTHOR DOMINANCE FACTOR
# ===========================================================================

def compute_author_dominance(
    articles: list[dict],
    top_n: int = 20,
) -> list[dict]:
    """Calcula dominance factor por autor.

    Dominance Factor = articulos como primer autor / total articulos.
    Valores altos indican liderazgo; bajos indican rol colaborativo.

    Returns:
        Lista de dicts ordenada por total descendente.
    """
    author_total: Counter = Counter()
    author_first: Counter = Counter()

    for a in articles:
        authors = a.get("authors", [])
        if not authors:
            continue
        first = authors[0].strip()
        for auth in authors:
            name = auth.strip()
            if name:
                author_total[name] += 1
        if first:
            author_first[first] += 1

    # Only authors with >= 2 articles
    results = []
    for name, total in author_total.most_common(top_n):
        if total < 2:
            continue
        first = author_first.get(name, 0)
        multi = total - first
        df = round(first / total, 3) if total > 0 else 0
        results.append({
            "author": name,
            "total": total,
            "first_authored": first,
            "multi_authored": multi,
            "dominance_factor": df,
        })

    results.sort(key=lambda x: x["total"], reverse=True)
    logger.info(f"Author dominance: {len(results)} authors with >=2 articles")
    return results


def plot_author_dominance_static(
    dominance_data: list[dict],
    output_dir: Path = FIGURES_DIR,
) -> Optional[Path]:
    """Genera grafico de dominance factor (matplotlib)."""
    if not dominance_data:
        return None

    data = dominance_data[:20]
    fig, ax = plt.subplots(figsize=(12, max(5, len(data) * 0.35)))

    names = [d["author"][:25] for d in data]
    first = [d["first_authored"] for d in data]
    multi = [d["multi_authored"] for d in data]
    df_vals = [d["dominance_factor"] for d in data]

    y_pos = range(len(names))
    ax.barh(y_pos, first, color="#2ca02c", label="First Author", edgecolor="white")
    ax.barh(y_pos, multi, left=first, color="#aec7e8", label="Co-Author", edgecolor="white")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Articles")
    ax.set_title("Author Dominance Factor", fontweight="bold")
    ax.legend(loc="lower right")

    # DF annotations
    for i, df_val in enumerate(df_vals):
        total = first[i] + multi[i]
        ax.text(total + 0.15, i, f"DF={df_val:.2f}",
                va="center", fontsize=7, color="#2ca02c", fontweight="bold")

    plt.tight_layout()
    out_path = output_dir / "author_dominance_factor.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info(f"Author dominance saved: {out_path}")
    return out_path


def plot_author_dominance_plotly(
    dominance_data: list[dict],
    output_dir: Path = PLOTLY_DIR,
) -> Optional[Path]:
    """Genera grafico de dominance factor interactivo (Plotly)."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    if not dominance_data:
        return None

    data = list(reversed(dominance_data[:20]))  # reverse for horizontal bar
    names = [d["author"][:25] for d in data]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=names, x=[d["first_authored"] for d in data],
        name="First Author", orientation="h", marker_color="#2ca02c",
    ))
    fig.add_trace(go.Bar(
        y=names, x=[d["multi_authored"] for d in data],
        name="Co-Author", orientation="h", marker_color="#aec7e8",
    ))

    # DF annotations
    for i, d in enumerate(data):
        fig.add_annotation(
            x=d["total"] + 0.2, y=d["author"][:25],
            text=f"DF={d['dominance_factor']:.2f}",
            showarrow=False, font=dict(size=9, color="#2ca02c"),
            xanchor="left",
        )

    fig.update_layout(
        title="Author Dominance Factor (first author / total)",
        xaxis_title="Number of Articles",
        barmode="stack",
        template="plotly_white",
        width=900, height=max(400, len(data) * 28),
        margin=dict(l=180),
    )

    out_path = output_dir / "author_dominance_factor.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    logger.info(f"Author dominance (Plotly) saved: {out_path}")
    return out_path


# ===========================================================================
# 5. CITATION HALF-LIFE
# ===========================================================================

def _extract_ref_years_from_crossref_cache(
    articles: list[dict], max_corpus_year: int,
) -> list[tuple[int, int]]:
    """Extrae anos de referencias desde Crossref cache (lookup por DOI hash).

    Much faster than scanning all cache files — directly looks up each
    article's DOI in the Crossref cache and extracts reference years.
    """
    ref_years = []
    try:
        from config import CROSSREF_CACHE_DIR
        import hashlib
        import json as _json

        n_hits = 0
        for a in articles:
            doi = a.get("doi", "")
            if not doi or doi == "nan":
                continue
            art_year = a.get("publication_year", max_corpus_year)

            cache_key = f"crossref:doi:{doi}"
            h = hashlib.md5(cache_key.encode()).hexdigest()
            cache_path = CROSSREF_CACHE_DIR / f"{h}.json"

            if not cache_path.exists():
                continue

            try:
                data = _json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                continue

            refs = data.get("reference", [])
            if not refs:
                continue

            n_hits += 1
            for ref in refs:
                ry = ref.get("year")
                if ry:
                    try:
                        ry = int(ry)
                        if 1900 <= ry <= max_corpus_year:
                            ref_years.append((ry, art_year))
                    except (ValueError, TypeError):
                        pass

        if ref_years:
            logger.info(
                f"Crossref cache: {n_hits} articles with references, "
                f"{len(ref_years)} ref-years extracted"
            )
    except Exception as e:
        logger.debug(f"Crossref cache lookup failed: {e}")

    return ref_years


def _extract_ref_years_from_cache(
    articles: list[dict], max_corpus_year: int,
) -> list[tuple[int, int]]:
    """Extrae anos de referencias desde OpenAlex cache (scan directo).

    Optimized: only scans cache if articles have referenced_works URLs.
    """
    # Early exit: check if any article has referenced_works to look up
    has_refs = any(
        a.get("referenced_works") and len(a["referenced_works"]) > 0
        for a in articles
    )
    if not has_refs:
        logger.info("No referenced_works URLs in articles, skipping cache scan")
        return []

    ref_years = []
    try:
        from config import OPENALEX_CACHE_DIR
        import json as _json
        import os

        # Collect all referenced URLs we need to look up
        needed_urls: set[str] = set()
        for a in articles:
            for ref_url in a.get("referenced_works", []):
                if ref_url:
                    needed_urls.add(ref_url)

        if not needed_urls:
            return []

        logger.info(f"Looking up {len(needed_urls)} reference URLs in cache...")

        # Build lookup: openalex_id -> publication_year from cache files
        ref_year_lookup: dict[str, int] = {}
        cache_dir = str(OPENALEX_CACHE_DIR)
        n_scanned = 0
        for fname in os.listdir(cache_dir):
            if not fname.endswith(".json"):
                continue
            n_scanned += 1
            try:
                fpath = os.path.join(cache_dir, fname)
                with open(fpath, "r", encoding="utf-8") as f:
                    data = _json.load(f)
                if isinstance(data, dict):
                    oa_id = data.get("id", "")
                    py = data.get("publication_year")
                    if oa_id and py and isinstance(py, int):
                        ref_year_lookup[oa_id] = py
            except Exception:
                continue

            # Progress every 5000 files
            if n_scanned % 5000 == 0:
                logger.info(f"  Cache scan: {n_scanned} files...")

        if not ref_year_lookup:
            return []

        logger.info(f"OpenAlex cache: {len(ref_year_lookup)} works with publication_year")

        for a in articles:
            art_year = a.get("publication_year", max_corpus_year)
            for ref_url in a.get("referenced_works", []):
                ry = ref_year_lookup.get(ref_url)
                if ry and 1900 <= ry <= max_corpus_year:
                    ref_years.append((ry, art_year))
    except Exception as e:
        logger.debug(f"Cache scan failed: {e}")

    return ref_years


def compute_citation_halflife(articles: list[dict]) -> dict:
    """Calcula half-life de citas y Price Index del corpus.

    Half-life = mediana de la edad de las referencias citadas (art_year - ref_year).
    Price Index = % de referencias con edad <= 5 anos al momento de publicacion
    del articulo citante (0 <= art_year - ref_year <= 5).

    Prioriza references_resolved (OpenAlex Phase 1i + S2/Scopus/WoS externos)
    y usa referenced_works_detailed (Crossref) como fallback por articulo.

    Returns:
        dict con 'halflife', 'median_ref_year', 'price_index', 'distribution'.
    """
    corpus_years = [a.get("publication_year", 0) for a in articles
                    if a.get("publication_year")]
    if not corpus_years:
        return {}

    max_corpus_year = max(corpus_years)

    # Extraer anos de referencias (prioridad por articulo: references_resolved -> referenced_works_detailed)
    ref_years = []

    for a in articles:
        art_year = a.get("publication_year", max_corpus_year)
        if not art_year or not isinstance(art_year, int):
            art_year = max_corpus_year

        art_ref_years = []

        # Prioridad 1: references_resolved (OpenAlex Phase 1i + S2/Scopus/WoS)
        for ref in a.get("references_resolved") or []:
            if not isinstance(ref, dict):
                continue
            ry = ref.get("year")
            if ry and isinstance(ry, (int, str)):
                try:
                    ry_int = int(ry)
                    if 1900 <= ry_int <= max_corpus_year:
                        art_ref_years.append((ry_int, art_year))
                except (ValueError, TypeError):
                    pass

        # Prioridad 2 (fallback si el articulo no tiene references_resolved): referenced_works_detailed (Crossref)
        if not art_ref_years:
            for ref in a.get("referenced_works_detailed") or []:
                if not isinstance(ref, dict):
                    continue
                ry = ref.get("year")
                if ry and isinstance(ry, (int, str)):
                    try:
                        ry_int = int(ry)
                        if 1900 <= ry_int <= max_corpus_year:
                            art_ref_years.append((ry_int, art_year))
                    except (ValueError, TypeError):
                        pass

        ref_years.extend(art_ref_years)

    # Fallback global 3: Crossref cache — lookup by DOI hash
    if not ref_years:
        ref_years = _extract_ref_years_from_crossref_cache(articles, max_corpus_year)

    # Fallback global 4: OpenAlex cache — scan resolved works for publication_year
    if not ref_years:
        ref_years = _extract_ref_years_from_cache(articles, max_corpus_year)

    if not ref_years:
        logger.warning("No reference year data for half-life calculation")
        return {}

    # Ages at time of citation
    ages = [art_y - ref_y for ref_y, art_y in ref_years if art_y >= ref_y]
    if not ages:
        return {}

    halflife = float(np.median(ages))
    mean_age = float(np.mean(ages))

    # Reference year distribution
    ref_year_dist = Counter(ry for ry, _ in ref_years)

    # Price Index (De Solla Price, 1970): % of references aged 0-5 years at citation time
    recent = sum(1 for age in ages if 0 <= age <= 5)
    price_index = round(recent / len(ages) * 100, 1) if ages else 0.0

    logger.info(
        f"Citation half-life: {halflife:.1f} years "
        f"(mean={mean_age:.1f}, Price Index={price_index}%, n_refs={len(ref_years)})"
    )

    return {
        "halflife": round(halflife, 1),
        "mean_age": round(mean_age, 1),
        "median_ref_year": int(np.median([ry for ry, _ in ref_years])),
        "price_index": price_index,
        "total_references": len(ref_years),
        "distribution": dict(sorted(ref_year_dist.items())),
        "max_corpus_year": max_corpus_year,
    }


def plot_citation_halflife_static(
    hl_data: dict,
    output_dir: Path = FIGURES_DIR,
) -> Optional[Path]:
    """Genera grafico de citation half-life (matplotlib)."""
    dist = hl_data.get("distribution", {})
    if not dist:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Distribution of reference years
    ax = axes[0]
    years_sorted = sorted(dist.keys())
    counts = [dist[y] for y in years_sorted]
    ax.bar(years_sorted, counts, color="#4e79a7", alpha=0.7, width=0.8)
    median_ry = hl_data.get("median_ref_year", 0)
    if median_ry:
        ax.axvline(median_ry, color="red", linestyle="--", linewidth=1.5,
                   label=f"Median ref year = {median_ry}")
    ax.set_xlabel("Reference Publication Year")
    ax.set_ylabel("Number of References")
    ax.set_title("Cited References Year Distribution", fontweight="bold")
    ax.legend(fontsize=9)

    # Age distribution
    ax = axes[1]
    max_year = hl_data.get("max_corpus_year", 2025)
    ages = []
    for y, cnt in dist.items():
        age = max_year - y
        ages.extend([age] * cnt)

    if ages:
        bins = range(0, max(ages) + 2)
        ax.hist(ages, bins=bins, color="#e15759", alpha=0.7, edgecolor="white")
        halflife = hl_data.get("halflife", 0)
        ax.axvline(halflife, color="blue", linestyle="--", linewidth=1.5,
                   label=f"Half-life = {halflife:.1f} years")
        ax.set_xlabel("Reference Age (years)")
        ax.set_ylabel("Number of References")
        ax.set_title("Citation Age Distribution (Half-life)", fontweight="bold")
        ax.legend(fontsize=9)

    plt.tight_layout()
    out_path = output_dir / "citation_halflife.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info(f"Citation half-life plot saved: {out_path}")
    return out_path


# ===========================================================================
# 6. CAGR (Compound Annual Growth Rate)
# ===========================================================================

def compute_cagr(articles: list[dict]) -> dict:
    """Calcula la tasa de crecimiento anual compuesta (CAGR) del campo.

    CAGR = (V_final / V_initial)^(1/n) - 1

    Tambien calcula doubling time y crecimiento anual simple.

    Returns:
        dict con 'cagr', 'doubling_time', 'annual_counts', 'growth_rates'.
    """
    year_counts: Counter = Counter()
    for a in articles:
        y = a.get("publication_year", 0)
        if y:
            year_counts[y] += 1

    if not year_counts:
        return {}

    years = sorted(year_counts.keys())
    counts = [year_counts[y] for y in years]

    if len(years) < 2:
        return {"cagr": 0, "annual_counts": dict(zip(years, counts))}

    # CAGR
    v_initial = counts[0]
    v_final = counts[-1]
    n_years = years[-1] - years[0]

    if v_initial > 0 and n_years > 0:
        cagr = (v_final / v_initial) ** (1 / n_years) - 1
    else:
        cagr = 0

    # Doubling time (Rule of 70)
    doubling_time = round(70 / (cagr * 100), 1) if cagr > 0 else float("inf")

    # Annual growth rates
    growth_rates = {}
    for i in range(1, len(years)):
        prev = counts[i - 1]
        curr = counts[i]
        if prev > 0:
            rate = (curr - prev) / prev
        else:
            rate = 0
        growth_rates[years[i]] = round(rate, 3)

    # Moving average (3-year)
    ma3 = {}
    for i in range(len(years)):
        window = counts[max(0, i - 1):i + 2]
        ma3[years[i]] = round(sum(window) / len(window), 1)

    logger.info(
        f"CAGR: {cagr:.1%} "
        f"(doubling time: {doubling_time} years, "
        f"period: {years[0]}-{years[-1]})"
    )

    return {
        "cagr": round(cagr, 4),
        "cagr_pct": round(cagr * 100, 1),
        "doubling_time": doubling_time,
        "period": (years[0], years[-1]),
        "annual_counts": dict(zip(years, counts)),
        "growth_rates": growth_rates,
        "moving_average_3y": ma3,
        "total_articles": sum(counts),
    }


def plot_cagr_static(
    cagr_data: dict,
    output_dir: Path = FIGURES_DIR,
) -> Optional[Path]:
    """Genera grafico CAGR con tendencia (matplotlib)."""
    annual = cagr_data.get("annual_counts", {})
    if not annual:
        return None

    years = sorted(annual.keys())
    counts = [annual[y] for y in years]
    ma3 = cagr_data.get("moving_average_3y", {})
    ma3_vals = [ma3.get(y, 0) for y in years]
    growth = cagr_data.get("growth_rates", {})

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Publication trend + MA3 + exponential fit
    ax = axes[0]
    ax.bar(years, counts, color="#4e79a7", alpha=0.7, label="Documents/year")
    ax.plot(years, ma3_vals, color="#e15759", linewidth=2, marker="o",
            markersize=4, label="3-year moving average")

    cagr_val = cagr_data.get("cagr_pct", 0)
    doubling = cagr_data.get("doubling_time", "inf")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Documents")
    ax.set_title(
        f"Annual Production (CAGR = {cagr_val:.1f}%, "
        f"doubling = {doubling} years)",
        fontweight="bold",
    )
    ax.legend(fontsize=9)

    # Growth rate
    ax = axes[1]
    gr_years = sorted(growth.keys())
    gr_vals = [growth[y] * 100 for y in gr_years]
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in gr_vals]
    ax.bar(gr_years, gr_vals, color=colors, alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(cagr_val, color="#ff7f0e", linestyle="--", linewidth=1.5,
               label=f"CAGR = {cagr_val:.1f}%")
    ax.set_xlabel("Year")
    ax.set_ylabel("Growth Rate (%)")
    ax.set_title("Annual Growth Rate", fontweight="bold")
    ax.legend(fontsize=9)

    plt.tight_layout()
    out_path = output_dir / "cagr_growth.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info(f"CAGR plot saved: {out_path}")
    return out_path


# ===========================================================================
# REPORTE MARKDOWN
# ===========================================================================

def generate_supplementary_report(
    cs_data: dict,
    trend_data: dict,
    cum_data: dict,
    dominance_data: list,
    halflife_data: dict,
    cagr_data: dict,
    output_dir: Path = REPORTS_DIR,
) -> Path:
    """Genera reporte Markdown de indicadores suplementarios."""
    lines = [
        "# Supplementary Bibliometric Indicators Report",
        "",
    ]

    # 1. Conceptual Structure
    lines.append("## 1. Conceptual Structure Map")
    lines.append("")
    if cs_data.get("coords"):
        method = cs_data.get("method", "ca").upper()
        explained = cs_data.get("explained_var", [])
        lines.append(f"- **Method:** {method}")
        lines.append(f"- **Keywords mapped:** {len(cs_data['coords'])}")
        if explained:
            lines.append(f"- **Dim 1 explained:** {explained[0]:.1f}%")
            if len(explained) > 1:
                lines.append(f"- **Dim 2 explained:** {explained[1]:.1f}%")

        clusters = cs_data.get("clusters", {})
        freqs = cs_data.get("frequencies", {})
        if clusters:
            n_cl = len(set(clusters.values()))
            lines.append(f"- **Clusters:** {n_cl}")
            lines.append("")
            for cid in sorted(set(clusters.values())):
                kws = [kw for kw, c in clusters.items() if c == cid]
                kws.sort(key=lambda k: freqs.get(k, 0), reverse=True)
                top = ", ".join(kws[:5])
                lines.append(f"  - **Cluster {cid + 1}** ({len(kws)} kw): {top}")
        lines.append("")
    else:
        lines.append("*Not enough data.*\n")

    # 2. Trend Topics
    lines.append("## 2. Trend Topics")
    lines.append("")
    td = trend_data.get("median_year_topics", [])
    if td:
        lines.append(f"- **Método:** {trend_data.get('metodo', '')}")
        lines.append(f"- **Palabras:** {len(td)}")
        lines.append("")
        lines.append("| Keyword | Freq | Q1 | Median year | Q3 |")
        lines.append("|---------|------|----|-------------|----|")
        for d in sorted(td, key=lambda d: (-d["year_med"], -d["freq"], d["keyword"]))[:40]:
            lines.append(f"| {d['keyword']} | {d['freq']} | {d['year_q1']} | {d['year_med']} | {d['year_q3']} |")
        lines.append("")
    else:
        lines.append("*Not enough data.*\n")

    # 3. Cumulative Production
    lines.append("## 3. Cumulative Production")
    lines.append("")
    if cum_data.get("authors"):
        lines.append("### Top Authors")
        lines.append("")
        years = cum_data["years"]
        for name, series in list(cum_data["authors"].items())[:10]:
            final = series.get(years[-1], 0) if years else 0
            lines.append(f"- **{name}**: {final} documents by {years[-1] if years else '?'}")
        lines.append("")
    if cum_data.get("sources"):
        lines.append("### Top Sources")
        lines.append("")
        years = cum_data["years"]
        for name, series in list(cum_data["sources"].items())[:10]:
            final = series.get(years[-1], 0) if years else 0
            lines.append(f"- **{name}**: {final} documents by {years[-1] if years else '?'}")
        lines.append("")

    # 4. Author Dominance
    lines.append("## 4. Author Dominance Factor")
    lines.append("")
    if dominance_data:
        lines.append("| Author | Total | First Auth | DF |")
        lines.append("|--------|-------|------------|-----|")
        for d in dominance_data[:15]:
            lines.append(
                f"| {d['author'][:30]} | {d['total']} | "
                f"{d['first_authored']} | {d['dominance_factor']:.2f} |"
            )
        lines.append("")
    else:
        lines.append("*Not enough data.*\n")

    # 5. Citation Half-life
    lines.append("## 5. Citation Half-life")
    lines.append("")
    if halflife_data:
        lines.append(f"- **Half-life:** {halflife_data.get('halflife', 'N/A')} years")
        lines.append(f"- **Mean reference age:** {halflife_data.get('mean_age', 'N/A')} years")
        lines.append(f"- **Median reference year:** {halflife_data.get('median_ref_year', 'N/A')}")
        lines.append(f"- **Price Index (refs <5 years):** {halflife_data.get('price_index', 'N/A')}%")
        lines.append(f"- **Total references analyzed:** {halflife_data.get('total_references', 0)}")
        lines.append("")
    else:
        lines.append("*No reference year data available (run Crossref enrichment first).*\n")

    # 6. CAGR
    lines.append("## 6. Growth Rate (CAGR)")
    lines.append("")
    if cagr_data:
        lines.append(f"- **CAGR:** {cagr_data.get('cagr_pct', 0):.1f}%")
        lines.append(f"- **Doubling time:** {cagr_data.get('doubling_time', 'N/A')} years")
        period = cagr_data.get("period", (0, 0))
        lines.append(f"- **Period:** {period[0]}-{period[1]}")
        lines.append(f"- **Total articles:** {cagr_data.get('total_articles', 0)}")
        lines.append("")

        annual = cagr_data.get("annual_counts", {})
        if annual:
            lines.append("| Year | Documents | Growth Rate |")
            lines.append("|------|-----------|-------------|")
            growth = cagr_data.get("growth_rates", {})
            for y in sorted(annual.keys()):
                gr = growth.get(y)
                gr_str = f"{gr * 100:.0f}%" if gr is not None else "—"
                lines.append(f"| {y} | {annual[y]} | {gr_str} |")
            lines.append("")
    else:
        lines.append("*Not enough data.*\n")

    out_path = output_dir / "supplementary_indicators_report.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Supplementary indicators report saved: {out_path}")
    return out_path


# ===========================================================================
# ORQUESTADOR PRINCIPAL
# ===========================================================================

def run_phase2e(articles: list[dict]) -> dict:
    """Ejecuta la Fase 2e: Indicadores bibliometricos suplementarios.

    Returns:
        dict con resultados de todos los sub-analisis.
    """
    logger.info("=" * 60)
    logger.info("=== FASE 2e: Indicadores Bibliometricos Suplementarios ===")
    logger.info("=" * 60)

    results = {}

    # 1. Conceptual Structure Map
    logger.info("--- 2f.1: Conceptual Structure Map ---")
    cs_data = compute_conceptual_structure(articles, top_n_keywords=50)
    results["conceptual_structure"] = cs_data
    if cs_data.get("coords"):
        plot_conceptual_structure_static(cs_data)
        plot_conceptual_structure_plotly(cs_data)

    # 2. Trend Topics
    logger.info("--- 2f.2: Trend Topics ---")
    trend_data = compute_trend_topics(articles, top_n_per_year=5, min_freq=2)
    results["trend_topics"] = trend_data
    if trend_data.get("median_year_topics"):
        plot_trend_topics_static(trend_data)
        plot_trend_topics_plotly(trend_data)

    # 3. Cumulative Production
    logger.info("--- 2f.3: Cumulative Production ---")
    cum_data = compute_cumulative_production(articles)
    results["cumulative_production"] = cum_data
    if cum_data.get("authors") or cum_data.get("sources"):
        plot_cumulative_production_static(cum_data)
        plot_cumulative_production_plotly(cum_data)

    # 4. Author Dominance Factor
    logger.info("--- 2f.4: Author Dominance Factor ---")
    dominance_data = compute_author_dominance(articles)
    results["author_dominance"] = dominance_data
    if dominance_data:
        plot_author_dominance_static(dominance_data)
        plot_author_dominance_plotly(dominance_data)

    # 5. Citation Half-life
    logger.info("--- 2f.5: Citation Half-life ---")
    halflife_data = compute_citation_halflife(articles)
    results["citation_halflife"] = halflife_data
    if halflife_data:
        plot_citation_halflife_static(halflife_data)

    # 6. CAGR
    logger.info("--- 2f.6: CAGR (Growth Rate) ---")
    cagr_data = compute_cagr(articles)
    results["cagr"] = cagr_data
    if cagr_data.get("annual_counts"):
        plot_cagr_static(cagr_data)

    # Reporte
    generate_supplementary_report(
        cs_data, trend_data, cum_data,
        dominance_data, halflife_data, cagr_data,
    )

    logger.info("=" * 60)
    logger.info("Fase 2e completada.")
    logger.info("=" * 60)

    return results

run_phase2f = run_phase2e
