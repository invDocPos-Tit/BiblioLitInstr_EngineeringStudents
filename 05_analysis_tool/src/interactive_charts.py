"""Interactive charts: bar charts, heatmaps, Sankey, reference graph,
citation trajectory, citation matrix, and RPYS spectroscopy."""
from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from config import PLOTLY_DIR, OPENALEX_CACHE_DIR
from src.utils import DiskCache

logger = logging.getLogger("bibliometrico.charts")

# Top N items for bar charts
CHART_TOP_N = 15


def _save_plotly_figure(fig, output_dir: Path, name: str) -> None:
    """Save Plotly figure as HTML and optionally PNG."""
    from src.advanced_visualizations import _save_plotly_figure as _save
    _save(fig, output_dir, name)


def _extract_dimension_values(article: dict, dimension: str) -> list[str]:
    """Extract values for a given dimension from an article.

    Supported dimensions: Authors, Countries, Institutions, Journals,
    Author Keywords, Keywords Plus, Languages.
    """
    openalex = article.get("openalex", {})

    if dimension == "Authors":
        return article.get("authors", [])
    elif dimension == "Countries":
        from src.modelo_corpus import article_countries
        return article_countries(article)          # definicion unica del modelo del corpus
    elif dimension == "Institutions":
        insts = set()
        for auth in openalex.get("authorships", []):
            for inst in auth.get("institutions", []):
                name = inst.get("name", "")
                if name:
                    insts.add(name)
        return list(insts)
    elif dimension == "Journals":
        source = article.get("publication_title", "")
        return [source] if source and source != "nan" else []
    elif dimension == "Author Keywords":
        return [kw.lower().strip() for kw in article.get("manual_tags", []) if kw.strip()]
    elif dimension == "Keywords Plus":
        return [kw.lower().strip() for kw in article.get("automatic_tags", []) if kw.strip()]
    elif dimension == "Languages":
        lang = article.get("language", "")
        return [lang] if lang and lang != "nan" else []
    return []


# ===========================================================================
# BAR CHARTS (Feature v)
# ===========================================================================

def _create_horizontal_bar(names: list[str], values: list[float],
                           title: str, x_label: str, output_dir: Path,
                           filename: str, color: str = "#4e79a7") -> None:
    """Helper to create a horizontal bar chart."""
    import plotly.graph_objects as go

    fig = go.Figure(go.Bar(
        y=names[::-1], x=values[::-1], orientation="h",
        marker_color=color,
        hovertemplate="%{y}: %{x}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title=x_label,
        width=1200, height=max(500, len(names) * 30 + 150),
        paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=250),
    )
    _save_plotly_figure(fig, output_dir, filename)


def create_bar_docs_per_year(articles: list[dict], output_dir: Path) -> None:
    """Bar chart: number of documents per year."""
    year_counts = Counter(
        a["publication_year"] for a in articles if a.get("publication_year")
    )
    if not year_counts:
        return
    years = sorted(year_counts.keys())
    counts = [year_counts[y] for y in years]

    import plotly.graph_objects as go
    fig = go.Figure(go.Bar(
        x=[str(y) for y in years], y=counts,
        marker_color="#4e79a7",
        hovertemplate="Year %{x}: %{y} documents<extra></extra>",
    ))
    fig.update_layout(
        title="Documents per Year",
        xaxis_title="Year", yaxis_title="Documents",
        width=1200, height=500, paper_bgcolor="white",
    )
    _save_plotly_figure(fig, output_dir, "bar_docs_per_year")


def create_bar_citations_per_year(articles: list[dict], output_dir: Path) -> None:
    """Bar chart: total citations per publication year."""
    year_cites: dict[int, int] = {}
    for a in articles:
        yr = a.get("publication_year")
        if yr:
            year_cites[yr] = year_cites.get(yr, 0) + a.get("cited_by_count", 0)
    if not year_cites:
        return
    years = sorted(year_cites.keys())
    cites = [year_cites[y] for y in years]

    import plotly.graph_objects as go
    fig = go.Figure(go.Bar(
        x=[str(y) for y in years], y=cites,
        marker_color="#f28e2b",
        hovertemplate="Year %{x}: %{y} citations<extra></extra>",
    ))
    fig.update_layout(
        title="Total Citations per Publication Year",
        xaxis_title="Year", yaxis_title="Total Citations",
        width=1200, height=500, paper_bgcolor="white",
    )
    _save_plotly_figure(fig, output_dir, "bar_citations_per_year")


def create_bar_past_citations_per_year(articles: list[dict], output_dir: Path) -> None:
    """Bar chart: average citations per document per publication year."""
    year_data: dict[int, list[int]] = {}
    for a in articles:
        yr = a.get("publication_year")
        if yr:
            year_data.setdefault(yr, []).append(a.get("cited_by_count", 0))
    if not year_data:
        return
    years = sorted(year_data.keys())
    avgs = [round(np.mean(year_data[y]), 2) for y in years]

    import plotly.graph_objects as go
    fig = go.Figure(go.Bar(
        x=[str(y) for y in years], y=avgs,
        marker_color="#e15759",
        hovertemplate="Year %{x}: %{y:.1f} avg citations<extra></extra>",
    ))
    fig.update_layout(
        title="Average Citations per Document by Year",
        xaxis_title="Year", yaxis_title="Avg Citations",
        width=1200, height=500, paper_bgcolor="white",
    )
    _save_plotly_figure(fig, output_dir, "bar_avg_citations_per_year")


def create_bar_lotka(articles: list[dict], output_dir: Path) -> None:
    """Bar chart: Lotka's productivity distribution."""
    author_counts = Counter()
    for a in articles:
        for auth in a.get("authors", []):
            author_counts[auth] += 1

    prod_dist = Counter(author_counts.values())
    if not prod_dist:
        return
    n_pubs = sorted(prod_dist.keys())
    n_authors = [prod_dist[n] for n in n_pubs]

    import plotly.graph_objects as go
    fig = go.Figure(go.Bar(
        x=[str(n) for n in n_pubs], y=n_authors,
        marker_color="#76b7b2",
        hovertemplate="%{x} publications: %{y} authors<extra></extra>",
    ))
    fig.update_layout(
        title="Lotka's Law: Author Productivity Distribution",
        xaxis_title="Number of Publications", yaxis_title="Number of Authors",
        width=1200, height=500, paper_bgcolor="white",
    )
    _save_plotly_figure(fig, output_dir, "bar_lotka")


def create_bar_bradford(articles: list[dict], output_dir: Path) -> None:
    """Bar chart: Bradford zones (top journals by document count)."""
    journal_counts = Counter()
    for a in articles:
        j = a.get("publication_title", "")
        if j and j != "nan":
            journal_counts[j] += 1
    if not journal_counts:
        return

    sorted_j = journal_counts.most_common(CHART_TOP_N)
    names = [j[:50] for j, _ in sorted_j]
    counts = [c for _, c in sorted_j]
    _create_horizontal_bar(names, counts, "Bradford's Law: Top Journals",
                           "Documents", output_dir, "bar_bradford", "#59a14f")


def create_bar_sources(articles: list[dict], output_dir: Path,
                       metric: str = "docs") -> None:
    """Bar chart: top sources by docs or citations."""
    source_data: dict[str, dict] = {}
    for a in articles:
        s = a.get("publication_title", "")
        if not s or s == "nan":
            continue
        if s not in source_data:
            source_data[s] = {"docs": 0, "citations": 0}
        source_data[s]["docs"] += 1
        source_data[s]["citations"] += a.get("cited_by_count", 0)
    if not source_data:
        return

    sorted_s = sorted(source_data.items(), key=lambda x: x[1][metric], reverse=True)[:CHART_TOP_N]
    names = [s[:50] for s, _ in sorted_s]
    values = [d[metric] for _, d in sorted_s]
    label = "Documents" if metric == "docs" else "Citations"
    _create_horizontal_bar(names, values, f"Top Sources by {label}",
                           label, output_dir, f"bar_sources_{metric}", "#edc948")


def create_bar_authors(articles: list[dict], output_dir: Path,
                       metric: str = "docs") -> None:
    """Bar chart: top authors by docs, citations, or h_index."""
    author_data: dict[str, dict] = {}
    for a in articles:
        tc = a.get("cited_by_count", 0)
        for auth in a.get("authors", []):
            if auth not in author_data:
                author_data[auth] = {"docs": 0, "citations": 0, "cite_list": []}
            author_data[auth]["docs"] += 1
            author_data[auth]["citations"] += tc
            author_data[auth]["cite_list"].append(tc)
    if not author_data:
        return

    if metric == "h_index":
        for auth, d in author_data.items():
            sorted_c = sorted(d["cite_list"], reverse=True)
            h = 0
            for i, c in enumerate(sorted_c, 1):
                if c >= i:
                    h = i
                else:
                    break
            d["h_index"] = h

    sorted_a = sorted(author_data.items(), key=lambda x: x[1].get(metric, 0), reverse=True)[:CHART_TOP_N]
    names = [a[:40] for a, _ in sorted_a]
    values = [d.get(metric, 0) for _, d in sorted_a]
    label = {"docs": "Documents", "citations": "Citations", "h_index": "H-index"}.get(metric, metric)
    _create_horizontal_bar(names, values, f"Top Authors by {label}",
                           label, output_dir, f"bar_authors_{metric}", "#b07aa1")


def create_bar_institutions(articles: list[dict], output_dir: Path,
                            metric: str = "docs") -> None:
    """Bar chart: top institutions by docs or citations."""
    inst_data: dict[str, dict] = {}
    for a in articles:
        tc = a.get("cited_by_count", 0)
        for val in _extract_dimension_values(a, "Institutions"):
            if val not in inst_data:
                inst_data[val] = {"docs": 0, "citations": 0}
            inst_data[val]["docs"] += 1
            inst_data[val]["citations"] += tc
    if not inst_data:
        return

    sorted_i = sorted(inst_data.items(), key=lambda x: x[1][metric], reverse=True)[:CHART_TOP_N]
    names = [i[:50] for i, _ in sorted_i]
    values = [d[metric] for _, d in sorted_i]
    label = "Documents" if metric == "docs" else "Citations"
    _create_horizontal_bar(names, values, f"Top Institutions by {label}",
                           label, output_dir, f"bar_institutions_{metric}", "#ff9da7")


def create_bar_countries(articles: list[dict], output_dir: Path,
                         metric: str = "docs") -> None:
    """Bar chart: top countries by docs or citations."""
    country_data: dict[str, dict] = {}
    for a in articles:
        tc = a.get("cited_by_count", 0)
        for val in _extract_dimension_values(a, "Countries"):
            if val not in country_data:
                country_data[val] = {"docs": 0, "citations": 0}
            country_data[val]["docs"] += 1
            country_data[val]["citations"] += tc
    if not country_data:
        return

    sorted_c = sorted(country_data.items(), key=lambda x: x[1][metric], reverse=True)[:CHART_TOP_N]
    names = [c for c, _ in sorted_c]
    values = [d[metric] for _, d in sorted_c]
    label = "Documents" if metric == "docs" else "Citations"
    _create_horizontal_bar(names, values, f"Top Countries by {label}",
                           label, output_dir, f"bar_countries_{metric}", "#9c755f")


def create_bar_languages(articles: list[dict], output_dir: Path) -> None:
    """Bar chart: documents by language."""
    lang_counts = Counter()
    for a in articles:
        lang = a.get("language", "")
        if lang and lang != "nan":
            lang_counts[lang] += 1
    if not lang_counts:
        return

    sorted_l = lang_counts.most_common(CHART_TOP_N)
    names = [l for l, _ in sorted_l]
    values = [c for _, c in sorted_l]
    _create_horizontal_bar(names, values, "Documents by Language",
                           "Documents", output_dir, "bar_languages", "#bab0ac")


def create_bar_keywords(articles: list[dict], output_dir: Path,
                        keyword_type: str = "author") -> None:
    """Bar chart: top keywords (author or plus)."""
    kw_counts = Counter()
    field = "manual_tags" if keyword_type == "author" else "automatic_tags"
    for a in articles:
        for kw in a.get(field, []):
            kw_counts[kw.lower().strip()] += 1
    if not kw_counts:
        return

    sorted_kw = kw_counts.most_common(CHART_TOP_N)
    names = [kw[:40] for kw, _ in sorted_kw]
    values = [c for _, c in sorted_kw]
    label = "Author Keywords" if keyword_type == "author" else "Keywords Plus"
    _create_horizontal_bar(names, values, f"Top {label}",
                           "Frequency", output_dir, f"bar_keywords_{keyword_type}", "#86bcb6")


def run_all_bar_charts(articles: list[dict], output_dir: Path) -> None:
    """Generate all bar chart variants."""
    logger.info("Generating bar charts...")
    chart_funcs = [
        lambda: create_bar_docs_per_year(articles, output_dir),
        lambda: create_bar_citations_per_year(articles, output_dir),
        lambda: create_bar_past_citations_per_year(articles, output_dir),
        lambda: create_bar_lotka(articles, output_dir),
        lambda: create_bar_bradford(articles, output_dir),
        lambda: create_bar_sources(articles, output_dir, metric="docs"),
        lambda: create_bar_sources(articles, output_dir, metric="citations"),
        lambda: create_bar_authors(articles, output_dir, metric="docs"),
        lambda: create_bar_authors(articles, output_dir, metric="citations"),
        lambda: create_bar_authors(articles, output_dir, metric="h_index"),
        lambda: create_bar_institutions(articles, output_dir, metric="docs"),
        lambda: create_bar_institutions(articles, output_dir, metric="citations"),
        lambda: create_bar_countries(articles, output_dir, metric="docs"),
        lambda: create_bar_countries(articles, output_dir, metric="citations"),
        lambda: create_bar_languages(articles, output_dir),
        lambda: create_bar_keywords(articles, output_dir, keyword_type="author"),
        lambda: create_bar_keywords(articles, output_dir, keyword_type="plus"),
    ]
    for fn in chart_funcs:
        try:
            fn()
        except Exception as e:
            logger.error(f"Bar chart failed: {e}")


# ===========================================================================
# HEATMAP XY (Feature p)
# ===========================================================================

def create_heatmap_xy(articles: list[dict], x_key: str, y_key: str,
                      output_dir: Path, top_n: int = 20) -> None:
    """Create interactive heatmap with two dimensions and article IDs on hover.

    Args:
        articles: List of enriched articles.
        x_key: Dimension for x-axis (e.g., 'Authors', 'Countries').
        y_key: Dimension for y-axis (e.g., 'Journals', 'Keywords').
        output_dir: Output directory.
        top_n: Number of top items per axis.
    """
    import plotly.graph_objects as go

    # Count co-occurrences and track article keys
    pair_count: dict[tuple[str, str], int] = {}
    pair_articles: dict[tuple[str, str], list[str]] = {}

    for a in articles:
        x_vals = _extract_dimension_values(a, x_key)
        y_vals = _extract_dimension_values(a, y_key)
        key = a.get("key", "")
        for xv in x_vals:
            for yv in y_vals:
                pair = (xv, yv)
                pair_count[pair] = pair_count.get(pair, 0) + 1
                pair_articles.setdefault(pair, []).append(key)

    if not pair_count:
        logger.warning(f"No data for heatmap {x_key} x {y_key}")
        return

    # Get top items by frequency
    x_freq = Counter()
    y_freq = Counter()
    for (xv, yv), c in pair_count.items():
        x_freq[xv] += c
        y_freq[yv] += c

    top_x = [x for x, _ in x_freq.most_common(top_n)]
    top_y = [y for y, _ in y_freq.most_common(top_n)]

    # Build matrix
    z = []
    hover_text = []
    for yi in top_y:
        row = []
        hover_row = []
        for xi in top_x:
            val = pair_count.get((xi, yi), 0)
            row.append(val)
            arts = pair_articles.get((xi, yi), [])
            art_str = ", ".join(arts[:5])
            if len(arts) > 5:
                art_str += f"... (+{len(arts)-5})"
            hover_row.append(f"{xi} × {yi}<br>Count: {val}<br>Articles: {art_str}")
        z.append(row)
        hover_text.append(hover_row)

    fig = go.Figure(go.Heatmap(
        z=z,
        x=[str(x)[:30] for x in top_x],
        y=[str(y)[:40] for y in top_y],
        hovertext=hover_text,
        hoverinfo="text",
        colorscale="YlOrRd",
        colorbar=dict(title="Count"),
    ))
    fig.update_layout(
        title=f"Heatmap: {x_key} × {y_key}",
        xaxis=dict(title=x_key, tickangle=45),
        yaxis=dict(title=y_key),
        width=1400, height=900,
        paper_bgcolor="white",
    )

    safe_name = f"heatmap_{x_key.lower().replace(' ', '_')}_x_{y_key.lower().replace(' ', '_')}"
    _save_plotly_figure(fig, output_dir, safe_name)


# ===========================================================================
# SANKEY (Feature n)
# ===========================================================================

def create_sankey_generic(articles: list[dict], dimensions: list[str],
                          output_dir: Path, top_n: int = 12) -> None:
    """Create a Sankey diagram from any combination of dimension keys.

    Args:
        articles: List of enriched articles.
        dimensions: List of dimension names (2-4), e.g. ['Countries', 'Institutions', 'Authors'].
        output_dir: Output directory.
        top_n: Top items per dimension.
    """
    import plotly.graph_objects as go

    if len(dimensions) < 2:
        logger.warning("Sankey needs at least 2 dimensions")
        return

    # Get top items per dimension
    dim_freq: list[Counter] = []
    for dim in dimensions:
        freq = Counter()
        for a in articles:
            for v in _extract_dimension_values(a, dim):
                freq[v] += 1
        dim_freq.append(freq)

    dim_top: list[list[str]] = []
    for freq in dim_freq:
        dim_top.append([v for v, _ in freq.most_common(top_n)])

    # Build node list
    node_labels = []
    node_index = {}
    for d_idx, (dim, tops) in enumerate(zip(dimensions, dim_top)):
        for val in tops:
            key = (d_idx, val)
            node_index[key] = len(node_labels)
            node_labels.append(f"{val} ({dim})")

    # Build links between adjacent dimension pairs
    sources, targets, values = [], [], []
    for d_idx in range(len(dimensions) - 1):
        link_counts: dict[tuple[str, str], int] = {}
        set_left = set(dim_top[d_idx])
        set_right = set(dim_top[d_idx + 1])

        for a in articles:
            left_vals = [v for v in _extract_dimension_values(a, dimensions[d_idx]) if v in set_left]
            right_vals = [v for v in _extract_dimension_values(a, dimensions[d_idx + 1]) if v in set_right]
            for lv in left_vals:
                for rv in right_vals:
                    link_counts[(lv, rv)] = link_counts.get((lv, rv), 0) + 1

        for (lv, rv), count in link_counts.items():
            src = node_index.get((d_idx, lv))
            tgt = node_index.get((d_idx + 1, rv))
            if src is not None and tgt is not None:
                sources.append(src)
                targets.append(tgt)
                values.append(count)

    if not sources:
        logger.warning(f"No links for Sankey: {dimensions}")
        return

    fig = go.Figure(go.Sankey(
        node=dict(pad=20, thickness=25, label=node_labels,
                  line=dict(color="white", width=1)),
        link=dict(source=sources, target=targets, value=values),
    ))
    dim_str = " → ".join(dimensions)
    fig.update_layout(
        title=f"Sankey: {dim_str}",
        width=1400, height=800, paper_bgcolor="white",
    )
    safe_name = "sankey_" + "_".join(d.lower().replace(" ", "_") for d in dimensions)
    _save_plotly_figure(fig, output_dir, safe_name)


# ===========================================================================
# REFERENCE GRAPH (Feature w)
# ===========================================================================

def create_main_reference_graph(articles: list[dict], output_dir: Path,
                                top_n: int = 40) -> None:
    """Create an interactive network of the most cited references.

    Nodes = top references cited by corpus articles.
    Edges = references co-cited together.
    """
    import plotly.graph_objects as go

    # Count how many corpus articles cite each reference
    ref_citing = Counter()
    for a in articles:
        for ref in a.get("referenced_works", []):
            ref_citing[ref] += 1

    top_refs = [ref for ref, _ in ref_citing.most_common(top_n)]
    top_ref_set = set(top_refs)

    if len(top_refs) < 3:
        logger.warning("Not enough references for reference graph")
        return

    # Resolve labels
    cache = DiskCache(OPENALEX_CACHE_DIR)
    ref_labels = {}
    for ref_url in top_refs:
        work_id = ref_url.split("/")[-1] if "/" in ref_url else ref_url
        cached = cache.get(f"work:{work_id}")
        if cached and isinstance(cached, dict):
            authorships = cached.get("authorships", [])
            first_author = "Unknown"
            if authorships:
                first_author = authorships[0].get("author", {}).get("display_name", "Unknown")
                parts = first_author.split()
                if parts:
                    first_author = parts[-1]
            year = cached.get("publication_year", "?")
            ref_labels[ref_url] = f"{first_author} ({year})"
        else:
            ref_labels[ref_url] = work_id

    # Build co-citation edges among top refs
    ref_to_articles: dict[str, set] = {}
    for a in articles:
        refs = set(a.get("referenced_works", []))
        for ref in refs & top_ref_set:
            ref_to_articles.setdefault(ref, set()).add(a.get("key", ""))

    import networkx as nx
    G = nx.Graph()
    for ref in top_refs:
        G.add_node(ref, label=ref_labels.get(ref, ref),
                   cited_by=ref_citing[ref])

    for i in range(len(top_refs)):
        for j in range(i + 1, len(top_refs)):
            shared = ref_to_articles.get(top_refs[i], set()) & ref_to_articles.get(top_refs[j], set())
            if shared:
                G.add_edge(top_refs[i], top_refs[j], weight=len(shared))

    if G.number_of_edges() == 0:
        logger.info("No co-citation edges for reference graph")
        return

    # Layout
    pos = nx.spring_layout(G, k=2.0 / np.sqrt(max(G.number_of_nodes(), 1)),
                           seed=42, iterations=100, weight="weight")

    # Edges
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=0.8, color="rgba(180,180,180,0.5)"),
        hoverinfo="none", showlegend=False,
    )

    # Nodes
    max_cited = max(ref_citing[r] for r in top_refs)
    node_x, node_y, node_text, node_hover, node_size = [], [], [], [], []
    for ref in top_refs:
        if ref not in pos:
            continue
        x, y = pos[ref]
        node_x.append(x)
        node_y.append(y)
        label = ref_labels.get(ref, ref)
        cited = ref_citing[ref]
        node_text.append(label)
        node_hover.append(f"<b>{label}</b><br>Cited by {cited} corpus articles")
        node_size.append(10 + 30 * np.sqrt(cited / max(max_cited, 1)))

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        text=node_text, textposition="top center", textfont=dict(size=8),
        hovertext=node_hover, hoverinfo="text",
        marker=dict(size=node_size, color="#e15759",
                    line=dict(width=1, color="white")),
        showlegend=False,
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=f"Main Reference Graph (Top {len(top_refs)} cited references)",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        width=1400, height=1000, paper_bgcolor="white", plot_bgcolor="white",
        hovermode="closest",
    )
    _save_plotly_figure(fig, output_dir, "reference_graph")


# ===========================================================================
# CITATION TRAJECTORY (Feature x)
# ===========================================================================

def create_citation_trajectory(articles: list[dict], output_dir: Path,
                               top_n: int = 15) -> None:
    """Create annual citation line chart per top reference.

    Uses counts_by_year from OpenAlex cache if available.
    """
    import plotly.graph_objects as go

    # Count citations from corpus per reference per year
    ref_year_count: dict[str, dict[int, int]] = {}
    ref_total = Counter()

    for a in articles:
        yr = a.get("publication_year")
        if not yr:
            continue
        for ref in a.get("referenced_works", []):
            ref_total[ref] += 1
            ref_year_count.setdefault(ref, {})
            ref_year_count[ref][yr] = ref_year_count[ref].get(yr, 0) + 1

    top_refs = [ref for ref, _ in ref_total.most_common(top_n)]
    if not top_refs:
        return

    # Resolve labels
    cache = DiskCache(OPENALEX_CACHE_DIR)
    ref_labels = {}
    for ref_url in top_refs:
        work_id = ref_url.split("/")[-1] if "/" in ref_url else ref_url
        cached = cache.get(f"work:{work_id}")
        if cached and isinstance(cached, dict):
            authorships = cached.get("authorships", [])
            first_author = "Unknown"
            if authorships:
                first_author = authorships[0].get("author", {}).get("display_name", "Unknown")
                parts = first_author.split()
                if parts:
                    first_author = parts[-1]
            year = cached.get("publication_year", "?")
            ref_labels[ref_url] = f"{first_author} ({year})"
        else:
            ref_labels[ref_url] = work_id[:20]

    # Build figure
    all_years = set()
    for ref in top_refs:
        if ref in ref_year_count:
            all_years.update(ref_year_count[ref].keys())
    if not all_years:
        return

    years_range = sorted(all_years)

    fig = go.Figure()
    for ref in top_refs:
        yc = ref_year_count.get(ref, {})
        counts = [yc.get(y, 0) for y in years_range]
        fig.add_trace(go.Scatter(
            x=[str(y) for y in years_range], y=counts,
            mode="lines+markers", name=ref_labels.get(ref, ref)[:35],
            hovertemplate="%{x}: %{y} citations<extra>%{fullData.name}</extra>",
        ))

    fig.update_layout(
        title=f"Citation Trajectory (Top {len(top_refs)} references)",
        xaxis_title="Year", yaxis_title="Times Cited by Corpus",
        width=1400, height=700, paper_bgcolor="white",
        legend=dict(font=dict(size=9)),
    )
    _save_plotly_figure(fig, output_dir, "citation_trajectory")


# ===========================================================================
# CITATION MATRIX (Feature y)
# ===========================================================================

def create_citation_matrix(articles: list[dict], output_dir: Path,
                           top_refs_n: int = 30) -> None:
    """Create a binary heatmap: references (rows) x citing articles (columns).

    Value = 1 if article cites the reference, 0 otherwise.
    """
    import plotly.graph_objects as go

    # Count and select top references
    ref_citing = Counter()
    for a in articles:
        for ref in a.get("referenced_works", []):
            ref_citing[ref] += 1

    top_refs = [ref for ref, _ in ref_citing.most_common(top_refs_n)]
    top_ref_set = set(top_refs)

    if len(top_refs) < 3:
        logger.warning("Not enough references for citation matrix")
        return

    # Resolve reference labels
    cache = DiskCache(OPENALEX_CACHE_DIR)
    ref_labels = {}
    for ref_url in top_refs:
        work_id = ref_url.split("/")[-1] if "/" in ref_url else ref_url
        cached = cache.get(f"work:{work_id}")
        if cached and isinstance(cached, dict):
            authorships = cached.get("authorships", [])
            first_author = "Unknown"
            if authorships:
                first_author = authorships[0].get("author", {}).get("display_name", "Unknown")
                parts = first_author.split()
                if parts:
                    first_author = parts[-1]
            year = cached.get("publication_year", "?")
            ref_labels[ref_url] = f"{first_author} ({year})"
        else:
            ref_labels[ref_url] = work_id[:20]

    # Article labels
    article_labels = []
    article_refs_list = []
    for a in articles:
        refs = set(a.get("referenced_works", []))
        if refs & top_ref_set:
            first_author = a["authors"][0].split(",")[0] if a.get("authors") else "?"
            yr = a.get("publication_year", "?")
            article_labels.append(f"{first_author} ({yr})")
            article_refs_list.append(refs)

    if not article_labels:
        return

    # Build matrix
    z = []
    hover_text = []
    for ref in top_refs:
        row = []
        h_row = []
        for i, refs in enumerate(article_refs_list):
            val = 1 if ref in refs else 0
            row.append(val)
            h_row.append(f"{ref_labels.get(ref, '?')} → {article_labels[i]}: {'Yes' if val else 'No'}")
        z.append(row)
        hover_text.append(h_row)

    fig = go.Figure(go.Heatmap(
        z=z,
        x=article_labels,
        y=[ref_labels.get(r, r)[:35] for r in top_refs],
        hovertext=hover_text,
        hoverinfo="text",
        colorscale=[[0, "white"], [1, "#e15759"]],
        showscale=False,
    ))
    fig.update_layout(
        title=f"Citation Matrix (Top {len(top_refs)} references × {len(article_labels)} articles)",
        xaxis=dict(title="Citing Articles", tickangle=45, tickfont=dict(size=7)),
        yaxis=dict(title="References", tickfont=dict(size=8)),
        width=max(1200, len(article_labels) * 20 + 400),
        height=max(700, len(top_refs) * 22 + 200),
        paper_bgcolor="white",
    )
    _save_plotly_figure(fig, output_dir, "citation_matrix")


# ===========================================================================
# RPYS - Reference Publication Year Spectroscopy (Feature z)
# ===========================================================================

def create_rpys_chart(articles: list[dict], output_dir: Path) -> None:
    """Figura RPYS de la Fase 4: grafica la serie que calcula la Fase 2g, sin recalcular.

    Antes esta funcion tenia su propio RPYS (obras distintas, mediana global, suavizado
    gaussiano), distinto del de la 2g: la figura y el reporte podian no coincidir. Ahora hay
    una sola implementacion (src/reference_spectroscopy.compute_enhanced_rpys).
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from config import SPECTROSCOPY_DIR
    from src.utils import load_json

    data = load_json(SPECTROSCOPY_DIR / "spectroscopy_data.json") or {}
    rpys = data.get("rpys") or {}
    series = rpys.get("series") or []
    if not series:
        logger.warning(
            "RPYS: no hay serie en output/spectroscopy/spectroscopy_data.json "
            "(ejecute la Fase 2g con la version actual); se omite la figura rpys.html."
        )
        return

    years = [str(r["year"]) for r in series]
    inc = [bool(r.get("ventana_incompleta")) for r in series]
    unidad = str(rpys.get("unidad", "citas")).split(" (")[0]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=(f"{unidad.capitalize()} por año de publicación de la referencia",
                                        "Desviación respecto de la mediana de 5 años (t−2 … t+2)"))
    fig.add_trace(go.Bar(x=years, y=[r["citas"] for r in series], name=unidad.capitalize(),
                         marker_color=["#bab0ac" if i else "#4e79a7" for i in inc],
                         hovertemplate="Año %{x}: %{y}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=years, y=[r["mediana_5"] for r in series], mode="lines",
                             name="Mediana 5 años", line=dict(color="black", width=1.5, dash="dot")),
                  row=1, col=1)
    fig.add_trace(go.Bar(x=years, y=[r["desviacion"] for r in series], name="Desviación",
                         marker_color=["#bab0ac" if i else ("#e15759" if r["desviacion"] > 0 else "#76b7b2")
                                       for r, i in zip(series, inc)],
                         hovertemplate="Año %{x}: desviación %{y:.0f}<extra></extra>"), row=2, col=1)
    fig.update_layout(title="Reference Publication Year Spectroscopy (RPYS)",
                      width=1600, height=900, paper_bgcolor="white", showlegend=True)
    fig.update_xaxes(title_text="Año de publicación de la referencia", row=2, col=1)
    fig.update_yaxes(title_text=unidad.capitalize(), row=1, col=1)
    fig.update_yaxes(title_text="Desviación", row=2, col=1)
    _save_plotly_figure(fig, output_dir, "rpys")


# ===========================================================================
# RUNNER
# ===========================================================================

def run_interactive_charts(articles: list[dict]) -> None:
    """Run all interactive chart generation."""
    output_dir = PLOTLY_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Interactive Charts ===")

    # Bar charts
    try:
        run_all_bar_charts(articles, output_dir)
    except Exception as e:
        logger.error(f"Bar charts failed: {e}")

    # Heatmaps
    heatmap_pairs = [
        ("Countries", "Author Keywords"),
        ("Authors", "Journals"),
        ("Countries", "Institutions"),
    ]
    for x_key, y_key in heatmap_pairs:
        try:
            create_heatmap_xy(articles, x_key, y_key, output_dir)
        except Exception as e:
            logger.error(f"Heatmap {x_key} x {y_key} failed: {e}")

    # Sankey diagrams
    sankey_configs = [
        ["Countries", "Institutions", "Authors"],
        ["Countries", "Journals"],
        ["Authors", "Author Keywords"],
    ]
    for dims in sankey_configs:
        try:
            create_sankey_generic(articles, dims, output_dir)
        except Exception as e:
            logger.error(f"Sankey {dims} failed: {e}")

    # Reference graph
    try:
        create_main_reference_graph(articles, output_dir)
    except Exception as e:
        logger.error(f"Reference graph failed: {e}")

    # Citation trajectory
    try:
        create_citation_trajectory(articles, output_dir)
    except Exception as e:
        logger.error(f"Citation trajectory failed: {e}")

    # Citation matrix
    try:
        create_citation_matrix(articles, output_dir)
    except Exception as e:
        logger.error(f"Citation matrix failed: {e}")

    # RPYS
    try:
        create_rpys_chart(articles, output_dir)
    except Exception as e:
        logger.error(f"RPYS failed: {e}")

    logger.info("Interactive charts completed.")
