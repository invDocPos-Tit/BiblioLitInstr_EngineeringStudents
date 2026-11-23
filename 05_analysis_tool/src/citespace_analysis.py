"""Fase 2h: CiteSpace Analysis Suite.

Implements 8 CiteSpace-inspired analyses:
1. Kleinberg Burst Detection (keywords, authors, references, journals)
2. Timeline Visualization (co-citation clusters)
3. Timezone View (concentric rings by year)
4. Pivot Node Detection (inter-cluster bridge nodes)
5. Sigma (S) Metric = (centrality + 1) ^ burstness
6. Research Front Identification (intellectual base → citing front)
7. Cluster Labeling (TF-IDF, LLR, MI)
8. CiteSpace Report (markdown + JSON + Plotly charts)
"""
from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from config import CITESPACE_DIR, ENRICHED_PATH, OUTPUT_DIR
from src.utils import (
    save_json, resolve_openalex_label_cached, is_noisy_legacy_concept,
    safe_write_plotly_png,
)

logger = logging.getLogger("bibliometrico.citespace_analysis")


def _is_openalex_url(s: str) -> bool:
    """Check if a string is an opaque OpenAlex URL."""
    return isinstance(s, str) and "openalex.org/" in s


# ======================================================================
# 1. KLEINBERG BURST DETECTION
# ======================================================================

def _kleinberg_burst_detection(
    event_counts: Dict[int, int],
    s: float = 2.0,
    gamma: float = 1.0,
    year_totals: Optional[Dict[int, int]] = None,
) -> List[Dict[str, Any]]:
    """Kleinberg (2003) batched binomial automaton burst detection via Viterbi/DP.

    Args:
        event_counts: {year: count} for an entity
        s: state scaling multiplier (default 2.0)
        gamma: state transition cost coefficient (default 1.0)
        year_totals: {year: total_events_or_docs_in_year} across the corpus

    Returns:
        List of burst intervals: {level, begin_year, end_year, strength, raw_count, duration}
    """
    if not event_counts:
        return []

    obs_years = sorted(event_counts.keys())
    if year_totals:
        all_years = sorted(set(year_totals.keys()) | set(obs_years))
        # Restrict span from first active corpus year to last active corpus year
        min_y = min(obs_years[0], min(year_totals.keys()))
        max_y = max(obs_years[-1], max(year_totals.keys()))
        years = [y for y in all_years if min_y <= y <= max_y]
    else:
        if len(obs_years) < 2:
            return []
        years = list(range(obs_years[0], obs_years[-1] + 1))

    if len(years) < 2:
        return []

    counts = [int(event_counts.get(y, 0)) for y in years]
    n = len(counts)
    total_r = sum(counts)

    if total_r == 0:
        return []

    if year_totals:
        d_t = [max(counts[idx], int(year_totals.get(y, 0)), 1) for idx, y in enumerate(years)]
    else:
        max_c = max(counts)
        d_t = [max(counts[idx], max_c * 5, 10) for idx in range(n)]

    total_d = sum(d_t)
    if total_d <= total_r:
        total_d = total_r * 2

    # Base probability p0 = R / D (Kleinberg 2003, Section 3)
    p0 = max(min(total_r / total_d, 0.49), 1e-6)

    # Number of states (max burst level)
    k = max(2, int(math.ceil(math.log(max(total_d, 2), max(s, 1.1)))) + 1)

    # State probabilities: p_i = p0 * s^i (capped at 0.999)
    rates = [min(p0 * (s ** i), 0.999) for i in range(k)]

    # Binomial negative log-likelihood emission cost (up to additive log(C(d_t, r_t)) constant)
    def _cost(r: int, d: int, state: int) -> float:
        p = rates[state]
        if p <= 0.0 or p >= 1.0:
            return 1e10
        rem = max(0, d - r)
        return -(r * math.log(max(p, 1e-15)) + rem * math.log(max(1.0 - p, 1e-15)))

    # Transition cost: gamma * ln(n) * (j - i) for going from state i to j > i
    def _tau(i: int, j: int) -> float:
        if j <= i:
            return 0.0
        return gamma * math.log(max(n, 2)) * (j - i)

    # Viterbi DP
    dp = [[float("inf")] * k for _ in range(n)]
    backptr = [[0] * k for _ in range(n)]

    for j in range(k):
        dp[0][j] = _tau(0, j) + _cost(counts[0], d_t[0], j)

    for t in range(1, n):
        for j in range(k):
            c = _cost(counts[t], d_t[t], j)
            for i in range(k):
                val = dp[t - 1][i] + _tau(i, j) + c
                if val < dp[t][j]:
                    dp[t][j] = val
                    backptr[t][j] = i

    # Backtrack optimal state sequence
    states = [0] * n
    states[n - 1] = int(np.argmin(dp[n - 1]))
    for t in range(n - 2, -1, -1):
        states[t] = backptr[t + 1][states[t + 1]]

    # Extract burst intervals (state > 0) and compute Kleinberg log-likelihood burst weight
    bursts = []
    i = 0
    while i < n:
        if states[i] > 0:
            level = states[i]
            begin = i
            while i < n and states[i] >= level:
                i += 1
            end = i - 1
            # Trim leading/trailing zero-count years inside the interval if any
            while begin <= end and counts[begin] == 0:
                begin += 1
            while end >= begin and counts[end] == 0:
                end -= 1
            if begin <= end:
                ll_weight = sum(
                    max(0.0, _cost(counts[t], d_t[t], 0) - _cost(counts[t], d_t[t], states[t]))
                    for t in range(begin, end + 1)
                )
                raw_count = sum(counts[begin : end + 1])
                bursts.append({
                    "level": int(max(states[begin : end + 1])),
                    "begin_year": years[begin],
                    "end_year": years[end],
                    "strength": round(float(ll_weight), 2),
                    "raw_count": int(raw_count),
                    "duration": years[end] - years[begin] + 1,
                })
        else:
            i += 1

    return bursts


def _extract_entity_year_counts(
    articles: List[dict],
    entity_type: str,
) -> Dict[str, Dict[int, int]]:
    """Extract entity-year frequency counts from articles.

    Args:
        entity_type: 'keywords', 'authors', 'references', 'journals'

    Returns:
        {entity_name: {year: count}}
    """
    counts: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    kw_normalizer = None
    if entity_type == "keywords":
        from src.utils import build_corpus_keyword_normalizer, extract_normalized_article_keywords
        kw_normalizer = build_corpus_keyword_normalizer(articles)

    for article in articles:
        year = article.get("publication_year")
        if not year:
            continue
        try:
            year = int(year)
        except (ValueError, TypeError):
            continue

        if entity_type == "keywords":
            tags = extract_normalized_article_keywords(
                article, corpus_map=kw_normalizer, drop_stopwords=True
            )
            for tag in set(tags):  # deduplicate per article
                if not is_noisy_legacy_concept(tag):
                    counts[tag][year] += 1

        elif entity_type == "authors":
            authors = article.get("authors_parsed", [])
            if not authors:
                auth_raw = article.get("authors", "")
                if isinstance(auth_raw, list):
                    authors = [str(a).strip() for a in auth_raw if str(a).strip()]
                elif isinstance(auth_raw, str) and auth_raw.strip():
                    authors = [
                        a.strip() for a in auth_raw.split(";") if a.strip()
                    ]
            for author in set(authors):
                name = author.strip() if isinstance(author, str) else str(author)
                if name:
                    counts[name][year] += 1

        elif entity_type == "references":
            refs = article.get("referenced_works", [])
            if isinstance(refs, list):
                for ref in set(refs):
                    ref_str = str(ref).strip()
                    if ref_str:
                        counts[ref_str][year] += 1

        elif entity_type == "journals":
            journal = article.get("publication_title", "")
            if isinstance(journal, str) and journal.strip():
                counts[journal.strip()][year] += 1

    return dict(counts)


def detect_bursts(
    articles: List[dict],
    entity_type: str = "keywords",
    s: float = 2.0,
    gamma: float = 1.0,
    min_occurrences: int = 3,
) -> List[Dict[str, Any]]:
    """Detect bursts for all entities of a given type.

    Returns sorted list of burst records with entity info.
    """
    entity_counts = _extract_entity_year_counts(articles, entity_type)

    # Compute annual denominator d_t (articles per year in corpus)
    year_totals: Dict[int, int] = defaultdict(int)
    for article in articles:
        yr = article.get("publication_year")
        try:
            yr_int = int(yr)
            year_totals[yr_int] += 1
        except (ValueError, TypeError):
            continue

    # Mismos parametros para todos los tipos de entidad. Antes se fijaban a mano gamma = 0.6 y
    # min_occurrences = 2 para autores y revistas ("corpus compactos"), pisando --citespace-burst-gamma.
    effective_min = min_occurrences
    effective_gamma = gamma

    all_bursts = []
    for entity, year_counts in entity_counts.items():
        total = sum(year_counts.values())
        if total < effective_min:
            continue

        bursts = _kleinberg_burst_detection(
            year_counts, s=s, gamma=effective_gamma, year_totals=dict(year_totals)
        )
        for b in bursts:
            b["entity"] = entity
            b["entity_type"] = entity_type
            b["total_occurrences"] = total
            all_bursts.append(b)

    # Sort by strength descending, then total_occurrences descending, then entity name
    all_bursts.sort(key=lambda x: (-x["strength"], -x["total_occurrences"], x["entity"]))
    return all_bursts


def plot_burst_timeline(
    bursts: List[Dict[str, Any]],
    entity_type: str,
    output_dir: Path,
    top_n: int = 30,
) -> Optional[Path]:
    """Create Plotly horizontal timeline of burst intervals."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        logger.warning("plotly not available for burst timeline")
        return None

    if not bursts:
        return None

    # Take top N by strength
    top_bursts = bursts[:top_n]
    top_bursts.reverse()  # bottom to top for display

    fig = go.Figure()

    # Color by burst level
    max_level = max(b["level"] for b in top_bursts) if top_bursts else 1
    colors = [
        f"rgba({min(255, int(50 + 200 * b['level'] / max_level))}, "
        f"{max(0, int(200 - 150 * b['level'] / max_level))}, "
        f"{max(0, int(100 - 80 * b['level'] / max_level))}, 0.8)"
        for b in top_bursts
    ]

    entities = []
    for i, b in enumerate(top_bursts):
        entity_label = b["entity"]
        if len(entity_label) > 40:
            entity_label = entity_label[:37] + "..."
        entities.append(entity_label)

        fig.add_trace(go.Bar(
            y=[entity_label],
            x=[b["end_year"] - b["begin_year"] + 1],
            base=[b["begin_year"]],
            orientation="h",
            marker=dict(color=colors[i]),
            text=f"L{b['level']} ({b['begin_year']}-{b['end_year']}) S={b['strength']}",
            textposition="inside",
            hovertemplate=(
                f"<b>{b['entity']}</b><br>"
                f"Period: {b['begin_year']}-{b['end_year']}<br>"
                f"Level: {b['level']}<br>"
                f"Strength: {b['strength']}<br>"
                f"Duration: {b['duration']} years<extra></extra>"
            ),
            showlegend=False,
        ))

    fig.update_layout(
        title=f"Burst Detection Timeline — {entity_type.title()}",
        xaxis_title="Year",
        yaxis_title="",
        height=max(400, len(top_bursts) * 25 + 100),
        width=1000,
        barmode="overlay",
        margin=dict(l=250),
    )

    output_path = output_dir / f"burst_{entity_type}.html"
    fig.write_html(str(output_path))

    # PNG export
    safe_write_plotly_png(fig, output_path.with_suffix(".png"), scale=2, log=logger)

    return output_path


# ======================================================================
# 2. TIMELINE VISUALIZATION (co-citation clusters)
# ======================================================================

def _build_cocitation_network_internal(
    articles: List[dict], min_cocitations: int = 2
) -> nx.Graph:
    """Build co-citation network from articles (internal, no API calls)."""
    G = nx.Graph()
    ref_to_articles: Dict[str, List[int]] = defaultdict(list)

    for i, article in enumerate(articles):
        refs = article.get("referenced_works", [])
        if isinstance(refs, list):
            for ref in refs:
                ref_str = str(ref).strip()
                if ref_str:
                    ref_to_articles[ref_str].append(i)

    # Build edges: two refs co-cited if they appear together in same article
    ref_list = list(ref_to_articles.keys())
    for i in range(len(ref_list)):
        for j in range(i + 1, len(ref_list)):
            common = set(ref_to_articles[ref_list[i]]) & set(
                ref_to_articles[ref_list[j]]
            )
            if len(common) >= min_cocitations:
                G.add_edge(ref_list[i], ref_list[j], weight=len(common))

    # Add node attributes
    for ref in ref_list:
        if ref in G:
            G.nodes[ref]["cited_by_count"] = len(ref_to_articles[ref])

    # Prune disconnected micro-islands (< 4 nodes) so Louvain does not report 100+ 2-node pairs
    if G.number_of_nodes() > 0:
        valid_nodes = set()
        for comp in nx.connected_components(G):
            if len(comp) >= 4:
                valid_nodes.update(comp)
        if len(valid_nodes) >= 10:
            G = G.subgraph(valid_nodes).copy()

    return G


def _detect_communities_internal(G: nx.Graph) -> dict:
    """Detect communities using Louvain (fails explicitly if python-louvain is missing)."""
    if G.number_of_nodes() < 2:
        return {}
    import community as community_louvain
    partition = community_louvain.best_partition(G, random_state=42)
    nx.set_node_attributes(G, partition, "community")
    return partition


def compute_modularity_silhouette(
    G: nx.Graph, partition: dict,
) -> Dict[str, float]:
    """Compute Modularity Q and mean Silhouette S for a clustered network.

    These are the two primary quality metrics reported in CiteSpace analyses:
    - Q > 0.3: significant clustering structure
    - Q 0.4-0.8: reasonable
    - S > 0.5: reasonable clustering
    - S > 0.7: convincing clustering

    Returns dict with 'modularity_q', 'mean_silhouette', per-cluster silhouettes.
    """
    if not partition or G.number_of_nodes() < 3:
        return {"modularity_q": 0.0, "mean_silhouette": 0.0, "cluster_silhouettes": {}}

    # --- Modularity Q ---
    try:
        import community as community_louvain
        q = community_louvain.modularity(partition, G)
    except (ImportError, Exception):
        # Manual computation: Q = (1/2m) * sum_ij [A_ij - k_i*k_j/(2m)] * delta(c_i, c_j)
        m2 = sum(d.get("weight", 1) for _, _, d in G.edges(data=True))
        if m2 == 0:
            q = 0.0
        else:
            q = 0.0
            for u in G.nodes():
                for v in G.nodes():
                    if partition.get(u) == partition.get(v):
                        a_uv = G[u][v].get("weight", 1) if G.has_edge(u, v) else 0
                        k_u = sum(G[u][n].get("weight", 1) for n in G.neighbors(u))
                        k_v = sum(G[v][n].get("weight", 1) for n in G.neighbors(v))
                        q += a_uv - k_u * k_v / m2
            q /= m2

    # --- Mean Silhouette S ---
    # For each node: s(i) = (b(i) - a(i)) / max(a(i), b(i))
    # a(i) = avg distance to same-cluster nodes
    # b(i) = min avg distance to nodes of any other cluster
    # Distance = 1 / (weight + 1) for co-occurrence; 1 for unconnected
    clusters: Dict[int, List] = defaultdict(list)
    for node, comm in partition.items():
        if node in G:
            clusters[comm].append(node)

    # Shortest path lengths (or use simple weight-based distance)
    node_silhouettes: Dict[str, float] = {}
    cluster_silhouettes: Dict[int, float] = {}

    for cid, members in clusters.items():
        if len(members) < 2:
            for m in members:
                node_silhouettes[m] = 0.0
            cluster_silhouettes[cid] = 0.0
            continue

        member_sils = []
        for node in members:
            # a(i): average distance to same-cluster nodes
            same_dists = []
            for other in members:
                if other == node:
                    continue
                if G.has_edge(node, other):
                    w = G[node][other].get("weight", 1)
                    same_dists.append(1.0 / (w + 1))
                else:
                    same_dists.append(1.0)
            a_i = np.mean(same_dists) if same_dists else 0.0

            # b(i): min average distance to any other cluster
            b_i = float("inf")
            for other_cid, other_members in clusters.items():
                if other_cid == cid or not other_members:
                    continue
                other_dists = []
                for other in other_members:
                    if G.has_edge(node, other):
                        w = G[node][other].get("weight", 1)
                        other_dists.append(1.0 / (w + 1))
                    else:
                        other_dists.append(1.0)
                avg_dist = np.mean(other_dists) if other_dists else 1.0
                b_i = min(b_i, avg_dist)

            if b_i == float("inf"):
                b_i = 1.0

            denom = max(a_i, b_i)
            s_i = (b_i - a_i) / denom if denom > 0 else 0.0
            node_silhouettes[node] = s_i
            member_sils.append(s_i)

        cluster_silhouettes[cid] = float(np.mean(member_sils)) if member_sils else 0.0

    all_sils = list(node_silhouettes.values())
    mean_s = float(np.mean(all_sils)) if all_sils else 0.0

    return {
        "modularity_q": round(q, 4),
        "mean_silhouette": round(mean_s, 4),
        "cluster_silhouettes": {k: round(v, 4) for k, v in cluster_silhouettes.items()},
    }


def _extract_ref_year(ref_id: str, articles: List[dict]) -> Optional[int]:
    """Try to extract publication year for a reference."""
    # Check if any article's resolved_refs has year info
    for article in articles:
        resolved = article.get("resolved_references", {})
        if isinstance(resolved, dict):
            ref_data = resolved.get(ref_id, {})
            if isinstance(ref_data, dict):
                year = ref_data.get("year") or ref_data.get("publication_year")
                if year:
                    try:
                        return int(year)
                    except (ValueError, TypeError):
                        pass

    # Try extracting year from OpenAlex-style URL
    # e.g., https://openalex.org/W... — no year info in URL itself
    return None


def plot_cocitation_timeline(
    articles: List[dict],
    G_cocit: Optional[nx.Graph] = None,
    output_dir: Path = CITESPACE_DIR,
) -> Optional[Path]:
    """Create timeline visualization of co-citation clusters.

    Y-axis = cluster, X-axis = year, nodes = references
    Size = citations, color = cluster
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        logger.warning("plotly not available")
        return None

    if G_cocit is None:
        G_cocit = _build_cocitation_network_internal(articles, min_cocitations=2)

    if G_cocit.number_of_nodes() < 3:
        logger.info("Co-citation network too small for timeline")
        return None

    partition = _detect_communities_internal(G_cocit)
    if not partition:
        return None

    # Get years for reference nodes
    ref_years = {}
    for node in G_cocit.nodes():
        year = _extract_ref_year(node, articles)
        if year:
            ref_years[node] = year

    if not ref_years:
        # Fallback: use citing article years
        ref_to_years: Dict[str, List[int]] = defaultdict(list)
        for article in articles:
            year = article.get("publication_year")
            if not year:
                continue
            try:
                year = int(year)
            except (ValueError, TypeError):
                continue
            refs = article.get("referenced_works", [])
            if isinstance(refs, list):
                for ref in refs:
                    ref_str = str(ref).strip()
                    if ref_str in G_cocit:
                        ref_to_years[ref_str].append(year)

        for ref, yrs in ref_to_years.items():
            if yrs:
                ref_years[ref] = min(yrs)  # Use earliest citing year

    if not ref_years:
        logger.info("No year data available for co-citation timeline")
        return None

    # Group by cluster
    clusters: Dict[int, List[str]] = defaultdict(list)
    for node, comm in partition.items():
        clusters[comm].append(node)

    # Sort clusters by size
    sorted_clusters = sorted(clusters.items(), key=lambda x: -len(x[1]))
    cluster_y_map = {cid: i for i, (cid, _) in enumerate(sorted_clusters)}

    # Colorscale
    n_clusters = len(sorted_clusters)
    import plotly.express as px
    palette = px.colors.qualitative.Set3 if n_clusters <= 12 else px.colors.qualitative.Alphabet

    fig = go.Figure()

    for cid, members in sorted_clusters:
        y_val = cluster_y_map[cid]
        xs = []
        ys = []
        sizes = []
        texts = []
        for node in members:
            year = ref_years.get(node)
            if year is None:
                continue
            cited = G_cocit.nodes[node].get("cited_by_count", 1)
            xs.append(year)
            ys.append(y_val)
            sizes.append(max(5, min(30, cited * 2)))
            label = resolve_openalex_label_cached(node, articles) if _is_openalex_url(node) else node
            if len(label) > 50:
                label = label[:47] + "..."
            texts.append(f"{label}<br>Year: {year}<br>Citations: {cited}")

        if xs:
            color = palette[cid % len(palette)]
            fig.add_trace(go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                marker=dict(size=sizes, color=color, opacity=0.7,
                            line=dict(width=1, color="white")),
                text=texts,
                hoverinfo="text",
                name=f"Cluster {cid} (n={len(members)})",
            ))

    fig.update_layout(
        title="Co-citation Timeline (Clusters × Years)",
        xaxis_title="Year",
        yaxis_title="Cluster",
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(sorted_clusters))),
            ticktext=[f"C{cid} (n={len(m)})"
                      for cid, m in sorted_clusters],
        ),
        height=max(400, len(sorted_clusters) * 60 + 100),
        width=1200,
        showlegend=True,
    )

    output_path = output_dir / "cocitation_timeline.html"
    fig.write_html(str(output_path))
    safe_write_plotly_png(fig, output_path.with_suffix(".png"), scale=2, log=logger)
    return output_path


# ======================================================================
# 3. TIMEZONE VIEW (concentric rings)
# ======================================================================

def plot_timezone_view(
    articles: List[dict],
    G_cocit: Optional[nx.Graph] = None,
    output_dir: Path = CITESPACE_DIR,
) -> Optional[Path]:
    """Create timezone view: concentric rings by year.

    Inner ring = oldest, outer ring = most recent.
    Nodes positioned by year (radius) and cluster (angle).
    """
    try:
        import plotly.graph_objects as go
        import plotly.express as px
    except ImportError:
        logger.warning("plotly not available")
        return None

    if G_cocit is None:
        G_cocit = _build_cocitation_network_internal(articles, min_cocitations=2)

    if G_cocit.number_of_nodes() < 3:
        logger.info("Co-citation network too small for timezone view")
        return None

    partition = _detect_communities_internal(G_cocit)

    # Get years
    ref_to_years: Dict[str, List[int]] = defaultdict(list)
    for article in articles:
        year = article.get("publication_year")
        if not year:
            continue
        try:
            year = int(year)
        except (ValueError, TypeError):
            continue
        refs = article.get("referenced_works", [])
        if isinstance(refs, list):
            for ref in refs:
                ref_str = str(ref).strip()
                if ref_str in G_cocit:
                    ref_to_years[ref_str].append(year)

    ref_years = {r: min(yrs) for r, yrs in ref_to_years.items() if yrs}

    if not ref_years:
        logger.info("No year data for timezone view")
        return None

    years_list = sorted(set(ref_years.values()))
    if len(years_list) < 2:
        return None

    min_year = years_list[0]
    max_year = years_list[-1]
    year_range = max_year - min_year

    if year_range == 0:
        return None

    # Group nodes by cluster
    clusters: Dict[int, List[str]] = defaultdict(list)
    for node, comm in partition.items():
        clusters[comm].append(node)

    n_clusters = max(len(clusters), 1)
    palette = px.colors.qualitative.Set3 if n_clusters <= 12 else px.colors.qualitative.Alphabet

    fig = go.Figure()

    # Draw concentric rings for each year
    for year in years_list:
        radius = 0.3 + 0.7 * (year - min_year) / year_range
        theta = np.linspace(0, 2 * np.pi, 100)
        fig.add_trace(go.Scatter(
            x=radius * np.cos(theta),
            y=radius * np.sin(theta),
            mode="lines",
            line=dict(color="rgba(200,200,200,0.3)", width=1),
            showlegend=False,
            hoverinfo="skip",
        ))

    # Place nodes
    for cid, members in clusters.items():
        angle_base = 2 * np.pi * cid / n_clusters
        xs, ys, sizes, texts = [], [], [], []

        for idx, node in enumerate(members):
            year = ref_years.get(node)
            if year is None:
                continue
            radius = 0.3 + 0.7 * (year - min_year) / year_range
            # Spread nodes within cluster sector
            angle = angle_base + (idx * 0.15 / max(1, len(members)))
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            cited = G_cocit.nodes[node].get("cited_by_count", 1)

            xs.append(x)
            ys.append(y)
            sizes.append(max(5, min(25, cited * 2)))
            label = resolve_openalex_label_cached(node, articles) if _is_openalex_url(node) else node
            if len(label) > 50:
                label = label[:47] + "..."
            texts.append(f"{label}<br>Year: {year}<br>Cluster: {cid}")

        if xs:
            color = palette[cid % len(palette)]
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="markers",
                marker=dict(size=sizes, color=color, opacity=0.7,
                            line=dict(width=1, color="white")),
                text=texts, hoverinfo="text",
                name=f"Cluster {cid}",
            ))

    # Year labels on right side
    for year in years_list[::max(1, len(years_list) // 8)]:
        radius = 0.3 + 0.7 * (year - min_year) / year_range
        fig.add_annotation(
            x=radius + 0.05, y=0, text=str(year),
            showarrow=False, font=dict(size=9, color="gray"),
        )

    fig.update_layout(
        title="Timezone View (Concentric Rings by Year)",
        xaxis=dict(visible=False, scaleanchor="y"),
        yaxis=dict(visible=False),
        width=900, height=900,
        showlegend=True,
    )

    output_path = output_dir / "timezone_view.html"
    fig.write_html(str(output_path))
    safe_write_plotly_png(fig, output_path.with_suffix(".png"), scale=2, log=logger)
    return output_path


# ======================================================================
# 4. PIVOT NODE DETECTION
# ======================================================================

def detect_pivot_nodes(
    G: nx.Graph,
    partition: Optional[dict] = None,
    top_n: int = 20,
) -> List[Dict[str, Any]]:
    """Detect pivot nodes: bridge nodes between clusters.

    Bridge score = n_neighbor_clusters * betweenness_centrality
    """
    if G.number_of_nodes() < 3:
        return []

    if partition is None:
        partition = _detect_communities_internal(G)

    if not partition:
        return []

    # Compute betweenness centrality
    from src.utils import strength_betweenness
    betweenness = strength_betweenness(G)   # distancia = 1/peso (antes el peso se tomaba como distancia)

    pivot_nodes = []
    for node in G.nodes():
        if node not in partition:
            continue

        bc = betweenness.get(node, 0)
        node_cluster = partition[node]

        # Find neighbor clusters
        neighbor_clusters = set()
        for neighbor in G.neighbors(node):
            nc = partition.get(neighbor)
            if nc is not None and nc != node_cluster:
                neighbor_clusters.add(nc)

        n_clusters = len(neighbor_clusters)
        if n_clusters > 0:
            bridge_score = n_clusters * bc
            pivot_nodes.append({
                "node": node,
                "cluster": node_cluster,
                "betweenness": round(bc, 6),
                "n_neighbor_clusters": n_clusters,
                "neighbor_clusters": sorted(neighbor_clusters),
                "bridge_score": round(bridge_score, 6),
                "degree": G.degree(node),
            })

    pivot_nodes.sort(key=lambda x: x["bridge_score"], reverse=True)
    return pivot_nodes[:top_n]


def plot_pivot_nodes(
    pivot_nodes: List[Dict[str, Any]],
    output_dir: Path = CITESPACE_DIR,
) -> Optional[Path]:
    """Scatter plot of pivot nodes: betweenness vs neighbor clusters."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    if not pivot_nodes:
        return None

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[p["betweenness"] for p in pivot_nodes],
        y=[p["n_neighbor_clusters"] for p in pivot_nodes],
        mode="markers+text",
        marker=dict(
            size=[max(8, min(40, p["bridge_score"] * 500)) for p in pivot_nodes],
            color=[p["bridge_score"] for p in pivot_nodes],
            colorscale="YlOrRd",
            showscale=True,
            colorbar=dict(title="Bridge Score"),
            opacity=0.7,
            line=dict(width=1, color="white"),
        ),
        text=[p["node"][:30] for p in pivot_nodes],
        textposition="top center",
        textfont=dict(size=8),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Betweenness: %{x:.4f}<br>"
            "Neighbor clusters: %{y}<br>"
            "Bridge score: %{customdata[1]:.4f}<br>"
            "Degree: %{customdata[2]}<extra></extra>"
        ),
        customdata=[(p["node"], p["bridge_score"], p["degree"])
                     for p in pivot_nodes],
    ))

    fig.update_layout(
        title="Pivot Nodes (Inter-cluster Bridges)",
        xaxis_title="Betweenness Centrality",
        yaxis_title="Number of Neighbor Clusters",
        width=900, height=600,
    )

    output_path = output_dir / "pivot_nodes.html"
    fig.write_html(str(output_path))
    safe_write_plotly_png(fig, output_path.with_suffix(".png"), scale=2, log=logger)
    return output_path


# ======================================================================
# 5. SIGMA (Σ) METRIC
# ======================================================================

def compute_sigma(
    G: nx.Graph,
    bursts_by_node: Dict[str, float],
) -> List[Dict[str, Any]]:
    """Compute Sigma metric: S = (centrality + 1) ^ burstness.

    Args:
        G: network graph
        bursts_by_node: {node: max_burst_strength}

    Returns:
        Sorted list of {node, betweenness, burstness, sigma}
    """
    if G.number_of_nodes() < 2:
        return []

    from src.utils import strength_betweenness
    betweenness = strength_betweenness(G)   # distancia = 1/peso (antes el peso se tomaba como distancia)

    sigma_list = []
    for node in G.nodes():
        bc = betweenness.get(node, 0)
        burst = bursts_by_node.get(node, 0)
        s = (bc + 1) ** burst if burst > 0 else bc + 1
        sigma_list.append({
            "node": node,
            "betweenness": round(bc, 6),
            "burstness": round(burst, 4),
            "sigma": round(s, 6),
            "degree": G.degree(node),
        })

    sigma_list.sort(key=lambda x: x["sigma"], reverse=True)
    return sigma_list


def plot_sigma_scatter(
    sigma_list: List[Dict[str, Any]],
    output_dir: Path = CITESPACE_DIR,
    top_n: int = 50,
) -> Optional[Path]:
    """Scatter: betweenness vs burstness, size = Sigma."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    if not sigma_list:
        return None

    top = sigma_list[:top_n]

    fig = go.Figure()

    max_sigma = max(s["sigma"] for s in top) if top else 1
    fig.add_trace(go.Scatter(
        x=[s["betweenness"] for s in top],
        y=[s["burstness"] for s in top],
        mode="markers+text",
        marker=dict(
            size=[max(6, min(50, s["sigma"] / max(max_sigma, 0.001) * 50))
                  for s in top],
            color=[s["sigma"] for s in top],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Sigma (Σ)"),
            opacity=0.7,
            line=dict(width=1, color="white"),
        ),
        text=[s["node"][:25] for s in top],
        textposition="top center",
        textfont=dict(size=7),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Betweenness: %{x:.4f}<br>"
            "Burstness: %{y:.2f}<br>"
            "Sigma: %{customdata[1]:.4f}<extra></extra>"
        ),
        customdata=[(s["node"], s["sigma"]) for s in top],
    ))

    fig.update_layout(
        title="Sigma (Σ) Metric: Betweenness × Burstness",
        xaxis_title="Betweenness Centrality",
        yaxis_title="Burstness (max burst strength)",
        width=900, height=600,
    )

    output_path = output_dir / "sigma_scatter.html"
    fig.write_html(str(output_path))
    safe_write_plotly_png(fig, output_path.with_suffix(".png"), scale=2, log=logger)
    return output_path


# ======================================================================
# 6. RESEARCH FRONT IDENTIFICATION
# ======================================================================

def identify_research_fronts(
    articles: List[dict],
    G_cocit: Optional[nx.Graph] = None,
    partition: Optional[dict] = None,
) -> List[Dict[str, Any]]:
    """Identify research fronts from co-citation clusters.

    Intellectual base = cluster members (cited references).
    Research front = corpus articles that cite the cluster.
    Labels from keywords of citing articles.
    """
    if G_cocit is None:
        G_cocit = _build_cocitation_network_internal(articles, min_cocitations=2)

    if G_cocit.number_of_nodes() < 3:
        return []

    if partition is None:
        partition = _detect_communities_internal(G_cocit)

    if not partition:
        return []

    # Group references by cluster
    clusters: Dict[int, set] = defaultdict(set)
    for node, comm in partition.items():
        clusters[comm].add(node)

    # Find citing articles for each cluster
    fronts = []
    for cid, cluster_refs in clusters.items():
        citing_articles = []
        citing_years = []
        citing_keywords = []

        for article in articles:
            refs = article.get("referenced_works", [])
            if not isinstance(refs, list):
                continue
            article_refs = set(str(r).strip() for r in refs if r)
            overlap = article_refs & cluster_refs

            if overlap:
                citing_articles.append(article.get("key", ""))
                year = article.get("publication_year")
                if year:
                    try:
                        citing_years.append(int(year))
                    except (ValueError, TypeError):
                        pass

                # Collect keywords (all sources)
                for field in ("manual_tags", "automatic_tags"):
                    val = article.get(field, "")
                    if isinstance(val, str) and val.strip():
                        citing_keywords.extend(
                            t.strip().lower()
                            for t in val.split(";")
                            if t.strip()
                        )
                    elif isinstance(val, list):
                        citing_keywords.extend(t.strip().lower() for t in val if isinstance(t, str) and t.strip())
                for kw_obj in article.get("openalex", {}).get("keywords", []):
                    kw = kw_obj.get("keyword", "") if isinstance(kw_obj, dict) else ""
                    if kw.strip():
                        citing_keywords.append(kw.strip().lower())
                for mesh in article.get("mesh_terms", []):
                    if isinstance(mesh, str) and mesh.strip():
                        citing_keywords.append(mesh.strip().lower())

        if not citing_articles:
            continue

        # Top keywords by frequency (filtering noisy legacy MAG concepts)
        kw_counts = Counter(kw for kw in citing_keywords if not is_noisy_legacy_concept(kw))
        top_kw = [kw for kw, _ in kw_counts.most_common(10)]

        front = {
            "cluster_id": cid,
            "intellectual_base_size": len(cluster_refs),
            "research_front_size": len(citing_articles),
            "citing_articles": citing_articles[:20],  # limit for JSON
            "top_keywords": top_kw,
            "label": "; ".join(top_kw[:3]) if top_kw else f"Cluster {cid}",
            "year_range": (
                [min(citing_years), max(citing_years)] if citing_years else None
            ),
        }
        fronts.append(front)

    fronts.sort(key=lambda x: x["research_front_size"], reverse=True)
    return fronts


def plot_research_fronts(
    fronts: List[Dict[str, Any]],
    output_dir: Path = CITESPACE_DIR,
) -> Optional[Path]:
    """Bar chart: intellectual base vs research front per cluster."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    if not fronts:
        return None

    labels = [f["label"][:40] for f in fronts]
    base_sizes = [f["intellectual_base_size"] for f in fronts]
    front_sizes = [f["research_front_size"] for f in fronts]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=base_sizes, orientation="h",
        name="Intellectual Base (cited refs)",
        marker_color="steelblue",
    ))
    fig.add_trace(go.Bar(
        y=labels, x=front_sizes, orientation="h",
        name="Research Front (citing articles)",
        marker_color="coral",
    ))

    fig.update_layout(
        title="Research Fronts: Intellectual Base vs Citing Front",
        xaxis_title="Count",
        barmode="group",
        height=max(400, len(fronts) * 40 + 100),
        width=1000,
        margin=dict(l=300),
    )

    output_path = output_dir / "research_fronts.html"
    fig.write_html(str(output_path))
    safe_write_plotly_png(fig, output_path.with_suffix(".png"), scale=2, log=logger)
    return output_path


# ======================================================================
# 7. CLUSTER LABELING (TF-IDF, LLR, MI)
# ======================================================================

def label_clusters(
    articles: List[dict],
    G_cocit: Optional[nx.Graph] = None,
    partition: Optional[dict] = None,
    top_n_terms: int = 5,
) -> Dict[int, Dict[str, Any]]:
    """Label co-citation clusters using TF-IDF, LLR, and MI.

    Uses titles/abstracts of citing articles for each cluster.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        logger.warning("scikit-learn required for cluster labeling")
        return {}

    if G_cocit is None:
        G_cocit = _build_cocitation_network_internal(articles, min_cocitations=2)

    if G_cocit.number_of_nodes() < 3:
        return {}

    if partition is None:
        partition = _detect_communities_internal(G_cocit)

    if not partition:
        return {}

    # Group references by cluster
    clusters: Dict[int, set] = defaultdict(set)
    for node, comm in partition.items():
        clusters[comm].add(node)

    # Collect text for each cluster from citing articles
    cluster_texts: Dict[int, List[str]] = defaultdict(list)
    cluster_articles: Dict[int, List[str]] = defaultdict(list)

    for article in articles:
        refs = article.get("referenced_works", [])
        if not isinstance(refs, list):
            continue
        article_refs = set(str(r).strip() for r in refs if r)

        for cid, cluster_refs in clusters.items():
            if article_refs & cluster_refs:
                # Combine title and abstract
                title = article.get("title", "")
                abstract = article.get("abstract", "")
                text = f"{title} {abstract}".strip()
                if text:
                    cluster_texts[cid].append(text)
                    cluster_articles[cid].append(article.get("key", ""))

    if not cluster_texts:
        return {}

    # Combine all cluster texts for TF-IDF
    all_cluster_ids = sorted(cluster_texts.keys())
    all_docs = []
    doc_cluster_map = []
    for cid in all_cluster_ids:
        # Combine all texts for this cluster into one document
        combined = " ".join(cluster_texts[cid])
        all_docs.append(combined)
        doc_cluster_map.append(cid)

    if len(all_docs) < 2:
        return {}

    # TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words="english",
        min_df=1,
        max_df=0.9,
        ngram_range=(1, 2),
    )
    try:
        tfidf_matrix = vectorizer.fit_transform(all_docs)
    except ValueError:
        return {}

    feature_names = vectorizer.get_feature_names_out()

    cluster_labels: Dict[int, Dict[str, Any]] = {}

    for idx, cid in enumerate(all_cluster_ids):
        tfidf_scores = tfidf_matrix[idx].toarray().flatten()
        top_indices = tfidf_scores.argsort()[-top_n_terms:][::-1]
        tfidf_terms = [
            {"term": feature_names[i], "score": round(float(tfidf_scores[i]), 4)}
            for i in top_indices
            if tfidf_scores[i] > 0
        ]

        # LLR (Log-Likelihood Ratio)
        llr_terms = _compute_llr(
            cluster_texts[cid], cluster_texts, cid, top_n_terms
        )

        # MI (Mutual Information)
        mi_terms = _compute_mi(
            cluster_texts[cid], cluster_texts, cid, top_n_terms
        )

        cluster_labels[cid] = {
            "cluster_id": cid,
            "size": len(clusters[cid]),
            "n_citing_articles": len(cluster_articles[cid]),
            "tfidf_terms": tfidf_terms,
            "llr_terms": llr_terms,
            "mi_terms": mi_terms,
            "label_tfidf": "; ".join(t["term"] for t in tfidf_terms[:3]),
            "label_llr": "; ".join(t["term"] for t in llr_terms[:3]) if llr_terms else "",
        }

    return cluster_labels


def _compute_llr(
    cluster_docs: List[str],
    all_cluster_texts: Dict[int, List[str]],
    target_cid: int,
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """Compute Log-Likelihood Ratio for terms in a cluster.

    LLR measures how distinctive a term is for this cluster vs others.
    """
    # Tokenize
    def _tokenize(text: str) -> List[str]:
        return re.findall(r'\b[a-z]{3,}\b', text.lower())

    # Term frequencies in target cluster
    target_tf: Counter = Counter()
    for doc in cluster_docs:
        target_tf.update(_tokenize(doc))

    # Term frequencies in other clusters
    other_tf: Counter = Counter()
    for cid, docs in all_cluster_texts.items():
        if cid == target_cid:
            continue
        for doc in docs:
            other_tf.update(_tokenize(doc))

    # English stop words (minimal set)
    stop_words = {
        "the", "and", "for", "are", "but", "not", "you", "all",
        "can", "had", "her", "was", "one", "our", "out", "has", "have",
        "this", "that", "with", "from", "they", "been", "said", "each",
        "which", "their", "will", "other", "about", "many", "then",
        "them", "these", "some", "more", "into", "also", "than", "were",
        "study", "results", "paper", "research", "article", "based",
        "using", "used", "analysis", "data", "method", "approach",
    }

    # Compute LLR for each term
    n_target = sum(target_tf.values())
    n_other = sum(other_tf.values())

    if n_target == 0 or n_other == 0:
        return []

    llr_scores = []
    for term, count in target_tf.items():
        if term in stop_words or len(term) < 3:
            continue

        a = count  # term in target
        b = other_tf.get(term, 0)  # term in others
        c = n_target - a  # non-term in target
        d = n_other - b  # non-term in others
        n = a + b + c + d

        if a == 0 or n == 0:
            continue

        # G-test / LLR
        def _safe_log(x):
            return math.log(max(x, 1e-15))

        expected_a = (a + b) * (a + c) / n
        expected_b = (a + b) * (b + d) / n

        if expected_a <= 0 or expected_b <= 0:
            continue

        llr = 2 * (
            a * _safe_log(a / expected_a)
            + (b * _safe_log(b / expected_b) if b > 0 else 0)
        )

        llr_scores.append({"term": term, "score": round(llr, 4)})

    llr_scores.sort(key=lambda x: x["score"], reverse=True)
    return llr_scores[:top_n]


def _compute_mi(
    cluster_docs: List[str],
    all_cluster_texts: Dict[int, List[str]],
    target_cid: int,
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """Compute Mutual Information for terms in a cluster."""
    def _tokenize(text: str) -> set:
        return set(re.findall(r'\b[a-z]{3,}\b', text.lower()))

    # Document-level presence
    target_docs_tokens = [_tokenize(doc) for doc in cluster_docs]
    other_docs_tokens = []
    for cid, docs in all_cluster_texts.items():
        if cid == target_cid:
            continue
        for doc in docs:
            other_docs_tokens.append(_tokenize(doc))

    n_target = len(target_docs_tokens)
    n_other = len(other_docs_tokens)
    n_total = n_target + n_other

    if n_total == 0 or n_target == 0:
        return []

    # Count document presence
    all_terms = set()
    for tokens in target_docs_tokens + other_docs_tokens:
        all_terms.update(tokens)

    stop_words = {
        "the", "and", "for", "are", "but", "not", "you", "all",
        "can", "had", "her", "was", "one", "our", "out", "has", "have",
        "this", "that", "with", "from", "they", "been", "said", "each",
        "which", "their", "will", "other", "about", "many", "then",
        "study", "results", "paper", "research", "article", "based",
        "using", "used", "analysis", "data", "method", "approach",
    }

    mi_scores = []
    for term in all_terms:
        if term in stop_words or len(term) < 3:
            continue

        a = sum(1 for t in target_docs_tokens if term in t)
        b = sum(1 for t in other_docs_tokens if term in t)

        if a < 2:
            continue

        p_term = (a + b) / n_total
        p_cluster = n_target / n_total
        p_joint = a / n_total

        if p_term <= 0 or p_cluster <= 0 or p_joint <= 0:
            continue

        mi = math.log2(p_joint / (p_term * p_cluster))
        mi_scores.append({"term": term, "score": round(mi, 4)})

    mi_scores.sort(key=lambda x: x["score"], reverse=True)
    return mi_scores[:top_n]


# ======================================================================
# 8. KEYWORD CO-OCCURRENCE NETWORK (CiteSpace-style visualizations)
# ======================================================================

def _label_keyword_clusters(
    G_kw: nx.Graph,
    partition: dict,
) -> Dict[int, Dict[str, Any]]:
    """Label keyword clusters by most frequent keywords in each cluster.

    Unlike label_clusters() (which uses TF-IDF on citing article text),
    this directly uses the keyword nodes themselves — simpler and always
    works regardless of data availability.
    """
    if not partition:
        return {}

    clusters: Dict[int, List[str]] = defaultdict(list)
    for node, comm in partition.items():
        if node in G_kw:
            clusters[comm].append(node)

    labels: Dict[int, Dict[str, Any]] = {}
    for cid, members in clusters.items():
        # Rank by frequency (node attribute), then by degree
        ranked = sorted(
            members,
            key=lambda n: (
                G_kw.nodes[n].get("frequency", 0),
                G_kw.degree(n),
            ),
            reverse=True,
        )
        top_terms = [
            {"term": n, "score": round(float(G_kw.nodes[n].get("frequency", 1)), 1)}
            for n in ranked[:5]
        ]
        labels[cid] = {
            "cluster_id": cid,
            "size": len(members),
            "tfidf_terms": top_terms,
            "label_tfidf": "; ".join(t["term"] for t in top_terms[:3]),
            "label_llr": "",
        }
    return labels



def _build_keyword_cooccurrence_network(
    articles: List[dict],
    min_cooccurrences: int = 2,
    max_keywords: int = 200,
) -> nx.Graph:
    """Build keyword co-occurrence network from articles.

    Uses manual_tags (primary) and automatic_tags (fallback).
    Also incorporates OpenAlex concepts if available.
    """
    _empty = {"", "nan", "none", "nan ", " nan"}

    def _valid(tag):
        if not tag:
            return None
        t = str(tag).strip().lower()
        if not t or t in _empty or is_noisy_legacy_concept(t):
            return None
        return t

    # Collect keywords per article
    article_kws: List[set] = []
    kw_freq: Counter = Counter()

    for article in articles:
        kws: set = set()
        # Primary: manual_tags
        for tag in article.get("manual_tags", []):
            v = _valid(tag)
            if v:
                kws.add(v)
        # Fallback: automatic_tags
        if not kws:
            for tag in article.get("automatic_tags", []):
                v = _valid(tag)
                if v:
                    kws.add(v)
        # OpenAlex keywords (always supplement)
        for kw_obj in article.get("openalex", {}).get("keywords", []):
            kw = kw_obj.get("keyword", "") if isinstance(kw_obj, dict) else ""
            v = _valid(kw)
            if v:
                kws.add(v)
        # MeSH terms (always supplement)
        for mesh in article.get("mesh_terms", []):
            v = _valid(mesh)
            if v:
                kws.add(v)
        # Supplement with OpenAlex CWTS Leiden topics (or filtered legacy concepts as fallback)
        if len(kws) < 3:
            topics = article.get("topics") or article.get("openalex", {}).get("topics", [])
            if topics:
                for t in topics:
                    if isinstance(t, dict):
                        name = t.get("display_name") or t.get("name") or ""
                        v = _valid(name)
                        if v:
                            kws.add(v)
            else:
                for c in article.get("concepts", []):
                    if isinstance(c, dict):
                        score = c.get("score", 0) or 0
                        name = c.get("name", "")
                        v = _valid(name)
                        if score >= 0.5 and v:
                            kws.add(v)
        # primary_topic as last resort
        if len(kws) < 3:
            pt = article.get("openalex", {}).get("primary_topic", {}) or {}
            for field in ("name", "subfield"):
                val = pt.get(field, "")
                v = _valid(val)
                if v:
                    kws.add(v)
        article_kws.append(kws)
        for kw in sorted(kws):
            kw_freq[kw] += 1

    # Keep top N most frequent keywords (deterministic alphabetical tie-breaking)
    top_kws = {
        kw for kw, _ in sorted(kw_freq.items(), key=lambda x: (-x[1], x[0]))[:max_keywords]
    }

    # Build co-occurrence edges
    G = nx.Graph()
    for kws in article_kws:
        filtered = sorted(kws & top_kws)
        for i in range(len(filtered)):
            for j in range(i + 1, len(filtered)):
                if G.has_edge(filtered[i], filtered[j]):
                    G[filtered[i]][filtered[j]]["weight"] += 1
                else:
                    G.add_edge(filtered[i], filtered[j], weight=1)

    # Remove weak edges
    to_remove = [
        (u, v) for u, v, d in G.edges(data=True)
        if d["weight"] < min_cooccurrences
    ]
    G.remove_edges_from(to_remove)
    G.remove_nodes_from(list(nx.isolates(G)))

    # Node attributes
    for node in G.nodes():
        G.nodes[node]["frequency"] = kw_freq.get(node, 1)

    # Pathfinder network pruning (removes redundant edges)
    # Use conservative q=2 for dense keyword networks (only 2-hop triangles)
    n_before = G.number_of_nodes()
    G_pruned = _pathfinder_pruning(G, q=2)
    # Fallback: if too aggressive (< 30% of original nodes), skip pruning
    if G_pruned.number_of_nodes() < max(20, n_before * 0.3):
        logger.debug(
            f"Pathfinder too aggressive ({G_pruned.number_of_nodes()} nodes), "
            f"keeping original ({n_before} nodes)"
        )
    else:
        G = G_pruned

    # Keep only largest connected component for visualization clarity
    if G.number_of_nodes() > 0:
        components = sorted(nx.connected_components(G), key=len, reverse=True)
        if components:
            largest_cc = components[0]
            removed = G.number_of_nodes() - len(largest_cc)
            if removed > 0:
                G = G.subgraph(largest_cc).copy()
                logger.debug(
                    f"Kept largest component: {G.number_of_nodes()} nodes "
                    f"({removed} disconnected removed)"
                )

    return G


def _pathfinder_pruning(G: nx.Graph, q: int = -1, r: float = float("inf")) -> nx.Graph:
    """Pathfinder Network Scaling (PFNET) — CiteSpace's default pruning algorithm.

    Removes edges that violate the triangle inequality: an edge (u,v) is kept
    only if no alternative path u→...→v exists where all intermediate edges
    have weight >= the direct edge weight.

    Uses proximity (weight) as similarity — higher weight = stronger link.
    An edge is redundant if an alternative path exists through stronger links.

    Args:
        G: weighted graph (weight = co-occurrence count = similarity)
        q: path length limit (-1 = N-1 = strongest pruning)
        r: Minkowski parameter (inf = minimax metric)

    Returns:
        Pruned graph.
    """
    if G.number_of_edges() < 3:
        return G

    if q == -1:
        q = G.number_of_nodes() - 1

    # Convert to distance: dist = 1/weight (high co-occurrence = short distance)
    edges_to_remove = []
    for u, v, data in G.edges(data=True):
        w_uv = data.get("weight", 1)
        dist_uv = 1.0 / max(w_uv, 0.001)

        # Check if an alternative path exists with max-edge-dist <= dist_uv
        # (i.e., all edges on path have weight >= w_uv)
        # Use BFS/DFS limited to q hops, only through edges with weight >= w_uv
        found_alt = False
        visited = {u}
        frontier = [(u, 0)]  # (node, hops)
        while frontier:
            curr, hops = frontier.pop(0)
            if hops >= q:
                continue
            for nbr in G.neighbors(curr):
                if nbr == v and curr != u:
                    # Found alternative path through stronger/equal edges
                    found_alt = True
                    break
                if nbr not in visited and nbr != v:
                    w_edge = G[curr][nbr].get("weight", 1)
                    if w_edge >= w_uv:  # Only traverse stronger edges
                        visited.add(nbr)
                        frontier.append((nbr, hops + 1))
            if found_alt:
                break

        if found_alt:
            edges_to_remove.append((u, v))

    G_pruned = G.copy()
    G_pruned.remove_edges_from(edges_to_remove)
    G_pruned.remove_nodes_from(list(nx.isolates(G_pruned)))

    logger.debug(
        f"Pathfinder pruning: {G.number_of_edges()} → {G_pruned.number_of_edges()} edges "
        f"({len(edges_to_remove)} removed)"
    )
    return G_pruned


def _get_cluster_color_palette(n_clusters: int, dark: bool = False):
    """Generate distinct colors for clusters."""
    if dark:
        # Bright saturated colors for dark backgrounds
        base_colors = [
            "#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8",
            "#F7DC6F", "#BB8FCE", "#85C1E9", "#F8C471", "#82E0AA",
            "#F1948A", "#AED6F1", "#D7BDE2", "#A3E4D7", "#FAD7A0",
            "#A9CCE3", "#D5F5E3", "#FADBD8", "#D6EAF8", "#E8DAEF",
            "#FCF3CF", "#D1F2EB", "#FDEBD0", "#D4E6F1", "#F9E79F",
        ]
    else:
        # CiteSpace-like pastel/medium colors for light backgrounds
        base_colors = [
            "#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6",
            "#1ABC9C", "#E67E22", "#2980B9", "#27AE60", "#D35400",
            "#8E44AD", "#16A085", "#C0392B", "#2C3E50", "#F1C40F",
            "#7D3C98", "#148F77", "#CB4335", "#1F618D", "#229954",
            "#B7950B", "#6C3483", "#117A65", "#A93226", "#1A5276",
        ]
    # Ensure enough colors
    while len(base_colors) < n_clusters:
        base_colors.extend(base_colors)
    return base_colors[:max(n_clusters, 1)]


def _compute_spring_layout_by_cluster(
    G: nx.Graph, partition: dict,
    k: float = 1.0, iterations: int = 100, seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """Compute spring layout with cluster-aware positioning.

    Places cluster centroids apart first, then runs spring layout
    with stronger intra-cluster attraction.
    """
    if G.number_of_nodes() == 0:
        return {}

    # Group by cluster
    clusters: Dict[int, List[str]] = defaultdict(list)
    for node, comm in partition.items():
        if node in G:
            clusters[comm].append(node)

    n_clusters = len(clusters)
    if n_clusters == 0:
        return {}

    # Initial positions: arrange clusters in a circle
    init_pos = {}
    cluster_ids = sorted(clusters.keys())
    for idx, cid in enumerate(cluster_ids):
        angle = 2 * np.pi * idx / max(n_clusters, 1)
        cx = 3.0 * np.cos(angle)
        cy = 3.0 * np.sin(angle)
        for i, node in enumerate(clusters[cid]):
            # Small random offset within cluster
            rng = np.random.RandomState(seed + hash(node) % 10000)
            ox = rng.normal(0, 0.5)
            oy = rng.normal(0, 0.5)
            init_pos[node] = (cx + ox, cy + oy)

    # Run spring layout with initial positions
    pos = nx.spring_layout(
        G, pos=init_pos, k=k, iterations=iterations,
        seed=seed, weight="weight",
    )
    return pos


def plot_cluster_network(
    articles: List[dict],
    G: Optional[nx.Graph] = None,
    partition: Optional[dict] = None,
    cluster_labels: Optional[Dict[int, Dict]] = None,
    output_dir: Path = CITESPACE_DIR,
    dark_theme: bool = False,
    max_display_nodes: int = 300,
    max_display_edges: int = 1500,
) -> Optional[Path]:
    """CiteSpace-style cluster network visualization.

    Generates force-directed network graph with:
    - Nodes colored by cluster (Louvain community)
    - Node size proportional to frequency/degree
    - Cluster labels at centroids (with #N prefix)
    - Inter-cluster bridge edges highlighted
    - Top nodes labeled
    - Dark or light theme

    Like citespaceFig.png (dark) and citespaceFig2.png (light).
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        logger.warning("plotly not available for cluster network")
        return None

    # Build network if not provided
    if G is None:
        G = _build_keyword_cooccurrence_network(articles, min_cooccurrences=2)

    if G.number_of_nodes() < 5:
        logger.info("Network too small for cluster visualization")
        return None

    # Detect communities if not provided
    if partition is None:
        partition = _detect_communities_internal(G)
    if not partition:
        return None

    # Limit nodes for visualization performance
    if G.number_of_nodes() > max_display_nodes:
        # Keep top nodes by degree
        degree_sorted = sorted(G.degree(), key=lambda x: x[1], reverse=True)
        keep_nodes = {n for n, _ in degree_sorted[:max_display_nodes]}
        G = G.subgraph(keep_nodes).copy()
        partition = {n: c for n, c in partition.items() if n in G}

    # Layout
    pos = _compute_spring_layout_by_cluster(G, partition, k=1.2, iterations=150)
    if not pos:
        return None

    # Group by cluster
    clusters: Dict[int, List[str]] = defaultdict(list)
    for node, comm in partition.items():
        if node in G:
            clusters[comm].append(node)

    # Sort clusters by size (largest first)
    sorted_cluster_ids = sorted(clusters.keys(), key=lambda c: -len(clusters[c]))
    n_clusters = len(sorted_cluster_ids)
    colors = _get_cluster_color_palette(n_clusters, dark=dark_theme)
    cluster_color_map = {cid: colors[i] for i, cid in enumerate(sorted_cluster_ids)}

    # Get label for each cluster
    cluster_label_text = {}
    for rank, cid in enumerate(sorted_cluster_ids):
        if cluster_labels and cid in cluster_labels:
            cl = cluster_labels[cid]
            lbl = cl.get("label_tfidf", "") or cl.get("label_llr", "")
            if lbl:
                # Use first term only for cleaner labels
                lbl = lbl.split(";")[0].strip()
        else:
            # Derive label from most frequent keyword in cluster
            kw_freq: Counter = Counter()
            for node in clusters[cid]:
                kw_freq[node] = G.nodes[node].get("frequency", G.degree(node))
            top = kw_freq.most_common(1)
            lbl = top[0][0] if top else f"cluster {cid}"
        cluster_label_text[cid] = f"#{rank} {lbl}"

    # Theme settings
    if dark_theme:
        bg_color = "#0a0a0a"
        text_color = "#ffffff"
        edge_color = "rgba(100,100,100,0.15)"
        bridge_edge_color = "rgba(255,255,255,0.25)"
        node_line_color = "rgba(0,0,0,0.5)"
        label_font_color = "#ffffff"
        grid_color = "rgba(50,50,50,0.3)"
    else:
        bg_color = "#ffffff"
        text_color = "#333333"
        edge_color = "rgba(180,180,180,0.2)"
        bridge_edge_color = "rgba(200,100,150,0.35)"
        node_line_color = "rgba(255,255,255,0.8)"
        label_font_color = "#333333"
        grid_color = "rgba(200,200,200,0.3)"

    fig = go.Figure()

    # --- Draw edges ---
    # Separate intra-cluster and inter-cluster edges
    intra_edge_x, intra_edge_y = [], []
    bridge_edge_x, bridge_edge_y = [], []

    edge_count = 0
    edges_sorted = sorted(
        G.edges(data=True), key=lambda e: e[2].get("weight", 1), reverse=True
    )
    for u, v, data in edges_sorted:
        if edge_count >= max_display_edges:
            break
        if u not in pos or v not in pos:
            continue
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        is_bridge = partition.get(u) != partition.get(v)
        if is_bridge:
            # Curved inter-cluster edges
            # Quadratic bezier: midpoint offset perpendicular to line
            mx = (x0 + x1) / 2
            my = (y0 + y1) / 2
            dx = x1 - x0
            dy = y1 - y0
            length = max(math.sqrt(dx * dx + dy * dy), 0.01)
            # Perpendicular offset
            offset = length * 0.15
            cx = mx - dy / length * offset
            cy = my + dx / length * offset
            # Generate bezier points
            for t in np.linspace(0, 1, 15):
                bx = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t ** 2 * x1
                by = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t ** 2 * y1
                bridge_edge_x.append(bx)
                bridge_edge_y.append(by)
            bridge_edge_x.append(None)
            bridge_edge_y.append(None)
        else:
            intra_edge_x.extend([x0, x1, None])
            intra_edge_y.extend([y0, y1, None])
        edge_count += 1

    # Intra-cluster edges
    fig.add_trace(go.Scatter(
        x=intra_edge_x, y=intra_edge_y,
        mode="lines",
        line=dict(width=0.5, color=edge_color),
        hoverinfo="skip",
        showlegend=False,
    ))

    # Inter-cluster bridge edges
    if bridge_edge_x:
        fig.add_trace(go.Scatter(
            x=bridge_edge_x, y=bridge_edge_y,
            mode="lines",
            line=dict(width=1.5, color=bridge_edge_color),
            hoverinfo="skip",
            showlegend=False,
        ))

    # --- Draw nodes by cluster ---
    from src.utils import strength_betweenness
    betweenness = strength_betweenness(G)   # misma betweenness que pivotes y sigma

    for rank, cid in enumerate(sorted_cluster_ids):
        members = clusters[cid]
        if not members:
            continue

        xs, ys, sizes, texts, labels = [], [], [], [], []
        for node in members:
            if node not in pos:
                continue
            x, y = pos[node]
            freq = G.nodes[node].get("frequency", G.degree(node))
            bc = betweenness.get(node, 0)
            size = max(6, min(45, math.sqrt(freq) * 5 + bc * 100))

            xs.append(x)
            ys.append(y)
            sizes.append(size)
            texts.append(
                f"<b>{node}</b><br>"
                f"Frequency: {freq}<br>"
                f"Degree: {G.degree(node)}<br>"
                f"Betweenness: {bc:.4f}<br>"
                f"Cluster: #{rank}"
            )
            # Label top nodes
            if freq >= 15 or bc >= 0.05 or G.degree(node) >= 10:
                labels.append(node)
            else:
                labels.append("")

        color = cluster_color_map[cid]

        # Node markers
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="markers+text",
            marker=dict(
                size=sizes,
                color=color,
                opacity=0.85,
                line=dict(width=1, color=node_line_color),
            ),
            text=labels,
            textposition="top center",
            textfont=dict(
                size=9,
                color=color if dark_theme else label_font_color,
            ),
            hovertext=texts,
            hoverinfo="text",
            name=cluster_label_text.get(cid, f"#{rank}"),
        ))

    # --- Cluster label annotations at centroids ---
    for rank, cid in enumerate(sorted_cluster_ids):
        members = [n for n in clusters[cid] if n in pos]
        if not members:
            continue
        cx = np.mean([pos[n][0] for n in members])
        cy = np.mean([pos[n][1] for n in members])
        label = cluster_label_text.get(cid, f"#{rank}")
        color = cluster_color_map[cid]

        fig.add_annotation(
            x=cx, y=cy - 0.15,
            text=f"<b>{label}</b>",
            showarrow=False,
            font=dict(
                size=12,
                color=color if dark_theme else color,
                family="Arial Black",
            ),
            bgcolor="rgba(0,0,0,0.5)" if dark_theme else "rgba(255,255,255,0.7)",
            borderpad=2,
        )

    # --- Layout ---
    suffix = "dark" if dark_theme else "light"
    title = "Keyword Co-occurrence Cluster Network"

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16, color=text_color),
        ),
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        xaxis=dict(visible=False, scaleanchor="y"),
        yaxis=dict(visible=False),
        width=1400,
        height=1000,
        showlegend=True,
        legend=dict(
            font=dict(color=text_color, size=10),
            bgcolor="rgba(0,0,0,0.3)" if dark_theme else "rgba(255,255,255,0.8)",
            bordercolor="rgba(100,100,100,0.5)",
            borderwidth=1,
        ),
        margin=dict(l=20, r=20, t=60, b=20),
    )

    fname = f"cluster_network_{suffix}"
    output_path = output_dir / f"{fname}.html"
    fig.write_html(str(output_path))
    safe_write_plotly_png(fig, output_path.with_suffix(".png"), scale=2, log=logger)
    logger.info(f"Cluster network ({suffix}): {output_path}")
    return output_path


def plot_treering_network(
    articles: List[dict],
    G: Optional[nx.Graph] = None,
    partition: Optional[dict] = None,
    cluster_labels: Optional[Dict[int, Dict]] = None,
    bursts_by_node: Optional[Dict[str, float]] = None,
    output_dir: Path = CITESPACE_DIR,
    max_display_nodes: int = 150,
) -> Optional[Path]:
    """CiteSpace tree-ring network visualization.

    Each node is rendered as concentric rings where:
    - Each ring = one year; thickness = keyword frequency in that year
    - Ring color = temporal gradient (blue→green→yellow→red)
    - Purple outer ring = high betweenness centrality (>= 0.05)
    - Red inner glow = burst detection present
    - Node total size = overall frequency
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    if G is None:
        G = _build_keyword_cooccurrence_network(articles, min_cooccurrences=2)
    if G.number_of_nodes() < 5:
        return None
    if partition is None:
        partition = _detect_communities_internal(G)
    if not partition:
        return None

    # Limit for performance
    if G.number_of_nodes() > max_display_nodes:
        degree_sorted = sorted(G.degree(), key=lambda x: x[1], reverse=True)
        keep_nodes = {n for n, _ in degree_sorted[:max_display_nodes]}
        G = G.subgraph(keep_nodes).copy()
        partition = {n: c for n, c in partition.items() if n in G}

    # Compute layout
    pos = _compute_spring_layout_by_cluster(G, partition, k=1.5, iterations=150)
    if not pos:
        return None

    # Compute betweenness centrality
    from src.utils import strength_betweenness
    betweenness = strength_betweenness(G)   # misma betweenness que pivotes y sigma

    # Build keyword → year → count mapping
    _empty = {"", "nan", "none"}
    kw_year_freq: Dict[str, Counter] = defaultdict(Counter)
    for article in articles:
        year = article.get("publication_year")
        if not year:
            continue
        try:
            year = int(year)
        except (ValueError, TypeError):
            continue
        # Vocabulario comun del pipeline: las mismas palabras normalizadas que los nodos de G
        from src.utils import article_keywords
        kws = set(article_keywords(article))
        for kw in kws:
            if kw in G:
                kw_year_freq[kw][year] += 1

    # Year range for color mapping
    all_years = set()
    for counts in kw_year_freq.values():
        all_years.update(counts.keys())
    if not all_years:
        return None
    min_year = min(all_years)
    max_year = max(all_years)
    year_span = max(max_year - min_year, 1)

    # Temporal color gradient (blue → green → yellow → red)
    def _year_color(year: int) -> str:
        t = (year - min_year) / year_span  # 0 to 1
        if t < 0.33:
            r, g, b = 50, int(100 + 155 * t / 0.33), 200
        elif t < 0.66:
            tt = (t - 0.33) / 0.33
            r, g, b = int(50 + 200 * tt), 255, int(200 - 200 * tt)
        else:
            tt = (t - 0.66) / 0.34
            r, g, b = 255, int(255 - 200 * tt), 50
        return f"rgb({r},{g},{b})"

    fig = go.Figure()

    # Draw edges first (thin, behind nodes)
    edge_x, edge_y = [], []
    for u, v in G.edges():
        if u in pos and v in pos:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=0.3, color="rgba(150,150,150,0.15)"),
        hoverinfo="skip", showlegend=False,
    ))

    # Draw tree-ring nodes
    # For each node, draw concentric circles from outer (oldest) to inner (newest)
    sorted_years = sorted(all_years)
    n_years = len(sorted_years)

    # Collect node labels for the most important nodes
    top_labels_x, top_labels_y, top_labels_text = [], [], []

    for node in G.nodes():
        if node not in pos:
            continue
        cx, cy = pos[node]
        freq = G.nodes[node].get("frequency", 1)
        bc = betweenness.get(node, 0)
        is_burst = (bursts_by_node or {}).get(node, 0) > 0
        burst_str = (bursts_by_node or {}).get(node, 0)

        # Base radius proportional to sqrt(frequency)
        base_radius = max(0.05, min(0.35, math.sqrt(freq) * 0.04))

        year_counts = kw_year_freq.get(node, {})
        if not year_counts:
            # Simple circle if no year data
            fig.add_trace(go.Scatter(
                x=[cx], y=[cy], mode="markers",
                marker=dict(size=max(8, freq * 2), color="gray", opacity=0.5),
                hovertext=f"<b>{node}</b><br>Freq: {freq}",
                hoverinfo="text", showlegend=False,
            ))
            continue

        max_yearly = max(year_counts.values()) if year_counts else 1

        # Purple outer ring for high centrality (>= 0.05)
        if bc >= 0.05:
            theta = np.linspace(0, 2 * np.pi, 40)
            r_purple = base_radius * 1.15
            fig.add_trace(go.Scatter(
                x=cx + r_purple * np.cos(theta),
                y=cy + r_purple * np.sin(theta),
                mode="lines", fill="toself",
                fillcolor="rgba(128,0,128,0.25)",
                line=dict(color="purple", width=2),
                hoverinfo="skip", showlegend=False,
            ))

        # Draw rings from outermost (earliest year) to innermost (latest)
        for ring_idx, year in enumerate(sorted_years):
            count = year_counts.get(year, 0)
            if count == 0:
                continue
            # Ring radius: outer = earliest, inner = latest
            ring_frac = 1.0 - ring_idx / max(n_years - 1, 1)
            r = base_radius * (0.3 + 0.7 * ring_frac)
            # Thickness proportional to count
            thickness = max(1, count / max_yearly * 4)
            color = _year_color(year)

            theta = np.linspace(0, 2 * np.pi, 40)
            fig.add_trace(go.Scatter(
                x=cx + r * np.cos(theta),
                y=cy + r * np.sin(theta),
                mode="lines",
                line=dict(color=color, width=thickness),
                hoverinfo="skip", showlegend=False,
            ))

        # Red center for burst
        if is_burst:
            fig.add_trace(go.Scatter(
                x=[cx], y=[cy], mode="markers",
                marker=dict(
                    size=base_radius * 80,
                    color="rgba(255,0,0,0.4)",
                    line=dict(width=0),
                ),
                hoverinfo="skip", showlegend=False,
            ))

        # Center dot + hover
        cid = partition.get(node, 0)
        hover = (
            f"<b>{node}</b><br>"
            f"Frequency: {freq}<br>"
            f"Betweenness: {bc:.4f}<br>"
            f"Cluster: {cid}<br>"
            f"Years active: {min(year_counts.keys())}-{max(year_counts.keys())}"
        )
        if is_burst:
            hover += f"<br>Burst strength: {burst_str:.2f}"

        fig.add_trace(go.Scatter(
            x=[cx], y=[cy], mode="markers",
            marker=dict(size=4, color="white",
                        line=dict(width=0.5, color="gray")),
            hovertext=hover, hoverinfo="text", showlegend=False,
        ))

        # Label important nodes
        if freq >= 15 or bc >= 0.05 or G.degree(node) >= 10:
            top_labels_x.append(cx)
            top_labels_y.append(cy + base_radius + 0.03)
            top_labels_text.append(node)

    # Labels trace
    if top_labels_x:
        fig.add_trace(go.Scatter(
            x=top_labels_x, y=top_labels_y,
            mode="text", text=top_labels_text,
            textposition="top center",
            textfont=dict(size=9, color="#333"),
            hoverinfo="skip", showlegend=False,
        ))

    # Year color legend (annotation)
    fig.add_annotation(
        x=1.0, y=0.0, xref="paper", yref="paper",
        text=(
            f"<b>Tree-ring legend</b><br>"
            f"Ring color: <span style='color:rgb(50,100,200)'>{min_year}</span>"
            f" → <span style='color:rgb(255,55,50)'>{max_year}</span><br>"
            f"<span style='color:purple'>Purple ring</span> = high centrality<br>"
            f"<span style='color:red'>Red center</span> = burst detected"
        ),
        showarrow=False, font=dict(size=10),
        align="left", bgcolor="rgba(255,255,255,0.9)",
        bordercolor="gray", borderpad=5, borderwidth=1,
    )

    fig.update_layout(
        title="Keyword Co-occurrence Network (Tree-Ring Visualization)",
        plot_bgcolor="#fafafa",
        paper_bgcolor="#ffffff",
        xaxis=dict(visible=False, scaleanchor="y"),
        yaxis=dict(visible=False),
        width=1400, height=1000,
        showlegend=False,
        margin=dict(l=20, r=100, t=60, b=20),
    )

    output_path = output_dir / "treering_network.html"
    fig.write_html(str(output_path))
    safe_write_plotly_png(fig, output_path.with_suffix(".png"), scale=2, log=logger)
    logger.info(f"Tree-ring network: {output_path}")
    return output_path


def plot_cluster_timeline_network(
    articles: List[dict],
    G: Optional[nx.Graph] = None,
    partition: Optional[dict] = None,
    cluster_labels: Optional[Dict[int, Dict]] = None,
    output_dir: Path = CITESPACE_DIR,
) -> Optional[Path]:
    """CiteSpace-style timeline cluster view (like citespaceFig1.png).

    Nodes arranged by:
    - Y-axis: cluster (row per cluster)
    - X-axis: year (publication year / first appearance)
    - Size: frequency / citation count
    - Color: cluster
    - Edges: co-occurrence links between keywords
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    if G is None:
        G = _build_keyword_cooccurrence_network(articles, min_cooccurrences=2)
    if G.number_of_nodes() < 5:
        return None
    if partition is None:
        partition = _detect_communities_internal(G)
    if not partition:
        return None

    # Compute first-appearance year for each keyword
    kw_first_year: Dict[str, int] = {}
    kw_year_counts: Dict[str, Counter] = defaultdict(Counter)
    _empty = {"", "nan", "none"}
    for article in articles:
        year = article.get("publication_year")
        if not year:
            continue
        try:
            year = int(year)
        except (ValueError, TypeError):
            continue
        # Vocabulario comun del pipeline: las mismas palabras normalizadas que los nodos de G
        from src.utils import article_keywords
        kws = set(article_keywords(article))
        for kw in kws:
            if kw in G:
                kw_year_counts[kw][year] += 1
                if kw not in kw_first_year or year < kw_first_year[kw]:
                    kw_first_year[kw] = year

    if not kw_first_year:
        return None

    # Clusters sorted by size
    clusters: Dict[int, List[str]] = defaultdict(list)
    for node, comm in partition.items():
        if node in G:
            clusters[comm].append(node)
    sorted_cids = sorted(clusters.keys(), key=lambda c: -len(clusters[c]))
    cluster_y = {cid: i for i, cid in enumerate(sorted_cids)}

    n_clusters = len(sorted_cids)
    colors = _get_cluster_color_palette(n_clusters, dark=False)
    cluster_color_map = {cid: colors[i] for i, cid in enumerate(sorted_cids)}

    # Cluster label text
    cluster_label_text = {}
    for rank, cid in enumerate(sorted_cids):
        if cluster_labels and cid in cluster_labels:
            lbl = cluster_labels[cid].get("label_tfidf", "").split(";")[0].strip()
        else:
            kw_freq: Counter = Counter()
            for node in clusters[cid]:
                kw_freq[node] = G.nodes[node].get("frequency", 1)
            top = kw_freq.most_common(1)
            lbl = top[0][0] if top else ""
        cluster_label_text[cid] = f"#{rank} {lbl}"

    fig = go.Figure()

    # Draw edges (only within same cluster or between adjacent years)
    edge_x, edge_y = [], []
    for u, v, data in G.edges(data=True):
        if u not in kw_first_year or v not in kw_first_year:
            continue
        if u not in partition or v not in partition:
            continue
        xu = kw_first_year[u]
        xv = kw_first_year[v]
        cu = partition[u]
        cv = partition[v]
        yu = cluster_y.get(cu)
        yv = cluster_y.get(cv)
        if yu is None or yv is None:
            continue
        # Add small jitter to y for visual separation
        rng = np.random.RandomState(hash(u + v) % 100000)
        jitter_u = rng.uniform(-0.3, 0.3)
        jitter_v = rng.uniform(-0.3, 0.3)
        edge_x.extend([xu, xv, None])
        edge_y.extend([yu + jitter_u, yv + jitter_v, None])

    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=0.3, color="rgba(180,180,180,0.2)"),
        hoverinfo="skip",
        showlegend=False,
    ))

    # Draw nodes
    for rank, cid in enumerate(sorted_cids):
        members = clusters[cid]
        xs, ys, sizes, texts, labels = [], [], [], [], []
        for node in members:
            year = kw_first_year.get(node)
            if year is None:
                continue
            freq = G.nodes[node].get("frequency", 1)
            size = max(5, min(35, math.sqrt(freq) * 4))
            rng = np.random.RandomState(hash(node) % 100000)
            jitter = rng.uniform(-0.3, 0.3)

            xs.append(year)
            ys.append(cluster_y[cid] + jitter)
            sizes.append(size)
            texts.append(
                f"<b>{node}</b><br>First year: {year}<br>"
                f"Frequency: {freq}<br>Cluster: #{rank}"
            )
            if freq >= 15 or G.degree(node) >= 8:
                labels.append(node)
            else:
                labels.append("")

        if xs:
            color = cluster_color_map[cid]
            fig.add_trace(go.Scatter(
                x=xs, y=ys,
                mode="markers+text",
                marker=dict(
                    size=sizes, color=color, opacity=0.8,
                    line=dict(width=0.5, color="white"),
                ),
                text=labels,
                textposition="top center",
                textfont=dict(size=8, color="#333"),
                hovertext=texts,
                hoverinfo="text",
                name=cluster_label_text.get(cid, f"#{rank}"),
            ))

    # Year range
    all_years = sorted(set(kw_first_year.values()))
    min_year = min(all_years) if all_years else 2000
    max_year = max(all_years) if all_years else 2025

    fig.update_layout(
        title="Keyword Cluster Timeline",
        xaxis=dict(
            title="Year",
            tickmode="linear",
            dtick=2,
            range=[min_year - 1, max_year + 1],
            gridcolor="rgba(200,200,200,0.3)",
        ),
        yaxis=dict(
            title="Cluster",
            tickmode="array",
            tickvals=list(range(n_clusters)),
            ticktext=[cluster_label_text.get(cid, f"#{i}")
                      for i, cid in enumerate(sorted_cids)],
            gridcolor="rgba(200,200,200,0.3)",
        ),
        height=max(500, n_clusters * 80 + 150),
        width=1400,
        plot_bgcolor="#fafafa",
        showlegend=True,
        legend=dict(font=dict(size=9)),
        margin=dict(l=200, r=20, t=60, b=60),
    )

    output_path = output_dir / "cluster_timeline_network.html"
    fig.write_html(str(output_path))
    safe_write_plotly_png(fig, output_path.with_suffix(".png"), scale=2, log=logger)
    logger.info(f"Cluster timeline network: {output_path}")
    return output_path


def _plot_quality_metrics(
    quality: Dict[str, Any],
    cluster_labels: Optional[Dict[int, Dict]] = None,
    output_dir: Path = CITESPACE_DIR,
) -> Optional[Path]:
    """Bar chart of Modularity Q and per-cluster Silhouette S."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    q = quality.get("modularity_q", 0)
    mean_s = quality.get("mean_silhouette", 0)
    cluster_sils = quality.get("cluster_silhouettes", {})

    if not cluster_sils:
        return None

    # Sort clusters by size (largest first)
    sorted_cids = sorted(cluster_sils.keys())
    labels = []
    values = []
    for cid in sorted_cids:
        if cluster_labels and cid in cluster_labels:
            lbl = cluster_labels[cid].get("label_tfidf", "").split(";")[0].strip()
            labels.append(f"#{cid} {lbl[:25]}")
        else:
            labels.append(f"#{cid}")
        values.append(cluster_sils[cid])

    fig = go.Figure()

    # Per-cluster silhouette bars
    colors = ["#2ecc71" if v > 0.7 else "#f39c12" if v > 0.5 else "#e74c3c"
              for v in values]
    fig.add_trace(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        name="Cluster Silhouette",
        text=[f"{v:.3f}" for v in values],
        textposition="auto",
    ))

    # Threshold lines
    fig.add_hline(y=0.7, line_dash="dash", line_color="green",
                  annotation_text="S=0.7 (convincing)")
    fig.add_hline(y=0.5, line_dash="dash", line_color="orange",
                  annotation_text="S=0.5 (reasonable)")

    fig.update_layout(
        title=(
            f"Clustering Quality — Modularity Q={q:.4f}, "
            f"Mean Silhouette S={mean_s:.4f}"
        ),
        xaxis_title="Cluster",
        yaxis_title="Silhouette Score",
        yaxis=dict(range=[-0.2, 1.05]),
        height=500, width=900,
        showlegend=False,
    )

    output_path = output_dir / "quality_metrics.html"
    fig.write_html(str(output_path))
    safe_write_plotly_png(fig, output_path.with_suffix(".png"), scale=2, log=logger)
    return output_path


# ======================================================================
# 9. CITESPACE REPORT
# ======================================================================

def _generate_citespace_report(
    burst_results: Dict[str, List[Dict]],
    pivot_nodes: List[Dict],
    sigma_list: List[Dict],
    research_fronts: List[Dict],
    cluster_labels: Dict[int, Dict],
    kw_cluster_labels: Optional[Dict[int, Dict]] = None,
    quality_metrics: Optional[Dict[str, Any]] = None,
    output_dir: Path = CITESPACE_DIR,
) -> Tuple[Path, Path]:
    """Generate citespace_report.md and citespace_data.json."""

    # --- Markdown Report ---
    lines = [
        "# CiteSpace-style Analysis Report (own implementation)",
        "",
        "> **Not produced by CiteSpace.** Own implementation, in this pipeline, of the analyses that CiteSpace "
        "popularized, following their published definitions: burst detection (Kleinberg, 2002), betweenness "
        "centrality (Freeman, 1977), sigma = (centrality + 1)^burst (Chen et al., 2009), modularity Q and "
        "silhouette for cluster quality, and log-likelihood ratio (LLR) cluster labels. It has not been "
        "validated against the CiteSpace software; cite the methods, not the tool.",
        "",
        "---",
        "",
    ]

    # Network Quality Metrics
    if quality_metrics:
        q = quality_metrics.get("modularity_q", 0)
        s = quality_metrics.get("mean_silhouette", 0)
        q_interp = "significant" if q > 0.3 else "weak"
        s_interp = "convincing" if s > 0.7 else ("reasonable" if s > 0.5 else "weak")
        lines.append("## Network Quality")
        lines.append("")
        lines.append(f"- **Modularity Q** = {q:.4f} ({q_interp}; Q > 0.3 = significant)")
        lines.append(f"- **Mean Silhouette S** = {s:.4f} ({s_interp}; S > 0.7 = convincing)")
        lines.append("")
        cs = quality_metrics.get("cluster_silhouettes", {})
        if cs:
            lines.append("| Cluster | Silhouette | Quality |")
            lines.append("|---------|-----------|---------|")
            for cid, sv in sorted(cs.items()):
                q_label = "convincing" if sv > 0.7 else ("reasonable" if sv > 0.5 else "weak")
                lines.append(f"| #{cid} | {sv:.4f} | {q_label} |")
            lines.append("")

    # Burst Detection
    lines.append("## 1. Burst Detection (Kleinberg)")
    lines.append("")
    for entity_type, bursts in burst_results.items():
        lines.append(f"### {entity_type.title()} Bursts")
        lines.append("")
        if bursts:
            lines.append("| Entity | Begin | End | Duration | Level | Strength |")
            lines.append("|--------|-------|-----|----------|-------|----------|")
            for b in bursts[:20]:
                entity = b["entity"]
                if len(entity) > 50:
                    entity = entity[:47] + "..."
                lines.append(
                    f"| {entity} | {b['begin_year']} | {b['end_year']} | "
                    f"{b['duration']} | {b['level']} | {b['strength']} |"
                )
            lines.append("")
        else:
            lines.append("No significant bursts detected.")
            lines.append("")

    # Pivot Nodes
    lines.append("## 2. Pivot Nodes (Inter-cluster Bridges)")
    lines.append("")
    if pivot_nodes:
        lines.append("| Node | Cluster | Betweenness | Neighbor Clusters | Bridge Score |")
        lines.append("|------|---------|------------|-------------------|-------------|")
        for p in pivot_nodes[:15]:
            node = p["node"]
            if len(node) > 50:
                node = node[:47] + "..."
            lines.append(
                f"| {node} | {p['cluster']} | {p['betweenness']:.4f} | "
                f"{p['n_neighbor_clusters']} | {p['bridge_score']:.4f} |"
            )
        lines.append("")
    else:
        lines.append("No pivot nodes detected.")
        lines.append("")

    # Sigma Rankings
    lines.append("## 3. Sigma (Σ) Rankings")
    lines.append("")
    if sigma_list:
        lines.append("| Rank | Node | Betweenness | Burstness | Sigma |")
        lines.append("|------|------|------------|-----------|-------|")
        for i, s in enumerate(sigma_list[:20], 1):
            node = s["node"]
            if len(node) > 50:
                node = node[:47] + "..."
            lines.append(
                f"| {i} | {node} | {s['betweenness']:.4f} | "
                f"{s['burstness']:.2f} | {s['sigma']:.4f} |"
            )
        lines.append("")
    else:
        lines.append("No Sigma data available.")
        lines.append("")

    # Research Fronts
    lines.append("## 4. Research Fronts")
    lines.append("")
    if research_fronts:
        for f in research_fronts:
            yr = f.get("year_range")
            yr_str = f"{yr[0]}-{yr[1]}" if yr else "N/A"
            lines.append(
                f"### Cluster {f['cluster_id']}: {f['label']}"
            )
            lines.append(f"- Intellectual base: {f['intellectual_base_size']} references")
            lines.append(f"- Research front: {f['research_front_size']} articles")
            lines.append(f"- Year range: {yr_str}")
            lines.append(f"- Top keywords: {', '.join(f['top_keywords'][:5])}")
            lines.append("")
    else:
        lines.append("No research fronts identified.")
        lines.append("")

    # Keyword Network Clusters
    lines.append("## 5. Keyword Co-occurrence Clusters")
    lines.append("")
    if kw_cluster_labels:
        lines.append("| Cluster | Size | Top Keywords |")
        lines.append("|---------|------|-------------|")
        for cid, cl in sorted(kw_cluster_labels.items()):
            lines.append(
                f"| #{cid} | {cl['size']} | {cl['label_tfidf'][:60]} |"
            )
        lines.append("")
    else:
        lines.append("No keyword clusters available.")
        lines.append("")

    # Co-citation Cluster Labels
    lines.append("## 6. Co-citation Cluster Labels")
    lines.append("")
    if cluster_labels:
        lines.append("| Cluster | Size | TF-IDF Label | LLR Label |")
        lines.append("|---------|------|-------------|-----------|")
        for cid, cl in sorted(cluster_labels.items()):
            lines.append(
                f"| {cid} | {cl['size']} | "
                f"{cl['label_tfidf'][:40]} | "
                f"{cl.get('label_llr', '')[:40]} |"
            )
        lines.append("")

        for cid, cl in sorted(cluster_labels.items()):
            lines.append(f"### Cluster {cid} Detail")
            lines.append(f"- Size: {cl['size']} references, {cl['n_citing_articles']} citing articles")
            lines.append(f"- **TF-IDF terms:** {', '.join(t['term'] for t in cl['tfidf_terms'])}")
            if cl.get("llr_terms"):
                lines.append(f"- **LLR terms:** {', '.join(t['term'] for t in cl['llr_terms'])}")
            if cl.get("mi_terms"):
                lines.append(f"- **MI terms:** {', '.join(t['term'] for t in cl['mi_terms'])}")
            lines.append("")
    else:
        lines.append("No cluster labels available.")
        lines.append("")

    lines.append("---")
    lines.append("*Generated by the pipeline (Phase 2h): CiteSpace-style analyses, own implementation; not the CiteSpace software.*")

    report_path = output_dir / "citespace_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    # --- JSON Data ---
    data = {
        "quality_metrics": quality_metrics or {},
        "burst_detection": burst_results,
        "pivot_nodes": pivot_nodes,
        "sigma_rankings": sigma_list[:50],
        "research_fronts": research_fronts,
        "cluster_labels": {
            str(k): v for k, v in cluster_labels.items()
        },
        "keyword_clusters": {
            str(k): v for k, v in (kw_cluster_labels or {}).items()
        },
    }
    json_path = output_dir / "citespace_data.json"
    save_json(data, json_path)

    return report_path, json_path


# ======================================================================
# OPENALEX LABEL RESOLUTION FOR CITESPACE OUTPUTS
# ======================================================================

def _resolve_citespace_labels(
    burst_results: Dict[str, List[Dict]],
    pivot_nodes: List[Dict],
    sigma_list: List[Dict],
    articles: List[dict],
) -> None:
    """Resolve OpenAlex URLs in-place in CiteSpace data structures.

    Must be called AFTER all computations that use raw IDs as dict keys
    (bursts_by_node, sigma matching) and BEFORE plotting/reporting.
    """
    # Resolve reference burst entities
    for entity_type, bursts in burst_results.items():
        for b in bursts:
            if _is_openalex_url(b.get("entity", "")):
                b["entity"] = resolve_openalex_label_cached(b["entity"], articles)

    # Resolve pivot node labels
    for p in pivot_nodes:
        if _is_openalex_url(p.get("node", "")):
            p["node"] = resolve_openalex_label_cached(p["node"], articles)

    # Resolve sigma node labels
    for s in sigma_list:
        if _is_openalex_url(s.get("node", "")):
            s["node"] = resolve_openalex_label_cached(s["node"], articles)


# ======================================================================
# MAIN ENTRY POINT
# ======================================================================

def run_phase2h(
    articles: List[dict],
    networks: Optional[Dict[str, nx.Graph]] = None,
    burst_s: float = 2.0,
    burst_gamma: float = 1.0,
) -> Dict[str, Any]:
    """Entry point for Phase 2h: CiteSpace Analysis Suite.

    Agnostic implementation: works with any bibliographic data that has
    keywords/tags, titles, abstracts, and publication years. Does NOT
    require external API calls or CiteSpace software.

    Builds two networks:
    - Keyword co-occurrence (always available from tags/keywords)
    - Co-citation (if referenced_works data exists)

    Args:
        articles: enriched article list
        networks: pre-built networks from Phase 3 (optional)
        burst_s: Kleinberg state transition cost
        burst_gamma: Kleinberg difficulty parameter

    Returns:
        Summary dict with counts and paths.
    """
    logger.info("=" * 60)
    logger.info("Fase 2h: CiteSpace Analysis Suite")
    logger.info("=" * 60)

    output_dir = CITESPACE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    TOTAL_STEPS = 13
    results: Dict[str, Any] = {
        "n_bursts": 0,
        "n_pivot_nodes": 0,
        "n_clusters": 0,
        "n_kw_clusters": 0,
        "modularity_q": 0.0,
        "mean_silhouette": 0.0,
        "charts": [],
    }

    # ------------------------------------------------------------------
    # 1. Burst Detection (all entity types)
    # ------------------------------------------------------------------
    logger.info(f"1/{TOTAL_STEPS} Burst Detection (Kleinberg)...")
    burst_results: Dict[str, List[Dict]] = {}
    total_bursts = 0

    for entity_type in ("keywords", "authors", "references", "journals"):
        bursts = detect_bursts(
            articles, entity_type=entity_type, s=burst_s, gamma=burst_gamma
        )
        burst_results[entity_type] = bursts
        total_bursts += len(bursts)

        # Defer reference burst plotting until after labels are resolved
        if entity_type != "references":
            chart = plot_burst_timeline(bursts, entity_type, output_dir)
            if chart:
                results["charts"].append(str(chart))

        logger.info(
            "  %s: %d bursts detected", entity_type, len(bursts)
        )

    results["n_bursts"] = total_bursts

    # ------------------------------------------------------------------
    # Build keyword co-occurrence network (agnostic, always available)
    # ------------------------------------------------------------------
    logger.info("Building keyword co-occurrence network...")
    G_kw = _build_keyword_cooccurrence_network(
        articles, min_cooccurrences=2, max_keywords=200
    )
    kw_partition = _detect_communities_internal(G_kw)
    n_kw_clusters = len(set(kw_partition.values())) if kw_partition else 0
    results["n_kw_clusters"] = n_kw_clusters
    logger.info(
        "  Keyword network: %d nodes, %d edges, %d clusters",
        G_kw.number_of_nodes(), G_kw.number_of_edges(), n_kw_clusters,
    )

    # ------------------------------------------------------------------
    # Get or build co-citation network (if referenced_works available)
    # ------------------------------------------------------------------
    G_cocit = None
    if networks and "cocitation" in networks:
        G_cocit = networks["cocitation"].copy()
        valid_nodes = {n for comp in nx.connected_components(G_cocit) if len(comp) >= 4 for n in comp}
        if len(valid_nodes) >= 10:
            G_cocit = G_cocit.subgraph(valid_nodes).copy()
        logger.info("Using pre-built co-citation network from Phase 3 (%d nodes in clusters >= 4)", G_cocit.number_of_nodes())
    else:
        logger.info("Building co-citation network internally...")
        G_cocit = _build_cocitation_network_internal(articles, min_cocitations=2)
        logger.info(
            "  Co-citation network: %d nodes, %d edges",
            G_cocit.number_of_nodes(),
            G_cocit.number_of_edges(),
        )

    # Use keyword network as primary if co-citation is too sparse
    G_primary = G_cocit if G_cocit.number_of_nodes() >= 10 else G_kw
    primary_type = "cocitation" if G_primary is G_cocit else "keyword"
    logger.info(f"  Primary network for analyses: {primary_type}")

    partition = _detect_communities_internal(G_primary)
    n_clusters = len(set(partition.values())) if partition else 0
    results["n_clusters"] = n_clusters

    # Compute Modularity Q and Silhouette S on primary network (G_primary) and keyword network (G_kw)
    primary_quality = compute_modularity_silhouette(G_primary, partition)
    kw_quality = compute_modularity_silhouette(G_kw, kw_partition)
    primary_quality["network_type"] = primary_type
    primary_quality["keyword_network_quality"] = {
        "modularity_q": kw_quality["modularity_q"],
        "mean_silhouette": kw_quality["mean_silhouette"],
    }
    results["modularity_q"] = primary_quality["modularity_q"]
    results["mean_silhouette"] = primary_quality["mean_silhouette"]
    results["keyword_modularity_q"] = kw_quality["modularity_q"]
    results["keyword_mean_silhouette"] = kw_quality["mean_silhouette"]
    q = primary_quality["modularity_q"]
    s = primary_quality["mean_silhouette"]
    q_interp = "significant" if q > 0.3 else "weak"
    s_interp = "convincing" if s > 0.7 else ("reasonable" if s > 0.5 else "weak")
    logger.info(
        f"  Primary ({primary_type}) network quality: Modularity Q={q:.4f} ({q_interp}), "
        f"Silhouette S={s:.4f} ({s_interp})"
    )
    for cid, cs in sorted(primary_quality["cluster_silhouettes"].items()):
        logger.info(f"    Cluster {cid}: silhouette={cs:.4f}")

    # ------------------------------------------------------------------
    # 2. Timeline Visualization
    # ------------------------------------------------------------------
    logger.info(f"2/{TOTAL_STEPS} Co-citation Timeline...")
    chart = plot_cocitation_timeline(articles, G_cocit, output_dir)
    if chart:
        results["charts"].append(str(chart))

    # ------------------------------------------------------------------
    # 3. Timezone View
    # ------------------------------------------------------------------
    logger.info(f"3/{TOTAL_STEPS} Timezone View...")
    chart = plot_timezone_view(articles, G_cocit, output_dir)
    if chart:
        results["charts"].append(str(chart))

    # ------------------------------------------------------------------
    # 4. Pivot Node Detection
    # ------------------------------------------------------------------
    logger.info(f"4/{TOTAL_STEPS} Pivot Node Detection...")
    pivot_nodes = detect_pivot_nodes(G_primary, partition)
    results["n_pivot_nodes"] = len(pivot_nodes)
    logger.info("  %d pivot nodes detected", len(pivot_nodes))

    # ------------------------------------------------------------------
    # 5. Sigma Metric
    # ------------------------------------------------------------------
    logger.info(f"5/{TOTAL_STEPS} Sigma (Σ) Metric...")
    # Build burst-by-node mapping
    # Use keyword bursts for keyword network, reference bursts for cocitation
    if primary_type == "keyword":
        burst_source = burst_results.get("keywords", [])
    else:
        burst_source = burst_results.get("references", [])
    bursts_by_node: Dict[str, float] = {}
    for b in burst_source:
        entity = b["entity"]
        if entity in bursts_by_node:
            bursts_by_node[entity] = max(bursts_by_node[entity], b["strength"])
        else:
            bursts_by_node[entity] = b["strength"]

    sigma_list = compute_sigma(G_primary, bursts_by_node)

    # ------------------------------------------------------------------
    # Resolve OpenAlex URLs → human-readable labels (AFTER all computation,
    # BEFORE any plotting/reporting that displays node labels)
    # ------------------------------------------------------------------
    _resolve_citespace_labels(burst_results, pivot_nodes, sigma_list, articles)

    # Now plot reference bursts (deferred so labels are resolved)
    ref_bursts = burst_results.get("references", [])
    if ref_bursts:
        chart = plot_burst_timeline(ref_bursts, "references", output_dir)
        if chart:
            results["charts"].append(str(chart))

    # Re-plot pivot nodes with resolved labels
    chart = plot_pivot_nodes(pivot_nodes, output_dir)
    if chart:
        results["charts"].append(str(chart))

    chart = plot_sigma_scatter(sigma_list, output_dir)
    if chart:
        results["charts"].append(str(chart))

    # ------------------------------------------------------------------
    # 6. Research Front Identification
    # ------------------------------------------------------------------
    logger.info(f"6/{TOTAL_STEPS} Research Front Identification...")
    fronts = identify_research_fronts(articles, G_cocit, partition)
    logger.info("  %d research fronts identified", len(fronts))

    chart = plot_research_fronts(fronts, output_dir)
    if chart:
        results["charts"].append(str(chart))

    # ------------------------------------------------------------------
    # 7. Cluster Labeling
    # ------------------------------------------------------------------
    logger.info(f"7/{TOTAL_STEPS} Cluster Labeling (TF-IDF / LLR / MI)...")
    cluster_labels = label_clusters(articles, G_cocit, partition)
    logger.info("  %d clusters labeled", len(cluster_labels))

    # Also label keyword clusters (used for network visualizations)
    kw_cluster_labels = _label_keyword_clusters(G_kw, kw_partition)

    # ------------------------------------------------------------------
    # 8. Cluster Network — Light theme (like CiteSpace cluster view)
    # ------------------------------------------------------------------
    logger.info(f"8/{TOTAL_STEPS} Cluster Network (light theme)...")
    chart = plot_cluster_network(
        articles, G=G_kw, partition=kw_partition,
        cluster_labels=kw_cluster_labels,
        output_dir=output_dir, dark_theme=False,
    )
    if chart:
        results["charts"].append(str(chart))

    # ------------------------------------------------------------------
    # 9. Cluster Network — Dark theme (CiteSpace landscape)
    # ------------------------------------------------------------------
    logger.info(f"9/{TOTAL_STEPS} Cluster Network (dark theme)...")
    chart = plot_cluster_network(
        articles, G=G_kw, partition=kw_partition,
        cluster_labels=kw_cluster_labels,
        output_dir=output_dir, dark_theme=True,
    )
    if chart:
        results["charts"].append(str(chart))

    # ------------------------------------------------------------------
    # 10. Cluster Timeline Network
    # ------------------------------------------------------------------
    logger.info(f"10/{TOTAL_STEPS} Cluster Timeline Network...")
    chart = plot_cluster_timeline_network(
        articles, G=G_kw, partition=kw_partition,
        cluster_labels=kw_cluster_labels,
        output_dir=output_dir,
    )
    if chart:
        results["charts"].append(str(chart))

    # ------------------------------------------------------------------
    # 11. Tree-Ring Network (signature CiteSpace visualization)
    # ------------------------------------------------------------------
    logger.info(f"11/{TOTAL_STEPS} Tree-Ring Network...")
    # Build keyword bursts mapping for tree-ring red center
    kw_bursts_map: Dict[str, float] = {}
    for b in burst_results.get("keywords", []):
        entity = b["entity"]
        if entity in kw_bursts_map:
            kw_bursts_map[entity] = max(kw_bursts_map[entity], b["strength"])
        else:
            kw_bursts_map[entity] = b["strength"]
    chart = plot_treering_network(
        articles, G=G_kw, partition=kw_partition,
        cluster_labels=kw_cluster_labels,
        bursts_by_node=kw_bursts_map,
        output_dir=output_dir,
    )
    if chart:
        results["charts"].append(str(chart))

    # ------------------------------------------------------------------
    # 12. Modularity & Silhouette Summary Chart
    # ------------------------------------------------------------------
    logger.info(f"12/{TOTAL_STEPS} Quality Metrics Chart...")
    active_labels = cluster_labels if (primary_type == "cocitation" and cluster_labels) else kw_cluster_labels
    chart = _plot_quality_metrics(primary_quality, active_labels, output_dir)
    if chart:
        results["charts"].append(str(chart))

    # ------------------------------------------------------------------
    # 13. Report Generation
    # ------------------------------------------------------------------
    logger.info(f"13/{TOTAL_STEPS} Report Generation...")
    report_path, json_path = _generate_citespace_report(
        burst_results=burst_results,
        pivot_nodes=pivot_nodes,
        sigma_list=sigma_list,
        research_fronts=fronts,
        cluster_labels=cluster_labels,
        kw_cluster_labels=kw_cluster_labels,
        quality_metrics=primary_quality,
        output_dir=output_dir,
    )
    results["report_path"] = str(report_path)
    results["json_path"] = str(json_path)

    # Summary
    logger.info("-" * 40)
    logger.info("CiteSpace Analysis Summary:")
    logger.info("  Total bursts: %d", total_bursts)
    for et, bs in burst_results.items():
        logger.info("    %s: %d", et, len(bs))
    logger.info("  Keyword clusters: %d (%d nodes)",
                n_kw_clusters, G_kw.number_of_nodes())
    logger.info("  Primary (%s) Modularity Q: %.4f, Mean Silhouette S: %.4f",
                primary_type, primary_quality["modularity_q"], primary_quality["mean_silhouette"])
    logger.info("  Co-citation clusters: %d", n_clusters)
    logger.info("  Pivot nodes: %d", len(pivot_nodes))
    logger.info("  Research fronts: %d", len(fronts))
    logger.info("  Labeled clusters: %d", len(cluster_labels))
    logger.info("  Charts generated: %d", len(results["charts"]))
    logger.info("  Report: %s", report_path)
    logger.info("-" * 40)

    return results

run_phase2i = run_phase2h
