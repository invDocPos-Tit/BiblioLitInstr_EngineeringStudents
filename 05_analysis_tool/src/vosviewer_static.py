"""Generate high-impact static images replicating VOSviewer's 3 visualization types.

Produces PNG (300 DPI) and SVG for each network:
  1. Network visualization  - nodes coloured by cluster
  2. Overlay visualization  - nodes coloured by score (year / citations)
  3. Density visualization  - kernel-density heatmap
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from config import FIGURES_DIR

logger = logging.getLogger("bibliometrico.vosviewer_static")

# ---------------------------------------------------------------------------
# VOSviewer colour palette (20 cluster colours)
# ---------------------------------------------------------------------------
_VOS_CLUSTER_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#ff7f00", "#984ea3",
    "#a65628", "#f781bf", "#66c2a5", "#fc8d62",
    "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f", "#e5c494",
    "#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#999999", "#b3b3b3",
]

_VOS_OVERLAY_CMAP = LinearSegmentedColormap.from_list(
    "vos_overlay",
    ["#0000ff", "#00cccc", "#00ff00", "#ccff00", "#ffff00", "#ff6600"],
    N=256,
)

OUTPUT_SUBDIR = "vosviewer_static"

# ---------------------------------------------------------------------------
# layout helpers
# ---------------------------------------------------------------------------

def _compute_layout(G: nx.Graph) -> dict:
    """Reuse the community-aware layout from network_analysis when available."""
    try:
        from src.network_analysis import _community_layout
        communities = nx.get_node_attributes(G, "community")
        return _community_layout(G, communities)
    except ImportError:
        pass

    # Fallback: spring layout
    k = 3.0 / max(np.sqrt(G.number_of_nodes()), 1)
    return nx.spring_layout(G, k=k, iterations=80, seed=42)


def _compute_node_sizes(G: nx.Graph, min_size: float = 20, max_size: float = 500) -> dict:
    """Scale node sizes using sqrt of weighted degree."""
    degrees = dict(G.degree(weight="weight"))
    max_deg = max(degrees.values()) if degrees else 1
    return {
        n: min_size + (max_size - min_size) * np.sqrt(degrees[n] / max_deg)
        for n in G.nodes()
    }


def _filter_top_edges(G: nx.Graph, fraction: float = 0.25) -> list:
    """Return only the top-fraction strongest edges."""
    edges = list(G.edges(data=True))
    if not edges:
        return []
    weights = [d.get("weight", 1) for _, _, d in edges]
    threshold = np.percentile(weights, (1 - fraction) * 100)
    return [(u, v, d) for u, v, d in edges if d.get("weight", 1) >= threshold]


# ---------------------------------------------------------------------------
# smart labels
# ---------------------------------------------------------------------------

def _draw_smart_labels(
    ax, pos, degrees, node_labels, n_labels: int = 40,
    color: str = "#222222", bg_color: str = "white", bg_alpha: float = 0.75,
    communities: dict = None,
):
    """Draw labels for the most important nodes, distributed across clusters."""
    max_deg = max(degrees.values()) if degrees else 1

    if communities:
        comm_nodes = {}
        for n, c in communities.items():
            if n in degrees:
                comm_nodes.setdefault(c, []).append(n)

        n_comms = len(comm_nodes)
        min_per = min(2, n_labels // max(n_comms, 1))
        remaining = n_labels - min_per * n_comms
        total = sum(len(ns) for ns in comm_nodes.values())

        top_nodes = []
        for c, nodes in comm_nodes.items():
            extra = max(0, int(remaining * len(nodes) / max(total, 1)))
            k = min(len(nodes), min_per + extra)
            sorted_n = sorted(nodes, key=lambda n: degrees.get(n, 0), reverse=True)
            top_nodes.extend(sorted_n[:k])

        if len(top_nodes) < n_labels:
            used = set(top_nodes)
            rest = sorted(
                [n for n in degrees if n not in used],
                key=lambda n: degrees[n], reverse=True,
            )
            top_nodes.extend(rest[:n_labels - len(top_nodes)])
    else:
        top_nodes = sorted(degrees, key=lambda n: degrees[n], reverse=True)[:n_labels]

    try:
        from adjustText import adjust_text

        texts = []
        for n in top_nodes:
            if n not in pos:
                continue
            d = degrees.get(n, 0)
            lbl = node_labels.get(n, str(n))[:36]
            sz = 7.5 + 4.5 * (d / max_deg)
            t = ax.text(
                pos[n][0], pos[n][1], lbl,
                fontsize=min(sz, 12), fontweight="bold",
                ha="center", va="center", zorder=5, color=color,
                bbox=dict(boxstyle="round,pad=0.2", facecolor=bg_color,
                          edgecolor="none", alpha=bg_alpha),
            )
            texts.append(t)
        if texts:
            adjust_text(
                texts, ax=ax,
                arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.5),
                expand=(1.3, 1.5), force_text=(0.8, 1.0),
            )
    except ImportError:
        for n in top_nodes:
            if n not in pos:
                continue
            d = degrees.get(n, 0)
            lbl = node_labels.get(n, str(n))[:36]
            sz = 7.5 + 4.5 * (d / max_deg)
            ax.text(
                pos[n][0], pos[n][1], lbl,
                fontsize=min(sz, 12), fontweight="bold",
                ha="center", va="center", zorder=5, color=color,
                bbox=dict(boxstyle="round,pad=0.2", facecolor=bg_color,
                          edgecolor="none", alpha=bg_alpha),
            )


# ---------------------------------------------------------------------------
# score determination
# ---------------------------------------------------------------------------

def _get_score_for_network(G: nx.Graph, name: str, articles: list[dict]) -> tuple:
    """Return (score_dict, score_label) for overlay colouring.

    Returns:
        score_dict: {node: float} or None
        score_label: human-readable label for color bar
    """
    # Networks with explicit year attribute
    if name == "bibliographic_coupling":
        years = nx.get_node_attributes(G, "year")
        valid = {n: y for n, y in years.items() if y and y > 0}
        if valid:
            return valid, "Publication Year"

    # Keyword network: average publication year
    if name == "keyword_cooccurrence":
        from src.vosviewer_export import _compute_keyword_scores
        kw_scores = _compute_keyword_scores(articles)
        score = {}
        for n in G.nodes():
            if n in kw_scores and kw_scores[n]["avg_year"] > 0:
                score[n] = kw_scores[n]["avg_year"]
        if score:
            return score, "Avg. Publication Year"

    # Co-authorship / coupling: compute avg year from articles
    if name.startswith("coauthorship_") or name.startswith("coupling_"):
        level = name.split("_", 1)[1]
        from src.vosviewer_export import _compute_entity_scores
        ent_scores = _compute_entity_scores(articles, level)
        score = {}
        for n in G.nodes():
            if n in ent_scores and ent_scores[n]["avg_year"] > 0:
                score[n] = ent_scores[n]["avg_year"]
        if score:
            return score, "Avg. Publication Year"

    # Cocitation networks: use cocitation count as score
    if "cocitation" in name:
        counts = nx.get_node_attributes(G, "cocitation_count")
        valid = {n: c for n, c in counts.items() if c and c > 0}
        if valid:
            return valid, "Co-citation Count"

    return None, ""


# ---------------------------------------------------------------------------
# 1. Network Visualization
# ---------------------------------------------------------------------------

def render_network_visualization(
    G: nx.Graph, name: str, output_dir: Path,
    pos: dict, node_sizes: dict,
) -> None:
    """Cluster-coloured network visualization."""
    communities = nx.get_node_attributes(G, "community")
    node_labels = nx.get_node_attributes(G, "label")
    degrees = dict(G.degree(weight="weight"))
    max_deg = max(degrees.values()) if degrees else 1

    # Node colours by cluster (1-based cluster IDs)
    node_colors = {
        n: _VOS_CLUSTER_COLORS[max(0, communities.get(n, 1) - 1) % len(_VOS_CLUSTER_COLORS)]
        for n in G.nodes()
    }

    # Edges
    vis_edges = _filter_top_edges(G, fraction=0.25)
    edge_weights = [d.get("weight", 1) for _, _, d in vis_edges]
    max_ew = max(edge_weights) if edge_weights else 1

    fig, ax = plt.subplots(figsize=(16, 12))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fafafa")

    for u, v, d in vis_edges:
        w = d.get("weight", 1)
        alpha = 0.06 + 0.3 * (w / max_ew)
        lw = 0.2 + 2.0 * (w / max_ew)
        cu = node_colors.get(u, "#999999")
        ax.plot(
            [pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
            color=cu, alpha=alpha * 0.5, linewidth=lw, zorder=1,
        )

    for n in G.nodes():
        ax.scatter(
            pos[n][0], pos[n][1], s=node_sizes[n], c=node_colors[n],
            edgecolors="white", linewidths=0.6, alpha=0.85, zorder=3,
        )

    ax.margins(x=0.14, y=0.12)
    n_labels = min(40, max(15, G.number_of_nodes() // 4))
    _draw_smart_labels(ax, pos, degrees, node_labels, n_labels=n_labels,
                       communities=communities)

    # Legend sorted by cluster size (excluding singletons when multi-node clusters exist)
    if communities:
        from collections import Counter as _Cnt
        comm_counts = _Cnt(communities.values())
        multi_comms = [c for c, cnt in sorted(comm_counts.items(), key=lambda x: (-x[1], x[0])) if cnt >= 2]
        display_comms = multi_comms[:12] if multi_comms else [c for c, _ in comm_counts.most_common(12)]
        handles = []
        for c in display_comms:
            color = _VOS_CLUSTER_COLORS[max(0, c - 1) % len(_VOS_CLUSTER_COLORS)]
            count = comm_counts[c]
            patch = plt.Line2D(
                [0], [0], marker="o", color="w", markerfacecolor=color,
                markersize=9, label=f"Cluster {c} (n={count})",
            )
            handles.append(patch)
        ax.legend(
            handles=handles, loc="upper left", bbox_to_anchor=(-0.03, 1.02),
            fontsize=8, framealpha=0.92, title="Communities", title_fontsize=9.5,
            borderpad=0.9, handletextpad=0.5,
        )

    n_clusters = len([c for c, cnt in _Cnt(communities.values()).items() if cnt >= 2]) if communities else "?"
    if n_clusters == 0 and communities:
        n_clusters = len(set(communities.values()))
    ax.set_title(
        f"{name.replace('_', ' ').title()}\n"
        f"{G.number_of_nodes()} items · {G.number_of_edges()} links · {n_clusters} clusters",
        fontsize=15, fontweight="bold", pad=18,
    )
    ax.axis("off")
    plt.tight_layout(pad=2)

    base = f"{name}_vos_network"
    fig.savefig(output_dir / f"{base}.png", dpi=300, facecolor="white")
    fig.savefig(output_dir / f"{base}.svg", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved {base}.png + .svg")


# ---------------------------------------------------------------------------
# 2. Overlay Visualization
# ---------------------------------------------------------------------------

def render_overlay_visualization(
    G: nx.Graph, name: str, output_dir: Path,
    pos: dict, node_sizes: dict,
    score_dict: dict, score_label: str,
) -> None:
    """Overlay visualization with continuous colour mapping."""
    communities = nx.get_node_attributes(G, "community")
    node_labels = nx.get_node_attributes(G, "label")
    degrees = dict(G.degree(weight="weight"))

    nodes_list = list(G.nodes())
    score_values = np.array([score_dict.get(n, np.nan) for n in nodes_list])
    valid_mask = ~np.isnan(score_values)
    if valid_mask.sum() == 0:
        logger.info(f"No valid scores for overlay: {name}")
        return

    vmin = np.nanmin(score_values[valid_mask])
    vmax = np.nanmax(score_values[valid_mask])
    if vmin == vmax:
        vmax = vmin + 1

    vis_edges = _filter_top_edges(G, fraction=0.25)
    edge_weights = [d.get("weight", 1) for _, _, d in vis_edges]
    max_ew = max(edge_weights) if edge_weights else 1

    fig, ax = plt.subplots(figsize=(16, 12))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fafafa")

    for u, v, d in vis_edges:
        w = d.get("weight", 1)
        alpha = 0.04 + 0.2 * (w / max_ew)
        lw = 0.2 + 1.5 * (w / max_ew)
        ax.plot(
            [pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
            color="#cccccc", alpha=alpha, linewidth=lw, zorder=1,
        )

    # Nodes with score
    xs = [pos[n][0] for n in nodes_list]
    ys = [pos[n][1] for n in nodes_list]
    sizes = [node_sizes[n] for n in nodes_list]

    # Grey nodes for missing scores
    for i, n in enumerate(nodes_list):
        if not valid_mask[i]:
            ax.scatter(
                xs[i], ys[i], s=sizes[i], c="#cccccc",
                edgecolors="white", linewidths=0.6, alpha=0.5, zorder=2,
            )

    # Coloured nodes
    valid_xs = [xs[i] for i in range(len(nodes_list)) if valid_mask[i]]
    valid_ys = [ys[i] for i in range(len(nodes_list)) if valid_mask[i]]
    valid_sizes = [sizes[i] for i in range(len(nodes_list)) if valid_mask[i]]
    valid_scores = score_values[valid_mask]

    sc = ax.scatter(
        valid_xs, valid_ys, s=valid_sizes, c=valid_scores,
        cmap=_VOS_OVERLAY_CMAP, vmin=vmin, vmax=vmax,
        edgecolors="white", linewidths=0.6, alpha=0.9, zorder=3,
    )

    cbar = plt.colorbar(sc, ax=ax, shrink=0.45, aspect=30, pad=0.02)
    cbar.set_label(score_label, fontsize=11, fontweight="bold")
    cbar.ax.tick_params(labelsize=9)

    n_labels = min(35, max(12, G.number_of_nodes() // 5))
    _draw_smart_labels(ax, pos, degrees, node_labels, n_labels=n_labels,
                       color="#333333", communities=communities)

    ax.set_title(
        f"{name.replace('_', ' ').title()} — Overlay by {score_label}\n"
        f"{G.number_of_nodes()} items · {G.number_of_edges()} links",
        fontsize=15, fontweight="bold", pad=18,
    )
    ax.axis("off")
    plt.tight_layout(pad=2)

    base = f"{name}_vos_overlay"
    fig.savefig(output_dir / f"{base}.png", dpi=300, facecolor="white")
    fig.savefig(output_dir / f"{base}.svg", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved {base}.png + .svg")


# ---------------------------------------------------------------------------
# 3. Density Visualization
# ---------------------------------------------------------------------------

def render_density_visualization(
    G: nx.Graph, name: str, output_dir: Path, pos: dict,
) -> None:
    """Kernel-density heatmap visualization."""
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        logger.warning("scipy not available, skipping density visualization")
        return

    node_labels = nx.get_node_attributes(G, "label")
    communities = nx.get_node_attributes(G, "community")
    degrees = dict(G.degree(weight="weight"))

    xs = np.array([pos[n][0] for n in G.nodes()])
    ys = np.array([pos[n][1] for n in G.nodes()])
    ws = np.array([degrees[n] for n in G.nodes()], dtype=float)
    n_nodes = G.number_of_nodes()

    # Build density grid (adaptive resolution for performance)
    pad = 0.5
    x_min, x_max = xs.min() - pad, xs.max() + pad
    y_min, y_max = ys.min() - pad, ys.max() + pad
    grid_res = min(500, 150 + n_nodes // 3)
    density = np.zeros((grid_res, grid_res))

    for x, y, w in zip(xs, ys, ws):
        ix = int((x - x_min) / (x_max - x_min) * (grid_res - 1))
        iy = int((y - y_min) / (y_max - y_min) * (grid_res - 1))
        ix = min(max(ix, 0), grid_res - 1)
        iy = min(max(iy, 0), grid_res - 1)
        density[iy, ix] += w

    # Adaptive sigma
    sigma = grid_res / (6 + n_nodes / 40)
    density = gaussian_filter(density, sigma=sigma)

    fig, ax = plt.subplots(figsize=(16, 12))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    im = ax.imshow(
        density, origin="lower",
        extent=[x_min, x_max, y_min, y_max],
        cmap="viridis", aspect="auto", interpolation="bicubic",
    )

    # Labels
    n_labels = min(30, max(10, n_nodes // 5))
    _draw_smart_labels(
        ax, pos, degrees, node_labels, n_labels=n_labels,
        color="white", bg_color="black", bg_alpha=0.5,
        communities=communities,
    )

    cbar = plt.colorbar(im, ax=ax, shrink=0.4, aspect=30, pad=0.02)
    cbar.set_label("Item Density", fontsize=11, fontweight="bold")
    cbar.ax.tick_params(labelsize=8)

    ax.set_title(
        f"{name.replace('_', ' ').title()} — Density Visualization\n"
        f"{n_nodes} items",
        fontsize=15, fontweight="bold", pad=18,
    )
    ax.axis("off")
    plt.tight_layout(pad=2)

    base = f"{name}_vos_density"
    fig.savefig(output_dir / f"{base}.png", dpi=300, facecolor="white")
    fig.savefig(output_dir / f"{base}.svg", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved {base}.png + .svg")


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------

def run_vosviewer_static(articles: list[dict], networks: dict) -> None:
    """Generate high-impact static visualizations for all networks.

    Args:
        articles: List of enriched article dicts.
        networks: Dict mapping network name -> nx.Graph.
    """
    output_dir = FIGURES_DIR / OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 50)
    logger.info("VOSviewer Static Visualizations (300 DPI)")
    logger.info("=" * 50)

    for name, G in networks.items():
        if G.number_of_nodes() < 3:
            logger.info(f"Skipping {name}: too few nodes ({G.number_of_nodes()})")
            continue

        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        logger.info(f"Rendering {name} ({n_nodes} nodes, {n_edges} edges)")

        # Filter out isolated nodes (degree == 0) for cleaner static visualization if connected nodes >= 3
        connected_nodes = [n for n, d in G.degree() if d > 0]
        if len(connected_nodes) >= 3 and len(connected_nodes) < G.number_of_nodes():
            G = G.subgraph(connected_nodes).copy()
            n_nodes = G.number_of_nodes()

        # For very large networks, filter to top nodes for visualization
        if n_nodes > 200:
            deg = dict(G.degree(weight="weight"))
            n_keep = min(150, n_nodes)
            thr = sorted(deg.values(), reverse=True)[min(n_keep - 1, len(deg) - 1)]
            top_nodes = [n for n, d in deg.items() if d >= thr][:n_keep]
            G = G.subgraph(top_nodes).copy()
            logger.info(f"  Filtered for visualization: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        try:
            from src.network_analysis import detect_communities
            detect_communities(G)
        except Exception:
            pass

        try:
            pos = _compute_layout(G)
            node_sizes = _compute_node_sizes(G)

            # 1. Network visualization
            render_network_visualization(G, name, output_dir, pos, node_sizes)

            # 2. Overlay visualization
            score_dict, score_label = _get_score_for_network(G, name, articles)
            if score_dict:
                render_overlay_visualization(
                    G, name, output_dir, pos, node_sizes, score_dict, score_label,
                )
            else:
                logger.info(f"No score data for overlay: {name}")

            # 3. Density visualization
            render_density_visualization(G, name, output_dir, pos)

        except Exception as e:
            logger.error(f"Failed to render {name}: {e}")

    logger.info(f"Static visualizations complete -> {output_dir}")
