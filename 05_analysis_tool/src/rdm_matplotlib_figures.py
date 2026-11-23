"""Publication-quality matplotlib figures for Phase 6 (RDM).

Generates 5 high-impact static figures in English (400 DPI) inspired by
VOSviewer, CiteSpace, and Callon strategic diagram styles.

Figures:
  1. Thematic Overlay Map — VOSviewer-style bubble clusters with keyword lists
  2. Strategic Diagram (Callon) — Multi-panel four-quadrant thematic evolution
  3. Country Collaboration Network — Gradient-colored network graph
  4. Thematic Landscape 2D — Scatter with convex hull cluster regions
  5. CiteSpace Timeline — Horizontal timeline with cluster bands and Bezier arcs
"""
from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap, to_rgba
import numpy as np

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

logger = logging.getLogger("bibliometrico.rdm_mpl")

# ---------------------------------------------------------------------------
# Global styling
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Georgia"],
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

FIGURE_DPI = 400
HULL_ALPHA = 0.15
BUBBLE_EDGE_ALPHA = 0.4

CLUSTER_PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabebe", "#469990",
    "#e6beff", "#9A6324", "#fffac8", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#a9a9a9", "#e41a1c",
]


def _get_cluster_color(cid: int) -> str:
    return CLUSTER_PALETTE[cid % len(CLUSTER_PALETTE)]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_CACHED_CORPUS_MAP: Optional[dict] = None


def _extract_keywords(article: dict, normalizer=None) -> List[str]:
    """Extract normalized keywords from an article via domain-agnostic KB."""
    from src.utils import extract_normalized_article_keywords
    if normalizer is not None:
        return extract_normalized_article_keywords(article, corpus_map=normalizer)
    from src.utils import article_keywords
    return article_keywords(article)


def _build_keyword_clusters(
    articles: List[dict],
    min_freq: int = 2,
    max_clusters: int = 10,
) -> Optional[Tuple]:
    """Build keyword co-occurrence network, detect communities, compute layout.

    Returns: (G_kw, partition, layout, kw_freq, cluster_labels, cluster_members)
    or None if insufficient data.
    """
    if not HAS_NX:
        return None

    G_kw = nx.Graph()
    kw_freq: Counter = Counter()

    for a in articles:
        kws = _extract_keywords(a)
        for kw in kws:
            kw_freq[kw] += 1
        for i in range(len(kws)):
            for j in range(i + 1, len(kws)):
                if G_kw.has_edge(kws[i], kws[j]):
                    G_kw[kws[i]][kws[j]]["weight"] += 1
                else:
                    G_kw.add_edge(kws[i], kws[j], weight=1)

    # Filter low-frequency
    low = [n for n in G_kw if kw_freq.get(n, 0) < min_freq]
    G_kw.remove_nodes_from(low)

    if len(G_kw) < 5:
        logger.warning("Insufficient keywords for cluster analysis")
        return None

    # Community detection
    try:
        import community as community_louvain
        partition = community_louvain.best_partition(G_kw, random_state=42)
    except ImportError:
        partition = {}
        for idx, comp in enumerate(nx.connected_components(G_kw)):
            for node in comp:
                partition[node] = idx

    # Limit clusters
    cluster_sizes = Counter(partition.values())
    top_clusters = [c for c, _ in cluster_sizes.most_common(max_clusters)]
    # Reassign small clusters to nearest large cluster
    top_set = set(top_clusters)
    for node in list(partition.keys()):
        if partition[node] not in top_set:
            # Assign to the most common neighbor cluster
            neighbors = [partition[nb] for nb in G_kw.neighbors(node)
                         if partition.get(nb) in top_set]
            if neighbors:
                partition[node] = Counter(neighbors).most_common(1)[0][0]
            else:
                partition[node] = top_clusters[0]

    # Layout
    k = 3.0 / max(np.sqrt(G_kw.number_of_nodes()), 1)
    layout = nx.spring_layout(G_kw, k=k, iterations=80, seed=42, weight="weight")

    # Cluster labels and members
    cluster_members: Dict[int, List[str]] = defaultdict(list)
    for kw, cid in partition.items():
        cluster_members[cid].append(kw)

    cluster_labels = {}
    for cid, members in cluster_members.items():
        sorted_m = sorted(members, key=lambda m: kw_freq.get(m, 0), reverse=True)
        cluster_labels[cid] = sorted_m[0] if sorted_m else f"cluster_{cid}"

    return G_kw, partition, layout, kw_freq, cluster_labels, dict(cluster_members)


def _compute_callon_metrics_by_period(
    articles: List[dict],
    n_periods: int = 3,
) -> Optional[Tuple]:
    """Segment corpus, compute Callon centrality/density per period.

    Returns: (periods, period_clusters, cluster_tracking) or None.
    """
    if not HAS_NX:
        return None

    years = [a.get("publication_year") or a.get("year") or 2020 for a in articles]
    valid_years = sorted(set(y for y in years if y > 1950))
    if len(valid_years) < 2:
        return None

    year_min, year_max = min(valid_years), max(valid_years)
    span = year_max - year_min + 1
    period_size = max(1, span // n_periods)

    periods = []
    for p in range(n_periods):
        start = year_min + p * period_size
        end = start + period_size - 1 if p < n_periods - 1 else year_max
        periods.append((start, end))

    period_clusters = []

    for period_idx, (p_start, p_end) in enumerate(periods):
        period_articles = [a for a in articles
                           if p_start <= (a.get("publication_year") or a.get("year") or 0) <= p_end]
        if len(period_articles) < 5:
            period_clusters.append([])
            continue

        G_kw = nx.Graph()
        kw_freq: Counter = Counter()
        for a in period_articles:
            kws = _extract_keywords(a)
            for kw in kws:
                kw_freq[kw] += 1
            for i in range(len(kws)):
                for j in range(i + 1, len(kws)):
                    if G_kw.has_edge(kws[i], kws[j]):
                        G_kw[kws[i]][kws[j]]["weight"] += 1
                    else:
                        G_kw.add_edge(kws[i], kws[j], weight=1)

        low_freq = [n for n in G_kw if kw_freq.get(n, 0) < 2]
        G_kw.remove_nodes_from(low_freq)

        if len(G_kw) < 3:
            period_clusters.append([])
            continue

        try:
            import community as community_louvain
            partition = community_louvain.best_partition(G_kw, random_state=42)
        except ImportError:
            partition = {}
            for idx, comp in enumerate(nx.connected_components(G_kw)):
                for node in comp:
                    partition[node] = idx

        clusters_in_period = []
        cluster_ids = set(partition.values())

        for cid in cluster_ids:
            members = [n for n, c in partition.items() if c == cid]
            if len(members) < 2:
                continue

            # Callon et al. (1991): misma implementacion que la Fase 3b
            from src.strategic_thematic_analysis import callon_centrality_density
            centrality, density = callon_centrality_density(G_kw, members, kw_freq)
            total_freq = sum(kw_freq.get(m, 0) for m in members)
            sorted_members = sorted(members, key=lambda m: kw_freq.get(m, 0), reverse=True)
            label = "; ".join(sorted_members[:3])

            clusters_in_period.append({
                "period_idx": period_idx,
                "period_label": f"{p_start}\u2013{p_end}",
                "cluster_id": cid,
                "label": label,
                "keywords": set(members),
                "n_keywords": len(members),
                "centrality": round(centrality, 4),
                "density": round(density, 4),
                "total_frequency": total_freq,
                "top_keywords": sorted_members[:5],
            })

        period_clusters.append(clusters_in_period)

    # Track clusters across periods (Jaccard)
    global_cluster_id = 0
    cluster_tracking = {}

    for period_idx, clusters in enumerate(period_clusters):
        if period_idx == 0:
            for c in clusters:
                cluster_tracking[(period_idx, c["cluster_id"])] = global_cluster_id
                c["global_id"] = global_cluster_id
                global_cluster_id += 1
        else:
            prev_clusters = period_clusters[period_idx - 1]
            for c in clusters:
                best_match = -1
                best_jaccard = 0.3
                for pc in prev_clusters:
                    intersection = len(c["keywords"] & pc["keywords"])
                    union = len(c["keywords"] | pc["keywords"])
                    jaccard = intersection / union if union > 0 else 0
                    if jaccard > best_jaccard:
                        best_jaccard = jaccard
                        best_match = pc.get("global_id", -1)
                if best_match >= 0:
                    c["global_id"] = best_match
                    cluster_tracking[(period_idx, c["cluster_id"])] = best_match
                else:
                    c["global_id"] = global_cluster_id
                    cluster_tracking[(period_idx, c["cluster_id"])] = global_cluster_id
                    global_cluster_id += 1

    return periods, period_clusters, cluster_tracking


_COUNTRY_ALIASES = {
    "PEOPLES R CHINA": "CHINA", "P.R. CHINA": "CHINA", "PR CHINA": "CHINA",
    "UNITED STATES": "USA", "UNITED STATES OF AMERICA": "USA", "U.S.A.": "USA",
    "UNITED KINGDOM": "UK", "ENGLAND": "UK", "SCOTLAND": "UK", "WALES": "UK",
    "SOUTH KOREA": "SOUTH KOREA", "KOREA": "SOUTH KOREA",
    "RUSSIAN FEDERATION": "RUSSIA", "DEUTSCHLAND": "GERMANY",
    "NEDERLAND": "NETHERLANDS", "ESPANA": "SPAIN",
    "TURKIYE": "TURKEY", "HONG KONG": "CHINA (HONG KONG)",
    "TAIWAN": "TAIWAN", "MACAU": "CHINA (MACAU)",
}


def _extract_countries_from_affiliations(article: dict) -> set:
    """Extract country names from affiliations_raw field."""
    countries = set()

    # Try explicit country fields first
    for field in ("author_countries", "countries"):
        ac = article.get(field) or []
        if isinstance(ac, str):
            ac = [c.strip() for c in ac.split(";") if c.strip()]
        if isinstance(ac, list):
            for c in ac:
                if isinstance(c, str) and c.strip():
                    countries.add(c.strip().upper())
                elif isinstance(c, dict):
                    name = c.get("country", c.get("name", ""))
                    if name:
                        countries.add(name.strip().upper())

    # Try authors_parsed
    if not countries:
        authors = article.get("authors_parsed", [])
        if isinstance(authors, list):
            for author in authors:
                if isinstance(author, dict):
                    country = author.get("country", "")
                    if country:
                        countries.add(country.strip().upper())

    # Try OpenAlex authorships (countries / institutions)
    if not countries:
        oa = article.get("openalex") or {}
        for auth in oa.get("authorships", []) if isinstance(oa, dict) else []:
            if not isinstance(auth, dict):
                continue
            for cc in auth.get("countries", []) or []:
                if isinstance(cc, str) and cc.strip():
                    countries.add(cc.strip().upper())
            for inst in auth.get("institutions", []) or []:
                if isinstance(inst, dict) and inst.get("country"):
                    countries.add(str(inst["country"]).strip().upper())

    # Parse from affiliations_raw (last token after last comma is usually country)
    if not countries:
        import re
        # US state abbreviations to filter out
        _US_STATES = {
            "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
            "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
            "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
            "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
            "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
            "DC",
        }
        raw = article.get("affiliations_raw", "")
        if isinstance(raw, str) and raw:
            for aff in raw.split(";"):
                aff = aff.strip()
                if not aff:
                    continue
                parts = [p.strip() for p in aff.split(",")]
                if len(parts) >= 2:
                    candidate = parts[-1].strip().upper()
                    # Remove zip codes
                    candidate = re.sub(r'\b\d{4,6}\b', '', candidate).strip()
                    # Skip US state abbreviations — mark as USA
                    if candidate in _US_STATES or re.match(r'^[A-Z]{2}\s+USA$', candidate):
                        countries.add("USA")
                        continue
                    if candidate == "USA" or candidate == "US":
                        countries.add("USA")
                        continue
                    # Skip if too short or numeric
                    if len(candidate) < 3 or candidate.isdigit():
                        continue
                    # Skip university/institution names mistakenly at end
                    if any(w in candidate for w in ["UNIV", "INST", "COLL", "DEPT", "SCH"]):
                        continue
                    countries.add(candidate)

    # Normalize country names via knowledge base ISO-2 map and aliases
    from src.utils import load_knowledge_base
    iso2_map = {k.upper(): v.upper() for k, v in load_knowledge_base().get("country_iso2", {}).items()}
    normalized = set()
    for c in countries:
        c = iso2_map.get(c, c)
        c = _COUNTRY_ALIASES.get(c, c)
        c = iso2_map.get(c, c)
        if len(c) >= 2:
            normalized.add(c)

    return normalized


def _build_country_coauthorship(
    articles: List[dict],
    min_pubs: int = 2,
) -> Optional[nx.Graph]:
    """Build weighted country co-authorship graph from articles."""
    if not HAS_NX:
        return None

    country_count: Counter = Counter()
    coauthor_count: Counter = Counter()

    for a in articles:
        countries = list(_extract_countries_from_affiliations(a))
        if not countries:
            continue

        for c in countries:
            country_count[c] += 1
        for i in range(len(countries)):
            for j in range(i + 1, len(countries)):
                pair = tuple(sorted([countries[i], countries[j]]))
                coauthor_count[pair] += 1

    # Filter by minimum publications
    valid = {c for c, n in country_count.items() if n >= min_pubs}
    if len(valid) < 3:
        return None

    G = nx.Graph()
    for c in valid:
        G.add_node(c, count=country_count[c])

    for (c1, c2), w in coauthor_count.items():
        if c1 in valid and c2 in valid:
            G.add_edge(c1, c2, weight=w)

    return G


def _bezier_2d(p0, p1, curvature: float = 0.3, n_points: int = 30):
    """2D quadratic Bezier curve for CiteSpace-style arcs."""
    p0 = np.array(p0, dtype=float)
    p1 = np.array(p1, dtype=float)
    mid = (p0 + p1) / 2
    dist = np.linalg.norm(p1 - p0)
    # Control point offset perpendicular to the line
    direction = p1 - p0
    normal = np.array([-direction[1], direction[0]])
    norm_len = np.linalg.norm(normal)
    if norm_len > 0:
        normal = normal / norm_len
    control = mid + normal * dist * curvature

    t = np.linspace(0, 1, n_points)
    curve = (np.outer((1 - t) ** 2, p0) +
             np.outer(2 * (1 - t) * t, control) +
             np.outer(t ** 2, p1))
    return curve[:, 0], curve[:, 1]


def _convex_hull_polygon(points, color, alpha=0.15, ax=None, padding=0.02):
    """Draw a filled convex hull with padding around a set of 2D points."""
    from scipy.spatial import ConvexHull

    points = np.array(points)
    if len(points) < 3:
        if len(points) == 2:
            # Draw ellipse between two points
            center = points.mean(axis=0)
            width = np.linalg.norm(points[1] - points[0]) + padding * 4
            height = width * 0.4
            angle = np.degrees(np.arctan2(points[1, 1] - points[0, 1],
                                          points[1, 0] - points[0, 0]))
            ellipse = mpatches.Ellipse(center, width, height, angle=angle,
                                       facecolor=to_rgba(color, alpha),
                                       edgecolor=to_rgba(color, alpha * 2.5),
                                       linewidth=1.5, zorder=1)
            if ax:
                ax.add_patch(ellipse)
            return
        elif len(points) == 1:
            circle = mpatches.Circle(points[0], radius=padding * 3,
                                     facecolor=to_rgba(color, alpha),
                                     edgecolor=to_rgba(color, alpha * 2.5),
                                     linewidth=1.5, zorder=1)
            if ax:
                ax.add_patch(circle)
            return
        return

    try:
        hull = ConvexHull(points)
    except Exception:
        return

    hull_points = points[hull.vertices]
    # Add padding by expanding from centroid
    centroid = hull_points.mean(axis=0)
    expanded = centroid + (hull_points - centroid) * (1 + padding * 3)

    polygon = plt.Polygon(expanded, closed=True,
                          facecolor=to_rgba(color, alpha),
                          edgecolor=to_rgba(color, min(1.0, alpha * 2.5)),
                          linewidth=1.5, zorder=1)
    if ax:
        ax.add_patch(polygon)


def _save_mpl_figure(fig, output_dir: Path, name: str) -> Path:
    """Save matplotlib figure at publication DPI."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.png"
    fig.savefig(str(path), dpi=FIGURE_DPI, facecolor="white",
                bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    logger.info(f"Saved: {path}")
    return path


# ===================================================================
# FIGURE 1: Thematic Overlay Map (VOSviewer style - Example 01)
# ===================================================================

def fig01_thematic_overlay_map(
    articles: List[dict],
    output_dir: Path,
    max_clusters: int = 8,
    top_keywords_per_cluster: int = 12,
) -> Optional[Path]:
    """Thematic overlay map with colored bubble clusters and keyword lists."""
    result = _build_keyword_clusters(articles, min_freq=2, max_clusters=max_clusters)
    if result is None:
        logger.warning("Cannot build thematic overlay map: insufficient data")
        return None

    G_kw, partition, layout, kw_freq, cluster_labels, cluster_members = result

    fig, ax = plt.subplots(figsize=(20, 16), facecolor="white")
    ax.set_facecolor("white")

    # Sort clusters by size (largest first for drawing order)
    sorted_cids = sorted(cluster_members.keys(),
                         key=lambda c: len(cluster_members[c]), reverse=True)

    # Remap cluster IDs to sequential for clean labeling
    cid_remap = {cid: idx for idx, cid in enumerate(sorted_cids)}

    # Draw convex hulls per cluster
    for cid in sorted_cids:
        members = cluster_members[cid]
        color = _get_cluster_color(cid_remap[cid])
        points = [layout[kw] for kw in members if kw in layout]
        if len(points) >= 1:
            _convex_hull_polygon(points, color, alpha=HULL_ALPHA, ax=ax, padding=0.03)

    # Draw edges (subtle)
    for u, v, d in G_kw.edges(data=True):
        if u in layout and v in layout:
            w = d.get("weight", 1)
            ax.plot([layout[u][0], layout[v][0]],
                    [layout[u][1], layout[v][1]],
                    color="#cccccc", linewidth=max(0.2, min(1.5, w * 0.15)),
                    alpha=0.3, zorder=2)

    # Draw nodes
    for cid in sorted_cids:
        members = cluster_members[cid]
        color = _get_cluster_color(cid_remap[cid])
        for kw in members:
            if kw not in layout:
                continue
            freq = kw_freq.get(kw, 1)
            size = max(15, min(300, freq * 8))
            ax.scatter(layout[kw][0], layout[kw][1], s=size,
                       c=color, alpha=0.7, edgecolors="white",
                       linewidths=0.5, zorder=3)

    # Cluster labels and keyword lists
    for cid in sorted_cids:
        members = cluster_members[cid]
        color = _get_cluster_color(cid_remap[cid])
        idx = cid_remap[cid]

        points = [layout[kw] for kw in members if kw in layout]
        if not points:
            continue
        centroid = np.mean(points, axis=0)

        # Bold cluster label at centroid
        main_label = cluster_labels.get(cid, f"cluster {idx}")
        ax.text(centroid[0], centroid[1], f"#{idx} {main_label}",
                fontsize=11, fontweight="bold", color=color,
                ha="center", va="center", zorder=6,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=color, alpha=0.85, linewidth=1.5))

        # Keyword list around cluster
        sorted_members = sorted(members, key=lambda m: kw_freq.get(m, 0),
                                reverse=True)[:top_keywords_per_cluster]
        # Position keywords in a column near the cluster
        # Find the side with more space
        x_range = max(p[0] for p in points) - min(p[0] for p in points) if len(points) > 1 else 0.1
        # Place keyword list to the side of the cluster
        offset_x = x_range * 0.5 + 0.06
        if centroid[0] > 0:
            text_x = centroid[0] + offset_x
            ha = "left"
        else:
            text_x = centroid[0] - offset_x
            ha = "right"

        kw_text = "\n".join(sorted_members)
        ax.text(text_x, centroid[1], kw_text,
                fontsize=7, color=color, ha=ha, va="center",
                fontfamily="serif", linespacing=1.4, zorder=5,
                alpha=0.85)

    # Legend
    legend_handles = []
    for cid in sorted_cids:
        idx = cid_remap[cid]
        color = _get_cluster_color(idx)
        label = f"#{idx} {cluster_labels.get(cid, '')}"
        legend_handles.append(mpatches.Patch(facecolor=color, alpha=0.5,
                                             edgecolor=color, label=label[:40]))

    ax.legend(handles=legend_handles, loc="lower left", fontsize=8,
              framealpha=0.9, edgecolor="#cccccc", ncol=2,
              title="Thematic Clusters", title_fontsize=9)

    ax.set_title("Thematic Overlay Map", fontsize=16, fontweight="bold", pad=20)
    ax.axis("off")

    return _save_mpl_figure(fig, output_dir, "rdm_thematic_overlay_map")


# ===================================================================
# FIGURE 2: Strategic Diagram / Callon Panels (Example 02)
# ===================================================================

def fig02_strategic_diagram_panels(
    articles: List[dict],
    output_dir: Path,
    n_periods: int = 3,
) -> Optional[Path]:
    """Multi-panel Callon strategic diagram across time periods."""
    result = _compute_callon_metrics_by_period(articles, n_periods=n_periods)
    if result is None:
        logger.warning("Cannot build strategic diagram: insufficient data")
        return None

    periods, period_clusters, cluster_tracking = result

    # Filter out empty periods
    active_periods = [(i, periods[i], period_clusters[i])
                      for i in range(len(periods)) if period_clusters[i]]
    if not active_periods:
        logger.warning("No active periods for strategic diagram")
        return None

    n_active = len(active_periods)
    fig, axes = plt.subplots(1, n_active, figsize=(7 * n_active, 8),
                             facecolor="white", squeeze=False)
    axes = axes[0]

    # Compute global ranges for consistent axes
    all_cent = [c["centrality"] for _, _, cs in active_periods for c in cs]
    all_dens = [c["density"] for _, _, cs in active_periods for c in cs]
    if not all_cent:
        plt.close(fig)
        return None

    cent_med = np.median(all_cent)
    dens_med = np.median(all_dens)
    cent_range = (min(all_cent) - 0.1, max(all_cent) + 0.1)
    dens_range = (min(all_dens) - 0.05, max(all_dens) + 0.05)

    for ax_idx, (period_idx, (p_start, p_end), clusters) in enumerate(active_periods):
        ax = axes[ax_idx]
        ax.set_facecolor("#fafafa")

        # Median reference lines
        ax.axhline(dens_med, color="#bbbbbb", linestyle="--", linewidth=1, zorder=1)
        ax.axvline(cent_med, color="#bbbbbb", linestyle="--", linewidth=1, zorder=1)

        # Quadrant labels
        quadrant_labels = [
            (0.95, 0.95, "Motor Themes", "right", "top"),
            (0.05, 0.95, "Niche Themes", "left", "top"),
            (0.05, 0.05, "Emerging /\nDeclining", "left", "bottom"),
            (0.95, 0.05, "Basic /\nTransversal", "right", "bottom"),
        ]
        for qx, qy, qlabel, qha, qva in quadrant_labels:
            ax.text(qx, qy, qlabel, transform=ax.transAxes,
                    fontsize=9, color="#aaaaaa", fontstyle="italic",
                    ha=qha, va=qva, zorder=1)

        # Draw clusters as bubbles
        for c in clusters:
            color = _get_cluster_color(c.get("global_id", c["cluster_id"]))
            size = max(100, min(2000, c["total_frequency"] * 15))
            ax.scatter(c["centrality"], c["density"], s=size,
                       c=color, alpha=0.6, edgecolors=color,
                       linewidths=1.5, zorder=3)

            # Cluster label
            label_text = c["top_keywords"][0] if c["top_keywords"] else ""
            if len(c["top_keywords"]) > 1:
                label_text += f"\n{c['top_keywords'][1]}"
            ax.text(c["centrality"], c["density"] + 0.005, label_text,
                    fontsize=7.5, fontweight="bold", color="#333333",
                    ha="center", va="bottom", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor="none", alpha=0.7))

        ax.set_xlabel("Relevance Degree\n(Callon Centrality)", fontsize=10)
        if ax_idx == 0:
            ax.set_ylabel("Development Degree\n(Callon Density)", fontsize=10)
        ax.set_title(f"{p_start}\u2013{p_end}", fontsize=12, fontweight="bold")
        ax.tick_params(labelsize=8)

        # Consistent ranges
        ax.set_xlim(cent_range)
        ax.set_ylim(dens_range)

    fig.suptitle("Strategic Thematic Diagram (Callon)", fontsize=16,
                 fontweight="bold", y=1.02)
    fig.tight_layout()

    return _save_mpl_figure(fig, output_dir, "rdm_strategic_diagram_callon")


# ===================================================================
# FIGURE 3: Country Collaboration Network (Example 03)
# ===================================================================

def fig03_country_collaboration_network(
    articles: List[dict],
    output_dir: Path,
    min_publications: int = 2,
) -> Optional[Path]:
    """Country collaboration network with gradient-colored nodes."""
    G = _build_country_coauthorship(articles, min_pubs=min_publications)
    if G is None or len(G) < 3:
        logger.warning("Cannot build country collaboration network: insufficient data")
        return None

    fig, ax = plt.subplots(figsize=(16, 12), facecolor="white")
    ax.set_facecolor("white")

    # Layout
    k = 2.5 / max(np.sqrt(len(G)), 1)
    pos = nx.spring_layout(G, k=k, iterations=100, seed=42, weight="weight")

    # Node sizes and colors
    counts = nx.get_node_attributes(G, "count")
    max_count = max(counts.values()) if counts else 1

    # Gradient colormap (magenta to light pink, like example 03)
    cmap = LinearSegmentedColormap.from_list("country_grad",
        ["#fce4ec", "#f48fb1", "#e91e63", "#880e4f"], N=256)

    node_sizes = {n: 100 + 1500 * np.sqrt(counts.get(n, 1) / max_count) for n in G.nodes()}
    node_colors = {n: counts.get(n, 1) / max_count for n in G.nodes()}

    # Draw edges
    edge_weights = [d.get("weight", 1) for _, _, d in G.edges(data=True)]
    max_ew = max(edge_weights) if edge_weights else 1

    for u, v, d in G.edges(data=True):
        w = d.get("weight", 1)
        lw = max(0.3, min(5.0, 4.0 * w / max_ew))
        alpha = max(0.1, min(0.6, 0.5 * w / max_ew))
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color="#e0e0e0", linewidth=lw, alpha=alpha, zorder=1,
                solid_capstyle="round")

    # Draw nodes with gradient
    nodes_sorted = sorted(G.nodes(), key=lambda n: counts.get(n, 0))
    for n in nodes_sorted:
        color_val = node_colors[n]
        color = cmap(color_val)
        size = node_sizes[n]

        # Outer ring (halo)
        ax.scatter(pos[n][0], pos[n][1], s=size * 1.6,
                   c=[to_rgba(color, 0.2)], edgecolors="none", zorder=2)
        # Main node
        ax.scatter(pos[n][0], pos[n][1], s=size,
                   c=[color], edgecolors="white", linewidths=1.5, zorder=3)

    # Labels
    try:
        from adjustText import adjust_text
        texts = []
        for n in G.nodes():
            count_n = counts.get(n, 0)
            fontsize = 7 + 5 * np.sqrt(count_n / max_count)
            fontsize = min(fontsize, 13)
            t = ax.text(pos[n][0], pos[n][1], n,
                        fontsize=fontsize, fontweight="bold",
                        ha="center", va="center", zorder=5,
                        color="#333333",
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                                  edgecolor="none", alpha=0.75))
            texts.append(t)
        if texts:
            adjust_text(texts, ax=ax,
                        arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.5),
                        expand=(1.3, 1.5), force_text=(0.8, 1.0))
    except ImportError:
        for n in G.nodes():
            count_n = counts.get(n, 0)
            fontsize = 7 + 5 * np.sqrt(count_n / max_count)
            fontsize = min(fontsize, 13)
            ax.text(pos[n][0], pos[n][1], n,
                    fontsize=fontsize, fontweight="bold",
                    ha="center", va="center", zorder=5,
                    color="#333333",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                              edgecolor="none", alpha=0.75))

    ax.set_title("Country Collaboration Network", fontsize=16,
                 fontweight="bold", pad=20)
    ax.text(0.5, -0.02, f"{len(G.nodes())} countries \u00b7 {len(G.edges())} collaborations",
            transform=ax.transAxes, ha="center", fontsize=10, color="#666666")
    ax.axis("off")

    return _save_mpl_figure(fig, output_dir, "rdm_country_collaboration")


# ===================================================================
# FIGURE 4: Thematic Landscape 2D (Example 04)
# ===================================================================

def fig04_thematic_landscape_2d(
    articles: List[dict],
    output_dir: Path,
    max_clusters: int = 10,
) -> Optional[Path]:
    """2D thematic landscape with convex hull cluster regions."""
    result = _build_keyword_clusters(articles, min_freq=2, max_clusters=max_clusters)
    if result is None:
        logger.warning("Cannot build thematic landscape 2D: insufficient data")
        return None

    G_kw, partition, layout, kw_freq, cluster_labels, cluster_members = result

    fig, ax = plt.subplots(figsize=(14, 10), facecolor="white")
    ax.set_facecolor("white")

    sorted_cids = sorted(cluster_members.keys(),
                         key=lambda c: len(cluster_members[c]), reverse=True)
    cid_remap = {cid: idx for idx, cid in enumerate(sorted_cids)}

    # Draw convex hulls
    for cid in sorted_cids:
        members = cluster_members[cid]
        color = _get_cluster_color(cid_remap[cid])
        points = [layout[kw] for kw in members if kw in layout]
        if len(points) >= 1:
            _convex_hull_polygon(points, color, alpha=0.2, ax=ax, padding=0.04)

    # Draw subtle edges
    for u, v, d in G_kw.edges(data=True):
        if u in layout and v in layout and partition.get(u) == partition.get(v):
            ax.plot([layout[u][0], layout[v][0]],
                    [layout[u][1], layout[v][1]],
                    color="#e0e0e0", linewidth=0.3, alpha=0.4, zorder=2)

    # Draw nodes
    for cid in sorted_cids:
        members = cluster_members[cid]
        color = _get_cluster_color(cid_remap[cid])
        for kw in members:
            if kw not in layout:
                continue
            freq = kw_freq.get(kw, 1)
            size = max(10, min(250, freq * 6))
            ax.scatter(layout[kw][0], layout[kw][1], s=size,
                       c=color, alpha=0.65, edgecolors="white",
                       linewidths=0.4, zorder=3)

    # Cluster labels at centroids
    for cid in sorted_cids:
        members = cluster_members[cid]
        color = _get_cluster_color(cid_remap[cid])
        idx = cid_remap[cid]
        points = [layout[kw] for kw in members if kw in layout]
        if not points:
            continue
        centroid = np.mean(points, axis=0)

        # Top 3 keywords as label
        sorted_m = sorted(members, key=lambda m: kw_freq.get(m, 0), reverse=True)
        label = f"#{idx} {sorted_m[0]}"
        if len(sorted_m) > 1:
            label += f"\n{sorted_m[1]}"
        if len(sorted_m) > 2:
            label += f"\n{sorted_m[2]}"

        ax.text(centroid[0], centroid[1], label,
                fontsize=8, fontweight="bold", color=color,
                ha="center", va="center", zorder=6,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=color, alpha=0.8, linewidth=1))

    ax.set_title("Thematic Landscape", fontsize=16, fontweight="bold", pad=20)
    ax.axis("off")

    return _save_mpl_figure(fig, output_dir, "rdm_thematic_landscape_2d")


# ===================================================================
# FIGURE 5: CiteSpace Timeline (Example 05 - KEY FIGURE)
# ===================================================================

def fig05_citespace_timeline(
    articles: List[dict],
    networks: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Path] = None,
    top_n_labels: int = 80,
    max_arcs: int = 100,
) -> Optional[Path]:
    """CiteSpace-style horizontal timeline with cluster bands and Bezier arcs."""
    if not HAS_NX:
        logger.warning("NetworkX not available; skipping CiteSpace timeline")
        return None

    # Build keyword co-occurrence network
    G_kw = nx.Graph()
    kw_freq: Counter = Counter()
    kw_first_year: Dict[str, int] = {}
    kw_articles: Dict[str, List[str]] = defaultdict(list)

    for a in articles:
        key = a.get("key", a.get("doi", ""))
        year = a.get("publication_year") or a.get("year")
        if not year:
            continue
        kws = _extract_keywords(a)
        for kw in kws:
            kw_freq[kw] += 1
            kw_articles[kw].append(key)
            if kw not in kw_first_year or year < kw_first_year[kw]:
                kw_first_year[kw] = year
        for i in range(len(kws)):
            for j in range(i + 1, len(kws)):
                if G_kw.has_edge(kws[i], kws[j]):
                    G_kw[kws[i]][kws[j]]["weight"] += 1
                else:
                    G_kw.add_edge(kws[i], kws[j], weight=1)

    # Filter low-frequency keywords
    min_freq = max(2, len(articles) // 50)
    low_freq = [n for n in G_kw if kw_freq.get(n, 0) < min_freq]
    G_kw.remove_nodes_from(low_freq)

    if len(G_kw) < 5:
        logger.warning("Insufficient keywords for CiteSpace timeline")
        return None

    # Community detection
    try:
        import community as community_louvain
        partition = community_louvain.best_partition(G_kw, random_state=42)
    except ImportError:
        partition = {}
        for idx, comp in enumerate(nx.connected_components(G_kw)):
            for node in comp:
                partition[node] = idx

    # Burst detection (optional)
    kw_year_counts: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for a in articles:
        year = a.get("publication_year") or a.get("year")
        if not year:
            continue
        for kw in _extract_keywords(a):
            if kw in partition:
                kw_year_counts[kw][year] += 1

    kw_burst_strength: Dict[str, float] = {}
    try:
        from src.citespace_analysis import _kleinberg_burst_detection
        for kw, yc in kw_year_counts.items():
            if sum(yc.values()) < 3:
                continue
            try:
                bursts = _kleinberg_burst_detection(yc)
                if bursts:
                    kw_burst_strength[kw] = max(b.get("strength", b.get("level", 0))
                                                for b in bursts)
            except Exception:
                continue
    except ImportError:
        pass

    # Cluster members and labels
    cluster_members: Dict[int, List[str]] = defaultdict(list)
    for kw, cid in partition.items():
        cluster_members[cid].append(kw)

    cluster_labels = {}
    for cid, members in cluster_members.items():
        sorted_m = sorted(members, key=lambda m: kw_freq.get(m, 0), reverse=True)
        cluster_labels[cid] = "; ".join(sorted_m[:3])

    # Sort clusters by size
    sorted_clusters = sorted(cluster_members.keys(),
                             key=lambda c: len(cluster_members[c]), reverse=True)
    # Limit to top clusters
    max_display_clusters = 12
    sorted_clusters = sorted_clusters[:max_display_clusters]
    displayed_kws = set()
    for cid in sorted_clusters:
        displayed_kws.update(cluster_members[cid])

    cluster_y = {c: idx for idx, c in enumerate(sorted_clusters)}
    n_clusters = len(sorted_clusters)

    if n_clusters < 2:
        logger.warning("Insufficient clusters for CiteSpace timeline")
        return None

    # Year range
    years_in_data = sorted(set(kw_first_year[kw] for kw in displayed_kws
                               if kw in kw_first_year))
    if len(years_in_data) < 2:
        return None
    year_min, year_max = years_in_data[0], years_in_data[-1]

    # --- Create figure ---
    fig, ax = plt.subplots(figsize=(24, 10), facecolor="white")
    ax.set_facecolor("white")

    # Band height
    band_h = 1.0

    # Draw alternating cluster bands
    for i, cid in enumerate(sorted_clusters):
        y_bottom = i * band_h - band_h * 0.5
        bg_color = "#f9f9f9" if i % 2 == 0 else "#ffffff"
        ax.axhspan(y_bottom, y_bottom + band_h, facecolor=bg_color,
                   edgecolor="#eeeeee", linewidth=0.5, zorder=0)

        # Cluster separator (dotted line at top of band)
        if i < n_clusters - 1:
            ax.axhline(y_bottom + band_h, color="#dddddd", linestyle=":",
                       linewidth=0.7, zorder=1)

    # Cross-cluster co-occurrence arcs (Bezier curves)
    cross_edges = []
    for u, v, d in G_kw.edges(data=True):
        cu = partition.get(u)
        cv = partition.get(v)
        if cu != cv and cu in cluster_y and cv in cluster_y:
            w = d.get("weight", 1)
            if w >= 2:
                cross_edges.append((u, v, w))

    cross_edges.sort(key=lambda e: e[2], reverse=True)
    cross_edges = cross_edges[:max_arcs]

    for u, v, w in cross_edges:
        year_u = kw_first_year.get(u, year_min)
        year_v = kw_first_year.get(v, year_min)
        cid_u = partition[u]
        cid_v = partition[v]
        y_u = cluster_y[cid_u] * band_h
        y_v = cluster_y[cid_v] * band_h

        # Jitter within band
        jitter_u = (hash(u) % 100) / 100 * 0.4 - 0.2
        jitter_v = (hash(v) % 100) / 100 * 0.4 - 0.2

        p0 = (year_u, y_u + jitter_u)
        p1 = (year_v, y_v + jitter_v)

        # Curvature proportional to vertical distance
        vert_dist = abs(y_u - y_v) / (n_clusters * band_h)
        curvature = 0.15 + 0.2 * vert_dist

        bx, by = _bezier_2d(p0, p1, curvature=curvature, n_points=40)
        color_u = _get_cluster_color(sorted_clusters.index(cid_u)
                                     if cid_u in sorted_clusters
                                     else cid_u)
        alpha = max(0.15, min(0.55, 0.30 * w / max(1, cross_edges[0][2])))
        lw = max(1.0, min(5.0, 3.0 * w / max(1, cross_edges[0][2])))
        ax.plot(bx, by, color=color_u, linewidth=lw, alpha=alpha, zorder=2,
                solid_capstyle="round")

    # Draw keyword nodes
    max_freq = max(kw_freq.get(kw, 1) for kw in displayed_kws)

    for cid in sorted_clusters:
        members = cluster_members[cid]
        color = _get_cluster_color(sorted_clusters.index(cid))
        y_base = cluster_y[cid] * band_h

        for kw in members:
            if kw not in kw_first_year:
                continue
            year = kw_first_year[kw]
            freq = kw_freq.get(kw, 1)

            jitter = (hash(kw) % 100) / 100 * 0.35 - 0.175
            y = y_base + jitter

            size = max(12, min(280, 190 * np.sqrt(freq / max_freq)))
            ax.scatter(year, y, s=size, c=color, alpha=0.75,
                       edgecolors="white", linewidths=0.5, zorder=4)

    # Labels for top keywords per cluster
    labeled_positions = []
    for cid in sorted_clusters:
        members = cluster_members[cid]
        sorted_m = sorted(members, key=lambda m: kw_freq.get(m, 0), reverse=True)
        n_labels = min(8, len(sorted_m))
        y_base = cluster_y[cid] * band_h

        for kw in sorted_m[:n_labels]:
            if kw not in kw_first_year:
                continue
            year = kw_first_year[kw]
            jitter = (hash(kw) % 100) / 100 * 0.35 - 0.175
            y = y_base + jitter
            freq = kw_freq.get(kw, 1)
            fsize = max(7, min(11.5, 7 + 4.5 * np.sqrt(freq / max_freq)))

            color = _get_cluster_color(sorted_clusters.index(cid))
            ax.text(year + 0.15, y + 0.05, kw[:25],
                    fontsize=fsize, color="#333333",
                    ha="left", va="bottom", zorder=6,
                    fontfamily="serif",
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="white",
                              edgecolor="none", alpha=0.6))

    # Cluster labels on the right margin
    for cid in sorted_clusters:
        idx = sorted_clusters.index(cid)
        y_base = cluster_y[cid] * band_h
        color = _get_cluster_color(idx)
        label = cluster_labels.get(cid, f"#{cid}")
        ax.text(year_max + 0.8, y_base, f"#{idx}  {label[:35]}",
                fontsize=11, fontweight="bold", color=color,
                ha="left", va="center", zorder=6,
                fontfamily="serif")

    # Axes styling
    ax.set_xlim(year_min - 0.5, year_max + 8)
    ax.set_ylim(-band_h * 0.6, (n_clusters - 1) * band_h + band_h * 0.6)

    # X-axis: years
    year_range = year_max - year_min
    if year_range <= 15:
        ax.set_xticks(range(year_min, year_max + 1))
    else:
        ax.set_xticks(range(year_min, year_max + 1, 2))
    ax.tick_params(axis="x", labelsize=11, rotation=45)

    # Y-axis: cluster IDs
    ax.set_yticks([cluster_y[c] * band_h for c in sorted_clusters])
    ax.set_yticklabels([f"#{sorted_clusters.index(c)}" for c in sorted_clusters],
                       fontsize=11)

    # Timeline arrow at bottom
    ax.annotate("", xy=(year_max + 0.5, -band_h * 0.55),
                xytext=(year_min - 0.3, -band_h * 0.55),
                arrowprops=dict(arrowstyle="->", color="#999999", lw=1.5))
    ax.text((year_min + year_max) / 2, -band_h * 0.62, "Timeline",
            ha="center", fontsize=11, color="#999999", fontstyle="italic")

    ax.set_xlabel("First Appearance (Year)", fontsize=13, labelpad=15)
    ax.set_ylabel("Thematic Cluster", fontsize=13)
    ax.set_title("CiteSpace-Style Keyword Co-occurrence Timeline",
                 fontsize=18, fontweight="bold", pad=20)

    # Minimal spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")

    fig.tight_layout()
    return _save_mpl_figure(fig, output_dir, "rdm_citespace_timeline")


# ===================================================================
# Entry point
# ===================================================================

def run_rdm_matplotlib_figures(
    articles: List[dict],
    networks: Optional[Dict[str, Any]] = None,
    rdm_records: Optional[List[dict]] = None,
    output_dir: Optional[Path] = None,
) -> dict:
    """Generate all 5 publication-quality matplotlib figures.

    Returns dict with charts_saved count and file paths.
    """
    if output_dir is None:
        from config import RDM_DIR
        output_dir = RDM_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Vocabulario comun del pipeline (se recalcula solo si este corpus no es el vigente)
    from src.utils import prepare_corpus_vocabulary, _VOCABULARIO, _huella_corpus
    if _VOCABULARIO.get("clave") != _huella_corpus(articles):
        prepare_corpus_vocabulary(articles, write_report=False)

    charts_saved = 0
    paths = {}

    # Figure 4 first (simplest, validates pipeline)
    logger.info("Generating Figure 4: Thematic Landscape 2D...")
    try:
        p = fig04_thematic_landscape_2d(articles, output_dir)
        if p:
            charts_saved += 1
            paths["thematic_landscape_2d"] = str(p)
    except Exception as e:
        logger.error(f"Figure 4 (Thematic Landscape 2D) failed: {e}")

    # Figure 1: Thematic Overlay Map
    logger.info("Generating Figure 1: Thematic Overlay Map...")
    try:
        p = fig01_thematic_overlay_map(articles, output_dir)
        if p:
            charts_saved += 1
            paths["thematic_overlay_map"] = str(p)
    except Exception as e:
        logger.error(f"Figure 1 (Thematic Overlay Map) failed: {e}")

    # Figure 2: Strategic Diagram (Callon)
    logger.info("Generating Figure 2: Strategic Diagram (Callon)...")
    try:
        p = fig02_strategic_diagram_panels(articles, output_dir)
        if p:
            charts_saved += 1
            paths["strategic_diagram_callon"] = str(p)
    except Exception as e:
        logger.error(f"Figure 2 (Strategic Diagram) failed: {e}")

    # Figure 3: Country Collaboration Network
    logger.info("Generating Figure 3: Country Collaboration Network...")
    try:
        p = fig03_country_collaboration_network(articles, output_dir)
        if p:
            charts_saved += 1
            paths["country_collaboration"] = str(p)
    except Exception as e:
        logger.error(f"Figure 3 (Country Collaboration) failed: {e}")

    # Figure 5: CiteSpace Timeline
    logger.info("Generating Figure 5: CiteSpace Timeline...")
    try:
        p = fig05_citespace_timeline(articles, networks, output_dir)
        if p:
            charts_saved += 1
            paths["citespace_timeline"] = str(p)
    except Exception as e:
        logger.error(f"Figure 5 (CiteSpace Timeline) failed: {e}")

    logger.info(f"Matplotlib publication figures complete: {charts_saved}/5 saved")
    return {"charts_saved": charts_saved, "paths": paths}
