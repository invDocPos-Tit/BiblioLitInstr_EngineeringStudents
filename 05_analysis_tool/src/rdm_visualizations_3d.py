"""Visualizaciones 3D avanzadas adicionales para Phase 6.

Dos analisis 3D inspirados en CiteSpace:

1. Citation Chronotope 3D — Timeline de clusters con arcos de citacion curvos
   (inspirado en CiteSpace timeline view + citation arcs, ejemplo 05)
   Ejes: X=Tiempo, Y=Cluster tematico, Z=Impacto (citaciones normalizadas)

2. Thematic Landscape 3D — Paisaje tematico evolutivo
   (inspirado en CiteSpace cluster view + strategic diagram, ejemplos 01/02)
   Ejes: X=Centralidad de Callon, Y=Densidad de Callon, Z=Periodo temporal
"""
from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

logger = logging.getLogger("bibliometrico.rdm_viz3d")

# ---------------------------------------------------------------------------
# Paleta de colores para clusters
# ---------------------------------------------------------------------------
CLUSTER_PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabebe", "#469990",
    "#e6beff", "#9A6324", "#fffac8", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#a9a9a9", "#000000",
]


def _get_cluster_color(cid: int) -> str:
    return CLUSTER_PALETTE[cid % len(CLUSTER_PALETTE)]


def _save_figure(fig, output_dir: Path, name: str) -> None:
    """Save Plotly figure as HTML and optionally PNG."""
    from src.utils import safe_write_plotly_png

    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / f"{name}.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    logger.info(f"Saved: {html_path}")
    png_path = output_dir / f"{name}.png"
    safe_write_plotly_png(fig, png_path, width=2400, height=1500, scale=3, log=logger)


def _extract_keywords(article: dict) -> List[str]:
    """Palabras clave normalizadas con el vocabulario comun del pipeline (src.utils.article_keywords)."""
    from src.utils import article_keywords
    return article_keywords(article)


def _bezier_3d(p0, p1, height_factor: float = 0.3, n_points: int = 30,
               max_height: float = 0.0):
    """Genera curva de Bezier cuadratica 3D entre dos puntos.

    El punto de control se eleva en Z por encima del plano de los dos puntos.
    Si max_height > 0, la elevacion se limita a ese valor absoluto para
    evitar que los arcos se salgan del rango visible.
    """
    p0 = np.array(p0, dtype=float)
    p1 = np.array(p1, dtype=float)
    mid = (p0 + p1) / 2
    dist = np.linalg.norm(p1 - p0)
    # Punto de control: elevado en Z
    elevation = dist * height_factor
    if max_height > 0:
        elevation = min(elevation, max_height)
    control = mid.copy()
    control[2] += elevation

    t = np.linspace(0, 1, n_points)
    curve = np.outer((1 - t) ** 2, p0) + np.outer(2 * (1 - t) * t, control) + np.outer(t ** 2, p1)
    return curve[:, 0], curve[:, 1], curve[:, 2]


# ===================================================================
# ANALISIS 1: Citation Chronotope 3D
# ===================================================================

def plot_citation_chronotope_3d(
    articles: List[dict],
    networks: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Path] = None,
    top_n: int = 80,
) -> Optional[Path]:
    """Citation Chronotope 3D — Timeline con clusters y arcos de citacion.

    Inspirado en CiteSpace timeline view (ejemplo 05):
    - X = Ano de publicacion
    - Y = Cluster tematico (comunidad en red de acoplamiento bibliografico)
    - Z = Impacto normalizado (citaciones / max)

    Arcos de Bezier 3D conectan articulos que se citan mutuamente dentro
    del corpus, coloreados por el cluster del articulo citante.

    Args:
        articles: Lista de articulos enriquecidos
        networks: Dict con grafos NetworkX (necesita 'bibliographic_coupling')
        output_dir: Directorio de salida
        top_n: Numero maximo de articulos a mostrar
    """
    if not HAS_PLOTLY or not HAS_NX:
        logger.warning("Plotly/NetworkX no disponible; omitiendo Citation Chronotope 3D")
        return None

    if output_dir is None:
        from config import RDM_DIR
        output_dir = RDM_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building Citation Chronotope 3D...")

    # --- Build article lookup ---
    art_by_key = {}
    art_by_oa_id = {}
    for a in articles:
        key = a.get("key", a.get("doi", ""))
        if key:
            art_by_key[key] = a
        oa_id = a.get("openalex_id") or a.get("openalex", {}).get("id", "")
        if oa_id:
            art_by_oa_id[oa_id] = a
        doi = a.get("doi", "")
        if doi:
            art_by_key[doi] = a

    # --- Get community assignments from network ---
    node_community = {}
    if networks and "bibliographic_coupling" in networks:
        G = networks["bibliographic_coupling"]
        for node, data in G.nodes(data=True):
            comm = data.get("community", 0)
            node_community[node] = comm
    else:
        # Fallback: assign communities by keyword similarity groups
        logger.info("No network available; assigning clusters by keyword groups")
        kw_counter = Counter()
        for a in articles:
            for kw in _extract_keywords(a):
                kw_counter[kw] += 1
        top_kws = [kw for kw, _ in kw_counter.most_common(8)]
        for a in articles:
            key = a.get("key", a.get("doi", ""))
            kws = _extract_keywords(a)
            best_cluster = 0
            for idx, tkw in enumerate(top_kws):
                if tkw in kws:
                    best_cluster = idx
                    break
            node_community[key] = best_cluster

    # --- Select top articles by citation ---
    sorted_articles = sorted(articles, key=lambda a: a.get("cited_by_count", 0), reverse=True)
    selected = sorted_articles[:top_n]

    # --- Build internal citation edges ---
    # Article A cites Article B if B's OpenAlex ID is in A's referenced_works
    all_oa_ids = set()
    key_to_oa = {}
    for a in selected:
        key = a.get("key", a.get("doi", ""))
        oa_id = a.get("openalex_id") or a.get("openalex", {}).get("id", "")
        if oa_id:
            all_oa_ids.add(oa_id)
            key_to_oa[key] = oa_id

    oa_to_key = {v: k for k, v in key_to_oa.items()}

    citation_edges = []  # (citing_key, cited_key)
    for a in selected:
        citing_key = a.get("key", a.get("doi", ""))
        refs = a.get("referenced_works", [])
        for ref in refs:
            ref_id = ref if isinstance(ref, str) else ref.get("id", "")
            if ref_id in oa_to_key:
                cited_key = oa_to_key[ref_id]
                if cited_key != citing_key:
                    citation_edges.append((citing_key, cited_key))

    # --- Compute positions ---
    years = [a.get("publication_year") or a.get("year") or 2020 for a in selected]
    year_min, year_max = min(years), max(years)
    year_range = max(1, year_max - year_min)

    max_cites = max((a.get("cited_by_count", 0) for a in selected), default=1)
    max_cites = max(1, max_cites)

    # Get unique communities and sort
    communities = sorted(set(node_community.get(a.get("key", a.get("doi", "")), 0)
                             for a in selected))
    comm_to_y = {c: idx for idx, c in enumerate(communities)}
    n_clusters = len(communities)

    # Cluster labels from top keywords
    cluster_keywords = defaultdict(lambda: Counter())
    for a in selected:
        key = a.get("key", a.get("doi", ""))
        comm = node_community.get(key, 0)
        for kw in _extract_keywords(a):
            cluster_keywords[comm][kw] += 1

    cluster_labels = {}
    for comm, kw_counts in cluster_keywords.items():
        top3 = [kw for kw, _ in kw_counts.most_common(3)]
        cluster_labels[comm] = "; ".join(top3) if top3 else f"Cluster {comm}"

    # Position dict
    positions = {}
    year_cluster_count = defaultdict(lambda: defaultdict(int))

    for a in selected:
        key = a.get("key", a.get("doi", ""))
        year = a.get("publication_year") or a.get("year") or 2020
        cites = a.get("cited_by_count", 0)
        comm = node_community.get(key, 0)

        x = year  # keep as real year for readability
        y_base = comm_to_y.get(comm, 0)
        # Add jitter within cluster band
        jitter = year_cluster_count[year][comm] * 0.15
        year_cluster_count[year][comm] += 1
        y = y_base + jitter * 0.3

        z = math.log(1 + cites) / math.log(1 + max_cites)  # normalized log citations

        positions[key] = (x, y, z)

    # --- Build Plotly traces ---
    traces = []

    # 1. Citation arcs (Bezier curves)
    for citing_key, cited_key in citation_edges:
        if citing_key not in positions or cited_key not in positions:
            continue
        p0 = positions[citing_key]
        p1 = positions[cited_key]
        citing_comm = node_community.get(citing_key, 0)
        color = _get_cluster_color(citing_comm)

        bx, by, bz = _bezier_3d(p0, p1, height_factor=0.08, n_points=20,
                               max_height=0.3)
        traces.append(go.Scatter3d(
            x=bx.tolist(), y=by.tolist(), z=bz.tolist(),
            mode="lines",
            line=dict(width=1.5, color=color),
            opacity=0.25,
            hoverinfo="skip",
            showlegend=False,
        ))

    # 2. Article nodes (one trace per cluster)
    for comm in communities:
        comm_articles = [a for a in selected
                         if node_community.get(a.get("key", a.get("doi", "")), 0) == comm]
        if not comm_articles:
            continue

        color = _get_cluster_color(comm)
        label = cluster_labels.get(comm, f"Cluster {comm}")

        xs, ys, zs, sizes, hovers, texts = [], [], [], [], [], []
        for a in comm_articles:
            key = a.get("key", a.get("doi", ""))
            if key not in positions:
                continue
            x, y, z = positions[key]
            cites = a.get("cited_by_count", 0)
            title = (a.get("title") or "")[:70]
            year = a.get("publication_year") or a.get("year") or ""

            xs.append(x)
            ys.append(y)
            zs.append(z)
            sizes.append(max(4, min(25, math.sqrt(cites + 1) * 3)))
            hovers.append(
                f"<b>{title}</b><br>"
                f"Ano: {year} | Citas: {cites}<br>"
                f"Cluster: {label}"
            )
            # Show label only for top-cited articles
            if cites >= sorted([a2.get("cited_by_count", 0) for a2 in comm_articles],
                               reverse=True)[min(2, len(comm_articles) - 1)]:
                # First author + year
                authors = a.get("authors_parsed") or a.get("authors", [])
                if isinstance(authors, list) and authors:
                    first = authors[0]
                    if isinstance(first, dict):
                        first = first.get("family", first.get("name", ""))
                    elif isinstance(first, str):
                        first = first.split(",")[0].strip()
                    texts.append(f"{first} ({year})")
                else:
                    texts.append(f"({year})")
            else:
                texts.append("")

        traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="markers+text",
            marker=dict(
                size=sizes, color=color, opacity=0.85,
                line=dict(width=0.5, color="white"),
            ),
            text=texts,
            textposition="top center",
            textfont=dict(size=7, color="#333"),
            hovertext=hovers,
            hoverinfo="text",
            name=f"#{comm}: {label[:40]}",
        ))

    # 3. Cluster label annotations on Y axis
    annotations = []
    for comm in communities:
        y_pos = comm_to_y[comm]
        label = cluster_labels.get(comm, f"#{comm}")
        annotations.append(dict(
            x=year_min - 0.5,
            y=y_pos,
            z=0,
            text=f"<b>#{comm}</b> {label[:30]}",
            showarrow=False,
            font=dict(size=9, color=_get_cluster_color(comm)),
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text="Citation Chronotope 3D<br>"
                 "<sub>Time × Thematic Cluster × Impact — arcs = internal citations</sub>",
            x=0.5,
        ),
        scene=dict(
            xaxis=dict(title="Publication Year",
                       tickvals=list(range(year_min, year_max + 1, max(1, year_range // 8)))),
            yaxis=dict(title="Thematic Cluster",
                       tickvals=list(range(n_clusters)),
                       ticktext=[cluster_labels.get(c, f"#{c}")[:25] for c in communities]),
            zaxis=dict(title="Impact (log-citations normalized)", range=[0, 1.15]),
            camera=dict(eye=dict(x=1.8, y=-1.2, z=0.8)),
            annotations=annotations,
        ),
        legend=dict(
            yanchor="top", y=0.95, xanchor="left", x=0.01,
            bgcolor="rgba(255,255,255,0.85)", font=dict(size=10),
        ),
        width=1920, height=1200,
        margin=dict(l=0, r=0, t=100, b=0),
    )

    _save_figure(fig, output_dir, "citation_chronotope_3d")
    logger.info(f"Citation Chronotope 3D: {len(selected)} articles, "
                f"{len(citation_edges)} citation arcs, {n_clusters} clusters")
    return output_dir / "citation_chronotope_3d.html"


# ===================================================================
# ANALISIS 2: Thematic Landscape 3D
# ===================================================================

def plot_thematic_landscape_3d(
    articles: List[dict],
    networks: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Path] = None,
    n_periods: int = 3,
) -> Optional[Path]:
    """Thematic Landscape 3D — Mapa tematico estrategico evolutivo.

    Extiende el mapa de Callon 2D (Centrality x Density) a 3D anadiendo
    el eje temporal:
    - X = Centralidad de Callon (importancia externa del cluster)
    - Y = Densidad de Callon (cohesion interna del cluster)
    - Z = Periodo temporal

    Cada cluster es una esfera cuyo tamano refleja la frecuencia total
    de keywords. Trayectorias conectan el mismo cluster entre periodos.

    Cuadrantes del mapa tematico de Callon:
    - Q1 (alta centralidad, alta densidad): Temas motores
    - Q2 (baja centralidad, alta densidad): Temas nicho
    - Q3 (baja centralidad, baja densidad): Temas emergentes/declive
    - Q4 (alta centralidad, baja densidad): Temas basicos/transversales

    Args:
        articles: Lista de articulos enriquecidos
        networks: Dict con grafos NetworkX (opcional, reconstruye si falta)
        output_dir: Directorio de salida
        n_periods: Numero de periodos temporales
    """
    if not HAS_PLOTLY or not HAS_NX:
        logger.warning("Plotly/NetworkX no disponible; omitiendo Thematic Landscape 3D")
        return None

    if output_dir is None:
        from config import RDM_DIR
        output_dir = RDM_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building Thematic Landscape 3D...")

    # --- Segment corpus into temporal periods ---
    years = [a.get("publication_year") or a.get("year") or 2020 for a in articles]
    valid_years = sorted(set(y for y in years if y > 1950))
    if len(valid_years) < 2:
        logger.warning("Insufficient temporal range for Thematic Landscape 3D")
        return None

    year_min, year_max = min(valid_years), max(valid_years)
    span = year_max - year_min + 1
    period_size = max(1, span // n_periods)

    periods = []
    for p in range(n_periods):
        start = year_min + p * period_size
        end = start + period_size - 1 if p < n_periods - 1 else year_max
        periods.append((start, end))

    logger.info(f"Temporal periods: {periods}")

    # --- For each period, build keyword co-occurrence network and compute Callon metrics ---
    period_clusters = []  # list of dicts per period

    for period_idx, (p_start, p_end) in enumerate(periods):
        period_articles = [a for a in articles
                           if p_start <= (a.get("publication_year") or a.get("year") or 0) <= p_end]
        if len(period_articles) < 5:
            logger.info(f"Period {p_start}-{p_end}: only {len(period_articles)} articles, skipping")
            period_clusters.append([])
            continue

        # Build keyword co-occurrence network for this period
        G_kw = nx.Graph()
        kw_freq = Counter()
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

        # Add frequency as node attribute
        for kw, freq in kw_freq.items():
            if kw in G_kw:
                G_kw.nodes[kw]["frequency"] = freq

        # Filter: keep only keywords with freq >= 2
        low_freq = [n for n in G_kw if kw_freq.get(n, 0) < 2]
        G_kw.remove_nodes_from(low_freq)

        if len(G_kw) < 3:
            period_clusters.append([])
            continue

        # Community detection
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(G_kw, random_state=42)
        except ImportError:
            # Fallback: connected components as communities
            partition = {}
            for idx, comp in enumerate(nx.connected_components(G_kw)):
                for node in comp:
                    partition[node] = idx

        # Compute Callon centrality and density per cluster
        clusters_in_period = []
        cluster_ids = set(partition.values())

        for cid in cluster_ids:
            members = [n for n, c in partition.items() if c == cid]
            if len(members) < 2:
                continue

            # Callon et al. (1991): misma implementacion que la Fase 3b
            from src.strategic_thematic_analysis import callon_centrality_density
            centrality, density = callon_centrality_density(G_kw, members, kw_freq)

            # Total frequency
            total_freq = sum(kw_freq.get(m, 0) for m in members)

            # Top keywords for label
            sorted_members = sorted(members, key=lambda m: kw_freq.get(m, 0), reverse=True)
            label = "; ".join(sorted_members[:3])

            clusters_in_period.append({
                "period_idx": period_idx,
                "period_label": f"{p_start}-{p_end}",
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
        logger.info(f"Period {p_start}-{p_end}: {len(clusters_in_period)} clusters, "
                     f"{len(period_articles)} articles")

    # --- Track clusters across periods by Jaccard similarity ---
    # Assign consistent IDs
    global_cluster_id = 0
    cluster_tracking = {}  # (period_idx, local_cid) -> global_id

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
                best_jaccard = 0.3  # minimum threshold
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

    # --- Build Plotly traces ---
    traces = []

    # Normalize centrality and density across all periods for consistent axes
    all_centralities = [c["centrality"] for cs in period_clusters for c in cs]
    all_densities = [c["density"] for cs in period_clusters for c in cs]
    if not all_centralities:
        logger.warning("No clusters found for Thematic Landscape 3D")
        return None

    max_cent = max(all_centralities) if all_centralities else 1
    max_dens = max(all_densities) if all_densities else 1
    max_freq = max((c["total_frequency"] for cs in period_clusters for c in cs), default=1)

    # Median lines for quadrant classification
    med_cent = float(np.median(all_centralities)) if all_centralities else 0
    med_dens = float(np.median(all_densities)) if all_densities else 0

    # 1. Cluster spheres
    for period_idx, clusters in enumerate(period_clusters):
        for c in clusters:
            gid = c.get("global_id", 0)
            color = _get_cluster_color(gid)

            # Quadrant classification
            if c["centrality"] >= med_cent and c["density"] >= med_dens:
                quadrant = "Motor Themes"
            elif c["centrality"] < med_cent and c["density"] >= med_dens:
                quadrant = "Niche Themes"
            elif c["centrality"] < med_cent and c["density"] < med_dens:
                quadrant = "Emerging/Declining"
            else:
                quadrant = "Basic/Transversal"

            size = max(8, min(40, math.sqrt(c["total_frequency"]) * 4))

            traces.append(go.Scatter3d(
                x=[c["centrality"]],
                y=[c["density"]],
                z=[period_idx],
                mode="markers+text",
                marker=dict(
                    size=size, color=color, opacity=0.8,
                    line=dict(width=1, color="white"),
                ),
                text=[c["label"][:25]],
                textposition="top center",
                textfont=dict(size=8, color="#333"),
                hovertext=[
                    f"<b>{c['label']}</b><br>"
                    f"Period: {c['period_label']}<br>"
                    f"Centrality: {c['centrality']:.3f}<br>"
                    f"Density: {c['density']:.3f}<br>"
                    f"Keywords: {c['n_keywords']}<br>"
                    f"Total frequency: {c['total_frequency']}<br>"
                    f"Quadrant: {quadrant}<br>"
                    f"Top: {', '.join(c['top_keywords'][:5])}"
                ],
                hoverinfo="text",
                name=f"{c['period_label']}: {c['label'][:30]}",
                showlegend=False,
            ))

    # 2. Trajectory lines connecting same cluster across periods
    # Group by global_id
    gid_positions = defaultdict(list)
    for period_idx, clusters in enumerate(period_clusters):
        for c in clusters:
            gid = c.get("global_id", -1)
            gid_positions[gid].append({
                "period_idx": period_idx,
                "centrality": c["centrality"],
                "density": c["density"],
                "label": c["label"],
            })

    for gid, positions_list in gid_positions.items():
        if len(positions_list) < 2:
            continue
        positions_list.sort(key=lambda p: p["period_idx"])
        color = _get_cluster_color(gid)

        xs = [p["centrality"] for p in positions_list]
        ys = [p["density"] for p in positions_list]
        zs = [p["period_idx"] for p in positions_list]

        traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines+markers",
            line=dict(width=4, color=color),
            marker=dict(size=3, color=color),
            hoverinfo="skip",
            name=f"Trayectoria: {positions_list[0]['label'][:25]}",
            showlegend=True,
        ))

        # Add arrow annotation at the end
        if len(xs) >= 2:
            dx = xs[-1] - xs[-2]
            dy = ys[-1] - ys[-2]
            direction = ""
            if dx > 0.01:
                direction += "\u2192 more central"
            elif dx < -0.01:
                direction += "\u2190 less central"
            if dy > 0.01:
                direction += " \u2191 denser"
            elif dy < -0.01:
                direction += " \u2193 less dense"
            if direction:
                traces.append(go.Scatter3d(
                    x=[xs[-1]], y=[ys[-1]], z=[zs[-1]],
                    mode="text",
                    text=[direction],
                    textfont=dict(size=7, color=color),
                    textposition="bottom center",
                    hoverinfo="skip",
                    showlegend=False,
                ))

    # 3. Median reference planes (per period)
    for period_idx in range(len(periods)):
        # Vertical line at median centrality
        traces.append(go.Scatter3d(
            x=[med_cent, med_cent],
            y=[0, max_dens * 1.1],
            z=[period_idx, period_idx],
            mode="lines",
            line=dict(width=1, color="rgba(128,128,128,0.3)", dash="dash"),
            hoverinfo="skip", showlegend=False,
        ))
        # Horizontal line at median density
        traces.append(go.Scatter3d(
            x=[0, max_cent * 1.1],
            y=[med_dens, med_dens],
            z=[period_idx, period_idx],
            mode="lines",
            line=dict(width=1, color="rgba(128,128,128,0.3)", dash="dash"),
            hoverinfo="skip", showlegend=False,
        ))

    # 4. Quadrant labels per period
    period_labels = [f"{s}-{e}" for s, e in periods]

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text="Thematic Landscape 3D<br>"
                 "<sub>Callon Centrality × Callon Density × Temporal Period</sub>",
            x=0.5,
        ),
        scene=dict(
            xaxis=dict(title="Callon Centrality (external importance)"),
            yaxis=dict(title="Callon Density (internal cohesion)"),
            zaxis=dict(
                title="Period",
                tickvals=list(range(len(periods))),
                ticktext=period_labels,
            ),
            camera=dict(eye=dict(x=1.5, y=-1.5, z=0.9)),
        ),
        legend=dict(
            yanchor="top", y=0.95, xanchor="left", x=0.01,
            bgcolor="rgba(255,255,255,0.85)", font=dict(size=10),
        ),
        width=1920, height=1200,
        margin=dict(l=0, r=0, t=100, b=0),
    )

    _save_figure(fig, output_dir, "thematic_landscape_3d")
    total_clusters = sum(len(cs) for cs in period_clusters)
    n_tracked = sum(1 for v in gid_positions.values() if len(v) >= 2)
    logger.info(f"Thematic Landscape 3D: {total_clusters} cluster-period instances, "
                f"{n_tracked} tracked trajectories across periods")
    return output_dir / "thematic_landscape_3d.html"


# ===================================================================
# ANALISIS 3: Cluster Evolution Timeline 3D (estilo CiteSpace ejemplo 05)
# ===================================================================

def plot_cluster_evolution_3d(
    articles: List[dict],
    networks: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Cluster Evolution Timeline 3D — Replica el estilo CiteSpace ejemplo 05.

    Similar al timeline CiteSpace pero en 3D:
    - X = Ano de publicacion (timeline horizontal)
    - Y = Cluster ID (bandas horizontales, una por cluster)
    - Z = Burst strength del keyword (elevacion = intensidad de burst)

    Arcos curvos 3D conectan keywords que co-ocurren entre clusters
    diferentes, mostrando puentes tematicos.

    Args:
        articles: Lista de articulos enriquecidos
        networks: Dict con grafos NetworkX
        output_dir: Directorio de salida
    """
    if not HAS_PLOTLY or not HAS_NX:
        logger.warning("Plotly/NetworkX no disponible; omitiendo Cluster Evolution 3D")
        return None

    if output_dir is None:
        from config import RDM_DIR
        output_dir = RDM_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building Cluster Evolution Timeline 3D...")

    # --- Build keyword co-occurrence network ---
    G_kw = nx.Graph()
    kw_freq = Counter()
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
        logger.warning("Insufficient keywords for Cluster Evolution 3D")
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

    # --- Compute burst levels per keyword ---
    kw_year_counts: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for a in articles:
        year = a.get("publication_year") or a.get("year")
        if not year:
            continue
        for kw in _extract_keywords(a):
            if kw in partition:  # only tracked keywords
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
        logger.info("Burst detection not available; Z axis will use frequency")

    # --- Cluster labels ---
    cluster_members = defaultdict(list)
    for kw, cid in partition.items():
        cluster_members[cid].append(kw)

    cluster_labels = {}
    for cid, members in cluster_members.items():
        sorted_m = sorted(members, key=lambda m: kw_freq.get(m, 0), reverse=True)
        cluster_labels[cid] = "; ".join(sorted_m[:3])

    # Sort clusters by size — use spacing factor to stretch Y axis
    sorted_clusters = sorted(cluster_members.keys(),
                             key=lambda c: len(cluster_members[c]), reverse=True)
    Y_SPACING = 1.8  # stretch cluster bands apart
    cluster_y = {c: idx * Y_SPACING for idx, c in enumerate(sorted_clusters)}
    n_clusters = len(sorted_clusters)

    # --- Max burst for Z normalization ---
    max_burst = max(kw_burst_strength.values(), default=1)
    max_burst = max(1, max_burst)
    max_freq_log = math.log(1 + max(kw_freq.values(), default=1))

    # --- Build traces ---
    traces = []

    # 1. Cross-cluster co-occurrence arcs
    cross_edges = []
    for u, v, d in G_kw.edges(data=True):
        if partition.get(u) != partition.get(v):
            w = d.get("weight", 1)
            if w >= 2:  # only significant co-occurrences
                cross_edges.append((u, v, w))

    # Sort by weight and take top edges to avoid clutter
    cross_edges.sort(key=lambda e: e[2], reverse=True)
    cross_edges = cross_edges[:min(100, len(cross_edges))]

    for u, v, w in cross_edges:
        year_u = kw_first_year.get(u, 2020)
        year_v = kw_first_year.get(v, 2020)
        cid_u = partition[u]
        cid_v = partition[v]
        y_u = cluster_y.get(cid_u, 0)
        y_v = cluster_y.get(cid_v, 0)

        # Z position based on burst or frequency — lift baseline slightly
        Z_LIFT = 0.15
        z_u = Z_LIFT + (kw_burst_strength.get(u, 0) / max_burst if max_burst > 0 else 0)
        z_v = Z_LIFT + (kw_burst_strength.get(v, 0) / max_burst if max_burst > 0 else 0)
        if not kw_burst_strength:  # fallback to log frequency
            z_u = Z_LIFT + math.log(1 + kw_freq.get(u, 0)) / max_freq_log
            z_v = Z_LIFT + math.log(1 + kw_freq.get(v, 0)) / max_freq_log

        p0 = (year_u, y_u, z_u)
        p1 = (year_v, y_v, z_v)
        bx, by, bz = _bezier_3d(p0, p1, height_factor=0.12, n_points=30,
                                max_height=0.35)

        color_u = _get_cluster_color(cid_u)
        traces.append(go.Scatter3d(
            x=bx.tolist(), y=by.tolist(), z=bz.tolist(),
            mode="lines",
            line=dict(width=max(1, min(4, w * 0.5)), color=color_u),
            opacity=0.3,
            hoverinfo="skip",
            showlegend=False,
        ))

    # 2. Keyword nodes per cluster
    for cid in sorted_clusters:
        members = cluster_members[cid]
        color = _get_cluster_color(cid)
        label = cluster_labels.get(cid, f"#{cid}")
        y_base = cluster_y[cid]

        xs, ys, zs, sizes, hovers, texts = [], [], [], [], [], []

        for kw in members:
            year = kw_first_year.get(kw, 2020)
            freq = kw_freq.get(kw, 1)

            # Z: burst strength or frequency — lifted baseline
            Z_LIFT = 0.15
            if kw_burst_strength:
                z = Z_LIFT + kw_burst_strength.get(kw, 0) / max_burst
            else:
                z = Z_LIFT + math.log(1 + freq) / max_freq_log

            # Jitter Y within cluster band (wider with Y_SPACING)
            jitter = (hash(kw) % 100) / 100 * 0.8 - 0.4
            y = y_base + jitter

            xs.append(year)
            ys.append(y)
            zs.append(z)
            sizes.append(max(3, min(20, math.sqrt(freq) * 3)))

            burst_info = f"Burst: {kw_burst_strength.get(kw, 0):.1f}" if kw in kw_burst_strength else "No burst"
            hovers.append(
                f"<b>{kw}</b><br>"
                f"Cluster: {label}<br>"
                f"First appearance: {year}<br>"
                f"Frequency: {freq}<br>"
                f"{burst_info}"
            )

            # Show text only for high-frequency keywords
            if freq >= sorted([kw_freq.get(m, 0) for m in members],
                              reverse=True)[min(4, len(members) - 1)]:
                texts.append(kw[:20])
            else:
                texts.append("")

        traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="markers+text",
            marker=dict(
                size=sizes, color=color, opacity=0.8,
                line=dict(width=0.5, color="white"),
            ),
            text=texts,
            textposition="top center",
            textfont=dict(size=7, color="#333"),
            hovertext=hovers,
            hoverinfo="text",
            name=f"#{cid}: {label[:35]}",
        ))

    # 3. Cluster band separators
    years_range = sorted(kw_first_year.values())
    if years_range:
        year_min_kw, year_max_kw = years_range[0], years_range[-1]
        for i in range(n_clusters - 1):
            y_sep = (i + 0.5) * Y_SPACING
            traces.append(go.Scatter3d(
                x=[year_min_kw, year_max_kw],
                y=[y_sep, y_sep],
                z=[0, 0],
                mode="lines",
                line=dict(width=1, color="rgba(200,200,200,0.4)", dash="dot"),
                hoverinfo="skip", showlegend=False,
            ))

    fig = go.Figure(data=traces)

    z_title = "Burst Strength (normalized)" if kw_burst_strength else "Frequency (log-normalized)"

    fig.update_layout(
        title=dict(
            text="Cluster Evolution Timeline 3D<br>"
                 "<sub>Time × Thematic Cluster × Burst — arcs = inter-cluster co-occurrences</sub>",
            x=0.5,
        ),
        scene=dict(
            xaxis=dict(title="First Appearance (year)"),
            yaxis=dict(
                title="",
                tickvals=[i * Y_SPACING for i in range(n_clusters)],
                ticktext=[f"#{c}: {cluster_labels.get(c, '')[:20]}" for c in sorted_clusters],
                tickfont=dict(size=11),
            ),
            zaxis=dict(title=z_title, range=[0, 1.5]),
            camera=dict(eye=dict(x=2.0, y=-1.0, z=0.7)),
        ),
        legend=dict(
            yanchor="top", y=0.95, xanchor="left", x=0.01,
            bgcolor="rgba(255,255,255,0.90)", font=dict(size=13),
            bordercolor="#cccccc", borderwidth=1,
        ),
        width=1920, height=1200,
        margin=dict(l=0, r=0, t=100, b=0),
    )

    _save_figure(fig, output_dir, "cluster_evolution_3d")
    n_kw = sum(len(cluster_members[c]) for c in sorted_clusters)
    logger.info(f"Cluster Evolution 3D: {n_kw} keywords, {n_clusters} clusters, "
                f"{len(cross_edges)} cross-cluster arcs")
    return output_dir / "cluster_evolution_3d.html"


# ===================================================================
# Entry point
# ===================================================================

def run_rdm_visualizations_3d(
    articles: List[dict],
    networks: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Path] = None,
) -> dict:
    """Ejecuta las tres visualizaciones 3D adicionales.

    Returns:
        Dict con estadisticas de las visualizaciones generadas.
    """
    if output_dir is None:
        from config import RDM_DIR
        output_dir = RDM_DIR

    results = {"charts_saved": 0}

    # 1. Citation Chronotope 3D
    try:
        path = plot_citation_chronotope_3d(articles, networks, output_dir)
        if path:
            results["citation_chronotope"] = str(path)
            results["charts_saved"] += 1
    except Exception as e:
        logger.error(f"Citation Chronotope 3D failed: {e}")

    # 2. Thematic Landscape 3D
    try:
        path = plot_thematic_landscape_3d(articles, networks, output_dir)
        if path:
            results["thematic_landscape"] = str(path)
            results["charts_saved"] += 1
    except Exception as e:
        logger.error(f"Thematic Landscape 3D failed: {e}")

    # 3. Cluster Evolution Timeline 3D
    try:
        path = plot_cluster_evolution_3d(articles, networks, output_dir)
        if path:
            results["cluster_evolution"] = str(path)
            results["charts_saved"] += 1
    except Exception as e:
        logger.error(f"Cluster Evolution 3D failed: {e}")

    return results
