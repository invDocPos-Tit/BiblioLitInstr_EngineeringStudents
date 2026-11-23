"""Visualizaciones avanzadas con Plotly: Sankey, historiografo, mapa mundial, redes interactivas."""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np

from src.utils import resolve_openalex_label_cached, safe_write_plotly_png

logger = logging.getLogger("bibliometrico.advviz")

# Paleta de colores para clusters
_CLUSTER_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#ff7f00", "#984ea3",
    "#a65628", "#f781bf", "#66c2a5", "#fc8d62", "#8da0cb",
    "#e78ac3", "#a6d854", "#ffd92f", "#e5c494", "#1b9e77",
    "#d95f02", "#7570b3", "#e7298a", "#999999", "#b3b3b3",
]


def _save_plotly_figure(fig, output_dir: Path, name: str) -> None:
    """Guarda figura Plotly como HTML y opcionalmente como PNG."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # HTML interactivo
    html_path = output_dir / f"{name}.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    logger.info(f"Saved: {html_path}")

    # PNG estatico (protegido contra cuelgues de Kaleido/Chrome)
    png_path = output_dir / f"{name}.png"
    safe_write_plotly_png(fig, png_path, width=1600, height=1000, scale=2, log=logger)


# ===========================================================================
# 1. SANKEY TEMATICO - Evolucion de keywords por periodos
# ===========================================================================

def create_thematic_sankey(articles: list[dict], output_dir: Path,
                           periods: list[tuple[int, int]] | None = None,
                           top_n: int = 18) -> None:
    """Crea diagrama Sankey de evolucion tematica por periodos.

    Args:
        articles: Lista de articulos enriquecidos.
        output_dir: Directorio de salida para archivos Plotly.
        periods: Lista de tuplas (year_start, year_end). Si None, auto-calcula 3 periodos.
        top_n: Numero de keywords top por periodo.
    """
    import plotly.graph_objects as go

    years = [a["publication_year"] for a in articles if a.get("publication_year")]
    if not years:
        logger.warning("No year data for Sankey")
        return

    min_year, max_year = min(years), max(years)

    if periods is None:
        span = max_year - min_year + 1
        if span <= 3:
            periods = [(min_year, max_year)]
        elif span <= 5:
            mid = min_year + span // 2
            periods = [(min_year, mid - 1), (mid, max_year)]
        else:
            t1 = min_year + span // 3
            t2 = t1 + span // 3
            periods = [(min_year, t1), (t1 + 1, t2), (t2 + 1, max_year)]

    if len(periods) < 2:
        logger.info("Only 1 period, skipping Sankey (need >=2)")
        return

    # Extraer keywords por periodo
    period_keywords: list[Counter] = []
    for start, end in periods:
        kw_counter = Counter()
        for a in articles:
            yr = a.get("publication_year", 0)
            if start <= yr <= end:
                for kw in a.get("manual_tags", []):
                    kw_counter[kw.lower().strip()] += 1
        period_keywords.append(kw_counter)

    # Top keywords por periodo
    period_top: list[list[str]] = []
    for counter in period_keywords:
        top = [kw for kw, _ in counter.most_common(top_n)]
        period_top.append(top)

    # Construir nodos y links del Sankey
    node_labels = []
    node_colors = []
    node_index = {}  # (period_idx, keyword) -> index

    for p_idx, top_kws in enumerate(period_top):
        for kw in top_kws:
            key = (p_idx, kw)
            if key not in node_index:
                node_index[key] = len(node_labels)
                # Color por cluster: hash del keyword para consistencia
                color_idx = hash(kw) % len(_CLUSTER_COLORS)
                node_labels.append(f"{kw} ({periods[p_idx][0]}-{periods[p_idx][1]})")
                node_colors.append(_CLUSTER_COLORS[color_idx])

    sources, targets, values, link_colors = [], [], [], []

    for p_idx in range(len(periods) - 1):
        kws_current = set(period_top[p_idx])
        kws_next = set(period_top[p_idx + 1])

        for kw in kws_current:
            if kw in kws_next:
                src = node_index.get((p_idx, kw))
                tgt = node_index.get((p_idx + 1, kw))
                if src is not None and tgt is not None:
                    # Valor = min de frecuencias en ambos periodos
                    val = min(period_keywords[p_idx][kw], period_keywords[p_idx + 1][kw])
                    sources.append(src)
                    targets.append(tgt)
                    values.append(max(val, 1))
                    color_idx = hash(kw) % len(_CLUSTER_COLORS)
                    # Link color semitransparente
                    hex_c = _CLUSTER_COLORS[color_idx]
                    link_colors.append(f"rgba({int(hex_c[1:3],16)},{int(hex_c[3:5],16)},{int(hex_c[5:7],16)},0.4)")

    if not sources:
        logger.warning("No keyword overlaps between periods for Sankey")
        return

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=20,
            thickness=25,
            line=dict(color="white", width=1),
            label=node_labels,
            color=node_colors,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
        ),
    )])

    period_labels = [f"{s}-{e}" for s, e in periods]
    fig.update_layout(
        title=dict(
            text=f"Thematic Evolution: {' → '.join(period_labels)}<br>"
                 f"<sub>Top {top_n} keywords per period · Links show persistent themes</sub>",
            font=dict(size=18),
        ),
        font=dict(size=12),
        width=1400,
        height=800,
        paper_bgcolor="white",
    )

    _save_plotly_figure(fig, output_dir, "sankey_thematic_evolution")


# ===========================================================================
# 2. HISTORIOGRAFO - Red de citas dentro del corpus
# ===========================================================================

def create_historiograph(articles: list[dict], output_dir: Path,
                         top_n: int = 45) -> None:
    """Crea historiografo de citas internas del corpus.

    Muestra los articulos mas citados conectados por relaciones de citacion
    dentro del corpus. X = año, Y = posicion escalonada.
    """
    import plotly.graph_objects as go

    # Construir mapa OpenAlex ID -> articulo
    id_to_article = {}
    for a in articles:
        oa_id = a.get("openalex", {}).get("openalex_id", "")
        if oa_id:
            id_to_article[oa_id] = a

    # Seleccionar top N por citaciones
    sorted_articles = sorted(articles, key=lambda x: x.get("cited_by_count", 0), reverse=True)
    top_articles = sorted_articles[:top_n]
    top_ids = {a.get("openalex", {}).get("openalex_id", "") for a in top_articles}
    top_ids.discard("")

    if len(top_ids) < 3:
        logger.warning("Not enough articles with OpenAlex IDs for historiograph")
        return

    # Construir grafo dirigido: A -> B si A cita B (ambos en corpus)
    G = nx.DiGraph()
    for a in top_articles:
        oa_id = a.get("openalex", {}).get("openalex_id", "")
        if not oa_id:
            continue
        first_author = a["authors"][0].split(",")[0] if a.get("authors") else "Unknown"
        year = a.get("publication_year") or 0
        label = f"{first_author} ({year})"
        G.add_node(oa_id, label=label, year=year,
                   citations=a.get("cited_by_count") or 0,
                   title=(a.get("title") or "")[:60])

    for a in top_articles:
        oa_id = a.get("openalex", {}).get("openalex_id", "")
        if not oa_id:
            continue
        for ref_url in a.get("referenced_works", []):
            if ref_url in top_ids and ref_url != oa_id:
                G.add_edge(oa_id, ref_url)

    if G.number_of_edges() == 0:
        logger.info("No internal citations found among top articles for historiograph")
        return

    # Layout: X = año, Y = posicion escalonada
    year_nodes = defaultdict(list)
    for node in G.nodes():
        yr = G.nodes[node].get("year", 0)
        year_nodes[yr].append(node)

    pos = {}
    for yr, nodes in year_nodes.items():
        nodes_sorted = sorted(nodes, key=lambda n: G.nodes[n].get("citations", 0), reverse=True)
        for i, node in enumerate(nodes_sorted):
            pos[node] = (yr, -i)

    # Dibujar con Plotly
    edge_traces = []
    for u, v in G.edges():
        if u in pos and v in pos:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_traces.append(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode="lines",
                line=dict(width=1.2, color="rgba(150,150,150,0.5)"),
                hoverinfo="none",
                showlegend=False,
            ))

    # Nodos
    node_x, node_y, node_text, node_hover, node_size, node_color = [], [], [], [], [], []
    citations_list = [G.nodes[n].get("citations", 0) for n in G.nodes() if n in pos]
    max_cites = max(citations_list) if citations_list else 1

    for node in G.nodes():
        if node not in pos:
            continue
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        label = G.nodes[node].get("label", "")
        cites = G.nodes[node].get("citations", 0)
        title = G.nodes[node].get("title", "")
        node_text.append(label)
        node_hover.append(f"{label}<br>Citations: {cites}<br>{title}")
        node_size.append(12 + 30 * np.sqrt(cites / max(max_cites, 1)))
        node_color.append(G.nodes[node].get("year", 2020))

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        textfont=dict(size=9),
        hovertext=node_hover,
        hoverinfo="text",
        marker=dict(
            size=node_size,
            color=node_color,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Year", thickness=15),
            line=dict(width=1, color="white"),
        ),
    )

    fig = go.Figure(data=edge_traces + [node_trace])

    all_years = sorted(set(G.nodes[n].get("year", 0) for n in G.nodes()))
    fig.update_layout(
        title=dict(
            text=f"Citation Historiograph<br>"
                 f"<sub>Top {len(G.nodes())} articles · {G.number_of_edges()} citation links within corpus</sub>",
            font=dict(size=18),
        ),
        xaxis=dict(title="Publication Year", tickmode="array", tickvals=all_years,
                    gridcolor="rgba(200,200,200,0.3)"),
        yaxis=dict(title="", showticklabels=False, gridcolor="rgba(200,200,200,0.1)"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=1600,
        height=900,
        showlegend=False,
        hovermode="closest",
    )

    _save_plotly_figure(fig, output_dir, "historiograph")


# ===========================================================================
# 3. MAPA MUNDIAL DE COLABORACION
# ===========================================================================

def _iso2_to_iso3(code: str) -> str:
    """Convierte ISO 3166-1 alpha-2 a alpha-3. Fallback: devuelve el mismo codigo."""
    try:
        import pycountry
        c = pycountry.countries.get(alpha_2=code.upper())
        return c.alpha_3 if c else code
    except (ImportError, AttributeError):
        # Mapeo manual de los paises mas comunes
        _MAP = {
            "US": "USA", "CN": "CHN", "GB": "GBR", "DE": "DEU", "FR": "FRA",
            "JP": "JPN", "IN": "IND", "BR": "BRA", "AU": "AUS", "CA": "CAN",
            "IT": "ITA", "ES": "ESP", "KR": "KOR", "NL": "NLD", "SE": "SWE",
            "CH": "CHE", "PL": "POL", "TR": "TUR", "IR": "IRN", "RU": "RUS",
            "MX": "MEX", "SA": "SAU", "TW": "TWN", "SG": "SGP", "MY": "MYS",
            "TH": "THA", "PK": "PAK", "EG": "EGY", "ZA": "ZAF", "CL": "CHL",
            "CO": "COL", "AR": "ARG", "AT": "AUT", "BE": "BEL", "DK": "DNK",
            "FI": "FIN", "NO": "NOR", "PT": "PRT", "IE": "IRL", "CZ": "CZE",
            "HU": "HUN", "RO": "ROU", "GR": "GRC", "IL": "ISR", "NZ": "NZL",
            "HK": "HKG", "BD": "BGD", "VN": "VNM", "NG": "NGA", "PE": "PER",
        }
        return _MAP.get(code.upper(), code)


def _iso2_to_name(code: str) -> str:
    """Convierte ISO 3166-1 alpha-2 a nombre del pais."""
    try:
        import pycountry
        c = pycountry.countries.get(alpha_2=code.upper())
        return c.name if c else code
    except (ImportError, AttributeError):
        return code


def create_world_collaboration_map(articles: list[dict], output_dir: Path) -> None:
    """Crea mapa mundial choropleth con lineas de colaboracion.

    - Color por numero de articulos por pais.
    - Lineas entre paises que co-publican.
    """
    import plotly.graph_objects as go

    # Contar articulos por pais y co-autorias
    country_articles = Counter()
    country_pairs = Counter()

    for a in articles:
        authorships = a.get("openalex", {}).get("authorships", [])
        article_countries = set()
        for auth in authorships:
            for c in auth.get("countries", []):
                if c:
                    article_countries.add(c.upper())

        for c in article_countries:
            country_articles[c] += 1

        country_list = sorted(article_countries)
        for i in range(len(country_list)):
            for j in range(i + 1, len(country_list)):
                pair = (country_list[i], country_list[j])
                country_pairs[pair] += 1

    if not country_articles:
        logger.warning("No country data for world map")
        return

    # Choropleth
    iso3_codes = [_iso2_to_iso3(c) for c in country_articles.keys()]
    country_names = [_iso2_to_name(c) for c in country_articles.keys()]
    counts = list(country_articles.values())

    fig = go.Figure()

    fig.add_trace(go.Choropleth(
        locations=iso3_codes,
        z=counts,
        text=country_names,
        colorscale="Blues",
        colorbar=dict(title="Articles", thickness=15, len=0.5),
        marker_line_color="white",
        marker_line_width=0.5,
        hovertemplate="%{text}<br>Articles: %{z}<extra></extra>",
    ))

    # Lineas de colaboracion
    # Coordenadas aproximadas de paises (centroides)
    _CENTROIDS = {
        "US": (39.8, -98.6), "CN": (35.0, 105.0), "GB": (54.0, -2.0),
        "DE": (51.0, 10.0), "FR": (46.0, 2.0), "JP": (36.0, 138.0),
        "IN": (22.0, 78.0), "BR": (-10.0, -55.0), "AU": (-25.0, 135.0),
        "CA": (56.0, -106.0), "IT": (42.0, 12.0), "ES": (40.0, -4.0),
        "KR": (36.0, 128.0), "NL": (52.0, 5.0), "SE": (62.0, 15.0),
        "CH": (47.0, 8.0), "PL": (52.0, 20.0), "TR": (39.0, 35.0),
        "IR": (32.0, 53.0), "RU": (60.0, 100.0), "MX": (23.0, -102.0),
        "SA": (24.0, 45.0), "TW": (23.5, 121.0), "SG": (1.3, 103.8),
        "MY": (4.0, 109.5), "TH": (15.0, 100.0), "PK": (30.0, 69.0),
        "EG": (27.0, 30.0), "ZA": (-29.0, 24.0), "CL": (-33.0, -70.5),
        "CO": (4.0, -74.0), "AR": (-34.0, -64.0), "AT": (47.3, 13.3),
        "BE": (50.8, 4.3), "DK": (56.0, 10.0), "FI": (64.0, 26.0),
        "NO": (60.5, 8.5), "PT": (39.5, -8.0), "IE": (53.0, -8.0),
        "CZ": (49.8, 15.5), "HU": (47.0, 20.0), "RO": (46.0, 25.0),
        "GR": (39.0, 22.0), "IL": (31.0, 34.8), "NZ": (-41.0, 174.0),
        "BD": (24.0, 90.0), "VN": (16.0, 106.0), "NG": (10.0, 8.0),
    }

    if country_pairs:
        max_collab = max(country_pairs.values())
        for (c1, c2), count in country_pairs.most_common(50):
            coords1 = _CENTROIDS.get(c1)
            coords2 = _CENTROIDS.get(c2)
            if coords1 and coords2:
                width = 1 + 4 * (count / max(max_collab, 1))
                opacity = 0.3 + 0.5 * (count / max(max_collab, 1))
                fig.add_trace(go.Scattergeo(
                    lat=[coords1[0], coords2[0]],
                    lon=[coords1[1], coords2[1]],
                    mode="lines",
                    line=dict(width=width, color=f"rgba(255,69,0,{opacity})"),
                    hoverinfo="text",
                    text=f"{_iso2_to_name(c1)} - {_iso2_to_name(c2)}: {count} collaborations",
                    showlegend=False,
                ))

    fig.update_layout(
        title=dict(
            text=f"Global Research Collaboration Map<br>"
                 f"<sub>{len(country_articles)} countries · {sum(country_pairs.values())} collaboration links</sub>",
            font=dict(size=18),
        ),
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor="rgba(200,200,200,0.5)",
            projection_type="natural earth",
            showland=True,
            landcolor="rgb(243,243,243)",
            showocean=True,
            oceancolor="rgb(230,240,250)",
            showcountries=True,
            countrycolor="rgba(200,200,200,0.5)",
        ),
        width=1500,
        height=800,
        paper_bgcolor="white",
    )

    _save_plotly_figure(fig, output_dir, "world_collaboration_map")


# ===========================================================================
# 4. REDES INTERACTIVAS PLOTLY
# ===========================================================================

def create_interactive_network(G: nx.Graph, output_dir: Path, name: str,
                               title: str = "") -> None:
    """Convierte un grafo networkx existente a red interactiva Plotly.

    - Nodos con hover (titulo, citas, comunidad).
    - Labels de comunidad como anotaciones en centroide.
    - Colores por comunidad.
    """
    import plotly.graph_objects as go

    if G.number_of_nodes() < 3:
        logger.info(f"Network {name} too small for interactive visualization")
        return

    communities = nx.get_node_attributes(G, "community")
    node_labels = nx.get_node_attributes(G, "label")
    degrees = dict(G.degree(weight="weight"))
    max_deg = max(degrees.values()) if degrees else 1

    # Layout
    if communities:
        # Use community-aware layout
        comm_nodes = defaultdict(list)
        for n, c in communities.items():
            comm_nodes[c].append(n)

        n_comms = len(comm_nodes)
        comm_centers = {}
        radius = 3.0
        for i, c in enumerate(sorted(comm_nodes.keys())):
            angle = 2 * np.pi * i / max(n_comms, 1)
            comm_centers[c] = np.array([radius * np.cos(angle), radius * np.sin(angle)])

        pos = {}
        for c, nodes in comm_nodes.items():
            if len(nodes) == 1:
                pos[nodes[0]] = comm_centers[c] + np.random.randn(2) * 0.05
                continue
            subG = G.subgraph(nodes)
            k = 1.0 / np.sqrt(max(len(nodes), 1))
            local_pos = nx.spring_layout(subG, k=k, seed=42, iterations=80, weight="weight")
            coords = np.array(list(local_pos.values()))
            spread = max(np.ptp(coords, axis=0).max(), 0.01)
            scale = min(0.8, 0.3 + 0.1 * np.sqrt(len(nodes)))
            for n in nodes:
                normalized = (local_pos[n] - coords.mean(axis=0)) / spread * scale
                pos[n] = comm_centers[c] + normalized
    else:
        k = 3.0 / np.sqrt(max(G.number_of_nodes(), 1))
        pos = nx.spring_layout(G, k=k, seed=42, iterations=150, weight="weight")

    # Edges
    edge_x, edge_y = [], []
    edge_weights = [d.get("weight", 1) for _, _, d in G.edges(data=True)]
    max_ew = max(edge_weights) if edge_weights else 1
    # Filter to top 30% edges for clarity
    threshold = np.percentile(edge_weights, 70) if len(edge_weights) > 10 else 0
    for u, v, d in G.edges(data=True):
        if d.get("weight", 1) >= threshold and u in pos and v in pos:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=0.5, color="rgba(180,180,180,0.3)"),
        hoverinfo="none",
        showlegend=False,
    )

    # Nodes by community
    node_traces = []
    if communities:
        comm_groups = defaultdict(list)
        for n in G.nodes():
            c = communities.get(n, 0)
            comm_groups[c].append(n)

        for c, nodes in sorted(comm_groups.items()):
            nx_list, ny_list, texts, hovers, sizes = [], [], [], [], []
            for n in nodes:
                if n not in pos:
                    continue
                x, y = pos[n]
                nx_list.append(x)
                ny_list.append(y)
                label = node_labels.get(n, str(n))
                if not label or "openalex.org/" in label:
                    label = resolve_openalex_label_cached(n)
                deg = degrees.get(n, 0)
                texts.append(label if deg > max_deg * 0.3 else "")
                title_attr = G.nodes[n].get("title", "")
                cites = G.nodes[n].get("citations", G.nodes[n].get("cocitation_count", ""))
                hover = f"<b>{label}</b><br>Community: {c}<br>Degree: {deg}"
                if title_attr:
                    hover += f"<br>{title_attr}"
                if cites:
                    hover += f"<br>Citations: {cites}"
                hovers.append(hover)
                sizes.append(8 + 25 * np.sqrt(deg / max(max_deg, 1)))

            color = _CLUSTER_COLORS[c % len(_CLUSTER_COLORS)]
            node_traces.append(go.Scatter(
                x=nx_list, y=ny_list,
                mode="markers+text",
                text=texts,
                textposition="top center",
                textfont=dict(size=9, color="#333"),
                hovertext=hovers,
                hoverinfo="text",
                marker=dict(
                    size=sizes,
                    color=color,
                    line=dict(width=1, color="white"),
                    opacity=0.85,
                ),
                name=f"Cluster {c} (n={len(nodes)})",
            ))
    else:
        # Single trace, no communities
        nx_list, ny_list, texts, hovers, sizes = [], [], [], [], []
        for n in G.nodes():
            if n not in pos:
                continue
            x, y = pos[n]
            nx_list.append(x)
            ny_list.append(y)
            label = node_labels.get(n, str(n))
            if not label or "openalex.org/" in label:
                label = resolve_openalex_label_cached(n)
            deg = degrees.get(n, 0)
            texts.append(label if deg > max_deg * 0.3 else "")
            hovers.append(f"<b>{label}</b><br>Degree: {deg}")
            sizes.append(8 + 25 * np.sqrt(deg / max(max_deg, 1)))

        node_traces.append(go.Scatter(
            x=nx_list, y=ny_list,
            mode="markers+text",
            text=texts,
            textposition="top center",
            textfont=dict(size=9),
            hovertext=hovers,
            hoverinfo="text",
            marker=dict(
                size=sizes,
                color="#4e79a7",
                line=dict(width=1, color="white"),
            ),
            showlegend=False,
        ))

    fig = go.Figure(data=[edge_trace] + node_traces)

    # Community label annotations at centroids
    if communities:
        annotations = []
        comm_nodes_dict = defaultdict(list)
        for n, c in communities.items():
            if n in pos:
                comm_nodes_dict[c].append(n)

        for c, nodes in comm_nodes_dict.items():
            if len(nodes) < 2:
                continue
            cx = np.mean([pos[n][0] for n in nodes])
            cy = np.mean([pos[n][1] for n in nodes])
            # Find most representative keyword/label for this community
            top_node = max(nodes, key=lambda n: degrees.get(n, 0))
            comm_label = node_labels.get(top_node, f"C{c}")
            annotations.append(dict(
                x=cx, y=cy,
                text=f"<b>Cluster {c}</b>",
                showarrow=False,
                font=dict(size=13, color=_CLUSTER_COLORS[c % len(_CLUSTER_COLORS)]),
                opacity=0.6,
            ))
        fig.update_layout(annotations=annotations)

    n_clusters = len(set(communities.values())) if communities else "N/A"
    display_title = title or name.replace("_", " ").title()
    fig.update_layout(
        title=dict(
            text=f"{display_title}<br>"
                 f"<sub>{G.number_of_nodes()} items · {G.number_of_edges()} links · {n_clusters} clusters</sub>",
            font=dict(size=16),
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=1400,
        height=1000,
        hovermode="closest",
        legend=dict(
            yanchor="top", y=0.99, xanchor="left", x=0.01,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(200,200,200,0.5)",
            borderwidth=1,
        ),
    )

    _save_plotly_figure(fig, output_dir, f"network_{name}")


# ===========================================================================
# 5. COUNTRY QUALITY CHART (3 subplots interactivos)
# ===========================================================================

def create_country_quality_chart(articles: list[dict], output_dir: Path) -> None:
    """Plotly 3-subplot: total citations, avg citations, h-index por pais."""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    # Agrupar articulos por pais
    country_data: dict[str, list[int]] = {}
    for a in articles:
        from src.modelo_corpus import article_countries as _paises
        article_countries = set(_paises(a))        # definicion unica (antes: solo paises de instituciones)
        tc = a.get("cited_by_count", 0)
        for c in article_countries:
            country_data.setdefault(c, []).append(tc)

    if not country_data:
        logger.warning("No country data for quality chart")
        return

    rows = []
    for country, cites_list in country_data.items():
        total_cites = sum(cites_list)
        n_articles = len(cites_list)
        avg_cites = total_cites / n_articles if n_articles else 0
        sorted_c = sorted(cites_list, reverse=True)
        h = 0
        for i, c in enumerate(sorted_c, 1):
            if c >= i:
                h = i
            else:
                break
        rows.append({"country": country, "total_cites": total_cites,
                     "avg_cites": round(avg_cites, 2), "h_index": h,
                     "n_articles": n_articles})

    import pandas as pd
    df = pd.DataFrame(rows)

    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=("Total Citations", "Average Citations", "H-index"),
                        shared_yaxes=False)

    # Panel 1: Total cites
    top_tc = df.nlargest(15, "total_cites")
    fig.add_trace(go.Bar(
        y=top_tc["country"], x=top_tc["total_cites"], orientation="h",
        marker_color="#4e79a7", name="Total Citations",
        hovertemplate="%{y}: %{x} citations<extra></extra>",
    ), row=1, col=1)

    # Panel 2: Avg cites
    top_avg = df.nlargest(15, "avg_cites")
    fig.add_trace(go.Bar(
        y=top_avg["country"], x=top_avg["avg_cites"], orientation="h",
        marker_color="#f28e2b", name="Avg Citations",
        hovertemplate="%{y}: %{x:.1f} avg<extra></extra>",
    ), row=1, col=2)

    # Panel 3: H-index
    top_h = df.nlargest(15, "h_index")
    fig.add_trace(go.Bar(
        y=top_h["country"], x=top_h["h_index"], orientation="h",
        marker_color="#e15759", name="H-index",
        hovertemplate="%{y}: h=%{x}<extra></extra>",
    ), row=1, col=3)

    fig.update_layout(
        title=dict(
            text="Country Quality Metrics<br>"
                 f"<sub>{len(country_data)} countries</sub>",
            font=dict(size=18),
        ),
        width=1600, height=700,
        paper_bgcolor="white",
        showlegend=False,
    )
    # Invert y-axes so highest is at top
    fig.update_yaxes(autorange="reversed")

    _save_plotly_figure(fig, output_dir, "country_quality_interactive")


# ===========================================================================
# RUNNER PRINCIPAL
# ===========================================================================

def run_advanced_visualizations(articles: list[dict], output_dir: Path,
                                networks: dict[str, nx.Graph] | None = None) -> None:
    """Ejecuta todas las visualizaciones avanzadas.

    Args:
        articles: Lista de articulos enriquecidos.
        output_dir: Directorio base de salida (se crea subdirectorio 'plotly').
        networks: Dict de grafos networkx {nombre: Graph} para redes interactivas.
    """
    plotly_dir = output_dir / "plotly"
    plotly_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Visualizaciones Avanzadas (Plotly) ===")

    # 1. Sankey tematico
    logger.info("Creating thematic Sankey diagram...")
    try:
        create_thematic_sankey(articles, plotly_dir)
    except Exception as e:
        logger.error(f"Sankey failed: {e}")

    # 2. Historiografo
    logger.info("Creating citation historiograph...")
    try:
        create_historiograph(articles, plotly_dir)
    except Exception as e:
        logger.error(f"Historiograph failed: {e}")

    # 3. Mapa mundial
    logger.info("Creating world collaboration map...")
    try:
        create_world_collaboration_map(articles, plotly_dir)
    except Exception as e:
        logger.error(f"World map failed: {e}")

    # 4. Country quality chart
    logger.info("Creating country quality chart...")
    try:
        create_country_quality_chart(articles, plotly_dir)
    except Exception as e:
        logger.error(f"Country quality chart failed: {e}")

    # 5. Redes interactivas
    if networks:
        for net_name, G in networks.items():
            logger.info(f"Creating interactive network: {net_name}...")
            try:
                create_interactive_network(G, plotly_dir, net_name,
                                           title=net_name.replace("_", " ").title())
            except Exception as e:
                logger.error(f"Interactive network {net_name} failed: {e}")

    logger.info(f"Advanced visualizations saved to: {plotly_dir}")
