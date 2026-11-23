"""Fase 3b: Analisis tematico estrategico (agnostico).

Incluye:
  - Mapa tematico / Diagrama estrategico (centralidad x densidad)
  - Evolucion tematica (Sankey basado en clusters temporales)
  - Analisis SCP vs MCP (Single/Multiple Country Publications)
"""
from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

from config import FIGURES_DIR, PLOTLY_DIR, REPORTS_DIR

logger = logging.getLogger("bibliometrico.thematic_strategy")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_keywords(article: dict, articles: Optional[list[dict]] = None) -> list[str]:
    """Palabras clave normalizadas con el vocabulario comun del pipeline (src.utils.article_keywords).

    Antes esta fase tenia su propio extractor (solo minusculas, autor O automaticas), distinto del
    de la Fase 3: 'palabra clave' no significaba lo mismo en el mapa tematico y en las redes."""
    from src.utils import article_keywords
    return article_keywords(article, articles)


def _extract_countries(article: dict) -> list[str]:
    """Paises de un articulo: definicion unica del modelo del corpus."""
    from src.modelo_corpus import article_countries
    return article_countries(article)


def _build_kw_cooccurrence(articles: list[dict],
                           min_cooccurrence: int = 2) -> nx.Graph:
    """Construye red de co-ocurrencia de keywords."""
    G = nx.Graph()
    kw_freq: Counter = Counter()
    cooc: Counter = Counter()

    for a in articles:
        kws = _extract_keywords(a, articles)
        for kw in kws:
            kw_freq[kw] += 1
        for i in range(len(kws)):
            for j in range(i + 1, len(kws)):
                pair = tuple(sorted([kws[i], kws[j]]))
                cooc[pair] += 1

    for (k1, k2), cnt in cooc.items():
        if cnt >= min_cooccurrence:
            if not G.has_node(k1):
                G.add_node(k1, frequency=kw_freq[k1])
            if not G.has_node(k2):
                G.add_node(k2, frequency=kw_freq[k2])
            G.add_edge(k1, k2, weight=cnt)

    return G


def callon_centrality_density(G: nx.Graph, members, freq: dict) -> tuple[float, float]:
    """Centralidad y densidad de un tema segun Callon, Courtial y Laville (1991), como Cobo et al. (2011)
    y bibliometrix::thematicMap.

    - Indice de equivalencia de cada enlace: e_ij = c_ij^2 / (c_i * c_j), con c_ij = co-ocurrencias y
      c_i = frecuencia de la palabra (normaliza por la frecuencia: un termino muy comun no domina).
    - Centralidad = 10 * suma de e_ij de los enlaces que salen del tema.
    - Densidad = 100 * suma de e_ij de los enlaces internos / numero de palabras del tema.

    Unica implementacion para la Fase 3b y los diagramas estrategicos de la Fase 6. La version
    anterior usaba conteos brutos, centralidad / n y densidad / pares posibles, lo que castigaba a
    los temas grandes (el tema de autoeficacia salia 'emergente o en declive').
    """
    s = set(members)
    ext = inn = 0.0
    for u in s:
        if u not in G:
            continue
        fu = max(float(freq.get(u, 0) or 0), 1.0)
        for v in G.neighbors(u):
            fv = max(float(freq.get(v, 0) or 0), 1.0)
            c = float(G[u][v].get("weight", 1))
            e = c * c / (fu * fv)
            if v in s:
                inn += e          # cada enlace interno se recorre dos veces
            else:
                ext += e
    inn /= 2.0
    n = len(s) or 1
    return 10.0 * ext, 100.0 * inn / n


def _detect_communities(G: nx.Graph) -> dict:
    """Louvain community detection. Returns {node: community_id}."""
    if G.number_of_nodes() == 0:
        return {}
    import community as community_louvain
    partition = community_louvain.best_partition(G, random_state=42)
    nx.set_node_attributes(G, partition, "community")
    return partition


def _country_code_to_name(code: str) -> str:
    """Convierte ISO-2 a nombre legible."""
    try:
        import pycountry
        c = pycountry.countries.get(alpha_2=code.upper())
        return c.name if c else code
    except ImportError:
        return code


# ===========================================================================
# 1. MAPA TEMATICO / DIAGRAMA ESTRATEGICO
# ===========================================================================

def compute_thematic_map(
    articles: list[dict],
    min_cooccurrence: int = 2,
    min_cluster_size: int = 3,
    kw_network: Optional[nx.Graph] = None,
) -> dict:
    """Calcula centralidad y densidad de clusters tematicos.

    Para cada cluster (comunidad Louvain en red de co-ocurrencia):
      - **Centralidad** y **Densidad** segun Callon et al. (1991): ver
        callon_centrality_density (indice de equivalencia; centralidad = 10 * suma externa;
        densidad = 100 * suma interna / numero de palabras).

    Args:
        kw_network: Optional pre-built keyword cooccurrence graph (from Phase 3).
                    If provided, uses a copy (Callon modifies attributes).

    Returns:
        dict con 'clusters', 'graph', 'partition' para visualizacion.
    """
    if kw_network is not None and kw_network.number_of_nodes() >= 5:
        G = kw_network.copy()
        logger.info(
            f"Reusing pre-built keyword network: {G.number_of_nodes()} nodes, "
            f"{G.number_of_edges()} edges"
        )
    else:
        G = _build_kw_cooccurrence(articles, min_cooccurrence=min_cooccurrence)
    if G.number_of_nodes() < 5:
        logger.warning("Too few keywords for thematic map")
        return {"clusters": [], "graph": G, "partition": {}}

    partition = _detect_communities(G)
    if not partition:
        return {"clusters": [], "graph": G, "partition": {}}

    # Agrupar keywords por cluster
    cluster_nodes: dict[int, list[str]] = defaultdict(list)
    for node, cid in partition.items():
        cluster_nodes[cid].append(node)

    clusters = []
    for cid, nodes in cluster_nodes.items():
        if len(nodes) < min_cluster_size:
            continue

        n = len(nodes)
        centrality, density = callon_centrality_density(
            G, nodes, {kw: G.nodes[kw].get("frequency", 0) for kw in G.nodes()}
        )

        # Nombre del cluster: top-3 keywords por frecuencia
        kw_freqs = [(kw, G.nodes[kw].get("frequency", 0)) for kw in nodes]
        kw_freqs.sort(key=lambda x: x[1], reverse=True)
        name = "; ".join(kw for kw, _ in kw_freqs[:3])

        # Frecuencia total del cluster
        total_freq = sum(G.nodes[kw].get("frequency", 0) for kw in nodes)

        clusters.append({
            "id": cid,
            "name": name,
            "keywords": nodes,
            "n_keywords": n,
            "centrality": round(centrality, 4),
            "density": round(density, 4),
            "total_frequency": total_freq,
            "top_keywords": kw_freqs[:5],
        })

    clusters.sort(key=lambda c: c["total_frequency"], reverse=True)
    logger.info(f"Thematic map: {len(clusters)} clusters from {G.number_of_nodes()} keywords")

    return {"clusters": clusters, "graph": G, "partition": partition}


def plot_thematic_map_static(
    thematic_data: dict,
    output_dir: Path = FIGURES_DIR,
) -> Optional[Path]:
    """Genera diagrama estrategico estatico (matplotlib, 300 DPI)."""
    clusters = thematic_data.get("clusters", [])
    if not clusters:
        logger.warning("No clusters for thematic map plot")
        return None

    centralities = [c["centrality"] for c in clusters]
    densities = [c["density"] for c in clusters]

    med_c = np.median(centralities)
    med_d = np.median(densities)

    fig, ax = plt.subplots(figsize=(12, 10))

    # Cuadrantes
    ax.axhline(med_d, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline(med_c, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    # Etiquetas de cuadrantes
    x_min = min(centralities) - 0.5
    x_max = max(centralities) + 0.5
    y_min = min(densities) - 0.5
    y_max = max(densities) + 0.5

    quad_labels = {
        "NICHE THEMES\n(well-developed,\nisolated)": (x_min * 0.3 + med_c * 0.7, y_max * 0.7 + med_d * 0.3),
        "MOTOR THEMES\n(well-developed,\ncentral)": (x_max * 0.7 + med_c * 0.3, y_max * 0.7 + med_d * 0.3),
        "EMERGING/\nDECLINING THEMES": (x_min * 0.3 + med_c * 0.7, y_min * 0.7 + med_d * 0.3),
        "BASIC THEMES\n(transversal,\ngeneral)": (x_max * 0.7 + med_c * 0.3, y_min * 0.7 + med_d * 0.3),
    }

    for label, (x, y) in quad_labels.items():
        ax.text(x, y, label, fontsize=8, ha="center", va="center",
                alpha=0.35, style="italic")

    # Burbujas
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(clusters), 1)))
    for i, cl in enumerate(clusters):
        size = max(100, cl["total_frequency"] * 30)
        ax.scatter(
            cl["centrality"], cl["density"],
            s=size, alpha=0.65,
            color=colors[i % len(colors)],
            edgecolors="black", linewidth=0.5,
            zorder=3,
        )
        # Nombre (top keywords)
        short_name = cl["name"][:40]
        ax.annotate(
            short_name,
            (cl["centrality"], cl["density"]),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=7,
            fontweight="bold",
            zorder=4,
        )

    ax.set_xlabel("Centrality (Callon)", fontsize=11)
    ax.set_ylabel("Density (Callon)", fontsize=11)
    ax.set_title("Strategic Diagram / Thematic Map", fontsize=14, fontweight="bold")

    # Leyenda de cuadrantes
    q_patches = [
        mpatches.Patch(color="#2ca02c", alpha=0.3, label="Motor (high C, high D)"),
        mpatches.Patch(color="#ff7f0e", alpha=0.3, label="Niche (low C, high D)"),
        mpatches.Patch(color="#1f77b4", alpha=0.3, label="Basic (high C, low D)"),
        mpatches.Patch(color="#d62728", alpha=0.3, label="Emerging/Declining (low C, low D)"),
    ]
    ax.legend(handles=q_patches, loc="upper left", fontsize=8, framealpha=0.8)

    plt.tight_layout()
    out_path = output_dir / "thematic_map_strategic_diagram.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info(f"Thematic map saved: {out_path}")
    return out_path


def plot_thematic_map_plotly(
    thematic_data: dict,
    output_dir: Path = PLOTLY_DIR,
) -> Optional[Path]:
    """Genera diagrama estrategico interactivo (Plotly)."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        logger.warning("plotly not installed, skipping interactive thematic map")
        return None

    clusters = thematic_data.get("clusters", [])
    if not clusters:
        return None

    centralities = [c["centrality"] for c in clusters]
    densities = [c["density"] for c in clusters]
    med_c = np.median(centralities)
    med_d = np.median(densities)

    # Clasificar cuadrantes
    quadrant_labels = []
    quadrant_colors = []
    for cl in clusters:
        if cl["centrality"] >= med_c and cl["density"] >= med_d:
            quadrant_labels.append("Motor")
            quadrant_colors.append("#2ca02c")
        elif cl["centrality"] < med_c and cl["density"] >= med_d:
            quadrant_labels.append("Niche")
            quadrant_colors.append("#ff7f0e")
        elif cl["centrality"] >= med_c and cl["density"] < med_d:
            quadrant_labels.append("Basic")
            quadrant_colors.append("#1f77b4")
        else:
            quadrant_labels.append("Emerging/Declining")
            quadrant_colors.append("#d62728")

    hover_texts = []
    for cl, ql in zip(clusters, quadrant_labels):
        top_kws = ", ".join(kw for kw, _ in cl["top_keywords"])
        hover_texts.append(
            f"<b>{cl['name']}</b><br>"
            f"Quadrant: {ql}<br>"
            f"Centrality: {cl['centrality']:.3f}<br>"
            f"Density: {cl['density']:.3f}<br>"
            f"Keywords ({cl['n_keywords']}): {top_kws}<br>"
            f"Total frequency: {cl['total_frequency']}"
        )

    sizes = [max(20, cl["total_frequency"] * 4) for cl in clusters]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=centralities,
        y=densities,
        mode="markers+text",
        marker=dict(
            size=sizes,
            color=quadrant_colors,
            opacity=0.7,
            line=dict(width=1, color="black"),
        ),
        text=[cl["name"][:30] for cl in clusters],
        textposition="top center",
        textfont=dict(size=9),
        hovertext=hover_texts,
        hoverinfo="text",
    ))

    # Lineas de mediana
    fig.add_hline(y=med_d, line_dash="dash", line_color="gray", opacity=0.5,
                  annotation_text=f"Density median = {med_d:.3f}",
                  annotation_position="top right")
    fig.add_vline(x=med_c, line_dash="dash", line_color="gray", opacity=0.5,
                  annotation_text=f"Centrality median = {med_c:.3f}",
                  annotation_position="top right")

    # Annotations de cuadrantes
    pad_x = (max(centralities) - min(centralities)) * 0.15 or 0.5
    pad_y = (max(densities) - min(densities)) * 0.15 or 0.5
    quad_annotations = [
        dict(x=med_c - pad_x, y=med_d + pad_y, text="<b>NICHE</b>",
             showarrow=False, font=dict(size=12, color="rgba(255,127,14,0.4)")),
        dict(x=med_c + pad_x, y=med_d + pad_y, text="<b>MOTOR</b>",
             showarrow=False, font=dict(size=12, color="rgba(44,160,44,0.4)")),
        dict(x=med_c - pad_x, y=med_d - pad_y, text="<b>EMERGING/<br>DECLINING</b>",
             showarrow=False, font=dict(size=12, color="rgba(214,39,40,0.4)")),
        dict(x=med_c + pad_x, y=med_d - pad_y, text="<b>BASIC</b>",
             showarrow=False, font=dict(size=12, color="rgba(31,119,180,0.4)")),
    ]

    fig.update_layout(
        title="Strategic Diagram / Thematic Map",
        xaxis_title="Centrality (Callon)",
        yaxis_title="Density (Callon)",
        showlegend=False,
        annotations=quad_annotations,
        template="plotly_white",
        width=900, height=700,
    )

    out_path = output_dir / "thematic_map_strategic_diagram.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    logger.info(f"Interactive thematic map saved: {out_path}")

    return out_path


# ===========================================================================
# 2. EVOLUCION TEMATICA → Script R para bibliometrix (mejor calidad)
# ===========================================================================

def generate_thematic_evolution_r_script(
    articles: list[dict],
    output_dir: Path = REPORTS_DIR,
) -> Optional[Path]:
    """Genera script R dedicado para thematic evolution con bibliometrix.

    Produce un .R listo para ejecutar que genera:
      - Thematic evolution (alluvial Sankey)
      - Thematic map con periodos separados
      - Datos exportados para post-procesamiento del usuario

    Requiere: bibliometrix_export.csv generado por Phase 4.
    """
    years = [a.get("publication_year", 0) for a in articles if a.get("publication_year")]
    if not years:
        logger.warning("No years for thematic evolution R script")
        return None

    min_y, max_y = min(years), max(years)
    span = max_y - min_y + 1

    # Auto-calculate year cuts
    if span >= 9:
        third = span // 3
        cuts = [min_y, min_y + third, min_y + 2 * third, max_y]
    elif span >= 5:
        mid = min_y + span // 2
        cuts = [min_y, mid, max_y]
    else:
        cuts = [min_y, max_y]

    cuts_r = f"c({', '.join(str(c) for c in cuts)})"

    script = f'''#!/usr/bin/env Rscript
# ============================================================================
# Thematic Evolution Analysis — bibliometrix
# Generated by Fase 3b (strategic_thematic_analysis.py)
# ============================================================================
#
# USAGE:
#   Rscript thematic_evolution.R
#
# PREREQUISITES:
#   1. Run `python main.py --phases 5` to generate bibliometrix_export.csv
#   2. Install R packages: install.packages(c("bibliometrix", "ggplot2"))
#
# The user can modify parameters below before running.
# ============================================================================

# --- Configuration (adjust as needed) ---
CSV_FILE    <- "bibliometrix_export.csv"   # Relative to script dir
FIELD       <- "DE"                        # "DE" (author kw) or "ID" (kw plus)
MIN_FREQ    <- 2                           # Min keyword frequency per period
N_WORDS     <- 200                         # Max keywords per period
YEAR_CUTS   <- {cuts_r}  # Period boundaries (auto: {min_y}-{max_y})
OUTPUT_DIR  <- "."                         # Output directory

# --- Load libraries ---
if (!requireNamespace("bibliometrix", quietly = TRUE)) {{
  install.packages("bibliometrix", repos = "https://cloud.r-project.org/")
}}
library(bibliometrix)

# --- Locate data ---
script_dir <- tryCatch(dirname(sys.frame(1)$ofile), error = function(e) getwd())
csv_path <- file.path(script_dir, CSV_FILE)
if (!file.exists(csv_path)) {{
  alt <- file.path(script_dir, "..", "output", CSV_FILE)
  if (file.exists(alt)) csv_path <- alt
  else {{
    alt2 <- file.path(script_dir, "..", CSV_FILE)
    if (file.exists(alt2)) csv_path <- alt2
    else stop("CSV not found. Run: python main.py --phases 5")
  }}
}}

cat("Loading:", csv_path, "\\n")
# Read CSV directly (convert2df fails with our pre-formatted export)
M <- read.csv(csv_path, stringsAsFactors = FALSE, na.strings = c("", "NA", "nan"))
cat("Articles loaded:", nrow(M), "\\n")

# Ensure PY is numeric
M$PY <- as.numeric(M$PY)

# Build SR field required by bibliometrix (first author + year + source)
first_au <- sub(";.*", "", M$AU)
M$SR <- paste(first_au, M$PY, M$SO, sep = ", ")
M$SR <- make.unique(M$SR)
M$SR_FULL <- M$SR

# Set class so bibliometrix functions recognize the data frame
class(M) <- c("bibliometrixDB", "data.frame")

# --- 1. Thematic Evolution (Alluvial/Sankey) ---
cat("\\n=== THEMATIC EVOLUTION ===\\n")
cat("Field:", FIELD, "\\n")
cat("Year cuts:", paste(YEAR_CUTS, collapse = " -> "), "\\n")

M_te <- M[!is.na(M[[FIELD]]) & M[[FIELD]] != "", ]
cat("Articles with keywords:", nrow(M_te), "\\n")

if (nrow(M_te) >= 10 && length(YEAR_CUTS) >= 3) {{
  tryCatch({{
    nexus <- thematicEvolution(M_te, field = FIELD, years = YEAR_CUTS,
                                n = N_WORDS, minFreq = MIN_FREQ)

    # Save plot
    pdf(file.path(OUTPUT_DIR, "thematic_evolution.pdf"), width = 18, height = 10)
    plotThematicEvolution(nexus$Nodes, nexus$Edges)
    dev.off()
    cat("Saved: thematic_evolution.pdf\\n")

    # Export data for post-processing
    write.csv(nexus$Nodes, file.path(OUTPUT_DIR, "thematic_evolution_nodes.csv"),
              row.names = FALSE)
    write.csv(nexus$Edges, file.path(OUTPUT_DIR, "thematic_evolution_edges.csv"),
              row.names = FALSE)
    cat("Saved: thematic_evolution_nodes.csv, thematic_evolution_edges.csv\\n")
  }}, error = function(e) cat("Error:", e$message, "\\n"))
}} else {{
  cat("Not enough data for thematic evolution.\\n")
}}

# --- 2. Thematic Map per Period ---
cat("\\n=== THEMATIC MAPS PER PERIOD ===\\n")
for (i in 1:(length(YEAR_CUTS) - 1)) {{
  yr_start <- YEAR_CUTS[i]
  yr_end   <- YEAR_CUTS[i + 1]
  period_label <- paste0(yr_start, "-", yr_end)
  cat("Period:", period_label, "\\n")

  M_period <- M_te[M_te$PY >= yr_start & M_te$PY <= yr_end, ]
  if (nrow(M_period) >= 5) {{
    tryCatch({{
      Map <- thematicMap(M_period, field = FIELD, n = N_WORDS,
                          minfreq = max(1, MIN_FREQ - 1),
                          stemming = FALSE, size = 0.5, repel = TRUE)

      pdf(file.path(OUTPUT_DIR, paste0("thematic_map_", period_label, ".pdf")),
          width = 14, height = 10)
      print(Map$map)
      dev.off()
      cat("  Saved: thematic_map_", period_label, ".pdf\\n")

      # Export cluster data
      if (!is.null(Map$clusters)) {{
        write.csv(Map$clusters,
                  file.path(OUTPUT_DIR, paste0("thematic_clusters_", period_label, ".csv")),
                  row.names = FALSE)
      }}
    }}, error = function(e) cat("  Error:", e$message, "\\n"))
  }} else {{
    cat("  Not enough articles in period.\\n")
  }}
}}

# --- 3. Global Thematic Map ---
cat("\\n=== GLOBAL THEMATIC MAP ===\\n")
tryCatch({{
  Map_global <- thematicMap(M_te, field = FIELD, n = N_WORDS,
                             minfreq = MIN_FREQ,
                             stemming = FALSE, size = 0.5, repel = TRUE)
  pdf(file.path(OUTPUT_DIR, "thematic_map_global.pdf"), width = 14, height = 10)
  print(Map_global$map)
  dev.off()
  cat("Saved: thematic_map_global.pdf\\n")

  if (!is.null(Map_global$clusters)) {{
    write.csv(Map_global$clusters,
              file.path(OUTPUT_DIR, "thematic_clusters_global.csv"),
              row.names = FALSE)
    cat("Saved: thematic_clusters_global.csv\\n")
  }}
}}, error = function(e) cat("Error:", e$message, "\\n"))

cat("\\n=== DONE ===\\n")
cat("Post-processing: edit YEAR_CUTS, FIELD, MIN_FREQ as needed and re-run.\\n")
'''

    out_path = output_dir / "thematic_evolution.R"
    out_path.write_text(script, encoding="utf-8")
    logger.info(f"Thematic evolution R script saved: {out_path}")
    logger.info(f"  Year cuts: {cuts}")
    logger.info(f"  Run: Rscript {out_path}")
    return out_path


# ===========================================================================
# 3. SCP vs MCP (Single / Multiple Country Publications)
# ===========================================================================

def compute_scp_mcp(articles: list[dict]) -> dict:
    """Clasifica articulos como SCP o MCP y agrega por pais.

    - SCP: todos los autores del mismo pais
    - MCP: autores de 2+ paises

    Returns:
        dict con 'articles_classified', 'by_country', 'summary'.
    """
    classified = []
    country_stats: dict[str, dict] = defaultdict(
        lambda: {"total": 0, "scp": 0, "mcp": 0, "citations": 0}
    )

    n_no_country = 0
    for a in articles:
        countries = _extract_countries(a)
        if not countries:
            # Fallback: buscar en affiliations string
            affs = a.get("affiliations", [])
            # No country data: skip
            n_no_country += 1
            classified.append({
                "key": a.get("key", ""),
                "type": "unknown",
                "countries": [],
            })
            continue

        pub_type = "SCP" if len(countries) == 1 else "MCP"
        cites = a.get("cited_by_count", 0)

        classified.append({
            "key": a.get("key", ""),
            "type": pub_type,
            "countries": countries,
        })

        # Atribuir a cada pais participante
        for c in countries:
            country_stats[c]["total"] += 1
            country_stats[c][pub_type.lower()] += 1
            country_stats[c]["citations"] += cites

    # Calcular ratios
    by_country = []
    for code, stats in country_stats.items():
        name = _country_code_to_name(code)
        total = stats["total"]
        by_country.append({
            "country_code": code,
            "country_name": name,
            "total": total,
            "scp": stats["scp"],
            "mcp": stats["mcp"],
            "mcp_ratio": round(stats["mcp"] / total, 3) if total else 0,
            "citations": stats["citations"],
            "avg_citations": round(stats["citations"] / total, 2) if total else 0,
        })

    by_country.sort(key=lambda x: x["total"], reverse=True)

    # Summary
    total_classified = sum(1 for c in classified if c["type"] != "unknown")
    total_scp = sum(1 for c in classified if c["type"] == "SCP")
    total_mcp = sum(1 for c in classified if c["type"] == "MCP")

    summary = {
        "total_articles": len(articles),
        "classified": total_classified,
        "unknown_country": n_no_country,
        "scp": total_scp,
        "mcp": total_mcp,
        "mcp_ratio": round(total_mcp / total_classified, 3) if total_classified else 0,
        "n_countries": len(by_country),
    }

    logger.info(
        f"SCP/MCP: {total_scp} SCP, {total_mcp} MCP "
        f"({summary['mcp_ratio']:.1%} international), "
        f"{n_no_country} unknown"
    )

    return {
        "articles_classified": classified,
        "by_country": by_country,
        "summary": summary,
    }


def plot_scp_mcp_plotly(
    scp_data: dict,
    top_n: int = 20,
    output_dir: Path = PLOTLY_DIR,
) -> Optional[Path]:
    """Genera grafico SCP vs MCP por pais (Plotly stacked bar)."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        logger.warning("plotly not installed, skipping SCP/MCP plot")
        return None

    by_country = scp_data.get("by_country", [])[:top_n]
    if not by_country:
        return None

    countries = [c["country_name"] for c in by_country]
    scp_vals = [c["scp"] for c in by_country]
    mcp_vals = [c["mcp"] for c in by_country]
    mcp_ratios = [c["mcp_ratio"] for c in by_country]

    # Reverse for horizontal bar
    countries.reverse()
    scp_vals.reverse()
    mcp_vals.reverse()
    mcp_ratios.reverse()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=countries, x=scp_vals, name="SCP",
        orientation="h",
        marker_color="#4e79a7",
        hovertemplate="<b>%{y}</b><br>SCP: %{x}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=countries, x=mcp_vals, name="MCP",
        orientation="h",
        marker_color="#e15759",
        hovertemplate="<b>%{y}</b><br>MCP: %{x}<extra></extra>",
    ))

    # Annotations: MCP ratio
    for i, (c, ratio) in enumerate(zip(countries, mcp_ratios)):
        total = scp_vals[i] + mcp_vals[i]
        if total > 0:
            fig.add_annotation(
                x=total + 0.3, y=c,
                text=f"{ratio:.0%}",
                showarrow=False,
                font=dict(size=9, color="#e15759"),
                xanchor="left",
            )

    summary = scp_data.get("summary", {})
    fig.update_layout(
        title=f"Corresponding Author's Country — SCP vs MCP "
              f"(MCP ratio: {summary.get('mcp_ratio', 0):.1%})",
        xaxis_title="Number of Documents",
        yaxis_title="",
        barmode="stack",
        template="plotly_white",
        width=900, height=max(400, top_n * 28),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=150),
    )

    out_path = output_dir / "scp_mcp_analysis.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    logger.info(f"SCP/MCP plot saved: {out_path}")

    return out_path


def plot_scp_mcp_static(
    scp_data: dict,
    top_n: int = 20,
    output_dir: Path = FIGURES_DIR,
) -> Optional[Path]:
    """Genera grafico SCP vs MCP estatico (matplotlib)."""
    by_country = scp_data.get("by_country", [])[:top_n]
    if not by_country:
        return None

    countries = [c["country_name"] for c in by_country]
    scp_vals = [c["scp"] for c in by_country]
    mcp_vals = [c["mcp"] for c in by_country]

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.35)))

    y_pos = range(len(countries))
    ax.barh(y_pos, scp_vals, color="#4e79a7", label="SCP", edgecolor="white")
    ax.barh(y_pos, mcp_vals, left=scp_vals, color="#e15759", label="MCP", edgecolor="white")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(countries, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Documents")
    ax.set_title("Corresponding Author's Country — SCP vs MCP", fontweight="bold")
    ax.legend(loc="lower right")

    # MCP ratio annotations
    for i, c in enumerate(by_country):
        total = c["scp"] + c["mcp"]
        if total > 0 and c["mcp_ratio"] > 0:
            ax.text(total + 0.2, i, f"{c['mcp_ratio']:.0%}",
                    va="center", fontsize=7, color="#e15759")

    plt.tight_layout()
    out_path = output_dir / "scp_mcp_analysis.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info(f"SCP/MCP static plot saved: {out_path}")
    return out_path


# ===========================================================================
# REPORTE MARKDOWN
# ===========================================================================

def generate_report(
    thematic_data: dict,
    r_script_path: Optional[str],
    scp_data: dict,
    output_dir: Path = REPORTS_DIR,
) -> Path:
    """Genera reporte Markdown consolidado."""
    lines = [
        "# Strategic Thematic Analysis Report",
        "",
        "## 1. Thematic Map (Strategic Diagram)",
        "",
    ]

    clusters = thematic_data.get("clusters", [])
    if clusters:
        centralities = [c["centrality"] for c in clusters]
        densities = [c["density"] for c in clusters]
        med_c = np.median(centralities)
        med_d = np.median(densities)

        lines.append(f"- **Clusters detected:** {len(clusters)}")
        lines.append(f"- **Centrality median:** {med_c:.3f}")
        lines.append(f"- **Density median:** {med_d:.3f}")
        lines.append("")
        lines.append("| Cluster | Keywords | Centrality | Density | Quadrant | Freq |")
        lines.append("|---------|----------|------------|---------|----------|------|")

        for cl in clusters:
            if cl["centrality"] >= med_c and cl["density"] >= med_d:
                quad = "Motor"
            elif cl["centrality"] < med_c and cl["density"] >= med_d:
                quad = "Niche"
            elif cl["centrality"] >= med_c and cl["density"] < med_d:
                quad = "Basic"
            else:
                quad = "Emerging/Declining"
            lines.append(
                f"| {cl['name'][:35]} | {cl['n_keywords']} | "
                f"{cl['centrality']:.3f} | {cl['density']:.3f} | "
                f"{quad} | {cl['total_frequency']} |"
            )
        lines.append("")
    else:
        lines.append("*No clusters detected.*\n")

    # Thematic Evolution
    lines.append("## 2. Thematic Evolution")
    lines.append("")

    if r_script_path:
        lines.append(
            f"R script generated for post-processing with bibliometrix: "
            f"`{r_script_path}`"
        )
        lines.append("")
        lines.append("Run with: `Rscript thematic_evolution.R`")
        lines.append("")
        lines.append("Configurable parameters in the script:")
        lines.append("- `YEAR_CUTS`: Period boundaries")
        lines.append("- `FIELD`: `DE` (author keywords) or `ID` (keywords plus)")
        lines.append("- `MIN_FREQ`: Minimum keyword frequency per period")
        lines.append("")
    else:
        lines.append("*R script not generated (no year data).*\n")

    # SCP/MCP
    lines.append("## 3. SCP vs MCP Analysis")
    lines.append("")

    summary = scp_data.get("summary", {})
    if summary:
        lines.append(f"- **Total articles:** {summary['total_articles']}")
        lines.append(f"- **Classified:** {summary['classified']}")
        lines.append(f"- **SCP (single country):** {summary['scp']}")
        lines.append(f"- **MCP (multi country):** {summary['mcp']}")
        lines.append(f"- **MCP ratio:** {summary['mcp_ratio']:.1%}")
        lines.append(f"- **Countries represented:** {summary['n_countries']}")
        lines.append(f"- **Unknown country:** {summary['unknown_country']}")
        lines.append("")

        by_country = scp_data.get("by_country", [])[:15]
        if by_country:
            lines.append("| Country | Total | SCP | MCP | MCP% | Cites | Avg Cites |")
            lines.append("|---------|-------|-----|-----|------|-------|-----------|")
            for c in by_country:
                lines.append(
                    f"| {c['country_name']} | {c['total']} | {c['scp']} | "
                    f"{c['mcp']} | {c['mcp_ratio']:.0%} | "
                    f"{c['citations']} | {c['avg_citations']:.1f} |"
                )
            lines.append("")
    else:
        lines.append("*No country data available.*\n")

    out_path = output_dir / "strategic_thematic_report.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Strategic thematic report saved: {out_path}")
    return out_path


# ===========================================================================
# ORQUESTADOR PRINCIPAL
# ===========================================================================

def run_phase3b(articles: list[dict], networks: Optional[dict] = None) -> dict:
    """Ejecuta la Fase 3b: Analisis tematico estrategico completo.

    Args:
        articles: List of enriched article dicts.
        networks: Optional dict of pre-built networkx graphs from Phase 3.
                  If it contains 'keyword_cooccurrence', reuses it for thematic map.

    Returns:
        dict con resultados de las tres sub-analisis.
    """
    logger.info("=" * 60)
    logger.info("=== FASE 3b: Analisis Tematico Estrategico ===")
    logger.info("=" * 60)

    results = {}

    # Extract pre-built keyword network if available
    kw_net = (networks or {}).get("keyword_cooccurrence")

    # 1. Mapa tematico
    logger.info("--- 3b.1: Thematic Map (Strategic Diagram) ---")
    thematic_data = compute_thematic_map(articles, kw_network=kw_net)
    results["thematic_map"] = thematic_data

    if thematic_data["clusters"]:
        plot_thematic_map_static(thematic_data)
        plot_thematic_map_plotly(thematic_data)

    # 2. Evolucion tematica -> Script R para bibliometrix
    logger.info("--- 3b.2: Thematic Evolution (R script) ---")
    r_script_path = generate_thematic_evolution_r_script(articles)
    results["thematic_evolution_r_script"] = str(r_script_path) if r_script_path else None

    # 3. SCP / MCP
    logger.info("--- 3b.3: SCP vs MCP Analysis ---")
    scp_data = compute_scp_mcp(articles)
    results["scp_mcp"] = scp_data

    if scp_data["by_country"]:
        plot_scp_mcp_plotly(scp_data)
        plot_scp_mcp_static(scp_data)

    # Reporte
    generate_report(thematic_data, results.get("thematic_evolution_r_script"), scp_data)

    logger.info("=" * 60)
    logger.info("Fase 3b completada.")
    logger.info("=" * 60)

    return results
