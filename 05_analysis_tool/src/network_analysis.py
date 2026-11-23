"""Fase 3: Analisis de redes - acoplamiento, co-citacion, co-palabras."""
from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

from tqdm import tqdm
import time

from config import (
    NETWORKS_DIR, FIGURES_DIR, OPENALEX_CACHE_DIR,
    OPENALEX_EMAIL, OPENALEX_API_KEY, OPENALEX_RATE_LIMIT,
)
from src.utils import (
    RateLimiter,
    DiskCache,
    create_session,
    safe_request,
    resolve_openalex_label_cached,
    build_corpus_keyword_normalizer,
    extract_normalized_article_keywords,
    normalize_author_display_name,
    normalize_institution_name,
    build_corpus_institution_normalizer,
    format_journal_display_name,
    _journal_match_key,
    disambiguate_citation_labels,
    load_knowledge_base,
)

logger = logging.getLogger("bibliometrico.networks")


import re as _re

# Pre-compiled regex: match W followed by digits anywhere in the string
_WORK_ID_RE = _re.compile(r'(W\d{5,})')


def _extract_work_id(ref_url: str) -> str:
    """Extract a clean OpenAlex work ID (W-prefixed) from a reference URL.

    Handles multiple formats found in referenced_works:
    - "https://openalex.org/W2345678" → "W2345678"
    - "doi_fragment+openalex:W2345678" → "W2345678"  (malformed from some APIs)
    - "doi_fragment openalex:W2345678" → "W2345678"
    - "omid:br/... DOI" → "" (no W-ID, unresolvable)
    - "W2345678" → "W2345678"
    """
    # Fast path: standard OpenAlex URL
    if ref_url.startswith("https://openalex.org/W"):
        wid = ref_url.rsplit("/", 1)[-1]
        if wid.startswith("W") and wid[1:].isdigit():
            return wid

    # Try regex extraction for any format
    m = _WORK_ID_RE.search(ref_url)
    if m:
        return m.group(1)

    # Fallback: old behavior (split on /)
    wid = ref_url.split("/")[-1] if "/" in ref_url else ref_url
    if wid.startswith("W") and len(wid) > 1 and wid[1:].isdigit():
        return wid

    return ""


def _extract_label(data: dict) -> str:
    """Extract 'LastName (Year)' label from an OpenAlex work dict."""
    first_author = "Unknown"
    authorships = data.get("authorships", [])
    if authorships:
        first_author = authorships[0].get("author", {}).get("display_name", "Unknown")
        if not first_author or first_author == "Unknown":
            first_author = authorships[0].get("name", "Unknown")
        parts = first_author.split()
        if parts:
            first_author = parts[-1]
    year = data.get("publication_year") or "?"
    return f"{first_author} ({year})"


def _build_labels_from_resolved_refs(articles: list[dict]) -> dict[str, str]:
    """Extract 'FirstAuthor (Year)' labels from pre-resolved references (Phase 1i).

    Avoids API calls by using data already stored in article['references_resolved'].
    Returns {work_id: 'LastName (Year)'} for all resolvable references.
    """
    labels: dict[str, str] = {}
    for a in articles:
        for ref in a.get("references_resolved") or []:
            if not isinstance(ref, dict):
                continue
            oa_id = ref.get("openalex_id", "")
            wid = _extract_work_id(oa_id)
            if not wid or wid in labels:
                continue
            first_author = ref.get("first_author", "")
            year = ref.get("year", "")
            if first_author and year:
                # Extract last name
                parts = first_author.split()
                last_name = parts[-1] if parts else first_author
                labels[wid] = f"{last_name} ({year})"
    return labels


def resolve_openalex_ids(
    openalex_urls: list[str],
    batch_size: int = 25,
    base_backoff: float = 1.0,
    rate_limit: float | None = None,
    articles: list[dict] | None = None,
) -> dict[str, str]:
    """Resuelve OpenAlex work URLs a labels legibles 'PrimerAutor (Año)'.
    Usa cache en disco y batch API (hasta 50 IDs por request) para velocidad.
    Soporta API key para el premium pool de OpenAlex.
    """
    cache = DiskCache(OPENALEX_CACHE_DIR)
    labels = {}

    urls_to_fetch = []
    skipped_malformed = 0
    for url in openalex_urls:
        work_id = _extract_work_id(url)
        if not work_id:
            skipped_malformed += 1
            continue
        cached = cache.get(f"work:{work_id}")
        if cached and isinstance(cached, dict):
            labels[work_id] = _extract_label(cached)
        else:
            urls_to_fetch.append(work_id)
    if skipped_malformed:
        logger.info(f"Skipped {skipped_malformed} references without valid OpenAlex W-ID")

    if not urls_to_fetch:
        return labels

    logger.info(f"Resolving {len(urls_to_fetch)} OpenAlex IDs (cached: {len(labels)})...")

    session = create_session()
    # Build base params: prefer API key, fallback to mailto
    base_params = {}
    if OPENALEX_API_KEY:
        base_params["api_key"] = OPENALEX_API_KEY
        logger.info("Using OpenAlex API key (premium pool)")
    elif OPENALEX_EMAIL:
        base_params["mailto"] = OPENALEX_EMAIL

    # Batch API: /works?filter=openalex_id:W123|W456|...
    # Usar lotes conservadores para reducir 503
    BATCH_SIZE = max(1, min(50, batch_size))
    # With API key: up to 100 req/s; without: 10 req/s
    rate = min(OPENALEX_RATE_LIMIT * (5 if OPENALEX_API_KEY else 1), 50)
    if rate_limit:
        rate = min(max(rate_limit, 0.1), 50)
    rate_limiter = RateLimiter(rate)

    batches = [urls_to_fetch[i:i + BATCH_SIZE]
               for i in range(0, len(urls_to_fetch), BATCH_SIZE)]

    resolved_count = 0
    failed_batches = 0
    for batch in tqdm(batches, desc="Resolving references (batch)"):
        rate_limiter.wait()
        # OpenAlex filter requires full URLs: https://openalex.org/W123
        full_urls = [f"https://openalex.org/{wid}" for wid in batch]
        filter_str = "|".join(full_urls)
        params = {
            **base_params,
            "filter": f"openalex_id:{filter_str}",
            "select": "id,doi,authorships,publication_year,title,cited_by_count,concepts,primary_location,biblio",
            "per_page": str(BATCH_SIZE),
        }
        try:
            resp = session.get(
                "https://api.openalex.org/works",
                params=params,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            for work in results:
                oa_id = work.get("id", "")
                wid = oa_id.split("/")[-1] if "/" in oa_id else oa_id
                cache.set(f"work:{wid}", work)
                labels[wid] = _extract_label(work)
                resolved_count += 1
            # IDs not returned are non-existent; mark them directly
            returned_ids = set()
            for work in results:
                oa_id = work.get("id", "")
                returned_ids.add(oa_id.split("/")[-1] if "/" in oa_id else oa_id)
            for wid in batch:
                if wid not in returned_ids and wid not in labels:
                    full_url = f"https://openalex.org/{wid}" if wid.startswith("W") else wid
                    labels[wid] = resolve_openalex_label_cached(full_url, articles)
        except Exception as e:
            failed_batches += 1
            if failed_batches <= 3:
                logger.warning(f"Batch request failed ({e}), marking {len(batch)} IDs as unresolved")
            # Backoff incremental conservador
            time.sleep(base_backoff * min(failed_batches, 5))
            # Mark all IDs in failed batch as unresolved (don't fall back to individual)
            for wid in batch:
                if wid not in labels:
                    full_url = f"https://openalex.org/{wid}" if wid.startswith("W") else wid
                    labels[wid] = resolve_openalex_label_cached(full_url, articles)

    # Mark unresolved IDs
    for wid in urls_to_fetch:
        if wid not in labels:
            full_url = f"https://openalex.org/{wid}" if wid.startswith("W") else wid
            labels[wid] = resolve_openalex_label_cached(full_url, articles)

    logger.info(f"Resolved {resolved_count}/{len(urls_to_fetch)} references")
    return labels


def build_bibliographic_coupling_network(articles: list[dict]) -> nx.Graph:
    """Red de acoplamiento bibliográfico con índice invertido.
    Dos artículos del corpus están conectados si comparten referencias.
    Construye ref -> {artículos} y acumula pesos por pares de artículos citantes.
    """
    from collections import Counter as _Counter

    G = nx.Graph()

    # Nodos con label legible: "PrimerAutor (Año)" desambiguado (a, b, c) y atributos básicos
    article_refs: dict[str, set] = {}
    raw_labels: dict[str, str] = {}
    node_attrs: dict[str, dict] = {}
    for a in articles:
        refs = set(a.get("referenced_works", []) or [])
        if not refs:
            continue
        first_author = a["authors"][0].split(",")[0] if a.get("authors") else "Unknown"
        year = a.get("publication_year") or "?"
        raw_labels[a["key"]] = f"{first_author} ({year})"
        node_attrs[a["key"]] = {
            "title": (a.get("title") or "")[:80],
            "year": a.get("publication_year") or 0,
            "citations": a.get("cited_by_count") or 0,
            "num_refs": len(refs),
        }
        article_refs[a["key"]] = refs

    clean_labels = disambiguate_citation_labels(raw_labels)
    for key, attrs in node_attrs.items():
        G.add_node(key, label=clean_labels.get(key, raw_labels[key]), **attrs)

    if G.number_of_nodes() == 0:
        logger.info("Bibliographic coupling: 0 nodes, 0 edges")
        return G

    # Índice invertido: ref -> artículos que la citan
    ref_to_articles: dict[str, set] = {}
    for key, refs in article_refs.items():
        for r in refs:
            ref_to_articles.setdefault(r, set()).add(key)

    # Contador de pares de artículos co-referenciando
    edge_counter = _Counter()
    for citing_set in ref_to_articles.values():
        if len(citing_set) < 2:
            continue
        ks = sorted(citing_set)
        for i in range(len(ks)):
            ki = ks[i]
            for j in range(i + 1, len(ks)):
                kj = ks[j]
                edge_counter[(ki, kj)] += 1

    # Construir aristas con pesos
    for (u, v), w in edge_counter.items():
        G.add_edge(u, v, weight=int(w))

    logger.info(
        f"Bibliographic coupling: {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges"
    )
    return G


def build_cocitation_network(
    articles: list[dict],
    min_cocitations: int = 2,
    resolve_labels: bool = True,
    resolve_rate_limit: float | None = None,
    resolve_batch_size: int = 25,
    resolve_backoff: float = 1.0,
) -> nx.Graph:
    """Red de co-citacion.
    Dos trabajos externos estan conectados si son co-citados por articulos del corpus.
    Uses inverted index approach: iterate articles once, count co-citation pairs.
    """
    from collections import Counter as _Counter

    # Build ref -> citing articles map AND count co-citations via inverted index
    ref_to_articles: dict[str, set] = {}
    for a in articles:
        for ref in a.get("referenced_works", []):
            ref_to_articles.setdefault(ref, set()).add(a["key"])

    # Filter to refs cited by >=2 articles
    refs_list = [r for r, citing in ref_to_articles.items() if len(citing) >= 2]
    logger.info(f"References cited by >=2 corpus articles: {len(refs_list)}")

    # Resolve labels (optional)
    if resolve_labels:
        # First: try labels from references_resolved (Phase 1i) — zero API calls
        ref_labels = _build_labels_from_resolved_refs(articles)
        pre_resolved = sum(1 for r in refs_list
                           if _extract_work_id(r) in ref_labels)
        remaining = [r for r in refs_list
                     if _extract_work_id(r) not in ref_labels]
        if pre_resolved:
            logger.info(
                f"Labels resolved from references_resolved: {pre_resolved}/{len(refs_list)} "
                f"({len(remaining)} remaining for API)"
            )
        # Then: resolve remaining via API (if any)
        if remaining:
            api_labels = resolve_openalex_ids(
                remaining,
                batch_size=resolve_batch_size,
                base_backoff=resolve_backoff,
                rate_limit=resolve_rate_limit,
                articles=articles,
            )
            ref_labels.update(api_labels)
    else:
        ref_labels = {}

    # Build co-citation counts via inverted index (iterate articles, not O(n^2) refs)
    refs_set = set(refs_list)
    cocite_counter = _Counter()
    for a in articles:
        # Get this article's refs that are in our filtered set
        art_refs = sorted(r for r in a.get("referenced_works", []) if r in refs_set)
        # All pairs of refs in this article are co-cited
        for i in range(len(art_refs)):
            for j in range(i + 1, len(art_refs)):
                cocite_counter[(art_refs[i], art_refs[j])] += 1

    # Build graph from co-citation pairs
    G = nx.Graph()
    for (ref_i, ref_j), count in cocite_counter.items():
        if count >= min_cocitations:
            id_i = _extract_work_id(ref_i) or ref_i
            id_j = _extract_work_id(ref_j) or ref_j

            if not G.has_node(id_i):
                label_i = ref_labels.get(id_i, id_i)
                clean_url_i = f"https://openalex.org/{id_i}" if id_i.startswith("W") else ref_i
                G.add_node(id_i, label=label_i, openalex_url=clean_url_i,
                           cocitation_count=len(ref_to_articles[ref_i]))
            if not G.has_node(id_j):
                label_j = ref_labels.get(id_j, id_j)
                clean_url_j = f"https://openalex.org/{id_j}" if id_j.startswith("W") else ref_j
                G.add_node(id_j, label=label_j, openalex_url=clean_url_j,
                           cocitation_count=len(ref_to_articles[ref_j]))

            G.add_edge(id_i, id_j, weight=count)

    # Desambiguar homonimos 'Autor (Año)' -> 'Autor (Añoa)', 'Autor (Añob)'
    if G.number_of_nodes() > 0:
        raw_node_labels = {n: G.nodes[n].get("label", str(n)) for n in G.nodes()}
        disamb_labels = disambiguate_citation_labels(raw_node_labels)
        for n, lbl in disamb_labels.items():
            G.nodes[n]["label"] = lbl

    logger.info(
        f"Co-citation network: {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges"
    )
    return G


def build_keyword_cooccurrence_network(
    articles: list[dict], min_cooccurrence: int = 2
) -> nx.Graph:
    """Red de co-ocurrencia de palabras clave con el vocabulario comun del pipeline (reglas agnosticas
    aprendidas del corpus; ver src.utils.build_corpus_keyword_normalizer)."""
    from src.utils import article_keywords
    G = nx.Graph()

    kw_freq = Counter()
    cooccurrence = Counter()

    for a in articles:
        keywords = article_keywords(a, articles)
        for kw in keywords:
            kw_freq[kw] += 1
        for i in range(len(keywords)):
            for j in range(i + 1, len(keywords)):
                pair = (keywords[i], keywords[j])
                cooccurrence[pair] += 1

    # Agregar nodos y aristas
    for (kw1, kw2), count in cooccurrence.items():
        if count >= min_cooccurrence:
            if not G.has_node(kw1):
                G.add_node(kw1, label=kw1, frequency=kw_freq[kw1])
            if not G.has_node(kw2):
                G.add_node(kw2, label=kw2, frequency=kw_freq[kw2])
            G.add_edge(kw1, kw2, weight=count)

    logger.info(
        f"Keyword co-occurrence: {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges"
    )
    return G


def detect_communities(G: nx.Graph) -> dict:
    """Deteccion de comunidades con Louvain renumeradas por tamaño descendente (Cluster 1 = mayor)."""
    if G.number_of_nodes() == 0:
        return {}
    import community as community_louvain
    raw_partition = community_louvain.best_partition(G, random_state=42)
    counts = Counter(raw_partition.values())
    # Ordenar comunidades de mayor a menor tamaño (1-based como en VOSviewer)
    sorted_comms = [c for c, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0]))]
    remap = {old_c: idx + 1 for idx, old_c in enumerate(sorted_comms)}
    partition = {n: remap[c] for n, c in raw_partition.items()}
    nx.set_node_attributes(G, partition, "community")
    n_communities = len(sorted_comms)
    logger.info(f"Communities detected: {n_communities}")
    return partition


def compute_network_metrics(G: nx.Graph) -> dict:
    """Calcula metricas de red."""
    if G.number_of_nodes() == 0:
        return {"nodes": 0, "edges": 0}

    metrics = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": round(nx.density(G), 4),
    }

    # Componentes conectados
    if not nx.is_directed(G):
        components = list(nx.connected_components(G))
        metrics["connected_components"] = len(components)
        largest_cc = max(components, key=len)
        metrics["largest_component_size"] = len(largest_cc)

    # Centralidades (en componente mas grande si es desconectado)
    if not nx.is_directed(G) and nx.is_connected(G):
        subG = G
    elif not nx.is_directed(G):
        largest_cc = max(nx.connected_components(G), key=len)
        subG = G.subgraph(largest_cc)
    else:
        subG = G

    if subG.number_of_nodes() > 1:
        degree_cent = nx.degree_centrality(subG)
        # Exacta y con distancia = 1/peso. Antes: peso tomado como distancia y, con mas de 500 nodos,
        # muestreo aleatorio SIN semilla (k=100): el resultado cambiaba entre corridas.
        from src.utils import strength_betweenness
        betweenness = strength_betweenness(subG)
        pagerank = nx.pagerank(subG, weight="weight")

        # Top 5 por cada metrica
        metrics["top_degree"] = sorted(
            degree_cent.items(), key=lambda x: x[1], reverse=True
        )[:5]
        metrics["top_betweenness"] = sorted(
            betweenness.items(), key=lambda x: x[1], reverse=True
        )[:5]
        metrics["top_pagerank"] = sorted(
            pagerank.items(), key=lambda x: x[1], reverse=True
        )[:5]

        # Asignar a nodos para exportacion
        nx.set_node_attributes(G, degree_cent, "degree_centrality")
        nx.set_node_attributes(G, betweenness, "betweenness_centrality")
        nx.set_node_attributes(G, pagerank, "pagerank")

    return metrics


def export_network(G: nx.Graph, name: str, output_dir: Path = NETWORKS_DIR) -> None:
    """Exporta red en multiples formatos."""
    if G.number_of_nodes() == 0:
        logger.warning(f"Empty network: {name}, skipping export")
        return

    # Sanitize None values — GEXF does not support NoneType attributes
    _numeric_keys = {
        "year", "citations", "cited_by_count", "num_refs",
        "publication_year", "frequency", "total_cites",
        "cocitation_count", "community", "weight",
        "degree_centrality", "betweenness_centrality", "pagerank",
    }
    for _, data in G.nodes(data=True):
        for k, v in list(data.items()):
            if v is None:
                data[k] = 0 if k in _numeric_keys else ""
    for _, _, data in G.edges(data=True):
        for k, v in list(data.items()):
            if v is None:
                data[k] = 0 if k in _numeric_keys else ""

    # GEXF (Gephi)
    gexf_path = output_dir / f"{name}.gexf"
    nx.write_gexf(G, str(gexf_path))
    logger.info(f"Exported: {gexf_path}")

    # GraphML (VOSviewer)
    graphml_path = output_dir / f"{name}.graphml"
    nx.write_graphml(G, str(graphml_path))
    logger.info(f"Exported: {graphml_path}")

    # HTML interactivo (pyvis) — prune large networks for performance
    try:
        from pyvis.network import Network

        # For very large networks, build pyvis from a top-node subgraph
        MAX_PYVIS_EDGES = 5000
        if G.number_of_edges() > MAX_PYVIS_EDGES:
            deg = dict(G.degree(weight="weight"))
            n_keep = min(200, G.number_of_nodes())
            threshold = sorted(deg.values(), reverse=True)[min(n_keep - 1, len(deg) - 1)]
            top_nodes = [n for n, d in deg.items() if d >= threshold][:n_keep]
            G_pyvis = G.subgraph(top_nodes).copy()
            logger.info(
                f"pyvis pruned for {name}: {G_pyvis.number_of_nodes()} nodes, "
                f"{G_pyvis.number_of_edges()} edges (from {G.number_of_nodes()}/{G.number_of_edges()})"
            )
        else:
            G_pyvis = G

        net = Network(
            height="750px", width="100%",
            bgcolor="#ffffff", font_color="#333333",
        )

        # Agregar nodos con labels legibles
        node_labels = nx.get_node_attributes(G_pyvis, "label")
        node_titles = nx.get_node_attributes(G_pyvis, "title")  # tooltips
        communities = nx.get_node_attributes(G_pyvis, "community")
        color_palette = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
                         "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
                         "#9c755f", "#bab0ac"]

        for node in G_pyvis.nodes():
            label = node_labels.get(node, str(node))
            tooltip = node_titles.get(node, label)
            comm = communities.get(node, 0)
            color = color_palette[comm % len(color_palette)]
            degree = G_pyvis.degree(node, weight="weight")
            size = max(10, min(50, 5 + degree * 2))
            net.add_node(str(node), label=label, title=tooltip,
                         color=color, size=size)

        for u, v, data in G_pyvis.edges(data=True):
            weight = data.get("weight", 1)
            net.add_edge(str(u), str(v), value=weight, title=f"Weight: {weight}")

        net.toggle_physics(True)
        net.set_options("""
        {
            "nodes": {"font": {"size": 14, "face": "arial"}},
            "edges": {"color": {"inherit": "both"}, "smooth": false},
            "physics": {
                "forceAtlas2Based": {
                    "gravitationalConstant": -80,
                    "centralGravity": 0.01,
                    "springLength": 120,
                    "springConstant": 0.02
                },
                "solver": "forceAtlas2Based",
                "stabilization": {"iterations": 200}
            },
            "interaction": {"hover": true, "tooltipDelay": 100}
        }
        """)
        html_path = output_dir / f"{name}.html"
        net.save_graph(str(html_path))
        logger.info(f"Exported: {html_path}")
    except ImportError:
        logger.warning("pyvis not installed, skipping HTML export")


def _community_layout(G: nx.Graph, communities: dict, seed: int = 42) -> dict:
    """Layout que separa comunidades espacialmente con atraccion inter-cluster suave.
    1. Posiciona centros de comunidades multi-nodo en un anillo compacto.
    2. Dentro de cada comunidad, usa spring_layout local (o circular ordenado por grado si es disperso).
    3. Aplica atraccion baricentrica suave entre comunidades conectadas para evitar vacios artificiales.
    """
    if G.number_of_nodes() <= 1:
        return {n: np.array([0.0, 0.0]) for n in G.nodes()}

    if not communities:
        k = 3.0 / np.sqrt(G.number_of_nodes())
        return nx.spring_layout(G, k=k, seed=seed, iterations=200, weight="weight")

    # Agrupar nodos por comunidad
    comm_nodes: dict[int, list] = {}
    for n, c in communities.items():
        if n in G:
            comm_nodes.setdefault(c, []).append(n)

    # Separar comunidades con >=2 nodos de singletons aislados
    multi_comms = [c for c in sorted(comm_nodes.keys()) if len(comm_nodes[c]) >= 2]
    single_comms = [c for c in sorted(comm_nodes.keys()) if len(comm_nodes[c]) == 1]

    comm_centers = {}
    n_multi = max(len(multi_comms), 1)
    base_radius = 1.35 if n_multi <= 4 else (1.55 if n_multi <= 8 else 1.75)

    for i, c in enumerate(multi_comms):
        # Rotar ligeramente el angulo inicial (pi/6) para evitar que un cluster caiga exactamente en la esquina superior izquierda de la leyenda
        angle = (2 * np.pi * i / n_multi) - (np.pi / 6)
        comm_centers[c] = np.array([base_radius * np.cos(angle), base_radius * np.sin(angle)])

    for i, c in enumerate(single_comms):
        angle = (2 * np.pi * i / max(len(single_comms), 1)) + (np.pi / 12)
        outer_r = base_radius * 1.28
        comm_centers[c] = np.array([outer_r * np.cos(angle), outer_r * np.sin(angle)])

    # Layout local dentro de cada comunidad
    pos = {}
    rng = np.random.RandomState(seed)
    for c, nodes in comm_nodes.items():
        if len(nodes) == 1:
            pos[nodes[0]] = comm_centers[c] + rng.randn(2) * 0.03
            continue

        subG = G.subgraph(nodes)
        n_internal_edges = subG.number_of_edges()
        n_nodes = len(nodes)

        if n_internal_edges < n_nodes * 0.3:
            deg_sorted = sorted(nodes, key=lambda n: G.degree(n, weight="weight"), reverse=True)
            local_pos = {}
            for idx, node in enumerate(deg_sorted):
                angle = 2 * np.pi * idx / n_nodes
                r = 0.45 + 0.35 * rng.rand()
                local_pos[node] = np.array([r * np.cos(angle), r * np.sin(angle)])
        else:
            local_k = 1.15 / np.sqrt(max(n_nodes, 1))
            local_pos = nx.spring_layout(subG, k=local_k, seed=seed, iterations=120, weight="weight")

        coords = np.array(list(local_pos.values()))
        spread = max(np.ptp(coords, axis=0).max(), 0.01)
        scale = min(0.95, 0.38 + 0.11 * np.sqrt(n_nodes))
        for n in nodes:
            normalized = (local_pos[n] - coords.mean(axis=0)) / spread * scale
            pos[n] = comm_centers[c] + normalized

    # Atraccion baricentrica suave hacia vecinos externos para cohesion visual
    if len(multi_comms) >= 2:
        new_pos = {}
        for n in G.nodes():
            c_n = communities.get(n)
            ext_vecs = []
            ext_weights = []
            for nb in G.neighbors(n):
                if communities.get(nb) != c_n and nb in pos:
                    w = G[n][nb].get("weight", 1.0)
                    ext_vecs.append(pos[nb])
                    ext_weights.append(w)
            if ext_vecs:
                w_arr = np.array(ext_weights, dtype=float)
                w_arr /= w_arr.sum()
                bary = np.average(np.array(ext_vecs), axis=0, weights=w_arr)
                new_pos[n] = 0.86 * pos[n] + 0.14 * bary
            else:
                new_pos[n] = pos[n]
        pos = new_pos

    return pos


def _filter_edges_for_vis(G: nx.Graph, top_fraction: float = 0.3) -> list:
    """Filtra aristas: solo muestra las mas fuertes para claridad visual."""
    edges = list(G.edges(data=True))
    if not edges:
        return []
    weights = [d.get("weight", 1) for _, _, d in edges]
    threshold = np.percentile(weights, (1 - top_fraction) * 100)
    return [(u, v, d) for u, v, d in edges if d.get("weight", 1) >= threshold]


# Paleta VOSviewer: colores saturados distinguibles
_VOS_CLUSTER_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#ff7f00", "#984ea3",
    "#a65628", "#f781bf", "#66c2a5", "#fc8d62",
    "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f", "#e5c494",
    "#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#999999", "#b3b3b3",
]


def _draw_labels(ax, pos, degrees, max_deg, node_labels_attr, n_labels: int = 35,
                 color: str = "#222222", bg_color: str = "white", bg_alpha: float = 0.75,
                 communities: dict = None):
    """Dibuja labels de los nodos mas importantes con adjustText.
    Si se proporcionan comunidades, distribuye labels proporcionalmente
    entre clusters para que todos tengan representacion.
    """
    if communities:
        # Distribute labels across communities proportionally, min 2 per community
        comm_nodes = {}
        for n, c in communities.items():
            if n in degrees:
                comm_nodes.setdefault(c, []).append(n)

        n_comms = len(comm_nodes)
        # At least 2 labels per community, rest distributed by size
        min_per_comm = min(2, n_labels // max(n_comms, 1))
        remaining = n_labels - min_per_comm * n_comms
        total_nodes = sum(len(ns) for ns in comm_nodes.values())

        top_nodes = []
        for c, nodes in comm_nodes.items():
            # Number of labels for this community
            extra = max(0, int(remaining * len(nodes) / max(total_nodes, 1)))
            n_for_comm = min(len(nodes), min_per_comm + extra)
            sorted_nodes = sorted(nodes, key=lambda n: degrees.get(n, 0), reverse=True)
            top_nodes.extend(sorted_nodes[:n_for_comm])

        # If we still have room, fill with top remaining by degree
        if len(top_nodes) < n_labels:
            used = set(top_nodes)
            remaining_nodes = sorted(
                [n for n in degrees if n not in used],
                key=lambda n: degrees[n], reverse=True
            )
            top_nodes.extend(remaining_nodes[:n_labels - len(top_nodes)])
    else:
        top_nodes = sorted(degrees.keys(), key=lambda n: degrees[n], reverse=True)[:n_labels]

    try:
        from adjustText import adjust_text
        texts = []
        for n in top_nodes:
            if n not in pos:
                continue
            d = degrees.get(n, 0)
            lbl = node_labels_attr.get(n, str(n))[:36]
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
            lbl = node_labels_attr.get(n, str(n))[:36]
            sz = 7.5 + 4.5 * (d / max_deg)
            ax.text(
                pos[n][0], pos[n][1], lbl,
                fontsize=min(sz, 12), fontweight="bold",
                ha="center", va="center", zorder=5, color=color,
                bbox=dict(boxstyle="round,pad=0.2", facecolor=bg_color,
                          edgecolor="none", alpha=bg_alpha),
            )


def visualize_network_vos(
    G: nx.Graph, name: str, output_dir: Path = FIGURES_DIR
) -> None:
    """Visualizacion de alta calidad estilo VOSviewer (cluster view)."""
    if G.number_of_nodes() == 0:
        return

    communities = nx.get_node_attributes(G, "community")
    node_labels_attr = nx.get_node_attributes(G, "label")
    pos = _community_layout(G, communities)

    # Grado ponderado
    degrees = dict(G.degree(weight="weight"))
    max_deg = max(degrees.values()) if degrees else 1
    if max_deg == 0:
        max_deg = 1

    # Tamaño: raiz cuadrada para evitar que nodos grandes dominen
    min_s, max_s = 20, 500
    node_sizes = {
        n: min_s + (max_s - min_s) * np.sqrt(degrees[n] / max_deg)
        for n in G.nodes()
    }

    # Colores
    node_colors = {}
    for n in G.nodes():
        c = token = communities.get(n, 0)
        node_colors[n] = _VOS_CLUSTER_COLORS[c % len(_VOS_CLUSTER_COLORS)]

    # Solo aristas fuertes
    vis_edges = _filter_edges_for_vis(G, top_fraction=0.25)
    edge_weights = [d.get("weight", 1) for _, _, d in vis_edges]
    max_ew = max(edge_weights) if edge_weights else 1

    fig, ax = plt.subplots(figsize=(18, 16))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fafafa")

    # Aristas
    for u, v, d in vis_edges:
        w = d.get("weight", 1)
        alpha = 0.06 + 0.3 * (w / max_ew)
        lw = 0.2 + 2.0 * (w / max_ew)
        cu = node_colors.get(u, "#999999")
        ax.plot(
            [pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
            color=cu, alpha=alpha * 0.5, linewidth=lw, zorder=1
        )

    # Nodos
    for n in G.nodes():
        ax.scatter(
            pos[n][0], pos[n][1], s=node_sizes[n], c=node_colors[n],
            edgecolors="white", linewidths=0.6, alpha=0.85, zorder=3
        )

    # Labels - distributed across all communities
    n_labels = min(40, max(15, G.number_of_nodes() // 4))
    _draw_labels(ax, pos, degrees, max_deg, node_labels_attr, n_labels=n_labels,
                 communities=communities)

    # Leyenda ordenada por tamaño de cluster (excluyendo singletons si hay clusters multi-nodo)
    if communities:
        comm_counts = Counter(communities.values())
        multi_comms = [c for c, cnt in sorted(comm_counts.items(), key=lambda x: (-x[1], x[0])) if cnt >= 2]
        display_comms = multi_comms[:12] if multi_comms else [c for c, _ in comm_counts.most_common(12)]
        legend_handles = []
        for c in display_comms:
            color = _VOS_CLUSTER_COLORS[(c - 1) % len(_VOS_CLUSTER_COLORS)]
            count = comm_counts[c]
            patch = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
                               markersize=9, label=f"Cluster {c} (n={count})")
            legend_handles.append(patch)
        ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(-0.03, 1.02),
                  fontsize=8, framealpha=0.92, title="Communities", title_fontsize=9.5,
                  borderpad=0.9, handletextpad=0.5)

    ax.margins(x=0.14, y=0.12)
    n_clusters = len([c for c, cnt in Counter(communities.values()).items() if cnt >= 2]) if communities else "?"
    if n_clusters == 0 and communities:
        n_clusters = len(set(communities.values()))
    ax.set_title(
        f"{name.replace('_', ' ').title()}\n"
        f"{G.number_of_nodes()} items · {G.number_of_edges()} links · {n_clusters} clusters",
        fontsize=15, fontweight="bold", pad=18
    )
    ax.axis("off")

    plt.tight_layout(pad=2)
    fig.savefig(output_dir / f"network_{name}_cluster.png", dpi=200, facecolor="white")
    plt.close(fig)
    logger.info(f"Saved: network_{name}_cluster.png")


def visualize_network_overlay(
    G: nx.Graph, name: str, year_attr: str = "year",
    output_dir: Path = FIGURES_DIR
) -> None:
    """Overlay por año estilo VOSviewer (gradiente temporal)."""
    if G.number_of_nodes() == 0:
        return

    years = nx.get_node_attributes(G, year_attr)
    if not years or all(v == 0 for v in years.values()):
        logger.info(f"No year data for overlay: {name}")
        return

    communities = nx.get_node_attributes(G, "community")
    pos = _community_layout(G, communities)
    node_labels_attr = nx.get_node_attributes(G, "label")
    degrees = dict(G.degree(weight="weight"))
    max_deg = max(degrees.values()) if degrees else 1
    if max_deg == 0:
        max_deg = 1

    # Tamaños
    min_s, max_s = 20, 500
    node_sizes = [
        min_s + (max_s - min_s) * np.sqrt(degrees[n] / max_deg)
        for n in G.nodes()
    ]

    # Años
    year_values = np.array([years.get(n, 0) for n in G.nodes()], dtype=float)
    valid_mask = year_values > 0
    if valid_mask.sum() == 0:
        return
    vmin = year_values[valid_mask].min()
    vmax = year_values[valid_mask].max()
    if vmin == vmax:
        vmax = vmin + 1

    # Aristas filtradas
    vis_edges = _filter_edges_for_vis(G, top_fraction=0.25)
    edge_weights = [d.get("weight", 1) for _, _, d in vis_edges]
    max_ew = max(edge_weights) if edge_weights else 1

    from matplotlib.colors import LinearSegmentedColormap
    vos_cmap = LinearSegmentedColormap.from_list(
        "vos_overlay", ["#0000ff", "#00cccc", "#00ff00", "#ccff00", "#ffff00", "#ff6600"], N=256
    )

    fig, ax = plt.subplots(figsize=(18, 16))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fafafa")

    # Aristas
    for u, v, d in vis_edges:
        w = d.get("weight", 1)
        alpha = 0.04 + 0.2 * (w / max_ew)
        lw = 0.2 + 1.5 * (w / max_ew)
        ax.plot(
            [pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
            color="#cccccc", alpha=alpha, linewidth=lw, zorder=1
        )

    # Nodos
    xs = [pos[n][0] for n in G.nodes()]
    ys = [pos[n][1] for n in G.nodes()]
    sc = ax.scatter(
        xs, ys, s=node_sizes, c=year_values, cmap=vos_cmap,
        vmin=vmin, vmax=vmax,
        edgecolors="white", linewidths=0.6, alpha=0.9, zorder=3
    )

    cbar = plt.colorbar(sc, ax=ax, shrink=0.45, aspect=30, pad=0.02)
    cbar.set_label("Publication Year", fontsize=11, fontweight="bold")
    cbar.ax.tick_params(labelsize=9)

    # Labels - distributed across communities
    n_labels = min(35, max(12, G.number_of_nodes() // 5))
    _draw_labels(ax, pos, degrees, max_deg, node_labels_attr, n_labels=n_labels,
                 color="#333333", communities=communities)

    ax.set_title(
        f"{name.replace('_', ' ').title()} — Overlay by Year\n"
        f"{G.number_of_nodes()} items · {G.number_of_edges()} links",
        fontsize=15, fontweight="bold", pad=18
    )
    ax.axis("off")

    plt.tight_layout(pad=2)
    fig.savefig(output_dir / f"network_{name}_overlay.png", dpi=200, facecolor="white")
    plt.close(fig)
    logger.info(f"Saved: network_{name}_overlay.png")


def visualize_network_density(
    G: nx.Graph, name: str, output_dir: Path = FIGURES_DIR
) -> None:
    """Mapa de densidad estilo VOSviewer."""
    if G.number_of_nodes() < 5:
        return

    from scipy.ndimage import gaussian_filter

    communities = nx.get_node_attributes(G, "community")
    pos = _community_layout(G, communities)
    node_labels_attr = nx.get_node_attributes(G, "label")
    degrees = dict(G.degree(weight="weight"))
    max_deg = max(degrees.values()) if degrees else 1
    if max_deg == 0:
        max_deg = 1

    xs = np.array([pos[n][0] for n in G.nodes()])
    ys = np.array([pos[n][1] for n in G.nodes()])
    ws = np.array([degrees[n] for n in G.nodes()], dtype=float)

    # Grid de densidad
    pad = 0.5
    x_min, x_max = xs.min() - pad, xs.max() + pad
    y_min, y_max = ys.min() - pad, ys.max() + pad
    grid_res = 400
    density = np.zeros((grid_res, grid_res))
    for x, y, w in zip(xs, ys, ws):
        ix = int((x - x_min) / (x_max - x_min) * (grid_res - 1))
        iy = int((y - y_min) / (y_max - y_min) * (grid_res - 1))
        ix = min(max(ix, 0), grid_res - 1)
        iy = min(max(iy, 0), grid_res - 1)
        density[iy, ix] += w

    # Sigma adaptativo: mas ancho para menos nodos
    sigma = grid_res / (8 + G.number_of_nodes() / 30)
    density = gaussian_filter(density, sigma=sigma)

    from matplotlib.colors import LinearSegmentedColormap
    density_cmap = LinearSegmentedColormap.from_list(
        "vos_density",
        ["#1a1a66", "#1a1aff", "#00aaff", "#00ddaa", "#88ee44",
         "#ffff00", "#ff8800", "#ff0000"],
        N=256
    )

    fig, ax = plt.subplots(figsize=(18, 16))
    bg_color = "#1a1a44"
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    im = ax.imshow(
        density, origin="lower", extent=[x_min, x_max, y_min, y_max],
        cmap=density_cmap, aspect="auto", interpolation="bicubic"
    )

    # Labels top nodos - distributed across communities
    _draw_labels(ax, pos, degrees, max_deg, node_labels_attr, n_labels=25,
                 color="white", bg_color="black", bg_alpha=0.5,
                 communities=communities)

    cbar = plt.colorbar(im, ax=ax, shrink=0.4, aspect=30, pad=0.02)
    cbar.set_label("Item Density", fontsize=11, color="white", fontweight="bold")
    cbar.ax.tick_params(labelsize=8, colors="white")

    ax.set_title(
        f"{name.replace('_', ' ').title()} — Density Visualization\n"
        f"{G.number_of_nodes()} items",
        fontsize=15, fontweight="bold", pad=18, color="white"
    )
    ax.axis("off")

    plt.tight_layout(pad=2)
    fig.savefig(output_dir / f"network_{name}_density.png", dpi=200, facecolor=bg_color)
    plt.close(fig)
    logger.info(f"Saved: network_{name}_density.png")


def build_coauthorship_network(articles: list[dict], level: str = "authors") -> nx.Graph:
    """Red de co-autoria a nivel autores/instituciones/paises normalizada.

    Args:
        articles: Lista de articulos enriquecidos.
        level: 'authors', 'institutions', or 'countries'.
    """
    G = nx.Graph()
    inst_map = build_corpus_institution_normalizer(articles) if level == "institutions" else None

    for a in articles:
        openalex = a.get("openalex", {})
        tc = a.get("cited_by_count") or 0

        if level == "authors":
            entities_set = set()
            for auth in openalex.get("authorships", []):
                name = normalize_author_display_name(auth.get("name", ""))
                if name:
                    entities_set.add(name)
            entities = sorted(entities_set)
        elif level == "institutions":
            entities_set = set()
            for auth in openalex.get("authorships", []):
                for inst in auth.get("institutions", []):
                    name = normalize_institution_name(inst.get("name", ""), corpus_inst_map=inst_map)
                    if name:
                        entities_set.add(name)
            entities = sorted(entities_set)
        elif level == "countries":
            from src.modelo_corpus import article_countries
            entities_set = set(article_countries(a))   # definicion unica del modelo del corpus
            entities = sorted(entities_set)
        else:
            continue

        # Add nodes
        for e in entities:
            if not G.has_node(e):
                G.add_node(e, label=e, frequency=0, total_cites=0)
            G.nodes[e]["frequency"] += 1
            G.nodes[e]["total_cites"] += tc

        # Add edges (co-authorship within same article)
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                if G.has_edge(entities[i], entities[j]):
                    G[entities[i]][entities[j]]["weight"] += 1
                else:
                    G.add_edge(entities[i], entities[j], weight=1)

    logger.info(
        f"Co-authorship ({level}): {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges"
    )
    return G


def build_bibliographic_coupling_by_source(
    articles: list[dict], level: str = "journals"
) -> nx.Graph:
    """Acoplamiento bibliografico a nivel journal/institution/country normalizado.

    Nodos = journals (o inst/countries) del articulo citante.
    Edge weight = numero de referencias compartidas entre las publicaciones de esos nodos.
    """
    entity_refs: dict[str, set] = {}
    entity_cites: dict[str, int] = {}
    inst_map = build_corpus_institution_normalizer(articles) if level == "institutions" else None
    journal_key_to_label: dict[str, str] = {}

    for a in articles:
        refs = set(a.get("referenced_works", []))
        if not refs:
            continue
        openalex = a.get("openalex", {})
        tc = a.get("cited_by_count") or 0

        if level == "journals":
            raw_j = a.get("publication_title", "")
            if raw_j and raw_j != "nan":
                j_key = _journal_match_key(raw_j)
                if j_key:
                    if j_key not in journal_key_to_label:
                        journal_key_to_label[j_key] = format_journal_display_name(raw_j)
                    entity = journal_key_to_label[j_key]
                    entity_refs.setdefault(entity, set()).update(refs)
                    entity_cites[entity] = entity_cites.get(entity, 0) + tc
        elif level == "institutions":
            insts = set()
            for auth in openalex.get("authorships", []):
                for inst in auth.get("institutions", []):
                    name = normalize_institution_name(inst.get("name", ""), corpus_inst_map=inst_map)
                    if name:
                        insts.add(name)
            for inst_name in insts:
                entity_refs.setdefault(inst_name, set()).update(refs)
                entity_cites[inst_name] = entity_cites.get(inst_name, 0) + tc
        elif level == "countries":
            from src.modelo_corpus import article_countries
            countries = set(article_countries(a))   # definicion unica del modelo del corpus
            for country in countries:
                entity_refs.setdefault(country, set()).update(refs)
                entity_cites[country] = entity_cites.get(country, 0) + tc

    G = nx.Graph()
    entities = list(entity_refs.keys())
    for e in entities:
        G.add_node(e, label=e, total_cites=entity_cites.get(e, 0),
                   num_refs=len(entity_refs[e]))

    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            shared = entity_refs[entities[i]] & entity_refs[entities[j]]
            if shared:
                G.add_edge(entities[i], entities[j], weight=len(shared))

    logger.info(
        f"Bib. coupling ({level}): {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges"
    )
    return G


def build_cocitation_by_source(articles: list[dict], min_cocitations: int = 2) -> nx.Graph:
    """Co-citacion a nivel journal con normalizacion de nombres de revista.

    Nodos = journals de las referencias citadas.
    Edge weight = numero de veces que dos journals son co-citados.
    """
    cache = DiskCache(OPENALEX_CACHE_DIR)

    ref_journal: dict[str, str] = {}
    ref_to_articles: dict[str, set] = {}
    journal_key_to_label: dict[str, str] = {}

    for a in articles:
        for ref_url in a.get("referenced_works", []):
            ref_to_articles.setdefault(ref_url, set()).add(a["key"])
            if ref_url not in ref_journal:
                work_id = _extract_work_id(ref_url) or ref_url
                cached = cache.get(f"work:{work_id}")
                if cached and isinstance(cached, dict):
                    primary = cached.get("primary_location", {}) or {}
                    source = primary.get("source", {}) or {}
                    journal = source.get("display_name", "")
                    if not journal:
                        for loc in cached.get("locations", []) or []:
                            src = (loc or {}).get("source", {}) or {}
                            j = src.get("display_name", "")
                            if j:
                                journal = j
                                break
                    if journal:
                        j_key = _journal_match_key(journal)
                        if j_key:
                            if j_key not in journal_key_to_label:
                                journal_key_to_label[j_key] = format_journal_display_name(journal)
                            journal = journal_key_to_label[j_key]
                    ref_journal[ref_url] = journal
                else:
                    ref_journal[ref_url] = ""

    # Agrupar refs por journal
    journal_citing_articles: dict[str, set] = {}
    for ref_url, journal in ref_journal.items():
        if journal:
            citing = ref_to_articles.get(ref_url, set())
            journal_citing_articles.setdefault(journal, set()).update(citing)

    # Build co-citation graph at journal level using inverted index
    G = nx.Graph()
    journals_set = {j for j, citing in journal_citing_articles.items() if len(citing) >= 2}

    for j in journals_set:
        G.add_node(j, label=j, cocitation_count=len(journal_citing_articles[j]))

    # Inverted index: for each article, find which journals it cites -> count co-citations
    article_journals: dict[str, list[str]] = {}
    for j in journals_set:
        for art_key in journal_citing_articles[j]:
            article_journals.setdefault(art_key, []).append(j)

    cocite_counter: Counter = Counter()
    for art_key, jlist in article_journals.items():
        jlist_sorted = sorted(jlist)
        for i in range(len(jlist_sorted)):
            for j_idx in range(i + 1, len(jlist_sorted)):
                cocite_counter[(jlist_sorted[i], jlist_sorted[j_idx])] += 1

    for (j1, j2), count in cocite_counter.items():
        if count >= min_cocitations:
            G.add_edge(j1, j2, weight=count)

    logger.info(
        f"Co-citation (journals): {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges"
    )
    return G


def compute_burt_constraint(G: nx.Graph) -> dict[str, float]:
    """Calcula Burt's constraint para cada nodo.

    Constraint baja = nodo ocupa un hueco estructural (broker).
    Para grafos >1000 nodos, solo computa en el componente gigante.

    Returns:
        Dict {nodo: constraint_value}.
    """
    if G.number_of_nodes() == 0:
        return {}

    # For large graphs, restrict to giant component
    if G.number_of_nodes() > 1000:
        largest_cc = max(nx.connected_components(G), key=len)
        G_work = G.subgraph(largest_cc).copy()
        logger.info(f"  Burt constraint: using giant component ({G_work.number_of_nodes()} nodes)")
    else:
        G_work = G

    try:
        constraint = nx.constraint(G_work)
    except Exception as e:
        logger.warning(f"  Burt constraint computation failed: {e}")
        return {}

    # Clean NaN/None values
    result = {}
    for node, val in constraint.items():
        if val is not None and not (isinstance(val, float) and (val != val)):  # NaN check
            result[node] = round(val, 4)

    return result


def compute_effective_size(G: nx.Graph) -> dict[str, float]:
    """Calcula effective size de Burt para cada nodo.

    Effective size = n_neighbors - redundancy.
    Alto effective size = mayor control de informacion.

    Returns:
        Dict {nodo: effective_size_value}.
    """
    if G.number_of_nodes() == 0:
        return {}

    if G.number_of_nodes() > 1000:
        largest_cc = max(nx.connected_components(G), key=len)
        G_work = G.subgraph(largest_cc).copy()
    else:
        G_work = G

    try:
        eff_size = nx.effective_size(G_work)
    except Exception as e:
        logger.warning(f"  Effective size computation failed: {e}")
        return {}

    result = {}
    for node, val in eff_size.items():
        if val is not None and not (isinstance(val, float) and (val != val)):
            result[node] = round(val, 3)

    return result


def identify_structural_holes(
    G: nx.Graph,
    partition: dict,
    constraint: dict[str, float],
    effective_size: dict[str, float],
) -> dict:
    """Identifica huecos estructurales usando metricas de Burt.

    Detecta:
    - Nodos broker (constraint baja, effective size alta)
    - Pares de comunidades sub-conectadas

    Returns:
        Dict con brokers, subcommunity gaps, y estadisticas.
    """
    if not constraint or not partition:
        return {"brokers": [], "community_gaps": []}

    # Identify brokers: low constraint nodes
    constraint_values = list(constraint.values())
    if not constraint_values:
        return {"brokers": [], "community_gaps": []}

    constraint_values_sorted = sorted(constraint_values)
    # Bottom 25% constraint = potential brokers
    threshold = constraint_values_sorted[max(0, len(constraint_values_sorted) // 4)]

    brokers = []
    for node, c in constraint.items():
        if c <= threshold:
            eff = effective_size.get(node, 0)
            comm = partition.get(node, -1)
            # Count neighbor communities
            neighbor_comms = set()
            for nb in G.neighbors(node):
                nb_comm = partition.get(nb, -1)
                if nb_comm != comm:
                    neighbor_comms.add(nb_comm)

            brokers.append({
                "node": node,
                "constraint": c,
                "effective_size": eff,
                "community": comm,
                "bridges_to_communities": list(neighbor_comms),
                "n_communities_bridged": len(neighbor_comms),
            })

    brokers.sort(key=lambda x: x["constraint"])
    brokers = brokers[:30]  # Top 30 brokers

    # Detect under-connected community pairs
    community_edges: dict[tuple, int] = {}
    for u, v in G.edges():
        cu = partition.get(u, -1)
        cv = partition.get(v, -1)
        if cu != cv:
            pair = (min(cu, cv), max(cu, cv))
            community_edges[pair] = community_edges.get(pair, 0) + 1

    # Count expected edges between communities
    comm_sizes = Counter(partition.values())
    community_gaps = []

    all_pairs = set()
    for c1 in comm_sizes:
        for c2 in comm_sizes:
            if c1 < c2:
                all_pairs.add((c1, c2))

    for pair in all_pairs:
        actual = community_edges.get(pair, 0)
        n1, n2 = comm_sizes[pair[0]], comm_sizes[pair[1]]
        max_possible = n1 * n2
        connectivity = actual / max_possible if max_possible > 0 else 0

        if connectivity < 0.05 and n1 >= 3 and n2 >= 3:  # Less than 5% connected
            community_gaps.append({
                "communities": list(pair),
                "actual_edges": actual,
                "possible_edges": max_possible,
                "connectivity": round(connectivity, 4),
                "sizes": [n1, n2],
            })

    community_gaps.sort(key=lambda x: x["connectivity"])

    return {
        "brokers": brokers,
        "community_gaps": community_gaps[:20],
        "n_total_nodes": len(constraint),
        "mean_constraint": round(sum(constraint.values()) / len(constraint), 4) if constraint else 0,
    }


def generate_structural_holes_report(
    G: nx.Graph,
    holes_data: dict,
    net_name: str,
    output_dir: Path | None = None,
) -> None:
    """Guarda reporte de huecos estructurales como JSON."""
    if output_dir is None:
        output_dir = NETWORKS_DIR

    report = {
        "network": net_name,
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        **holes_data,
    }

    try:
        path = output_dir / f"structural_holes_{net_name}.json"
        import json
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        logger.info(f"  Structural holes report: {path}")
    except Exception as e:
        logger.warning(f"  Could not save structural holes report: {e}")


def compute_interdisciplinarity_score(
    G: nx.Graph, node: str, partition: dict
) -> float:
    """Calcula score de interdisciplinariedad de un nodo.

    Score = proporcion de vecinos que estan en comunidades diferentes
    a la del nodo.

    Returns:
        Float entre 0 (todos vecinos en misma comunidad) y 1 (todos en otras).
    """
    node_comm = partition.get(node)
    if node_comm is None:
        return 0.0

    neighbors = list(G.neighbors(node))
    if not neighbors:
        return 0.0

    other_comm = sum(1 for nb in neighbors if partition.get(nb, node_comm) != node_comm)
    return round(other_comm / len(neighbors), 4)


def generate_bridge_articles_report(
    G: nx.Graph,
    partition: dict,
    articles: list[dict],
    constraint: dict[str, float] | None = None,
    top_n: int = 20,
) -> dict:
    """Genera reporte de articulos puente.

    Top N articulos por betweenness con:
    - Comunidades que conectan
    - Score de interdisciplinariedad
    - Constraint de Burt (si disponible)

    Args:
        G: Grafo de acoplamiento bibliografico.
        partition: Dict {nodo: community_id}.
        articles: Lista de articulos.
        constraint: Dict de constraint de Burt (opcional).
        top_n: Numero de articulos puente a reportar.

    Returns:
        Dict con lista de bridge articles y metadata.
    """
    # Get betweenness from node attributes or compute
    betweenness = {}
    for node in G.nodes():
        bc = G.nodes[node].get("betweenness_centrality", 0)
        if bc:
            betweenness[node] = bc

    if not betweenness:
        # Compute if not in attributes
        from src.utils import strength_betweenness
        betweenness = strength_betweenness(G)

    # Sort by betweenness
    top_nodes = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # Build article lookup
    key_to_article = {a.get("key", ""): a for a in articles}

    bridge_articles = []
    for node, bc in top_nodes:
        article = key_to_article.get(node, {})
        interdisc = compute_interdisciplinarity_score(G, node, partition)
        node_comm = partition.get(node, -1)

        # Find connected communities
        connected_comms = set()
        for nb in G.neighbors(node):
            nb_comm = partition.get(nb, -1)
            if nb_comm != node_comm:
                connected_comms.add(nb_comm)

        entry = {
            "key": node,
            "title": article.get("title", "")[:120],
            "year": article.get("publication_year", ""),
            "citations": article.get("cited_by_count", 0),
            "betweenness": round(bc, 6),
            "community": node_comm,
            "connected_communities": sorted(connected_comms),
            "n_communities_bridged": len(connected_comms),
            "interdisciplinarity_score": interdisc,
        }

        if constraint and node in constraint:
            entry["burt_constraint"] = constraint[node]

        bridge_articles.append(entry)

    report = {
        "top_n": top_n,
        "bridge_articles": bridge_articles,
        "n_total_nodes": G.number_of_nodes(),
    }

    # Save
    try:
        import json
        path = NETWORKS_DIR / "bridge_articles_report.json"
        path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        logger.info(f"  Bridge articles report: {path}")
    except Exception as e:
        logger.warning(f"  Could not save bridge articles report: {e}")

    return report


def run_phase3(
    articles: list[dict],
    resolve_ref_labels: bool = True,
    resolve_rate_limit: float | None = None,
    resolve_batch_size: int = 25,
    resolve_backoff: float = 1.0,
) -> dict:
    """Ejecuta la Fase 3 completa.

    Returns:
        Dict con grafos networkx: {nombre: nx.Graph} para uso en visualizaciones avanzadas.
    """
    logger.info("=== FASE 3: Analisis de Redes ===")
    results = {}
    graphs = {}

    # 1. Acoplamiento bibliografico
    G_coupling = build_bibliographic_coupling_network(articles)
    partition_coupling = detect_communities(G_coupling)
    metrics_coupling = compute_network_metrics(G_coupling)
    export_network(G_coupling, "bibliographic_coupling")
    # For visualization: filter to top nodes if network is large
    if G_coupling.number_of_nodes() > 150:
        deg_coup = dict(G_coupling.degree(weight="weight"))
        n_vis = min(120, G_coupling.number_of_nodes())
        thr = sorted(deg_coup.values(), reverse=True)[min(n_vis - 1, len(deg_coup) - 1)]
        top_coup = [n for n, d in deg_coup.items() if d >= thr][:n_vis]
        G_coup_vis = G_coupling.subgraph(top_coup).copy()
        detect_communities(G_coup_vis)
        logger.info(f"Bibliographic coupling vis filtered: {G_coup_vis.number_of_nodes()} nodes")
    else:
        G_coup_vis = G_coupling
    visualize_network_vos(G_coup_vis, "bibliographic_coupling")
    visualize_network_overlay(G_coup_vis, "bibliographic_coupling")
    visualize_network_density(G_coup_vis, "bibliographic_coupling")
    results["coupling"] = metrics_coupling
    graphs["bibliographic_coupling"] = G_coupling

    # 2. Co-citacion
    G_cocit = build_cocitation_network(
        articles,
        min_cocitations=2,
        resolve_labels=resolve_ref_labels,
        resolve_rate_limit=resolve_rate_limit,
        resolve_batch_size=resolve_batch_size,
        resolve_backoff=resolve_backoff,
    )
    detect_communities(G_cocit)
    metrics_cocit = compute_network_metrics(G_cocit)
    export_network(G_cocit, "cocitation")
    graphs["cocitation"] = G_cocit

    # Para visualizacion: filtrar a nodos con grado alto (top ~80-100)
    if G_cocit.number_of_nodes() > 100:
        degrees_cocit = dict(G_cocit.degree(weight="weight"))
        top_threshold = sorted(degrees_cocit.values(), reverse=True)[min(80, len(degrees_cocit)-1)]
        top_nodes = [n for n, d in degrees_cocit.items() if d >= top_threshold]
        G_cocit_vis = G_cocit.subgraph(top_nodes).copy()
        # Re-detectar comunidades en subgrafo
        detect_communities(G_cocit_vis)
        logger.info(f"Co-citation vis filtered: {G_cocit_vis.number_of_nodes()} nodes")
    else:
        G_cocit_vis = G_cocit
    visualize_network_vos(G_cocit_vis, "cocitation")
    visualize_network_density(G_cocit_vis, "cocitation")
    results["cocitation"] = metrics_cocit

    # 3. Co-ocurrencia de keywords
    G_kw = build_keyword_cooccurrence_network(articles, min_cooccurrence=2)
    detect_communities(G_kw)
    metrics_kw = compute_network_metrics(G_kw)
    export_network(G_kw, "keyword_cooccurrence")
    if G_kw.number_of_nodes() > 150:
        deg_kw = dict(G_kw.degree(weight="weight"))
        thr_kw = sorted(deg_kw.values(), reverse=True)[min(149, len(deg_kw) - 1)]
        top_kw = [n for n, d in deg_kw.items() if d >= thr_kw][:150]
        G_kw_vis = G_kw.subgraph(top_kw).copy()
        detect_communities(G_kw_vis)
    else:
        G_kw_vis = G_kw
    visualize_network_vos(G_kw_vis, "keyword_cooccurrence")
    visualize_network_density(G_kw_vis, "keyword_cooccurrence")
    results["keywords"] = metrics_kw
    graphs["keyword_cooccurrence"] = G_kw

    # 4. Co-authorship networks (authors, institutions, countries)
    for level in ("authors", "institutions", "countries"):
        net_name = f"coauthorship_{level}"
        G_coauth = build_coauthorship_network(articles, level=level)
        detect_communities(G_coauth)
        metrics_coauth = compute_network_metrics(G_coauth)
        export_network(G_coauth, net_name)
        conn_nodes = [n for n, d in G_coauth.degree() if d > 0]
        G_vis_base = G_coauth.subgraph(conn_nodes).copy() if len(conn_nodes) >= 3 else G_coauth
        # Visualize top nodes only for large networks
        if G_vis_base.number_of_nodes() > 80:
            deg = dict(G_vis_base.degree(weight="weight"))
            threshold = sorted(deg.values(), reverse=True)[min(60, len(deg) - 1)]
            top_nodes = [n for n, d in deg.items() if d >= threshold]
            G_vis = G_vis_base.subgraph(top_nodes).copy()
            detect_communities(G_vis)
        else:
            G_vis = G_vis_base.copy()
            detect_communities(G_vis)
        visualize_network_vos(G_vis, net_name)
        results[net_name] = metrics_coauth
        graphs[net_name] = G_coauth

    # 5. Bibliographic coupling by journals/institutions/countries
    for level in ("journals", "institutions", "countries"):
        net_name = f"coupling_{level}"
        G_coup_src = build_bibliographic_coupling_by_source(articles, level=level)
        if G_coup_src.number_of_nodes() > 2:
            detect_communities(G_coup_src)
            metrics_src = compute_network_metrics(G_coup_src)
            export_network(G_coup_src, net_name)
            conn_nodes = [n for n, d in G_coup_src.degree() if d > 0]
            G_vis_base = G_coup_src.subgraph(conn_nodes).copy() if len(conn_nodes) >= 3 else G_coup_src
            # Visualize top nodes for large networks
            if G_vis_base.number_of_nodes() > 60:
                deg = dict(G_vis_base.degree(weight="weight"))
                threshold = sorted(deg.values(), reverse=True)[min(50, len(deg) - 1)]
                top_nodes = [n for n, d in deg.items() if d >= threshold]
                G_vis = G_vis_base.subgraph(top_nodes).copy()
                detect_communities(G_vis)
            else:
                G_vis = G_vis_base.copy()
                detect_communities(G_vis)
            visualize_network_vos(G_vis, net_name)
            results[net_name] = metrics_src
            graphs[net_name] = G_coup_src

    # 6. Co-citation by journals
    G_cocit_journals = build_cocitation_by_source(articles, min_cocitations=2)
    if G_cocit_journals.number_of_nodes() > 2:
        detect_communities(G_cocit_journals)
        metrics_cocit_j = compute_network_metrics(G_cocit_journals)
        export_network(G_cocit_journals, "cocitation_journals")
        if G_cocit_journals.number_of_nodes() > 60:
            deg = dict(G_cocit_journals.degree(weight="weight"))
            threshold = sorted(deg.values(), reverse=True)[min(50, len(deg) - 1)]
            top_nodes = [n for n, d in deg.items() if d >= threshold]
            G_vis = G_cocit_journals.subgraph(top_nodes).copy()
            detect_communities(G_vis)
        else:
            G_vis = G_cocit_journals
        visualize_network_vos(G_vis, "cocitation_journals")
        results["cocitation_journals"] = metrics_cocit_j
        graphs["cocitation_journals"] = G_cocit_journals

    # 7. Burt's Structural Holes + Bridge Articles (on coupling network)
    if G_coupling.number_of_nodes() > 2:
        logger.info("Computing Burt structural holes...")
        try:
            constraint = compute_burt_constraint(G_coupling)
            eff_size = compute_effective_size(G_coupling)

            # Store as node attributes for GEXF export
            for node in G_coupling.nodes():
                G_coupling.nodes[node]["burt_constraint"] = constraint.get(node, 0)
                G_coupling.nodes[node]["effective_size"] = eff_size.get(node, 0)

            holes_data = identify_structural_holes(
                G_coupling, partition_coupling, constraint, eff_size
            )
            generate_structural_holes_report(G_coupling, holes_data, "bibliographic_coupling")

            # Re-export coupling with Burt attributes
            export_network(G_coupling, "bibliographic_coupling")

            # Bridge articles report
            bridge_report = generate_bridge_articles_report(
                G_coupling, partition_coupling, articles,
                constraint=constraint, top_n=20,
            )

            n_brokers = len(holes_data.get("brokers", []))
            n_gaps = len(holes_data.get("community_gaps", []))
            logger.info(
                f"  Structural holes: {n_brokers} brokers, "
                f"{n_gaps} community gaps, "
                f"{len(bridge_report.get('bridge_articles', []))} bridge articles"
            )
        except Exception as e:
            logger.error(f"  Burt structural holes failed: {e}")

    # Resumen
    for net_name, metrics in results.items():
        logger.info(
            f"  {net_name}: {metrics.get('nodes', 0)} nodes, "
            f"{metrics.get('edges', 0)} edges, "
            f"density={metrics.get('density', 'N/A')}"
        )

    logger.info("Fase 3 completada.")
    return graphs
