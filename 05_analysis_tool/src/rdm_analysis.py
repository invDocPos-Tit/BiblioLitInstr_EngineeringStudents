"""Phase 6: Research Dynamics Manifold (RDM).

Metrica bibliometrica original que mapea cada articulo en un espacio
tridimensional [0,1]^3 con tres ejes ortogonales:

  E (Emergencia)  — novedad temporal del articulo
  I (Impacto)     — difusion multi-canal (citacion + social)
  S (Conectividad) — posicion estructural en la red de conocimiento

Clasificacion en 8 arquetipos (octantes del cubo unitario) y
visualizacion 3D interactiva con Plotly.

Justificacion matematica:
  - Normalizacion por rango (CDF empirica): invarianza de escala
  - Power mean sub-lineal (alpha=0.5) en eje I: penaliza impacto
    unidimensional, recompensa difusion multi-canal
  - Score escalar RDM = ||r||_2 / sqrt(3) ∈ [0,1]
  - Corte por medianas: clasificacion adaptativa al corpus
"""
from __future__ import annotations

import json
import logging
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from config import RDM_DIR

logger = logging.getLogger("bibliometrico.rdm")

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
CURRENT_YEAR = datetime.now().year
PRICE_INDEX_WINDOW = 5
KEYWORD_NOVELTY_WINDOW = 2
POWER_MEAN_ALPHA = 0.5

# Pesos por eje
EMERGENCE_WEIGHTS = {"price_index": 0.40, "burst_level": 0.35, "keyword_novelty": 0.25}
CONNECTIVITY_WEIGHTS = {"betweenness": 0.30, "interdisciplinarity": 0.25,
                        "collaboration_breadth": 0.20, "sigma": 0.25}

# Paleta de colores para 8 arquetipos
ARCHETYPE_COLORS = {
    "Paradigm Shifters":  "#e6194b",  # rojo
    "Viral Novelties":    "#f58231",  # naranja
    "Emerging Bridges":   "#ffe119",  # amarillo
    "Isolated Seeds":     "#bfef45",  # lima
    "Established Pillars": "#3cb44b", # verde
    "Canonical Isolates": "#42d4f4",  # cyan
    "Silent Connectors":  "#4363d8",  # azul
    "Peripheral Works":   "#a9a9a9",  # gris
}

ARCHETYPE_DESCRIPTIONS = {
    "Paradigm Shifters":  "High emergence, high impact, well connected — the most dynamic articles",
    "Viral Novelties":    "Novel and high-impact but structurally isolated — breakout papers",
    "Emerging Bridges":   "Novel and well connected, future potential but not yet high impact",
    "Isolated Seeds":     "Early or niche innovations — isolated seeds of future research",
    "Established Pillars": "Intellectual backbone — foundational works well integrated in the network",
    "Canonical Isolates": "Highly cited classics not centrally positioned in the knowledge network",
    "Silent Connectors":  "Underestimated methodological bridges — low novelty and impact but structurally important",
    "Peripheral Works":   "Routine contributions — low across all three dimensions",
}

# Octant definitions for interpretation
ARCHETYPE_OCTANT_RULES = {
    "Paradigm Shifters":  {"E": "above median", "I": "above median", "S": "above median"},
    "Viral Novelties":    {"E": "above median", "I": "above median", "S": "below median"},
    "Emerging Bridges":   {"E": "above median", "I": "below median", "S": "above median"},
    "Isolated Seeds":     {"E": "above median", "I": "below median", "S": "below median"},
    "Established Pillars": {"E": "below median", "I": "above median", "S": "above median"},
    "Canonical Isolates": {"E": "below median", "I": "above median", "S": "below median"},
    "Silent Connectors":  {"E": "below median", "I": "below median", "S": "above median"},
    "Peripheral Works":   {"E": "below median", "I": "below median", "S": "below median"},
}


# ---------------------------------------------------------------------------
# Helpers matematicos
# ---------------------------------------------------------------------------

def _rank_normalize(values: Dict[str, float]) -> Dict[str, float]:
    """Normalizacion CDF empirica por rango (percentil).

    Cada valor se transforma a su percentil en [0, 1].
    Empates reciben el percentil medio del rango.
    """
    if not values:
        return {}
    items = sorted(values.items(), key=lambda x: x[1])
    n = len(items)
    result = {}
    i = 0
    while i < n:
        j = i
        while j < n and items[j][1] == items[i][1]:
            j += 1
        percentile = ((i + j - 1) / 2 + 1) / (n + 1)
        for k in range(i, j):
            result[items[k][0]] = round(percentile, 6)
        i = j
    return result


def _power_mean(values: List[float], alpha: float = POWER_MEAN_ALPHA) -> float:
    """Media potencial generalizada (power mean).

    M_alpha = [ (1/K) * sum(x_j^alpha) ] ^ (1/alpha)

    Con alpha=0.5: sub-lineal, penaliza desbalance entre canales.
    Filtra valores None/NaN. Si no hay valores validos, retorna 0.
    """
    valid = [v for v in values if v is not None and not math.isnan(v)]
    if not valid:
        return 0.0
    k = len(valid)
    s = sum(v ** alpha for v in valid)
    return round((s / k) ** (1.0 / alpha), 6)


def _classify_octant(e: float, i: float, s: float,
                     e_med: float, i_med: float, s_med: float) -> Tuple[str, int]:
    """Clasifica un articulo en uno de los 8 arquetipos segun octante."""
    e_high = e >= e_med
    i_high = i >= i_med
    s_high = s >= s_med

    if e_high and i_high and s_high:
        return "Paradigm Shifters", 1
    elif e_high and i_high and not s_high:
        return "Viral Novelties", 2
    elif e_high and not i_high and s_high:
        return "Emerging Bridges", 3
    elif e_high and not i_high and not s_high:
        return "Isolated Seeds", 4
    elif not e_high and i_high and s_high:
        return "Established Pillars", 5
    elif not e_high and i_high and not s_high:
        return "Canonical Isolates", 6
    elif not e_high and not i_high and s_high:
        return "Silent Connectors", 7
    else:
        return "Peripheral Works", 8


# ---------------------------------------------------------------------------
# Eje E: Emergencia (Novedad Temporal)
# ---------------------------------------------------------------------------

def _compute_price_index_per_article(articles: List[dict]) -> Dict[str, float]:
    """Calcula Price Index por articulo (% refs de ultimos 5 anos)."""
    result = {}
    for a in articles:
        key = a.get("key", a.get("doi", ""))
        if not key:
            continue
        pub_year = a.get("publication_year") or a.get("year")
        if not pub_year:
            result[key] = 0.0
            continue

        # Obtener anos de referencias
        ref_years = []
        for ref in (a.get("referenced_works_resolved") or []):
            ry = ref.get("year") or ref.get("publication_year")
            if ry and isinstance(ry, (int, float)):
                ref_years.append(int(ry))
        if not ref_years:
            # Intentar desde referenced_works raw con cache OpenAlex
            ref_years = a.get("_ref_years", [])
        if not ref_years:
            result[key] = 0.0
            continue

        recent = sum(1 for ry in ref_years if pub_year - ry <= PRICE_INDEX_WINDOW)
        result[key] = round(recent / len(ref_years) * 100, 2)
    return result


def _compute_burst_level_per_article(articles: List[dict]) -> Dict[str, float]:
    """Calcula nivel maximo de burst Kleinberg de las keywords de cada articulo.

    Usa deteccion de bursts por keyword y luego asigna a cada articulo
    el maximo burst de sus keywords en su ano de publicacion.
    """
    try:
        from src.citespace_analysis import _kleinberg_burst_detection
    except ImportError:
        logger.warning("citespace_analysis no disponible; burst_level = 0 para todos")
        return {a.get("key", a.get("doi", "")): 0.0 for a in articles}

    # Construir conteos keyword -> {year: count}
    kw_year_counts: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for a in articles:
        year = a.get("publication_year") or a.get("year")
        if not year:
            continue
        keywords = _extract_keywords(a)
        for kw in keywords:
            kw_year_counts[kw][year] += 1

    # Detectar bursts por keyword
    kw_bursts: Dict[str, List[dict]] = {}
    for kw, year_counts in kw_year_counts.items():
        total = sum(year_counts.values())
        if total < 3:
            continue
        try:
            bursts = _kleinberg_burst_detection(year_counts)
            if bursts:
                kw_bursts[kw] = bursts
        except Exception:
            continue

    # Asignar a cada articulo el max burst de sus keywords en su ano
    result = {}
    for a in articles:
        key = a.get("key", a.get("doi", ""))
        if not key:
            continue
        year = a.get("publication_year") or a.get("year")
        keywords = _extract_keywords(a)
        max_burst = 0.0
        for kw in keywords:
            for burst in kw_bursts.get(kw, []):
                if burst["begin_year"] <= year <= burst["end_year"]:
                    max_burst = max(max_burst, burst.get("strength", burst.get("level", 0)))
        result[key] = float(max_burst)
    return result


def _compute_keyword_novelty_per_article(articles: List[dict]) -> Dict[str, float]:
    """Fraccion de keywords de cada articulo que son nuevas en el corpus.

    Una keyword es 'nueva' si su primera aparicion en el corpus esta
    dentro de ±KEYWORD_NOVELTY_WINDOW anos de la publicacion del articulo.
    """
    # Primera aparicion de cada keyword
    kw_first_year: Dict[str, int] = {}
    for a in articles:
        year = a.get("publication_year") or a.get("year")
        if not year:
            continue
        for kw in _extract_keywords(a):
            if kw not in kw_first_year or year < kw_first_year[kw]:
                kw_first_year[kw] = year

    result = {}
    for a in articles:
        key = a.get("key", a.get("doi", ""))
        if not key:
            continue
        year = a.get("publication_year") or a.get("year")
        keywords = _extract_keywords(a)
        if not keywords or not year:
            result[key] = 0.0
            continue
        novel = sum(1 for kw in keywords
                    if abs(kw_first_year.get(kw, 9999) - year) <= KEYWORD_NOVELTY_WINDOW)
        result[key] = round(novel / len(keywords), 4)
    return result


def _extract_keywords(article: dict) -> List[str]:
    """Palabras clave normalizadas con el vocabulario comun del pipeline (src.utils.article_keywords)."""
    from src.utils import article_keywords
    return article_keywords(article)


def _compute_emergence_axis(articles: List[dict]) -> Tuple[Dict[str, float], Dict[str, dict]]:
    """Compute E (Emergence) axis. Returns (scores, sub_components)."""
    logger.info("Computing Emergence axis (E)...")
    price = _compute_price_index_per_article(articles)
    burst = _compute_burst_level_per_article(articles)
    novelty = _compute_keyword_novelty_per_article(articles)

    # Rank-normalize each sub-indicator
    price_n = _rank_normalize(price)
    burst_n = _rank_normalize(burst)
    novelty_n = _rank_normalize(novelty)

    w = EMERGENCE_WEIGHTS
    result = {}
    sub_components = {}
    all_keys = set(price_n) | set(burst_n) | set(novelty_n)
    for key in all_keys:
        p = price_n.get(key, 0.5)
        b = burst_n.get(key, 0.5)
        k = novelty_n.get(key, 0.5)
        result[key] = round(w["price_index"] * p + w["burst_level"] * b +
                            w["keyword_novelty"] * k, 6)
        sub_components[key] = {
            "price_index_raw": round(price.get(key, 0), 4),
            "price_index_norm": round(p, 4),
            "burst_level_raw": round(burst.get(key, 0), 4),
            "burst_level_norm": round(b, 4),
            "keyword_novelty_raw": round(novelty.get(key, 0), 4),
            "keyword_novelty_norm": round(k, 4),
        }
    logger.info(f"  Emergence: {len(result)} articles, "
                f"mean={np.mean(list(result.values())):.3f}")
    return result, sub_components


# ---------------------------------------------------------------------------
# Eje I: Impacto (Difusion Multi-canal)
# ---------------------------------------------------------------------------

def _compute_impact_axis(articles: List[dict]) -> Tuple[Dict[str, float], Dict[str, dict]]:
    """Compute I (Impact) axis using sub-linear power mean (alpha=0.5)."""
    logger.info("Computing Impact axis (I)...")

    raw_mncs = {}
    raw_fwci = {}
    raw_velocity = {}
    raw_social = {}

    for a in articles:
        key = a.get("key", a.get("doi", ""))
        if not key:
            continue

        # MNCS (ya calculado por leiden_indicators y almacenado en articulo)
        mncs = a.get("mncs")
        if mncs is not None:
            raw_mncs[key] = float(mncs)

        # FWCI / percentil de citacion
        fwci = None
        oa = a.get("openalex", {})
        if isinstance(oa, dict):
            perc = oa.get("cited_by_percentile_year")
            if isinstance(perc, dict):
                fwci = perc.get("max", perc.get("min"))
            elif isinstance(perc, (int, float)):
                fwci = float(perc)
        if fwci is not None:
            raw_fwci[key] = float(fwci)

        # Velocidad de citacion
        cited = a.get("cited_by_count") or a.get("times_cited", 0)
        if isinstance(cited, str):
            try:
                cited = int(cited)
            except ValueError:
                cited = 0
        pub_year = a.get("publication_year") or a.get("year")
        age = max(1, CURRENT_YEAR - pub_year) if pub_year else 1
        raw_velocity[key] = round(cited / age, 4)

        # Atencion social
        altmetric = 0
        if a.get("altmetric_score"):
            try:
                altmetric = float(a["altmetric_score"])
            except (ValueError, TypeError):
                pass
        events = 0
        if a.get("event_total"):
            try:
                events = int(a["event_total"])
            except (ValueError, TypeError):
                pass
        elif a.get("crossref_events_total"):
            try:
                events = int(a["crossref_events_total"])
            except (ValueError, TypeError):
                pass
        raw_social[key] = round(math.log(1 + altmetric + events), 4)

    # Rank-normalize
    mncs_n = _rank_normalize(raw_mncs)
    fwci_n = _rank_normalize(raw_fwci)
    vel_n = _rank_normalize(raw_velocity)
    soc_n = _rank_normalize(raw_social)

    # Power mean per article
    all_keys = set()
    for d in (mncs_n, fwci_n, vel_n, soc_n):
        all_keys.update(d.keys())

    result = {}
    sub_components = {}
    for key in all_keys:
        components = []
        if key in mncs_n:
            components.append(mncs_n[key])
        if key in fwci_n:
            components.append(fwci_n[key])
        if key in vel_n:
            components.append(vel_n[key])
        if key in soc_n:
            components.append(soc_n[key])
        if not components:
            result[key] = 0.5  # neutral imputation
        else:
            result[key] = _power_mean(components)
        sub_components[key] = {
            "mncs_raw": round(raw_mncs.get(key, 0), 4),
            "mncs_norm": round(mncs_n.get(key, 0.5), 4),
            "fwci_raw": round(raw_fwci.get(key, 0), 4),
            "fwci_norm": round(fwci_n.get(key, 0.5), 4),
            "citation_velocity_raw": round(raw_velocity.get(key, 0), 4),
            "citation_velocity_norm": round(vel_n.get(key, 0.5), 4),
            "social_attention_raw": round(raw_social.get(key, 0), 4),
            "social_attention_norm": round(soc_n.get(key, 0.5), 4),
            "n_channels": len(components),
        }
    logger.info(f"  Impact: {len(result)} articles, "
                f"mean={np.mean(list(result.values())):.3f}")
    return result, sub_components


# ---------------------------------------------------------------------------
# Eje S: Conectividad Estructural
# ---------------------------------------------------------------------------

def _compute_connectivity_axis(articles: List[dict],
                               networks: Optional[Dict[str, Any]] = None
                               ) -> Tuple[Dict[str, float], Dict[str, dict]]:
    """Compute S (Structural Connectivity) axis."""
    logger.info("Computing Structural Connectivity axis (S)...")

    article_keys = {a.get("key", a.get("doi", "")) for a in articles if a.get("key") or a.get("doi")}

    raw_bc = {}
    raw_interdisc = {}
    raw_collab = {}
    raw_sigma = {}

    has_network = False

    if networks and "bibliographic_coupling" in networks:
        G = networks["bibliographic_coupling"]
        has_network = True

        # Betweenness centrality (stored as node attribute)
        for node, data in G.nodes(data=True):
            bc = data.get("betweenness_centrality", 0.0)
            raw_bc[node] = float(bc)

            # Interdisciplinarity score
            interdisc = data.get("interdisciplinarity_score")
            if interdisc is not None:
                raw_interdisc[node] = float(interdisc)
            else:
                # Compute on the fly
                community = data.get("community")
                if community is not None:
                    neighbors = list(G.neighbors(node))
                    if neighbors:
                        other = sum(1 for nb in neighbors
                                    if G.nodes[nb].get("community") != community)
                        raw_interdisc[node] = round(other / len(neighbors), 4)

    if not has_network:
        logger.warning("No bibliographic coupling network available; S axis = 0.5 for all articles")
        neutral = {key: 0.5 for key in article_keys}
        neutral_sub = {key: {"betweenness_raw": 0, "betweenness_norm": 0.5,
                             "interdisciplinarity_raw": 0, "interdisciplinarity_norm": 0.5,
                             "collaboration_breadth_raw": 0, "collaboration_breadth_norm": 0.5,
                             "sigma_raw": 0, "sigma_norm": 0.5,
                             "has_network": False} for key in article_keys}
        return neutral, neutral_sub

    # Collaboration breadth (countries per article)
    max_countries = 1
    country_counts = {}
    for a in articles:
        key = a.get("key", a.get("doi", ""))
        if not key:
            continue
        countries = set()
        # From authors_parsed
        for auth in (a.get("authors_parsed") or []):
            if isinstance(auth, dict):
                c = auth.get("country")
                if c:
                    countries.add(c.strip().lower())
        # From article-level country fields
        for field in ("countries", "author_countries"):
            raw = a.get(field, [])
            if isinstance(raw, str):
                raw = [x.strip() for x in raw.split(";") if x.strip()]
            for c in (raw or []):
                if isinstance(c, str) and c.strip():
                    countries.add(c.strip().lower())
        country_counts[key] = len(countries)
        if len(countries) > max_countries:
            max_countries = len(countries)

    for key, count in country_counts.items():
        raw_collab[key] = round(count / max_countries, 4) if max_countries > 0 else 0.0

    # Sigma metric: try to compute from burst data
    try:
        from src.citespace_analysis import _kleinberg_burst_detection
        # Build burst levels per node
        kw_year_counts: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        article_keywords: Dict[str, List[str]] = {}
        article_years: Dict[str, int] = {}
        for a in articles:
            key = a.get("key", a.get("doi", ""))
            year = a.get("publication_year") or a.get("year")
            if key and year:
                kws = _extract_keywords(a)
                article_keywords[key] = kws
                article_years[key] = year
                for kw in kws:
                    kw_year_counts[kw][year] += 1

        # Compute max burst per keyword
        kw_max_burst: Dict[str, float] = {}
        for kw, yc in kw_year_counts.items():
            if sum(yc.values()) < 3:
                continue
            try:
                bursts = _kleinberg_burst_detection(yc)
                if bursts:
                    kw_max_burst[kw] = max(b.get("strength", b.get("level", 0)) for b in bursts)
            except Exception:
                continue

        # Sigma per article: (bc + 1) ^ max_burstness
        for key in raw_bc:
            bc = raw_bc[key]
            year = article_years.get(key)
            kws = article_keywords.get(key, [])
            burstness = 0.0
            for kw in kws:
                burstness = max(burstness, kw_max_burst.get(kw, 0.0))
            if burstness > 0:
                raw_sigma[key] = round((bc + 1) ** min(burstness, 10), 6)  # cap to avoid overflow
            else:
                raw_sigma[key] = 0.0
    except ImportError:
        logger.warning("citespace_analysis no disponible; sigma = 0 para todos")

    # Rank-normalize
    bc_n = _rank_normalize(raw_bc)
    interdisc_n = _rank_normalize(raw_interdisc)
    collab_n = _rank_normalize(raw_collab)
    sigma_n = _rank_normalize(raw_sigma)

    w = CONNECTIVITY_WEIGHTS
    all_keys = set(bc_n) | set(interdisc_n) | set(collab_n) | set(sigma_n)

    result = {}
    sub_components = {}
    for key in all_keys:
        bc_v = bc_n.get(key, 0.5)
        id_v = interdisc_n.get(key, 0.5)
        cl_v = collab_n.get(key, 0.5)
        sg_v = sigma_n.get(key, 0.5)
        result[key] = round(w["betweenness"] * bc_v + w["interdisciplinarity"] * id_v +
                            w["collaboration_breadth"] * cl_v + w["sigma"] * sg_v, 6)
        sub_components[key] = {
            "betweenness_raw": round(raw_bc.get(key, 0), 6),
            "betweenness_norm": round(bc_v, 4),
            "interdisciplinarity_raw": round(raw_interdisc.get(key, 0), 4),
            "interdisciplinarity_norm": round(id_v, 4),
            "collaboration_breadth_raw": round(raw_collab.get(key, 0), 4),
            "collaboration_breadth_norm": round(cl_v, 4),
            "sigma_raw": round(raw_sigma.get(key, 0), 6),
            "sigma_norm": round(sg_v, 4),
            "has_network": True,
        }

    # Articles not in network get neutral score
    for key in article_keys:
        if key not in result:
            result[key] = 0.5
            sub_components[key] = {
                "betweenness_raw": 0, "betweenness_norm": 0.5,
                "interdisciplinarity_raw": 0, "interdisciplinarity_norm": 0.5,
                "collaboration_breadth_raw": 0, "collaboration_breadth_norm": 0.5,
                "sigma_raw": 0, "sigma_norm": 0.5,
                "has_network": False,
            }

    logger.info(f"  Connectivity: {len(result)} articles, "
                f"mean={np.mean(list(result.values())):.3f}")
    return result, sub_components


# ---------------------------------------------------------------------------
# Compute RDM (funcion maestra)
# ---------------------------------------------------------------------------

def compute_rdm(articles: List[dict],
                networks: Optional[Dict[str, Any]] = None) -> List[dict]:
    """Calcula Research Dynamics Manifold para todos los articulos.

    Returns:
        Lista de dicts con keys: key, title, year, E, I, S, RDM,
        archetype, archetype_id, dominant_axis.
    """
    if len(articles) < 5:
        logger.warning(f"Corpus muy pequeno ({len(articles)} articulos); RDM puede ser inestable")

    # Compute three axes (each returns scores + sub-components)
    e_values, e_sub = _compute_emergence_axis(articles)
    i_values, i_sub = _compute_impact_axis(articles)
    s_values, s_sub = _compute_connectivity_axis(articles, networks)

    # Collect all keys
    all_keys = set(e_values) | set(i_values) | set(s_values)

    # Build article metadata lookup
    meta = {}
    for a in articles:
        key = a.get("key", a.get("doi", ""))
        if key:
            meta[key] = {
                "title": (a.get("title") or "")[:120],
                "year": a.get("publication_year") or a.get("year") or 0,
                "doi": a.get("doi", ""),
                "cited_by_count": a.get("cited_by_count", 0),
            }

    # Compute medians for classification
    e_vals = [e_values.get(k, 0.5) for k in all_keys]
    i_vals = [i_values.get(k, 0.5) for k in all_keys]
    s_vals = [s_values.get(k, 0.5) for k in all_keys]
    e_med = float(np.median(e_vals))
    i_med = float(np.median(i_vals))
    s_med = float(np.median(s_vals))
    logger.info(f"Medians — E: {e_med:.3f}, I: {i_med:.3f}, S: {s_med:.3f}")

    if len(articles) < 30:
        logger.warning("Corpus < 30 articulos: clasificacion en octantes puede ser inestable")

    # Build records
    records = []
    for key in sorted(all_keys):
        e = e_values.get(key, 0.5)
        i = i_values.get(key, 0.5)
        s = s_values.get(key, 0.5)
        rdm_score = round(math.sqrt(e**2 + i**2 + s**2) / math.sqrt(3), 6)
        archetype, arch_id = _classify_octant(e, i, s, e_med, i_med, s_med)

        # Dominant axis
        axes = {"E": e, "I": i, "S": s}
        dominant = max(axes, key=axes.get)

        m = meta.get(key, {})
        records.append({
            "key": key,
            "title": m.get("title", ""),
            "year": m.get("year", 0),
            "doi": m.get("doi", ""),
            "cited_by_count": m.get("cited_by_count", 0),
            "E": round(e, 4),
            "I": round(i, 4),
            "S": round(s, 4),
            "RDM": round(rdm_score, 4),
            "archetype": archetype,
            "archetype_id": arch_id,
            "dominant_axis": dominant,
            "emergence_components": e_sub.get(key, {}),
            "impact_components": i_sub.get(key, {}),
            "connectivity_components": s_sub.get(key, {}),
        })

    records.sort(key=lambda r: r["RDM"], reverse=True)
    logger.info(f"RDM computed for {len(records)} articles")
    return records


# ---------------------------------------------------------------------------
# Visualizacion 3D
# ---------------------------------------------------------------------------

def plot_rdm_3d(records: List[dict], output_dir: Path) -> Optional[Path]:
    """Genera scatter 3D interactivo con Plotly."""
    if not HAS_PLOTLY:
        logger.warning("Plotly not available; skipping visualizacion 3D")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    # Separate by archetype
    traces = []
    for archetype, color in ARCHETYPE_COLORS.items():
        subset = [r for r in records if r["archetype"] == archetype]
        if not subset:
            continue

        e_vals = [r["E"] for r in subset]
        i_vals = [r["I"] for r in subset]
        s_vals = [r["S"] for r in subset]
        sizes = [max(4, r["RDM"] * 18) for r in subset]
        hover = [
            f"<b>{r['title'][:80]}</b><br>"
            f"Ano: {r['year']}<br>"
            f"Arquetipo: {r['archetype']}<br>"
            f"E={r['E']:.3f} | I={r['I']:.3f} | S={r['S']:.3f}<br>"
            f"RDM={r['RDM']:.3f} | Eje dominante: {r['dominant_axis']}<br>"
            f"Citas: {r['cited_by_count']}"
            for r in subset
        ]

        traces.append(go.Scatter3d(
            x=e_vals, y=i_vals, z=s_vals,
            mode="markers",
            marker=dict(
                size=sizes,
                color=color,
                opacity=0.8,
                line=dict(width=0.5, color="white"),
            ),
            text=hover,
            hoverinfo="text",
            name=f"{archetype} ({len(subset)})",
        ))

    # Top 10 articles highlighted
    top10 = records[:10]
    if top10:
        traces.append(go.Scatter3d(
            x=[r["E"] for r in top10],
            y=[r["I"] for r in top10],
            z=[r["S"] for r in top10],
            mode="markers+text",
            marker=dict(
                size=8,
                color="gold",
                symbol="diamond",
                opacity=1.0,
                line=dict(width=2, color="black"),
            ),
            text=[f"#{i+1}" for i in range(len(top10))],
            textposition="top center",
            textfont=dict(size=9, color="black"),
            hovertext=[
                f"<b>TOP {i+1}: {r['title'][:60]}</b><br>"
                f"RDM={r['RDM']:.3f}"
                for i, r in enumerate(top10)
            ],
            hoverinfo="text",
            name="Top 10 RDM",
            showlegend=True,
        ))

    # Median planes
    e_med = float(np.median([r["E"] for r in records]))
    i_med = float(np.median([r["I"] for r in records]))
    s_med = float(np.median([r["S"] for r in records]))

    # E = median plane
    traces.append(go.Mesh3d(
        x=[e_med, e_med, e_med, e_med],
        y=[0, 1, 1, 0],
        z=[0, 0, 1, 1],
        i=[0, 0], j=[1, 2], k=[2, 3],
        color="gray", opacity=0.08,
        name="Median E", showlegend=False, hoverinfo="skip",
    ))
    # I = median plane
    traces.append(go.Mesh3d(
        x=[0, 1, 1, 0],
        y=[i_med, i_med, i_med, i_med],
        z=[0, 0, 1, 1],
        i=[0, 0], j=[1, 2], k=[2, 3],
        color="gray", opacity=0.08,
        name="Median I", showlegend=False, hoverinfo="skip",
    ))
    # S = median plane
    traces.append(go.Mesh3d(
        x=[0, 1, 1, 0],
        y=[0, 1, 1, 0],
        z=[s_med, s_med, s_med, s_med],
        i=[0, 0], j=[1, 2], k=[2, 3],
        color="gray", opacity=0.08,
        name="Median S", showlegend=False, hoverinfo="skip",
    ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(
            text="Research Dynamics Manifold (RDM)<br>"
                 "<sub>Emergence × Impact × Structural Connectivity</sub>",
            x=0.5,
        ),
        scene=dict(
            xaxis=dict(title="Emergence (E)", range=[0, 1.05],
                       tickvals=[0, 0.25, 0.5, 0.75, 1.0]),
            yaxis=dict(title="Impact (I)", range=[0, 1.05],
                       tickvals=[0, 0.25, 0.5, 0.75, 1.0]),
            zaxis=dict(title="Connectivity (S)", range=[0, 1.05],
                       tickvals=[0, 0.25, 0.5, 0.75, 1.0]),
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.0)),
        ),
        legend=dict(
            yanchor="top", y=0.95,
            xanchor="left", x=0.01,
            bgcolor="rgba(255,255,255,0.85)",
            font=dict(size=10),
        ),
        width=1920, height=1200,
        margin=dict(l=0, r=0, t=100, b=0),
    )

    _save_figure(fig, output_dir, "rdm_scatter_3d")
    return output_dir / "rdm_scatter_3d.html"


# ---------------------------------------------------------------------------
# Proyecciones 2D
# ---------------------------------------------------------------------------

def plot_rdm_2d_projections(records: List[dict], output_dir: Path) -> Optional[Path]:
    """Genera 3 proyecciones 2D facetadas (E-I, E-S, I-S)."""
    if not HAS_PLOTLY:
        logger.warning("Plotly not available; skipping proyecciones 2D")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = [("E", "I"), ("E", "S"), ("I", "S")]
    axis_labels = {"E": "Emergence (E)", "I": "Impact (I)", "S": "Connectivity (S)"}

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[f"{axis_labels[a]} vs {axis_labels[b]}" for a, b in pairs],
        horizontal_spacing=0.08,
    )

    for col, (ax, ay) in enumerate(pairs, 1):
        for archetype, color in ARCHETYPE_COLORS.items():
            subset = [r for r in records if r["archetype"] == archetype]
            if not subset:
                continue
            fig.add_trace(go.Scatter(
                x=[r[ax] for r in subset],
                y=[r[ay] for r in subset],
                mode="markers",
                marker=dict(
                    size=[max(4, r["RDM"] * 14) for r in subset],
                    color=color, opacity=0.7,
                    line=dict(width=0.5, color="white"),
                ),
                text=[f"{r['title'][:50]}... | RDM={r['RDM']:.3f}" for r in subset],
                hoverinfo="text",
                name=archetype,
                showlegend=(col == 1),
                legendgroup=archetype,
            ), row=1, col=col)

        # Median lines
        med_x = float(np.median([r[ax] for r in records]))
        med_y = float(np.median([r[ay] for r in records]))
        for val, axis in [(med_x, "x"), (med_y, "y")]:
            is_x = axis == "x"
            fig.add_trace(go.Scatter(
                x=[val, val] if is_x else [0, 1],
                y=[0, 1] if is_x else [val, val],
                mode="lines",
                line=dict(color="gray", width=1, dash="dash"),
                showlegend=False, hoverinfo="skip",
            ), row=1, col=col)

    fig.update_layout(
        title=dict(text="RDM \u2014 2D Projections", x=0.5),
        width=1920, height=800,
        margin=dict(t=100, b=60),
    )
    for col, (ax, ay) in enumerate(pairs, 1):
        fig.update_xaxes(title_text=axis_labels[ax], range=[0, 1.05], row=1, col=col)
        fig.update_yaxes(title_text=axis_labels[ay], range=[0, 1.05], row=1, col=col)

    _save_figure(fig, output_dir, "rdm_projections_2d")
    return output_dir / "rdm_projections_2d.html"


# ---------------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------------

def generate_rdm_report(records: List[dict], output_dir: Path) -> Path:
    """Genera reporte markdown con estadisticas y arquetipos."""
    output_dir.mkdir(parents=True, exist_ok=True)

    n = len(records)
    e_vals = [r["E"] for r in records]
    i_vals = [r["I"] for r in records]
    s_vals = [r["S"] for r in records]
    rdm_vals = [r["RDM"] for r in records]

    # Archetype distribution
    arch_counts = Counter(r["archetype"] for r in records)

    lines = [
        "# Research Dynamics Manifold (RDM) — Report",
        "",
        "## Formulation",
        "",
        "Each article is mapped into a three-dimensional space [0,1]^3:",
        "",
        "- **E (Emergence)**: Price Index (0.40) + Kleinberg Burst (0.35) + Keyword Novelty (0.25)",
        "- **I (Impact)**: Sub-linear power mean (alpha=0.5) of MNCS, FWCI, citation velocity, social attention",
        "- **S (Connectivity)**: Betweenness (0.30) + Interdisciplinarity (0.25) + Collaboration (0.20) + Sigma (0.25)",
        "",
        f"**RDM score** = ||r||_2 / sqrt(3) in [0,1]",
        "",
        "## Descriptive Statistics",
        "",
        f"| Metric  | E | I | S | RDM |",
        f"|---------|---|---|---|-----|",
        f"| N       | {n} | {n} | {n} | {n} |",
        f"| Mean    | {np.mean(e_vals):.3f} | {np.mean(i_vals):.3f} | {np.mean(s_vals):.3f} | {np.mean(rdm_vals):.3f} |",
        f"| Median  | {np.median(e_vals):.3f} | {np.median(i_vals):.3f} | {np.median(s_vals):.3f} | {np.median(rdm_vals):.3f} |",
        f"| Std     | {np.std(e_vals):.3f} | {np.std(i_vals):.3f} | {np.std(s_vals):.3f} | {np.std(rdm_vals):.3f} |",
        f"| Min     | {min(e_vals):.3f} | {min(i_vals):.3f} | {min(s_vals):.3f} | {min(rdm_vals):.3f} |",
        f"| Max     | {max(e_vals):.3f} | {max(i_vals):.3f} | {max(s_vals):.3f} | {max(rdm_vals):.3f} |",
        "",
        "## Archetype Distribution",
        "",
        "| # | Archetype | N | % | Description |",
        "|---|-----------|---|---|-------------|",
    ]

    for arch_id in range(1, 9):
        name = [n for n, c in ARCHETYPE_COLORS.items()][arch_id - 1]
        count = arch_counts.get(name, 0)
        pct = round(count / n * 100, 1) if n > 0 else 0
        desc = ARCHETYPE_DESCRIPTIONS.get(name, "")
        lines.append(f"| {arch_id} | **{name}** | {count} | {pct}% | {desc} |")

    lines.extend(["", "## Top 15 Articles by RDM Score", ""])
    lines.append("| # | RDM | E | I | S | Archetype | Title |")
    lines.append("|---|-----|---|---|---|-----------|-------|")
    for idx, r in enumerate(records[:15], 1):
        lines.append(
            f"| {idx} | {r['RDM']:.3f} | {r['E']:.3f} | {r['I']:.3f} | "
            f"{r['S']:.3f} | {r['archetype']} | {r['title'][:70]} |"
        )

    # Top 5 per archetype
    lines.extend(["", "## Top 5 per Archetype", ""])
    for archetype in ARCHETYPE_COLORS:
        subset = [r for r in records if r["archetype"] == archetype]
        if not subset:
            continue
        lines.append(f"### {archetype} ({len(subset)} articles)")
        lines.append("")
        lines.append("| RDM | E | I | S | Titulo |")
        lines.append("|-----|---|---|---|--------|")
        for r in subset[:5]:
            lines.append(
                f"| {r['RDM']:.3f} | {r['E']:.3f} | {r['I']:.3f} | "
                f"{r['S']:.3f} | {r['title'][:70]} |"
            )
        lines.append("")

    # Mathematical justification
    lines.extend([
        "## Mathematical Justification",
        "",
        "### Rank Normalization (Empirical CDF)",
        "Each sub-indicator x is transformed to its percentile F(x) = P(X <= x).",
        "This guarantees scale invariance and comparability across corpora of different sizes.",
        "",
        "### Sub-linear Power Mean (alpha = 0.5)",
        "The I axis uses a generalized power mean:",
        "```",
        "M_{0.5}(x_1,...,x_K) = [ (1/K) * sum(x_j^{0.5}) ]^{2}",
        "```",
        "Key property: penalizes imbalance across diffusion channels.",
        "An article with high impact in a single channel (e.g., only citations,",
        "no social attention) scores lower than one with moderate impact across",
        "multiple channels. This captures the notion of 'broad diffusion'.",
        "",
        "### Axis Orthogonality",
        "- E (temporal novelty) vs I (accumulated citations): correlation ~0",
        "  in the literature (new articles have not had time to accumulate citations).",
        "- E vs S (network centrality): independent (novelty does not imply central position).",
        "- I vs S: weak correlation (~0.2-0.3 typical); rank normalization",
        "  minimizes residual dependence.",
        "",
        "### RDM Scalar Score",
        "```",
        "RDM_i = ||r_i||_2 / sqrt(3) = sqrt(E^2 + I^2 + S^2) / sqrt(3)",
        "```",
        "Range [0,1]. Articles at the corner (1,1,1) achieve RDM=1.",
        "Division by sqrt(3) normalizes the unit cube diagonal.",
        "",
        "### Octant Classification",
        "Corpus medians define the cutting planes, creating an",
        "empirically adaptive taxonomy (no fixed thresholds).",
        "Each octant corresponds to an interpretable archetype.",
        "",
    ])

    report_path = output_dir / "rdm_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Saved: {report_path}")
    return report_path


# ---------------------------------------------------------------------------
# Helper para guardar figuras
# ---------------------------------------------------------------------------

def _save_figure(fig, output_dir: Path, name: str) -> None:
    """Save Plotly figure as HTML and optionally PNG."""
    from src.utils import safe_write_plotly_png

    output_dir.mkdir(parents=True, exist_ok=True)

    html_path = output_dir / f"{name}.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    logger.info(f"Saved: {html_path}")

    png_path = output_dir / f"{name}.png"
    safe_write_plotly_png(fig, png_path, width=2400, height=1500, scale=3, log=logger)


# ---------------------------------------------------------------------------
# Entry point (Phase 6)
# ---------------------------------------------------------------------------

def run_phase6_rdm(articles: List[dict],
                    networks: Optional[Dict[str, Any]] = None) -> dict:
    """Entry point para Phase 6: Research Dynamics Manifold.

    Args:
        articles: Lista de articulos enriquecidos.
        networks: Dict de grafos NetworkX (de Phase 3). Opcional.

    Returns:
        Dict con estadisticas de la fase.
    """
    logger.info("=" * 60)
    logger.info("Phase 6: Research Dynamics Manifold (RDM)")
    logger.info("=" * 60)

    output_dir = RDM_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute RDM
    records = compute_rdm(articles, networks)

    if not records:
        logger.warning("No RDM records generated")
        return {"n_articles": 0, "n_archetypes": 0, "charts_saved": 0}

    # Compute corpus-level statistics for interpretation
    e_vals = [r["E"] for r in records]
    i_vals = [r["I"] for r in records]
    s_vals = [r["S"] for r in records]
    rdm_vals = [r["RDM"] for r in records]
    arch_dist = dict(Counter(r["archetype"] for r in records))
    dominant_dist = dict(Counter(r["dominant_axis"] for r in records))

    # Build enriched JSON with metadata envelope
    enriched_data = {
        "metadata": {
            "description": (
                "Research Dynamics Manifold (RDM) maps each article into a 3D space "
                "[0,1]^3 along Emergence (E), Impact (I), and Structural Connectivity (S) "
                "axes, then classifies it into one of 8 interpretive archetypes based on "
                "corpus-relative medians."
            ),
            "n_articles": len(records),
            "generated_by": "Phase 6 — RDM Analysis",
            "version": "2.0",
        },
        "methodology": {
            "axes": {
                "E_emergence": {
                    "description": "Temporal novelty — how cutting-edge is the article relative to the corpus?",
                    "formula": "E = 0.40 * price_index + 0.35 * burst_level + 0.25 * keyword_novelty",
                    "components": {
                        "price_index": {
                            "weight": 0.40,
                            "description": f"Percentage of references published within the last {PRICE_INDEX_WINDOW} years of the article",
                            "interpretation": "Higher = article cites recent work, suggesting frontier research",
                        },
                        "burst_level": {
                            "weight": 0.35,
                            "description": "Maximum Kleinberg burst strength among the article's keywords at publication year",
                            "interpretation": "Higher = article uses keywords experiencing a sudden popularity surge",
                        },
                        "keyword_novelty": {
                            "weight": 0.25,
                            "description": f"Fraction of keywords whose first corpus appearance is within +/-{KEYWORD_NOVELTY_WINDOW} years",
                            "interpretation": "Higher = article introduces or adopts newly emerging terminology",
                        },
                    },
                },
                "I_impact": {
                    "description": "Multi-channel diffusion — how widely has the article been disseminated?",
                    "formula": "I = power_mean(MNCS, FWCI, citation_velocity, social_attention, alpha=0.5)",
                    "aggregation": (
                        f"Sub-linear power mean (alpha={POWER_MEAN_ALPHA}) penalizes single-channel dominance; "
                        "an article with moderate impact across multiple channels scores higher than one with "
                        "extreme impact in only one channel."
                    ),
                    "components": {
                        "mncs": {
                            "description": "Mean Normalized Citation Score — field-adjusted citation count",
                            "interpretation": "Higher = more citations than field average",
                        },
                        "fwci": {
                            "description": "Field-Weighted Citation Impact — OpenAlex citation percentile by year",
                            "interpretation": "Higher = article is in the top percentile of its field-year cohort",
                        },
                        "citation_velocity": {
                            "description": "cited_by_count / article_age_in_years",
                            "interpretation": "Higher = faster accumulation of citations over time",
                        },
                        "social_attention": {
                            "description": "log(1 + altmetric_score + crossref_events)",
                            "interpretation": "Higher = broader social/media attention beyond academic citations",
                        },
                    },
                },
                "S_connectivity": {
                    "description": "Structural position in the knowledge network — how central and bridging is the article?",
                    "formula": "S = 0.30 * betweenness + 0.25 * interdisciplinarity + 0.20 * collaboration_breadth + 0.25 * sigma",
                    "components": {
                        "betweenness": {
                            "weight": 0.30,
                            "description": "Betweenness centrality in the bibliographic coupling network",
                            "interpretation": "Higher = article sits on many shortest paths between other articles (information broker)",
                        },
                        "interdisciplinarity": {
                            "weight": 0.25,
                            "description": "Fraction of network neighbors belonging to different communities",
                            "interpretation": "Higher = article bridges multiple research communities",
                        },
                        "collaboration_breadth": {
                            "weight": 0.20,
                            "description": "Number of distinct author countries, normalized by corpus max",
                            "interpretation": "Higher = broader international collaboration",
                        },
                        "sigma": {
                            "weight": 0.25,
                            "description": "(betweenness + 1) ^ max_keyword_burstness",
                            "interpretation": "Higher = structurally central article using trending keywords (bridge + burst synergy)",
                        },
                    },
                },
            },
            "normalization": {
                "method": "Rank normalization (empirical CDF)",
                "description": "Each raw sub-indicator is transformed to its corpus percentile in [0,1]. This ensures scale invariance and comparability across corpora of different sizes.",
            },
            "rdm_score": {
                "formula": "RDM = sqrt(E^2 + I^2 + S^2) / sqrt(3)",
                "range": "[0, 1]",
                "interpretation": "Euclidean distance to origin in the unit cube, normalized by the cube diagonal. Articles at (1,1,1) achieve RDM=1.",
            },
            "classification": {
                "method": "Octant classification using corpus medians as decision thresholds",
                "description": "Each axis is split at its median, creating 2^3 = 8 octants. Each octant corresponds to an interpretive archetype. This is empirically adaptive — thresholds adjust to each corpus.",
            },
        },
        "archetypes": {
            name: {
                "id": idx + 1,
                "description": ARCHETYPE_DESCRIPTIONS[name],
                "octant_rule": ARCHETYPE_OCTANT_RULES[name],
                "color": ARCHETYPE_COLORS[name],
                "count": arch_dist.get(name, 0),
                "percentage": round(arch_dist.get(name, 0) / len(records) * 100, 1) if records else 0,
            }
            for idx, name in enumerate(ARCHETYPE_COLORS.keys())
        },
        "corpus_statistics": {
            "medians": {
                "E": round(float(np.median(e_vals)), 4),
                "I": round(float(np.median(i_vals)), 4),
                "S": round(float(np.median(s_vals)), 4),
                "RDM": round(float(np.median(rdm_vals)), 4),
            },
            "means": {
                "E": round(float(np.mean(e_vals)), 4),
                "I": round(float(np.mean(i_vals)), 4),
                "S": round(float(np.mean(s_vals)), 4),
                "RDM": round(float(np.mean(rdm_vals)), 4),
            },
            "std": {
                "E": round(float(np.std(e_vals)), 4),
                "I": round(float(np.std(i_vals)), 4),
                "S": round(float(np.std(s_vals)), 4),
                "RDM": round(float(np.std(rdm_vals)), 4),
            },
            "ranges": {
                "E": [round(min(e_vals), 4), round(max(e_vals), 4)],
                "I": [round(min(i_vals), 4), round(max(i_vals), 4)],
                "S": [round(min(s_vals), 4), round(max(s_vals), 4)],
                "RDM": [round(min(rdm_vals), 4), round(max(rdm_vals), 4)],
            },
            "archetype_distribution": arch_dist,
            "dominant_axis_distribution": dominant_dist,
        },
        "interpretation_guide": {
            "how_to_read_a_record": {
                "E": "Emergence score [0,1] — higher = more novel/cutting-edge",
                "I": "Impact score [0,1] — higher = broader multi-channel diffusion",
                "S": "Structural Connectivity score [0,1] — higher = more central in knowledge network",
                "RDM": "Overall RDM score [0,1] — composite distance from origin",
                "archetype": "Classification into one of 8 interpretive archetypes based on corpus medians",
                "dominant_axis": "Which axis (E/I/S) has the highest value for this article",
                "*_components": "Raw and rank-normalized values for each sub-indicator, enabling drill-down interpretation",
            },
            "cross_phase_value": (
                "RDM data enriches Phase 7 (Knowledge Map) by providing: "
                "(1) archetype distribution as a macro-level finding, "
                "(2) Paradigm Shifters as frontier research candidates, "
                "(3) Silent Connectors as undervalued bridges worth highlighting, "
                "(4) Isolated Seeds as potential emerging topics, "
                "(5) sub-component breakdown enabling causal interpretation of why an article scores high/low."
            ),
        },
        "articles": records,
    }

    # Save enriched data
    data_path = output_dir / "rdm_data.json"
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(enriched_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved: {data_path}")

    # Visualizations
    charts = 0
    if plot_rdm_3d(records, output_dir):
        charts += 1
    if plot_rdm_2d_projections(records, output_dir):
        charts += 1

    # Report
    generate_rdm_report(records, output_dir)

    # Additional 3D visualizations (CiteSpace-inspired)
    try:
        from src.rdm_visualizations_3d import run_rdm_visualizations_3d
        viz_results = run_rdm_visualizations_3d(articles, networks, output_dir)
        charts += viz_results.get("charts_saved", 0)
        logger.info(f"Additional 3D visualizations: {viz_results.get('charts_saved', 0)} charts")
    except ImportError as e:
        logger.warning(f"Additional 3D visualizations skipped: {e}")
    except Exception as e:
        logger.error(f"Additional 3D visualizations failed: {e}")

    # Publication-quality matplotlib figures (English, 400 DPI)
    try:
        from src.rdm_matplotlib_figures import run_rdm_matplotlib_figures
        mpl_results = run_rdm_matplotlib_figures(articles, networks, records, output_dir)
        charts += mpl_results.get("charts_saved", 0)
        logger.info(f"Matplotlib publication figures: {mpl_results.get('charts_saved', 0)} charts")
    except ImportError as e:
        logger.warning(f"Matplotlib publication figures skipped: {e}")
    except Exception as e:
        logger.error(f"Matplotlib publication figures failed: {e}")

    # Summary stats
    arch_counts = Counter(r["archetype"] for r in records)
    n_archetypes = sum(1 for c in arch_counts.values() if c > 0)

    logger.info(f"RDM complete: {len(records)} articles, {n_archetypes} archetypes active")
    for arch, count in sorted(arch_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {arch}: {count} ({count/len(records)*100:.1f}%)")

    return {
        "n_articles": len(records),
        "n_archetypes": n_archetypes,
        "charts_saved": charts,
        "archetype_distribution": dict(arch_counts),
        "top_rdm": records[0]["RDM"] if records else 0,
    }

run_phase10_rdm = run_phase6_rdm
