"""Phase 7: Knowledge Map — Síntesis Bibliométrica Determinista y Mapa Global de Hallazgos.

Recolecta hallazgos cuantitativos de todas las fases bibliométricas del pipeline,
calcula puntajes de impacto (significancia, recencia y corroboración entre fases),
evalúa la sensibilidad de pesos, sintetiza patrones transversales corroborados y
genera:
  - output/knowledge_map/knowledge_map_data.json
  - output/knowledge_map/impact_weight_sensitivity.json
  - output/knowledge_map/knowledge_map_synthesis.json
  - output/knowledge_map/knowledge_map_report.md
  - output/knowledge_map/knowledge_map_overview.html
"""
from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from html import escape as html_escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import (
    CITESPACE_DIR,
    IMPACT_EVOLUTION_DIR,
    LEIDEN_DIR,
    NETWORKS_DIR,
    OUTPUT_DIR,
    RDM_DIR,
    SOTA_DIR,
    SPECTROSCOPY_DIR,
)
from src.utils import (
    canonicalize_corpus_journals,
    load_json,
    normalize_language,
     resolve_openalex_label_cached,
    save_json,
)

logger = logging.getLogger("bibliometrico.knowledge_map")

KNOWLEDGE_MAP_DIR = OUTPUT_DIR / "knowledge_map"

# ---------------------------------------------------------------------------
# Finding types, categories, and sections
# ---------------------------------------------------------------------------

FINDING_TYPES = [
    "corpus_profile", "structural_gap", "declared_gap",
    "research_frontier", "burst_keyword", "pivot_node", "sigma_node",
    "leiden_outlier", "leiden_field_bias", "spectroscopy_lifecycle",
    "spectroscopy_seminal", "network_community", "bridge_article",
    "structural_hole", "thematic_quadrant", "scp_mcp_pattern",
    "citation_class", "sleeping_beauty",
    "fwci_outlier", "concept_evolution", "css_classification",
    "cagr_trend", "halflife_indicator",
    "sdg_alignment", "funding_pattern", "apc_pattern",
    "language_pattern", "metadata_completeness",
    "rdm_archetype", "rdm_paradigm_shifter", "rdm_axis_pattern",
]

FINDING_CATEGORIES = [
    "trend", "gap", "anomaly", "bridge", "sleeping_beauty",
    "convergence", "decline", "emergence",
]

RESULTS_SECTIONS = {
    1: "Perfil del corpus y crecimiento",
    2: "Estructura intelectual (co-citación, espectroscopía, Leiden)",
    3: "Estructura social (colaboración, SCP/MCP, puentes)",
    4: "Estructura conceptual (keywords, clusters, bursts)",
    5: "Frentes de investigación y tópicos emergentes",
    6: "Vacíos estructurales y declarados",
    7: "Indicadores de impacto y calidad (FWCI, MNCS, CSS, APC)",
    8: "Patrones transversales (hallazgos corroborados entre fases)",
}

# Colectores de la Fase 7: id -> (nombre de la fase en el RunRecorder de main.py,
# archivo que lee, o None si recalcula desde `articles` dentro de la propia Fase 7).
_COLLECTOR_SOURCES = {
    "2a":   ("2: Descriptivo", None),
    "2e":   ("2e: Indicadores suplementarios", None),
    "2f":   ("2f: SOTA/gaps", SOTA_DIR / "sota_results.json"),
    "2g":   ("2g: Espectroscopia", SPECTROSCOPY_DIR / "spectroscopy_data.json"),
    "2h":   ("2h: CiteSpace", CITESPACE_DIR / "citespace_data.json"),
    "2i":   ("2i: Leiden", LEIDEN_DIR / "leiden_report.json"),
    "3":    ("3: Redes", NETWORKS_DIR / "bridge_articles_report.json"),
    "3b":   ("3b: Tematico estrategico", None),
    "5":    ("5: Impacto evolucion", IMPACT_EVOLUTION_DIR / "impact_evolution_data.json"),
    "meta": (None, None),
}

# Fases que alimentan cada seccion (segun type_to_section de build_results_skeleton).
# Sirve para que una seccion vacia diga si esta vacia por falta de insumo y no por falta de hallazgos.
_SECTION_SOURCE_PHASES = {
    1: ["2a", "2e", "meta"],
    2: ["2g", "2i"],
    3: ["3", "3b"],
    4: ["2h", "3b"],
    5: ["2f", "2g", "2h", "5"],
    6: ["2f", "3"],
    7: ["5"],
}

_STATUS_LABEL = {
    "ok": "OK",
    "sin_hallazgos": "sin hallazgos",
    "fallida": "FALLIDA EN ESTA CORRIDA",
    "sin_insumo": "NO DISPONIBLE (falta el archivo)",
    "error_lectura": "ERROR AL LEER EL INSUMO",
}

_finding_counter = 0


def _next_finding_id() -> str:
    global _finding_counter
    _finding_counter += 1
    return f"F{_finding_counter:03d}"


def _reset_finding_counter() -> None:
    global _finding_counter
    _finding_counter = 0


def normalize_finding(
    raw: dict,
    phase_id: str,
    finding_type: str,
    category: str = "trend",
    tags: Optional[List[str]] = None,
) -> dict:
    """Convierte hallazgo crudo a esquema estándar."""
    return {
        "id": _next_finding_id(),
        "type": finding_type,
        "category": category if category in FINDING_CATEGORIES else "trend",
        "source_phase": phase_id,
        "title": raw.get("title", ""),
        "description": raw.get("description", ""),
        "impact_score": 0.0,
        "triangulation_score": 0,
        "triangulation_sources": [],
        "evidence": raw.get("evidence", {}),
        "connections": [],
        "tags": [t.lower().strip() for t in (tags or raw.get("tags", [])) if t and str(t).strip()],
        "related_articles": raw.get("related_articles", []),
    }


def _extract_tags_from_text(text: str, max_tags: int = 5) -> List[str]:
    """Extract meaningful tags from text."""
    if not text:
        return []
    stop = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "shall", "should", "may", "might", "can", "could", "this", "that",
        "these", "those", "it", "its", "not", "no", "so", "as", "if", "than",
        "which", "what", "who", "whom", "how", "where", "when", "why", "all",
        "each", "every", "both", "few", "more", "most", "other", "some", "such",
        "only", "own", "same", "also", "very", "just", "about", "above", "after",
        "between", "into", "through", "during", "before", "under", "study",
        "studies", "research", "paper", "article", "using", "based",
    }
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    tags = [w for w in words if w not in stop]
    return list(dict.fromkeys(tags))[:max_tags]


# ===================================================================
# Collectors (1 por fase bibliométrica)
# ===================================================================

def _collect_from_descriptive(articles: List[dict]) -> List[dict]:
    """Phase 2a: Lotka, Bradford, top cited, temporal."""
    findings = []
    if not articles:
        return findings

    canonicalize_corpus_journals(articles)

    # Temporal profile
    years = [a.get("publication_year") for a in articles if a.get("publication_year")]
    years = [int(y) for y in years if isinstance(y, (int, float)) and 1900 < y < 2100]
    if years:
        year_counts = Counter(years)
        min_y, max_y = min(years), max(years)
        peak_year = year_counts.most_common(1)[0] if year_counts else (0, 0)
        findings.append(normalize_finding(
            {
                "title": f"Corpus spans {min_y}-{max_y}, peak year {peak_year[0]} ({peak_year[1]} articles)",
                "description": f"Total {len(articles)} articles across {max_y - min_y + 1} years",
                "evidence": {
                    "year_range": [min_y, max_y],
                    "peak_year": peak_year[0],
                    "peak_count": peak_year[1],
                    "total_articles": len(articles),
                },
                "tags": ["temporal", "growth", "corpus profile"],
            },
            phase_id="2a", finding_type="corpus_profile", category="trend",
        ))

        if max_y - min_y >= 10:
            recent = sum(v for y, v in year_counts.items() if y > max_y - 5)
            earlier = sum(v for y, v in year_counts.items() if max_y - 10 < y <= max_y - 5)
            if earlier > 0:
                growth = (recent - earlier) / earlier * 100
                cat = "emergence" if growth > 50 else ("decline" if growth < -30 else "trend")
                findings.append(normalize_finding(
                    {
                        "title": f"Publication growth {growth:+.0f}% (last 5y vs prior 5y)",
                        "description": f"Recent 5y: {recent} articles, prior 5y: {earlier} articles",
                        "evidence": {"recent_count": recent, "earlier_count": earlier, "growth_pct": round(growth, 1)},
                        "tags": ["growth", "temporal", "trend"],
                    },
                    phase_id="2a", finding_type="corpus_profile", category=cat,
                ))

    # Top cited articles
    cited = sorted(articles, key=lambda a: a.get("cited_by_count", 0) or 0, reverse=True)
    top5 = cited[:5]
    if top5 and (top5[0].get("cited_by_count", 0) or 0) > 0:
        findings.append(normalize_finding(
            {
                "title": f"Top cited: '{top5[0].get('title', '')[:60]}' ({top5[0].get('cited_by_count', 0)} cites)",
                "description": "Top 5 most cited articles in corpus",
                "evidence": {
                    "top_articles": [
                        {
                            "key": a.get("key", ""),
                            "title": a.get("title", "")[:80],
                            "citations": a.get("cited_by_count", 0),
                            "year": a.get("publication_year"),
                        }
                        for a in top5
                    ]
                },
                "tags": ["citations", "impact", "top cited"],
                "related_articles": [a.get("key", "") for a in top5 if a.get("key")],
            },
            phase_id="2a", finding_type="corpus_profile", category="trend",
        ))

    # Lotka: author productivity
    author_counts = Counter()
    for a in articles:
        authors = a.get("authors", [])
        if isinstance(authors, list):
            for au in authors:
                au_str = str(au).strip()
                if au_str:
                    author_counts[au_str] += 1
        elif isinstance(authors, str):
            for au in authors.split(";"):
                au_str = au.strip()
                if au_str:
                    author_counts[au_str] += 1
    if author_counts:
        top_authors = author_counts.most_common(5)
        one_pub = sum(1 for v in author_counts.values() if v == 1)
        findings.append(normalize_finding(
            {
                "title": f"Lotka: {len(author_counts)} authors, {one_pub} ({one_pub*100//max(len(author_counts),1)}%) single-publication",
                "description": f"Top author: {top_authors[0][0]} ({top_authors[0][1]} articles)" if top_authors else "",
                "evidence": {
                    "n_authors": len(author_counts),
                    "single_pub_pct": round(one_pub / max(len(author_counts), 1) * 100, 1),
                    "top_authors": [{"name": n, "count": c} for n, c in top_authors],
                },
                "tags": ["authors", "lotka", "productivity"],
            },
            phase_id="2a", finding_type="corpus_profile", category="trend",
        ))

    # Bradford: journal concentration
    journal_counts = Counter()
    for a in articles:
        j = a.get("publication_title", "") or a.get("journal", "")
        if j and str(j).strip() != "nan":
            journal_counts[str(j).strip()] += 1
    if journal_counts:
        top_journals = journal_counts.most_common(5)
        core_coverage = sum(c for _, c in top_journals)
        findings.append(normalize_finding(
            {
                "title": f"Bradford: top 5 journals cover {core_coverage*100//max(len(articles),1)}% of corpus",
                "description": f"Core journal: {top_journals[0][0]} ({top_journals[0][1]} articles)" if top_journals else "",
                "evidence": {
                    "n_journals": len(journal_counts),
                    "top_journals": [{"name": n, "count": c} for n, c in top_journals],
                    "core_coverage_pct": round(core_coverage / max(len(articles), 1) * 100, 1),
                },
                "tags": ["journals", "bradford", "concentration"],
            },
            phase_id="2a", finding_type="corpus_profile", category="trend",
        ))

    return findings


def _collect_from_supplementary(articles: List[dict]) -> List[dict]:
    """Phase 2e: CAGR and Citation Half-life / Price Index from supplementary indicators."""
    findings = []
    if not articles:
        return findings

    years = [a.get("publication_year") for a in articles if a.get("publication_year")]
    years = [int(y) for y in years if isinstance(y, (int, float)) and 1900 < y < 2100]
    if len(set(years)) >= 5:
        year_counts = Counter(years)
        sorted_years = sorted(year_counts.keys())
        first_y, last_y = sorted_years[0], sorted_years[-1]
        span = last_y - first_y
        if span > 0 and year_counts[first_y] > 0:
            cagr = ((year_counts[last_y] / max(year_counts[first_y], 1)) ** (1.0 / span) - 1) * 100
            findings.append(normalize_finding(
                {
                    "title": f"Publication CAGR: {cagr:.1f}% ({first_y}-{last_y})",
                    "description": "Compound annual growth rate of publications",
                    "evidence": {
                        "cagr": round(cagr, 2),
                        "first_year": first_y,
                        "last_year": last_y,
                        "first_count": year_counts[first_y],
                        "last_count": year_counts[last_y],
                    },
                    "tags": ["cagr", "growth", "temporal"],
                },
                phase_id="2e", finding_type="cagr_trend", category="trend",
            ))

    try:
        from src.supplementary_indicators import compute_citation_halflife
        hl_data = compute_citation_halflife(articles)
        if hl_data and hl_data.get("halflife") is not None:
            findings.append(normalize_finding(
                {
                    "title": (
                        f"Citation half-life: {hl_data['halflife']:.1f} years "
                        f"(Price Index: {hl_data.get('price_index', 0):.1f}%)"
                    ),
                    "description": (
                        f"Median reference age is {hl_data['halflife']:.1f} years "
                        f"(mean {hl_data.get('mean_age', 0):.1f}y) across "
                        f"{hl_data.get('total_references', 0)} resolved references"
                    ),
                    "evidence": hl_data,
                    "tags": ["halflife", "price index", "reference age", "literature aging"],
                },
                phase_id="2e", finding_type="halflife_indicator", category="trend",
            ))
    except Exception as e:
        logger.debug(f"Supplementary half-life collection skipped: {e}")

    return findings


def _collect_from_sota(path: Optional[Path] = None) -> List[dict]:
    """Phase 2f: structural gaps, declared gaps, and frontiers from SOTA analysis."""
    findings = []
    sota_json = path or (SOTA_DIR / "sota_results.json")
    data = load_json(sota_json)
    if not data:
        return findings

    for gap in data.get("structural_gaps", [])[:10]:
        findings.append(normalize_finding(
            {
                "title": f"Gap: {gap.get('item_a', '?')} × {gap.get('item_b', '?')} unexplored",
                "description": f"Structural gap between {gap.get('dimension_a', '')} and {gap.get('dimension_b', '')}",
                "evidence": gap,
                "tags": [gap.get("item_a", ""), gap.get("item_b", "")],
            },
            phase_id="2f", finding_type="structural_gap", category="gap",
        ))

    for fr in data.get("frontiers", [])[:10]:
        gr = fr.get("growth_ratio", 0)
        try:
            gr_label = f"{float(gr):.1f}x"
        except (ValueError, TypeError):
            gr_label = str(gr)
        findings.append(normalize_finding(
            {
                "title": f"Frontier: '{fr.get('item', '')}' (growth {gr_label})",
                "description": "Emerging research frontier with high momentum",
                "evidence": fr,
                "tags": [fr.get("item", ""), fr.get("type", "")],
            },
            phase_id="2f", finding_type="research_frontier", category="emergence",
        ))

    for dg in data.get("declared_gaps", [])[:10]:
        findings.append(normalize_finding(
            {
                "title": f"Declared gap: '{dg.get('sentence', '')[:80]}'",
                "description": f"Gap explicitly mentioned in {dg.get('article_key', '')} ({dg.get('year', '')})",
                "evidence": dg,
                "tags": _extract_tags_from_text(dg.get("sentence", "")),
                "related_articles": [dg.get("article_key", "")] if dg.get("article_key") else [],
            },
            phase_id="2f", finding_type="declared_gap", category="gap",
        ))

    return findings


def _collect_from_spectroscopy(path: Optional[Path] = None) -> List[dict]:
    """Phase 2g: reference spectroscopy lifecycles and seminal works."""
    findings = []
    p = path or (SPECTROSCOPY_DIR / "spectroscopy_data.json")
    data = load_json(p)
    if not data:
        return findings

    seminal = data.get("seminal_works", {})
    # top_global: obras mas citadas POR EL CORPUS (una fila por obra). 'rows' esta ordenada por
    # concepto y decada, asi que sus 10 primeras filas no son las 10 obras mas citadas.
    rows = []
    if isinstance(seminal, dict):
        rows = seminal.get("top_global") or seminal.get("rows", [])
    for sw in rows[:10]:
        ref = sw.get("reference", "") or sw.get("ref", "") or sw.get("title", "") or str(sw)
        n_loc = sw.get("local_citations")
        findings.append(normalize_finding(
            {
                "title": f"Seminal work: {ref[:80]}"
                         + (f" ({sw.get('first_author', '')}, {sw.get('year', '')}; {n_loc} citas en el corpus)" if n_loc else ""),
                "description": "Reference most cited by the corpus (local citations), via reference spectroscopy",
                "evidence": sw,
                "tags": _extract_tags_from_text(ref),
            },
            phase_id="2g", finding_type="spectroscopy_seminal", category="trend",
        ))

    lifecycles = data.get("lifecycles", [])
    if isinstance(lifecycles, dict):
        lifecycles = lifecycles.get("concepts", [])
    for lc in (lifecycles[:10] if isinstance(lifecycles, list) else []):
        concept = lc.get("concept", "") or lc.get("name", "")
        if not concept:
            continue
        stage = (
            lc.get("stage", "")
            or lc.get("lifecycle_stage", "")
            or lc.get("status", "")
            or f"Peak {lc.get('peak_year', 'N/A')} ({lc.get('emergence_year', '')}–{lc.get('last_year', '')}, n={lc.get('total_refs', 0)})"
        )
        cat = "emergence" if ("emerg" in stage.lower() or "frontier" in stage.lower()) else ("decline" if "declin" in stage.lower() else "trend")
        findings.append(normalize_finding(
            {
                "title": f"Topic lifecycle: '{concept}' — {stage}",
                "description": f"Topic at lifecycle stage '{stage}' per reference spectroscopy",
                "evidence": lc,
                "tags": [concept.lower()],
            },
            phase_id="2g", finding_type="spectroscopy_lifecycle", category=cat,
        ))

    return findings


def _collect_from_citespace(
    path: Optional[Path] = None,
    articles: Optional[List[dict]] = None,
) -> List[dict]:
    """Phase 2h: bursts, pivot nodes, and sigma from CiteSpace analysis."""
    findings = []
    p = path or (CITESPACE_DIR / "citespace_data.json")
    data = load_json(p)
    if not data:
        return findings

    bursts = data.get("burst_detection", {})
    burst_items = []
    if isinstance(bursts, dict):
        burst_items = bursts.get("keywords", []) + bursts.get("references", []) + bursts.get("bursts", [])
    for b in burst_items[:15]:
        keyword = b.get("entity", "") or b.get("keyword", "") or b.get("term", "") or b.get("item", "")
        strength = b.get("strength", 0) or b.get("burst_strength", 0)
        start = b.get("begin_year", "") or b.get("start", "") or b.get("burst_start", "")
        end = b.get("end_year", "") or b.get("end", "") or b.get("burst_end", "")
        if not keyword:
            continue
        findings.append(normalize_finding(
            {
                "title": f"Burst: '{keyword[:60]}' {start}-{end} (strength {float(strength):.1f})",
                "description": "Entity experienced citation/usage burst (Kleinberg)",
                "evidence": {"entity": keyword, "strength": strength, "start": start, "end": end},
                "tags": _extract_tags_from_text(keyword) or [keyword.lower()],
            },
            phase_id="2h", finding_type="burst_keyword", category="emergence",
        ))

    pivots = data.get("pivot_nodes", [])
    for pv in pivots[:10]:
        node_raw = pv.get("node", "") or pv.get("label", "") or pv.get("id", "")
        if not node_raw:
            continue
        node_label = resolve_openalex_label_cached(node_raw, articles)
        findings.append(normalize_finding(
            {
                "title": f"Pivot node: {node_label[:65]} (betweenness={pv.get('betweenness', 0):.3f})",
                "description": "Node connecting different research clusters",
                "evidence": {**pv, "resolved_label": node_label},
                "tags": _extract_tags_from_text(node_label),
            },
            phase_id="2h", finding_type="pivot_node", category="bridge",
        ))

    sigmas = data.get("sigma_rankings", [])
    for sg in sigmas[:10]:
        node_raw = sg.get("node", "") or sg.get("label", "")
        sigma = sg.get("sigma", 0)
        if not node_raw or sigma < 1.0:
            continue
        node_label = resolve_openalex_label_cached(node_raw, articles)
        findings.append(normalize_finding(
            {
                "title": f"Sigma node: {node_label[:65]} (σ={sigma:.2f})",
                "description": "High sigma = high novelty × structural centrality",
                "evidence": {**sg, "resolved_label": node_label},
                "tags": _extract_tags_from_text(node_label),
            },
            phase_id="2h", finding_type="sigma_node", category="emergence" if sigma > 2 else "trend",
        ))

    return findings


def _collect_from_leiden(path: Optional[Path] = None) -> List[dict]:
    """Phase 2i: MNCS outliers and field biases from CWTS Leiden indicators."""
    findings = []
    p = path or (LEIDEN_DIR / "leiden_report.json")
    if not p.exists():
        p = LEIDEN_DIR / "leiden_indicators.json"
    data = load_json(p)
    if not data:
        return findings

    mncs = data.get("mncs", {})
    mean_mncs = mncs.get("mean", 0)
    n_above = mncs.get("n_above_1", 0)
    n_total = mncs.get("n_total", 0)
    if n_total > 0:
        findings.append(normalize_finding(
            {
                "title": f"Leiden MNCS: mean={mean_mncs:.2f}, {n_above}/{n_total} above world average",
                "description": "Mean Normalized Citation Score indicates corpus citation impact relative to field",
                "evidence": {"mean_mncs": mean_mncs, "n_above_1": n_above, "n_total": n_total},
                "tags": ["mncs", "leiden", "citation impact", "normalized"],
            },
            phase_id="2i", finding_type="leiden_outlier", category="trend",
        ))

    by_field = mncs.get("by_field", {})
    for field_name, field_data in sorted(
        by_field.items(), key=lambda x: x[1].get("mean_mncs", 0) if isinstance(x[1], dict) else 0, reverse=True
    )[:5]:
        if not isinstance(field_data, dict):
            continue
        fm = field_data.get("mean_mncs", 0)
        if fm > 1.5 or fm < 0.5:
            cat = "anomaly" if fm > 2.0 else ("trend" if fm > 1.0 else "decline")
            findings.append(normalize_finding(
                {
                    "title": f"Leiden field: '{field_name}' MNCS={fm:.2f}",
                    "description": f"Field {'outperforms' if fm > 1 else 'underperforms'} world average",
                    "evidence": {"field": field_name, "mean_mncs": fm, "n_articles": field_data.get("n_articles", 0)},
                    "tags": [field_name.lower(), "leiden", "field impact"],
                },
                phase_id="2i", finding_type="leiden_field_bias", category=cat,
            ))

    return findings


def _collect_from_networks(path: Optional[Path] = None) -> List[dict]:
    """Phase 3: communities, bridge articles, Burt structural holes."""
    findings = []

    bridge_path = path or (NETWORKS_DIR / "bridge_articles_report.json")
    bridge_data = load_json(bridge_path)
    if bridge_data:
        if isinstance(bridge_data, list):
            bridges = bridge_data
        else:
            # Esquema vigente de network_analysis.generate_bridge_articles_report: 'bridge_articles'
            bridges = bridge_data.get("bridge_articles", bridge_data.get("bridges"))
            if bridges is None:
                logger.warning(
                    f"  {bridge_path.name}: esquema no reconocido (claves {sorted(bridge_data)}); "
                    "se esperaba 'bridge_articles'. Revisar network_analysis.generate_bridge_articles_report."
                )
                bridges = []
        for br in (bridges[:10] if isinstance(bridges, list) else []):
            key = br.get("key", "") or br.get("article", "")
            findings.append(normalize_finding(
                {
                    "title": f"Bridge article: {key}",
                    "description": "Article connecting different research communities",
                    "evidence": br,
                    "tags": ["bridge", "interdisciplinary"],
                    "related_articles": [key] if key else [],
                },
                phase_id="3", finding_type="bridge_article", category="bridge",
            ))

    for net_type in [
        "cocitation", "bibliographic_coupling", "coupling", "coauthorship", "keyword",
        "coauthorship_authors", "coauthorship_institutions", "coauthorship_countries",
    ]:
        sh_path = NETWORKS_DIR / f"structural_holes_{net_type}.json"
        sh_data = load_json(sh_path)
        if not sh_data:
            continue
        # Esquema vigente de network_analysis.generate_structural_holes_report:
        # 'brokers' (ya ordenados por constraint de Burt ascendente, cuartil inferior) y
        # 'community_gaps' (pares de comunidades con <5% de los enlaces posibles, ordenados por conectividad).
        brokers = sh_data.get("brokers")
        community_gaps = sh_data.get("community_gaps")
        if brokers is None and community_gaps is None:
            logger.warning(
                f"  {sh_path.name}: esquema no reconocido (claves {sorted(sh_data)}); "
                "se esperaba 'brokers' / 'community_gaps'. Revisar network_analysis.generate_structural_holes_report."
            )
            continue
        mean_c = sh_data.get("mean_constraint")

        for br in (brokers or [])[:5]:
            node = br.get("node", "")
            val = br.get("constraint")
            if not node or not isinstance(val, (int, float)):
                continue
            findings.append(normalize_finding(
                {
                    "title": f"Structural hole broker: '{node}' in {net_type} (constraint={val:.3f})",
                    "description": (
                        f"Low Burt constraint in the {net_type} network"
                        + (f" (network mean {mean_c:.3f})" if isinstance(mean_c, (int, float)) else "")
                        + f"; links {br.get('n_communities_bridged', 0)} communities"
                    ),
                    "evidence": {**br, "network": net_type, "mean_constraint": mean_c},
                    "tags": [str(node).lower(), "structural hole", net_type],
                    "related_articles": [node] if net_type in ("bibliographic_coupling", "cocitation") else [],
                },
                phase_id="3", finding_type="structural_hole", category="bridge",
            ))

        for gap in (community_gaps or [])[:5]:
            comms = gap.get("communities", [])
            if len(comms) != 2:
                continue
            findings.append(normalize_finding(
                {
                    "title": (
                        f"Community gap in {net_type}: communities {comms[0]} and {comms[1]} "
                        f"({gap.get('actual_edges', 0)}/{gap.get('possible_edges', 0)} possible links)"
                    ),
                    "description": (
                        f"Communities of size {gap.get('sizes', ['?', '?'])[0]} and "
                        f"{gap.get('sizes', ['?', '?'])[1]} with connectivity {gap.get('connectivity', 0)} (<5%)"
                    ),
                    "evidence": {**gap, "network": net_type},
                    "tags": ["community gap", net_type],
                },
                phase_id="3", finding_type="community_gap", category="gap",
            ))

    # Antes se emitia aqui un hallazgo "Network built: X (N graph files)" por cada red:
    # contaba archivos en disco, no era un resultado, y entraba al ranking. Eliminado.
    return findings


def _load_bootstrap_ci() -> dict:
    """Load output/reproducibility/bootstrap_ci.json (Phase 5b) if available."""
    return load_json(OUTPUT_DIR / "reproducibility" / "bootstrap_ci.json") or {}


def _collect_from_thematic(articles: List[dict]) -> List[dict]:
    """Phase 3b: Callon thematic keywords and SCP/MCP collaboration."""
    findings = []
    from src.utils import article_keywords
    keyword_counts = Counter()
    for a in articles:
        for k in article_keywords(a, articles):   # vocabulario comun del pipeline
            keyword_counts[k] += 1

    if keyword_counts:
        top_kw = keyword_counts.most_common(10)
        findings.append(normalize_finding(
            {
                "title": f"Top keywords: {', '.join(k for k, _ in top_kw[:5])}",
                "description": "Most frequent author keywords across corpus",
                "evidence": {"top_keywords": [{"keyword": k, "count": c} for k, c in top_kw]},
                "tags": [k for k, _ in top_kw[:5]],
            },
            phase_id="3b", finding_type="thematic_quadrant", category="trend",
        ))

    try:
        from src.strategic_thematic_analysis import compute_scp_mcp
        summary = compute_scp_mcp(articles).get("summary", {})
    except Exception as e:
        logger.warning(f"SCP/MCP: compute_scp_mcp failed ({e})")
        summary = {}

    classified = summary.get("classified", 0)
    if classified > 0:
        scp_count = summary.get("scp", 0)
        mcp_count = summary.get("mcp", 0)
        mcp_pct = summary.get("mcp_ratio", 0) * 100
        evidence = {
            "scp": scp_count,
            "mcp": mcp_count,
            "mcp_pct": round(mcp_pct, 1),
            "classified": classified,
            "unknown_country": summary.get("unknown_country", 0),
            "corpus_total": summary.get("total_articles", len(articles)),
        }
        ci = _load_bootstrap_ci().get("mcp_rate", {})
        if ci:
            evidence["mcp_pct_ci95"] = [
                round(ci.get("ci_lo", 0) * 100, 1),
                round(ci.get("ci_hi", 0) * 100, 1),
            ]
        findings.append(normalize_finding(
            {
                "title": (
                    f"Collaboration: {mcp_pct:.1f}% multi-country (MCP), "
                    f"{100 - mcp_pct:.1f}% single-country (SCP)"
                ),
                "description": (
                    f"SCP={scp_count}, MCP={mcp_count} of {classified} "
                    f"classifiable articles (effective N; "
                    f"{summary.get('unknown_country', 0)} lack country data)"
                ),
                "evidence": evidence,
                "tags": ["collaboration", "scp", "mcp", "international"],
            },
            phase_id="3b", finding_type="scp_mcp_pattern", category="trend",
        ))

    return findings


def _collect_from_impact_evolution(path: Optional[Path] = None) -> List[dict]:
    """Phase 5 & 5b: FWCI, MNCS/PP(top10%), Price Index, Sleeping Beauties, CSS, APC."""
    findings = []
    p = path or (IMPACT_EVOLUTION_DIR / "impact_evolution_data.json")
    data = load_json(p) or {}
    boot = _load_bootstrap_ci()
    if not data and not boot:
        return findings

    fwci = data.get("fwci", {})
    fwci_mean = fwci.get("fwci_mean", 0)
    if fwci_mean:
        n_fwci = fwci.get("n_articles_with_fwci", 0)
        evidence = {
            "fwci_mean": fwci_mean,
            "fwci_median": fwci.get("fwci_median", 0),
            "pct_above_world_avg": fwci.get("pct_above_1", 0),
            "effective_n": n_fwci,
        }
        bf = boot.get("fwci", {}).get("FWCI_mean", {})
        if bf:
            evidence["fwci_mean_ci95"] = [
                round(bf.get("ci_lo", 0), 2), round(bf.get("ci_hi", 0), 2),
            ]
        findings.append(normalize_finding(
            {
                "title": f"FWCI corpus mean: {fwci_mean:.2f} (N={n_fwci})",
                "description": (
                    f"Field-Weighted Citation Impact "
                    f"{'above' if fwci_mean > 1 else 'below'} world average; "
                    f"median {fwci.get('fwci_median', 0):.2f}, "
                    f"{fwci.get('pct_above_1', 0):.1f}% of articles above world average"
                ),
                "evidence": evidence,
                "tags": ["fwci", "impact", "normalized", "citation"],
            },
            phase_id="5", finding_type="fwci_outlier", category="trend",
        ))

    mncs = boot.get("mncs_top10", {})
    if mncs:
        mncs_mean = mncs.get("MNCS_mean", {})
        pp10 = mncs.get("PP_top10_field", {})
        findings.append(normalize_finding(
            {
                "title": (
                    f"PP(top 10%): {pp10.get('point', 0) * 100:.1f}% "
                    f"({pp10.get('n_top10', 0)} articles in the field's top decile)"
                ),
                "description": (
                    f"Proportion of the corpus among the most-cited 10% of their field. "
                    f"Bootstrap MNCS={mncs_mean.get('point', 0):.2f} (N={mncs_mean.get('effective_n', 0)})."
                ),
                "evidence": {
                    "pp_top10": pp10.get("point", 0),
                    "pp_top10_ci95": [pp10.get("ci_lo", 0), pp10.get("ci_hi", 0)],
                    "n_top10": pp10.get("n_top10", 0),
                    "mncs_bootstrap_crosscheck": mncs_mean.get("point", 0),
                    "mncs_crosscheck_ci95": [mncs_mean.get("ci_lo", 0), mncs_mean.get("ci_hi", 0)],
                    "mncs_crosscheck_n": mncs_mean.get("effective_n", 0),
                },
                "tags": ["pp-top10", "mncs", "impact", "normalized", "leiden"],
            },
            phase_id="5", finding_type="fwci_outlier", category="trend",
        ))

    price = data.get("price_index", {})
    if price.get("price_index_median"):
        findings.append(normalize_finding(
            {
                "title": f"Price Index median: {price['price_index_median']:.1f}%",
                "description": (
                    f"{price['price_index_median']:.1f}% of cited references are "
                    f"<=5 years old (N={price.get('n_articles', 0)}); median "
                    f"reference age {price.get('median_ref_age_overall', 0):.1f} years"
                ),
                "evidence": price,
                "tags": ["price index", "reference age", "literature aging"],
            },
            phase_id="5", finding_type="halflife_indicator", category="trend",
        ))

    beauty = data.get("beauty_coefficient", {})
    for sb in beauty.get("top_sleeping_beauties", [])[:5]:
        coeff = sb.get("beauty_coefficient", 0)
        key = sb.get("key", "")
        if coeff <= 0:
            continue
        findings.append(normalize_finding(
            {
                "title": f"Sleeping beauty: {key} (B={coeff:.2f})",
                "description": (
                    f"Delayed-recognition pattern: peak {sb.get('peak_year', '?')}, "
                    f"{sb.get('sleeping_years', 0)} dormant years, "
                    f"{sb.get('cited_by_count', 0)} total citations"
                ),
                "evidence": sb,
                "tags": ["sleeping beauty", "delayed recognition", "beauty coefficient"],
                "related_articles": [key] if key else [],
            },
            phase_id="5", finding_type="sleeping_beauty", category="sleeping_beauty",
        ))

    css = data.get("css_classification", {})
    dist = css.get("class_distribution", {})
    if dist:
        outstanding = css.get("outstanding_articles", [])
        findings.append(normalize_finding(
            {
                "title": "CSS citation trajectories: " + ", ".join(f"{n} {cls.lower()}" for cls, n in dist.items()),
                "description": f"Characteristic Scores and Scales classification of {css.get('n_articles', 0)} articles",
                "evidence": {"class_distribution": dist, "outstanding_articles": outstanding},
                "tags": ["css", "citation trajectory", "classification"],
                "related_articles": [a.get("key", "") for a in outstanding if a.get("key")],
            },
            phase_id="5", finding_type="css_classification", category="trend",
        ))

    apc = data.get("apc_impact", {})
    if apc.get("n_apc"):
        evidence = dict(apc)
        adiff = boot.get("apc_difference", {})
        if adiff:
            dm = adiff.get("diff_mean", {})
            evidence["bootstrap_diff_mean"] = dm.get("point", 0)
            evidence["bootstrap_diff_ci95"] = [dm.get("ci_lo", 0), dm.get("ci_hi", 0)]
            evidence["mann_whitney_p"] = adiff.get("mann_whitney_p_two_sided")
        findings.append(normalize_finding(
            {
                "title": (
                    f"APC-paid articles: {apc['apc_mean_citations']:.2f} mean citations "
                    f"vs {apc.get('non_apc_mean_citations', 0):.2f} (no APC)"
                ),
                "description": (
                    f"{apc['n_apc']} APC-paid vs {apc.get('n_non_apc', 0)} non-APC; "
                    f"mean APC fee ${apc.get('apc_mean_usd', 0):.0f}."
                ),
                "evidence": evidence,
                "tags": ["apc", "open access", "citation impact"],
            },
            phase_id="5", finding_type="apc_pattern", category="trend",
        ))

    concept_evo = data.get("concept_evolution", {})
    lifecycles = concept_evo.get("lifecycles", [])
    if lifecycles:
        phase_counts = Counter(lc.get("phase", "unknown") for lc in lifecycles)
        top_by_phase: Dict[str, List[str]] = {}
        for lc in lifecycles:
            top_by_phase.setdefault(lc.get("phase", "unknown"), []).append(lc.get("concept", ""))
        findings.append(normalize_finding(
            {
                "title": "Topic lifecycle phases: " + ", ".join(f"{n} {ph}" for ph, n in phase_counts.most_common()),
                "description": f"Developmental-phase classification of {len(lifecycles)} top topics/concepts.",
                "evidence": {
                    "phase_counts": dict(phase_counts),
                    "concepts_by_phase": {ph: cs[:8] for ph, cs in top_by_phase.items()},
                },
                "tags": ["concept lifecycle", "concept evolution", "temporal"],
            },
            phase_id="5", finding_type="concept_evolution", category="trend",
        ))

    return findings


def _collect_from_corpus_metadata(articles: List[dict]) -> List[dict]:
    """Cross-phase: SDG, funding, language, APC, primary_topic, metadata completeness."""
    findings = []
    if not articles:
        return findings

    n = len(articles)

    sdg_counts = Counter()
    articles_with_sdg = 0
    for a in articles:
        sdgs = a.get("openalex", {}).get("sustainable_development_goals", [])
        if sdgs:
            articles_with_sdg += 1
            for sdg in sdgs:
                name = sdg.get("display_name", "")
                if name:
                    sdg_counts[name] += 1

    if sdg_counts:
        top_sdgs = sdg_counts.most_common(5)
        sdg_pct = round(articles_with_sdg / max(n, 1) * 100, 1)
        findings.append(normalize_finding(
            {
                "title": f"SDGs: {sdg_pct}% mapped, top={top_sdgs[0][0]} ({top_sdgs[0][1]} articles)",
                "description": f"{articles_with_sdg}/{n} articles have SDG mapping",
                "evidence": {"sdg_pct": sdg_pct, "top_sdgs": [{"name": s, "count": c} for s, c in top_sdgs]},
                "tags": ["sdg", "sustainability"] + [s.lower() for s, _ in top_sdgs[:3]],
            },
            phase_id="2a", finding_type="sdg_alignment", category="trend",
        ))

    funded = 0
    funder_counts = Counter()
    for a in articles:
        funders = a.get("funding", [])
        if funders:
            funded += 1
            for f in funders:
                name = f.get("name", "")
                if name:
                    funder_counts[name] += 1

    if funded > 0:
        funded_pct = round(funded / max(n, 1) * 100, 1)
        top_funders = funder_counts.most_common(5)
        findings.append(normalize_finding(
            {
                "title": f"Funding: {funded_pct}% articles funded, {len(funder_counts)} unique funders",
                "description": f"Top funder: {top_funders[0][0]} ({top_funders[0][1]} articles)" if top_funders else "",
                "evidence": {
                    "funded_pct": funded_pct,
                    "funded_count": funded,
                    "unique_funders": len(funder_counts),
                    "top_funders": [{"name": f, "count": c} for f, c in top_funders],
                },
                "tags": ["funding", "grants"] + [f.lower() for f, _ in top_funders[:2]],
            },
            phase_id="2a", finding_type="funding_pattern", category="trend",
        ))

    lang_counts = Counter()
    for a in articles:
        lang = normalize_language(a.get("language", "") or "")
        if lang and lang not in ("", "nan"):
            lang_counts[lang] += 1

    if lang_counts:
        top_lang = lang_counts.most_common(1)[0]
        top_pct = round(top_lang[1] / max(n, 1) * 100, 1)
        findings.append(normalize_finding(
            {
                "title": f"Language: {len(lang_counts)} languages, {top_lang[0]} dominant ({top_pct}%)",
                "description": "Publication language distribution across corpus",
                "evidence": {"top_languages": lang_counts.most_common(5), "n_languages": len(lang_counts)},
                "tags": ["language", "diversity", top_lang[0].lower()],
            },
            phase_id="2a", finding_type="language_pattern", category="trend",
        ))

    with_topic = sum(1 for a in articles if a.get("openalex", {}).get("primary_topic", {}).get("name"))
    domain_counts = Counter()
    for a in articles:
        pt = a.get("openalex", {}).get("primary_topic", {}) or {}
        d = pt.get("domain", "")
        if d:
            domain_counts[d] += 1

    if with_topic > 0 and domain_counts:
        topic_pct = round(with_topic / max(n, 1) * 100, 1)
        top_domain = domain_counts.most_common(1)[0]
        findings.append(normalize_finding(
            {
                "title": f"Research domains: {top_domain[0]} dominant ({top_domain[1]} articles), {topic_pct}% classified",
                "description": f"{len(domain_counts)} domains identified via OpenAlex primary_topic",
                "evidence": {"topic_coverage_pct": topic_pct, "domains": domain_counts.most_common()},
                "tags": ["domain", "classification", top_domain[0].lower()],
            },
            phase_id="2a", finding_type="corpus_profile", category="trend",
        ))

    fields_to_check = {
        "abstract": lambda a: bool(a.get("abstract") and a["abstract"] not in ("", "nan")),
        "doi": lambda a: bool(a.get("doi")),
        "affiliations": lambda a: bool(a.get("affiliations") or a.get("affiliations_raw")),
        "funding": lambda a: bool(a.get("funding")),
        "openalex_id": lambda a: bool(a.get("openalex", {}).get("openalex_id")),
        "language": lambda a: bool(a.get("language") and a["language"] not in ("", "nan")),
        "issn": lambda a: bool(a.get("issn") and a["issn"] not in ("", "nan")),
        "references": lambda a: bool(a.get("references_resolved") or a.get("referenced_works")),
    }
    completeness = {}
    for field, checker in fields_to_check.items():
        count = sum(1 for a in articles if checker(a))
        completeness[field] = {"count": count, "pct": round(count / max(n, 1) * 100, 1)}

    avg_completeness = round(
        sum(v["pct"] for v in completeness.values()) / max(len(completeness), 1), 1
    )
    gaps = {k: v for k, v in completeness.items() if v["pct"] < 50}
    findings.append(normalize_finding(
        {
            "title": f"Metadata completeness: {avg_completeness}% average across {len(fields_to_check)} fields",
            "description": "Gaps (<50%): " + ", ".join(f"{k}({v['pct']}%)" for k, v in gaps.items()) if gaps else "All core fields above 50% coverage",
            "evidence": {"completeness": completeness, "avg_pct": avg_completeness, "gaps": list(gaps.keys())},
            "tags": ["metadata", "completeness", "data quality"],
        },
        phase_id="1", finding_type="metadata_completeness",
        category="anomaly" if gaps else "trend",
    ))

    return findings


def _collect_from_rdm(path: Optional[Path] = None) -> List[dict]:
    """Phase 6: Research Dynamics Manifold — archetypes, dynamics, 3D patterns."""
    findings = []
    p = path or (RDM_DIR / "rdm_data.json")
    raw = load_json(p)
    if not raw:
        return findings

    if isinstance(raw, dict) and "articles" in raw:
        data = raw["articles"]
    elif isinstance(raw, list):
        data = raw
    else:
        return findings

    if not data:
        return findings

    n = len(data)
    arch_counts = Counter(r["archetype"] for r in data)
    dominant_arch = arch_counts.most_common(1)[0] if arch_counts else ("", 0)
    minority_archs = [a for a, c in arch_counts.items() if c <= max(1, n * 0.05)]
    dist_desc = "; ".join(f"{a}: {c} ({c/n*100:.0f}%)" for a, c in arch_counts.most_common())

    findings.append(normalize_finding(
        {
            "title": f"RDM: {len(arch_counts)} archetypes — dominant: {dominant_arch[0]} ({dominant_arch[1]}/{n})",
            "description": (
                f"Research Dynamics Manifold maps {n} articles into 8 archetypes "
                f"via Emergence x Impact x Connectivity axes. Distribution: {dist_desc}"
            ),
            "evidence": {
                "n_articles": n,
                "archetype_distribution": dict(arch_counts),
                "dominant_archetype": dominant_arch[0],
                "minority_archetypes": minority_archs,
            },
            "tags": ["rdm", "archetype", "dynamics", "emergence", "impact"],
        },
        phase_id="6", finding_type="rdm_archetype", category="trend",
    ))

    shifters = [r for r in data if r["archetype"] == "Paradigm Shifters"]
    if shifters:
        for r in sorted(shifters, key=lambda r: r["RDM"], reverse=True)[:5]:
            findings.append(normalize_finding(
                {
                    "title": f"Paradigm Shifter: {r['title'][:60]} (RDM={r['RDM']:.3f})",
                    "description": (
                        f"High emergence (E={r['E']:.2f}), high impact (I={r['I']:.2f}), "
                        f"high connectivity (S={r['S']:.2f}). Year {r['year']}, {r['cited_by_count']} citations."
                    ),
                    "evidence": r,
                    "tags": ["paradigm shifter", "high impact", "emergence", "connected"],
                    "related_articles": [r["key"]],
                },
                phase_id="6", finding_type="rdm_paradigm_shifter", category="emergence",
            ))

    bridges = [r for r in data if r["archetype"] == "Emerging Bridges"]
    if bridges:
        for r in sorted(bridges, key=lambda r: r["RDM"], reverse=True)[:3]:
            findings.append(normalize_finding(
                {
                    "title": f"Emerging Bridge: {r['title'][:60]}",
                    "description": (
                        f"Novel and well-connected article: E={r['E']:.2f}, S={r['S']:.2f}, I={r['I']:.2f}."
                    ),
                    "evidence": r,
                    "tags": ["emerging", "bridge", "potential", "connected"],
                    "related_articles": [r["key"]],
                },
                phase_id="6", finding_type="rdm_archetype", category="emergence",
            ))

    seeds = [r for r in data if r["archetype"] == "Isolated Seeds"]
    if len(seeds) >= 3:
        findings.append(normalize_finding(
            {
                "title": f"RDM: {len(seeds)} Isolated Seeds — niche innovations not yet integrated",
                "description": (
                    f"{len(seeds)} articles show high emergence but low impact and connectivity."
                ),
                "evidence": {
                    "count": len(seeds),
                    "sample_titles": [r["title"][:80] for r in seeds[:5]],
                    "mean_E": round(sum(r["E"] for r in seeds) / len(seeds), 3),
                },
                "tags": ["isolated", "niche", "innovation", "underexplored"],
                "related_articles": [r["key"] for r in seeds[:10]],
            },
            phase_id="6", finding_type="rdm_archetype", category="gap",
        ))

    connectors = [r for r in data if r["archetype"] == "Silent Connectors"]
    if connectors:
        for r in sorted(connectors, key=lambda r: r["S"], reverse=True)[:3]:
            findings.append(normalize_finding(
                {
                    "title": f"Silent Connector: {r['title'][:60]}",
                    "description": (
                        f"Structurally important (S={r['S']:.2f}) with I={r['I']:.2f}, E={r['E']:.2f}."
                    ),
                    "evidence": r,
                    "tags": ["bridge", "methodology", "structural"],
                    "related_articles": [r["key"]],
                },
                phase_id="6", finding_type="rdm_archetype", category="bridge",
            ))

    axis_counts = Counter(r["dominant_axis"] for r in data)
    dominant_axis = axis_counts.most_common(1)[0] if axis_counts else ("E", 0)
    axis_names = {"E": "Emergence", "I": "Impact", "S": "Connectivity"}
    findings.append(normalize_finding(
        {
            "title": (
                f"RDM axis dominance: {axis_names.get(dominant_axis[0], dominant_axis[0])} "
                f"({dominant_axis[1]}/{n}, {dominant_axis[1]/n*100:.0f}%)"
            ),
            "description": (
                f"Axis distribution: E={axis_counts.get('E', 0)}, "
                f"I={axis_counts.get('I', 0)}, S={axis_counts.get('S', 0)}."
            ),
            "evidence": dict(axis_counts),
            "tags": ["rdm", "dynamics", "axis", dominant_axis[0].lower()],
        },
        phase_id="6", finding_type="rdm_axis_pattern", category="trend",
    ))

    top5 = data[:5]
    if top5:
        findings.append(normalize_finding(
            {
                "title": f"Top RDM articles: {top5[0]['title'][:50]}... (RDM={top5[0]['RDM']:.3f})",
                "description": (
                    "Top 5 articles by RDM score: "
                    + "; ".join(f"{r['title'][:40]}... ({r['RDM']:.3f})" for r in top5)
                ),
                "evidence": {"top_5": top5},
                "tags": ["rdm", "ranking", "excellence", "multidimensional"],
                "related_articles": [r["key"] for r in top5],
            },
            phase_id="6", finding_type="rdm_axis_pattern", category="trend",
        ))

    return findings


# ===================================================================
# Core Algorithmic Functions
# ===================================================================

def _assess_phase_input(
    phase_id: str,
    n_findings: Optional[int],
    error: Optional[str],
    phase_status: Optional[dict],
) -> dict:
    """Clasifica el estado del insumo de un colector y de donde provienen sus datos."""
    run_name, input_path = _COLLECTOR_SOURCES.get(phase_id, (None, None))
    run_info = (phase_status or {}).get(run_name) if run_name else None
    run_state = run_info.get("status") if isinstance(run_info, dict) else None

    if input_path is None:
        origin = "recalculado en la Fase 7 desde el corpus"
    elif not input_path.exists():
        origin = "-"
    elif run_state == "ok":
        origin = "esta corrida"
    else:
        mtime = datetime.fromtimestamp(input_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        origin = f"corrida anterior (archivo del {mtime})"

    if run_state == "failed":
        status = "fallida"
    elif input_path is not None and not input_path.exists():
        status = "sin_insumo"
    elif error:
        status = "error_lectura"
    elif not n_findings:
        status = "sin_hallazgos"
    else:
        status = "ok"

    detail = ""
    if status == "fallida":
        detail = str(run_info.get("detail", ""))
    elif status == "error_lectura":
        detail = error or ""
    return {
        "phase": phase_id,
        "run_phase": run_name,
        "input": input_path.name if input_path is not None else None,
        "status": status,
        "origin": origin,
        "n_findings": n_findings or 0,
        "detail": detail,
    }


def collect_all_findings(
    articles: List[dict],
    phase_status: Optional[dict] = None,
    inputs_report: Optional[List[dict]] = None,
) -> List[dict]:
    """Collect findings from all active bibliometric phases.

    Si se pasa `inputs_report`, se le agrega el estado del insumo de cada colector
    (ok / sin_hallazgos / fallida / sin_insumo / error_lectura) para declararlo en el reporte.
    """
    _reset_finding_counter()
    all_findings = []

    collectors = [
        ("2a", lambda: _collect_from_descriptive(articles)),
        ("2e", lambda: _collect_from_supplementary(articles)),
        ("2f", lambda: _collect_from_sota()),
        ("2g", lambda: _collect_from_spectroscopy()),
        ("2h", lambda: _collect_from_citespace(articles=articles)),
        ("2i", lambda: _collect_from_leiden()),
        ("3",  lambda: _collect_from_networks()),
        ("3b", lambda: _collect_from_thematic(articles)),
        ("5",  lambda: _collect_from_impact_evolution()),
        # Fase 6 (RDM) excluida: indice propio sin validacion externa (desactivada por defecto en main.py)
        ("meta", lambda: _collect_from_corpus_metadata(articles)),
    ]

    for phase_id, collector in collectors:
        phase_findings: List[dict] = []
        error = None
        try:
            phase_findings = collector() or []
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            logger.warning(f"  Phase {phase_id}: collection failed — {error}")

        status = _assess_phase_input(phase_id, len(phase_findings), error, phase_status)
        if status["status"] == "fallida":
            # La fase fallo en esta corrida: lo que haya en disco es de una corrida anterior.
            status["n_findings"] = 0
            phase_findings = []
        if phase_findings:
            logger.info(f"  Phase {phase_id}: {len(phase_findings)} findings collected")
            all_findings.extend(phase_findings)
        if status["status"] in ("fallida", "sin_insumo", "error_lectura"):
            logger.warning(
                f"  Phase {phase_id}: {_STATUS_LABEL[status['status']]}"
                f"{' — ' + status['detail'] if status['detail'] else ''}. "
                "Sus hallazgos NO entran al Knowledge Map; el reporte lo declara."
            )
        if inputs_report is not None:
            inputs_report.append(status)

    logger.info(f"Total findings collected: {len(all_findings)}")
    return all_findings


DEFAULT_IMPACT_WEIGHTS = {
    "significance": 0.4, "recency": 0.2, "triangulation": 0.4,
}

_CATEGORY_WEIGHT = {
    "emergence": 0.9, "gap": 0.85, "anomaly": 0.8, "bridge": 0.75,
    "sleeping_beauty": 0.9, "convergence": 0.95, "decline": 0.7, "trend": 0.6,
}


def _normalize_impact_weights(weights: Optional[dict]) -> dict:
    keys = ("significance", "recency", "triangulation")
    if not isinstance(weights, dict):
        return dict(DEFAULT_IMPACT_WEIGHTS)
    try:
        vals = {k: max(0.0, float(weights[k])) for k in keys}
    except (KeyError, TypeError, ValueError):
        return dict(DEFAULT_IMPACT_WEIGHTS)
    s = sum(vals.values())
    if s <= 0:
        return dict(DEFAULT_IMPACT_WEIGHTS)
    return {k: v / s for k, v in vals.items()}


def _finding_components(f: dict) -> tuple:
    evidence = f.get("evidence", {})
    ev_richness = min(len(evidence) / 5.0, 1.0)
    cat_w = _CATEGORY_WEIGHT.get(f.get("category", "trend"), 0.5)
    significance = 0.5 * ev_richness + 0.5 * cat_w

    recency = 0.5
    ev_year = (
        evidence.get("year", 0)
        or evidence.get("peak_year", 0)
        or evidence.get("last_year", 0)
    )
    if ev_year and isinstance(ev_year, (int, float)):
        recency = min(max((ev_year - 2000) / 25.0, 0.0), 1.0)

    tri_score = f.get("triangulation_score", 0) / 5.0
    return significance, recency, tri_score


def compute_impact_scores(
    findings: List[dict],
    weights: Optional[dict] = None,
) -> List[dict]:
    if not findings:
        return findings
    w = _normalize_impact_weights(weights)
    for f in findings:
        sig, rec, tri = _finding_components(f)
        score = (
            w["significance"] * sig
            + w["recency"] * rec
            + w["triangulation"] * tri
        )
        f["impact_score"] = round(score, 4)
    return findings


def _impact_weight_sensitivity(
    findings: List[dict],
    base_weights: dict,
) -> dict:
    """Deterministically evaluate ranking stability across weight perturbations."""
    if not findings:
        return {"note": "no findings", "base_weights": base_weights}
    comps = {f["id"]: _finding_components(f) for f in findings}
    ids = list(comps.keys())

    def _order(w: dict):
        return sorted(
            ids,
            key=lambda i: -(
                w["significance"] * comps[i][0]
                + w["recency"] * comps[i][1]
                + w["triangulation"] * comps[i][2]
            ),
        )

    base_order = _order(base_weights)
    base_rank = {i: r for r, i in enumerate(base_order)}
    topn = min(20, len(ids))
    base_top = set(base_order[:topn])

    grid = {
        "equal": {"significance": 1, "recency": 1, "triangulation": 1},
        "significance_heavy": {"significance": 0.6, "recency": 0.2, "triangulation": 0.2},
        "recency_heavy": {"significance": 0.2, "recency": 0.6, "triangulation": 0.2},
        "triangulation_heavy": {"significance": 0.2, "recency": 0.2, "triangulation": 0.6},
    }
    n = len(ids)
    perts = []
    for name, raw in grid.items():
        w = _normalize_impact_weights(raw)
        order = _order(w)
        rank = {i: r for r, i in enumerate(order)}
        d2 = sum((base_rank[i] - rank[i]) ** 2 for i in ids)
        rho = 1.0 if n < 2 else 1.0 - (6.0 * d2) / (n * (n * n - 1))
        top = set(order[:topn])
        jac = len(base_top & top) / max(1, len(base_top | top))
        perts.append({
            "name": name,
            "weights": {k: round(v, 4) for k, v in w.items()},
            "spearman_vs_base": round(rho, 4),
            f"jaccard_top{topn}": round(jac, 4),
        })
    min_rho = min(p["spearman_vs_base"] for p in perts)
    interp = (
        "robusto: el ranking es estable ante reponderacion"
        if min_rho >= 0.8 else
        "moderado: el top depende de los pesos; reportar la sensibilidad"
        if min_rho >= 0.5 else
        "fragil: el ranking es muy sensible a los pesos; justificar la eleccion"
    )
    return {
        "base_weights": {k: round(v, 4) for k, v in base_weights.items()},
        "n_findings": n,
        "top_n": topn,
        "perturbations": perts,
        "min_spearman": round(min_rho, 4),
        "interpretation": interp,
    }


def detect_triangulation(
    findings: List[dict],
    weights: Optional[dict] = None,
) -> Tuple[List[dict], dict]:
    """Cross-reference findings across independent bibliometric phases."""
    try:
        from rapidfuzz import fuzz
    except ImportError:
        fuzz = None

    n = len(findings)
    matrix = {}

    for i in range(n):
        for j in range(i + 1, n):
            fa, fb = findings[i], findings[j]
            if fa["source_phase"] == fb["source_phase"]:
                continue

            score = 0.0
            tags_a = set(fa.get("tags", []))
            tags_b = set(fb.get("tags", []))
            if tags_a and tags_b:
                exact_overlap = tags_a & tags_b
                if exact_overlap:
                    score += len(exact_overlap) * 0.4
                if fuzz:
                    for ta in tags_a - exact_overlap:
                        for tb in tags_b - exact_overlap:
                            if fuzz.ratio(ta, tb) > 80:
                                score += 0.3

            arts_a = set(fa.get("related_articles", []))
            arts_b = set(fb.get("related_articles", []))
            if arts_a and arts_b:
                shared = arts_a & arts_b
                if shared:
                    score += len(shared) * 0.5

            if fa["category"] == fb["category"] and score > 0:
                score += 0.2

            if score > 0.3:
                matrix[(fa["id"], fb["id"])] = round(score, 3)

    tri_counts = Counter()
    tri_sources = {}
    for (id_a, id_b), sim in matrix.items():
        if sim > 0.3:
            tri_counts[id_a] += 1
            tri_counts[id_b] += 1
            tri_sources.setdefault(id_a, []).append(id_b)
            tri_sources.setdefault(id_b, []).append(id_a)

    id_to_finding = {f["id"]: f for f in findings}
    for f in findings:
        fid = f["id"]
        f["triangulation_score"] = tri_counts.get(fid, 0)
        src_ids = tri_sources.get(fid, [])
        f["triangulation_sources"] = list(set(
            f"{id_to_finding[sid]['source_phase']}:{id_to_finding[sid]['type']}"
            for sid in src_ids if sid in id_to_finding
        ))[:10]
        f["connections"] = src_ids[:20]

    findings = compute_impact_scores(findings, weights)
    return findings, matrix


def build_results_skeleton(findings: List[dict]) -> dict:
    """Group findings into standard bibliometric result sections."""
    skeleton = {sec_id: {"title": title, "findings": []} for sec_id, title in RESULTS_SECTIONS.items()}

    type_to_section = {
        "corpus_profile": 1, "cagr_trend": 1, "halflife_indicator": 1,
        "sdg_alignment": 1, "funding_pattern": 1, "language_pattern": 1,
        "metadata_completeness": 1,
        "spectroscopy_seminal": 2, "spectroscopy_lifecycle": 2,
        "leiden_outlier": 2, "leiden_field_bias": 2,
        "network_community": 3, "bridge_article": 3,
        "structural_hole": 3, "scp_mcp_pattern": 3,
        "thematic_quadrant": 4, "burst_keyword": 4,
        "pivot_node": 4, "sigma_node": 4,
        "research_frontier": 5, "concept_evolution": 5,
        "structural_gap": 6, "declared_gap": 6, "community_gap": 6,
        "fwci_outlier": 7, "sleeping_beauty": 7,
        "css_classification": 7, "citation_class": 7, "apc_pattern": 7,
        
    }

    type_to_secondary = {
        "burst_keyword": 5,
        "pivot_node": 5,
        "sigma_node": 5,
        "spectroscopy_lifecycle": 5,
    }

    for f in sorted(findings, key=lambda x: x.get("impact_score", 0), reverse=True):
        sec = type_to_section.get(f["type"], 8)
        skeleton[sec]["findings"].append(f["id"])

    for f in sorted(findings, key=lambda x: x.get("impact_score", 0), reverse=True):
        sec2 = type_to_secondary.get(f["type"])
        if sec2 and f["id"] not in skeleton[sec2]["findings"]:
            skeleton[sec2]["findings"].append(f["id"])

    for f in findings:
        if f.get("triangulation_score", 0) >= 2 and f["id"] not in skeleton[8]["findings"]:
            skeleton[8]["findings"].append(f["id"])

    return skeleton


def synthesize_algorithmic_themes(findings: List[dict]) -> dict:
    """Deterministically group corroborated findings into cross-cutting themes by tag/phase convergence."""
    corroborated = [f for f in findings if f.get("triangulation_score", 0) >= 1]
    if not corroborated:
        corroborated = sorted(findings, key=lambda f: f.get("impact_score", 0), reverse=True)[:15]

    tag_to_findings: Dict[str, List[dict]] = defaultdict(list)
    generic_tags = {"trend", "impact", "network", "citation", "citations", "corpus profile", "temporal"}
    for f in corroborated:
        for tag in f.get("tags", []):
            if tag and tag not in generic_tags and len(tag) > 2:
                tag_to_findings[tag].append(f)

    themes = []
    used_fids: set[str] = set()
    for tag, f_list in sorted(
        tag_to_findings.items(),
        key=lambda kv: (len({x["source_phase"] for x in kv[1]}), len(kv[1])),
        reverse=True,
    ):
        phases = sorted({x["source_phase"] for x in f_list})
        if len(phases) < 2 and len(themes) >= 3:
            continue
        fids = [x["id"] for x in f_list]
        if all(fid in used_fids for fid in fids):
            continue
        used_fids.update(fids)
        strength = "strong" if len(phases) >= 3 else ("moderate" if len(phases) == 2 else "suggestive")
        sample_titles = "; ".join(x["title"][:60] for x in f_list[:3])
        themes.append({
            "name": f"Convergence on '{tag.title()}'",
            "theme": f"Convergence on '{tag.title()}'",
            "evidence_strength": strength,
            "phases": phases,
            "finding_ids": fids[:10],
            "rationale": f"Corroborated across phases {', '.join(phases)}: {sample_titles}",
        })
        if len(themes) >= 8:
            break

    n_tri = sum(1 for f in findings if f.get("triangulation_score", 0) >= 2)
    return {
        "method": "deterministic_cross_phase_clustering",
        "cross_cutting_themes": themes,
        "meta_observation": (
            f"Algorithmic synthesis identified {len(themes)} cross-cutting patterns "
            f"across {len({f['source_phase'] for f in findings})} analytical phases, "
            f"with {n_tri} findings corroborated by 2 or more independent analyses."
        ),
    }


def generate_knowledge_map_report(
    findings: List[dict],
    skeleton: dict,
    themes: dict,
    stats: dict,
) -> Path:
    """Generate deterministic Markdown report summarizing all sections and 95% CIs."""
    KNOWLEDGE_MAP_DIR.mkdir(parents=True, exist_ok=True)
    report_path = KNOWLEDGE_MAP_DIR / "knowledge_map_report.md"
    f_by_id = {f["id"]: f for f in findings}
    boot = _load_bootstrap_ci()

    lines = [
        "# Knowledge Map — Deterministic Bibliometric Synthesis Report",
        "",
        f"- **Total Findings Collected**: {stats.get('total_findings', len(findings))}",
        f"- **Corroborated Findings (≥2 Phases)**: {stats.get('triangulated_findings', 0)}",
        f"- **Cross-Phase Links**: {stats.get('cross_phase_links', 0)}",
        f"- **Active Phases**: {', '.join(sorted(stats.get('by_phase', {}).keys()))}",
        "",
    ]

    phase_inputs = stats.get("phase_inputs", [])
    status_by_phase = {p["phase"]: p for p in phase_inputs}
    unavailable = [p for p in phase_inputs if p["status"] in ("fallida", "sin_insumo", "error_lectura")]
    if unavailable:
        lines.append(
            f"> **ATENCIÓN: {len(unavailable)} fase(s) sin insumo válido** "
            f"({', '.join(p['phase'] for p in unavailable)}). Sus hallazgos no forman parte de "
            "esta síntesis; las secciones que dependen de ellas están incompletas."
        )
        lines.append("")
    if phase_inputs:
        lines.append("## Estado de los insumos por fase")
        lines.append("")
        lines.append("| Fase | Insumo | Estado | Origen de los datos | Hallazgos | Detalle |")
        lines.append("|------|--------|--------|---------------------|-----------|---------|")
        for p in phase_inputs:
            detail = str(p.get("detail", "")).replace("|", "/")[:160]
            lines.append(
                f"| {p['phase']} | {p.get('input') or '—'} | {_STATUS_LABEL.get(p['status'], p['status'])} | "
                f"{p.get('origin', '')} | {p.get('n_findings', 0)} | {detail} |"
            )
        lines.append("")

    if boot:
        lines.append("## Headline Indicators with 95% Bootstrap Confidence Intervals")
        lines.append("")
        lines.append("| Indicator | Point Estimate | 95% CI | Effective N |")
        lines.append("|-----------|----------------|--------|-------------|")
        mcp = boot.get("mcp_rate", {})
        if mcp.get("effective_n", 0) > 0:
            lines.append(
                f"| Multi-Country Publications (MCP) | {mcp['point']*100:.2f}% | "
                f"[{mcp['ci_lo']*100:.2f}%, {mcp['ci_hi']*100:.2f}%] | {mcp['effective_n']} |"
            )
        pi = boot.get("price_index", {})
        if pi.get("effective_n", 0) > 0:
            lines.append(
                f"| Mean Article Price Index (≤5y refs) | {pi['point']*100:.2f}% | "
                f"[{pi['ci_lo']*100:.2f}%, {pi['ci_hi']*100:.2f}%] | {pi['effective_n']} |"
            )
        cites = boot.get("citations", {})
        if cites.get("effective_n", 0) > 0:
            cm = cites["citations_mean"]
            lines.append(
                f"| Mean Citations per Article | {cm['point']:.2f} | "
                f"[{cm['ci_lo']:.2f}, {cm['ci_hi']:.2f}] | {cites['effective_n']} |"
            )
        fw = boot.get("fwci", {})
        if fw.get("FWCI_mean"):
            fm = fw["FWCI_mean"]
            lines.append(
                f"| Field-Weighted Citation Impact (FWCI) | {fm['point']:.3f} | "
                f"[{fm['ci_lo']:.3f}, {fm['ci_hi']:.3f}] | {fw.get('effective_n', 0)} |"
            )
        mncs = boot.get("mncs_top10", {})
        if mncs.get("MNCS_mean"):
            mm = mncs["MNCS_mean"]
            lines.append(
                f"| Leiden MNCS (Field-Normalized) | {mm['point']:.3f} | "
                f"[{mm['ci_lo']:.3f}, {mm['ci_hi']:.3f}] | {mm.get('effective_n', 0)} |"
            )
        if mncs.get("PP_top10_field"):
            pp = mncs["PP_top10_field"]
            lines.append(
                f"| PP(top 10% of Field) | {pp['point']*100:.2f}% | "
                f"[{pp['ci_lo']*100:.2f}%, {pp['ci_hi']*100:.2f}%] | {pp.get('effective_n', 0)} |"
            )
        lines.append("")

    if themes and themes.get("cross_cutting_themes"):
        lines.append("## Cross-Cutting Corroborated Patterns")
        lines.append("")
        for th in themes["cross_cutting_themes"]:
            lines.append(
                f"- **{th.get('name', '')}** (`{th.get('evidence_strength', '')}`, "
                f"phases: {', '.join(th.get('phases', []))}): {th.get('rationale', '')}"
            )
        lines.append("")

    lines.append("## Section-by-Section Bibliometric Findings")
    lines.append("")
    for sec_id in sorted(RESULTS_SECTIONS.keys()):
        sec_info = skeleton.get(sec_id, {"title": RESULTS_SECTIONS[sec_id], "findings": []})
        sec_fids = sec_info.get("findings", [])
        sec_findings = [f_by_id[fid] for fid in sec_fids if fid in f_by_id]
        lines.append(f"### Section {sec_id}: {RESULTS_SECTIONS[sec_id]} ({len(sec_findings)} findings)")
        lines.append("")
        missing = [
            ph for ph in _SECTION_SOURCE_PHASES.get(sec_id, [])
            if status_by_phase.get(ph, {}).get("status") in ("fallida", "sin_insumo", "error_lectura")
        ]
        if sec_findings and missing:
            lines.append(
                f"_**Sección incompleta: falta el aporte de la(s) fase(s) {', '.join(missing)}.**_"
            )
            lines.append("")
        if not sec_findings:
            if missing:
                lines.append(
                    f"_**Sección sin datos: la(s) fase(s) {', '.join(missing)} no aportaron insumo "
                    f"válido en esta corrida.** La ausencia de hallazgos aquí NO significa que no los haya._"
                )
            else:
                lines.append("_No findings recorded for this section._")
            lines.append("")
            continue
        lines.append("| ID | Phase | Category | Impact | Corroborations | Finding | Details |")
        lines.append("|----|-------|----------|--------|----------------|---------|---------|")
        for f in sec_findings[:15]:
            title_clean = str(f.get("title", "")).replace("|", "/")
            desc_clean = str(f.get("description", ""))[:120].replace("|", "/")
            lines.append(
                f"| {f['id']} | {f['source_phase']} | {f['category']} | "
                f"{f.get('impact_score', 0):.3f} | {f.get('triangulation_score', 0)} | "
                f"**{title_clean}** | {desc_clean} |"
            )
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Knowledge Map report saved: {report_path}")
    return report_path


def generate_overview_html(
    findings: List[dict],
    matrix: dict,
    themes: Optional[dict] = None,
) -> Optional[Path]:
    """Generate interactive Plotly dashboard with 11 sections."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        logger.warning("Plotly not available — skipping HTML overview")
        return None

    KNOWLEDGE_MAP_DIR.mkdir(parents=True, exist_ok=True)

    CATEGORY_COLORS = {
        "trend": "#3498db", "gap": "#e74c3c", "anomaly": "#f39c12",
        "bridge": "#2ecc71", "sleeping_beauty": "#9b59b6",
        "convergence": "#1abc9c", "decline": "#95a5a6", "emergence": "#e67e22",
    }

    type_to_section = {
        "corpus_profile": 1, "cagr_trend": 1, "halflife_indicator": 1,
        "sdg_alignment": 1, "funding_pattern": 1, "language_pattern": 1,
        "metadata_completeness": 1,
        "spectroscopy_seminal": 2, "spectroscopy_lifecycle": 2,
        "leiden_outlier": 2, "leiden_field_bias": 2,
        "network_community": 3, "bridge_article": 3,
        "structural_hole": 3, "scp_mcp_pattern": 3,
        "thematic_quadrant": 4, "burst_keyword": 4, "pivot_node": 4, "sigma_node": 4,
        "research_frontier": 5, "concept_evolution": 5,
        "structural_gap": 6, "declared_gap": 6, "community_gap": 6,
        "fwci_outlier": 7, "sleeping_beauty": 7, "css_classification": 7,
        "citation_class": 7, "apc_pattern": 7,
        
    }

    def _sec_for(f):
        return type_to_section.get(f.get("type", ""), 8)

    type_to_secondary = {
        "burst_keyword": 5,
        "pivot_node": 5,
        "sigma_node": 5,
        "spectroscopy_lifecycle": 5,
    }

    skeleton = {sid: {"title": t, "findings": []} for sid, t in RESULTS_SECTIONS.items()}
    for f in sorted(findings, key=lambda x: x.get("impact_score", 0), reverse=True):
        skeleton[_sec_for(f)]["findings"].append(f)

    sec_ids_per_section = {sid: {ff["id"] for ff in sdata["findings"]} for sid, sdata in skeleton.items()}
    for f in sorted(findings, key=lambda x: x.get("impact_score", 0), reverse=True):
        sec2 = type_to_secondary.get(f.get("type"))
        if sec2 and f["id"] not in sec_ids_per_section.get(sec2, set()):
            skeleton[sec2]["findings"].append(f)
            sec_ids_per_section.setdefault(sec2, set()).add(f["id"])

    sec8_ids = sec_ids_per_section.get(8, set())
    for f in findings:
        if f.get("triangulation_score", 0) >= 2 and f["id"] not in sec8_ids:
            skeleton[8]["findings"].append(f)
            sec8_ids.add(f["id"])

    f_by_id = {f["id"]: f for f in findings}
    phases = sorted(set(f["source_phase"] for f in findings)) if findings else []
    categories_present = sorted(set(f.get("category", "trend") for f in findings)) if findings else []
    n_findings = len(findings)

    n_tri = sum(1 for f in findings if f.get("triangulation_score", 0) >= 2)
    n_phases = len(phases)
    n_crosslinks = len(matrix)
    impact_scores = [f.get("impact_score", 0) for f in findings]
    avg_impact = sum(impact_scores) / max(n_findings, 1)
    n_high_impact = sum(1 for s in impact_scores if s > 0.5)
    cat_counter = Counter(f.get("category", "trend") for f in findings)
    top_cat, top_cat_n = cat_counter.most_common(1)[0] if cat_counter else ("—", 0)
    avg_per_section = n_findings / 9.0

    def _cat_color(cat):
        return CATEGORY_COLORS.get(cat, "#3498db")

    # Figure 1: Impact Score Distribution
    fig_dist_html = ""
    try:
        fig_dist = make_subplots(
            rows=2, cols=1, row_heights=[0.3, 0.7],
            shared_xaxes=True, vertical_spacing=0.05,
        )
        fig_dist.add_trace(
            go.Box(x=impact_scores, name="Impact", marker_color="#3498db", boxmean=True),
            row=1, col=1,
        )
        for cat in categories_present:
            scores = [f.get("impact_score", 0) for f in findings if f.get("category") == cat]
            if scores:
                fig_dist.add_trace(go.Histogram(
                    x=scores, name=cat, marker_color=_cat_color(cat),
                    xbins=dict(start=0, end=1.05, size=0.05), opacity=0.7,
                ), row=2, col=1)
        fig_dist.update_layout(
            barmode="stack", height=400, title="Impact Score Distribution",
            xaxis2_title="Impact Score", yaxis2_title="Count",
            margin=dict(t=40, b=40),
        )
        fig_dist_html = json.dumps(fig_dist.to_dict(), default=str)
    except Exception as e:
        logger.warning(f"Impact distribution chart failed: {e}")

    # Figure 2: Sunburst
    fig_sun_html = ""
    try:
        sun_ids, sun_labels, sun_parents, sun_values, sun_colors = [], [], [], [], []
        sun_ids.append("root")
        sun_labels.append("All Findings")
        sun_parents.append("")
        sun_values.append(0)
        sun_colors.append("#ecf0f1")

        for sid, sdata in RESULTS_SECTIONS.items():
            sec_node = f"S{sid}"
            sun_ids.append(sec_node)
            sun_labels.append(f"S{sid}: {sdata[:25]}")
            sun_parents.append("root")
            sun_values.append(0)
            sun_colors.append("#bdc3c7")

            cats_in_sec = {f.get("category", "trend") for f in skeleton[sid]["findings"]}
            for cat in sorted(cats_in_sec):
                cat_node = f"S{sid}_{cat}"
                sun_ids.append(cat_node)
                sun_labels.append(cat)
                sun_parents.append(sec_node)
                sun_values.append(0)
                sun_colors.append(_cat_color(cat))

        leaf_findings = sorted(findings, key=lambda x: x.get("impact_score", 0), reverse=True)[:50]
        for f in leaf_findings:
            sec = _sec_for(f)
            cat = f.get("category", "trend")
            parent_node = f"S{sec}_{cat}"
            if parent_node not in sun_ids:
                sun_ids.append(parent_node)
                sun_labels.append(cat)
                sun_parents.append(f"S{sec}")
                sun_values.append(0)
                sun_colors.append(_cat_color(cat))
            sun_ids.append(f["id"])
            sun_labels.append(f"{f['id']}: {f.get('title', '')[:30]}")
            sun_parents.append(parent_node)
            sun_values.append(max(int(f.get("impact_score", 0) * 100), 1))
            sun_colors.append(_cat_color(cat))

        fig_sun = go.Figure(go.Sunburst(
            ids=sun_ids, labels=sun_labels, parents=sun_parents,
            values=sun_values, branchvalues="total",
            marker=dict(colors=sun_colors),
            hovertemplate="<b>%{label}</b><br>Value: %{value}<extra></extra>",
        ))
        fig_sun.update_layout(
            height=600, title="Findings Hierarchy: Section → Category → Finding",
            margin=dict(t=40, b=10),
        )
        fig_sun_html = json.dumps(fig_sun.to_dict(), default=str)
    except Exception as e:
        logger.warning(f"Sunburst chart failed: {e}")

    # Figure 3: Cross-Phase Corroboration Heatmap
    fig_tri_html = ""
    try:
        if phases and matrix:
            phase_idx = {p: i for i, p in enumerate(phases)}
            n_p = len(phases)
            tri_z = [[0.0] * n_p for _ in range(n_p)]
            tri_text = [[""] * n_p for _ in range(n_p)]
            for f in findings:
                pi = phase_idx.get(f["source_phase"])
                if pi is not None:
                    tri_z[pi][pi] += 1
                    tri_text[pi][pi] = f"{int(tri_z[pi][pi])} findings"
            for key_str, sim in matrix.items():
                parts = key_str.split("-")
                if len(parts) == 2:
                    fa, fb = f_by_id.get(parts[0]), f_by_id.get(parts[1])
                    if fa and fb:
                        pa = phase_idx.get(fa["source_phase"])
                        pb = phase_idx.get(fb["source_phase"])
                        if pa is not None and pb is not None and pa != pb:
                            sim_val = sim if isinstance(sim, (int, float)) else 0
                            tri_z[pa][pb] += sim_val
                            tri_z[pb][pa] += sim_val
                            pair = f"{parts[0]}-{parts[1]}"
                            tri_text[pa][pb] = (tri_text[pa][pb] + " " + pair).strip()
                            tri_text[pb][pa] = tri_text[pa][pb]
            fig_tri = go.Figure(go.Heatmap(
                z=tri_z, x=phases, y=phases, colorscale="Teal",
                text=tri_text,
                hovertemplate="Phase %{x} ↔ %{y}<br>Score: %{z:.2f}<br>%{text}<extra></extra>",
            ))
            fig_tri.update_layout(
                title="Cross-Phase Corroboration: Phase × Phase Similarity",
                height=450, xaxis_title="Phase", yaxis_title="Phase",
            )
            fig_tri_html = json.dumps(fig_tri.to_dict(), default=str)
    except Exception as e:
        logger.warning(f"Corroboration matrix chart failed: {e}")

    # Figure 4: Phase -> Section Sankey
    fig_sankey_html = ""
    try:
        if findings:
            sankey_phases = sorted(set(f["source_phase"] for f in findings))
            sankey_sections = list(RESULTS_SECTIONS.keys())
            phase_colors = [
                "#2c3e50", "#2980b9", "#27ae60", "#f39c12",
                "#e74c3c", "#9b59b6", "#1abc9c", "#e67e22",
                "#34495e", "#16a085", "#c0392b", "#8e44ad",
            ]
            sec_colors = [
                "#3498db", "#2ecc71", "#e74c3c", "#f39c12",
                "#9b59b6", "#1abc9c", "#e67e22", "#95a5a6", "#34495e",
            ]
            node_labels = (
                [f"Phase {p}" for p in sankey_phases]
                + [f"S{s}: {RESULTS_SECTIONS[s][:20]}" for s in sankey_sections]
            )
            node_colors = (
                [phase_colors[i % len(phase_colors)] for i in range(len(sankey_phases))]
                + [sec_colors[i % len(sec_colors)] for i in range(len(sankey_sections))]
            )
            n_sp = len(sankey_phases)
            flow = Counter()
            for f in findings:
                pi = sankey_phases.index(f["source_phase"]) if f["source_phase"] in sankey_phases else 0
                si = _sec_for(f)
                si_idx = sankey_sections.index(si) if si in sankey_sections else 7
                flow[(pi, n_sp + si_idx)] += 1
            sources, targets, values = [], [], []
            for (s, t), v in flow.items():
                sources.append(s)
                targets.append(t)
                values.append(v)
            fig_sankey = go.Figure(go.Sankey(
                node=dict(label=node_labels, color=node_colors, pad=15, thickness=20),
                link=dict(source=sources, target=targets, value=values),
            ))
            fig_sankey.update_layout(
                title="Flow: Source Phase → Results Section",
                height=500, margin=dict(t=40, b=20),
            )
            fig_sankey_html = json.dumps(fig_sankey.to_dict(), default=str)
    except Exception as e:
        logger.warning(f"Sankey chart failed: {e}")

    # Figure 5: Top-30 findings bar chart
    fig_bar_html = ""
    try:
        top_n = sorted(findings, key=lambda f: f.get("impact_score", 0), reverse=True)[:30]
        fig_bar = go.Figure(go.Bar(
            x=[f["impact_score"] for f in top_n],
            y=[f"{f['id']}: {f.get('title', '')[:50]}" for f in top_n],
            orientation="h",
            marker_color=[_cat_color(f.get("category", "trend")) for f in top_n],
            text=[f"Phase {f['source_phase']} | Corr={f.get('triangulation_score', 0)}" for f in top_n],
            textposition="inside",
            hovertext=[
                f"{f['id']}: {f.get('title', '')[:80]}<br>{f.get('description', '')[:120]}"
                for f in top_n
            ],
            hoverinfo="text",
        ))
        fig_bar.update_layout(
            title="Top 30 Findings by Impact Score",
            xaxis_title="Impact Score",
            yaxis=dict(autorange="reversed"),
            height=700, margin=dict(l=350),
        )
        fig_bar_html = json.dumps(fig_bar.to_dict(), default=str)
    except Exception as e:
        logger.warning(f"Bar chart failed: {e}")

    # Figure 6: Phase x Category heatmap
    fig_heat_html = ""
    try:
        if findings:
            heatmap_data = []
            for cat in categories_present:
                row = [
                    sum(1 for f in findings if f["source_phase"] == p and f.get("category") == cat)
                    for p in phases
                ]
                heatmap_data.append(row)
            fig_heat = go.Figure(go.Heatmap(
                z=heatmap_data, x=phases, y=categories_present,
                colorscale="YlOrRd", text=heatmap_data, texttemplate="%{text}",
            ))
            fig_heat.update_layout(
                title="Findings Distribution: Phase × Category",
                xaxis_title="Source Phase", yaxis_title="Category", height=400,
            )
            fig_heat_html = json.dumps(fig_heat.to_dict(), default=str)
    except Exception as e:
        logger.warning(f"Heatmap failed: {e}")

    # Figure 7: Network graph
    fig_net_html = ""
    try:
        top_net = sorted(findings, key=lambda f: f.get("impact_score", 0), reverse=True)[:40]
        n_nodes = len(top_net)
        node_positions = {}
        for i, f in enumerate(top_net):
            angle = 2 * math.pi * i / max(n_nodes, 1)
            node_positions[f["id"]] = (math.cos(angle), math.sin(angle))

        fig_net = go.Figure()
        for key_str, sim in matrix.items():
            parts = key_str.split("-")
            if len(parts) == 2:
                id_a, id_b = parts
                if id_a in node_positions and id_b in node_positions:
                    x0, y0 = node_positions[id_a]
                    x1, y1 = node_positions[id_b]
                    sim_val = sim if isinstance(sim, (int, float)) else 0.3
                    fig_net.add_trace(go.Scatter(
                        x=[x0, x1, None], y=[y0, y1, None],
                        mode="lines", line=dict(width=max(0.5, sim_val * 3), color="#ccc"),
                        hoverinfo="none", showlegend=False,
                    ))
        fig_net.add_trace(go.Scatter(
            x=[node_positions[f["id"]][0] for f in top_net],
            y=[node_positions[f["id"]][1] for f in top_net],
            mode="markers+text",
            marker=dict(
                size=[8 + f.get("impact_score", 0) * 22 for f in top_net],
                color=[_cat_color(f.get("category", "trend")) for f in top_net],
                line=dict(width=1, color="#fff"),
            ),
            text=[f["id"] for f in top_net],
            textposition="top center", textfont=dict(size=9),
            hovertext=[
                f"{f['id']}: {f.get('title', '')[:60]}<br>Phase {f['source_phase']}, "
                f"Cat={f.get('category', '')}, Impact={f.get('impact_score', 0):.2f}, "
                f"Corr={f.get('triangulation_score', 0)}"
                for f in top_net
            ],
            hoverinfo="text", showlegend=False,
        ))
        fig_net.update_layout(
            title="Knowledge Map: Cross-Phase Finding Connections (top 40)",
            showlegend=False, height=600,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )
        fig_net_html = json.dumps(fig_net.to_dict(), default=str)
    except Exception as e:
        logger.warning(f"Network chart failed: {e}")

    # Section tabs HTML
    tabs_buttons = ""
    tabs_panels = ""
    for sid, sdata in RESULTS_SECTIONS.items():
        sec_findings = skeleton[sid]["findings"]
        n_sf = len(sec_findings)
        active = "active" if sid == 1 else ""
        display = "block" if sid == 1 else "none"
        tabs_buttons += (
            f'<button class="tab-btn {active}" onclick="showSection(\'sec{sid}\')">'
            f'S{sid} <span class="badge" style="background:#7f8c8d">{n_sf}</span></button>\n'
        )
        rows_html = ""
        for f in sec_findings:
            imp = f.get("impact_score", 0)
            cat = f.get("category", "trend")
            tri = f.get("triangulation_score", 0)
            tri_marker = ' <span class="badge" style="background:#e74c3c">CORR</span>' if tri >= 2 else ""
            desc = html_escape(f.get("description", "")[:120])
            title = html_escape(f.get("title", ""))
            tags_html = "".join(
                f'<span class="badge" style="background:#ecf0f1;color:#2c3e50">{html_escape(t)}</span>'
                for t in f.get("tags", [])[:5]
            )
            rows_html += f"""<div class="finding-row">
  <span class="badge" style="background:{_cat_color(cat)}">{html_escape(f['id'])}</span>
  <strong>{title}</strong>{tri_marker}
  <div class="impact-bar-container"><div class="impact-bar" style="width:{imp*100:.0f}%;background:{_cat_color(cat)}"></div></div>
  <small>Phase {html_escape(str(f.get('source_phase', '')))} | {html_escape(f.get('type', ''))} | Impact: {imp:.2f}</small><br>
  <small>{desc}</small><br>{tags_html}
</div>\n"""
        tabs_panels += f'<div class="section-content" id="sec{sid}" style="display:{display}">\n'
        tabs_panels += f'<h3>{html_escape(sdata)}</h3>\n'
        tabs_panels += (rows_html if rows_html else '<p class="empty">No findings in this section.</p>')
        tabs_panels += '</div>\n'

    table_filter_badges = "".join(
        f'<span class="badge cat-filter" style="background:{_cat_color(c)};cursor:pointer" '
        f'onclick="toggleCatFilter(\'{c}\')">{html_escape(c)}</span>'
        for c in categories_present
    )
    table_rows = ""
    for f in sorted(findings, key=lambda x: x.get("impact_score", 0), reverse=True):
        cat = f.get("category", "trend")
        sec = _sec_for(f)
        tags_str = ", ".join(f.get("tags", [])[:3])
        table_rows += (
            f'<tr class="frow" data-cat="{html_escape(cat)}">'
            f'<td><span class="badge" style="background:{_cat_color(cat)}">{html_escape(f["id"])}</span></td>'
            f'<td>{html_escape(f.get("title", "")[:80])}</td>'
            f'<td>{html_escape(str(f.get("source_phase", "")))}</td>'
            f'<td>S{sec}</td>'
            f'<td>{html_escape(cat)}</td>'
            f'<td>{html_escape(f.get("type", ""))}</td>'
            f'<td data-value="{f.get("impact_score", 0):.4f}">{f.get("impact_score", 0):.3f}</td>'
            f'<td data-value="{f.get("triangulation_score", 0)}">{f.get("triangulation_score", 0)}</td>'
            f'<td>{html_escape(tags_str)}</td>'
            f'</tr>\n'
        )

    themes_html = ""
    if themes and themes.get("cross_cutting_themes"):
        for th in themes["cross_cutting_themes"]:
            strength = th.get("evidence_strength", "suggestive")
            strength_colors = {"strong": "#27ae60", "moderate": "#f39c12", "suggestive": "#95a5a6"}
            s_color = strength_colors.get(strength, "#95a5a6")
            th_name = html_escape(th.get("name", "Unnamed Theme"))
            rationale = html_escape(th.get("rationale", "")[:240])
            fids = ", ".join(th.get("finding_ids", [])[:10])
            ph_list = ", ".join(str(p) for p in th.get("phases", [])[:8])
            themes_html += f"""<div class="theme-card">
  <h4>{th_name} <span class="badge" style="background:{s_color}">{html_escape(strength)}</span></h4>
  <p>{rationale}</p>
  <small>Findings: {html_escape(fids)} | Phases: {html_escape(ph_list)}</small>
</div>\n"""
        if themes.get("meta_observation"):
            themes_html += f'<div class="meta-obs"><strong>Synthesis:</strong> {html_escape(themes["meta_observation"][:300])}</div>\n'
    else:
        themes_html = '<p class="empty">No cross-cutting patterns identified.</p>'

    def _chart_div(div_id, fig_json, fallback_msg):
        if fig_json:
            return f'<div class="chart" id="{div_id}"></div>'
        return f'<div class="chart"><p class="empty">{html_escape(fallback_msg)}</p></div>'

    chart_dist = _chart_div("distplot", fig_dist_html, "Impact distribution unavailable.")
    chart_sun = _chart_div("sunburst", fig_sun_html, "Sunburst unavailable.")
    chart_tri = _chart_div("trimatrix", fig_tri_html, "No cross-phase connections.")
    chart_sankey = _chart_div("sankey", fig_sankey_html, "Sankey flow unavailable.")
    chart_bar = _chart_div("bar", fig_bar_html, "Bar chart unavailable.")
    chart_heat = _chart_div("heatmap", fig_heat_html, "Heatmap unavailable.")
    chart_net = _chart_div("network", fig_net_html, "Network graph unavailable.")

    plotly_calls = ""
    for div_id, fig_json in [
        ("distplot", fig_dist_html), ("sunburst", fig_sun_html),
        ("trimatrix", fig_tri_html), ("sankey", fig_sankey_html),
        ("bar", fig_bar_html), ("heatmap", fig_heat_html),
        ("network", fig_net_html),
    ]:
        if fig_json:
            plotly_calls += (
                f"try {{ var d_{div_id} = {fig_json}; "
                f"Plotly.newPlot('{div_id}', d_{div_id}.data, d_{div_id}.layout); }} "
                f"catch(e) {{ console.warn('{div_id} chart error', e); }}\n"
            )

    css = """
body { font-family: Arial, sans-serif; margin: 0; padding: 0 20px 40px; background: #f8f9fa; }
.topnav { position: sticky; top: 0; background: #2c3e50; color: white; padding: 8px 20px;
  display: flex; flex-wrap: wrap; gap: 6px; z-index: 999; margin: 0 -20px; }
.topnav a { color: #ecf0f1; text-decoration: none; font-size: 0.85em; padding: 4px 10px;
  border-radius: 4px; transition: background 0.2s; }
.topnav a:hover { background: #34495e; }
h1 { color: #2c3e50; margin-top: 20px; }
h2 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 6px; }
.chart { background: white; border-radius: 8px; padding: 15px; margin: 20px 0;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
.stats { display: flex; gap: 15px; flex-wrap: wrap; margin: 20px 0; }
.stat-card { background: white; padding: 15px 20px; border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; min-width: 130px; }
.stat-card .number { font-size: 1.8em; font-weight: bold; }
.stat-card .label { color: #7f8c8d; font-size: 0.85em; }
.tab-container { display: flex; flex-wrap: wrap; gap: 4px; margin: 10px 0; }
.tab-btn { padding: 6px 14px; border: 1px solid #bdc3c7; border-radius: 4px 4px 0 0;
  background: #ecf0f1; cursor: pointer; font-size: 0.85em; }
.tab-btn.active { background: #3498db; color: white; border-color: #3498db; }
.section-content { background: white; border-radius: 0 8px 8px 8px; padding: 15px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
.finding-row { padding: 10px; border-bottom: 1px solid #ecf0f1; }
.finding-row:hover { background: #f8f9fa; }
.impact-bar-container { background: #ecf0f1; border-radius: 4px; height: 8px; width: 200px;
  display: inline-block; margin: 4px 8px; vertical-align: middle; }
.impact-bar { height: 100%; border-radius: 4px; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.78em;
  color: white; margin: 1px 2px; }
.theme-card { background: white; border-left: 4px solid #3498db; padding: 12px 16px;
  margin: 10px 0; border-radius: 0 8px 8px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
.meta-obs { background: #fef9e7; border-left: 4px solid #f39c12; padding: 12px 16px;
  margin: 10px 0; border-radius: 0 8px 8px 0; }
table.findings-table { width: 100%; border-collapse: collapse; font-size: 0.85em; }
table.findings-table thead { position: sticky; top: 52px; background: #2c3e50; color: white; z-index: 10; }
table.findings-table th { padding: 8px 6px; cursor: pointer; user-select: none; white-space: nowrap; }
table.findings-table th:hover { background: #34495e; }
table.findings-table td { padding: 6px; border-bottom: 1px solid #ecf0f1; }
table.findings-table tr:nth-child(even) { background: #f8f9fa; }
table.findings-table tr:hover { background: #eaf2f8; }
.empty { color: #95a5a6; font-style: italic; padding: 20px; text-align: center; }
.cat-filter { opacity: 1; transition: opacity 0.2s; }
.cat-filter.dim { opacity: 0.35; }
"""

    js = """
function showSection(secId) {
  document.querySelectorAll('.section-content').forEach(function(el) { el.style.display = 'none'; });
  var el = document.getElementById(secId);
  if (el) el.style.display = 'block';
  document.querySelectorAll('.tab-btn').forEach(function(b) { b.classList.remove('active'); });
  event.target.classList.add('active');
}

function sortTable(col) {
  var table = document.getElementById('allFindingsTable');
  var tbody = table.querySelector('tbody');
  var rows = Array.from(tbody.querySelectorAll('tr'));
  var asc = table.getAttribute('data-sort-col') == col && table.getAttribute('data-sort-dir') == 'asc';
  rows.sort(function(a, b) {
    var ca = a.children[col], cb = b.children[col];
    var va = ca.getAttribute('data-value') !== null ? parseFloat(ca.getAttribute('data-value')) : ca.textContent.trim();
    var vb = cb.getAttribute('data-value') !== null ? parseFloat(cb.getAttribute('data-value')) : cb.textContent.trim();
    if (typeof va === 'number' && typeof vb === 'number') return asc ? va - vb : vb - va;
    va = String(va); vb = String(vb);
    return asc ? va.localeCompare(vb) : vb.localeCompare(va);
  });
  rows.forEach(function(r) { tbody.appendChild(r); });
  table.setAttribute('data-sort-col', col);
  table.setAttribute('data-sort-dir', asc ? 'desc' : 'asc');
}

var catFilters = {};
function toggleCatFilter(cat) {
  catFilters[cat] = !catFilters[cat];
  var anyActive = Object.values(catFilters).some(function(v) { return v; });
  document.querySelectorAll('.cat-filter').forEach(function(b) {
    var c = b.textContent.trim();
    b.classList.toggle('dim', anyActive && !catFilters[c]);
  });
  document.querySelectorAll('.frow').forEach(function(r) {
    var rc = r.getAttribute('data-cat');
    r.style.display = (!anyActive || catFilters[rc]) ? '' : 'none';
  });
}
"""

    stat_cards = f"""
<div class="stat-card"><div class="number" style="color:#3498db">{n_findings}</div><div class="label">Total Findings</div></div>
<div class="stat-card"><div class="number" style="color:#2ecc71">{n_tri}</div><div class="label">Corroborated (≥2)</div></div>
<div class="stat-card"><div class="number" style="color:#9b59b6">{n_phases}</div><div class="label">Phases</div></div>
<div class="stat-card"><div class="number" style="color:#1abc9c">{n_crosslinks}</div><div class="label">Cross-links</div></div>
<div class="stat-card"><div class="number" style="color:#e67e22">{avg_impact:.2f}</div><div class="label">Avg Impact</div></div>
<div class="stat-card"><div class="number" style="color:{_cat_color(top_cat)}">{html_escape(top_cat)} ({top_cat_n})</div><div class="label">Top Category</div></div>
<div class="stat-card"><div class="number" style="color:#e74c3c">{n_high_impact}</div><div class="label">High-Impact (&gt;0.5)</div></div>
<div class="stat-card"><div class="number" style="color:#34495e">{avg_per_section:.1f}</div><div class="label">Findings/Section</div></div>
"""

    html_path = KNOWLEDGE_MAP_DIR / "knowledge_map_overview.html"
    nav_links = (
        '<a href="#summary">Summary</a>'
        '<a href="#sections">Sections</a>'
        '<a href="#distribution">Distribution</a>'
        '<a href="#sunburst">Hierarchy</a>'
        '<a href="#trimatrix">Corroboration</a>'
        '<a href="#sankey">Flow</a>'
        '<a href="#table">Table</a>'
        '<a href="#themes">Patterns</a>'
        '<a href="#bar">Top 30</a>'
        '<a href="#heatmap">Phase×Cat</a>'
        '<a href="#network">Network</a>'
    )
    empty_msg = '<p class="empty">No findings collected. Run pipeline phases first.</p>' if not findings else ""

    html_content = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Knowledge Map — Bibliometric Synthesis</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>{css}</style>
</head><body>

<nav class="topnav">{nav_links}</nav>

<h1>Knowledge Map — Deterministic Bibliometric Overview</h1>
{empty_msg}

<div id="summary"><h2>Executive Summary</h2>
<div class="stats">{stat_cards}</div></div>

<div id="sections"><h2>Section-by-Section Breakdown</h2>
<div class="tab-container">{tabs_buttons}</div>
{tabs_panels}</div>

<div id="distribution"><h2>Impact Score Distribution</h2>
{chart_dist}</div>

<div id="sunburst"><h2>Findings Hierarchy</h2>
{chart_sun}</div>

<div id="trimatrix"><h2>Cross-Phase Corroboration Matrix</h2>
{chart_tri}</div>

<div id="sankey"><h2>Phase → Section Flow</h2>
{chart_sankey}</div>

<div id="table"><h2>All Findings</h2>
<div style="margin:10px 0">{table_filter_badges}</div>
<div class="chart" style="overflow-x:auto">
<table class="findings-table" id="allFindingsTable">
<thead><tr>
  <th onclick="sortTable(0)">ID</th><th onclick="sortTable(1)">Title</th>
  <th onclick="sortTable(2)">Phase</th><th onclick="sortTable(3)">Section</th>
  <th onclick="sortTable(4)">Category</th><th onclick="sortTable(5)">Type</th>
  <th onclick="sortTable(6)">Impact</th><th onclick="sortTable(7)">Corr.</th>
  <th onclick="sortTable(8)">Tags</th>
</tr></thead>
<tbody>{table_rows}</tbody>
</table></div></div>

<div id="themes"><h2>Cross-Cutting Corroborated Patterns</h2>
{themes_html}</div>

<div id="bar"><h2>Top 30 by Impact</h2>
{chart_bar}</div>

<div id="heatmap"><h2>Phase × Category</h2>
{chart_heat}</div>

<div id="network"><h2>Finding Connections Network</h2>
{chart_net}</div>

<script>
{js}
{plotly_calls}
</script>
</body></html>"""

    html_path.write_text(html_content, encoding="utf-8")
    logger.info(f"HTML overview saved: {html_path}")
    return html_path


def build_knowledge_map(
    articles: List[dict],
    impact_weights: Optional[dict] = None,
    phase_status: Optional[dict] = None,
) -> dict:
    """Orchestrate deterministic Knowledge Map: collect -> score -> cross-reference -> skeleton."""
    logger.info("=" * 60)
    logger.info("Knowledge Map — Deterministic Bibliometric Synthesis")
    logger.info("=" * 60)

    eff_weights = _normalize_impact_weights(impact_weights)
    logger.info(f"Impact weights: {eff_weights}")

    phase_inputs: List[dict] = []
    findings = collect_all_findings(articles, phase_status=phase_status, inputs_report=phase_inputs)
    if not findings:
        logger.warning("No findings collected — skipping Knowledge Map")
        return {"findings": [], "skeleton": {}, "matrix": {}, "stats": {"phase_inputs": phase_inputs}}

    findings = compute_impact_scores(findings, eff_weights)
    findings, matrix = detect_triangulation(findings, eff_weights)
    skeleton = build_results_skeleton(findings)

    phase_counts = Counter(f["source_phase"] for f in findings)
    type_counts = Counter(f["type"] for f in findings)
    cat_counts = Counter(f["category"] for f in findings)
    tri_findings = [f for f in findings if f.get("triangulation_score", 0) >= 2]

    stats = {
        "total_findings": len(findings),
        "by_phase": dict(phase_counts),
        "by_type": dict(type_counts),
        "by_category": dict(cat_counts),
        "triangulated_findings": len(tri_findings),
        "cross_phase_links": len(matrix),
        "impact_weights": eff_weights,
        "phase_inputs": phase_inputs,
        "corroboration_method": {
            "method": "lexical_overlap",
            "components": ["tag_overlap", "article_overlap", "title_fuzzy"],
            "fuzzy_lib": "rapidfuzz",
            "link_threshold": 0.3,
        },
        "timestamp": datetime.now().isoformat(),
    }

    result = {
        "findings": findings,
        "skeleton": skeleton,
        "matrix": {f"{k[0]}-{k[1]}": v for k, v in matrix.items()},
        "stats": stats,
    }

    KNOWLEDGE_MAP_DIR.mkdir(parents=True, exist_ok=True)
    save_json(result, KNOWLEDGE_MAP_DIR / "knowledge_map_data.json")
    save_json(
        _impact_weight_sensitivity(findings, eff_weights),
        KNOWLEDGE_MAP_DIR / "impact_weight_sensitivity.json",
    )
    logger.info(
        f"Knowledge Map: {len(findings)} findings, "
        f"{len(tri_findings)} corroborated, {len(matrix)} cross-links"
    )
    return result


def run_knowledge_map(
    articles: List[dict],
    top_n: int = 20,
    impact_weights: Optional[dict] = None,
    phase_status: Optional[dict] = None,
) -> dict:
    """Run the complete deterministic Knowledge Map pipeline (Phase 7).

    `phase_status` es RunRecorder.phase_status de main.py: permite declarar como
    FALLIDA una fase que fallo en esta corrida en vez de omitirla en silencio.
    """
    logger.info("=" * 60)
    logger.info("PHASE 7 — Knowledge Map (Deterministic Synthesis)")
    logger.info("=" * 60)

    kmap = build_knowledge_map(articles, impact_weights=impact_weights, phase_status=phase_status)
    findings = kmap.get("findings", [])
    matrix = kmap.get("matrix", {})
    skeleton = kmap.get("skeleton", {})
    stats = kmap.get("stats", {})

    if not findings:
        logger.warning("No findings to process — Knowledge Map complete (empty)")
        return {"layer_a": kmap, "status": "ok"}

    themes = synthesize_algorithmic_themes(findings)
    save_json(themes, KNOWLEDGE_MAP_DIR / "knowledge_map_synthesis.json")

    report_path = generate_knowledge_map_report(findings, skeleton, themes, stats)
    html_path = generate_overview_html(findings, matrix, themes)

    return {
        "layer_a": kmap,
        "themes": themes,
        "report_path": str(report_path),
        "html_path": str(html_path) if html_path else None,
        "status": "ok",
    }

run_phase7 = run_knowledge_map
