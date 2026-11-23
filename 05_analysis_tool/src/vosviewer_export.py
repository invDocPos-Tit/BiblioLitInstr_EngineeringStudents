"""Export data in native VOSviewer format (map + network .txt files).

Generates tab-separated map files and sparse network files that can be
opened directly in VOSviewer via File > Open.  Also produces auxiliary
corpus, scores, and thesaurus files for text-mining workflows.
"""

import logging
import re
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np

from config import VOSVIEWER_DIR
from src.utils import (
    build_corpus_keyword_normalizer,
    extract_normalized_article_keywords,
    canonicalize_keyword,
    _iter_raw_article_keywords,
    normalize_author_display_name,
    normalize_institution_name,
    build_corpus_institution_normalizer,
    format_journal_display_name,
    _journal_match_key,
    load_knowledge_base,
)

logger = logging.getLogger("bibliometrico.vosviewer_export")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _sanitize_for_tsv(value) -> str:
    """Remove tabs, newlines, quotes and control chars that break TSV/VOSviewer format."""
    if value is None:
        return ""
    s = str(value)
    s = s.replace('"', "'")  # VOSviewer treats " as field delimiter
    s = re.sub(r"[\t\r\n]+", " ", s)
    s = re.sub(r"[\x00-\x1f]", "", s)
    return s.strip()


def _make_description(article: dict) -> str:
    """Build an HTML snippet for the VOSviewer description tooltip."""
    title = _sanitize_for_tsv(article.get("title", ""))
    source = _sanitize_for_tsv(article.get("publication_title", ""))
    year = article.get("publication_year", "")
    cites = article.get("cited_by_count", 0)
    parts = [f"<b>{title}</b>"]
    if source:
        parts.append(f"<i>{source}</i>")
    if year:
        parts.append(f"Year: {year}")
    parts.append(f"Citations: {cites}")
    return "<br>".join(parts)


def _get_doi_url(article: dict) -> str:
    doi = article.get("doi", "") or ""
    if doi and not doi.startswith("http"):
        return f"https://doi.org/{doi}"
    return doi


# ---------------------------------------------------------------------------
# score computation helpers
# ---------------------------------------------------------------------------

def _compute_keyword_scores(articles: list[dict]) -> dict:
    """Return {keyword: {'avg_year': float, 'avg_citations': float, 'frequency': int}}."""
    kw_years = defaultdict(list)
    kw_cites = defaultdict(list)
    kw_freq = Counter()
    corpus_map = build_corpus_keyword_normalizer(articles)

    for a in articles:
        year = a.get("publication_year", 0)
        cites = a.get("cited_by_count", 0)
        keywords = extract_normalized_article_keywords(
            a, corpus_map=corpus_map, drop_stopwords=True
        )
        for kw in keywords:
            kw_freq[kw] += 1
            if year and year > 0:
                kw_years[kw].append(year)
            kw_cites[kw].append(cites)

    result = {}
    for kw in kw_freq:
        avg_y = sum(kw_years[kw]) / len(kw_years[kw]) if kw_years[kw] else 0
        avg_c = sum(kw_cites[kw]) / len(kw_cites[kw]) if kw_cites[kw] else 0
        result[kw] = {
            "avg_year": round(avg_y, 2),
            "avg_citations": round(avg_c, 2),
            "frequency": kw_freq[kw],
        }
    return result


def _compute_entity_scores(articles: list[dict], level: str) -> dict:
    """Return {entity: {'avg_year': float, 'avg_citations': float, 'frequency': int}}.

    level: 'authors', 'institutions', 'countries', or 'journals'.
    """
    ent_years = defaultdict(list)
    ent_cites = defaultdict(list)
    ent_freq = Counter()
    inst_map = build_corpus_institution_normalizer(articles) if level == "institutions" else None
    journal_key_to_label: dict[str, str] = {}

    for a in articles:
        year = a.get("publication_year", 0)
        cites = a.get("cited_by_count", 0)
        openalex = a.get("openalex", {})
        entities = set()

        if level == "authors":
            for auth in openalex.get("authorships", []):
                name = normalize_author_display_name(auth.get("name", ""))
                if name:
                    entities.add(name)
        elif level == "institutions":
            for auth in openalex.get("authorships", []):
                for inst in auth.get("institutions", []):
                    name = normalize_institution_name(inst.get("name", ""), corpus_inst_map=inst_map)
                    if name:
                        entities.add(name)
        elif level == "countries":
            for auth in openalex.get("authorships", []):
                for c_name in auth.get("countries", []):
                    if c_name:
                        entities.add(str(c_name).strip().upper())
                for inst in auth.get("institutions", []):
                    country = inst.get("country", "")
                    if country:
                        entities.add(str(country).strip().upper())
        elif level == "journals":
            raw_j = a.get("publication_title", "")
            if raw_j and raw_j != "nan":
                j_key = _journal_match_key(raw_j)
                if j_key:
                    if j_key not in journal_key_to_label:
                        journal_key_to_label[j_key] = format_journal_display_name(raw_j)
                    entities.add(journal_key_to_label[j_key])

        for e in entities:
            ent_freq[e] += 1
            if year and year > 0:
                ent_years[e].append(year)
            ent_cites[e].append(cites)

    result = {}
    for e in ent_freq:
        avg_y = sum(ent_years[e]) / len(ent_years[e]) if ent_years[e] else 0
        avg_c = sum(ent_cites[e]) / len(ent_cites[e]) if ent_cites[e] else 0
        result[e] = {
            "avg_year": round(avg_y, 2),
            "avg_citations": round(avg_c, 2),
            "frequency": ent_freq[e],
        }
    return result


# ---------------------------------------------------------------------------
# map + network export
# ---------------------------------------------------------------------------

def _export_map_and_network(
    G: nx.Graph,
    name: str,
    articles: list[dict],
    article_map: dict,
    output_dir: Path,
) -> None:
    """Generate a VOSviewer map file and network file for a single network."""
    if G.number_of_nodes() == 0:
        logger.info(f"Skipping empty network: {name}")
        return

    nodes = list(G.nodes())
    node_to_id = {n: i + 1 for i, n in enumerate(nodes)}

    # Determine score/weight sources based on network type
    kw_scores = None
    entity_scores = None
    if name == "keyword_cooccurrence":
        kw_scores = _compute_keyword_scores(articles)
    elif name.startswith("coauthorship_"):
        level = name.replace("coauthorship_", "")
        entity_scores = _compute_entity_scores(articles, level)
    elif name.startswith("coupling_"):
        level = name.replace("coupling_", "")
        entity_scores = _compute_entity_scores(articles, level)

    # --- Compute layout for x/y coordinates ---
    communities = nx.get_node_attributes(G, "community")
    try:
        from src.network_analysis import _community_layout
        pos = _community_layout(G, communities)
    except ImportError:
        k = 3.0 / max(np.sqrt(G.number_of_nodes()), 1)
        pos = nx.spring_layout(G, k=k, iterations=80, seed=42)

    # --- Map file ---
    map_path = output_dir / f"{name}_map.txt"
    lines = []
    header = (
        "id\tlabel\tx\ty\tcluster\tweight<Links>\t"
        "weight<Total link strength>\tweight<Citations>\t"
        "score<Avg. pub. year>\tdescription\turl"
    )
    lines.append(header)
    degrees = dict(G.degree())
    weighted_degrees = dict(G.degree(weight="weight"))

    # VOSviewer requires cluster numbers 1-1000.
    # If community detection produced >1000 clusters, merge smallest into top 999.
    if communities:
        unique_clusters = sorted(set(communities.values()))
        if len(unique_clusters) > 999:
            from collections import Counter as _Cnt
            cluster_sizes = _Cnt(communities.values())
            # Keep top 999 clusters by size, merge rest into cluster 999
            top_999 = {c for c, _ in cluster_sizes.most_common(999)}
            remap = {}
            new_id = 1
            for c in unique_clusters:
                if c in top_999:
                    remap[c] = new_id
                    new_id += 1
                else:
                    remap[c] = 999  # merge small clusters
            communities = {n: remap[c] for n, c in communities.items()}
        else:
            # Remap to 1-based contiguous
            remap = {c: i + 1 for i, c in enumerate(unique_clusters)}
            communities = {n: remap[c] for n, c in communities.items()}

    for n in nodes:
        nid = node_to_id[n]
        label = _sanitize_for_tsv(G.nodes[n].get("label", str(n)))
        cluster = communities.get(n, 1)  # VOSviewer clusters are 1-based
        links = degrees.get(n, 0)
        total_link_strength = weighted_degrees.get(n, 0)

        # Determine citations weight and score
        citations = 0
        avg_year = ""
        description = ""
        url = ""

        if name == "bibliographic_coupling":
            citations = G.nodes[n].get("citations") or 0
            avg_year = G.nodes[n].get("year") or ""
            if avg_year == 0:
                avg_year = ""
            art = article_map.get(n, {})
            if art:
                description = _sanitize_for_tsv(_make_description(art))
                url = _get_doi_url(art)

        elif name == "cocitation":
            citations = G.nodes[n].get("cocitation_count") or 0
            raw_url = G.nodes[n].get("openalex_url") or ""
            # Ensure URL is valid for VOSviewer (must start with http)
            if raw_url and not raw_url.startswith("http"):
                # Try to reconstruct from node ID (W-prefixed)
                node_str = str(n)
                if node_str.startswith("W") and node_str[1:].isdigit():
                    raw_url = f"https://openalex.org/{node_str}"
                else:
                    raw_url = ""
            url = _sanitize_for_tsv(raw_url)

        elif name == "keyword_cooccurrence":
            if kw_scores and n in kw_scores:
                citations = kw_scores[n]["frequency"]
                avg_year = kw_scores[n]["avg_year"] if kw_scores[n]["avg_year"] > 0 else ""
            else:
                citations = G.nodes[n].get("frequency", 0)

        elif name.startswith("coauthorship_"):
            freq = G.nodes[n].get("frequency", 0)
            citations = freq
            if entity_scores and n in entity_scores:
                avg_year = entity_scores[n]["avg_year"] if entity_scores[n]["avg_year"] > 0 else ""
            if name == "coauthorship_countries":
                kb_countries = load_knowledge_base().get("country_iso_names", {})
                full_c = kb_countries.get(str(n).upper(), "")
                if full_c:
                    description = _sanitize_for_tsv(f"<b>{full_c} ({n})</b><br>Documents: {freq}")

        elif name.startswith("coupling_"):
            citations = G.nodes[n].get("total_cites", 0)
            if entity_scores and n in entity_scores:
                avg_year = entity_scores[n]["avg_year"] if entity_scores[n]["avg_year"] > 0 else ""
            if name == "coupling_countries":
                kb_countries = load_knowledge_base().get("country_iso_names", {})
                full_c = kb_countries.get(str(n).upper(), "")
                if full_c:
                    description = _sanitize_for_tsv(f"<b>{full_c} ({n})</b><br>Citations: {citations}")

        elif name == "cocitation_journals":
            citations = G.nodes[n].get("cocitation_count") or 0

        x_coord = round(pos[n][0], 4) if n in pos else 0.0
        y_coord = round(pos[n][1], 4) if n in pos else 0.0

        row = (
            f"{nid}\t{label}\t{x_coord}\t{y_coord}\t{cluster}\t{links}\t"
            f"{total_link_strength}\t{citations}\t"
            f"{avg_year}\t{description}\t{url}"
        )
        lines.append(row)

    map_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Saved VOSviewer map: {map_path.name} ({len(nodes)} items)")

    # --- Network file ---
    net_path = output_dir / f"{name}_network.txt"
    edge_lines = []
    for u, v, d in G.edges(data=True):
        uid = node_to_id[u]
        vid = node_to_id[v]
        w = d.get("weight", 1)
        edge_lines.append(f"{uid}\t{vid}\t{w}")

    net_path.write_text("\n".join(edge_lines), encoding="utf-8")
    logger.info(f"Saved VOSviewer network: {net_path.name} ({len(edge_lines)} links)")


# ---------------------------------------------------------------------------
# corpus + scores
# ---------------------------------------------------------------------------

def _export_corpus_and_scores(articles: list[dict], output_dir: Path) -> None:
    """Export corpus.txt and scores.txt for VOSviewer text mining."""
    # corpus.txt - one line per article: title + abstract
    corpus_lines = []
    score_lines = ["score<Pub. year>\tscore<Citations>"]

    for a in articles:
        title = _sanitize_for_tsv(a.get("title", ""))
        abstract = _sanitize_for_tsv(a.get("abstract", ""))
        line = f"{title}. {abstract}" if abstract else title
        corpus_lines.append(line)

        year = a.get("publication_year", "")
        cites = a.get("cited_by_count", 0)
        score_lines.append(f"{year}\t{cites}")

    corpus_path = output_dir / "corpus.txt"
    corpus_path.write_text("\n".join(corpus_lines), encoding="utf-8")
    logger.info(f"Saved corpus.txt ({len(corpus_lines)} documents)")

    scores_path = output_dir / "scores.txt"
    scores_path.write_text("\n".join(score_lines), encoding="utf-8")
    logger.info(f"Saved scores.txt")


# ---------------------------------------------------------------------------
# thesaurus files
# ---------------------------------------------------------------------------

def _export_thesaurus_keywords(articles: list[dict], output_dir: Path) -> None:
    """Generate native VOSviewer TSV keyword thesaurus (label\\treplace by) from corpus + knowledge base."""
    corpus_map = build_corpus_keyword_normalizer(articles)
    mappings: dict[str, str] = {}

    for a in articles:
        for raw in _iter_raw_article_keywords(a):
            if not isinstance(raw, str) or not raw.strip():
                continue
            raw_clean = _sanitize_for_tsv(raw.strip())
            canon = _sanitize_for_tsv(canonicalize_keyword(raw, corpus_map=corpus_map))
            if raw_clean and canon and raw_clean != canon:
                mappings[raw_clean] = canon
            low_clean = raw_clean.lower()
            if low_clean and canon and low_clean != canon:
                mappings[low_clean] = canon

    lines = ["label\treplace by"]
    for variant, preferred in sorted(mappings.items()):
        lines.append(f"{variant}\t{preferred}")

    if len(lines) > 1:
        path = output_dir / "thesaurus_keywords.txt"
        path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Saved thesaurus_keywords.txt ({len(lines) - 1} mappings)")


def _export_thesaurus_authors(articles: list[dict], output_dir: Path) -> None:
    """Generate native VOSviewer TSV author thesaurus (label\\treplace by) for variant author names."""
    mappings: dict[str, str] = {}
    by_norm_lower: dict[str, set[str]] = defaultdict(set)

    for a in articles:
        openalex = a.get("openalex", {})
        for auth in openalex.get("authorships", []):
            raw_name = auth.get("name", "")
            if not raw_name or not raw_name.strip():
                continue
            raw_clean = _sanitize_for_tsv(raw_name.strip())
            norm_name = _sanitize_for_tsv(normalize_author_display_name(raw_name))
            if raw_clean and norm_name and raw_clean != norm_name:
                mappings[raw_clean] = norm_name
            if norm_name:
                by_norm_lower[norm_name.lower()].add(raw_clean)
                by_norm_lower[norm_name.lower()].add(norm_name)

    for low_key, variants in by_norm_lower.items():
        if len(variants) > 1:
            preferred = normalize_author_display_name(max(variants, key=len))
            for v in sorted(variants):
                if v != preferred:
                    mappings[v] = preferred

    lines = ["label\treplace by"]
    for variant, preferred in sorted(mappings.items()):
        lines.append(f"{variant}\t{preferred}")

    if len(lines) > 1:
        path = output_dir / "thesaurus_authors.txt"
        path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Saved thesaurus_authors.txt ({len(lines) - 1} mappings)")


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------

def run_vosviewer_export(articles: list[dict], networks: dict) -> None:
    """Export all networks in native VOSviewer format.

    Args:
        articles: List of enriched article dicts.
        networks: Dict mapping network name -> nx.Graph (from Phase 3).
    """
    output_dir = VOSVIEWER_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 50)
    logger.info("VOSviewer Native Export")
    logger.info("=" * 50)

    # Build article lookup by key
    article_map = {}
    for a in articles:
        key = a.get("key", "")
        if key:
            article_map[key] = a

    # Export map + network for each graph
    for name, G in networks.items():
        try:
            _export_map_and_network(G, name, articles, article_map, output_dir)
        except Exception as e:
            logger.error(f"Failed to export {name}: {e}")

    # Auxiliary files
    try:
        _export_corpus_and_scores(articles, output_dir)
    except Exception as e:
        logger.error(f"Failed to export corpus/scores: {e}")

    try:
        _export_thesaurus_keywords(articles, output_dir)
    except Exception as e:
        logger.error(f"Failed to export keyword thesaurus: {e}")

    try:
        _export_thesaurus_authors(articles, output_dir)
    except Exception as e:
        logger.error(f"Failed to export author thesaurus: {e}")

    logger.info(f"VOSviewer export complete -> {output_dir}")
