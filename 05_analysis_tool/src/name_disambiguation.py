"""Fase 0b: Desambiguacion de autores e instituciones.

Agrupa variantes de nombres usando fuzzy matching (rapidfuzz) y ORCIDs
de OpenAlex para generar formas canonicas.
"""
from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from pathlib import Path

from rapidfuzz import fuzz
from tqdm import tqdm

from config import DATA_DIR

logger = logging.getLogger("bibliometrico.disambiguation")

DISAMBIGUATION_CACHE_DIR = DATA_DIR / "disambiguation_cache"


def _cluster_similar_names(
    names: list[str], threshold: float = 85.0
) -> list[list[str]]:
    """Agrupa nombres similares usando fuzzy matching greedy.

    Args:
        names: Lista de nombres unicos.
        threshold: Umbral minimo de similitud (0-100).

    Returns:
        Lista de clusters (cada cluster es una lista de variantes).
    """
    if not names:
        return []

    remaining = list(names)
    clusters: list[list[str]] = []

    while remaining:
        seed = remaining.pop(0)
        cluster = [seed]
        still_remaining = []

        for name in remaining:
            score = fuzz.ratio(seed.lower(), name.lower())
            if score >= threshold:
                cluster.append(name)
            else:
                still_remaining.append(name)

        clusters.append(cluster)
        remaining = still_remaining

    return clusters


def _pick_canonical(variants: list[str]) -> str:
    """Elige la forma canonica de un cluster de variantes.

    Preferencia: nombre mas largo (mas completo), luego el mas frecuente
    en el corpus si hay empate.
    """
    if len(variants) == 1:
        return variants[0]
    # Longest name is usually the most complete form
    return max(variants, key=lambda n: (len(n), n))


def _extract_orcids_from_openalex(articles: list[dict]) -> dict[str, str]:
    """Extrae mapeo nombre -> ORCID desde datos OpenAlex.

    Returns:
        Dict {author_name: orcid_url} para autores con ORCID.
    """
    orcid_map: dict[str, str] = {}
    for a in articles:
        openalex = a.get("openalex", {})
        for auth in openalex.get("authorships", []):
            name = auth.get("name", "")
            orcid = auth.get("orcid", "")
            if not orcid:
                # Try nested author dict
                orcid = (auth.get("author", {}) or {}).get("orcid", "") or ""
            if name and orcid:
                orcid_map[name] = orcid
    return orcid_map


def _build_orcid_clusters(
    orcid_map: dict[str, str],
) -> dict[str, list[str]]:
    """Agrupa nombres que comparten el mismo ORCID.

    Returns:
        Dict {orcid: [name1, name2, ...]}
    """
    clusters: dict[str, list[str]] = defaultdict(list)
    for name, orcid in orcid_map.items():
        clusters[orcid].append(name)
    return dict(clusters)


def disambiguate_authors(
    articles: list[dict], threshold: float = 90.0
) -> tuple[list[dict], dict]:
    """Desambigua autores en el corpus.

    1. Extrae todos los nombres de autores unicos.
    2. Agrupa por ORCID (gold standard).
    3. Fuzzy-match los restantes con umbral conservador.
    4. Asigna formas canonicas.

    Args:
        articles: Lista de articulos.
        threshold: Umbral de similitud para fuzzy matching.

    Returns:
        (articles_updated, report_dict)
    """
    logger.info("Desambiguando autores...")

    # Collect all author names
    all_names: set[str] = set()
    for a in articles:
        for author in a.get("authors", []):
            if author and author.strip():
                all_names.add(author.strip())
        # Also from OpenAlex authorships
        for auth in a.get("openalex", {}).get("authorships", []):
            name = auth.get("name", "")
            if name:
                all_names.add(name)

    if not all_names:
        logger.warning("No author names found for disambiguation")
        return articles, {"n_authors": 0, "n_clusters": 0, "n_merged": 0}

    logger.info(f"  {len(all_names)} nombres de autor unicos encontrados")

    # Step 1: ORCID-based clustering
    orcid_map = _extract_orcids_from_openalex(articles)
    orcid_clusters = _build_orcid_clusters(orcid_map)

    # Build name->canonical from ORCID clusters
    name_to_canonical: dict[str, str] = {}
    orcid_resolved = set()
    for orcid, names in orcid_clusters.items():
        if len(names) > 1:
            canonical = _pick_canonical(names)
            for n in names:
                name_to_canonical[n] = canonical
                orcid_resolved.add(n)

    # Step 2: Fuzzy match remaining names
    unresolved = [n for n in sorted(all_names) if n not in orcid_resolved]
    fuzzy_clusters = _cluster_similar_names(unresolved, threshold=threshold)

    n_merged = 0
    for cluster in fuzzy_clusters:
        if len(cluster) > 1:
            canonical = _pick_canonical(cluster)
            for variant in cluster:
                name_to_canonical[variant] = canonical
            n_merged += len(cluster) - 1

    # Step 3: Apply canonical names to articles
    for a in articles:
        original_authors = a.get("authors", [])
        canonical_authors = []
        for author in original_authors:
            canonical = name_to_canonical.get(author.strip(), author.strip())
            canonical_authors.append(canonical)

        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for ca in canonical_authors:
            if ca not in seen:
                seen.add(ca)
                deduped.append(ca)

        a["authors_canonical"] = deduped

    # Build ID map for reporting
    author_id_map = {}
    for variant, canonical in name_to_canonical.items():
        if variant != canonical:
            author_id_map.setdefault(canonical, []).append(variant)

    report = {
        "n_authors": len(all_names),
        "n_clusters": len(fuzzy_clusters) + len([c for c in orcid_clusters.values() if len(c) > 1]),
        "n_merged": n_merged + sum(len(v) for v in orcid_clusters.values() if len(v) > 1),
        "n_orcid_resolved": len(orcid_resolved),
        "author_id_map": author_id_map,
        "sample_merges": dict(list(author_id_map.items())[:10]),
    }

    logger.info(
        f"  Autores: {report['n_merged']} variantes fusionadas, "
        f"{report['n_orcid_resolved']} resueltas por ORCID"
    )

    return articles, report


def disambiguate_institutions(
    articles: list[dict], threshold: float = 90.0
) -> tuple[list[dict], dict]:
    """Desambigua instituciones usando IDs de OpenAlex + fuzzy matching.

    Returns:
        (articles_updated, report_dict)
    """
    logger.info("Desambiguando instituciones...")

    # Collect institutions: name -> openalex_id (if available)
    inst_names: set[str] = set()
    inst_id_map: dict[str, str] = {}  # openalex_id -> display_name

    for a in articles:
        for auth in a.get("openalex", {}).get("authorships", []):
            for inst in auth.get("institutions", []):
                name = inst.get("name", "")
                if name:
                    inst_names.add(name)
                    # Use OpenAlex institution ID for disambiguation
                    inst_id = inst.get("id", "") or inst.get("openalex_id", "")
                    if inst_id:
                        inst_id_map.setdefault(inst_id, name)

    if not inst_names:
        logger.warning("No institution names found")
        return articles, {"n_institutions": 0, "n_merged": 0}

    # Group by OpenAlex ID first
    id_to_names: dict[str, set[str]] = defaultdict(set)
    for a in articles:
        for auth in a.get("openalex", {}).get("authorships", []):
            for inst in auth.get("institutions", []):
                name = inst.get("name", "")
                inst_id = inst.get("id", "") or inst.get("openalex_id", "")
                if name and inst_id:
                    id_to_names[inst_id].add(name)

    # Build canonical map from IDs
    inst_canonical: dict[str, str] = {}
    id_resolved = set()
    for inst_id, names_set in id_to_names.items():
        names_list = list(names_set)
        if len(names_list) > 1:
            canonical = _pick_canonical(names_list)
            for n in names_list:
                inst_canonical[n] = canonical
                id_resolved.add(n)

    # Fuzzy match remaining
    unresolved = [n for n in sorted(inst_names) if n not in id_resolved]
    fuzzy_clusters = _cluster_similar_names(unresolved, threshold=threshold)

    n_merged = 0
    for cluster in fuzzy_clusters:
        if len(cluster) > 1:
            canonical = _pick_canonical(cluster)
            for variant in cluster:
                inst_canonical[variant] = canonical
            n_merged += len(cluster) - 1

    # Apply to articles (update affiliations list)
    for a in articles:
        original_affs = a.get("affiliations", [])
        canonical_affs = list(set(
            inst_canonical.get(aff, aff) for aff in original_affs
        ))
        if canonical_affs != original_affs:
            a["affiliations_canonical"] = canonical_affs

    report = {
        "n_institutions": len(inst_names),
        "n_merged": n_merged + sum(1 for v in id_to_names.values() if len(v) > 1),
        "n_id_resolved": len(id_resolved),
    }

    logger.info(
        f"  Instituciones: {report['n_merged']} variantes fusionadas, "
        f"{report['n_id_resolved']} resueltas por ID"
    )

    return articles, report


def run_phase0b(articles: list[dict], threshold: float = 90.0) -> list[dict]:
    """Ejecuta Fase 0b: Desambiguacion de autores e instituciones.

    Args:
        articles: Lista de articulos cargados.
        threshold: Umbral de similitud para fuzzy matching.

    Returns:
        articles con campos canonicos anadidos.
    """
    logger.info("=== FASE 0c: Desambiguacion de Nombres ===")

    DISAMBIGUATION_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    articles, author_report = disambiguate_authors(articles, threshold=threshold)
    articles, inst_report = disambiguate_institutions(articles, threshold=threshold)

    # Save report
    report = {
        "authors": author_report,
        "institutions": inst_report,
    }
    report_path = DISAMBIGUATION_CACHE_DIR / "disambiguation_report.json"
    try:
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"Reporte guardado: {report_path}")
    except Exception as e:
        logger.warning(f"No se pudo guardar reporte: {e}")

    logger.info("Fase 0b completada.")
    return articles

run_phase0c = run_phase0b
