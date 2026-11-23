"""Fase 2i: Indicadores inspirados en el Manifiesto de Leiden.

Calcula indicadores bibliometricos normalizados:
- MNCS: Mean Normalized Citation Score (normalizado por campo)
- PP(top 10%): Proporcion de articulos en top 10% de su campo
- MNJS: Mean Normalized Journal Score (basado en SJR)
- Deteccion de sesgos: idiomatico, geografico, tipo documental, cobertura

DISCLAIMER: La normalizacion se realiza intra-corpus, no contra una base
global. Los indicadores son aproximaciones utiles para analisis exploratorio
pero NO deben usarse para comparaciones inter-institucionales formales.
"""
from __future__ import annotations

import json
import logging
import math
from collections import Counter, defaultdict
from pathlib import Path

from config import OUTPUT_DIR, OPENALEX_CACHE_DIR
from src.utils import DiskCache, normalize_doi

logger = logging.getLogger("bibliometrico.leiden")

LEIDEN_DIR = OUTPUT_DIR / "leiden"


def _get_field_for_article(article: dict) -> str:
    """Obtiene el campo disciplinar de un articulo.

    Fallback chain: OpenAlex CWTS Leiden topic field > concepts level 0 > any concept > primary_topic > keywords > Unknown.
    """
    topics = article.get("topics") or article.get("openalex", {}).get("topics", [])
    if topics and isinstance(topics[0], dict):
        field = topics[0].get("field", "")
        if isinstance(field, dict):
            field = field.get("display_name", "")
        if field and str(field).strip():
            return str(field).strip()

    concepts = article.get("concepts", [])
    level0 = [c for c in concepts if c.get("level", -1) == 0]
    if level0:
        best = max(level0, key=lambda c: c.get("score", 0))
        return best.get("name", "Unknown")

    # Fallback: any concept
    if concepts:
        return concepts[0].get("name", "Unknown")

    # Fallback: primary_topic domain/field
    pt = article.get("primary_topic", {}) or article.get("openalex", {}).get("primary_topic", {}) or {}
    if isinstance(pt, dict):
        for key in ("field", "domain", "subfield", "name"):
            val = pt.get(key, "")
            if isinstance(val, dict):
                val = val.get("display_name", "")
            if val:
                return str(val).strip()

    # Fallback: first author keyword as broad field proxy
    for kw_field in ("author_keywords", "manual_tags"):
        kw_raw = article.get(kw_field, "") or ""
        if isinstance(kw_raw, str) and kw_raw.strip():
            first_kw = kw_raw.split(";")[0].strip()
            if first_kw and len(first_kw) > 2:
                return first_kw

    return "Unknown"


def compute_mncs(articles: list[dict]) -> list[dict]:
    """Calcula Mean Normalized Citation Score controlando por campo y año de publicacion.

    Prioriza el indicador mundial normalizado por campo y año de OpenAlex/CWTS (FWCI,
    donde 1.0 = media mundial para el mismo subcampo y año de publicacion) y utiliza
    normalizacion por cohorte (campo, año) como respaldo cuando un articulo carece
    de FWCI mundial. Incluye articulos con 0 citas tanto en el denominador como en
    el promedio global (evitando el sesgo de truncamiento post-normalizacion).

    Returns:
        Lista de dicts con key, field, year, citations, field_year_mean, mncs, mncs_intracorpus.
    """
    oa_cache = DiskCache(OPENALEX_CACHE_DIR)

    # Group by (field, year) and by field for cohort normalization
    cohort_cites: dict[tuple[str, int], list[float]] = defaultdict(list)
    field_cites: dict[str, list[float]] = defaultdict(list)
    article_meta: list[tuple[dict, str, int, float, float | None]] = []

    for a in articles:
        field = _get_field_for_article(a)
        try:
            year = int(a.get("publication_year") or 0)
        except (ValueError, TypeError):
            year = 0
        cites = float(a.get("cited_by_count", 0) or 0)
        cohort_cites[(field, year)].append(cites)
        field_cites[field].append(cites)

        # Look up OpenAlex world field+year normalized FWCI
        fwci_val = (a.get("openalex") or {}).get("fwci")
        if fwci_val is None:
            doi = a.get("doi", "")
            ndoi = normalize_doi(doi) if doi else ""
            if ndoi:
                cached = oa_cache.get(f"doi:{ndoi}")
                if isinstance(cached, dict):
                    fwci_val = cached.get("fwci")
        fwci_clean: float | None = None
        if isinstance(fwci_val, (int, float)) and not math.isnan(float(fwci_val)):
            fwci_clean = float(fwci_val)

        article_meta.append((a, field, year, cites, fwci_clean))

    field_means: dict[str, float] = {
        f: (sum(vals) / len(vals) if vals else 1.0)
        for f, vals in field_cites.items()
    }
    cohort_means: dict[tuple[str, int], float] = {}
    for (f, y), vals in cohort_cites.items():
        if len(vals) >= 3:
            cohort_means[(f, y)] = sum(vals) / len(vals)
        else:
            cohort_means[(f, y)] = field_means.get(f, 1.0)

    results = []
    for a, field, year, cites, fwci_clean in article_meta:
        c_mean = cohort_means.get((field, year), field_means.get(field, 1.0))
        mncs_intra = (cites / c_mean) if c_mean > 0 else 0.0
        # Primary MNCS uses OpenAlex world field+year normalized score when available, else cohort score
        mncs_primary = fwci_clean if fwci_clean is not None else mncs_intra

        a["mncs"] = round(mncs_primary, 3)
        a["mncs_intracorpus"] = round(mncs_intra, 3)
        a["mncs_field"] = field

        results.append({
            "key": a.get("key", ""),
            "field": field,
            "year": year,
            "citations": int(cites),
            "field_mean": round(c_mean, 2),
            "mncs": round(mncs_primary, 3),
            "mncs_intracorpus": round(mncs_intra, 3),
            "normalization_source": "openalex_world_fwci" if fwci_clean is not None else "cohort_field_year",
        })

    return results


def compute_pp_top10(articles: list[dict]) -> dict:
    """Calcula PP(top 10%) por campo.

    Proporcion de articulos del corpus que estan en el top 10%
    de citas de su campo (intra-corpus).

    Returns:
        Dict con pp_top10 global y por campo.
    """
    # Group by field
    field_articles: dict[str, list[dict]] = defaultdict(list)
    for a in articles:
        field = _get_field_for_article(a)
        field_articles[field].append(a)

    # Compute per-field top 10% threshold
    field_thresholds: dict[str, float] = {}
    for field, arts in field_articles.items():
        cites = sorted([a.get("cited_by_count", 0) or 0 for a in arts], reverse=True)
        idx_10 = max(0, int(len(cites) * 0.1) - 1)
        field_thresholds[field] = cites[idx_10] if cites else 0

    # Count articles in top 10%
    total_top10 = 0
    per_field = {}

    for field, arts in field_articles.items():
        threshold = field_thresholds[field]
        in_top = sum(1 for a in arts if (a.get("cited_by_count", 0) or 0) >= threshold and threshold > 0)
        per_field[field] = {
            "n_articles": len(arts),
            "threshold": threshold,
            "n_top10": in_top,
            "pp_top10": round(in_top / len(arts), 3) if arts else 0,
        }
        total_top10 += in_top

        # Tag articles
        for a in arts:
            a["in_top10_field"] = (a.get("cited_by_count", 0) or 0) >= threshold and threshold > 0

    return {
        "global_pp_top10": round(total_top10 / len(articles), 3) if articles else 0,
        "total_top10": total_top10,
        "per_field": per_field,
    }


_GLOBAL_SCIMAGO_MEAN_SJR_CACHE: dict[int, float] = {}


def _get_global_scimago_mean_sjr() -> dict[int, float]:
    """Calcula la media global anual de SJR sobre todas las revistas indexadas en SCImago."""
    if _GLOBAL_SCIMAGO_MEAN_SJR_CACHE:
        return _GLOBAL_SCIMAGO_MEAN_SJR_CACHE
    try:
        from src.scimago_enricher import SCImagoEnricher
        enricher = SCImagoEnricher()
        if enricher._load_csv():
            for yr, title_map in enricher._by_year_title.items():
                seen_src: set[str] = set()
                sjrs: list[float] = []
                for rec in title_map.values():
                    sid = rec.get("sourceid") or rec.get("title", "")
                    if sid in seen_src:
                        continue
                    seen_src.add(sid)
                    s = rec.get("sjr")
                    if isinstance(s, (int, float)) and s > 0:
                        sjrs.append(float(s))
                if sjrs:
                    _GLOBAL_SCIMAGO_MEAN_SJR_CACHE[yr] = sum(sjrs) / len(sjrs)
    except Exception:
        pass
    return _GLOBAL_SCIMAGO_MEAN_SJR_CACHE


def compute_mnjs(articles: list[dict]) -> list[dict]:
    """Calcula Mean Normalized Journal Score contra la media global anual de SCImago.

    Evita la tautologia algebraica de dividir por la media intra-corpus no nula
    (que siempre produce MNJS = 1.000). Normaliza el SJR de cada articulo respecto
    a la media global de todas las revistas indexadas en SCImago en el año de
    publicacion (o el año mas cercano disponible), reportando tambien la relacion
    intra-corpus por campo como referencia secundaria.

    Returns:
        Lista de dicts con key, journal, year, sjr, global_mean_sjr, field_mean_sjr, mnjs, mnjs_intracorpus.
    """
    global_sjr_by_year = _get_global_scimago_mean_sjr()
    default_global_sjr = (
        sum(global_sjr_by_year.values()) / len(global_sjr_by_year)
        if global_sjr_by_year
        else 1.0
    )

    field_sjrs: dict[str, list[float]] = defaultdict(list)
    article_data: list[tuple[dict, str, int, float]] = []

    for a in articles:
        field = _get_field_for_article(a)
        try:
            year = int(a.get("publication_year") or 0)
        except (ValueError, TypeError):
            year = 0
        sjr = a.get("scimago", {}).get("sjr", 0) if isinstance(a.get("scimago"), dict) else 0
        if not sjr:
            sjr = a.get("sjr", 0) or 0
        sjr = float(sjr) if sjr else 0.0
        field_sjrs[field].append(sjr)
        article_data.append((a, field, year, sjr))

    field_means: dict[str, float] = {}
    for field, sjrs in field_sjrs.items():
        nonzero = [s for s in sjrs if s > 0]
        field_means[field] = sum(nonzero) / len(nonzero) if nonzero else 0.0

    results = []
    for a, field, year, sjr in article_data:
        if global_sjr_by_year and year > 0:
            nearest_yr = min(global_sjr_by_year.keys(), key=lambda y: abs(y - year))
            g_mean = global_sjr_by_year[nearest_yr]
        else:
            g_mean = default_global_sjr

        f_mean = field_means[field]
        mnjs_global = (sjr / g_mean) if (g_mean > 0 and sjr > 0) else 0.0
        mnjs_intra = (sjr / f_mean) if (f_mean > 0 and sjr > 0) else 0.0

        a["mnjs"] = round(mnjs_global, 3)
        a["mnjs_intracorpus"] = round(mnjs_intra, 3)

        results.append({
            "key": a.get("key", ""),
            "journal": a.get("publication_title", ""),
            "year": year,
            "sjr": round(sjr, 3),
            "global_mean_sjr": round(g_mean, 3),
            "field_mean_sjr": round(f_mean, 3),
            "mnjs": round(mnjs_global, 3),
            "mnjs_intracorpus": round(mnjs_intra, 3),
        })

    return results


def detect_biases(articles: list[dict]) -> dict:
    """Detecta sesgos potenciales en el corpus.

    - Idiomatico: % de articulos no en ingles
    - Geografico: concentracion geografica
    - Tipo documental: distribucion de tipos
    - Cobertura: % sin datos de OpenAlex

    Returns:
        Dict con indicadores de sesgo.
    """
    n = len(articles)
    if n == 0:
        return {}

    # Language bias
    languages = Counter()
    for a in articles:
        lang = a.get("language", "") or a.get("detected_language", "") or ""
        lang = lang.lower().strip()
        if lang:
            languages[lang] += 1
        else:
            languages["unknown"] += 1

    english_count = languages.get("en", 0) + languages.get("english", 0) + languages.get("eng", 0)
    pct_non_english = round((n - english_count) / n * 100, 1)

    # Geographic bias
    countries = Counter()
    for a in articles:
        for auth in a.get("openalex", {}).get("authorships", []):
            for c in auth.get("countries", []):
                if c:
                    countries[c] += 1
            for inst in auth.get("institutions", []):
                cc = inst.get("country", "")
                if cc:
                    countries[cc] += 1

    top_countries = countries.most_common(5)
    total_country_mentions = sum(countries.values())
    geo_concentration = (
        top_countries[0][1] / total_country_mentions
        if top_countries and total_country_mentions > 0
        else 0
    )

    # Document type bias
    doc_types = Counter()
    for a in articles:
        dt = a.get("item_type", "") or a.get("openalex", {}).get("type", "") or "unknown"
        doc_types[dt.lower()] += 1

    # Coverage bias
    n_with_openalex = sum(1 for a in articles if a.get("openalex", {}).get("openalex_id"))
    n_with_concepts = sum(1 for a in articles if a.get("concepts"))
    n_with_sjr = sum(
        1 for a in articles
        if (isinstance(a.get("scimago"), dict) and a["scimago"].get("sjr"))
        or a.get("sjr")
    )

    return {
        "language_bias": {
            "pct_non_english": pct_non_english,
            "distribution": dict(languages.most_common(10)),
            "warning": pct_non_english > 30,
        },
        "geographic_bias": {
            "top_countries": [{"country": c, "count": n} for c, n in top_countries],
            "concentration_top1": round(geo_concentration, 3),
            "warning": geo_concentration > 0.5,
        },
        "document_type_bias": {
            "distribution": dict(doc_types.most_common(10)),
            "dominant_type": doc_types.most_common(1)[0] if doc_types else ("unknown", 0),
        },
        "coverage": {
            "pct_openalex": round(n_with_openalex / n * 100, 1),
            "pct_concepts": round(n_with_concepts / n * 100, 1),
            "pct_sjr": round(n_with_sjr / n * 100, 1),
            "warning": n_with_openalex / n < 0.7,
        },
    }


def run_phase2i(articles: list[dict]) -> dict:
    """Ejecuta Fase 2i: Indicadores Manifiesto de Leiden.

    Returns:
        Dict con todos los indicadores calculados.
    """
    logger.info("=== FASE 2i: Indicadores Leiden ===")
    LEIDEN_DIR.mkdir(parents=True, exist_ok=True)

    # Compute indicators
    mncs_results = compute_mncs(articles)
    pp_results = compute_pp_top10(articles)
    mnjs_results = compute_mnjs(articles)
    bias_results = detect_biases(articles)

    # Summary stats (including zero-citation articles in field-assigned corpus, consistent with Phase 5b)
    mncs_values = [
        r["mncs"]
        for r in mncs_results
        if str(r.get("field", "")).strip().lower() not in {"", "unknown", "none"}
    ]
    mean_mncs = sum(mncs_values) / len(mncs_values) if mncs_values else 0.0

    mncs_intra_values = [
        r["mncs_intracorpus"]
        for r in mncs_results
        if str(r.get("field", "")).strip().lower() not in {"", "unknown", "none"}
    ]
    mean_mncs_intra = sum(mncs_intra_values) / len(mncs_intra_values) if mncs_intra_values else 0.0

    mnjs_values = [r["mnjs"] for r in mnjs_results if r["mnjs"] > 0]
    mean_mnjs = sum(mnjs_values) / len(mnjs_values) if mnjs_values else 0.0

    results = {
        "disclaimer": (
            "NOTA: MNCS utiliza normalizacion mundial por subcampo y año de OpenAlex/CWTS (FWCI, "
            "media mundial = 1.0) con respaldo por cohorte (campo, año). MNJS normaliza el SJR de "
            "cada revista contra la media global anual de todas las revistas indexadas en SCImago."
        ),
        "mncs": {
            "mean": round(mean_mncs, 3),
            "mean_intracorpus_cohort": round(mean_mncs_intra, 3),
            "n_above_1": sum(1 for v in mncs_values if v > 1.0),
            "n_total": len(mncs_values),
            "by_field": {},
        },
        "pp_top10": pp_results,
        "mnjs": {
            "mean": round(mean_mnjs, 3),
            "n_with_sjr": sum(1 for r in mnjs_results if r["sjr"] > 0),
        },
        "biases": bias_results,
    }

    # Aggregate MNCS by field
    from collections import defaultdict as dd
    field_mncs: dict[str, list[float]] = dd(list)
    for r in mncs_results:
        field_mncs[r["field"]].append(r["mncs"])

    for field, values in field_mncs.items():
        results["mncs"]["by_field"][field] = {
            "mean_mncs": round(sum(values) / len(values), 3),
            "n_articles": len(values),
        }

    # Save detailed results
    try:
        # MNCS detail
        mncs_path = LEIDEN_DIR / "mncs_detail.json"
        mncs_path.write_text(
            json.dumps(mncs_results, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # MNJS detail
        mnjs_path = LEIDEN_DIR / "mnjs_detail.json"
        mnjs_path.write_text(
            json.dumps(mnjs_results, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # Full report
        report_path = LEIDEN_DIR / "leiden_report.json"
        report_path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # Markdown summary
        md_lines = [
            "# Leiden-Inspired Indicators Report",
            "",
            f"> {results['disclaimer']}",
            "",
            "## MNCS (Mean Normalized Citation Score)",
            f"- Corpus mean MNCS: {results['mncs']['mean']}",
            f"- Articles above field mean (MNCS > 1): {results['mncs']['n_above_1']}/{results['mncs']['n_total']}",
            "",
            "### By Field",
        ]
        for field, data in sorted(results["mncs"]["by_field"].items(), key=lambda x: -x[1]["n_articles"]):
            md_lines.append(f"- **{field}**: mean MNCS={data['mean_mncs']}, n={data['n_articles']}")

        md_lines.extend([
            "",
            f"## PP(top 10%): {results['pp_top10']['global_pp_top10']}",
            f"- Total in top 10%: {results['pp_top10']['total_top10']}/{len(articles)}",
            "",
            "## MNJS (Mean Normalized Journal Score)",
            f"- Mean MNJS: {results['mnjs']['mean']}",
            f"- Articles with SJR data: {results['mnjs']['n_with_sjr']}/{len(articles)}",
            "",
            "## Bias Detection",
        ])

        biases = results["biases"]
        if biases.get("language_bias", {}).get("warning"):
            md_lines.append(f"- **WARNING** Language bias: {biases['language_bias']['pct_non_english']}% non-English")
        if biases.get("geographic_bias", {}).get("warning"):
            md_lines.append(f"- **WARNING** Geographic concentration: top country = {biases['geographic_bias']['concentration_top1']:.0%}")
        if biases.get("coverage", {}).get("warning"):
            md_lines.append(f"- **WARNING** Low OpenAlex coverage: {biases['coverage']['pct_openalex']}%")

        summary_path = LEIDEN_DIR / "leiden_summary.md"
        summary_path.write_text("\n".join(md_lines), encoding="utf-8")

        logger.info(f"  Leiden results saved to {LEIDEN_DIR}")

    except Exception as e:
        logger.warning(f"Could not save Leiden results: {e}")

    # Log summary
    logger.info(
        f"  MNCS mean={results['mncs']['mean']}, "
        f"PP(top10%)={results['pp_top10']['global_pp_top10']}, "
        f"MNJS mean={results['mnjs']['mean']}"
    )

    warnings = []
    if bias_results.get("language_bias", {}).get("warning"):
        warnings.append("language")
    if bias_results.get("geographic_bias", {}).get("warning"):
        warnings.append("geographic")
    if bias_results.get("coverage", {}).get("warning"):
        warnings.append("coverage")
    if warnings:
        logger.warning(f"  Bias warnings: {', '.join(warnings)}")

    logger.info("Fase 2i completada.")
    return results

run_phase2j = run_phase2i
