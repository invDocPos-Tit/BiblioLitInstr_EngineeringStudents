"""Orquestador CLI del sistema determinista de analisis bibliometrico y cienciometrico."""

import argparse
import importlib
import logging
import os
import subprocess
import sys
from pathlib import Path


def _ensure_project_venv() -> None:
    """Garantiza que el pipeline se ejecute siempre con el .venv del proyecto y nunca con el Python del sistema."""
    project_root = Path(__file__).resolve().parent
    venv_py = (
        project_root / ".venv" / "Scripts" / "python.exe"
        if os.name == "nt"
        else project_root / ".venv" / "bin" / "python"
    )
    if not venv_py.exists():
        return
    try:
        current_py = Path(sys.executable).resolve()
        target_py = venv_py.resolve()
    except OSError:
        return
    if current_py != target_py and os.environ.get("_BIBLIO_VENV_REDIRECTED") != "1":
        sys.stderr.write(
            f"[AVISO] Detectado Python fuera de .venv ({current_py}). "
            f"Re-ejecutando automaticamente con {target_py}...\n"
        )
        sys.stderr.flush()
        env = os.environ.copy()
        env["_BIBLIO_VENV_REDIRECTED"] = "1"
        rc = subprocess.call([str(target_py), *sys.argv], env=env)
        sys.exit(rc)


def _verify_required_dependencies() -> None:
    """Verifica que las dependencias requeridas esten instaladas; falla explicitamente en vez de degradar en silencio."""
    required = {
        "community": "python-louvain",
        "networkx": "networkx",
        "scipy": "scipy",
        "sklearn": "scikit-learn",
        "prince": "prince",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "pandas": "pandas",
        "openpyxl": "openpyxl",
        "wordcloud": "wordcloud",
        "pyvis": "pyvis",
        "plotly": "plotly",
        "adjustText": "adjustText",
    }
    missing = []
    for mod_name, pkg_name in required.items():
        try:
            importlib.import_module(mod_name)
        except ImportError:
            missing.append(f"{pkg_name} (import {mod_name})")
    if missing:
        raise RuntimeError(
            f"Dependencias criticas ausentes en el entorno actual ({sys.executable}): "
            f"{', '.join(missing)}. Ejecute con .venv\\Scripts\\python.exe."
        )


_ensure_project_venv()
_verify_required_dependencies()
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from config import (
    CSV_PATH, DATA_DIR, ENRICHED_PATH, LOGS_DIR, OUTPUT_DIR,
    clean_dirs, clean_output, clean_review, count_review_state_files,
    ensure_dirs,
)
from src.utils import (
    setup_logging, add_file_logging, save_json, load_json,
    normalize_language, canonicalize_corpus_journals, get_canonical_oa_status,
    is_noisy_legacy_concept,
)
from src.provenance import RunRecorder, write_fingerprint, verify_fingerprint


VALID_PHASES = [0, 1, 2, 3, 4, 5, 6, 7]

_LEGACY_ARTICLE_KEYS = (
    "_inductive_idx",
    "inductive_analysis",
    "llm_classifications",
    "citation_contexts",
    "pdf_path",
    "pdf_text",
    "population_sample",
    "screening",
)


def _canonicalize_loaded_articles(articles: list[dict]) -> bool:
    """Asegura normalizacion de idiomas, revistas, topics y Open Access sobre el corpus en memoria."""
    changed = False
    for a in articles:
        for k in _LEGACY_ARTICLE_KEYS:
            if k in a:
                del a[k]
                changed = True
        if not a.get("topics") and a.get("openalex", {}).get("topics"):
            a["topics"] = a["openalex"]["topics"]
            changed = True
        for tag_field in ("automatic_tags", "all_tags"):
            tags = a.get(tag_field)
            if isinstance(tags, list):
                filtered = [t for t in tags if not is_noisy_legacy_concept(str(t))]
                if len(filtered) != len(tags):
                    a[tag_field] = filtered
                    changed = True
        oa_kws = a.get("openalex", {}).get("keywords")
        if isinstance(oa_kws, list):
            filtered_kws = [
                kw for kw in oa_kws
                if not is_noisy_legacy_concept(str(kw.get("keyword", "") if isinstance(kw, dict) else kw))
            ]
            if len(filtered_kws) != len(oa_kws):
                a["openalex"]["keywords"] = filtered_kws
                changed = True
        raw_lang = a.get("language", "")
        norm_lang = normalize_language(raw_lang)
        if norm_lang and norm_lang != raw_lang:
            a["language"] = norm_lang
            changed = True
        oa = a.get("open_access")
        if not isinstance(oa, dict):
            oa = {}
            a["open_access"] = oa
            changed = True
        canon_status = get_canonical_oa_status(a)
        if oa.get("status") != canon_status:
            oa["status"] = canon_status
            oa["is_oa"] = canon_status != "closed"
            changed = True

        # Reparar corrupcion de codificacion ('?' intrapalabra en autores, titulos y abstracts) de forma 100% agnostica al corpus
        import re as _re
        for _txt_field in ("title", "abstract"):
            _val = a.get(_txt_field)
            if isinstance(_val, str) and "?" in _val:
                _clean = _re.sub(r"(?<=[A-Za-z])\?s\b", "'s", _val)
                _clean = _re.sub(r"(?<=[A-Za-z]s)\?(?=\s+[a-z])", "'", _clean)
                _clean = _re.sub(r"(?<=[a-z])\?(?=[a-z])", "-", _clean)
                if _clean != _val:
                    a[_txt_field] = _clean
                    changed = True

    # Construir indice dinamico de apellidos limpios desde OpenAlex authorships y autores sin '?' del propio corpus
    import re as _re
    _clean_surnames: set[str] = set()
    for _art in articles:
        for _au in (_art.get("authors") or []):
            _s = str(_au).split(",")[0].strip()
            if _s and "?" not in _s and len(_s) >= 3:
                _clean_surnames.add(_s)
        for _oa_au in ((_art.get("openalex") or {}).get("authorships") or []):
            if isinstance(_oa_au, dict):
                _dn = (_oa_au.get("author") or {}).get("display_name") or _oa_au.get("display_name") or ""
                if _dn and "?" not in _dn:
                    _parts = _dn.strip().split()
                    if _parts:
                        _clean_surnames.add(_parts[-1].strip(",."))

    for a in articles:
        _authors = a.get("authors")
        if isinstance(_authors, list) and any("?" in str(au) for au in _authors):
            _oa_auths = (a.get("openalex") or {}).get("authorships") or []
            _new_authors = []
            for _idx, _au in enumerate(_authors):
                _au_str = str(_au).strip()
                if "?" in _au_str:
                    _repaired = None
                    # 1. Emparejar por posicion en OpenAlex authorships del mismo articulo
                    if _idx < len(_oa_auths) and isinstance(_oa_auths[_idx], dict):
                        _dn = (_oa_auths[_idx].get("author") or {}).get("display_name") or _oa_auths[_idx].get("display_name") or ""
                        if _dn and "?" not in _dn:
                            _parts = _dn.strip().split()
                            if len(_parts) >= 2:
                                _repaired = f"{_parts[-1]}, {'.'.join(p[0] for p in _parts[:-1] if p)}."
                    # 2. Emparejar el patron con '?' contra cualquier apellido limpio del corpus/OpenAlex
                    if not _repaired and "," in _au_str:
                        _sur_bad, _rest = _au_str.split(",", 1)
                        _pat_str = "^" + _re.escape(_sur_bad.strip()).replace(r"\?", ".{1,2}") + "$"
                        try:
                            _pat = _re.compile(_pat_str, _re.IGNORECASE)
                            for _cand in sorted(_clean_surnames):
                                if _pat.match(_cand):
                                    _repaired = f"{_cand},{_rest}"
                                    break
                        except _re.error:
                            pass
                    if _repaired:
                        _au_str = _repaired
                _new_authors.append(_au_str)
            if _new_authors != _authors:
                a["authors"] = _new_authors
                changed = True

    # Si hay articulos sin DOI, publication_title o ISSN pero existen caches de OpenAlex/Crossref, rescatarlos (0 API calls)
    if any(not a.get("doi") or not a.get("publication_title") or not a.get("issn") for a in articles):
        try:
            from src.metadata_completer import MetadataCompleter
            completer = MetadataCompleter()
            stage_a_stats = completer.stage_a(articles)
            if isinstance(stage_a_stats, dict) and any(v > 0 for v in stage_a_stats.values()):
                changed = True
        except Exception:
            pass

    # Integrar archivos locales opcionales en data/ (CitationScopus.csv, CitationWos.xlsx/csv, scopusReferences.ris, scopusCitadopor.ris)
    if (
        not any("scopus_citations_by_year" in a for a in articles)
        or not any("wos_citations_by_year" in a for a in articles)
    ):
        try:
            from src.local_citation_enricher import run_local_citation_enrichment
            local_stats = run_local_citation_enrichment(articles)
            if any(v > 0 for v in local_stats.values()):
                changed = True
        except Exception as e:
            logging.getLogger("bibliometrico.main").warning(
                f"Local citation enrichment warning: {e}"
            )

    # Si el CSV original no traia columna Keywords, poblar manual_tags desde all_tags / OpenAlex keywords
    for a in articles:
        if not a.get("manual_tags"):
            kws = list(a.get("all_tags") or [])
            if not kws:
                for kw_obj in (a.get("openalex") or {}).get("keywords", []):
                    kw_txt = kw_obj.get("keyword", "") if isinstance(kw_obj, dict) else str(kw_obj)
                    if kw_txt:
                        kws.append(kw_txt)
            if kws:
                a["manual_tags"] = list(dict.fromkeys(kws))
                changed = True

    # Palabras clave de autor en campo propio (vocabulario comun, opcion A). Checkpoints anteriores no lo traen.
    if any("author_keywords" not in a for a in articles) and CSV_PATH.exists():
        try:
            import pandas as pd
            from src.data_loader import load_csv, parse_tags
            df_ak = load_csv(CSV_PATH)
            if "__author_keywords" in df_ak.columns:
                por_key, por_doi = {}, {}
                for _, row in df_ak.iterrows():
                    val = row.get("__author_keywords")
                    tags = parse_tags(val) if pd.notna(val) else []
                    k = str(row.get("Key", "")).strip()
                    d = str(row.get("DOI", "")).strip().lower()
                    if k:
                        por_key[k] = tags
                    if d:
                        por_doi[d] = tags
                for a in articles:
                    if "author_keywords" in a:
                        continue
                    tags = por_key.get(a.get("key", ""))
                    if tags is None:
                        tags = por_doi.get((a.get("doi") or "").strip().lower(), [])
                    a["author_keywords"] = tags
                    changed = True
        except Exception as e:
            logging.getLogger("bibliometrico.main").warning(f"No se pudieron completar author_keywords: {e}")

    n_unified = canonicalize_corpus_journals(articles)
    if n_unified > 0:
        changed = True

    # Backfill merged_from / richness_score desde consolidado.csv y parsear referenced_works_raw si faltan
    if any(not a.get("merged_from") for a in articles) and CSV_PATH.exists():
        try:
            from src.data_loader import load_csv, _parse_structured_reference
            import pandas as pd
            df_csv = load_csv(CSV_PATH)
            by_key = {}
            by_doi = {}
            for _, row in df_csv.iterrows():
                k = str(row.get("Key", "")).strip()
                d = str(row.get("DOI", "")).strip().lower()
                mf = str(row.get("Merged From", "")).strip() if pd.notna(row.get("Merged From")) else ""
                rs = float(row["Richness Score"]) if pd.notna(row.get("Richness Score")) else None
                if k:
                    by_key[k] = (mf, rs)
                if d:
                    by_doi[d] = (mf, rs)
            for a in articles:
                hit = by_key.get(a.get("key", "")) or by_doi.get((a.get("doi") or "").strip().lower())
                if hit:
                    mf, rs = hit
                    if mf and a.get("merged_from") != mf:
                        a["merged_from"] = mf
                        changed = True
                    if rs is not None and a.get("richness_score") != rs:
                        a["richness_score"] = rs
                        changed = True
                if a.get("referenced_works_raw") and not a.get("referenced_works_detailed"):
                    for ref_str in a["referenced_works_raw"]:
                        parsed_ref = _parse_structured_reference(ref_str)
                        if parsed_ref:
                            a.setdefault("referenced_works_detailed", []).append(parsed_ref)
                            if parsed_ref["id"] not in a.get("referenced_works", []):
                                a.setdefault("referenced_works", []).append(parsed_ref["id"])
                    changed = True
        except Exception:
            pass

    # Si existen CSVs en scimago/ y el corpus aun no tiene enriquecimiento longitudinal SCImago, aplicarlo
    has_any_scimago = any(isinstance(a.get("scimago"), dict) and bool(a["scimago"]) for a in articles)
    needs_longitudinal_scimago = (
        not has_any_scimago
        or any(
            isinstance(a.get("scimago"), dict) and bool(a["scimago"]) and "sjr_by_year" not in a["scimago"]
            for a in articles
        )
    )
    if needs_longitudinal_scimago:
        from src.scimago_enricher import SCImagoEnricher
        enricher = SCImagoEnricher()
        if enricher._load_csv():
            for a in articles:
                old_sjr = a.get("sjr")
                old_q = a.get("sjr_quartile")
                old_scimago = dict(a["scimago"]) if isinstance(a.get("scimago"), dict) else None
                enricher.enrich_article(a)
                if (
                    a.get("sjr") != old_sjr
                    or a.get("sjr_quartile") != old_q
                    or a.get("scimago") != old_scimago
                ):
                    changed = True
                oa = a.get("open_access")
                if isinstance(oa, dict):
                    canon_status = get_canonical_oa_status(a)
                    if oa.get("status") != canon_status:
                        oa["status"] = canon_status
                        oa["is_oa"] = canon_status != "closed"
                        changed = True

    return changed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sistema de Enriquecimiento y Analisis Bibliometrico de Frontera"
    )
    parser.add_argument(
        "--phases", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7],
        help="Fases consecutivas a ejecutar (0-7). Ejemplo: --phases 0 1 2 3 4 5 6 7 o --phases 2 3 4 5 6 7",
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Ruta alternativa al CSV de entrada (default: consolidado.csv)",
    )
    parser.add_argument(
        "--skip-bert", action="store_true",
        help="Omitir analisis BERT/SciBERT (Fase 2c)",
    )
    parser.add_argument(
        "--bert-model", type=str, default=None,
        help="Modelo BERT a usar (default: allenai/scibert_scivocab_uncased)",
    )
    parser.add_argument(
        "--wordcloud-sources", nargs="+", type=str,
        default=["abstracts", "titles", "author_keywords", "keywords_plus"],
        help="Fuentes para wordclouds: abstracts titles author_keywords keywords_plus",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Nivel de logging",
    )
    parser.add_argument(
        "--run-id", type=str, default=None,
        help="Identificador de corrida para registrar metadatos en el manifiesto",
    )
    # --- Disambiguation (Phase 0b) ---
    parser.add_argument(
        "--skip-disambiguation", action="store_true",
        help="Omitir desambiguacion de autores/instituciones (Fase 0b)",
    )
    # --- Enrichment Sources (Phase 1) ---
    parser.add_argument(
        "--skip-crossref", action="store_true",
        help="Omitir enriquecimiento Crossref (Fase 1b)",
    )
    parser.add_argument(
        "--skip-s2", action="store_true",
        help="Omitir enriquecimiento Semantic Scholar (Fase 1c)",
    )
    parser.add_argument(
        "--openalex-rate", type=float, default=None,
        help="Limite de tasa (req/s) para OpenAlex (Fase 1/1d/1i/1j)",
    )
    parser.add_argument(
        "--skip-openalex-meta", action="store_true",
        help="Omitir enriquecimiento OpenAlex Meta - Sources/Authors (Fase 1d)",
    )
    parser.add_argument(
        "--skip-scimago", action="store_true",
        help="Omitir enriquecimiento SCImago SJR (Fase 1e)",
    )
    parser.add_argument(
        "--skip-altmetric", action="store_true", default=True,
        help="Omitir enriquecimiento Altmetric (Fase 1f). Desactivado por defecto (requiere API key)",
    )
    parser.add_argument(
        "--enable-altmetric", action="store_true",
        help="Activar enriquecimiento Altmetric (requiere ALTMETRIC_API_KEY)",
    )
    parser.add_argument(
        "--skip-opencitations", action="store_true",
        help="Omitir enriquecimiento OpenCitations (Fase 1g)",
    )
    parser.add_argument(
        "--skip-unpaywall", action="store_true",
        help="Omitir enriquecimiento Unpaywall (Fase 1h)",
    )
    parser.add_argument(
        "--skip-ref-resolve", action="store_true",
        help="Omitir resolucion mejorada de referencias en OpenAlex (Fase 1i)",
    )
    parser.add_argument(
        "--skip-metadata-complete", action="store_true",
        help="Omitir completado de metadatos desde caches (Fase 1j)",
    )
    parser.add_argument(
        "--skip-gap-filler", action="store_true",
        help="Omitir gap-filler multi-fuente (Fase 1k: Europe PMC, DOAJ, NLM, langdetect)",
    )
    parser.add_argument(
        "--skip-events", action="store_true",
        help="Omitir enriquecimiento Crossref Event Data (Fase 1l: impacto social)",
    )
    # --- Phase 2 subphases ---
    parser.add_argument(
        "--skip-quality", action="store_true",
        help="Omitir evaluacion estructural de calidad de abstracts (Fase 2d)",
    )
    parser.add_argument(
        "--quality-threshold", type=float, default=50.0,
        help="Umbral minimo de calidad de abstract (0-100). Default: 50.0",
    )
    parser.add_argument(
        "--skip-sota", action="store_true",
        help="Omitir analisis algoritmico SOTA y vacios de conocimiento (Fase 2f)",
    )
    parser.add_argument(
        "--skip-spectroscopy", action="store_true",
        help="Omitir espectroscopia de referencias RPYS (Fase 2g)",
    )
    parser.add_argument(
        "--skip-citespace", action="store_true",
        help="Omitir analisis CiteSpace (Fase 2h)",
    )
    parser.add_argument(
        "--citespace-burst-s", type=float, default=2.0,
        help="Costo de transicion de estado Kleinberg (default: 2.0)",
    )
    parser.add_argument(
        "--citespace-burst-gamma", type=float, default=1.0,
        help="Parametro de dificultad Kleinberg (default: 1.0)",
    )
    parser.add_argument(
        "--skip-leiden", action="store_true",
        help="Omitir indicadores CWTS Leiden (Fase 2i)",
    )
    # --- Phase 3 options ---
    parser.add_argument(
        "--skip-ref-resolution", action="store_true",
        help="Fase 3: omitir resolucion de etiquetas de referencias",
    )
    parser.add_argument(
        "--resolve-rate", type=float, default=None,
        help="Fase 3: limite de tasa (req/s) para resolver etiquetas de referencias",
    )
    parser.add_argument(
        "--resolve-batch-size", type=int, default=25,
        help="Fase 3: tamano de lote para resolver etiquetas de referencias",
    )
    parser.add_argument(
        "--resolve-backoff", type=float, default=1.0,
        help="Fase 3: backoff base (s) entre lotes fallidos",
    )
    # --- Stratified Analysis ---
    parser.add_argument(
        "--skip-stratified", action="store_true",
        help="Omitir analisis estratificado por tipo de documento",
    )
    parser.add_argument(
        "--min-stratum-size", type=int, default=15,
        help="Minimo de articulos por estrato (default: 15)",
    )
    # --- Phase 4 options ---
    parser.add_argument(
        "--skip-oa-export", action="store_true",
        help="Omitir exportacion de corpus OA (Fase 4b)",
    )
    # --- Bootstrap CIs (Phase 5b) ---
    parser.add_argument(
        "--skip-bootstrap-ci", action="store_true",
        help="Omitir bootstrap de IC95 sobre indicadores cabecera (Fase 5b)",
    )
    parser.add_argument(
        "--bootstrap-B", type=int, default=10000,
        help="Replicas del bootstrap no parametrico (default: 10000)",
    )
    parser.add_argument(
        "--bootstrap-seed", type=int, default=42,
        help="Seed del bootstrap (default: 42, reproducibilidad bit-a-bit)",
    )
    # --- Knowledge Map (Phase 7) ---
    parser.add_argument(
        "--permitir-insumos-incompletos", action="store_true",
        help="Continuar aunque falte por completo un insumo critico (palabras clave de autor, resumen o ano). "
             "Por defecto la corrida se detiene y explica que falta",
    )
    parser.add_argument(
        "--enable-rdm", action="store_true",
        help="Correr la Fase 6 (RDM). Desactivada por defecto: indice propio sin validacion externa; "
             "sus resultados no entran a la Fase 7",
    )
    parser.add_argument(
        "--skip-knowledge-map", action="store_true",
        help="Omitir sintesis de Knowledge Map (Fase 7)",
    )
    parser.add_argument(
        "--kmap-top-n", type=int, default=20,
        help="Top N findings para sintesis del Knowledge Map (default: 20)",
    )
    parser.add_argument(
        "--kmap-impact-weights", type=str, default=None,
        help="Pesos del impact score Fase 7 como 'sig,rec,tri' "
             "(renormalizados a suma 1). Default: 0.4,0.2,0.4",
    )
    # --- Limpieza ---
    parser.add_argument(
        "--clean", action="store_true",
        help="Limpiar data/ y output/ antes de ejecutar (pide confirmacion)",
    )
    parser.add_argument(
        "--clean-yes", action="store_true",
        help="Limpiar data/ y output/ sin confirmacion (uso en scripts)",
    )
    parser.add_argument(
        "--clean-output", action="store_true",
        help="Limpiar solo output/ (preserva data/caches) y salir sin ejecutar fases",
    )
    parser.add_argument(
        "--new-review", action="store_true",
        help="Reset entre revisiones: borra data/ y output/ EXCEPTO "
             "data/scimagojr.csv y consolidado.csv. Pide confirmacion y sale.",
    )
    parser.add_argument(
        "--new-review-yes", action="store_true",
        help="Igual que --new-review pero sin confirmacion (uso en scripts).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger("bibliometrico.main")

    # Procedencia: captura args completos + ts unico (log + manifiesto).
    recorder = RunRecorder(args)

    # Validacion de fases (0 a 7 consecutivas)
    valid_phases = set(VALID_PHASES)
    invalid = [p for p in args.phases if p not in valid_phases]
    if invalid:
        logger.error(f"Fases invalidas: {invalid}. Fases validas (consecutivas): {VALID_PHASES}")
        sys.exit(2)

    # Asegurar estructura de directorios
    ensure_dirs()

    # Reset entre revisiones: borra TODO el estado generado (data/ + output/)
    # salvo los inputs portables.
    if args.new_review or args.new_review_yes:
        n_files = count_review_state_files()
        if n_files == 0:
            logger.info("No hay estado de revision que borrar (ya esta limpio).")
        else:
            if not args.new_review_yes:
                resp = input(
                    f"Se eliminaran {n_files} archivos de data/ y output/ "
                    "(incl. data/enriched_articles.json y todos los caches). "
                    "Se preservan consolidado.csv, data/knowledge_base.json y los "
                    ".csv/.xlsx/.ris de la raiz de data/ (revisa que sean de la revision nueva). "
                    "Continuar? [s/N] "
                )
                if resp.strip().lower() != "s":
                    logger.info("Reset de revision cancelado por el usuario.")
                    sys.exit(0)
            clean_review()
        sys.exit(0)

    # Limpieza de SOLO output/ y salir (preserva data/ y sus caches)
    if args.clean_output:
        n_files = sum(
            1 for f in OUTPUT_DIR.rglob("*") if f.is_file()
        ) if OUTPUT_DIR.exists() else 0
        if n_files == 0:
            logger.info("output/ ya esta vacio, nada que limpiar.")
        else:
            removed = clean_output()
            logger.info(
                f"output/ limpiado ({removed} archivos eliminados); "
                "data/ y sus caches se conservan."
            )
        sys.exit(0)

    # Limpieza de directorios si se solicita
    if args.clean or args.clean_yes:
        n_files = sum(1 for d in (DATA_DIR, OUTPUT_DIR) if d.exists()
                      for f in d.rglob("*") if f.is_file())
        if n_files == 0:
            logger.info("Directorios ya vacios, nada que limpiar.")
        else:
            if not args.clean_yes:
                resp = input(
                    f"Se eliminaran {n_files} archivos en data/ y output/. "
                    "Continuar? [s/N] "
                )
                if resp.strip().lower() != "s":
                    logger.info("Limpieza cancelada por el usuario.")
                    sys.exit(0)
            removed = clean_dirs()
            logger.info(f"Directorios limpiados ({removed} elementos eliminados).")

    # Log persistido por corrida.
    add_file_logging(LOGS_DIR / f"run_{recorder.ts}.log", args.log_level)

    # Entrada de corpus efectiva (para huella agnostica + manifiesto).
    csv_path = Path(args.csv) if args.csv else CSV_PATH

    logger.info("=" * 60)
    logger.info("Sistema de Analisis Bibliometrico de Frontera (100% Determinista)")
    logger.info(f"Fases seleccionadas: {args.phases}")
    logger.info("=" * 60)

    articles = None
    networks = {}

    try:
        # Fase 0: Carga del CSV (con deteccion automatica de cambio de revision en consolidado.csv)
        if 0 in args.phases:
            status, detail = verify_fingerprint(csv_path)
            if status == "mismatch":
                logger.warning(
                    "Se detecto un nuevo consolidado.csv (%s). "
                    "Limpiando estado derivado y salidas de la revision anterior...",
                    detail,
                )
                clean_review()
            with recorder.phase("0: Carga CSV", swallow=False):
                from src.data_loader import run_phase0
                if args.csv:
                    try:
                        articles = run_phase0(csv_path=csv_path)
                    except Exception as e:
                        logger.error(f"Error cargando CSV {args.csv}: {e}")
                        sys.exit(1)
                else:
                    articles = run_phase0()
                _canonicalize_loaded_articles(articles)
                save_json(articles, ENRICHED_PATH)
                write_fingerprint(csv_path)

        # Cargar checkpoint si no se ejecuto fase 0
        if articles is None:
            articles = load_json(ENRICHED_PATH)
            if articles is None:
                logger.error(
                    "No se encontro checkpoint. Ejecute primero --phases 0"
                )
                sys.exit(1)
            logger.info(f"Checkpoint cargado: {len(articles)} articulos")

            status, detail = verify_fingerprint(csv_path)
            if status == "mismatch":
                logger.error(
                    "El checkpoint pertenece a OTRA revision (%s). "
                    "Ejecuta: python main.py --new-review  antes de continuar.",
                    detail,
                )
                sys.exit(1)
            if status == "absent":
                write_fingerprint(csv_path)
                logger.info(
                    "Sin huella previa: checkpoint sellado con corpus vigente "
                    "(auto-sanado). Si este checkpoint NO es de la revision "
                    "vigente, ejecuta: python main.py --new-review"
                )

            if _canonicalize_loaded_articles(articles):
                save_json(articles, ENRICHED_PATH)
                logger.info("Metadatos del checkpoint canonicalizados (idiomas, revistas, OA).")

        # Fase 0b: Desambiguacion de Autores/Instituciones
        if 0 in args.phases and not args.skip_disambiguation:
            with recorder.phase("0b: Desambiguacion"):
                from src.name_disambiguation import run_phase0b
                articles = run_phase0b(articles)
                save_json(articles, ENRICHED_PATH)

        # Re-key: migrar DOI-based keys a formato Apellido+Año legible
        with recorder.phase("rekey", swallow=False):
            from src.data_loader import rekey_articles, _is_doi_key, migrate_keys_in_outputs
            needs_rekey = any(
                not a.get("key") or _is_doi_key(a.get("key", ""))
                for a in articles
            )
            if needs_rekey:
                key_map = rekey_articles(articles)
                if key_map:
                    save_json(articles, ENRICHED_PATH)
                    logger.info(f"Checkpoint actualizado con {len(key_map)} keys migradas")
                    migrate_keys_in_outputs(key_map)

        # Fase 1: Enriquecimiento OpenAlex
        if 1 in args.phases:
            with recorder.phase("1: OpenAlex", swallow=False):
                from src.openalex_enricher import run_phase1
                articles = run_phase1(articles, rate_override=args.openalex_rate)

        # Fase 1b: Crossref
        if 1 in args.phases and not args.skip_crossref:
            with recorder.phase("1b: Crossref"):
                from src.crossref_enricher import run_phase1b
                articles = run_phase1b(articles)

        # Fase 1c: Semantic Scholar
        if 1 in args.phases and not args.skip_s2:
            with recorder.phase("1c: Semantic Scholar"):
                from src.semantic_scholar_enricher import run_phase1c
                articles = run_phase1c(articles)

        # Fase 1d: OpenAlex Meta (Sources + Authors)
        if 1 in args.phases and not args.skip_openalex_meta:
            with recorder.phase("1d: OpenAlex Meta"):
                from src.openalex_meta_enricher import run_phase1d
                articles = run_phase1d(articles, rate_override=args.openalex_rate)

        # Fase 1e: SCImago SJR (CSV local)
        if 1 in args.phases and not args.skip_scimago:
            with recorder.phase("1e: SCImago SJR"):
                from src.scimago_enricher import run_phase1e
                articles = run_phase1e(articles)

        # Fase 1f: Altmetric (impacto social) — desactivado por defecto
        if 1 in args.phases and args.enable_altmetric:
            with recorder.phase("1f: Altmetric"):
                from src.altmetric_enricher import run_phase1f
                articles = run_phase1f(articles)

        # Fase 1g: OpenCitations
        if 1 in args.phases and not args.skip_opencitations:
            with recorder.phase("1g: OpenCitations"):
                from src.opencitations_enricher import run_phase1g
                articles = run_phase1g(articles)

        # Fase 1h: Unpaywall (OA detallado)
        if 1 in args.phases and not args.skip_unpaywall:
            with recorder.phase("1h: Unpaywall"):
                from src.unpaywall_enricher import run_phase1h
                articles = run_phase1h(articles)

        # Fase 1i: Enhanced Reference Resolution (topics/concepts/keywords)
        if 1 in args.phases and not args.skip_ref_resolve:
            with recorder.phase("1i: Reference resolution"):
                from src.reference_resolver import run_phase1i
                articles = run_phase1i(articles, rate_override=args.openalex_rate)

        # Fase 1j: Metadata Completion (from caches + minimal API)
        if 1 in args.phases and not args.skip_metadata_complete:
            with recorder.phase("1j: Metadata completion"):
                from src.metadata_completer import run_phase1j
                articles = run_phase1j(articles, rate_override=args.openalex_rate)

        # Fase 1k: Metadata Gap-Filler (Europe PMC, DOAJ, NLM, langdetect)
        if 1 in args.phases and not args.skip_gap_filler:
            with recorder.phase("1k: Metadata gap-filler"):
                from src.metadata_gap_filler import run_phase1k
                articles = run_phase1k(articles)

        # Fase 1l: Crossref Event Data (social impact)
        if 1 in args.phases and not args.skip_events:
            with recorder.phase("1l: Crossref Event Data"):
                from src.crossref_events_enricher import run_phase1l
                articles = run_phase1l(articles)

        # Modelo del corpus (una vez por corrida, despues del enriquecimiento): palabras clave de autor
        # normalizadas, metodos y vacios declarados, paises e instituciones, siglas, IDF. Todas las
        # fases 2-7 leen de aqui (src/modelo_corpus.py) en lugar de extraer por su cuenta.
        if any(p >= 2 for p in args.phases):
            from src.modelo_corpus import build_corpus_model, InsumoIncompletoError
            try:
                with recorder.phase("Modelo del corpus", swallow=False):
                    build_corpus_model(articles, allow_incomplete=args.permitir_insumos_incompletos)
            except InsumoIncompletoError as e:
                logger.error(f"Corrida detenida: {e}")
                sys.exit(3)

        # Fase 2: Analisis Descriptivo y Subfases Bibliometricas
        if 2 in args.phases:
            with recorder.phase("2: Descriptivo", swallow=False):
                from src.descriptive_analysis import run_phase2
                run_phase2(articles)

            # EDA Report
            with recorder.phase("2: EDA report"):
                from src.eda_report import run_eda_report
                run_eda_report(articles)

        # Fase 2b: Analisis de Abstracts
        if 2 in args.phases:
            with recorder.phase("2b: Abstracts", swallow=False):
                from src.abstract_analysis import run_phase2b
                run_phase2b(articles)

            # WordClouds
            with recorder.phase("2b: WordClouds"):
                from src.wordcloud_generator import run_wordclouds
                run_wordclouds(articles, sources=args.wordcloud_sources)

            save_json(articles, ENRICHED_PATH)

        # Fase 2c: BERT/SciBERT Analysis (optional)
        if 2 in args.phases:
            with recorder.phase("2c: BERT/SciBERT"):
                from src.bert_topic_analysis import run_bert_analysis
                bert_kwargs = {"skip": args.skip_bert}
                if args.bert_model:
                    bert_kwargs["model_name"] = args.bert_model
                run_bert_analysis(articles, **bert_kwargs)
                save_json(articles, ENRICHED_PATH)

        # Fase 2d: Evaluacion Estructural de Calidad de Abstracts
        if 2 in args.phases and not args.skip_quality:
            with recorder.phase("2d: Calidad de abstracts"):
                from src.abstract_quality_assessment import run_phase2d_quality
                articles, high_quality_articles, quality_stats = run_phase2d_quality(
                    articles,
                    threshold=args.quality_threshold,
                )
                save_json(articles, ENRICHED_PATH)

                logger.info(
                    f"Calidad de abstracts: {quality_stats['above_threshold']}/{quality_stats['total']} "
                    f"superan umbral {args.quality_threshold}"
                )

                try:
                    from src.quality_report_generator import run_quality_report
                    quality_files = run_quality_report(articles, threshold=args.quality_threshold)
                    logger.info(f"Quality report: {len(quality_files)} files generated")
                except Exception as e:
                    logger.warning(f"Quality report generation failed: {e}")

        # Fase 2e: Indicadores Bibliometricos Suplementarios
        if 2 in args.phases:
            with recorder.phase("2e: Indicadores suplementarios"):
                from src.supplementary_indicators import run_phase2e
                run_phase2e(articles)

        # Fase 2f: SOTA + Vacios de Conocimiento (Algoritmico)
        if 2 in args.phases and not args.skip_sota:
            with recorder.phase("2f: SOTA/gaps"):
                from src.sota_gap_analysis import run_phase2f_sota
                sota_results = run_phase2f_sota(articles)
                logger.info(
                    f"SOTA analysis: {sota_results.get('taxonomy', {}).get('structural', {}).get('count', 0)} structural gaps, "
                    f"{sota_results.get('frontiers_count', 0)} frontiers"
                )

        # Fase 2g: Espectroscopia de Referencias (RPYS)
        if 2 in args.phases and not args.skip_spectroscopy:
            with recorder.phase("2g: Espectroscopia"):
                from src.reference_spectroscopy import run_phase2g
                spectro_results = run_phase2g(articles)
                logger.info(
                    f"Spectroscopy: {spectro_results.get('n_concepts', 0)} concepts, "
                    f"{spectro_results.get('n_refs_analyzed', 0)} refs analyzed"
                )

        # Fase 2h: CiteSpace Analysis Suite
        if 2 in args.phases and not args.skip_citespace:
            with recorder.phase("2h: CiteSpace"):
                from src.citespace_analysis import run_phase2h
                citespace_results = run_phase2h(
                    articles, networks=networks,
                    burst_s=args.citespace_burst_s,
                    burst_gamma=args.citespace_burst_gamma,
                )
                logger.info(
                    f"CiteSpace: {citespace_results.get('n_bursts', 0)} bursts, "
                    f"{citespace_results.get('n_pivot_nodes', 0)} pivot nodes, "
                    f"{citespace_results.get('n_clusters', 0)} labeled clusters"
                )

        # Fase 2i: Indicadores CWTS Leiden
        if 2 in args.phases and not args.skip_leiden:
            with recorder.phase("2i: Leiden"):
                from src.leiden_indicators import run_phase2i
                leiden_results = run_phase2i(articles)
                logger.info(
                    f"Leiden: MNCS={leiden_results.get('mncs', {}).get('mean', 0)}, "
                    f"PP(top10%)={leiden_results.get('pp_top10', {}).get('global_pp_top10', 0)}"
                )

        # Fase 3: Analisis de Redes
        if 3 in args.phases:
            with recorder.phase("3: Redes", swallow=False):
                from src.network_analysis import run_phase3
                networks = run_phase3(
                    articles,
                    resolve_ref_labels=not args.skip_ref_resolution,
                    resolve_rate_limit=args.resolve_rate,
                    resolve_batch_size=args.resolve_batch_size,
                    resolve_backoff=args.resolve_backoff,
                )

        # Fase 3b: Analisis Tematico Estrategico (Callon, evolucion, SCP/MCP)
        if 3 in args.phases:
            with recorder.phase("3b: Tematico estrategico"):
                from src.strategic_thematic_analysis import run_phase3b
                run_phase3b(articles, networks=networks)

        # Analisis Estratificado por Tipo de Documento
        if (2 in args.phases or 3 in args.phases) and not args.skip_stratified:
            with recorder.phase("Estratificado"):
                from src.stratified_analysis import run_stratified_analysis
                strat_results = run_stratified_analysis(
                    articles,
                    min_stratum_size=args.min_stratum_size,
                )
                if not strat_results.get("skipped"):
                    logger.info(
                        f"Stratified analysis: {strat_results['strata_count']} strata analyzed"
                    )

        # Validacion de datos antes de exportar
        if 4 in args.phases:
            try:
                from src.metadata_completer import validate_export_readiness
                validation = validate_export_readiness(articles)
                if validation["warnings"]:
                    for w in validation["warnings"]:
                        logger.warning(f"Export validation: {w}")
                if validation["info"]:
                    for i in validation["info"]:
                        logger.info(f"Export validation: {i}")
                logger.info(f"Export validation: {validation['summary']}")
            except ImportError:
                pass
            except Exception as e:
                logger.warning(f"Export validation failed: {e}")

        # Fase 4: Exportacion + Visualizaciones Avanzadas
        if 4 in args.phases:
            with recorder.phase("4: Exportacion", swallow=False):
                from src.export_integration import run_phase4
                run_phase4(articles)

            if not networks:
                logger.info("Rebuilding networks for Phase 4 outputs...")
                with recorder.phase("4: Rebuild redes"):
                    from src.network_analysis import (
                        build_bibliographic_coupling_network,
                        build_cocitation_network,
                        build_keyword_cooccurrence_network,
                        build_coauthorship_network,
                        build_bibliographic_coupling_by_source,
                        build_cocitation_by_source,
                        detect_communities,
                    )
                    G_coupling = build_bibliographic_coupling_network(articles)
                    detect_communities(G_coupling)
                    G_cocit = build_cocitation_network(articles, min_cocitations=2)
                    detect_communities(G_cocit)
                    G_kw = build_keyword_cooccurrence_network(articles, min_cooccurrence=2)
                    detect_communities(G_kw)
                    networks = {
                        "bibliographic_coupling": G_coupling,
                        "cocitation": G_cocit,
                        "keyword_cooccurrence": G_kw,
                    }
                    for level in ("authors", "institutions", "countries"):
                        G_ca = build_coauthorship_network(articles, level=level)
                        detect_communities(G_ca)
                        networks[f"coauthorship_{level}"] = G_ca
                    for level in ("journals", "institutions", "countries"):
                        G_cs = build_bibliographic_coupling_by_source(articles, level=level)
                        if G_cs.number_of_nodes() > 2:
                            detect_communities(G_cs)
                            networks[f"coupling_{level}"] = G_cs
                    G_cocit_j = build_cocitation_by_source(articles, min_cocitations=2)
                    if G_cocit_j.number_of_nodes() > 2:
                        detect_communities(G_cocit_j)
                        networks["cocitation_journals"] = G_cocit_j
                    logger.info(f"Networks rebuilt: {len(networks)} graphs available")

            # Visualizaciones avanzadas Plotly
            with recorder.phase("4: Visualizaciones avanzadas"):
                from src.advanced_visualizations import run_advanced_visualizations
                run_advanced_visualizations(articles, OUTPUT_DIR, networks=networks)

            # Interactive charts (bar, heatmap, sankey, reference graph, RPYS, etc.)
            with recorder.phase("4: Interactive charts"):
                from src.interactive_charts import run_interactive_charts
                run_interactive_charts(articles)

            # VOSviewer native export (.map + .net files)
            with recorder.phase("4: VOSviewer export"):
                from src.vosviewer_export import run_vosviewer_export
                run_vosviewer_export(articles, networks)

            # VOSviewer static high-impact visualizations
            with recorder.phase("4: VOSviewer static"):
                from src.vosviewer_static import run_vosviewer_static
                run_vosviewer_static(articles, networks)

            # Fase 4b: OA Corpus Export
            if not args.skip_oa_export:
                with recorder.phase("4b: OA corpus export"):
                    from src.oa_corpus_export import run_oa_corpus_export
                    oa_result = run_oa_corpus_export(articles)
                    if "error" not in oa_result:
                        logger.info(
                            f"OA corpus export: {oa_result['total_oa']} articles "
                            f"({oa_result['pct_oa']}% OA), {oa_result['with_pdf']} with PDF"
                        )

        # Fase 5: Impact Evolution Analysis
        if 5 in args.phases:
            with recorder.phase("5: Impacto evolucion"):
                from src.impact_evolution_analysis import run_phase5
                phase5_results = run_phase5(articles)
                logger.info(
                    f"Phase 5: {phase5_results.get('analyses_completed', 0)}/8 analyses, "
                    f"{phase5_results.get('charts_saved', 0)} charts"
                )

        # Fase 5b: Bootstrap CIs sobre indicadores cabecera (reproducibilidad)
        if 5 in args.phases and not args.skip_bootstrap_ci:
            with recorder.phase("5b: Bootstrap CIs"):
                from src.bootstrap_ci_phase import run_phase5b
                bootstrap_results = run_phase5b(
                    articles,
                    B=args.bootstrap_B,
                    seed=args.bootstrap_seed,
                )
                _mcp = bootstrap_results.get("mcp_rate", {})
                _fwci = bootstrap_results.get("fwci", {}).get("FWCI_mean", {})
                logger.info(
                    f"Phase 5b: bootstrap done (MCP={_mcp.get('point', 0)*100:.2f}%, "
                    f"FWCI mean={_fwci.get('point', float('nan')):.3f})"
                )

        # Fase 6: Research Dynamics Manifold (RDM). Desactivada por defecto: es un indice propio
        # sin validacion externa; solo corre con --enable-rdm y no alimenta la Fase 7.
        if 6 in args.phases and not args.enable_rdm:
            logger.info("Fase 6 (RDM) omitida: desactivada por defecto (no validada). Use --enable-rdm para correrla.")
        if 6 in args.phases and args.enable_rdm:
            with recorder.phase("6: RDM"):
                from src.rdm_analysis import run_phase6_rdm

                if not networks:
                    logger.info("Rebuilding networks for Phase 6 (RDM)...")
                    try:
                        from src.network_analysis import (
                            build_bibliographic_coupling_network,
                            detect_communities,
                            compute_network_metrics,
                        )
                        G_coupling = build_bibliographic_coupling_network(articles)
                        detect_communities(G_coupling)
                        compute_network_metrics(G_coupling)
                        networks = {"bibliographic_coupling": G_coupling}
                    except Exception as net_e:
                        logger.warning(f"Network rebuild failed; RDM will use neutral S axis: {net_e}")

                rdm_results = run_phase6_rdm(articles, networks=networks)
                logger.info(
                    f"Phase 6: RDM — {rdm_results.get('n_articles', 0)} articles, "
                    f"{rdm_results.get('n_archetypes', 0)} archetypes, "
                    f"{rdm_results.get('charts_saved', 0)} charts"
                )

        # Fase 7: Knowledge Map (se ejecuta al final para recoger findings de TODAS las fases 0-6)
        if 7 in args.phases and not args.skip_knowledge_map:
            with recorder.phase("7: Knowledge Map"):
                from src.knowledge_map import run_phase7

                impact_weights = None
                if args.kmap_impact_weights:
                    try:
                        parts = [float(x) for x in args.kmap_impact_weights.split(",")]
                        if len(parts) == 3:
                            impact_weights = {
                                "significance": parts[0],
                                "recency": parts[1],
                                "triangulation": parts[2],
                            }
                        else:
                            logger.warning(
                                "--kmap-impact-weights requiere 3 valores 'sig,rec,tri'; "
                                "usando defaults"
                            )
                    except ValueError:
                        logger.warning(
                            "--kmap-impact-weights invalido; usando defaults"
                        )

                kmap_results = run_phase7(
                    articles,
                    top_n=args.kmap_top_n,
                    impact_weights=impact_weights,
                    phase_status=recorder.phase_status,
                )
                n_findings = kmap_results.get("layer_a", {}).get("stats", {}).get("total_findings", 0)
                n_tri = kmap_results.get("layer_a", {}).get("stats", {}).get("triangulated_findings", 0)
                logger.info(
                    f"Phase 7: Knowledge Map — {n_findings} findings, "
                    f"{n_tri} convergent across phases"
                )

        # Guardar estado final
        save_json(articles, ENRICHED_PATH)
        logger.info("=" * 60)
        logger.info("Pipeline completado.")
        logger.info("=" * 60)
    finally:
        recorder.finalize(articles, csv_path)


if __name__ == "__main__":
    main()
