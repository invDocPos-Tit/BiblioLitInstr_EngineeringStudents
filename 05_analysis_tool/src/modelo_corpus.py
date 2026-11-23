"""Modelo del corpus: UNA sola extraccion de todo lo que las fases 2-7 necesitan del corpus.

Se construye una vez por corrida, despues del enriquecimiento (Fases 0 y 1) y antes de la Fase 2, y se
guarda en data/modelo_corpus.json con su huella. Las fases leen de aqui en lugar de extraer por su
cuenta: antes habia al menos seis extractores de palabras clave y siete de paises, cada uno con su
propia definicion, y daban cifras distintas para lo mismo.

Contenido:
  - por articulo: palabras clave de autor normalizadas, metodos declarados, vacios declarados,
    paises (ISO2) e instituciones normalizadas;
  - del corpus: mapa de normalizacion de palabras clave, siglas (inequivocas y ambiguas), vocabulario
    de metodos con su evidencia, IDF de los resumenes y los umbrales usados.

Todas las reglas son las mismas que ya documentan src/utils.py (vocabulario), src/metodos_declarados.py
y src/vacios_declarados.py; este modulo solo las ejecuta una vez y fija el resultado.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger("bibliometrico.modelo_corpus")

VERSION_MODELO = "1.0"
_MODELO: dict = {}


# ---------------------------------------------------------------------------
# Huella: el modelo solo vale para el corpus con el que se construyo
# ---------------------------------------------------------------------------

def _insumos_articulo(a: dict) -> dict:
    oa = a.get("openalex") or {}
    return {
        "key": a.get("key", ""),
        "year": a.get("publication_year"),
        "title": a.get("title", ""),
        "abstract": a.get("abstract", ""),
        "author_keywords": a.get("author_keywords", a.get("manual_tags", [])),
        "authorships": [
            {"countries": au.get("countries") or [],
             "institutions": [(i.get("name"), i.get("country")) for i in au.get("institutions") or []]}
            for au in oa.get("authorships") or []
        ],
    }


def corpus_fingerprint(articles: list[dict]) -> str:
    h = hashlib.sha256()
    for a in sorted(articles, key=lambda x: str(x.get("key", ""))):
        h.update(json.dumps(_insumos_articulo(a), sort_keys=True, ensure_ascii=False, default=str).encode("utf-8"))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Extracciones (una sola definicion por concepto)
# ---------------------------------------------------------------------------

def article_countries_iso(article: dict) -> list[str]:
    """Paises de un articulo (ISO2): paises de los autores UNION paises de sus instituciones.
    Definicion unica para todas las fases; se cuenta por articulo, no por afiliacion."""
    out = set()
    for au in (article.get("openalex") or {}).get("authorships") or []:
        for c in au.get("countries") or []:
            if c and str(c).strip():
                out.add(str(c).strip().upper())
        for inst in au.get("institutions") or []:
            c = inst.get("country")
            if c and str(c).strip():
                out.add(str(c).strip().upper())
    return sorted(out)


class InsumoIncompletoError(RuntimeError):
    """Falta por completo un insumo del que dependen varias fases."""


# (campo, descripcion, fases que dependen, como resolverlo)
_INSUMOS_CRITICOS = [
    ("author_keywords", "palabras clave de autor", "2e, 2h, 3, 3b y 7 (redes, mapa tematico, tendencias)",
     "exportar consolidado.csv con la columna Author_Keywords (bibcombiner la toma de Scopus/WoS)"),
    ("abstract", "resumen", "2b, 2d y 2f (metodos, calidad de resumenes, vacios)",
     "exportar consolidado.csv con la columna Abstract"),
    ("publication_year", "ano de publicacion", "casi todas (series temporales, RPYS, rafagas)",
     "exportar consolidado.csv con la columna Publication_Year"),
]


def check_inputs(articles: list[dict], allow_incomplete: bool = False) -> dict:
    """Verifica la cobertura de los insumos criticos. Si alguno falta en TODOS los articulos, se detiene
    con un mensaje que dice que falta, que fases afecta y como resolverlo (salvo allow_incomplete).
    La cobertura parcial se informa siempre, sin umbrales: decidir si alcanza es tarea de la persona."""
    n = len(articles)
    cobertura = {}
    for campo, desc, fases, solucion in _INSUMOS_CRITICOS:
        con = sum(1 for a in articles if a.get(campo) not in (None, "", "nan", []) and str(a.get(campo)).strip())
        cobertura[campo] = con
        if con == 0:
            msg = (f"Ningun articulo tiene {desc} (campo '{campo}'). Quedarian vacias las fases {fases}. "
                   f"Solucion: {solucion}. Si es intencional, ejecute con --permitir-insumos-incompletos.")
            if not allow_incomplete:
                raise InsumoIncompletoError(msg)
            logger.warning(msg)
    con_paises = sum(1 for a in articles if article_countries_iso(a))
    con_refs = sum(1 for a in articles if a.get("references_resolved") or a.get("referenced_works"))
    logger.info(
        f"Cobertura de insumos ({n} articulos): palabras clave de autor {cobertura['author_keywords']}, "
        f"resumen {cobertura['abstract']}, ano {cobertura['publication_year']}, paises {con_paises}, "
        f"referencias {con_refs}. Los articulos sin un insumo no aportan a los analisis que lo usan."
    )
    for campo, desc, fases, _ in _INSUMOS_CRITICOS:
        if 0 < cobertura[campo] < n:
            logger.warning(f"  {n - cobertura[campo]} de {n} articulos sin {desc}: no aportan a {fases}.")
    return {**cobertura, "paises": con_paises, "referencias": con_refs, "n": n}


def build_corpus_model(articles: list[dict], save: bool = True, allow_incomplete: bool = False) -> dict:
    """Construye el modelo del corpus, lo deja en memoria y (por defecto) lo guarda en data/."""
    import src.utils as u
    from src.metodos_declarados import build_method_vocabulary, article_methods
    from src.vacios_declarados import gap_sentences

    # Nunca leer de un modelo anterior (de otro corpus) mientras se construye este
    _MODELO.clear()
    cobertura = check_inputs(articles, allow_incomplete=allow_incomplete)

    # 1. Vocabulario de palabras clave (escribe output/reports/normalizacion_palabras_clave.csv)
    mapa = u.prepare_corpus_vocabulary(articles)
    siglas, ambiguas = u.learn_corpus_acronyms(articles)

    # 2. Metodos (escribe output/reports/metodos_declarados.csv)
    vocab_metodos, evidencia = build_method_vocabulary(articles)

    # 3. Instituciones normalizadas con el mapa del corpus
    inst_map = u.build_corpus_institution_normalizer(articles)

    por_articulo = {}
    for a in articles:
        key = str(a.get("key", ""))
        insts = set()
        for au in (a.get("openalex") or {}).get("authorships") or []:
            for inst in au.get("institutions") or []:
                n = u.normalize_institution_name(inst.get("name", ""), corpus_inst_map=inst_map)
                if n:
                    insts.add(n)
        abstract = a.get("abstract", "")
        por_articulo[key] = {
            "anio": a.get("publication_year"),
            "palabras_clave": u.extract_normalized_article_keywords(a, corpus_map=mapa),
            "metodos": article_methods(a, vocab_metodos),
            "vacios": [{"oracion": s_[:500], "cue": c} for s_, c in gap_sentences(str(abstract))]
                      if abstract and abstract != "nan" else [],
            "paises": article_countries_iso(a),
            "instituciones": sorted(insts),
        }

    modelo = {
        "meta": {
            "version": VERSION_MODELO,
            "generado": datetime.now().isoformat(timespec="seconds"),
            "huella_corpus": corpus_fingerprint(articles),
            "n_articulos": len(articles),
            "fuente_palabras_clave": "author_keywords (opcion A, 2026-11-22)",
            "definicion_paises": "paises de los autores UNION paises de sus instituciones (ISO2), por articulo",
            "umbral_metodos": evidencia.get("umbral"),
            "cobertura_insumos": cobertura,
        },
        "vocabulario": {"mapa": mapa, "siglas": siglas, "siglas_ambiguas": ambiguas},
        "metodos": {
            "vocabulario": sorted(vocab_metodos),
            "declarados_en": {m: int(evidencia["declarados"][m]) for m in vocab_metodos},
            "mencionados_en": {m: int(evidencia["menciones"][m]) for m in vocab_metodos},
        },
        "articulos": por_articulo,
    }
    # Terminos de interes declarados por el equipo (opcional): reporte aparte, no altera ningun analisis
    terminos = _reportar_terminos_de_interes(articles, por_articulo, mapa)
    if terminos:
        modelo["terminos_de_interes"] = terminos
    _MODELO.clear()
    _MODELO.update(modelo)
    if save:
        from config import DATA_DIR
        (DATA_DIR / "modelo_corpus.json").write_text(
            json.dumps(modelo, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    n_pk = sum(1 for v in por_articulo.values() if v["palabras_clave"])
    n_me = sum(1 for v in por_articulo.values() if v["metodos"])
    n_va = sum(len(v["vacios"]) for v in por_articulo.values())
    n_pa = sum(1 for v in por_articulo.values() if v["paises"])
    logger.info(
        f"Modelo del corpus: {len(articles)} articulos | palabras clave en {n_pk} | metodos en {n_me} "
        f"({len(vocab_metodos)} en el vocabulario) | {n_va} vacios declarados | paises en {n_pa} | "
        f"{len(siglas)} siglas ({len(ambiguas)} ambiguas)"
    )
    return modelo


def _reportar_terminos_de_interes(articles: list[dict], por_articulo: dict, mapa: dict) -> list[dict]:
    """Para cada termino de interes del equipo: en cuantos articulos es palabra clave de autor, en cuantos
    aparece en titulo o resumen, y su ano mediano. Escribe output/reports/terminos_de_interes.csv.
    OPCIONAL: si el equipo no declaro terminos, no hace nada."""
    import csv
    import numpy as np
    import src.utils as u
    from src.metodos_declarados import _menciona, _tokens_norm

    terminos = u.load_knowledge_base().get("terminos_de_interes") or []
    if not terminos:
        return []
    filas = []
    for t in terminos:
        canon = u.canonicalize_keyword(t, corpus_map=mapa) or t
        como_pk, en_texto, anos = 0, 0, []
        for a in articles:
            e = por_articulo.get(str(a.get("key", "")), {})
            es_pk = canon in (e.get("palabras_clave") or [])
            en_t = _menciona(_tokens_norm(f"{a.get('title', '')} {a.get('abstract', '')}"), canon)
            como_pk += es_pk
            en_texto += en_t
            if (es_pk or en_t) and isinstance(a.get("publication_year"), int):
                anos.append(a["publication_year"])
        filas.append({
            "termino_declarado": t, "termino_en_el_vocabulario": canon,
            "articulos_como_palabra_clave": como_pk, "articulos_en_titulo_o_resumen": en_texto,
            "articulos_en_total": len(anos),
            "ano_q1": int(np.percentile(anos, 25)) if anos else None,
            "ano_mediano": int(np.median(anos)) if anos else None,
            "ano_q3": int(np.percentile(anos, 75)) if anos else None,
            "primer_ano": min(anos) if anos else None, "ultimo_ano": max(anos) if anos else None,
        })
    try:
        from config import REPORTS_DIR
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(REPORTS_DIR / "terminos_de_interes.csv", "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(filas[0]))
            w.writeheader()
            w.writerows(filas)
        logger.info(f"Terminos de interes del equipo: {len(filas)} -> output/reports/terminos_de_interes.csv")
    except Exception as e:
        logger.warning(f"No se pudo escribir terminos_de_interes.csv: {e}")
    return filas


# ---------------------------------------------------------------------------
# Acceso unico
# ---------------------------------------------------------------------------

def get_model(articles: Optional[list[dict]] = None) -> dict:
    """Modelo vigente. Si se pasan `articles` y el modelo en memoria es de otro corpus, se reconstruye
    (en memoria, sin guardar) y se avisa: nunca se usa un modelo de otro corpus en silencio."""
    if _MODELO and (articles is None or _MODELO["meta"]["huella_corpus"] == corpus_fingerprint(articles)):
        return _MODELO
    if not _MODELO:
        from config import DATA_DIR
        p = DATA_DIR / "modelo_corpus.json"
        if p.exists():
            cargado = json.loads(p.read_text(encoding="utf-8"))
            if articles is None or cargado.get("meta", {}).get("huella_corpus") == corpus_fingerprint(articles):
                _MODELO.update(cargado)
                return _MODELO
    if articles is None:
        raise RuntimeError("No hay modelo del corpus: ejecute main.py (se construye antes de la Fase 2) "
                           "o llame a build_corpus_model(articles).")
    logger.warning("El modelo del corpus no corresponde a estos articulos: se reconstruye en memoria.")
    return build_corpus_model(articles, save=False)


def article_countries(article: dict) -> list[str]:
    """Paises (ISO2) de un articulo desde el modelo del corpus; si no hay modelo, misma definicion."""
    entrada = article_entry(article)
    return list(entrada["paises"]) if entrada is not None else article_countries_iso(article)


def article_entry(article: dict) -> Optional[dict]:
    """Entrada del modelo para un articulo (None si no hay modelo o el articulo no esta)."""
    if not _MODELO:
        return None
    return _MODELO["articulos"].get(str(article.get("key", "")))
