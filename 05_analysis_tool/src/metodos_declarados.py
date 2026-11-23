"""Metodos DECLARADOS por el corpus, sin lista de metodos ni de tecnicas (agnostico al dominio).

Unica implementacion usada por la Fase 2b (analisis de resumenes) y la Fase 2f (SOTA y vacios).

Un metodo es lo que los articulos declaran con las construcciones gramaticales con que cualquier
articulo, de cualquier campo, reporta su metodo:
    "using / via / through / by means of / employing / applying X"
    "X was/were used / applied / employed / conducted / performed / carried out / administered / adopted / implemented"
Lo unico fijo es el ingles funcional (determinantes, preposiciones, auxiliares), como una lista de
stopwords. Los nombres de los metodos salen del corpus y se filtran por datos:
  - un candidato de una sola palabra solo cuenta si el corpus tambien lo usa como palabra clave
    (descarta 'study', 'research' sin escribir una lista);
  - un metodo entra al vocabulario si lo declaran al menos max(2, N/50) articulos (mismo umbral que
    las demas dimensiones de la Fase 2f).
"""
from __future__ import annotations

import logging
import re
from collections import Counter

from src.utils import canonicalize_keyword, is_keyword_stopword, singularize_term_agnostic

logger = logging.getLogger("bibliometrico.metodos_declarados")


# Ingles funcional: lo unico fijo del detector (determinantes, pronombres, preposiciones, auxiliares).
_FUNCTION_WORDS = frozenset("""
a an the this that these those such each every all both either neither some any no
our their its his her my your we they it he she i you one two three several various many
of in on at to for from by with without within into onto upon via through across among between
and or nor but as than then so also more most less least very
is are was were be been being has have had do does did can could may might must shall should will would
which who whom whose what where when while whereas whether if because since
using used use based new other different further first second same own
""".split())
_METHOD_FORWARD_RE = re.compile(
    r"\b(?:using|via|through|by means of|by using|with the use of|employing|applying)\s+([a-z][a-z0-9\-' ]{2,80})"
)
_METHOD_BACKWARD_RE = re.compile(
    r"([a-z][a-z0-9\-' ]{2,80}?)\s+(?:was|were|has been|have been|had been)\s+"
    r"(?:used|applied|employed|conducted|performed|carried out|administered|adopted|implemented)\b"
)
_MAX_METHOD_TOKENS = 5


def _split_coordination(tokens: list[str]) -> list[list[str]]:
    """'exploratory and confirmatory factor analysis' -> [[exploratory factor analysis], [confirmatory factor analysis]];
    'interviews and surveys' -> [[interviews], [surveys]]. Solo usa la estructura, no el significado."""
    partes, actual = [], []
    for t in tokens:
        if t in ("and", "or"):
            if actual:
                partes.append(actual)
            actual = []
        else:
            actual.append(t)
    if actual:
        partes.append(actual)
    if len(partes) < 2:
        return partes
    ultima = partes[-1]
    if len(ultima) > 1 and all(len(p_) == 1 for p_ in partes[:-1]):
        return [p_ + ultima[1:] for p_ in partes[:-1]] + [ultima]
    return partes


def method_phrases(fragment: str, from_start: bool) -> list[str]:
    """Recorta el sintagma nominal adyacente a la construccion (hasta la primera palabra funcional,
    admitiendo coordinacion con 'and'/'or') y lo separa en sus metodos."""
    toks = re.split(r"\s+", fragment.strip())
    if from_start:                      # "using [structural equation modeling] to ..."
        out = []
        for t in toks:
            if not t or (t in _FUNCTION_WORDS and t not in ("and", "or")):
                if out:
                    break
                continue
            if t in ("and", "or") and not out:
                continue
            out.append(t)
            if len([x for x in out if x not in ("and", "or")]) >= _MAX_METHOD_TOKENS:
                break
    else:                               # "... [semi-structured interviews] were conducted"
        out = []
        for t in reversed(toks):
            if not t or (t in _FUNCTION_WORDS and t not in ("and", "or")):
                break
            out.insert(0, t)
            if len([x for x in out if x not in ("and", "or")]) >= _MAX_METHOD_TOKENS:
                break
    while out and out[-1] in ("and", "or"):
        out.pop()
    while out and out[0] in ("and", "or"):
        out.pop(0)
    frases = []
    for parte in _split_coordination(out):
        ph = " ".join(parte).strip(" -")
        if len(ph) >= 3 and any(ch.isalpha() for ch in ph):
            frases.append(ph)
    return frases


def method_candidates(text: str) -> list[str]:
    """Candidatos a metodo declarados en el texto (sin vocabulario de metodos)."""
    out = []
    for sent in re.split(r"[.;:!?()\[\]]", (text or "").lower()):
        for m in _METHOD_FORWARD_RE.finditer(sent):
            out.extend(method_phrases(re.split(r",", m.group(1))[0], from_start=True))
        for m in _METHOD_BACKWARD_RE.finditer(sent):
            out.extend(method_phrases(re.split(r",", m.group(1))[-1], from_start=False))
    return out




def _canon(term: str) -> str:
    from src.utils import _VOCABULARIO
    return canonicalize_keyword(term, corpus_map=_VOCABULARIO.get("mapa"))


def declared_methods(text: str) -> list[str]:
    """Metodos candidatos declarados en un texto, normalizados con el vocabulario comun."""
    out = []
    for cand in method_candidates(text):
        canon = _canon(singularize_term_agnostic(cand.replace("’", "'")))
        if canon and not is_keyword_stopword(canon):
            out.append(canon)
    return out


def min_freq_for(n_articles: int) -> int:
    return max(2, n_articles // 50)


# Un metodo debe estar declarado por al menos 2 articulos distintos (evidencia minima no unitaria).
MIN_ARTICULOS_QUE_DECLARAN = 2


def _tokens_norm(texto: str) -> list[str]:
    # El guion cuenta como espacio ('self-efficacy' = 'self efficacy'), igual que en el vocabulario comun
    toks = re.findall(r"[a-z0-9][a-z0-9']*", (texto or "").lower().replace("’", "'").replace("-", " "))
    return [singularize_term_agnostic(t) for t in toks]


def _menciona(tokens: list[str], termino: str) -> bool:
    objetivo = [singularize_term_agnostic(t) for t in termino.replace("-", " ").split()]
    n = len(objetivo)
    return any(tokens[i:i + n] == objetivo for i in range(len(tokens) - n + 1)) if n else False


def _menciones_articulo(article: dict, candidatos, keywords: set[str]) -> set[str]:
    tokens = _tokens_norm(f"{article.get('title', '')} {article.get('abstract', '')}")
    return {m for m in candidatos if m in keywords or _menciona(tokens, m)}


def build_method_vocabulary(articles: list[dict]) -> tuple[set[str], dict]:
    """Vocabulario de metodos del corpus.

    1. Que es un metodo: lo DECLARAN (con las construcciones de reporte) al menos 2 articulos.
    2. Para entrar al vocabulario, ademas debe MENCIONARSE (titulo, resumen o palabras clave) en
       al menos max(2, N/50) articulos, el mismo umbral que las demas dimensiones de la 2f.
    """
    from src.utils import article_keywords

    kw_por_art = [set(article_keywords(a, articles)) for a in articles]
    palabras_clave = set().union(*kw_por_art) if kw_por_art else set()
    declarados: Counter = Counter()
    for a in articles:
        for m in set(declared_methods(f"{a.get('title', '')}. {a.get('abstract', '')}")):
            if " " in m or m in palabras_clave:
                declarados[m] += 1
    candidatos = {m for m, n in declarados.items() if n >= MIN_ARTICULOS_QUE_DECLARAN}
    menciones: Counter = Counter()
    for a, kws in zip(articles, kw_por_art):
        for m in _menciones_articulo(a, candidatos, kws):
            menciones[m] += 1
    umbral = min_freq_for(len(articles))
    excluidos = _metodos_excluidos_de_la_revision()
    incluidos = _metodos_de_la_revision("metodos_incluidos")
    vocab = {m for m in candidatos if menciones[m] >= umbral and m not in excluidos}
    if incluidos:
        # Decision del equipo: se agregan aunque el corpus no los declare con frecuencia suficiente
        for a, kws in zip(articles, kw_por_art):
            for m in _menciones_articulo(a, incluidos - candidatos, kws):
                menciones[m] += 1
        candidatos |= incluidos
        vocab |= (incluidos - excluidos)
    logger.info(
        f"Metodos del corpus: {len(declarados)} candidatos declarados, {len(candidatos)} declarados por "
        f">= {MIN_ARTICULOS_QUE_DECLARAN} articulos, {len(vocab)} mencionados en >= {umbral} articulos"
        + (f"; {len(excluidos & candidatos)} excluidos por sinonimos_revision.json" if excluidos else "")
    )
    _escribir_reporte(candidatos, declarados, menciones, umbral, excluidos, vocab)
    return vocab, {"declarados": declarados, "menciones": menciones, "umbral": umbral}


def _metodos_de_la_revision(clave: str) -> set[str]:
    """Lista de metodos de data/sinonimos_revision.json (clave 'metodos_incluidos' o 'metodos_excluidos')."""
    import json
    from config import SINONIMOS_REVISION_PATH
    if not SINONIMOS_REVISION_PATH.exists():
        return set()
    try:
        data = json.loads(SINONIMOS_REVISION_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"No se pudo leer {SINONIMOS_REVISION_PATH}: {e}")
        return set()
    return {_canon(str(m)) for m in (data.get(clave) or []) if str(m).strip()}


def _metodos_excluidos_de_la_revision() -> set[str]:
    """Exclusiones propias de la revision (data/sinonimos_revision.json, clave 'metodos_excluidos').
    Vacio por defecto: el resultado es el que declara el corpus."""
    import json
    from config import SINONIMOS_REVISION_PATH
    if not SINONIMOS_REVISION_PATH.exists():
        return set()
    try:
        data = json.loads(SINONIMOS_REVISION_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"No se pudo leer {SINONIMOS_REVISION_PATH}: {e}")
        return set()
    return {_canon(str(m)) for m in (data.get("metodos_excluidos") or []) if str(m).strip()}


def _escribir_reporte(candidatos, declarados, menciones, umbral, excluidos, vocab) -> None:
    """output/reports/metodos_declarados.csv: evidencia de cada metodo para revisarla y declararla."""
    try:
        import csv
        from config import REPORTS_DIR
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(REPORTS_DIR / "metodos_declarados.csv", "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["metodo", "articulos_que_lo_declaran", "articulos_que_lo_mencionan",
                        "en_vocabulario", "motivo"])
            for m in sorted(candidatos, key=lambda x: (-menciones[x], x)):
                if m in excluidos:
                    motivo = "excluido por el equipo (sinonimos_revision.json)"
                elif m in _metodos_de_la_revision("metodos_incluidos"):
                    motivo = "incluido por el equipo (sinonimos_revision.json)"
                elif menciones[m] < umbral:
                    motivo = f"mencionado en menos de {umbral} articulos"
                else:
                    motivo = ""
                w.writerow([m, declarados[m], menciones[m], m in vocab, motivo])
    except Exception as e:
        logger.warning(f"No se pudo escribir metodos_declarados.csv: {e}")


def article_methods(article: dict, vocabulary: set[str], articles: list[dict] | None = None) -> list[str]:
    """Metodos del vocabulario del corpus que un articulo menciona (titulo, resumen o palabras clave).
    Es la UNICA definicion de 'metodo de un articulo' que usan las Fases 2b y 2f."""
    from src.modelo_corpus import article_entry
    entrada = article_entry(article)
    if entrada is not None and set(entrada["metodos"]) <= set(vocabulary):
        return list(entrada["metodos"])
    from src.utils import article_keywords
    return sorted(_menciones_articulo(article, vocabulary, set(article_keywords(article, articles))))
