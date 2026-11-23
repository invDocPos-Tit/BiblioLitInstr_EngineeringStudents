"""Vacios de conocimiento DECLARADOS en los resumenes: unica implementacion para las Fases 2b y 2f.

Antes habia dos detectores distintos que daban cifras distintas sobre el mismo corpus:
  - Fase 2b: 34 frases fijas, varias demasiado amplias ('is still', 'for the first time',
    'existing approaches'); 60 oraciones en el corpus de 347.
  - Fase 2f: expresiones regulares estrictas; 28 oraciones. Ademas leia 'future_research_sentences',
    un campo que ninguna fase escribe.

Criterio unico: la oracion declara explicitamente que falta investigacion, evidencia o comprension.
Lo fijo es lenguaje general de reporte cientifico en ingles (no vocabulario de un campo):
  - se exige que la frase hable de investigacion/estudios/evidencia/comprension;
  - no se cuentan los anuncios de novedad ('for the first time', 'to the best of our knowledge'),
    ni 'lack of' suelto ('lack of motivation' es un hallazgo, no un vacio), ni retos genericos.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger("bibliometrico.vacios_declarados")

_INV = r"(research|studies|study|work|evidence|data|attention|investigation|literature|knowledge|understanding)"

PATRONES_VACIO: list[tuple[str, re.Pattern]] = [
    ("vacio en la literatura", re.compile(r"(?i)\bgaps?\s+in\s+(the\s+)?(literature|knowledge|research|understanding)\b")),
    ("falta de investigacion", re.compile(rf"(?i)\b(lack|absence|scarcity|paucity|dearth)\s+of\s+(empirical\s+|prior\s+|previous\s+)?{_INV}\b")),
    ("poco estudiado", re.compile(r"(?i)\bremains?\s+(poorly|under|insufficiently|largely\s+un)\s*-?\s*(understood|studied|explored|investigated|examined)\b")),
    ("ningun estudio previo", re.compile(r"(?i)\bno\s+(previous|prior|existing)?\s*(study|studies|research|work)\s+(has|have)\b")),
    ("investigacion escasa", re.compile(rf"(?i)\b(limited|few|little|scarce|sparse)\s+(empirical\s+)?{_INV}\b")),
    ("se necesita mas investigacion", re.compile(r"(?i)\bneed\s+for\s+(further|more|additional)\s+(research|investigation|study|studies)\b")),
    ("investigacion futura", re.compile(r"(?i)\bfuture\s+(research|work|studies)\s+should\b")),
    ("inexplorado", re.compile(r"(?i)\b(under-?explored|under-?studied|under-?researched|overlooked|unexplored)\b")),
    ("investigacion requerida", re.compile(r"(?i)\bresearch\s+is\s+(needed|required|warranted)\b")),
    ("rara vez abordado", re.compile(r"(?i)\b(rarely|seldom)\s+(been\s+)?(addressed|studied|examined|investigated|explored)\b")),
    ("no comprendido", re.compile(r"(?i)\bnot\s+(well|fully|yet)\s+(understood|known|explored|established|examined)\b")),
    ("no estudiado aun", re.compile(r"(?i)\b(has|have)\s+not\s+(yet\s+)?been\s+(studied|examined|investigated|explored|addressed|evaluated|validated|tested)\b")),
    ("pregunta abierta", re.compile(r"(?i)\b(open|unresolved|unanswered)\s+(research\s+)?(questions?|problems?|issues?)\b")),
]

_SPLIT = re.compile(r"(?<=[.!?])\s+")


def gap_sentences(text: str) -> list[tuple[str, str]]:
    """Oraciones del texto que declaran un vacio, con la etiqueta del patron que las detecto."""
    out = []
    for sent in _SPLIT.split(text or ""):
        sent = sent.strip()
        if len(sent) < 20:
            continue
        for etiqueta, pat in PATRONES_VACIO:
            if pat.search(sent):
                out.append((sent, etiqueta))
                break               # una sola deteccion por oracion
    return out


def extract_declared_gaps(articles: list[dict]) -> list[dict]:
    """Vacios declarados en los resumenes del corpus (una entrada por oracion)."""
    gaps = []
    for a in articles:
        abstract = a.get("abstract", "")
        if not abstract or abstract == "nan":
            continue
        from src.modelo_corpus import article_entry
        entrada = article_entry(a)
        pares = ([(v["oracion"], v["cue"]) for v in entrada["vacios"]] if entrada is not None
                 else gap_sentences(str(abstract)))
        for sent, etiqueta in pares:
            gaps.append({
                "key": a.get("key", ""),
                "year": a.get("publication_year"),
                "sentence": sent[:500],
                "cue": etiqueta,
                "title": str(a.get("title", ""))[:60],
                "journal": a.get("publication_title", ""),
            })
    logger.info(f"Vacios declarados: {len(gaps)} oraciones en {len({g['key'] for g in gaps})} articulos")
    return gaps
