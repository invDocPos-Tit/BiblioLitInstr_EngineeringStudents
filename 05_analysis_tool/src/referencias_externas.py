# -*- coding: utf-8 -*-
"""Piezas comunes para rellenar bibliografias que OpenAlex no tiene.

La fase 1i resuelve las referencias desde `referenced_works` de OpenAlex. Si
el editor nunca deposito la bibliografia, OpenAlex tiene el articulo con la
lista vacia y ese articulo queda fuera de todo analisis de literatura citada.

Hay dos fuentes que cubren ese hueco y un paso final comun a las dos:

- Semantic Scholar, que extrae las referencias del PDF
  (`scripts/recuperar_referencias_s2.py`).
- La exportacion de referencias de Scopus, registro por registro
  (`scripts/importar_referencias_scopus.py`).

Las dos entregan, en el fondo, una lista de DOI. El paso comun es buscar esas
obras en OpenAlex y construir la entrada con `_build_resolved_entry`, la
MISMA funcion que usa la fase 1i.

Ese paso no es un rodeo evitable: las entradas de `references_resolved`
llevan conceptos, topicos y palabras clave de OpenAlex, y las fases que las
consumen cuentan con ellos. Una entrada armada solo con lo que trae la fuente
tendria esos campos vacios y falsearia en silencio los recuentos por
concepto. Por eso una referencia entra solo si resuelve en OpenAlex; las
demas se reportan como no incorporadas, con su motivo.
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from config import (
    OPENALEX_API_KEY,
    OPENALEX_BASE_URL,
    OPENALEX_EMAIL,
)
from src.reference_resolver import (
    ENHANCED_SELECT,
    _build_resolved_entry,
    _canonical_wid,
)

DOI_VALIDO = re.compile(r"^10\.\d{4,9}/\S+$")
LOTE_OA = 25
PAUSA_OA = 0.15
INTENTOS = 8
ESPERA_429 = 5


def pedir(url: str, cabeceras: dict, timeout: int = 45) -> tuple[dict | None, str]:
    """Devuelve (datos, estado): "ok" | "no_encontrado" | "fallo[:codigo]".

    Reintenta ante 429 y errores de red. Ocho intentos: con cinco quedaban
    tres consultas de cuarenta y tres marcadas como fallo que al repetirlas
    resultaron ser solo limite de tasa.
    """
    for intento in range(INTENTOS):
        try:
            req = urllib.request.Request(url, headers=cabeceras)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8", "replace")), "ok"
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None, "no_encontrado"
            if e.code == 429:
                time.sleep(ESPERA_429 * (intento + 1))
                continue
            if e.code in (401, 403):
                return None, f"fallo:{e.code}"
            time.sleep(2 * (intento + 1))
        except Exception:
            time.sleep(2 * (intento + 1))
    return None, "fallo"


def _cabeceras_openalex() -> dict:
    cab = {"User-Agent": f"biblio1403 (mailto:{OPENALEX_EMAIL})"}
    if OPENALEX_API_KEY:
        cab["Authorization"] = f"Bearer {OPENALEX_API_KEY}"
    return cab


def obras_de_openalex(dois: list[str]) -> dict[str, dict]:
    """Busca en OpenAlex, por lotes, las obras de esos DOI. {doi: obra}."""
    cabeceras = _cabeceras_openalex()
    encontradas: dict[str, dict] = {}
    limpios = [d for d in dict.fromkeys(dois) if DOI_VALIDO.match(d)]
    for i in range(0, len(limpios), LOTE_OA):
        lote = limpios[i:i + LOTE_OA]
        filtro = "doi:" + "|".join(lote)
        url = (f"{OPENALEX_BASE_URL}/works"
               f"?filter={urllib.parse.quote(filtro, safe=':|/.')}"
               f"&per-page={LOTE_OA}&select={ENHANCED_SELECT}")
        datos, estado = pedir(url, cabeceras)
        if estado == "ok":
            for obra in (datos or {}).get("results") or []:
                d = (obra.get("doi") or "").replace("https://doi.org/", "").lower()
                if d:
                    encontradas[d] = obra
        time.sleep(PAUSA_OA)
    return encontradas


def incorporar(articulo: dict, dois: list[str], fuente: str) -> tuple[int, int]:
    """Resuelve esos DOI y los adjunta al articulo. (incorporadas, sin_openalex).

    NO pisa nada: si el articulo ya tiene bibliografia, no la toca.
    """
    if articulo.get("references_resolved"):
        return 0, 0
    validos = [d for d in dict.fromkeys(dois) if DOI_VALIDO.match(d)]
    if not validos:
        return 0, 0
    obras = obras_de_openalex(validos)
    entradas, ids = [], []
    for d in validos:
        obra = obras.get(d)
        if not obra:
            continue
        entradas.append(_build_resolved_entry(obra))
        wid = _canonical_wid(obra.get("id") or "")
        if wid:
            ids.append(f"https://openalex.org/{wid}")
    if not entradas:
        return 0, len(validos)
    articulo["references_resolved"] = entradas
    existentes = list(articulo.get("referenced_works") or [])
    articulo["referenced_works"] = existentes + [x for x in ids if x not in existentes]
    articulo["referencias_fuente"] = fuente
    return len(entradas), len(validos) - len(entradas)
