#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""recuperar_referencias_s2.py - Rellena las referencias que OpenAlex no tiene.

    uv run python scripts/recuperar_referencias_s2.py            # solo informa
    uv run python scripts/recuperar_referencias_s2.py --aplicar  # escribe

POR QUE
-------
La fase 1i resuelve las referencias desde `referenced_works` de OpenAlex. Si
el editor nunca deposito la bibliografia, OpenAlex tiene el articulo pero con
la lista vacia, y el articulo queda fuera de todo analisis de literatura
citada: espectroscopia, cocitacion, acoplamiento, edad de las referencias.

Semantic Scholar construye su grafo extrayendo las referencias del PDF, no
esperando el deposito del editor. Cubre justamente ese hueco.

COMO
----
1. Toma los articulos SIN `references_resolved` y con DOI valido.
2. Pide a Semantic Scholar la lista de referencias de cada uno.
3. De cada referencia toma su DOI y busca la obra en OpenAlex, en lotes.
4. Construye la entrada con `_build_resolved_entry`, la MISMA funcion que usa
   la fase 1i.

El paso 3 no es un rodeo evitable: las entradas de `references_resolved`
llevan conceptos, topicos y palabras clave de OpenAlex, y las fases que las
consumen cuentan con ellos. Una entrada armada solo con lo que devuelve
Semantic Scholar tendria esos campos vacios y contaminaria en silencio los
recuentos por concepto. Por eso una referencia entra solo si se resuelve en
OpenAlex; las demas quedan en la trazabilidad con su motivo.

NO PISA NADA
------------
Solo toca articulos cuya lista de referencias esta vacia. Sin `--aplicar`
escribe un JSON aparte; con `--aplicar` respalda el original antes de
sobrescribir.

CLASE RESULTADO
---------------
Cambia lo que producen las fases 2, 3, 4, 5, 6 y 7. Al aplicarlo hay que
anotarlo en CHANGELOG.md y volver a correr esas fases.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import sys
import time
import urllib.parse
from datetime import datetime
from pathlib import Path

_RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_RAIZ))

from config import DATA_DIR, ENRICHED_PATH, S2_API_KEY, S2_BASE_URL  # noqa: E402
from src.referencias_externas import (  # noqa: E402
    DOI_VALIDO,
    incorporar,
    pedir,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("recuperar_refs_s2")

PAUSA_S2 = 1.1          # el limite de S2 es 1 req/s, acumulado
FUENTE = "semantic_scholar+openalex"


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def referencias_de_s2(doi: str) -> tuple[list[dict], str]:
    """Lista de referencias que Semantic Scholar tiene para ese DOI."""
    campos = "externalIds,title,year,authors"
    url = (f"{S2_BASE_URL}/paper/DOI:{urllib.parse.quote(doi)}/references"
           f"?fields={campos}&limit=1000")
    datos, estado = pedir(url, {"x-api-key": S2_API_KEY} if S2_API_KEY else {})
    if estado != "ok":
        return [], estado
    salida = []
    for item in (datos or {}).get("data") or []:
        cita = item.get("citedPaper") or {}
        ext = cita.get("externalIds") or {}
        salida.append({
            "doi": (ext.get("DOI") or "").lower().strip(),
            "titulo": cita.get("title") or "",
            "anio": cita.get("year"),
        })
    return salida, "ok"


# ---------------------------------------------------------------------------
# Proceso
# ---------------------------------------------------------------------------

def recuperar(articulos: list[dict], registrar=log.info) -> tuple[list[dict], list[dict], dict]:
    pendientes = [
        a for a in articulos
        if not (a.get("references_resolved") or [])
        and DOI_VALIDO.match(str(a.get("doi") or "").strip())
    ]
    sin_doi = sum(
        1 for a in articulos
        if not (a.get("references_resolved") or [])
        and not DOI_VALIDO.match(str(a.get("doi") or "").strip())
    )
    registrar(f"{len(articulos)} articulos | "
              f"{sum(1 for a in articulos if a.get('references_resolved'))} ya con referencias")
    registrar(f"{len(pendientes)} sin referencias y con DOI valido "
              f"({sin_doi} sin referencias y sin DOI: no se pueden consultar)")

    traza: list[dict] = []
    stats = {"consultados": 0, "sin_registro_en_s2": 0, "sin_referencias_en_s2": 0,
             "fallos_s2": 0, "articulos_recuperados": 0,
             "refs_de_s2": 0, "refs_resueltas_en_openalex": 0,
             "refs_sin_doi": 0, "refs_sin_openalex": 0}

    for n, art in enumerate(pendientes, 1):
        clave = art.get("key") or ""
        doi = str(art.get("doi")).strip()
        refs, estado = referencias_de_s2(doi)
        stats["consultados"] += 1
        time.sleep(PAUSA_S2)

        if estado == "no_encontrado":
            stats["sin_registro_en_s2"] += 1
            traza.append({"key": clave, "doi": doi, "resultado": "no esta en Semantic Scholar",
                          "refs_en_s2": 0, "refs_con_doi": 0, "refs_incorporadas": 0})
            continue
        if estado.startswith("fallo"):
            stats["fallos_s2"] += 1
            traza.append({"key": clave, "doi": doi, "resultado": f"fallo de consulta ({estado})",
                          "refs_en_s2": 0, "refs_con_doi": 0, "refs_incorporadas": 0})
            continue
        if not refs:
            stats["sin_referencias_en_s2"] += 1
            traza.append({"key": clave, "doi": doi, "resultado": "sin referencias en Semantic Scholar",
                          "refs_en_s2": 0, "refs_con_doi": 0, "refs_incorporadas": 0})
            continue

        stats["refs_de_s2"] += len(refs)
        con_doi = [r["doi"] for r in refs if DOI_VALIDO.match(r["doi"])]
        stats["refs_sin_doi"] += len(refs) - len(con_doi)

        incorporadas, sin_oa = incorporar(art, con_doi, FUENTE)
        stats["refs_sin_openalex"] += sin_oa
        if incorporadas:
            stats["articulos_recuperados"] += 1
            stats["refs_resueltas_en_openalex"] += incorporadas
            resultado = "recuperado"
        else:
            resultado = "ninguna referencia resolvio en OpenAlex"

        traza.append({"key": clave, "doi": doi, "resultado": resultado,
                      "refs_en_s2": len(refs), "refs_con_doi": len(con_doi),
                      "refs_incorporadas": incorporadas})
        registrar(f"  [{n}/{len(pendientes)}] {clave}: {len(refs)} en S2, "
                  f"{len(con_doi)} con DOI, {incorporadas} incorporadas")

    return articulos, traza, stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", type=Path, default=ENRICHED_PATH,
                    help="corpus enriquecido de entrada")
    ap.add_argument("--aplicar", action="store_true",
                    help="sobrescribe el corpus (respaldando antes). Sin esto, "
                         "escribe un archivo aparte y no toca nada")
    args = ap.parse_args(argv)

    if not S2_API_KEY:
        log.error("Falta S2_API_KEY. Va en ~/.automatizaciones/.env")
        return 2
    if not args.json.is_file():
        log.error(f"No existe {args.json}")
        return 2

    articulos = json.loads(args.json.read_text(encoding="utf-8"))
    antes = sum(1 for a in articulos if a.get("references_resolved"))

    articulos, traza, stats = recuperar(articulos)
    despues = sum(1 for a in articulos if a.get("references_resolved"))
    total = len(articulos)

    sello = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta_traza = DATA_DIR / f"referencias_s2_trazabilidad_{sello}.csv"
    if traza:
        with open(ruta_traza, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(traza[0]), delimiter=";",
                               quoting=csv.QUOTE_ALL)
            w.writeheader()
            w.writerows(traza)

    log.info("")
    log.info("-" * 62)
    log.info(f"Consultados en Semantic Scholar : {stats['consultados']}")
    log.info(f"  no estan en S2                : {stats['sin_registro_en_s2']}")
    log.info(f"  estan pero sin referencias    : {stats['sin_referencias_en_s2']}")
    log.info(f"  fallos de consulta            : {stats['fallos_s2']}")
    log.info(f"Referencias vistas en S2        : {stats['refs_de_s2']}")
    log.info(f"  sin DOI (no consultables)     : {stats['refs_sin_doi']}")
    log.info(f"  con DOI pero no en OpenAlex   : {stats['refs_sin_openalex']}")
    log.info(f"  INCORPORADAS                  : {stats['refs_resueltas_en_openalex']}")
    log.info("-" * 62)
    log.info(f"Cobertura de referencias: {antes}/{total} ({100*antes/total:.1f} %) "
             f"-> {despues}/{total} ({100*despues/total:.1f} %)")
    if traza:
        log.info(f"Trazabilidad: {ruta_traza.name}")

    if stats["articulos_recuperados"] == 0:
        log.info("Nada que escribir.")
        return 0

    if args.aplicar:
        respaldo = args.json.with_name(args.json.stem + f".ANTES_REFS_S2_{sello}.json")
        shutil.copy2(args.json, respaldo)
        args.json.write_text(json.dumps(articulos, ensure_ascii=False, indent=1),
                             encoding="utf-8")
        log.info(f"Respaldo : {respaldo.name}")
        log.info(f"Escrito  : {args.json.name}")
        log.info("")
        log.info("CLASE RESULTADO. Falta: anotarlo en CHANGELOG.md y volver a correr")
        log.info("las fases que leen las referencias:")
        log.info("  uv run python main.py --phases 2 3 4 5 6 7 --run-id <id>")
    else:
        salida = args.json.with_name(args.json.stem + f"_refs_s2_{sello}.json")
        salida.write_text(json.dumps(articulos, ensure_ascii=False, indent=1),
                          encoding="utf-8")
        log.info(f"Salida (no se toco el original): {salida.name}")
        log.info("Para aplicarlo: volver a correr con --aplicar")
    return 0


if __name__ == "__main__":
    sys.exit(main())
