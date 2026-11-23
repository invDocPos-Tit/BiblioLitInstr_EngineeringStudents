#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""importar_referencias_manuales.py - Incorpora bibliografias recogidas a mano.

    uv run python scripts/importar_referencias_manuales.py \\
        --dir <carpeta> --mapa <referencias_pendientes.csv>
    ... y otra vez con --aplicar para escribir.

POR QUE
-------
Cuando el editor no deposita la bibliografia, OpenAlex tiene el articulo con
la lista de referencias vacia y ese articulo queda fuera de todo analisis de
literatura citada. Semantic Scholar cubre parte del hueco; para el resto hay
que ir a buscarlas registro por registro.

QUE ESPERA
----------
Una carpeta con un archivo por articulo citante. Dos formatos:

- **RIS** exportado de Scopus o de Web of Science.
- **Texto plano** (`.txt`): la lista de referencias copiada del sitio del
  editor. Solo se aprovechan los DOI explicitos.

El nombre del archivo dice a quien pertenece la lista, y puede ser el
identificador de la base (`2-s2.0-85116777055.ris`, `WOS001856044200035.ris`,
que se traducen con el mapa) o directamente la clave del corpus
(`Yorulmaz2022.txt`). El nombre es lo unico que lo dice: una lista de
referencias no lleva dentro al citante.

El mapa es un CSV con columnas `eid` y `key`; sirve el
`referencias_pendientes.csv` que genera el propio informe.

COMO
----
De cada referencia toma su DOI y busca la obra en OpenAlex para construir la
entrada con la misma funcion de la fase 1i. El motivo de ese paso esta en
`src/referencias_externas.py`. Las referencias sin DOI, y las que no estan en
OpenAlex, quedan contadas en la trazabilidad.

NO PISA NADA
------------
Solo toca articulos cuya bibliografia esta vacia. Sin `--aplicar` escribe un
JSON aparte; con `--aplicar` respalda el original antes de sobrescribir.

CLASE RESULTADO
---------------
Cambia lo que producen las fases 2, 3, 4, 5, 6 y 7. Al aplicarlo hay
que anotarlo en CHANGELOG.md y volver a correrlas.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

_RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_RAIZ))

from config import DATA_DIR, ENRICHED_PATH  # noqa: E402
from src.referencias_externas import DOI_VALIDO, incorporar  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("importar_refs_manuales")

FUENTE = {"ris": "exportacion_manual+openalex",
          "texto": "lista_del_editor+openalex"}


_DOI_EN_TEXTO = re.compile(r"doi\.org/(10\.\d{4,9}/[^\s\]\[)>,;]+)", re.I)


def leer_referencias(ruta: Path) -> tuple[list[str], int, str]:
    """Devuelve (DOI de las referencias, total de entradas, formato).

    Acepta dos formas:

    - **RIS** exportado de Scopus o de Web of Science. Se abre con `utf-8-sig`
      y no con `utf-8`: la exportacion de Web of Science lleva BOM, y con
      `utf-8` el primer `TY` del archivo queda oculto tras el. Se perdia un
      registro por archivo, sin aviso.
    - **Texto plano**: la lista de referencias copiada del sitio del editor.
      Solo se extraen los DOI explicitos; una referencia sin DOI no se intenta
      resolver por su texto, porque emparejar una cita libre contra un indice
      produce falsos positivos y una referencia equivocada es peor que una
      ausente.
    """
    texto = ruta.read_text(encoding="utf-8-sig", errors="replace")
    if re.search(r"^TY  -", texto, re.M):
        dois, total = [], 0
        for linea in texto.splitlines():
            if linea.startswith("TY  -"):
                total += 1
            elif linea.startswith("DO  - "):
                d = linea[6:].strip().lower()
                if DOI_VALIDO.match(d):
                    dois.append(d)
        return dois, total, "ris"

    dois = [d.rstrip(".,;)").lower() for d in _DOI_EN_TEXTO.findall(texto)]
    dois = [d for d in dois if DOI_VALIDO.match(d)]
    total = sum(1 for l in texto.splitlines() if l.strip())
    return dois, total, "texto"


def leer_mapa(ruta: Path) -> dict[str, str]:
    """EID -> clave del corpus."""
    with open(ruta, encoding="utf-8-sig", newline="") as f:
        muestra = f.read(4096)
        f.seek(0)
        sep = ";" if muestra.count(";") >= muestra.count(",") else ","
        filas = list(csv.DictReader(f, delimiter=sep))
    mapa = {}
    for x in filas:
        eid = (x.get("eid") or "").strip()
        clave = (x.get("key") or "").strip()
        if eid and clave:
            mapa[eid] = clave
    return mapa


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, type=Path,
                    help="carpeta con un archivo por articulo (.ris o .txt)")
    ap.add_argument("--mapa", required=True, type=Path,
                    help="CSV con columnas eid y key")
    ap.add_argument("--json", type=Path, default=ENRICHED_PATH)
    ap.add_argument("--aplicar", action="store_true",
                    help="sobrescribe el corpus (respaldando antes)")
    args = ap.parse_args(argv)

    for ruta, que in ((args.dir, "carpeta"), (args.mapa, "mapa"), (args.json, "corpus")):
        if not ruta.exists():
            log.error(f"No existe la {que}: {ruta}")
            return 2

    mapa = leer_mapa(args.mapa)
    archivos = sorted(x for x in args.dir.iterdir()
                  if x.suffix.lower() in (".ris", ".txt"))
    log.info(f"{len(archivos)} archivos | {len(mapa)} EID en el mapa")
    if not archivos:
        log.error("No hay .ris ni .txt en esa carpeta")
        return 2

    articulos = json.loads(args.json.read_text(encoding="utf-8"))
    por_clave = {a.get("key"): a for a in articulos}
    antes = sum(1 for a in articulos if a.get("references_resolved"))
    total = len(articulos)

    traza = []
    st = {"archivos": 0, "sin_mapa": 0, "clave_ausente": 0, "ya_tenia": 0,
          "recuperados": 0, "refs_en_ris": 0, "refs_con_doi": 0,
          "refs_incorporadas": 0, "refs_sin_openalex": 0}

    for ruta in archivos:
        st["archivos"] += 1
        eid = ruta.stem
        # el nombre puede ser el EID (con su mapa) o la clave del corpus
        clave = mapa.get(eid) or (eid if eid in por_clave else None)
        dois, n_refs, formato = leer_referencias(ruta)
        st["refs_en_ris"] += n_refs
        st["refs_con_doi"] += len(dois)

        if not clave:
            st["sin_mapa"] += 1
            resultado = "el nombre no es un EID del mapa ni una clave del corpus"
            incorp = falt = 0
        elif clave not in por_clave:
            st["clave_ausente"] += 1
            resultado = f"la clave {clave} no esta en el corpus"
            incorp = falt = 0
        elif por_clave[clave].get("references_resolved"):
            st["ya_tenia"] += 1
            resultado = "ya tenia bibliografia; no se toca"
            incorp = falt = 0
        else:
            incorp, falt = incorporar(por_clave[clave], dois, FUENTE[formato])
            st["refs_incorporadas"] += incorp
            st["refs_sin_openalex"] += falt
            if incorp:
                st["recuperados"] += 1
                resultado = "incorporado"
            else:
                resultado = "ninguna referencia resolvio en OpenAlex"

        traza.append({"archivo": ruta.name, "eid": eid, "key": clave or "",
                      "formato": formato, "refs_en_el_archivo": n_refs, "refs_con_doi": len(dois),
                      "refs_incorporadas": incorp, "resultado": resultado})
        log.info(f"  {ruta.name:<30} {clave or '(sin mapa)':<22} "
                 f"{n_refs:>3} refs, {len(dois):>3} con DOI, {incorp:>3} incorporadas")

    despues = sum(1 for a in articulos if a.get("references_resolved"))
    sello = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta_traza = DATA_DIR / f"referencias_scopus_trazabilidad_{sello}.csv"
    with open(ruta_traza, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(traza[0]), delimiter=";",
                           quoting=csv.QUOTE_ALL)
        w.writeheader(); w.writerows(traza)

    log.info("")
    log.info("-" * 62)
    log.info(f"Archivos procesados        : {st['archivos']}")
    if st["sin_mapa"]:
        log.info(f"  EID fuera del mapa       : {st['sin_mapa']}")
    if st["clave_ausente"]:
        log.info(f"  clave fuera del corpus   : {st['clave_ausente']}")
    if st["ya_tenia"]:
        log.info(f"  ya tenian bibliografia   : {st['ya_tenia']}")
    log.info(f"Referencias en los archivos: {st['refs_en_ris']}")
    log.info(f"  con DOI                  : {st['refs_con_doi']}")
    log.info(f"  con DOI pero no en OA    : {st['refs_sin_openalex']}")
    log.info(f"  INCORPORADAS             : {st['refs_incorporadas']}")
    log.info("-" * 62)
    log.info(f"Cobertura: {antes}/{total} ({100*antes/total:.1f} %) -> "
             f"{despues}/{total} ({100*despues/total:.1f} %)")
    log.info(f"Trazabilidad: {ruta_traza.name}")

    if st["recuperados"] == 0:
        log.info("Nada que escribir.")
        return 0

    if args.aplicar:
        respaldo = args.json.with_name(args.json.stem + f".ANTES_REFS_SCOPUS_{sello}.json")
        shutil.copy2(args.json, respaldo)
        args.json.write_text(json.dumps(articulos, ensure_ascii=False, indent=1),
                             encoding="utf-8")
        log.info(f"Respaldo : {respaldo.name}")
        log.info(f"Escrito  : {args.json.name}")
        log.info("")
        log.info("CLASE RESULTADO. Falta anotarlo en CHANGELOG.md y rehacer:")
        log.info("  uv run python main.py --phases 2 3 4 5 6 7 --run-id <id>")
    else:
        salida = args.json.with_name(args.json.stem + f"_refs_scopus_{sello}.json")
        salida.write_text(json.dumps(articulos, ensure_ascii=False, indent=1),
                          encoding="utf-8")
        log.info(f"Salida (no se toco el original): {salida.name}")
        log.info("Para aplicarlo: volver a correr con --aplicar")
    return 0


if __name__ == "__main__":
    sys.exit(main())
