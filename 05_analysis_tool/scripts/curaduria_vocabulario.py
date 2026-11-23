"""Curaduria del vocabulario por el equipo de la revision (OPCIONAL).

El pipeline aprende del corpus el vocabulario de palabras clave, las siglas y los metodos. Esta planilla
permite que el equipo lo revise y lo corrija ANTES de los analisis, en Excel, sin editar JSON a mano.

    1) Exportar (despues de las Fases 0 y 1; requiere data/enriched_articles.json):
         .venv\\Scripts\\python.exe scripts/curaduria_vocabulario.py --exportar
       -> data/curaduria_vocabulario.xlsx  (si ya existe, se conserva una copia .anterior.xlsx)

    2) El equipo completa las columnas de decision y justificacion (menus desplegables).

    3) Importar:
         .venv\\Scripts\\python.exe scripts/curaduria_vocabulario.py --importar
       -> valida las decisiones y escribe data/sinonimos_revision.json (copia previa: .anterior.json)

    4) Correr las fases 2-7 con main.py. Los reportes distinguen lo decidido por el equipo de lo automatico.

Si el equipo no hace nada de esto, el pipeline usa solo lo aprendido del corpus (sinonimos_revision.json vacio).
Toda decision distinta de la automatica exige justificacion: queda en el JSON y se puede citar en el paper.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PLANILLA = ROOT / "data" / "curaduria_vocabulario.xlsx"
REVISION = ROOT / "data" / "sinonimos_revision.json"

DEC_UNIF = ["aceptar", "no unificar", "cambiar"]
DEC_METODO = ["automatico", "incluir", "excluir"]


# ---------------------------------------------------------------------------
# Exportar
# ---------------------------------------------------------------------------

def _revision_actual() -> dict:
    if REVISION.exists():
        return json.loads(REVISION.read_text(encoding="utf-8"))
    return {}


def exportar() -> Path:
    import logging
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    logging.basicConfig(level=logging.WARNING)
    import src.utils as u
    from src.metodos_declarados import build_method_vocabulary

    arts = json.loads((ROOT / "data" / "enriched_articles.json").read_text(encoding="utf-8"))
    rev = _revision_actual()
    just = rev.get("justificacion") or {}
    no_unif = set(rev.get("no_unificar") or [])
    syn_rev = rev.get("keyword_synonyms") or {}
    m_incl = set(rev.get("metodos_incluidos") or [])
    m_excl = set(rev.get("metodos_excluidos") or [])

    trace: dict = {}
    mapa = u.build_corpus_keyword_normalizer(arts, trace=trace)
    siglas, ambiguas = u.learn_corpus_acronyms(arts)
    _, ev = build_method_vocabulary(arts)
    freq = Counter(k for a in arts for k in u.extract_normalized_article_keywords(a, corpus_map=mapa))

    wb = Workbook()
    encabezado = Font(bold=True, color="FFFFFF")
    relleno = PatternFill("solid", fgColor="4E79A7")
    editable = PatternFill("solid", fgColor="FFF2CC")

    def hoja(titulo, columnas, anchos, editables=()):
        ws = wb.create_sheet(titulo)
        ws.append(columnas)
        for i, c in enumerate(ws[1], 1):
            c.font, c.fill = encabezado, relleno
            c.alignment = Alignment(wrap_text=True, vertical="top")
            ws.column_dimensions[c.column_letter].width = anchos[i - 1]
        ws.freeze_panes = "A2"
        ws._editables = editables
        return ws

    def pintar(ws, n_filas):
        for col in ws._editables:
            for r in range(2, n_filas + 2):
                ws.cell(row=r, column=col).fill = editable

    def lista(ws, valores, col, n_filas):
        dv = DataValidation(type="list", formula1='"' + ",".join(valores) + '"', allow_blank=False)
        ws.add_data_validation(dv)
        letra = ws.cell(row=1, column=col).column_letter
        dv.add(f"{letra}2:{letra}{max(n_filas + 1, 2)}")

    # LEEME
    ws = wb.active
    ws.title = "LEEME"
    for linea in [
        "Curaduria del vocabulario de la revision (OPCIONAL)",
        "",
        "El pipeline aprendio este vocabulario del propio corpus. Revise y decida solo en las celdas AMARILLAS.",
        "Si no cambia nada, el analisis usa lo automatico. Toda decision distinta de la automatica exige justificacion.",
        "",
        "Hojas:",
        "  Unificaciones   : variantes que el algoritmo unio. aceptar / no unificar / cambiar (indique el termino nuevo).",
        "  Siglas ambiguas : siglas con varias expansiones en el corpus. Elija la opcion correcta o deje 'ninguna'.",
        "  Sinonimos nuevos: equivalencias que el algoritmo no detecto (variante -> termino canonico).",
        "  Metodos         : candidatos a metodo con su evidencia. automatico / incluir / excluir.",
        "  Terminos interes: terminos que el equipo quiere seguir siempre (reporte aparte; no cambia los analisis).",
        "  Exclusiones     : terminos a excluir del vocabulario (p. ej. etiquetas de indexacion del corpus).",
        "  Vocabulario     : SOLO CONSULTA. Todos los terminos del corpus con su frecuencia.",
        "",
        "Al terminar: .venv\\Scripts\\python.exe scripts/curaduria_vocabulario.py --importar",
        "y luego correr las fases 2 a 7 con main.py.",
    ]:
        ws.append([linea])
    ws.column_dimensions["A"].width = 120
    ws["A1"].font = Font(bold=True, size=13)

    # Unificaciones
    ws = hoja("Unificaciones", ["variante", "termino_final_automatico", "regla", "frecuencia_variante",
                                "decision", "termino_nuevo (si 'cambiar')", "justificacion"],
              [34, 34, 34, 10, 14, 30, 50], editables=(5, 6, 7))
    filas = sorted(((v, t) for v, t in trace.items() if not t["regla"].startswith("no unificar")),
                   key=lambda kv: (-kv[1]["frecuencia_variante"], kv[0]))
    for v, t in filas:
        if v in no_unif:
            dec, nuevo = "no unificar", ""
        elif v in syn_rev and syn_rev[v] != t["canonico"]:
            dec, nuevo = "cambiar", syn_rev[v]
        else:
            dec, nuevo = "aceptar", ""
        ws.append([v, t["canonico"], t["regla"], t["frecuencia_variante"], dec, nuevo, just.get(v, "")])
    for v in sorted(no_unif - {v for v, _ in filas}):
        ws.append([v, v, "no unificar (decision del equipo)", freq.get(v, 0), "no unificar", "", just.get(v, "")])
    lista(ws, DEC_UNIF, 5, ws.max_row - 1)
    pintar(ws, ws.max_row - 1)

    # Siglas ambiguas
    ws = hoja("Siglas ambiguas", ["sigla", "opcion_1", "opcion_2", "opcion_3", "opcion_4", "eleccion", "justificacion"],
              [12, 40, 40, 40, 40, 12, 50], editables=(6, 7))
    for sf, lfs in sorted(ambiguas.items()):
        ops = (lfs + ["", "", "", ""])[:4]
        eleccion = "ninguna"
        if sf in syn_rev and syn_rev[sf] in lfs:
            eleccion = str(lfs.index(syn_rev[sf]) + 1)
        ws.append([sf, *ops, eleccion, just.get(sf, "")])
    lista(ws, ["ninguna", "1", "2", "3", "4"], 6, ws.max_row - 1)
    pintar(ws, ws.max_row - 1)

    # Sinonimos nuevos (los de la revision que no vienen de las otras hojas)
    ws = hoja("Sinonimos nuevos", ["variante", "termino_canonico", "justificacion"], [40, 40, 60], editables=(1, 2, 3))
    usados = {v for v, _ in filas} | set(ambiguas)
    for v, c in sorted(syn_rev.items()):
        if v not in usados:
            ws.append([v, c, just.get(v, "")])
    pintar(ws, max(ws.max_row - 1, 30))

    # Metodos
    ws = hoja("Metodos", ["metodo_candidato", "articulos_que_lo_declaran", "articulos_que_lo_mencionan",
                          "entra_automaticamente", "decision", "justificacion"], [40, 14, 14, 14, 14, 50], editables=(5, 6))
    umbral = ev["umbral"]
    cands = sorted(set(ev["declarados"]) | m_incl | m_excl, key=lambda m: (-ev["menciones"].get(m, 0), m))
    for m in cands:
        if ev["declarados"].get(m, 0) < 2 and m not in m_incl | m_excl:
            continue          # solo candidatos declarados por al menos 2 articulos (regla del pipeline)
        auto = ev["menciones"].get(m, 0) >= umbral
        dec = "incluir" if m in m_incl else "excluir" if m in m_excl else "automatico"
        ws.append([m, ev["declarados"].get(m, 0), ev["menciones"].get(m, 0), "si" if auto else "no", dec, just.get(m, "")])
    lista(ws, DEC_METODO, 5, ws.max_row - 1)
    pintar(ws, ws.max_row - 1)

    # Terminos de interes y exclusiones
    ws = hoja("Terminos interes", ["termino", "justificacion"], [50, 70], editables=(1, 2))
    for t in rev.get("terminos_de_interes") or []:
        ws.append([t, just.get(t, "")])
    pintar(ws, max(ws.max_row - 1, 30))
    ws = hoja("Exclusiones", ["termino", "justificacion"], [50, 70], editables=(1, 2))
    for t in rev.get("keyword_stopwords") or []:
        ws.append([t, just.get(t, "")])
    pintar(ws, max(ws.max_row - 1, 30))

    # Vocabulario (consulta)
    ws = hoja("Vocabulario", ["termino", "articulos"], [50, 12])
    for t, n in freq.most_common():
        ws.append([t, n])

    if PLANILLA.exists():
        shutil.copy2(PLANILLA, PLANILLA.with_suffix(".anterior.xlsx"))
    wb.save(PLANILLA)
    print(f"Planilla: {PLANILLA}  ({len(filas)} unificaciones, {len(ambiguas)} siglas ambiguas, "
          f"{len(freq)} terminos en el vocabulario)")
    return PLANILLA


# ---------------------------------------------------------------------------
# Importar
# ---------------------------------------------------------------------------

def _filas(wb, nombre):
    ws = wb[nombre]
    cab = [str(c.value or "").strip() for c in ws[1]]
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row is None or all(v in (None, "") for v in row):
            continue
        yield {cab[i]: ("" if v is None else str(v).strip()) for i, v in enumerate(row) if i < len(cab)}


def importar() -> Path:
    from openpyxl import load_workbook
    from src.utils import clean_keyword_syntax

    if not PLANILLA.exists():
        raise SystemExit(f"No existe {PLANILLA}. Primero: scripts/curaduria_vocabulario.py --exportar")
    wb = load_workbook(PLANILLA, data_only=True)
    errores: list[str] = []
    syn, no_unif, m_incl, m_excl, interes, stop, just = {}, [], [], [], [], [], {}

    def norm(x):
        return clean_keyword_syntax(x) if x else ""

    def requiere_just(clave, j, donde):
        if not j:
            errores.append(f"{donde}: '{clave}' tiene una decision distinta de la automatica y no tiene justificacion.")
        else:
            just[clave] = j

    for f in _filas(wb, "Unificaciones"):
        v, dec, nuevo, j = norm(f.get("variante")), f.get("decision", "aceptar").lower(), norm(f.get("termino_nuevo (si 'cambiar')")), f.get("justificacion", "")
        if dec not in DEC_UNIF:
            errores.append(f"Unificaciones: decision '{dec}' no valida para '{v}' (use {DEC_UNIF}).")
        elif dec == "no unificar":
            no_unif.append(v); requiere_just(v, j, "Unificaciones")
        elif dec == "cambiar":
            if not nuevo:
                errores.append(f"Unificaciones: '{v}' dice 'cambiar' pero no indica el termino nuevo.")
            else:
                syn[v] = nuevo; requiere_just(v, j, "Unificaciones")
    for f in _filas(wb, "Siglas ambiguas"):
        sf, el, j = norm(f.get("sigla")), f.get("eleccion", "ninguna").lower(), f.get("justificacion", "")
        if el in ("", "ninguna"):
            continue
        op = norm(f.get(f"opcion_{el}", ""))
        if not op:
            errores.append(f"Siglas ambiguas: '{sf}' elige la opcion {el}, que esta vacia.")
        else:
            syn[sf] = op; requiere_just(sf, j, "Siglas ambiguas")
    for f in _filas(wb, "Sinonimos nuevos"):
        v, c, j = norm(f.get("variante")), norm(f.get("termino_canonico")), f.get("justificacion", "")
        if v and c:
            if v == c:
                errores.append(f"Sinonimos nuevos: '{v}' apunta a si mismo.")
            syn[v] = c; requiere_just(v, j, "Sinonimos nuevos")
        elif v or c:
            errores.append(f"Sinonimos nuevos: fila incompleta ('{v}' -> '{c}').")
    for f in _filas(wb, "Metodos"):
        m, dec, j = norm(f.get("metodo_candidato")), f.get("decision", "automatico").lower(), f.get("justificacion", "")
        if dec not in DEC_METODO:
            errores.append(f"Metodos: decision '{dec}' no valida para '{m}' (use {DEC_METODO}).")
        elif dec == "incluir":
            m_incl.append(m); requiere_just(m, j, "Metodos")
        elif dec == "excluir":
            m_excl.append(m); requiere_just(m, j, "Metodos")
    for hoja_, destino in (("Terminos interes", interes), ("Exclusiones", stop)):
        for f in _filas(wb, hoja_):
            t, j = norm(f.get("termino")), f.get("justificacion", "")
            if t:
                destino.append(t); requiere_just(t, j, hoja_)

    # Coherencia: sin ciclos ni cadenas contradictorias
    for v, c in syn.items():
        visto, x = {v}, c
        while x in syn:
            if x in visto:
                errores.append(f"Sinonimos: ciclo detectado que pasa por '{v}' -> ... -> '{x}'.")
                break
            visto.add(x); x = syn[x]
        if v in no_unif:
            errores.append(f"'{v}' esta a la vez en 'no unificar' y con un sinonimo.")
    for m in set(m_incl) & set(m_excl):
        errores.append(f"Metodos: '{m}' esta a la vez incluido y excluido.")

    if errores:
        print("La planilla tiene errores; NO se modifico sinonimos_revision.json:\n  - " + "\n  - ".join(errores))
        raise SystemExit(1)

    rev = _revision_actual()
    if REVISION.exists():
        shutil.copy2(REVISION, REVISION.with_suffix(".anterior.json"))
    rev.update({
        "keyword_synonyms": dict(sorted(syn.items())),
        "no_unificar": sorted(set(no_unif)),
        "metodos_incluidos": sorted(set(m_incl)),
        "metodos_excluidos": sorted(set(m_excl)),
        "terminos_de_interes": list(dict.fromkeys(interes)),
        "keyword_stopwords": sorted(set(stop)),
        "justificacion": dict(sorted(just.items())),
    })
    rev.setdefault("_meta", {})["curaduria"] = (
        "Generado desde data/curaduria_vocabulario.xlsx con scripts/curaduria_vocabulario.py --importar")
    REVISION.write_text(json.dumps(rev, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Escrito {REVISION}: {len(syn)} sinonimos, {len(set(no_unif))} 'no unificar', {len(set(m_incl))} metodos "
          f"incluidos, {len(set(m_excl))} excluidos, {len(interes)} terminos de interes, {len(set(stop))} exclusiones.")
    print("Siguiente paso: correr las fases 2 a 7 con main.py.")
    return REVISION


def main() -> None:
    ap = argparse.ArgumentParser(description="Curaduria opcional del vocabulario por el equipo de la revision.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--exportar", action="store_true", help="Generar data/curaduria_vocabulario.xlsx")
    g.add_argument("--importar", action="store_true", help="Validar la planilla y escribir data/sinonimos_revision.json")
    a = ap.parse_args()
    exportar() if a.exportar else importar()


if __name__ == "__main__":
    main()
