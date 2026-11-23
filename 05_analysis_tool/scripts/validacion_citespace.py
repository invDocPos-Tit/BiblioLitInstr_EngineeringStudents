"""Validacion de la Fase 2h ('CiteSpace-style', implementacion propia) sin el software CiteSpace.

Se valida cada pieza contra una referencia independiente del codigo de la 2h:
  1. Rafagas de Kleinberg (2002), modelo binomial por periodos (r_t de d_t documentos):
     a) referencia EXACTA de 2 estados escrita aqui desde las ecuaciones del articulo (Viterbi completo);
     (El paquete `burst_detection` 0.1.3 se descarto: esta abandonado y falla con NumPy actual en dos
     puntos, np.float y la asignacion de np.where; ademas decodifica de forma voraz, no con Viterbi.)
     La 2h usa varios niveles; se compara la condicion 'en rafaga' (nivel >= 1) por entidad y por ano.
  2. Sigma de Chen et al. (2009): sigma = (betweenness + 1) ^ burstness, recalculado desde lo guardado.

Uso (desde la raiz del proyecto, despues de la Fase 2):
    .venv\\Scripts\\python.exe scripts/validacion_citespace.py
Salida: validacion_externa/citespace/informe_validacion_citespace.md y detalle CSV.
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "validacion_externa" / "citespace"
S, GAMMA, MIN_OCC = 2.0, 1.0, 3        # valores por defecto de main.py (--citespace-burst-s/-gamma)


def _ln_binom_cost(r: int, d: int, p: float) -> float:
    """-ln P(r | d, p) sin la constante combinatoria (igual en todos los estados)."""
    p = min(max(p, 1e-12), 1 - 1e-12)
    return -(r * math.log(p) + (d - r) * math.log(1 - p))


def referencia_kleinberg_2_estados(r: list[int], d: list[int], s: float, gamma: float) -> list[int]:
    """Kleinberg (2002), seccion de rafagas en lotes: 2 estados, p0 = R/D, p1 = s*p0,
    costo de subir de estado = gamma * ln(n); bajar no cuesta. Viterbi completo."""
    n = len(r)
    p0 = sum(r) / max(sum(d), 1)
    p = [p0, min(s * p0, 0.99999)]
    tau_up = gamma * math.log(max(n, 2))
    cost = [[0.0, 0.0] for _ in range(n)]
    back = [[0, 0] for _ in range(n)]
    for j in (0, 1):
        cost[0][j] = (tau_up if j == 1 else 0.0) + _ln_binom_cost(r[0], d[0], p[j])
    for t in range(1, n):
        for j in (0, 1):
            cands = [cost[t - 1][i] + (tau_up if j > i else 0.0) for i in (0, 1)]
            i_best = int(np.argmin(cands))
            back[t][j] = i_best
            cost[t][j] = cands[i_best] + _ln_binom_cost(r[t], d[t], p[j])
    q = [0] * n
    q[-1] = int(np.argmin(cost[-1]))
    for t in range(n - 1, 0, -1):
        q[t - 1] = back[t][q[t]]
    return q


def main() -> None:
    import logging
    logging.basicConfig(level=logging.WARNING)
    import src.utils as u
    from src.citespace_analysis import _extract_entity_year_counts, _kleinberg_burst_detection

    OUT.mkdir(parents=True, exist_ok=True)
    arts = json.loads((ROOT / "data" / "enriched_articles.json").read_text(encoding="utf-8"))
    u.prepare_corpus_vocabulary(arts, write_report=False)

    year_totals: dict[int, int] = defaultdict(int)
    for a in arts:
        try:
            year_totals[int(a.get("publication_year"))] += 1
        except (TypeError, ValueError):
            pass
    years = sorted(year_totals)

    L = ["# Validación de la Fase 2h sin el software CiteSpace", "",
         "La Fase 2h es una implementación propia de los análisis que popularizó CiteSpace. "
         "Aquí se compara cada pieza con una referencia independiente de su código.", "",
         f"Parámetros: s = {S}, γ = {GAMMA}, mínimo de apariciones = {MIN_OCC} (por defecto en `main.py`).", ""]
    filas, resumen = [], []
    for tipo in ("keywords", "references", "journals", "authors"):
        counts = _extract_entity_year_counts(arts, tipo)
        n_ent = acuerdo_ref = acuerdo_pkg = 0
        anos_pipe = anos_ref = anos_pkg = inter_ref = inter_pkg = 0
        for ent, yc in counts.items():
            if sum(yc.values()) < MIN_OCC:
                continue
            n_ent += 1
            r = [int(yc.get(y, 0)) for y in years]
            d = [max(r[i], year_totals[y], 1) for i, y in enumerate(years)]
            # 2h (varios niveles): anos con nivel >= 1
            bursts = _kleinberg_burst_detection(yc, s=S, gamma=GAMMA, year_totals=dict(year_totals))
            pipe = {y for b in bursts if b.get("level", 1) >= 1
                    for y in range(int(b["begin_year"]), int(b["end_year"]) + 1)}
            # referencia exacta de 2 estados
            q_ref = referencia_kleinberg_2_estados(r, d, S, GAMMA)
            ref = {y for y, q in zip(years, q_ref) if q == 1}
            pkg = set()
            acuerdo_ref += (bool(pipe) == bool(ref))
            acuerdo_pkg += (bool(pipe) == bool(pkg))
            anos_pipe += len(pipe); anos_ref += len(ref); anos_pkg += len(pkg)
            inter_ref += len(pipe & ref); inter_pkg += len(pipe & pkg)
            filas.append({"tipo": tipo, "entidad": ent, "total": sum(yc.values()),
                          "rafaga_2h": sorted(pipe), "rafaga_ref_2_estados": sorted(ref)})
        if not n_ent:
            continue
        j_ref = inter_ref / (anos_pipe + anos_ref - inter_ref) if (anos_pipe + anos_ref - inter_ref) else 1.0   # sin rafagas en ninguna = acuerdo
        j_pkg = inter_pkg / max(anos_pipe + anos_pkg - inter_pkg, 1)
        resumen.append((tipo, n_ent, acuerdo_ref / n_ent, j_ref, anos_pipe, anos_ref))

    L += ["## 1. Ráfagas de Kleinberg (2002)", "",
          "Concordancia de la condición «en ráfaga» (nivel ≥ 1). *Acuerdo por entidad*: la entidad tiene o no "
          "alguna ráfaga en ambas. *Jaccard*: coincidencia de los años en ráfaga.", "",
          "| Entidades | n | Acuerdo por entidad | Jaccard de años | Años en ráfaga (2h / referencia) |",
          "|---|---|---|---|---|"]
    for t, n, ar, jr, a1, a2 in resumen:
        L.append(f"| {t} | {n} | {ar:.3f} | {jr:.3f} | {a1} / {a2} |")
    L += ["", "- La referencia exacta implementa Viterbi completo con 2 estados; la 2h usa varios niveles, así que "
          "pueden diferir donde un nivel alto cambia el camino óptimo.",
          "- El paquete `burst_detection` se descartó como contraste: no corre con NumPy actual y decodifica de forma voraz.", ""]
    pd.DataFrame(filas).to_csv(OUT / "rafagas_detalle.csv", index=False, encoding="utf-8")

    # 2. Sigma
    data = json.loads((ROOT / "output" / "citespace" / "citespace_data.json").read_text(encoding="utf-8"))
    sig = data.get("sigma_rankings") or []
    if isinstance(sig, dict):
        sig = next((v for v in sig.values() if isinstance(v, list)), [])
    errores = []
    for row in sig:
        bc, b, sg = row.get("betweenness"), row.get("burstness"), row.get("sigma")
        if None in (bc, b, sg):
            continue
        esperado = (bc + 1) ** b if b > 0 else bc + 1
        errores.append(abs(esperado - sg))
    L += ["## 2. Sigma (Chen et al., 2009)", ""]
    if errores:
        L.append(f"- Filas comprobadas: {len(errores)}; error absoluto máximo frente a (betweenness + 1)^burstness: "
                 f"{max(errores):.2e} (los valores guardados están redondeados).")
    else:
        L.append("- No hay filas de sigma guardadas para comprobar.")
    L += ["- Advertencia de método: la betweenness se calcula con `weight=\"weight\"`, y networkx interpreta el peso "
          "como **distancia**. En co-citación el peso es fuerza de vínculo, así que los vínculos fuertes cuentan como "
          "caminos largos. Debe revisarse (usar 1/peso como distancia o betweenness sin pesos).", ""]
    (OUT / "informe_validacion_citespace.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
