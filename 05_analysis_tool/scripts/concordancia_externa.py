"""Concordancia entre el pipeline y herramientas externas (bibliometrix y VOSviewer).

Compara, sobre el mismo corpus:
  A. Estructura (clusters) de 4 redes: pipeline (.graphml, Louvain python-louvain),
     VOSviewer 1.6.20 (re-agrupado por el usuario, validacion_externa/vosviewer/) y
     bibliometrix 5.2.1 (scripts/validacion_bibliometrix.R, validacion_externa/bibliometrix/).
     Metricas: ARI y NMI sobre los nodos comunes. Referencias para interpretarlas
     (sin umbrales fijados a mano):
       - estabilidad interna: ARI entre 20 semillas del MISMO algoritmo;
       - azar: ARI contra etiquetas permutadas (200 permutaciones).
  B. Indicadores y conteos: lo que muestran las figuras del pipeline (datos embebidos
     en output/plotly/*.html) frente a las tablas de bibliometrix.

Uso (desde la raiz del proyecto):
    .venv\\Scripts\\python.exe scripts/concordancia_externa.py

Salidas en validacion_externa/: informe_concordancia.md, concordancia_resultados.json
y concordancia_detalle/*.csv (emparejamientos y discrepancias, para auditar).
No modifica nada del pipeline.
"""
from __future__ import annotations

import base64
import hashlib
import itertools
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import OPENALEX_CACHE_DIR  # noqa: E402
from src.export_integration import _iso2_to_country_name, _resolve_cr_from_cache  # noqa: E402
from src.utils import DiskCache  # noqa: E402

VAL = ROOT / "validacion_externa"
BX = VAL / "bibliometrix"
VOS = VAL / "vosviewer"
DET = VAL / "concordancia_detalle"
NETWORKS = ROOT / "output" / "networks"
VOS_PIPE = ROOT / "output" / "vosviewer"
PLOTLY = ROOT / "output" / "plotly"

SEMILLAS = list(range(1, 21))
N_PERMUTACIONES = 200
SEMILLA_PERMUTACION = 0

# Redes comparadas: nombre pipeline -> (archivo bibliometrix, criterio de emparejamiento)
REDES = {
    "bibliographic_coupling": ("acoplamiento", "clave de articulo"),
    "cocitation": ("cocitacion", "primer autor + ano de la referencia (formato de bibliometrix)"),
    "keyword_cooccurrence": ("coocurrencia_DE", "palabra clave en minusculas"),
    "coauthorship_countries": ("colaboracion_paises", "pais (ISO2)"),
}

# Diferencias de vocabulario de paises de bibliometrix frente al mapa ISO del pipeline.
# Se declaran explicitamente; los nombres que no se resuelven se reportan, no se descartan en silencio.
_ALIAS_PAIS_BX = {
    "KOREA": "KR", "UNITED KINGDOM": "GB", "U ARAB EMIRATES": "AE", "UNITED ARAB EMIRATES": "AE",
    "RUSSIA": "RU", "IRAN": "IR", "VIETNAM": "VN", "TURKEY": "TR", "TAIWAN": "TW",
    "CZECH REPUBLIC": "CZ", "SAUDI ARABIA": "SA",
}


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def _norm(s) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^A-Za-z0-9 ]+", " ", s.upper())
    return re.sub(r"\s+", " ", s).strip()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _leer_csv_bx(nombre: str) -> pd.DataFrame:
    df = pd.read_csv(BX / nombre)
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):   # pandas 3: texto es StringDtype, no object
            # fillna antes de astype: si no, una celda vacia se convierte en el texto 'nan'
            df[c] = df[c].fillna("").astype(str).str.strip()
    return df


def _trazas_plotly(nombre: str) -> list[dict]:
    html = (PLOTLY / f"{nombre}.html").read_text(encoding="utf-8")
    m = re.search(r'Plotly\.newPlot\(\s*"[^"]+",\s*(\[.*?\])\s*,\s*\{', html, re.S)
    if not m:
        raise ValueError(f"No se encontraron datos Plotly en {nombre}.html")
    trazas = json.loads(m.group(1))
    for t in trazas:
        for eje in ("x", "y"):
            v = t.get(eje)
            if isinstance(v, dict) and "bdata" in v:
                t[eje] = np.frombuffer(base64.b64decode(v["bdata"]), dtype=np.dtype(v["dtype"])).tolist()
    return trazas


_ISO_POR_NOMBRE = None


def _iso_de_nombre(nombre: str) -> str | None:
    """Nombre de pais (pipeline o bibliometrix) -> ISO2, usando el mismo mapa que escribe C1."""
    global _ISO_POR_NOMBRE
    if _ISO_POR_NOMBRE is None:
        import string
        _ISO_POR_NOMBRE = {}
        for a, b in itertools.product(string.ascii_uppercase, repeat=2):
            n = _iso2_to_country_name(a + b)
            if n and n != a + b:
                _ISO_POR_NOMBRE.setdefault(_norm(n), a + b)
    k = _norm(nombre)
    if k in _ALIAS_PAIS_BX:
        return _ALIAS_PAIS_BX[k]
    if k in _ISO_POR_NOMBRE:
        return _ISO_POR_NOMBRE[k]
    if len(k) == 2 and k.isalpha():
        return k
    return None


# ---------------------------------------------------------------------------
# A. Estructura: particiones
# ---------------------------------------------------------------------------

def _estabilidad(particiones: list[dict]) -> dict:
    """ARI entre todos los pares de particiones (mismos nodos)."""
    nodos = sorted(set.intersection(*[set(p) for p in particiones]))
    aris = [
        adjusted_rand_score([a[n] for n in nodos], [b[n] for n in nodos])
        for a, b in itertools.combinations(particiones, 2)
    ]
    return {
        "n_particiones": len(particiones),
        "ari_media": round(float(np.mean(aris)), 4),
        "ari_min": round(float(np.min(aris)), 4),
        "ari_p05": round(float(np.percentile(aris, 5)), 4),
    }


def _comparar(a: dict, b: dict, nombre_a: str, nombre_b: str) -> dict:
    comunes = sorted(set(a) & set(b))
    res = {
        "par": f"{nombre_a} vs {nombre_b}",
        "n_a": len(a), "n_b": len(b), "n_comunes": len(comunes),
        "cobertura_a": round(len(comunes) / len(a), 3) if a else 0.0,
        "cobertura_b": round(len(comunes) / len(b), 3) if b else 0.0,
    }
    if len(comunes) < 3:
        res["nota"] = "menos de 3 nodos comunes: no se calcula concordancia"
        return res
    la = [a[n] for n in comunes]
    lb = [b[n] for n in comunes]
    rng = np.random.default_rng(SEMILLA_PERMUTACION)
    nulos = [adjusted_rand_score(la, rng.permutation(lb)) for _ in range(N_PERMUTACIONES)]
    res.update({
        "k_a": len(set(la)), "k_b": len(set(lb)),
        "ari": round(float(adjusted_rand_score(la, lb)), 4),
        "nmi": round(float(normalized_mutual_info_score(la, lb)), 4),
        "ari_azar_media": round(float(np.mean(nulos)), 4),
        "ari_azar_p95": round(float(np.percentile(nulos, 95)), 4),
    })
    return res


def _pipeline_particiones(red: str) -> tuple[dict, list[dict], nx.Graph, dict]:
    """Particion del .graphml y la de 20 semillas con el mismo metodo del pipeline."""
    import community as community_louvain

    G = nx.read_graphml(NETWORKS / f"{red}.graphml")
    oficial = {n: int(d["community"]) for n, d in G.nodes(data=True) if "community" in d}
    semillas = [community_louvain.best_partition(G, random_state=s) for s in SEMILLAS]
    control = community_louvain.best_partition(G, random_state=42)
    reproduce = adjusted_rand_score(
        [oficial[n] for n in oficial], [control[n] for n in oficial]
    )
    return oficial, semillas, G, {"ari_semilla42_vs_graphml": round(float(reproduce), 4)}


# Variantes de normalizacion de VOSviewer: sufijo del map file guardado por el usuario.
# La primera es la principal (la que se usa en las tablas generales).
VOS_VARIANTES = {
    "LinLog/modularity": "_map_linLog.txt",          # la que se reporta en el paper
    "association strength": "_map.txt",
    "sin normalizacion": "_map_sin_normalizacion.txt",
}


def _vos_particion(red: str, G: nx.Graph, sufijo: str = "_map.txt") -> dict:
    """Clusters de VOSviewer, llevados a los nodos del .graphml por id (orden de G.nodes())."""
    nodos = list(G.nodes())
    v = pd.read_csv(VOS / f"{red}{sufijo}", sep="\t", encoding="utf-8-sig", quoting=3)
    p = pd.read_csv(VOS_PIPE / f"{red}_map.txt", sep="\t", encoding="utf-8-sig", quoting=3)
    if len(v) != len(nodos) or len(p) != len(nodos):
        raise ValueError(f"{red}: tamanos distintos (graphml {len(nodos)}, map {len(p)}, VOSviewer {len(v)})")
    for i, lab in zip(p["id"], p["label"]):
        if str(G.nodes[nodos[i - 1]].get("label", nodos[i - 1])) != str(lab).strip('"'):   # VOSviewer cita etiquetas con comas
            raise ValueError(f"{red}: el id {i} no corresponde al mismo nodo en graphml y map file")
    return {nodos[int(i) - 1]: int(c) for i, c in zip(v["id"], v["cluster"])}


def _clave_cr_bibliometrix(cr: str) -> str:
    """'HU L, 1999, JOURNAL, ...' -> 'HU L 1999', como etiqueta bibliometrix los nodos de co-citacion."""
    partes = [x.strip() for x in cr.split(",")]
    return _norm(" ".join(partes[:2]))


_PERDIDA_CR: dict = {}


def _mapa_cocitacion_pipeline(G: nx.Graph) -> tuple[dict, dict]:
    """W-id del pipeline -> clave 'AUTOR AÑO' usando la misma resolucion que escribe el CR del export.

    Se resuelve referencia por referencia: _resolve_cr_from_cache omite las que no puede
    resolver, asi que la lista completa no se alinea con referenced_works. De paso se cuenta
    cuantas referencias no llegan al CR del export (bibliometrix no las ve).
    """
    arts = json.loads((ROOT / "data" / "enriched_articles.json").read_text(encoding="utf-8"))
    cache = DiskCache(OPENALEX_CACHE_DIR)
    mem: dict = {}
    w_a_clave: dict[str, str] = {}
    n_refs = n_sin_cr = 0
    sin_cr: set = set()
    for a in arts:
        rr = a.get("references_resolved", [])
        for w in a.get("referenced_works", []):
            wid = w.split("/")[-1]
            n_refs += 1
            cr = _resolve_cr_from_cache([w], cache, mem_cache=mem, refs_resolved=rr)
            if not cr:
                n_sin_cr += 1
                sin_cr.add(wid)
                continue
            w_a_clave.setdefault(wid, _clave_cr_bibliometrix(cr[0]))
    _PERDIDA_CR.update({
        "referencias_totales": n_refs,
        "referencias_sin_CR_en_export": n_sin_cr,
        "obras_distintas_sin_CR": len(sin_cr),
        "nodos_cocitacion_sin_CR": sum(1 for n in G.nodes() if n in sin_cr),
    })
    return w_a_clave, {n: G.nodes[n].get("cocitation_count", 0) for n in G.nodes()}


def _proyectar(part: dict, clave_de: dict, peso: dict | None = None) -> tuple[dict, int]:
    """Lleva una particion a otro espacio de claves; si varias caen en la misma clave,
    se queda con el nodo de mayor peso (y cuenta la colision)."""
    out: dict = {}
    mejor: dict = {}
    colisiones = 0
    for n, c in part.items():
        k = clave_de.get(n)
        if not k:
            continue
        w = (peso or {}).get(n, 0)
        if k in out:
            colisiones += 1
            if w <= mejor[k]:
                continue
        out[k] = c
        mejor[k] = w
    return out, colisiones


def _association_strength(G: nx.Graph) -> nx.Graph:
    """Normalizacion de VOSviewer (van Eck y Waltman, 2009): a_ij = 2m c_ij / (s_i s_j)."""
    s = dict(G.degree(weight="weight"))
    m = sum(s.values()) / 2
    H = nx.Graph()
    H.add_nodes_from(G)
    for u, v, d in G.edges(data=True):
        H.add_edge(u, v, weight=2 * m * float(d.get("weight", 1)) / (s[u] * s[v]))
    return H


def sensibilidad_normalizacion(G: nx.Graph, vos: dict, bx_por_nodo: dict) -> list[dict]:
    """Louvain del pipeline sobre conteos brutos y sobre association strength (10 semillas),
    comparado con VOSviewer y con bibliometrix. Aisla el efecto de la normalizacion."""
    import community as community_louvain

    G = G.copy()
    G.remove_nodes_from([n for n in list(G) if G.degree(n) == 0])
    nodos_v = [n for n in G if n in vos]
    nodos_b = [n for n in G if bx_por_nodo.get(n) is not None]
    filas = []
    for etiqueta, grafo in (("conteos brutos (pipeline actual)", G), ("association strength", _association_strength(G))):
        ps = [community_louvain.best_partition(grafo, random_state=s) for s in range(1, 11)]
        filas.append({
            "normalizacion": etiqueta,
            "k_medio": round(float(np.mean([len(set(p.values())) for p in ps])), 1),
            "ari_vs_vosviewer": round(float(np.mean([adjusted_rand_score([p[n] for n in nodos_v], [vos[n] for n in nodos_v]) for p in ps])), 3),
            "ari_vs_bibliometrix": round(float(np.mean([adjusted_rand_score([p[n] for n in nodos_b], [bx_por_nodo[n] for n in nodos_b]) for p in ps])), 3) if len(nodos_b) > 2 else None,
        })
    return filas


def analizar_red(red: str) -> dict:
    archivo_bx, criterio = REDES[red]
    oficial, semillas_pipe, G, control = _pipeline_particiones(red)
    vos_desactualizado = {}
    vos, vos_principal = {}, None
    for nombre_var, sufijo in VOS_VARIANTES.items():      # principal: la primera variante disponible y vigente
        if not (VOS / f"{red}{sufijo}").exists():
            continue
        try:
            vos = _vos_particion(red, G, sufijo)
            vos_principal = nombre_var
            break
        except ValueError as e:
            # El map file de VOSviewer es de una version anterior de la red: no se puede emparejar
            vos_desactualizado[nombre_var] = str(e)
    # Los nodos aislados forman cada uno su propio cluster en todas las herramientas: no
    # aportan estructura e inflan k. Se excluyen de las comparaciones y se reporta cuantos son.
    aislados = {n for n in G.nodes() if G.degree(n) == 0}
    oficial = {n: c for n, c in oficial.items() if n not in aislados}
    vos = {n: c for n, c in vos.items() if n not in aislados}
    semillas_pipe = [{n: c for n, c in p.items() if n not in aislados} for p in semillas_pipe]

    bx_ref = _leer_csv_bx(f"12_{archivo_bx}_clusters_igraph.csv")
    bx_sem = _leer_csv_bx(f"12_{archivo_bx}_louvain_semillas.csv")
    cols_sem = [c for c in bx_sem.columns if c.startswith("semilla_")]

    # Claves comunes con bibliometrix segun el tipo de red
    nota_emp = ""
    if red == "bibliographic_coupling":
        clave_bx = lambda s: str(s).split()[0]                       # 'Ahn2014 NA 2014' -> 'Ahn2014'
        clave_pipe = {n: n for n in G.nodes()}
        peso = None
    elif red == "cocitation":
        clave_bx = _norm
        clave_pipe, peso = _mapa_cocitacion_pipeline(G)
        clave_pipe = {n: clave_pipe.get(n) for n in G.nodes()}
    elif red == "keyword_cooccurrence":
        clave_bx = lambda s: str(s).strip().lower()
        clave_pipe = {n: str(n).strip().lower() for n in G.nodes()}
        peso = None
        nota_emp = ("El pipeline canonicaliza palabras clave (sinonimos de knowledge_base.json, "
                    "singularizacion); bibliometrix usa DE crudas. La cobertura mide ese efecto.")
    else:
        clave_bx = lambda s: _iso_de_nombre(s) or f"?{s}"
        clave_pipe = {n: str(n).upper() for n in G.nodes()}
        peso = None

    bx = {clave_bx(n): int(c) for n, c in zip(bx_ref["nodo"], bx_ref["cluster"])}
    # Sensibilidad (solo palabras clave): red de bibliometrix sin umbral de co-ocurrencia
    sens_umbral1 = None
    f1 = BX / f"12_{archivo_bx}_umbral1_clusters_igraph.csv"
    if f1.exists():
        b1 = _leer_csv_bx(f1.name)
        bx1 = {clave_bx(n): int(c) for n, c in zip(b1["nodo"], b1["cluster"])}
    bx_semillas = [{clave_bx(n): int(c) for n, c in zip(bx_sem["nodo"], bx_sem[col])} for col in cols_sem]

    pipe_k, col_pipe = _proyectar(oficial, clave_pipe, peso)
    if f1.exists():
        sens_umbral1 = {**_comparar(pipe_k, bx1, "pipeline", "bibliometrix (sin umbral)"), "nodos_bibliometrix": len(bx1)}
        fs1 = BX / f"12_{archivo_bx}_umbral1_louvain_semillas.csv"
        if fs1.exists():
            s1 = _leer_csv_bx(fs1.name)
            sens_umbral1["estabilidad_bibliometrix_sin_umbral"] = _estabilidad(
                [{clave_bx(n): int(c) for n, c in zip(s1["nodo"], s1[col])} for col in s1.columns if col.startswith("semilla_")]
            )["ari_media"]
        sens_umbral1["estabilidad_bibliometrix_con_umbral"] = _estabilidad(bx_semillas)["ari_media"]
        sens_umbral1["estabilidad_pipeline"] = _estabilidad(semillas_pipe)["ari_media"]
        # Mismo umbral, palabras crudas (DE del export) frente a las normalizadas del pipeline
        import community as community_louvain
        exp = pd.read_csv(ROOT / "output" / "bibliometrix_export.csv", dtype=str, keep_default_na=False)
        G_crudo = _grafo_de_listas([[k.strip().lower() for k in de.split(";") if k.strip()] for de in exp["DE"]], min_cooc=2)
        comunes = [n for n in G_crudo if n in oficial]
        if len(comunes) > 2:
            aris = [adjusted_rand_score([p_[n] for n in comunes], [oficial[n] for n in comunes])
                    for p_ in (community_louvain.best_partition(G_crudo, random_state=s_) for s_ in range(1, 11))]
            sens_umbral1["ari_crudas_vs_normalizadas_mismo_umbral"] = round(float(np.mean(aris)), 4)
    vos_k, _ = _proyectar(vos, clave_pipe, peso)

    no_resueltos = sorted(k for k in bx if k.startswith("?")) if red == "coauthorship_countries" else []

    comp = [
        *([_comparar(oficial, vos, "pipeline", "VOSviewer")] if vos else []),
        _comparar(pipe_k, bx, "pipeline", "bibliometrix"),
        *([_comparar(vos_k, bx, "VOSviewer", "bibliometrix")] if vos else []),
    ]

    DET.mkdir(parents=True, exist_ok=True)
    filas = []
    for n in G.nodes():
        k = clave_pipe.get(n)
        filas.append({"nodo_pipeline": n, "clave_bibliometrix": k,
                      "cluster_pipeline": oficial.get(n), "cluster_vosviewer": vos.get(n),
                      "cluster_bibliometrix": bx.get(k) if k else None})
    pd.DataFrame(filas).to_csv(DET / f"{red}_emparejamiento.csv", index=False, encoding="utf-8")

    bx_por_nodo = {n: bx.get(clave_pipe.get(n)) for n in G.nodes() if clave_pipe.get(n)}
    sens = sensibilidad_normalizacion(G, vos, bx_por_nodo) if vos else []

    # Variantes de normalizacion de VOSviewer presentes en la carpeta
    variantes = []
    for nombre_var, sufijo in VOS_VARIANTES.items():
        if not (VOS / f"{red}{sufijo}").exists():
            continue
        try:
            pv = {n: c for n, c in _vos_particion(red, G, sufijo).items() if n not in aislados}
        except ValueError as e:
            vos_desactualizado[nombre_var] = str(e)
            continue
        pv_k, _ = _proyectar(pv, clave_pipe, peso)
        c1 = _comparar(oficial, pv, "pipeline", f"VOSviewer ({nombre_var})")
        c2 = _comparar(pv_k, bx, f"VOSviewer ({nombre_var})", "bibliometrix")
        variantes.append({"normalizacion_vosviewer": nombre_var, "archivo": f"{red}{sufijo}",
                          "vs_pipeline": c1, "vs_bibliometrix": c2})

    return {
        "red": red,
        "criterio_emparejamiento_bibliometrix": criterio,
        "nota": nota_emp,
        "nodos": {"pipeline": len(oficial), "vosviewer": len(vos), "bibliometrix": len(bx)},
        "vosviewer_desactualizado": vos_desactualizado,
        "sensibilidad_umbral_bibliometrix": sens_umbral1,
        "vosviewer_principal": vos_principal,
        "aislados_excluidos": len(aislados),
        "perdida_cr_export": dict(_PERDIDA_CR) if red == "cocitation" else {},
        "colisiones_pipeline_en_clave_bibliometrix": col_pipe,
        "paises_bibliometrix_no_resueltos": no_resueltos,
        "control_reproduccion_pipeline": control,
        "estabilidad": {
            "pipeline (python-louvain, 20 semillas)": _estabilidad(semillas_pipe),
            "bibliometrix (igraph Louvain, 20 semillas)": _estabilidad(bx_semillas),
        },
        "comparaciones": comp,
        "sensibilidad_normalizacion": sens,
        "variantes_vosviewer": variantes,
    }


# ---------------------------------------------------------------------------
# B. Indicadores y conteos
# ---------------------------------------------------------------------------

def indicadores() -> dict:
    out: dict = {}

    # Produccion anual
    t = _trazas_plotly("bar_docs_per_year")[0]
    pipe = {int(x): int(y) for x, y in zip(t["x"], t["y"])}
    bxp = _leer_csv_bx("02_produccion_anual.csv")
    bxd = {int(y): int(a) for y, a in zip(bxp["Year"], bxp["Articles"])}
    anos = sorted(set(pipe) | set(bxd))
    dif = [{"ano": a, "pipeline": pipe.get(a), "bibliometrix": bxd.get(a)} for a in anos if pipe.get(a) != bxd.get(a)]
    out["produccion_anual"] = {"anos": len(anos), "total_pipeline": sum(pipe.values()),
                               "total_bibliometrix": sum(bxd.values()), "diferencias": dif}

    # Crecimiento
    rep = (ROOT / "output" / "reports" / "supplementary_indicators_report.md").read_text(encoding="utf-8")
    m = re.search(r"\*\*CAGR:\*\*\s*([\d.]+)%", rep)
    out["crecimiento_anual_pct"] = {
        "pipeline": float(m.group(1)) if m else None,
        "bibliometrix": float(_leer_csv_bx("02b_tasa_crecimiento.csv")["valor"].iloc[0]),
    }

    # Colaboracion internacional por articulo (conjunto de paises)
    pr = _leer_csv_bx("00_paises_por_registro.csv").replace("nan", "")
    filas = []
    for _, r in pr.iterrows():
        sb = {(_iso_de_nombre(x) or f"?{x}") for x in str(r["AU_CO_bibliometrix"]).split(";") if x}
        sp = {(_iso_de_nombre(x) or f"?{x}") for x in str(r["AU_CO_PIPELINE"]).split(";") if x}
        filas.append({"KEY": r["KEY"], "paises_pipeline": ";".join(sorted(sp)),
                      "paises_bibliometrix": ";".join(sorted(sb)),
                      "mcp_pipeline": len(sp) > 1, "mcp_bibliometrix": len(sb) > 1,
                      "sin_pais_pipeline": not sp, "sin_pais_bibliometrix": not sb})
    dfp = pd.DataFrame(filas)
    dfp[(dfp.mcp_pipeline != dfp.mcp_bibliometrix) | (dfp.paises_pipeline != dfp.paises_bibliometrix)].to_csv(
        DET / "paises_discrepancias.csv", index=False, encoding="utf-8")
    out["colaboracion_internacional"] = {
        "mcp_pipeline": int(dfp.mcp_pipeline.sum()), "mcp_bibliometrix": int(dfp.mcp_bibliometrix.sum()),
        "articulos_mcp_discrepantes": dfp[dfp.mcp_pipeline != dfp.mcp_bibliometrix]["KEY"].tolist(),
        "conjuntos_de_paises_identicos": int((dfp.paises_pipeline == dfp.paises_bibliometrix).sum()),
        "sin_pais_pipeline": int(dfp.sin_pais_pipeline.sum()),
        "sin_pais_bibliometrix": int(dfp.sin_pais_bibliometrix.sum()),
        "detalle": "concordancia_detalle/paises_discrepancias.csv",
    }
    au1 = (pr["AU1_CO_bibliometrix"].map(lambda x: _iso_de_nombre(x) if x else "")
           == pr["AU1_CO_PIPELINE"].map(lambda x: _iso_de_nombre(x) if x else ""))
    out["pais_autor_correspondencia_coincide"] = {"coinciden": int(au1.sum()), "de": int(len(au1))}

    # Rankings top: fuentes y autores
    def ranking(plot: str, csv: str, col: str, norm_pipe, norm_bx):
        t = _trazas_plotly(plot)[0]
        pipe = sorted(zip(t["y"], t["x"]), key=lambda z: -z[1])
        b = _leer_csv_bx(csv)
        bxr = list(zip(b[col], b["Articles"]))
        n = min(len(pipe), len(bxr))
        kp = [norm_pipe(x) for x, _ in pipe[:n]]
        kb = [norm_bx(x) for x, _ in bxr[:n]]
        comun = [k for k in kp if k in kb]
        rho = spearmanr([kp.index(k) for k in comun], [kb.index(k) for k in comun]).statistic if len(comun) > 2 else None
        pd.DataFrame({"pipeline": [x for x, _ in pipe[:n]], "n_pipeline": [v for _, v in pipe[:n]],
                      "bibliometrix": [x for x, _ in bxr[:n]], "n_bibliometrix": [v for _, v in bxr[:n]]}
                     ).to_csv(DET / f"ranking_{plot}.csv", index=False, encoding="utf-8")
        corte = pipe[n - 1][1]
        return {"top_n": n, "valor_de_corte": corte,
                "empatados_en_el_corte_pipeline": sum(1 for _, v in pipe if v == corte),
                "jaccard": round(len(set(kp) & set(kb)) / len(set(kp) | set(kb)), 3),
                "spearman_en_comunes": None if rho is None or np.isnan(rho) else round(float(rho), 3),
                "maximo_pipeline": pipe[0], "maximo_bibliometrix": bxr[0]}

    out["fuentes_top"] = ranking("bar_sources_docs", "03_fuentes_top.csv", "Sources",
                                 lambda s: _norm(s)[:40], lambda s: _norm(s)[:40])

    def autor_pipe(s):  # 'Oliveira, Soraia' -> 'OLIVEIRA S'
        ap, _, nom = str(s).partition(",")
        return _norm(f"{ap} {nom.strip()[:1]}")

    def autor_bx(s):  # 'OLIVEIRA SORAIA' / 'TRUYOL ME' -> 'OLIVEIRA S' / 'TRUYOL M'
        p = _norm(s).split()
        return f"{p[0]} {p[1][:1]}" if len(p) > 1 else (p[0] if p else "")

    out["autores_top"] = ranking("bar_authors_docs", "04_autores_top.csv", "Authors", autor_pipe, autor_bx)

    # Lotka
    t = _trazas_plotly("bar_lotka")[0]
    lb = _leer_csv_bx("08_lotka_productividad.csv")
    out["lotka_autores_por_productividad"] = {
        "pipeline": {int(x): int(y) for x, y in zip(t["x"], t["y"])},
        "bibliometrix": {int(d): int(n) for d, n in zip(lb["Documents written"], lb["N. of Authors"])},
        "ajuste_bibliometrix": _leer_csv_bx("08b_lotka_ajuste.csv").iloc[0].to_dict(),
    }

    # RPYS
    tr = _trazas_plotly("rpys")
    refs = {int(x): float(y) for x, y in zip(tr[0]["x"], tr[0]["y"])}
    desv = {int(x): float(y) for x, y in zip(tr[2]["x"], tr[2]["y"])}
    rb = _leer_csv_bx("11_rpys_por_ano.csv")
    bref = {int(y): float(c) for y, c in zip(rb["Year"], rb["Citations"])}
    bdesv = {int(y): float(d) for y, d in zip(rb["Year"], rb["diffMedian5"])}
    anos = sorted(set(refs) & set(bref))
    rho = spearmanr([refs[a] for a in anos], [bref[a] for a in anos]).statistic
    pico_p = [a for a, _ in sorted(desv.items(), key=lambda z: -z[1])[:10]]
    pico_b = [a for a, _ in sorted(bdesv.items(), key=lambda z: -z[1])[:10]]
    out["rpys"] = {
        "anos_comunes": len(anos), "rango_pipeline": [min(refs), max(refs)], "rango_bibliometrix": [min(bref), max(bref)],
        "spearman_citas_por_ano": round(float(rho), 3),
        "citas_totales_pipeline": sum(refs[a] for a in anos), "citas_totales_bibliometrix": sum(bref[a] for a in anos),
        "picos_top10_pipeline": sorted(pico_p), "picos_top10_bibliometrix": sorted(pico_b),
        "picos_comunes": sorted(set(pico_p) & set(pico_b)),
        "nota": "Ambos: desviacion respecto de la mediana de 5 anos (pipeline: Fase 2g, RPYS estandar; bibliometrix: diffMedian5). bibliometrix solo ve las referencias que llegan al campo CR del export.",
    }
    return out


# ---------------------------------------------------------------------------
# C. Mapa tematico (Callon) y tendencia de topicos
# ---------------------------------------------------------------------------

def _cuadrante(c: float, d: float, med_c: float, med_d: float) -> str:
    """Diagrama estrategico de Callon, con corte en la mediana (regla del pipeline)."""
    if c > med_c and d > med_d:
        return "motor"
    if c > med_c:
        return "basico/transversal"
    if d > med_d:
        return "nicho"
    return "emergente o en declive"


def _cuadrantes(clusters: list[dict], kc: str, kd: str) -> dict:
    med_c = float(np.median([c[kc] for c in clusters]))
    med_d = float(np.median([c[kd] for c in clusters]))
    return {c["id"]: _cuadrante(c[kc], c[kd], med_c, med_d) for c in clusters}


def _grafo_de_listas(listas: list[list[str]], min_cooc: int = 2) -> nx.Graph:
    """Co-ocurrencia con la misma regla que src.strategic_thematic_analysis._build_kw_cooccurrence."""
    freq: Counter = Counter()
    cooc: Counter = Counter()
    for kws in listas:
        kws = sorted(set(kws))
        freq.update(kws)
        for i in range(len(kws)):
            for j in range(i + 1, len(kws)):
                cooc[(kws[i], kws[j])] += 1
    G = nx.Graph()
    for (a, b), n in cooc.items():
        if n >= min_cooc:
            G.add_node(a, frequency=freq[a])
            G.add_node(b, frequency=freq[b])
            G.add_edge(a, b, weight=n)
    return G


def _comparar_temas(pipe_clusters: list[dict], bx_words: pd.DataFrame, bx_clusters: pd.DataFrame, etiqueta: str) -> dict:
    """Pertenencia (ARI sobre palabras comunes), emparejamiento de temas (Jaccard) y cuadrantes."""
    pal_pipe = {kw: c["id"] for c in pipe_clusters for kw in c["keywords"]}
    pal_bx = {str(w).strip().lower(): int(c) for w, c in zip(bx_words["Words"], bx_words["Cluster"])}
    comp = _comparar(pal_pipe, pal_bx, etiqueta, "bibliometrix")
    comunes = set(pal_pipe) & set(pal_bx)

    cuad_p = _cuadrantes(pipe_clusters, "centrality", "density")
    bxc = [{"id": int(g), "centrality": float(c), "density": float(d), "name": str(n)}
           for g, c, d, n in zip(bx_clusters["groups"], bx_clusters["centrality"], bx_clusters["density"], bx_clusters["name"])]
    cuad_b = _cuadrantes(bxc, "centrality", "density")
    nombre_p = {c["id"]: c["name"] for c in pipe_clusters}

    pares = []
    for b in bxc:
        wb = {w for w, c in pal_bx.items() if c == b["id"] and w in comunes}
        mejor, jac = None, 0.0
        for p in pipe_clusters:
            wp = {w for w in p["keywords"] if w in comunes}
            j = len(wb & wp) / len(wb | wp) if (wb | wp) else 0.0
            if j > jac:
                mejor, jac = p["id"], j
        pares.append({
            "tema_bibliometrix": b["name"], "cuadrante_bibliometrix": cuad_b[b["id"]],
            "palabras_comunes_en_el_tema": len(wb),
            "tema_pipeline_mas_parecido": nombre_p.get(mejor, "—"),
            "cuadrante_pipeline": cuad_p.get(mejor, "—"),
            "jaccard": round(jac, 3),
            "mismo_cuadrante": mejor is not None and cuad_p.get(mejor) == cuad_b[b["id"]],
        })
    return {"pertenencia": comp, "palabras_comunes": len(comunes),
            "k_pipeline": len(pipe_clusters), "k_bibliometrix": len(bxc), "temas": pares}


def comparar_mapa_tematico() -> dict:
    """Mapa tematico del pipeline (Fase 3b) frente a thematicMap de bibliometrix (campo DE).

    Dos versiones del lado del pipeline, para separar causas:
      1. la oficial: red de palabras canonicalizadas de la Fase 3 (lo que se publica);
      2. el metodo del pipeline sobre las palabras clave de autor crudas (DE del export),
         que son las mismas que usa bibliometrix. Si (2) concuerda mucho mas que (1), la
         diferencia la produce la canonicalizacion/fuentes de palabras, no el metodo.
    """
    from src.strategic_thematic_analysis import compute_thematic_map

    arts = json.loads((ROOT / "data" / "enriched_articles.json").read_text(encoding="utf-8"))
    G = nx.read_graphml(NETWORKS / "keyword_cooccurrence.graphml")
    for u, v, d in G.edges(data=True):
        d["weight"] = float(d.get("weight", 1))
    for n, d in G.nodes(data=True):
        d["frequency"] = int(float(d.get("frequency", 0)))
    oficial = compute_thematic_map(arts, kw_network=G)["clusters"]

    # Control: el recalculo reproduce la figura publicada (datos embebidos en el HTML)
    t = _trazas_plotly("thematic_map_strategic_diagram")[0]
    fig_c = sorted(round(float(x), 3) for x in t["x"])
    rec_c = sorted(round(float(c["centrality"]), 3) for c in oficial)
    control = {"clusters_figura": len(fig_c), "clusters_recalculo": len(rec_c),
               "centralidades_identicas": fig_c == rec_c}

    exp = pd.read_csv(ROOT / "output" / "bibliometrix_export.csv", dtype=str, keep_default_na=False)
    listas = [[k.strip().lower() for k in str(de).split(";") if k.strip()] for de in exp["DE"]]
    G_crudo = _grafo_de_listas(listas)
    crudo = compute_thematic_map(arts, kw_network=G_crudo)["clusters"]

    bx_words = _leer_csv_bx("09_mapa_tematico_DE_palabras.csv")
    bx_clusters = _leer_csv_bx("09_mapa_tematico_DE_clusters.csv")
    return {
        "control_figura": control,
        "oficial": _comparar_temas(oficial, bx_words, bx_clusters, "pipeline (palabras canonicalizadas)"),
        "metodo_pipeline_sobre_DE_crudas": _comparar_temas(crudo, bx_words, bx_clusters, "pipeline sobre DE crudas"),
        "nota": "bibliometrix: Louvain sobre las 250 palabras mas frecuentes (minfreq 5). "
                "Pipeline: Louvain (semilla 42) sobre toda la red con co-ocurrencia >= 2 y clusters de 3 o mas palabras.",
    }


def comparar_tendencia_topicos() -> dict:
    """Tendencia de topicos: ano mediano de uso de cada palabra, pipeline vs bibliometrix (DE).

    bibliometrix (fieldByYear) ubica cada palabra en el ano mediano de los documentos que la usan.
    La figura del pipeline muestra otra cosa (las 5 palabras mas frecuentes de cada ano). Para
    comparar lo mismo, el ano mediano se calcula aqui con las palabras y conteos del pipeline.
    """
    from src.supplementary_indicators import compute_trend_topics

    arts = json.loads((ROOT / "data" / "enriched_articles.json").read_text(encoding="utf-8"))
    tt = compute_trend_topics(arts, top_n_per_year=5, min_freq=2)
    anos_por_kw: dict[str, list[int]] = defaultdict(list)
    for y, c in tt["kw_year"].items():
        for kw, n in c.items():
            anos_por_kw[kw].extend([int(y)] * int(n))
    pipe = {kw: (len(ys), float(np.median(ys))) for kw, ys in anos_por_kw.items() if len(ys) >= 5}

    archivo = BX / "10_tendencia_topicos_DE.csv"
    if not archivo.exists():
        return {"nota": "Falta 10_tendencia_topicos_DE.csv: vuelva a correr scripts/validacion_bibliometrix.R"}
    b = _leer_csv_bx(archivo.name)
    # fieldByYear devuelve todas las palabras; min.freq solo filtra lo que grafica
    bx = {str(i).strip().lower(): (int(f), float(m)) for i, f, m in zip(b["item"], b["freq"], b["year_med"]) if int(f) >= 5}
    comunes = sorted(set(pipe) & set(bx))
    rho = spearmanr([pipe[k][1] for k in comunes], [bx[k][1] for k in comunes]).statistic if len(comunes) > 2 else None
    dif = [abs(pipe[k][1] - bx[k][1]) for k in comunes]
    def recientes(d):
        return [k for k, _ in sorted(d.items(), key=lambda kv: (-kv[1][1], -kv[1][0], kv[0]))[:15]]
    rp, rb = recientes(pipe), recientes(bx)

    # Que muestra la figura del pipeline: palabras mas frecuentes por ano
    por_ano = defaultdict(list)
    for r in tt["trend_data"]:
        por_ano[r["year"]].append(r["keyword"])
    primeras = Counter(v[0] for v in por_ano.values() if v)
    pd.DataFrame([{"palabra": k, "n_pipeline": pipe.get(k, (0, None))[0], "ano_mediano_pipeline": pipe.get(k, (0, None))[1],
                   "n_bibliometrix": bx.get(k, (0, None))[0], "ano_mediano_bibliometrix": bx.get(k, (0, None))[1]}
                  for k in sorted(set(pipe) | set(bx))]).to_csv(DET / "tendencia_topicos.csv", index=False, encoding="utf-8")
    return {
        "palabras_freq5_pipeline": len(pipe), "palabras_freq5_bibliometrix": len(bx), "comunes": len(comunes),
        "spearman_ano_mediano": None if rho is None or np.isnan(rho) else round(float(rho), 3),
        "diferencia_media_anos": round(float(np.mean(dif)), 2) if dif else None,
        "coinciden_exacto_ano_mediano": int(sum(d == 0 for d in dif)),
        "mas_recientes_pipeline": rp, "mas_recientes_bibliometrix": rb,
        "recientes_en_comun": sorted(set(rp) & set(rb)),
        "figura_pipeline": {
            "metodo": "5 palabras mas frecuentes de cada ano (no ano mediano)",
            "palabra_mas_frecuente_por_ano": dict(primeras.most_common()),
            "anos": len(por_ano),
        },
        "detalle": "concordancia_detalle/tendencia_topicos.csv",
    }


# ---------------------------------------------------------------------------
# Informe
# ---------------------------------------------------------------------------

def _fmt(v):
    return "—" if v is None else v


def escribir_informe(redes: list[dict], ind: dict, entradas: dict) -> Path:
    L = ["# Informe de concordancia externa", "",
         f"Generado: {datetime.now():%Y-%m-%d %H:%M} con `scripts/concordancia_externa.py`.", "",
         "## Entradas", "", "| Archivo | SHA-256 (16) |", "|---|---|"]
    L += [f"| `{k}` | `{v}` |" for k, v in entradas.items()]
    L += ["", "## Cómo leer las métricas de clusters", "",
          "- **ARI** (índice de Rand ajustado) y **NMI** (información mutua normalizada) sobre los nodos comunes.",
          "- No hay umbral fijado a mano. Cada valor se lee contra dos referencias medidas en los mismos datos:",
          "  - **estabilidad interna**: ARI entre 20 semillas del mismo algoritmo (cuánto varía la herramienta contra sí misma);",
          "  - **azar**: ARI contra etiquetas permutadas (200 permutaciones).",
          "- Si la concordancia entre herramientas se acerca a la estabilidad interna, la estructura no depende de la herramienta.",
          "  Si queda cerca del azar, la estructura reportada es un artefacto del algoritmo o de la construcción de la red.", "",
          "**Qué agrupa cada herramienta.** El pipeline y bibliometrix aplican Louvain (modularidad) sobre conteos brutos. "
          "La tabla principal usa VOSviewer con normalización **LinLog/modularity**, que optimiza el mismo criterio: "
          "la diferencia que queda es del algoritmo. La tabla de sensibilidad recalcula el Louvain del pipeline sobre "
          "*association strength* para mostrar cuánto cambia la estructura con la normalización. Parámetros de VOSviewer: "
          "`validacion_externa/vosviewer/parametros.txt`.", ""]

    L += ["## A. Estructura de las redes", ""]
    for r in redes:
        L += [f"### {r['red']}", "",
              f"- Nodos con al menos un enlace: pipeline {r['nodos']['pipeline']}, VOSviewer {r['nodos']['vosviewer']}, "
              f"bibliometrix {r['nodos']['bibliometrix']}. Aislados excluidos del pipeline/VOSviewer: {r['aislados_excluidos']}.",
              f"- Emparejamiento con bibliometrix: {r['criterio_emparejamiento_bibliometrix']}."
              + (f" Colisiones (varios nodos del pipeline en una misma clave): "
                 f"{r['colisiones_pipeline_en_clave_bibliometrix']}." if r['colisiones_pipeline_en_clave_bibliometrix'] else ""),
              f"- Control: la semilla 42 reproduce la partición del `.graphml` con ARI = "
              f"{r['control_reproduccion_pipeline']['ari_semilla42_vs_graphml']}."]
        if r["nota"]:
            L.append(f"- {r['nota']}")
        if r.get("perdida_cr_export"):
            p = r["perdida_cr_export"]
            L.append(f"- Referencias que no llegan al campo CR del export (bibliometrix no las ve): "
                     f"{p['referencias_sin_CR_en_export']} de {p['referencias_totales']} "
                     f"({p['obras_distintas_sin_CR']} obras distintas; {p['nodos_cocitacion_sin_CR']} son nodos de esta red).")
        if r["paises_bibliometrix_no_resueltos"]:
            L.append(f"- Países de bibliometrix sin código ISO: {', '.join(r['paises_bibliometrix_no_resueltos'])}.")
        L += ["", "| Estabilidad interna | ARI medio | ARI mínimo | ARI p5 |", "|---|---|---|---|"]
        for k, v in r["estabilidad"].items():
            L.append(f"| {k} | {v['ari_media']} | {v['ari_min']} | {v['ari_p05']} |")
        L += ["", "| Comparación | Nodos comunes | Cobertura (a / b) | k (a / b) | ARI | NMI | ARI azar (media / p95) |",
              "|---|---|---|---|---|---|---|"]
        for c in r["comparaciones"]:
            if "ari" not in c:
                L.append(f"| {c['par']} | {c['n_comunes']} | {c['cobertura_a']} / {c['cobertura_b']} | — | — | — | {c.get('nota', '')} |")
                continue
            L.append(f"| {c['par']} | {c['n_comunes']} | {c['cobertura_a']} / {c['cobertura_b']} | "
                     f"{c['k_a']} / {c['k_b']} | {c['ari']} | {c['nmi']} | {c['ari_azar_media']} / {c['ari_azar_p95']} |")
        L.append(f"- VOSviewer usado en la tabla principal: {r.get('vosviewer_principal') or 'ninguno (no hay map file vigente)'}.")
        su = r.get("sensibilidad_umbral_bibliometrix")
        if su and "ari" in su:
            L += [f"- Bibliometrix se construye con el **mismo umbral que el pipeline** (al menos 2 co-ocurrencias) para comparar "
                  f"redes iguales. Sensibilidad: con la red por defecto de bibliometrix, sin umbral ({su['nodos_bibliometrix']} nodos), "
                  f"el ARI pipeline vs bibliometrix baja a {su['ari']} (NMI {su['nmi']}).",
                  "- **Justificación del umbral de 2 co-ocurrencias.** Un enlace con una sola co-ocurrencia proviene de un único "
                  "artículo: muestra que dos palabras aparecieron juntas una vez, no que estén asociadas en el campo. Exigir al "
                  "menos dos artículos es la evidencia mínima de una asociación que se repite.",
                  f"  - Estabilidad de bibliometrix entre semillas: sin umbral {_fmt(su.get('estabilidad_bibliometrix_sin_umbral'))}, "
                  f"con umbral {_fmt(su.get('estabilidad_bibliometrix_con_umbral'))}; pipeline {_fmt(su.get('estabilidad_pipeline'))}.",
                  f"  - Con el mismo umbral, palabras crudas frente a normalizadas: ARI {_fmt(su.get('ari_crudas_vs_normalizadas_mismo_umbral'))} "
                  "(cuanto más se acerque a la estabilidad del pipeline, menos influye la normalización).",
                  "  - Si con umbral la concordancia se acerca a la estabilidad de cada herramienta, la diferencia sin umbral la "
                  "produce el umbral y no el método."]
        if r.get("vosviewer_desactualizado"):
            L += ["", f"> **VOSviewer desactualizado para esta red** ({', '.join(r['vosviewer_desactualizado'])}): "
                  "el map file corresponde a una versión anterior de la red. Volver a agrupar en VOSviewer "
                  f"con `output/vosviewer/{r['red']}_map.txt` y `_network.txt` actuales."]
        if not r["sensibilidad_normalizacion"]:
            L += ["", f"Detalle nodo a nodo: `concordancia_detalle/{r['red']}_emparejamiento.csv`.", ""]
            continue
        L += ["", "Sensibilidad a la normalización (Louvain del pipeline, 10 semillas, resolución 1):", "",
              "| Normalización | k medio | ARI vs VOSviewer | ARI vs bibliometrix |", "|---|---|---|---|"]
        for s in r["sensibilidad_normalizacion"]:
            L.append(f"| {s['normalizacion']} | {s['k_medio']} | {s['ari_vs_vosviewer']} | {_fmt(s['ari_vs_bibliometrix'])} |")
        if len(r.get("variantes_vosviewer", [])) > 1:
            L += ["", "Normalización usada en VOSviewer:", "",
                  "| VOSviewer con | k (pipeline / VOS) | ARI vs pipeline | NMI vs pipeline | ARI vs bibliometrix |",
                  "|---|---|---|---|---|"]
            for vv in r["variantes_vosviewer"]:
                a, b = vv["vs_pipeline"], vv["vs_bibliometrix"]
                L.append(f"| {vv['normalizacion_vosviewer']} | {a.get('k_a', '—')} / {a.get('k_b', '—')} | "
                         f"{a.get('ari', '—')} | {a.get('nmi', '—')} | {b.get('ari', '—')} |")
        L += ["", f"Detalle nodo a nodo: `concordancia_detalle/{r['red']}_emparejamiento.csv`.", ""]

    L += ["## B. Indicadores y conteos", "",
          "Pipeline = datos embebidos en las figuras `output/plotly/*.html` (lo que se publicaría). "
          "Bibliometrix = tablas de `validacion_externa/bibliometrix/`.", ""]
    pa = ind["produccion_anual"]
    L += ["### Producción anual", "",
          f"- Total: pipeline {pa['total_pipeline']}, bibliometrix {pa['total_bibliometrix']} ({pa['anos']} años).",
          f"- Años con diferencia: {len(pa['diferencias'])}"
          + (": " + ", ".join(f"{d['ano']} ({d['pipeline']} vs {d['bibliometrix']})" for d in pa['diferencias']) if pa['diferencias'] else "."), ""]
    cr = ind["crecimiento_anual_pct"]
    L += ["### Crecimiento anual", "", f"- Pipeline {_fmt(cr['pipeline'])} %, bibliometrix {cr['bibliometrix']} %.", ""]
    ci = ind["colaboracion_internacional"]
    L += ["### Colaboración internacional (MCP)", "",
          f"- Artículos multipaís: pipeline {ci['mcp_pipeline']}, bibliometrix {ci['mcp_bibliometrix']}.",
          f"- Artículos que cambian de clase: {', '.join(ci['articulos_mcp_discrepantes']) or 'ninguno'}.",
          f"- Conjunto de países idéntico en {ci['conjuntos_de_paises_identicos']} de 347 artículos. "
          f"Sin país: pipeline {ci['sin_pais_pipeline']}, bibliometrix {ci['sin_pais_bibliometrix']}.",
          f"- País del autor de correspondencia coincide en {ind['pais_autor_correspondencia_coincide']['coinciden']} "
          f"de {ind['pais_autor_correspondencia_coincide']['de']}.",
          f"- Detalle: `{ci['detalle']}`.", ""]
    for clave, titulo in (("fuentes_top", "Fuentes más productivas"), ("autores_top", "Autores más productivos")):
        f = ind[clave]
        L += [f"### {titulo}", "",
              f"- Top {f['top_n']}: Jaccard {f['jaccard']}, Spearman en los comunes {_fmt(f['spearman_en_comunes'])}. "
              f"Valor en el corte: {f['valor_de_corte']} documentos, con {f['empatados_en_el_corte_pipeline']} "
              "empatados dentro del top del pipeline (el orden entre empatados es arbitrario en ambas herramientas).",
              f"- Primero en el pipeline: {f['maximo_pipeline'][0]} ({f['maximo_pipeline'][1]}). "
              f"Primero en bibliometrix: {f['maximo_bibliometrix'][0]} ({f['maximo_bibliometrix'][1]}).", ""]
    lo = ind["lotka_autores_por_productividad"]
    L += ["### Lotka (autores por número de documentos)", "",
          f"- Pipeline: {lo['pipeline']}. Bibliometrix: {lo['bibliometrix']}.",
          f"- Ajuste de bibliometrix: β = {lo['ajuste_bibliometrix'].get('beta', 0):.3f}, "
          f"R² = {lo['ajuste_bibliometrix'].get('R2', 0):.3f}, p (KS) = {lo['ajuste_bibliometrix'].get('p_value_ks', 0):.3f}.", ""]
    rp = ind["rpys"]
    L += ["### RPYS", "",
          f"- Años comunes {rp['anos_comunes']}; rango pipeline {rp['rango_pipeline']}, bibliometrix {rp['rango_bibliometrix']}.",
          f"- Spearman de citas por año: {rp['spearman_citas_por_ano']}. "
          f"Citas totales en años comunes: pipeline {rp['citas_totales_pipeline']:.0f}, bibliometrix {rp['citas_totales_bibliometrix']:.0f}.",
          f"- Picos (top 10 por desviación) comunes: {rp['picos_comunes']}.",
          f"  Pipeline {rp['picos_top10_pipeline']}; bibliometrix {rp['picos_top10_bibliometrix']}.",
          f"- {rp['nota']}", ""]
    for r in ind.get("redes_solo_vosviewer", []):
        e = r["estabilidad_pipeline"]
        L += [f"### {r['red']} (solo pipeline vs VOSviewer; sin equivalente en bibliometrix)", "",
              f"- Aislados excluidos: {r['aislados_excluidos']}. Estabilidad del pipeline (20 semillas): ARI medio "
              f"{e['ari_media']}, mínimo {e['ari_min']}, p5 {e['ari_p05']}.", "",
              "| VOSviewer con | Nodos comunes | k (pipeline / VOS) | ARI | NMI | ARI azar (media / p95) |", "|---|---|---|---|---|---|"]
        for c in r["comparaciones"]:
            L.append(f"| {c['normalizacion_vosviewer']} | {c['n_comunes']} | {c.get('k_a', '—')} / {c.get('k_b', '—')} | "
                     f"{c.get('ari', '—')} | {c.get('nmi', '—')} | {c.get('ari_azar_media', '—')} / {c.get('ari_azar_p95', '—')} |")
        if r["desactualizado"]:
            L.append(f"- Desactualizado: {', '.join(r['desactualizado'])}.")
        L.append("")
    mt = ind.get("mapa_tematico")
    if mt:
        cf = mt["control_figura"]
        L += ["## C. Mapa temático (Callon)", "",
              f"- {mt['nota']}",
              f"- Control: el recálculo reproduce la figura publicada: {cf['centralidades_identicas']} "
              f"({cf['clusters_figura']} temas en la figura, {cf['clusters_recalculo']} en el recálculo).",
              "- Cuadrantes con corte en la mediana de centralidad y densidad de cada herramienta "
              "(con 4 o 5 temas, un tema cerca de la mediana cambia de cuadrante con facilidad).", ""]
        for clave, titulo in (("oficial", "Pipeline oficial (palabras canonicalizadas)"),
                              ("metodo_pipeline_sobre_DE_crudas", "Método del pipeline sobre las mismas palabras de autor que bibliometrix")):
            r = mt[clave]
            c = r["pertenencia"]
            L += [f"### {titulo}", "",
                  f"- Temas: pipeline {r['k_pipeline']}, bibliometrix {r['k_bibliometrix']}. Palabras comunes: {r['palabras_comunes']}.",
                  f"- Pertenencia de palabras a temas: ARI {c.get('ari', '—')}, NMI {c.get('nmi', '—')} "
                  f"(azar: media {c.get('ari_azar_media', '—')}, p95 {c.get('ari_azar_p95', '—')}).", "",
                  "| Tema bibliometrix | Cuadrante | Tema del pipeline más parecido | Cuadrante | Jaccard | ¿Mismo cuadrante? |",
                  "|---|---|---|---|---|---|"]
            for t in r["temas"]:
                L.append(f"| {t['tema_bibliometrix']} | {t['cuadrante_bibliometrix']} | {t['tema_pipeline_mas_parecido']} | "
                         f"{t['cuadrante_pipeline']} | {t['jaccard']} | {'sí' if t['mismo_cuadrante'] else 'no'} |")
            L.append("")
    tp = ind.get("tendencia_topicos")
    if tp:
        L += ["## D. Tendencia de tópicos", ""]
        if "nota" in tp:
            L += [f"- {tp['nota']}", ""]
        else:
            fp = tp["figura_pipeline"]
            L += ["- Se compara el año mediano de uso de cada palabra con frecuencia ≥ 5 (método de `fieldByYear`, "
                  "que es también el que usa ahora la figura del pipeline).",
                  "- Las fuentes no son las mismas: el pipeline usa el vocabulario común normalizado (palabras de autor, "
                  "automáticas, OpenAlex y MeSH); bibliometrix, solo las palabras de autor (DE) sin normalizar. "
                  "Por eso el pipeline tiene más palabras y la comparación se hace sobre las comunes.",
                  f"- Palabras: pipeline {tp['palabras_freq5_pipeline']}, bibliometrix {tp['palabras_freq5_bibliometrix']}, "
                  f"comunes {tp['comunes']}.",
                  f"- Spearman del año mediano en las comunes: {_fmt(tp['spearman_ano_mediano'])}; diferencia media "
                  f"{_fmt(tp['diferencia_media_anos'])} años; año idéntico en {tp['coinciden_exacto_ano_mediano']}.",
                  f"- 15 más recientes en común: {', '.join(tp['recientes_en_comun']) or 'ninguna'}.",
                  f"  - Pipeline: {', '.join(tp['mas_recientes_pipeline'])}.",
                  f"  - Bibliometrix: {', '.join(tp['mas_recientes_bibliometrix'])}.",
                  f"- Referencia: con el método anterior de la figura ({fp['metodo']}), la palabra más frecuente en cada "
                  f"uno de los {fp['anos']} años habría sido {fp['palabra_mas_frecuente_por_ano']}: no muestra tendencia.",
                  f"- Detalle: `{tp['detalle']}`.", ""]
    p = VAL / "informe_concordancia.md"
    p.write_text("\n".join(L), encoding="utf-8")
    return p


REDES_SOLO_VOS = ["coauthorship_authors", "coupling_journals", "cocitation_journals"]


def analizar_red_solo_vos(red: str) -> dict | None:
    """Redes sin equivalente en el script de R: pipeline vs VOSviewer y estabilidad interna."""
    variantes = [(n, s_) for n, s_ in VOS_VARIANTES.items() if (VOS / f"{red}{s_}").exists()]
    if not variantes:
        return None
    oficial, semillas_pipe, G, control = _pipeline_particiones(red)
    aislados = {n for n in G.nodes() if G.degree(n) == 0}
    oficial = {n: c for n, c in oficial.items() if n not in aislados}
    semillas_pipe = [{n: c for n, c in p_.items() if n not in aislados} for p_ in semillas_pipe]
    comps, errores = [], {}
    for nombre_var, sufijo in variantes:
        try:
            pv = {n: c for n, c in _vos_particion(red, G, sufijo).items() if n not in aislados}
        except ValueError as e:
            errores[nombre_var] = str(e)
            continue
        comps.append({"normalizacion_vosviewer": nombre_var, **_comparar(oficial, pv, "pipeline", f"VOSviewer ({nombre_var})")})
    return {"red": red, "aislados_excluidos": len(aislados), "control": control,
            "estabilidad_pipeline": _estabilidad(semillas_pipe), "comparaciones": comps, "desactualizado": errores}


def main() -> None:
    DET.mkdir(parents=True, exist_ok=True)
    entradas = {}
    for f in sorted(list(VOS.glob("*_map.txt")) + list(BX.glob("12_*_clusters_igraph.csv"))
                    + [ROOT / "output" / "bibliometrix_export.csv"] + list(NETWORKS.glob("*.graphml"))):
        base = f.name.split(".")[0].replace("_map", "")
        if f.parent == BX or f.name == "bibliometrix_export.csv" or base in REDES:
            entradas[str(f.relative_to(ROOT)).replace("\\", "/")] = _sha(f)

    redes = []
    for red in REDES:
        print(f"Red {red}...", flush=True)
        redes.append(analizar_red(red))
    solo_vos = [r for r in (analizar_red_solo_vos(x) for x in REDES_SOLO_VOS) if r]
    print("Indicadores...", flush=True)
    ind = indicadores()
    ind["redes_solo_vosviewer"] = solo_vos
    print("Mapa tematico...", flush=True)
    ind["mapa_tematico"] = comparar_mapa_tematico()
    print("Tendencia de topicos...", flush=True)
    ind["tendencia_topicos"] = comparar_tendencia_topicos()
    res = {"generado": datetime.now().isoformat(timespec="seconds"), "entradas": entradas,
           "redes": redes, "indicadores": ind}
    (VAL / "concordancia_resultados.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    p = escribir_informe(redes, ind, entradas)
    print(f"Informe: {p}")


if __name__ == "__main__":
    main()
