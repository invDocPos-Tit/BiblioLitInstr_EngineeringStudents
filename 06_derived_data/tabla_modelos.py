"""Builds Table S1 (association models M1-M5) from calculos_modelos.json, in English and Spanish (markdown), so that every
value of the table is traceable to the model output. Run after calculos_modelos.py:  python -X utf8 -I tabla_modelos.py"""
import json, pathlib
HERE = pathlib.Path(__file__).resolve().parent
D = json.load(open(HERE / "calculos_modelos.json", encoding="utf-8"))["model_details"]
LAB = {"en": {"Intercept": "Intercept", "t": "Year (per year since 2014)", "yc": "Year (per year)", "journal": "Journal article",
              "technical": "Names technical competencies", "transversal": "Names transversal competencies",
              "selfreport": "Self-report instrument terms", "lref": "Resolved references (log)", "labw": "Abstract words (log)",
              "oa": "Open access", "mcp": "Multi-country authorship", "specific": "Specific measurement-property terms",
              "roots": "Cites a psychometric root"},
       "es": {"Intercept": "Intercepto", "t": "Año (por año desde 2014)", "yc": "Año (por año)", "journal": "Artículo de revista",
              "technical": "Nombra competencias técnicas", "transversal": "Nombra competencias transversales",
              "selfreport": "Términos de instrumentos de autoinforme", "lref": "Referencias resueltas (log)", "labw": "Palabras del resumen (log)",
              "oa": "Acceso abierto", "mcp": "Autoría de varios países", "specific": "Términos específicos de propiedades de medición",
              "roots": "Cita una raíz psicométrica"}}
OUTC = {"en": {"M1": "Documents per year, 2014–2024", "M2": "Names technical competencies", "M3": "Names performance or observation instrument terms",
               "M4": "Cites at least one classical psychometric root", "M4b": "Uses specific measurement-property terms", "M5": "Citations"},
        "es": {"M1": "Documentos por año, 2014–2024", "M2": "Nombra competencias técnicas", "M3": "Nombra términos de instrumentos de desempeño u observación",
               "M4": "Cita al menos una raíz psicométrica clásica", "M4b": "Usa términos específicos de propiedades de medición", "M5": "Citas"}}
TXT = {"en": dict(head="| Model and outcome | Term | Estimate (95% CI) | p |\n|---|---|---|---|", nb="negative binomial", lo="logistic",
                  n="n", ev="events", al="dispersion α", r2="McFadden R²", ref="Reference", yearf="publication year entered as a factor (reference 2014; coefficients not shown)",
                  note="Estimates are rate ratios (negative binomial) or odds ratios (logistic) with 95% Wald intervals; the intercept is the expected count or odds at the reference values. M1 also yields an expected 2025 count of {e} documents from the 2014–2024 trend, against 76 observed. In M1, α = 0.00 indicates no overdispersion relative to the Poisson model.",
                  title="**Table S1.** Association models M1–M5: specification and estimates."),
       "es": dict(head="| Modelo y variable dependiente | Término | Estimación (IC 95 %) | p |\n|---|---|---|---|", nb="binomial negativo", lo="logístico",
                  n="n", ev="eventos", al="dispersión α", r2="R² de McFadden", ref="Referencia", yearf="año de publicación como factor (referencia 2014; coeficientes no mostrados)",
                  note="Las estimaciones son razones de tasas (binomial negativo) o razones de momios (logístico) con intervalos de Wald del 95 %; el intercepto es el recuento o los momios esperados en los valores de referencia. M1 da además {e} documentos esperados en 2025 según la tendencia 2014–2024, frente a 76 observados. En M1, α = 0,00 indica que no hay sobredispersión respecto del modelo de Poisson.",
                  title="**Tabla S1.** Modelos de asociación M1–M5: especificación y estimaciones.")}
REF = {"en": {"M1": "", "M2": "conference paper", "M3": "conference paper", "M4": "conference paper", "M4b": "conference paper",
              "M5": "conference paper, closed access, single-country authorship"},
       "es": {"M1": "", "M2": "trabajo de congreso", "M3": "trabajo de congreso", "M4": "trabajo de congreso", "M4b": "trabajo de congreso",
              "M5": "trabajo de congreso, acceso cerrado, autoría de un solo país"}}
def fmt(v, lang):   # two decimals; values below 0.01 with two significant figures (e.g. intercepts at log(1 + x) = 0)
    import math
    s = f"{v:.2f}" if abs(v) >= 0.01 or v == 0 else f"{v:.{1 - math.floor(math.log10(abs(v)))}f}"   # two significant figures, no exponent
    return s if lang == "en" else s.replace(".", ",")
for lang in ("en", "es"):
    T = TXT[lang]; rows = [T["title"], "", T["head"]]
    for k, d in D.items():
        kind = T["nb"] if d["type"].startswith("negative") else T["lo"]
        info = [f"{T['n']} = {d['n']}"]
        if "events" in d: info.append(f"{T['ev']} = {d['events']}"); info.append(f"{T['r2']} = {fmt(d['mcfadden_r2'], lang)}")
        if "dispersion_alpha" in d: info.append(f"{T['al']} = {fmt(d['dispersion_alpha'], lang)}")
        if REF[lang][k]: info.append(f"{T['ref']}: {REF[lang][k]}")
        if k == "M5": info.append(T["yearf"])
        head = f"**{k}** ({kind}): {OUTC[lang][k]}; " + "; ".join(info)
        for i, t in enumerate(d["terms"]):
            p = f"{t['p']:.3f}" if lang == "en" else f"{t['p']:.3f}".replace(".", ",")
            if t["p"] < 0.001: p = "< 0.001" if lang == "en" else "< 0,001"
            est = f"{fmt(t['estimate'], lang)} ({fmt(t['ci_low'], lang)}–{fmt(t['ci_high'], lang)})"
            rows.append(f"| {head if i == 0 else ''} | {LAB[lang][t['term']]} | {est} | {p} |")
    expected = D["M1"]["note"].split(": ")[1].split(";")[0]          # e.g. "29.7"
    if lang == "es": expected = expected.replace(".", ",")
    rows += ["", "*" + ("Note." if lang == "en" else "Nota.") + "* " + T["note"].format(e=expected)]
    (HERE / f"tabla_s1_modelos_{lang}.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
print("ok")
