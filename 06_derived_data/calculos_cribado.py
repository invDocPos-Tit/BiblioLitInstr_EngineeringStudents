"""Screening figures reported in Section 2.3 and Figure 1 (read-only; nothing is written outside this folder).
Sources: human pilots (INSTRUMENTOS_ING/03_seleccion/piloto_50_v0{1,2}/analisis_revisado_v01/resultados_diagnostico.json),
the contrast of the two blind AI rounds and the reconciliation phase (.../screening_completo_1128_v01/revision_fase3_v01/contraste_1128.csv),
the 1,083 screened records of the November search (repositorio_datos/02_screening/screening_outcomes_1083.csv) and the
eligibility routes and exclusion causes (../../conteos_rescate.json, produced by ../../rescatar.py).
Run: python -X utf8 -I calculos_cribado.py  ->  calculos_cribado.json"""
import csv, json, collections, pathlib
HERE = pathlib.Path(__file__).resolve().parent; ROOT = HERE.parent.parent
SEL = pathlib.Path(r"G:\REVISIONES\INSTRUMENTOS_ING\03_seleccion")
pil = {k: json.load(open(SEL / f"piloto_50_{k}" / "analisis_revisado_v01" / "resultados_diagnostico.json", encoding="utf-8"))["voto"] for k in ("v01", "v02")}
con = {r["UID"]: r for r in csv.DictReader(open(SEL / "screening_completo_1128_v01" / "revision_fase3_v01" / "contraste_1128.csv", encoding="utf-8-sig"), delimiter=";")}
ids = {r["record_id"] for r in csv.DictReader(open(ROOT / "repositorio_datos" / "02_screening" / "screening_outcomes_1083.csv", encoding="utf-8-sig"))}
ai = [con[i] for i in ids if i in con]; human = sorted(ids - set(con))
def kappa(rows):
    n = len(rows); a = sum(r["Ronda1"] == r["Ronda2"] for r in rows)
    c1 = collections.Counter(r["Ronda1"] for r in rows); c2 = collections.Counter(r["Ronda2"] for r in rows)
    pe = sum(c1[k] * c2[k] for k in set(c1) | set(c2)) / n / n
    return {"n": n, "agreement": a, "agreement_pct": round(100 * a / n, 1), "cohen_kappa_3cat": round((a / n - pe) / (1 - pe), 3)}
cr = json.load(open(ROOT / "conteos_rescate.json", encoding="utf-8"))
nc = cr["no_elegibles_noviembre_por_causa"]
R = {"human_pilot_1": {"n": 50, "agreement": round(pil["v01"]["acuerdo"] * 50), "agreement_pct": round(100 * pil["v01"]["acuerdo"], 1), "cohen_kappa": round(pil["v01"]["kappa"], 3), "note": "title and abstract without year; unweighted, before consensus"},
     "human_pilot_2": {"n": 50, "agreement": round(pil["v02"]["acuerdo"] * 50), "agreement_pct": round(100 * pil["v02"]["acuerdo"], 1), "cohen_kappa": round(pil["v02"]["kappa"], 3), "note": "with year; unweighted, before consensus; five discrepancies ratified by consensus (36 EXCLUDE, 14 UNCERTAIN)",
                       "records_in_1083": len(human)},
     "ai_two_blind_rounds_on_1083": {**kappa(ai), "contrast_types": dict(collections.Counter(r["Tipo_contraste"] for r in ai))},
     "ai_two_blind_rounds_all_1128": kappa(list(con.values())),
     "eligible_by_route_352": cr["por_via"], "corpus_by_route_347": cr["por_via_corpus_final"],
     "not_eligible_731": nc,
     "not_eligible_by_A1_A6": sum(v for k, v in nc.items() if k[:2] in ("A1", "A2", "A4", "A5", "A6")),
     "not_eligible_A1_A6_not_confirmed": sum(v for k, v in nc.items() if k[:2] not in ("A1", "A2", "A4", "A5", "A6")),
     "uncertain_resolved_at_full_text": cr["por_via"]["RESCATE_UNCERTAIN_INCLUIDO_EN_TEXTO_COMPLETO"] + nc["UNCERTAIN_EXCLUIDO_EN_TEXTO_COMPLETO"] + nc["UNCERTAIN_SIN_PDF_NO_EVALUADO"] + nc["UNCERTAIN_RETIRO_DOCUMENTAL"]}
assert R["ai_two_blind_rounds_on_1083"]["n"] + len(human) == 1083 == cr["cribados_de_noviembre"]
assert R["not_eligible_by_A1_A6"] + R["not_eligible_A1_A6_not_confirmed"] == 731
(HERE / "calculos_cribado.json").write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding="utf-8"); print(json.dumps(R, ensure_ascii=False, indent=1))
