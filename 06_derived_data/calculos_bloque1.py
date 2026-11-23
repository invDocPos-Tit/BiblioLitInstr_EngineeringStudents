"""Block 1 of the search string (population and context): discipline-specific terms (electronic* engineering, electrical
engineering) versus the general engineering-education terms, in the title, abstract and author keywords of the 347 documents.
Same text and source as calculos_bloques_cadena.py (enriched_articles.json of the analysis tool; read-only).
Run: python -X utf8 -I calculos_bloque1.py  ->  calculos_bloque1.json"""
import json, pathlib, re
HERE = pathlib.Path(__file__).resolve().parent
arts = json.load(open(r"G:\AUTOMATIZACIONES\biblio140302_11\data\enriched_articles.json", encoding="utf-8"))
arts = arts if isinstance(arts, list) else arts["articles"]; assert len(arts) == 347
def text(a):
    kw = a.get("author_keywords") or []; kw = kw if isinstance(kw, list) else [kw]
    return " ".join([str(a.get("title") or ""), str(a.get("abstract") or ""), " ".join(map(str, kw))]).lower()
SPEC = re.compile(r"\belectronic\w* engineering|\belectrical engineering")
GEN = re.compile(r"\bengineering education|\bengineering students?\b|\bengineering undergraduates?\b|\bengineering curricul")
s = [bool(SPEC.search(text(a))) for a in arts]; g = [bool(GEN.search(text(a))) for a in arts]
R = {"n": 347, "discipline_specific": sum(s), "discipline_specific_pct": round(100 * sum(s) / 347, 1),
     "discipline_specific_without_general": sum(x and not y for x, y in zip(s, g)),
     "neither_in_title_abstract_keywords": sum((not x) and (not y) for x, y in zip(s, g)),
     "general_terms": sum(g)}
(HERE / "calculos_bloque1.json").write_text(json.dumps(R, indent=1), encoding="utf-8"); print(R)
