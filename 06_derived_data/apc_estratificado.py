"""Comparación de APC estratificada. Marca de APC por documento con la misma regla de la herramienta (src/bootstrap_ci_phase.py:_article_apc), sobre data/enriched_articles.json (solo lectura).
Uso (desde manuscrito/): python -X utf8 calculos_ampliados/apc_estratificado.py"""
import json, os, numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
OUT=os.path.dirname(os.path.abspath(__file__))
A=json.load(open('G:/AUTOMATIZACIONES/biblio140302_11/data/enriched_articles.json',encoding='utf-8'))
def apc(a):
    p=(a.get('openalex') or {}).get('apc_paid')
    if isinstance(p,dict):
        v=p.get('value_usd') or p.get('value')
        if v: return float(v)
    jm=a.get('journal_metrics') or {}
    return float(jm['apc_usd']) if jm.get('apc_usd') else None
d=pd.DataFrame([dict(key=a['key'],apc_usd=apc(a),anio=int(a['publication_year']),tipo=a['item_type'],citas=a['cited_by_count']) for a in A])
d['grupo']=np.where(d.apc_usd.fillna(0)>0,'APC','sin APC'); d.to_csv(f'{OUT}/apc_por_documento.csv',index=False)
rows=[]
def res(x,lab):
    a=x[x.grupo=='APC'].citas; b=x[x.grupo=='sin APC'].citas
    rows.append(dict(estrato=lab,n_apc=len(a),n_sin=len(b),media_apc=round(a.mean(),1),media_sin=round(b.mean(),1),mediana_apc=a.median(),mediana_sin=b.median(),p_mann_whitney=round(mannwhitneyu(a,b,alternative='two-sided').pvalue,4)))
res(d,'todos los documentos'); j=d[d.tipo=='Journal Article']; res(j,'solo artículos de revista')
for lo,hi in [(2014,2021),(2022,2024),(2025,2026)]: res(j[(j.anio>=lo)&(j.anio<=hi)],f'revistas {lo}-{hi}')
r=pd.DataFrame(rows); r.to_csv(f'{OUT}/apc_estratificado.csv',index=False); print(r.to_string())
