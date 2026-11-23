# Sistema de Enriquecimiento y Análisis Bibliométrico de Frontera

Pipeline modular, **100 % determinista** y agnóstico al dominio para enriquecimiento multi-fuente y análisis bibliométrico/cienciométrico de estado del arte. Procesa cualquier corpus bibliográfico en formato CSV (como `consolidado.csv`) y genera indicadores avanzados, análisis de redes, detección de ráfagas, espectroscopía de referencias, intervalos de confianza bootstrap, variedad de dinámicas de investigación (RDM) y síntesis algorítmica de hallazgos sin depender de LLMs ni heurísticas inductivas subjetivas.

---

## Qué hace

A partir de un CSV bibliográfico (`consolidado.csv`), el sistema ejecuta un pipeline estructurado en **8 fases deterministas** (`0, 1, 2, 3, 4, 5, 6, 7`) que:
1. Limpia, deduplica (por DOI y título normalizado) y canonicaliza idiomas, revistas (ISSN + clave normalizada) y colores de Acceso Abierto (`diamond` frente a `gold`).
2. Enriquece cada artículo con metadatos y bibliografías desde **11 fuentes** (OpenAlex CWTS Leiden `topics`, Crossref, Semantic Scholar, SCImago SJR, Altmetric, OpenCitations, Unpaywall, Europe PMC, DOAJ, NLM, Crossref Event Data).
3. Ejecuta **más de 25 análisis bibliométricos y cienciométricos** reproducibles bit a bit y exporta reportes Markdown, Excel multi-hoja, CSV/BibTeX/script R para Bibliometrix, redes GEXF/GraphML/VOSviewer y visualizaciones estáticas (PNG 300 DPI) e interactivas (Plotly HTML).

### Capacidades principales

| Capacidad | Descripción | Fase |
|-----------|-------------|------|
| **Carga, deduplicación y canonicalización** | Auto-detección de delimitador CSV, deduplicación defensiva por DOI/título y normalización de idiomas y revistas | 0 |
| **Desambiguación de nombres** | Unifica variantes de autores (ORCID + fuzzy matching) e instituciones (IDs OpenAlex) | 0b |
| **Enriquecimiento multi-fuente** | OpenAlex (`topics` CWTS Leiden), Crossref, Semantic Scholar, SCImago, Altmetric, OpenCitations, Unpaywall, Europe PMC, DOAJ, NLM, Event Data | 1 (1a–1l) |
| **Análisis descriptivo y EDA** | Ley de Lotka, zonas de Bradford, tendencias temporales, distribución geográfica, SDG mapping, OA (`diamond`/`gold`/`green`/`hybrid`/`bronze`/`closed`) | 2 / EDA |
| **Análisis de abstracts y léxico metodológico** | TF-IDF, NMF, extracción agnóstica multidisciplinaria de diseños/métodos con límites de palabra (`\b...\b`), brechas declaradas y WordClouds | 2b |
| **BERT / SciBERT** | Clustering temático opcional con embeddings científicos + UMAP + HDBSCAN | 2c |
| **Calidad estructural de abstracts** | Evaluación algorítmica multidimensional (estructura IMRaD, completitud informativa) | 2d |
| **Indicadores suplementarios** | Estructura conceptual (CA/MDS), temas de tendencia, factor de dominancia, semivida de citación e Índice de Price sobre `references_resolved` | 2e |
| **SOTA y vacíos estructurales** | Matriz de co-ocurrencia método × aplicación, taxonomía algorítmica de vacíos estructurales y frentes emergentes | 2f |
| **Espectroscopía de referencias (RPYS)** | Reference Publication Year Spectroscopy, raíces históricas y genealogía conceptual sobre `topics` CWTS Leiden | 2g |
| **CiteSpace Analysis Suite** | Detección de ráfagas de Kleinberg, línea temporal de co-citación, nodos pivote, métrica Sigma ($\Sigma$), frentes de investigación y etiquetado TF-IDF/LLR/MI con $Q$ y $S$ medidos sobre la red primaria | 2h |
| **Indicadores CWTS Leiden** | MNCS, PP(top 10 %) y MNJS normalizados por campo + diagnóstico de sesgos de cobertura e idioma | 2i |
| **10 redes bibliométricas + Burt** | Acoplamiento bibliográfico, co-citación, co-ocurrencia de keywords y coautoría (con Louvain), huecos estructurales de Burt y artículos puente | 3 |
| **Análisis temático estratégico** | Diagrama estratégico de Callon (centralidad vs. densidad), evolución temática multi-período y colaboración internacional SCP/MCP | 3b |
| **Análisis estratificado** | Comparación sistemática entre estratos documentales (artículos de revista vs. actas de congreso, etc.) | Estratificado |
| **Exportación e integración** | CSV/BibTeX/script R nativo para Bibliometrix, Excel multi-hoja, gráficos interactivos Plotly, exportación nativa VOSviewer y corpus Open Access | 4 / 4b |
| **Evolución de impacto** | Trayectorias de citación, FWCI, coeficiente de belleza (*Sleeping Beauties*), genealogía de conceptos, flujo Sankey de tópicos y análisis APC | 5 |
| **Bootstrap de intervalos de confianza** | IC 95 % no paramétricos ($B = 10\,000$, semilla fija) para MCP, FWCI, MNCS, PP(top 10 %), citas medias, prima APC e Índice de Price | 5b |
| **Research Dynamics Manifold (RDM)** | Variedad 3D de dinámicas de investigación $(S, I, T)$, clasificación en 8 arquetipos y superficies de potencial | 6 |
| **Knowledge Map (Síntesis determinista)** | Recolección algorítmica de hallazgos de todas las fases, corroboración cruzada inter-fase, análisis de sensibilidad de pesos y panel HTML interactivo | 7 |

---

## Inicio rápido

### Requisitos

- Python 3.10+ (probado y bloqueado en Python 3.12 con `uv`)
- Instalación del entorno reproducible:
  ```bash
  uv sync
  # Opcional: extras para embeddings SciBERT o detección de idioma
  uv sync --extra semantica --extra idioma
  ```

### Configuración mínima (`~/.automatizaciones/.env` o `.env` local)

```bash
OPENALEX_EMAIL=your.email@example.org
OPENALEX_API_KEY=...        # Opcional (aumenta límite de tasa en OpenAlex)
S2_API_KEY=...              # Opcional (para enriquecimiento y rescate de referencias en Semantic Scholar)
ALTMETRIC_API_KEY=...       # Opcional (solo si se activa --enable-altmetric)
```

### Ejecución sobre cualquier corpus `consolidado.csv`

```bash
# 1. Iniciar una nueva revisión limpia (borra data/ y output/ derivados, conserva consolidado.csv y data/scimagojr.csv)
.venv\Scripts\python.exe main.py --new-review-yes

# 2. Ejecutar el pipeline completo (Fases 0, 1, 2, 3, 5, 6, 8, 10)
.venv\Scripts\python.exe main.py --phases 0 1 2 3 4 5 6 7 --run-id mi_revision_v1

# 3. Re-ejecutar únicamente las fases analíticas reutilizando el corpus enriquecido (data/enriched_articles.json)
.venv\Scripts\python.exe main.py --phases 2 3 4 5 6 7 --skip-bert --skip-ref-resolution --run-id analisis_v2
```

### Scripts auxiliares de recuperación de referencias citadas

Cuando las editoriales no depositan las listas de referencias en Crossref/OpenAlex, el sistema incluye dos herramientas deterministas que resuelven las referencias faltantes contra OpenAlex usando la misma estructura de la Fase 1i:
- `scripts/recuperar_referencias_s2.py` — recupera referencias extraídas de PDFs por Semantic Scholar.
- `scripts/importar_referencias_manuales.py` — importa listas de referencias exportadas en formato `.ris` (Scopus / Web of Science) o `.txt` (listas con DOI).

---

## Arquitectura del pipeline

```
                     ┌─────────────────────────────────┐
                     │  consolidado.csv (entrada)       │
                     └──────────────┬──────────────────┘
                                    │
                     ┌──────────────▼──────────────────┐
                     │  FASE 0 + 0b: Carga, dedup,      │
                     │  canonicalización y ORCID/fuzzy  │
                     └──────────────┬──────────────────┘
                                    │
          ┌─────────────────────────▼─────────────────────────┐
          │  FASE 1 (1a–1l): Enriquecimiento multi-fuente      │
          │  OpenAlex (topics CWTS Leiden) > Crossref > S2 >   │
          │  OpenAlex Meta > SCImago > OpenCitations >         │
          │  Unpaywall > Ref Resolution (1i) > Gap-Filler      │
          └─────────────────────────┬─────────────────────────┘
                                    │
     ┌──────────────────────────────▼──────────────────────────────┐
     │  FASE 2 (2a–2i): Análisis Bibliométrico y Cienciométrico     │
     │  Descriptivo + EDA > Abstracts/Métodos > SciBERT > Calidad >│
     │  Suplementarios (Price Index / Half-life) > SOTA + Gaps >   │
     │  Espectroscopía RPYS > CiteSpace Suite > Indicadores Leiden │
     └──────────────────────────────┬──────────────────────────────┘
                                    │
     ┌──────────────────────────────▼──────────────────────────────┐
     │  FASE 3 + 3b + Estratificado: Redes y Estructura Temática    │
     │  10 redes bibliométricas + Huecos de Burt + Artículos puente│
     │  + Diagrama estratégico de Callon + Evolución + SCP/MCP     │
     └──────────────────────────────┬──────────────────────────────┘
                                    │
     ┌──────────────────────────────▼──────────────────────────────┐
     │  FASE 4 + 4b: Exportación e Integración                     │
     │  Bibliometrix (CSV/BibTeX/R) > Excel multi-hoja > Plotly >  │
     │  VOSviewer (.map/.net + estáticos) > Exportación Corpus OA  │
     └──────────────────────────────┬──────────────────────────────┘
                                    │
     ┌──────────────────────────────▼──────────────────────────────┐
     │  FASE 5 + 5b: Evolución de Impacto + Bootstrap IC 95 %      │
     │  Trayectorias, FWCI, Sleeping Beauties, Genealogía, Sankey  │
     │  + Bootstrap no paramétrico (B=10 000) sobre indicadores    │
     └──────────────────────────────┬──────────────────────────────┘
                                    │
     ┌──────────────────────────────▼──────────────────────────────┐
     │  FASE 6: Research Dynamics Manifold (RDM)                  │
     │  Variedad 3D (S, I, T), 8 arquetipos y superficies          │
     └──────────────────────────────┬──────────────────────────────┘
                                    │
     ┌──────────────────────────────▼──────────────────────────────┐
     │  FASE 7: Knowledge Map (Síntesis Determinista)              │
     │  Ledger de evidencia, corroboración cruzada inter-fase,     │
     │  sensibilidad de pesos e informe/dashboard integrado        │
     └─────────────────────────────────────────────────────────────┘
```

---

## Estructura de salidas (`output/`)

```
output/
├── bibliometrix_export.csv    # CSV compatible con R bibliometrix / Biblioshiny
├── bibliometrix_analysis.R    # Script R automatizado (13 análisis)
├── resultados.bib             # BibTeX canónico del corpus
├── citespace/                 # CiteSpace suite (ráfagas, pivotes, Sigma, frentes, Q/S)
├── figures/                   # 30+ figuras estáticas de alta resolución (PNG)
├── impact_evolution/          # Fase 5 (FWCI, Sleeping Beauties, genealogía, Sankey)
├── knowledge_map/             # Fase 7 (knowledge_map_data.json, synthesis, report.md, HTML)
├── leiden/                    # Fase 2i (MNCS, PP top 10%, MNJS, diagnóstico de sesgos)
├── logs/                      # Logs por corrida y manifiestos JSON (run_manifest_<ts>.json)
├── networks/                  # Redes GEXF/GraphML, métricas, Burt constraint y Bridge Articles
├── oa_corpus/                 # Fase 4b (catálogo y resumen de artículos Open Access)
├── plotly/                    # 40+ visualizaciones interactivas HTML
├── quality/                   # Fase 2d (reporte de calidad estructural de abstracts)
├── rdm/                       # Fase 6 (Research Dynamics Manifold: datos, reporte y figuras)
├── reproducibility/           # Fase 5b (bootstrap_ci.json y bootstrap_ci.md)
├── reports/                   # Reportes Markdown (descriptivo, EDA, abstracts, suplementarios) y Excel
├── sota_analysis/             # Fase 2f (matrices de vacíos estructurales y fronteras)
├── spectroscopy/              # Fase 2g (RPYS, obras seminales y genealogía de referencias)
├── stratified/                # Análisis estratificado por tipo documental
├── thematic_strategy/         # Fase 3b (diagrama de Callon, evolución temática, SCP/MCP)
└── vosviewer/                 # Redes nativas VOSviewer (.map + .net) y mapas estáticos
```
