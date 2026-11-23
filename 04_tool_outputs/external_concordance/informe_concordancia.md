# Informe de concordancia externa

Generado: 2026-11-22 15:51 con `scripts/concordancia_externa.py`.

## Entradas

| Archivo | SHA-256 (16) |
|---|---|
| `output/bibliometrix_export.csv` | `0617149ee2164c17` |
| `output/networks/bibliographic_coupling.graphml` | `4d527f8a242de4ff` |
| `output/networks/coauthorship_countries.graphml` | `7ac8266d811c619e` |
| `output/networks/cocitation.graphml` | `581ace3c00483e6a` |
| `output/networks/keyword_cooccurrence.graphml` | `2b744fe1eee13dca` |
| `validacion_externa/bibliometrix/12_acoplamiento_clusters_igraph.csv` | `9baf297c3d6d1a9f` |
| `validacion_externa/bibliometrix/12_cocitacion_clusters_igraph.csv` | `f0e28bc7862666b3` |
| `validacion_externa/bibliometrix/12_colaboracion_paises_clusters_igraph.csv` | `3e067f303ffc6133` |
| `validacion_externa/bibliometrix/12_coocurrencia_DE_clusters_igraph.csv` | `72b8776c1748557b` |
| `validacion_externa/bibliometrix/12_coocurrencia_DE_umbral1_clusters_igraph.csv` | `db3bb0d5492d03ee` |

## Cómo leer las métricas de clusters

- **ARI** (índice de Rand ajustado) y **NMI** (información mutua normalizada) sobre los nodos comunes.
- No hay umbral fijado a mano. Cada valor se lee contra dos referencias medidas en los mismos datos:
  - **estabilidad interna**: ARI entre 20 semillas del mismo algoritmo (cuánto varía la herramienta contra sí misma);
  - **azar**: ARI contra etiquetas permutadas (200 permutaciones).
- Si la concordancia entre herramientas se acerca a la estabilidad interna, la estructura no depende de la herramienta.
  Si queda cerca del azar, la estructura reportada es un artefacto del algoritmo o de la construcción de la red.

**Qué agrupa cada herramienta.** El pipeline y bibliometrix aplican Louvain (modularidad) sobre conteos brutos. La tabla principal usa VOSviewer con normalización **LinLog/modularity**, que optimiza el mismo criterio: la diferencia que queda es del algoritmo. La tabla de sensibilidad recalcula el Louvain del pipeline sobre *association strength* para mostrar cuánto cambia la estructura con la normalización. Parámetros de VOSviewer: `validacion_externa/vosviewer/parametros.txt`.

## A. Estructura de las redes

### bibliographic_coupling

- Nodos con al menos un enlace: pipeline 299, VOSviewer 299, bibliometrix 300. Aislados excluidos del pipeline/VOSviewer: 43.
- Emparejamiento con bibliometrix: clave de articulo.
- Control: la semilla 42 reproduce la partición del `.graphml` con ARI = 1.0.

| Estabilidad interna | ARI medio | ARI mínimo | ARI p5 |
|---|---|---|---|
| pipeline (python-louvain, 20 semillas) | 0.4902 | 0.3007 | 0.3534 |
| bibliometrix (igraph Louvain, 20 semillas) | 0.5294 | 0.282 | 0.3528 |

| Comparación | Nodos comunes | Cobertura (a / b) | k (a / b) | ARI | NMI | ARI azar (media / p95) |
|---|---|---|---|---|---|---|
| pipeline vs VOSviewer | 299 | 1.0 / 1.0 | 8 / 11 | 0.417 | 0.5141 | -0.0004 / 0.0102 |
| pipeline vs bibliometrix | 299 | 1.0 / 0.997 | 8 / 9 | 0.4342 | 0.5147 | 0.0002 / 0.0108 |
| VOSviewer vs bibliometrix | 299 | 1.0 / 0.997 | 11 / 9 | 0.6279 | 0.6453 | -0.0005 / 0.0075 |
- VOSviewer usado en la tabla principal: LinLog/modularity.

Sensibilidad a la normalización (Louvain del pipeline, 10 semillas, resolución 1):

| Normalización | k medio | ARI vs VOSviewer | ARI vs bibliometrix |
|---|---|---|---|
| conteos brutos (pipeline actual) | 9.8 | 0.473 | 0.445 |
| association strength | 19.1 | 0.183 | 0.183 |

Detalle nodo a nodo: `concordancia_detalle/bibliographic_coupling_emparejamiento.csv`.

### cocitation

- Nodos con al menos un enlace: pipeline 639, VOSviewer 639, bibliometrix 575. Aislados excluidos del pipeline/VOSviewer: 0.
- Emparejamiento con bibliometrix: primer autor + ano de la referencia (formato de bibliometrix). Colisiones (varios nodos del pipeline en una misma clave): 11.
- Control: la semilla 42 reproduce la partición del `.graphml` con ARI = 1.0.
- Referencias que no llegan al campo CR del export (bibliometrix no las ve): 5752 de 16595 (5315 obras distintas; 12 son nodos de esta red).

| Estabilidad interna | ARI medio | ARI mínimo | ARI p5 |
|---|---|---|---|
| pipeline (python-louvain, 20 semillas) | 0.7585 | 0.565 | 0.6214 |
| bibliometrix (igraph Louvain, 20 semillas) | 0.8056 | 0.5245 | 0.6334 |

| Comparación | Nodos comunes | Cobertura (a / b) | k (a / b) | ARI | NMI | ARI azar (media / p95) |
|---|---|---|---|---|---|---|
| pipeline vs VOSviewer | 639 | 1.0 / 1.0 | 98 / 95 | 0.7038 | 0.9007 | -0.0004 / 0.0085 |
| pipeline vs bibliometrix | 498 | 0.816 / 0.866 | 49 / 44 | 0.6484 | 0.8588 | -0.0002 / 0.0098 |
| VOSviewer vs bibliometrix | 498 | 0.816 / 0.866 | 46 / 44 | 0.8766 | 0.9481 | 0.0001 / 0.0099 |
- VOSviewer usado en la tabla principal: LinLog/modularity.

Sensibilidad a la normalización (Louvain del pipeline, 10 semillas, resolución 1):

| Normalización | k medio | ARI vs VOSviewer | ARI vs bibliometrix |
|---|---|---|---|
| conteos brutos (pipeline actual) | 96.3 | 0.705 | 0.646 |
| association strength | 105.9 | 0.599 | 0.647 |

Detalle nodo a nodo: `concordancia_detalle/cocitation_emparejamiento.csv`.

### keyword_cooccurrence

- Nodos con al menos un enlace: pipeline 545, VOSviewer 0, bibliometrix 591. Aislados excluidos del pipeline/VOSviewer: 0.
- Emparejamiento con bibliometrix: palabra clave en minusculas.
- Control: la semilla 42 reproduce la partición del `.graphml` con ARI = 1.0.
- El pipeline canonicaliza palabras clave (sinonimos de knowledge_base.json, singularizacion); bibliometrix usa DE crudas. La cobertura mide ese efecto.

| Estabilidad interna | ARI medio | ARI mínimo | ARI p5 |
|---|---|---|---|
| pipeline (python-louvain, 20 semillas) | 0.6194 | 0.2718 | 0.3522 |
| bibliometrix (igraph Louvain, 20 semillas) | 0.6444 | 0.4738 | 0.5492 |

| Comparación | Nodos comunes | Cobertura (a / b) | k (a / b) | ARI | NMI | ARI azar (media / p95) |
|---|---|---|---|---|---|---|
| pipeline vs bibliometrix | 505 | 0.927 / 0.854 | 8 / 6 | 0.5673 | 0.5057 | 0.0002 / 0.0216 |
- VOSviewer usado en la tabla principal: ninguno (no hay map file vigente).
- Bibliometrix se construye con el **mismo umbral que el pipeline** (al menos 2 co-ocurrencias) para comparar redes iguales. Sensibilidad: con la red por defecto de bibliometrix, sin umbral (2541 nodos), el ARI pipeline vs bibliometrix baja a 0.1405 (NMI 0.2439).
- **Justificación del umbral de 2 co-ocurrencias.** Un enlace con una sola co-ocurrencia proviene de un único artículo: muestra que dos palabras aparecieron juntas una vez, no que estén asociadas en el campo. Exigir al menos dos artículos es la evidencia mínima de una asociación que se repite.
  - Estabilidad de bibliometrix entre semillas: sin umbral 0.3736, con umbral 0.6444; pipeline 0.6194.
  - Con el mismo umbral, palabras crudas frente a normalizadas: ARI 0.5291 (cuanto más se acerque a la estabilidad del pipeline, menos influye la normalización).
  - Si con umbral la concordancia se acerca a la estabilidad de cada herramienta, la diferencia sin umbral la produce el umbral y no el método.

> **VOSviewer desactualizado para esta red** (LinLog/modularity): el map file corresponde a una versión anterior de la red. Volver a agrupar en VOSviewer con `output/vosviewer/keyword_cooccurrence_map.txt` y `_network.txt` actuales.

Detalle nodo a nodo: `concordancia_detalle/keyword_cooccurrence_emparejamiento.csv`.

### coauthorship_countries

- Nodos con al menos un enlace: pipeline 36, VOSviewer 36, bibliometrix 35. Aislados excluidos del pipeline/VOSviewer: 26.
- Emparejamiento con bibliometrix: pais (ISO2).
- Control: la semilla 42 reproduce la partición del `.graphml` con ARI = 1.0.

| Estabilidad interna | ARI medio | ARI mínimo | ARI p5 |
|---|---|---|---|
| pipeline (python-louvain, 20 semillas) | 1.0 | 1.0 | 1.0 |
| bibliometrix (igraph Louvain, 20 semillas) | 1.0 | 1.0 | 1.0 |

| Comparación | Nodos comunes | Cobertura (a / b) | k (a / b) | ARI | NMI | ARI azar (media / p95) |
|---|---|---|---|---|---|---|
| pipeline vs VOSviewer | 36 | 1.0 / 1.0 | 7 / 7 | 0.9396 | 0.9589 | -0.002 / 0.0696 |
| pipeline vs bibliometrix | 35 | 0.972 / 1.0 | 7 / 7 | 1.0 | 1.0 | -0.0042 / 0.0574 |
| VOSviewer vs bibliometrix | 35 | 0.972 / 1.0 | 7 / 7 | 0.9351 | 0.9579 | -0.0047 / 0.066 |
- VOSviewer usado en la tabla principal: LinLog/modularity.

Sensibilidad a la normalización (Louvain del pipeline, 10 semillas, resolución 1):

| Normalización | k medio | ARI vs VOSviewer | ARI vs bibliometrix |
|---|---|---|---|
| conteos brutos (pipeline actual) | 7.0 | 0.94 | 1.0 |
| association strength | 9.8 | 0.618 | 0.582 |

Detalle nodo a nodo: `concordancia_detalle/coauthorship_countries_emparejamiento.csv`.

## B. Indicadores y conteos

Pipeline = datos embebidos en las figuras `output/plotly/*.html` (lo que se publicaría). Bibliometrix = tablas de `validacion_externa/bibliometrix/`.

### Producción anual

- Total: pipeline 347, bibliometrix 347 (13 años).
- Años con diferencia: 0.

### Crecimiento anual

- Pipeline 17.5 %, bibliometrix 17.48 %.

### Colaboración internacional (MCP)

- Artículos multipaís: pipeline 36, bibliometrix 35.
- Artículos que cambian de clase: Fini2018.
- Conjunto de países idéntico en 344 de 347 artículos. Sin país: pipeline 0, bibliometrix 0.
- País del autor de correspondencia coincide en 345 de 347.
- Detalle: `concordancia_detalle/paises_discrepancias.csv`.

### Fuentes más productivas

- Top 15: Jaccard 0.765, Spearman en los comunes 0.995. Valor en el corte: 3 documentos, con 1 empatados dentro del top del pipeline (el orden entre empatados es arbitrario en ambas herramientas).
- Primero en el pipeline: ASEE Annual Conference and Exposition, Conference  (50). Primero en bibliometrix: ASEE ANNUAL CONFERENCE AND EXPOSITION CONFERENCE PROCEEDINGS (50).

### Autores más productivos

- Top 15: Jaccard 0.364, Spearman en los comunes 0.333. Valor en el corte: 2 documentos, con 14 empatados dentro del top del pipeline (el orden entre empatados es arbitrario en ambas herramientas).
- Primero en el pipeline: Truyol, M.E. (3). Primero en bibliometrix: TRUYOL ME (3).

### Lotka (autores por número de documentos)

- Pipeline: {1: 1163, 2: 31, 3: 1}. Bibliometrix: {1: 1153, 2: 36, 3: 1}.
- Ajuste de bibliometrix: β = 6.265, R² = 0.975, p (KS) = 0.518.

### RPYS

- Años comunes 73; rango pipeline [1951, 2026], bibliometrix [1926, 2023].
- Spearman de citas por año: 0.997. Citas totales en años comunes: pipeline 14990, bibliometrix 9861.
- Picos (top 10 por desviación) comunes: [2000, 2006, 2009, 2019, 2020].
  Pipeline [1977, 1991, 1996, 2000, 2006, 2009, 2013, 2019, 2020, 2024]; bibliometrix [2000, 2006, 2007, 2009, 2012, 2016, 2017, 2018, 2019, 2020].
- Ambos: desviacion respecto de la mediana de 5 anos (pipeline: Fase 2g, RPYS estandar; bibliometrix: diffMedian5). bibliometrix solo ve las referencias que llegan al campo CR del export.

### coauthorship_authors (solo pipeline vs VOSviewer; sin equivalente en bibliometrix)

- Aislados excluidos: 47. Estabilidad del pipeline (20 semillas): ARI medio 1.0, mínimo 1.0, p5 1.0.

| VOSviewer con | Nodos comunes | k (pipeline / VOS) | ARI | NMI | ARI azar (media / p95) |
|---|---|---|---|---|---|
| LinLog/modularity | 1067 | 266 / 266 | 1.0 | 1.0 | 0.0 / 0.0026 |

### coupling_journals (solo pipeline vs VOSviewer; sin equivalente en bibliometrix)

- Aislados excluidos: 24. Estabilidad del pipeline (20 semillas): ARI medio 0.5096, mínimo 0.1888, p5 0.3324.

| VOSviewer con | Nodos comunes | k (pipeline / VOS) | ARI | NMI | ARI azar (media / p95) |
|---|---|---|---|---|---|
| LinLog/modularity | 145 | 4 / 7 | 0.3385 | 0.4586 | -0.0006 / 0.0211 |

### cocitation_journals (solo pipeline vs VOSviewer; sin equivalente en bibliometrix)

- Aislados excluidos: 90. Estabilidad del pipeline (20 semillas): ARI medio 0.6214, mínimo 0.372, p5 0.4302.

| VOSviewer con | Nodos comunes | k (pipeline / VOS) | ARI | NMI | ARI azar (media / p95) |
|---|---|---|---|---|---|
| LinLog/modularity | 890 | 8 / 90 | 0.0047 | 0.0938 | 0.0 / 0.0072 |

## C. Mapa temático (Callon)

- bibliometrix: Louvain sobre las 250 palabras mas frecuentes (minfreq 5). Pipeline: Louvain (semilla 42) sobre toda la red con co-ocurrencia >= 2 y clusters de 3 o mas palabras.
- Control: el recálculo reproduce la figura publicada: True (6 temas en la figura, 6 en el recálculo).
- Cuadrantes con corte en la mediana de centralidad y densidad de cada herramienta (con 4 o 5 temas, un tema cerca de la mediana cambia de cuadrante con facilidad).

### Pipeline oficial (palabras canonicalizadas)

- Temas: pipeline 6, bibliometrix 4. Palabras comunes: 196.
- Pertenencia de palabras a temas: ARI 0.2822, NMI 0.2457 (azar: media 0.0002, p95 0.0202).

| Tema bibliometrix | Cuadrante | Tema del pipeline más parecido | Cuadrante | Jaccard | ¿Mismo cuadrante? |
|---|---|---|---|---|---|
| engineering education | basico/transversal | engineering education; student; curriculum | basico/transversal | 0.606 | sí |
| engineering students | motor | self efficacy; engineering student; survey | motor | 0.412 | sí |
| higher education | emergente o en declive | higher education; artificial intelligence; critical thinking | basico/transversal | 0.304 | no |
| confirmatory factor analysis | nicho | self efficacy; engineering student; survey | motor | 0.151 | no |

### Método del pipeline sobre las mismas palabras de autor que bibliometrix

- Temas: pipeline 5, bibliometrix 4. Palabras comunes: 250.
- Pertenencia de palabras a temas: ARI 0.4837, NMI 0.4199 (azar: media -0.0006, p95 0.0176).

| Tema bibliometrix | Cuadrante | Tema del pipeline más parecido | Cuadrante | Jaccard | ¿Mismo cuadrante? |
|---|---|---|---|---|---|
| engineering education | basico/transversal | engineering education; students; curricula | basico/transversal | 0.661 | sí |
| engineering students | motor | engineering students; self efficacy; surveys | motor | 0.712 | sí |
| higher education | emergente o en declive | higher education; artificial intelligence; critical thinking | emergente o en declive | 0.37 | sí |
| confirmatory factor analysis | nicho | education; confirmatory factor analysis; learning | nicho | 0.514 | sí |

## D. Tendencia de tópicos

- Se compara el año mediano de uso de cada palabra con frecuencia ≥ 5 (método de `fieldByYear`, que es también el que usa ahora la figura del pipeline).
- Las fuentes no son las mismas: el pipeline usa el vocabulario común normalizado (palabras de autor, automáticas, OpenAlex y MeSH); bibliometrix, solo las palabras de autor (DE) sin normalizar. Por eso el pipeline tiene más palabras y la comparación se hace sobre las comunes.
- Palabras: pipeline 141, bibliometrix 183, comunes 124.
- Spearman del año mediano en las comunes: 0.98; diferencia media 0.2 años; año idéntico en 95.
- 15 más recientes en común: artificial intelligence, augmented reality, design methodology, e-learning, generative artificial intelligence, modeling, outcome-based education, student engagement, testing.
  - Pipeline: artificial intelligence, generative artificial intelligence, e-learning, design methodology, modeling, outcome-based education, testing, virtual reality, augmented reality, scale development, technology, visualization, feedback, student engagement, experimental groups.
  - Bibliometrix: artificial intelligence, generative ai, e-learning, printing, learning (artificial intelligence), outcome-based education, design methodology, generative artificial intelligence, modeling, student engagement, ai literacy, testing, augmented reality, challenge-based learning, design thinking.
- Referencia: con el método anterior de la figura (5 palabras mas frecuentes de cada ano (no ano mediano)), la palabra más frecuente en cada uno de los 13 años habría sido {'engineering education': 11, 'student': 2}: no muestra tendencia.
- Detalle: `concordancia_detalle/tendencia_topicos.csv`.
