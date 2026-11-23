# Manual de uso — Análisis bibliométrico y cienciométrico

Guía para correr una revisión **de principio a fin sin asistencia**: preparar insumos, ejecutar, revisar el vocabulario (opcional), validar contra herramientas externas y saber qué se puede publicar.

> Versión del flujo: 2026-11-22. Los cambios de método están en `CHANGELOG.md`. `manual.md` es documentación histórica y **no** describe el comportamiento actual.

---

## 0. Qué hace y qué no hace

- **No depende del tema.** El vocabulario de palabras clave, las siglas, los métodos y los vacíos de conocimiento se aprenden del propio corpus, con reglas generales que sirven para cualquier campo. No hay listas de un tema ni sinónimos de revisiones anteriores.
- **Es determinista.** La misma entrada da la misma salida: los algoritmos con azar usan semilla fija.
- **Solo funciona en inglés.** La detección de métodos y de vacíos, el singularizador, las siglas y las stopwords asumen inglés. Los resúmenes en otros idiomas aportan poco a esos análisis.
- **Está validado contra bibliometrix 5.2.1 y VOSviewer 1.6.20** en un corpus de educación en ingeniería (347 artículos, noviembre de 2026). Que funcione igual en otros temas es una hipótesis: para comprobarlo hay que validar cada revisión nueva con el procedimiento de la sección 6.
- **No reemplaza el juicio de expertos.** La concordancia entre herramientas indica que un resultado no depende del software, no que sea correcto. Para validar temas, métodos o vacíos hace falta una revisión humana, según `docs/PROTOCOLO_VALIDACION.md` del monorepo.

---

## 1. Requisitos (una sola vez por máquina)

| Qué | Para qué | Cómo |
|---|---|---|
| `uv` y Python 3.12 | Pipeline | `uv sync` en la carpeta del proyecto (crea `.venv` desde `uv.lock`) |
| Credenciales de API | Enriquecimiento (Fase 1) | Archivo central `~/.automatizaciones/.env`. Desde la raíz del monorepo: `python scripts/credenciales.py --init`. Sin `S2_API_KEY`, Semantic Scholar es lento (errores 429), pero funciona |
| R 4.5 + `bibliometrix` + `igraph` | Validación externa | `install.packages(c("bibliometrix", "igraph"))` |
| VOSviewer 1.6.20 | Validación externa (paso manual) | <https://www.vosviewer.com> |

Todo se ejecuta desde una terminal PowerShell en la carpeta del proyecto. `main.py` se redirige solo a `.venv` si se lo llama con otro Python.

---

## 2. Insumos de una revisión

### 2.1 `consolidado.csv` (obligatorio, en la raíz)

Es la exportación consolidada de la revisión, normalmente generada con `bibcombiner5` y **copiada** a esta carpeta: los proyectos del monorepo no leen archivos de carpetas hermanas. Va separado por `;`.

| Columna | Necesidad | Si falta |
|---|---|---|
| `Title`, `Publication_Year`, `Abstract` | Imprescindible | Si **ningún** artículo tiene año o resumen, la corrida se detiene con un mensaje claro (sección 8) |
| `Author_Keywords` | Imprescindible para los análisis de palabras clave | Si ningún artículo las tiene, la corrida se detiene. Si faltan en algunos, esos artículos no entran a redes, mapa temático ni tendencias, y el log informa cuántos son |
| `Authors`, `DOI`, `Journal_Name`, `Affiliations`, `Index_Keywords`, `Source_DB`, `Merged_From` | Recomendadas | Se completan en parte con OpenAlex y Crossref; sin ellas hay menos precisión |

> **Palabras clave de autor.** El vocabulario común usa **solo** `Author_Keywords`. Las palabras automáticas (`Index_Keywords`, Keywords Plus, OpenAlex) no entran, porque las asigna una máquina y meten ruido (convención de bibliometrix para `DE`).

### 2.2 Archivos de `data/` que son de cada revisión (opcionales)

| Archivo | Qué aporta |
|---|---|
| `CitationScopus.csv` | Citas por año en Scopus; procedencia de los registros |
| `CitationWos.xlsx` o `CitationWos.csv` | Citas por año en WoS; DOIs faltantes |
| `scopusReferences.ris` | Referencias citadas (RPYS, co-citación) |
| `scopusCitadopor.ris` | Artículos que citan al corpus |
| `sinonimos_revision.json` | Decisiones del equipo sobre el vocabulario (sección 5). Puede no existir |
| `curaduria_vocabulario.xlsx` | Planilla de curaduría (sección 5). Puede no existir |

**Si un archivo es de la revisión anterior, contamina los resultados sin dar ningún error.** `--new-review` los conserva y los lista en su aviso: reemplácelos o bórrelos.

### 2.3 Archivos que se conservan entre revisiones

- `scimago/scimagojr AAAA.csv`: SJR y cuartil por año.
- `data/knowledge_base.json`: recursos **genéricos** (países, subencabezados MeSH, etiquetas de indexación). No debe tener sinónimos de un tema; si los tiene, el log avisa.

---

## 3. Flujo de una revisión nueva

```powershell
# 1. Reset de la revisión anterior (borra data/ derivado y output/; conserva los insumos)
.\.venv\Scripts\python.exe main.py --new-review
#    -> Lea el aviso: lista los archivos conservados que pueden ser de la revisión anterior.
#       Reemplácelos o bórrelos. Para empezar sin decisiones de vocabulario:
#       borre data\sinonimos_revision.json y data\curaduria_vocabulario.xlsx

# 2. Copie el consolidado.csv nuevo a la raíz del proyecto

# 3. Carga y enriquecimiento (Fases 0 y 1; consulta APIs; puede tardar)
.\.venv\Scripts\python.exe main.py --phases 0 1 --run-id <revision>_carga

# 4. (OPCIONAL) Curaduría del vocabulario por el equipo -> sección 5

# 5. Análisis (Fases 2 a 7)
.\.venv\Scripts\python.exe main.py --phases 2 3 4 5 7 --run-id <revision>_analisis

# 6. Validación externa -> sección 6
```

**Reglas de trazabilidad:**
- Use siempre `--run-id`. Cada corrida deja `output/logs/run_<fecha>.log` y `run_manifest_<fecha>.json`, con los parámetros, el estado de cada fase y la huella de los insumos.
- **No modifique el código mientras corre el pipeline.** Cada fase carga el código en el momento en que empieza, y una corrida con versiones mezcladas no sirve.
- Las cifras del paper deben salir de una corrida registrada (con manifiesto), no de pruebas sueltas.

### Qué pasa al correr las fases 2 a 7

Antes de la Fase 2 se construye el **modelo del corpus** (`data/modelo_corpus.json`). Es una sola extracción de lo que usan todas las fases: palabras clave de autor normalizadas, métodos y vacíos declarados, países (código ISO, contados por artículo), instituciones, siglas y vocabulario de métodos. En el log aparecen la línea de **cobertura de insumos** y la del **modelo del corpus**. Revíselas.

| Fase | Contenido |
|---|---|
| 2 | 2a descriptivo · 2b resúmenes (métodos y vacíos declarados) · 2c SciBERT · 2d calidad de resúmenes · 2e indicadores (tendencia de tópicos por año mediano) · 2f estado del arte y vacíos · 2g RPYS (Marx et al., 2014) · 2h análisis al estilo CiteSpace · 2i indicadores CWTS Leiden |
| 3 | Redes (acoplamiento, co-citación, co-ocurrencia, coautoría) y diagrama estratégico de Callon (3b) |
| 4 | Exportaciones: bibliometrix, VOSviewer, Excel, figuras interactivas |
| 5 | Evolución del impacto e intervalos de confianza por bootstrap (5b) |
| 6 | RDM. **Desactivada por defecto** (índice propio sin validación); solo corre con `--enable-rdm` y no entra a la Fase 7 |
| 7 | Knowledge Map. Úselo solo como **tabla de indicadores** (sección 7) |

---

## 4. Qué revisar después de cada corrida

1. **Log, línea de cobertura:** cuántos artículos tienen palabras clave de autor, resumen, año, países y referencias.
2. **`output/reports/normalizacion_palabras_clave.csv`:** cada variante unificada, con su término final, la regla que la produjo y su frecuencia. Hay que revisar las **siglas ambiguas**, que no se unifican solas (p. ej. PBL: *problem-based* o *project-based learning*).
3. **`output/reports/metodos_declarados.csv`:** cada candidato a método, con los artículos que lo declaran y los que lo mencionan, y por qué entra o no al vocabulario.
4. **Manifiesto** (`output/logs/run_manifest_*.json`): todas las fases deben estar en `ok`. `skipped` en la fase 6 es lo esperado.

Si algo de los puntos 2 o 3 no corresponde a su campo, use la curaduría (sección 5).

---

## 5. Curaduría del vocabulario por el equipo (OPCIONAL)

El vocabulario que se aprende del corpus es una **propuesta**. El equipo puede revisarlo y corregirlo en Excel antes del análisis. Si no lo hace, se usa lo automático.

```powershell
# a) Después de las Fases 0 y 1:
.\.venv\Scripts\python.exe scripts\curaduria_vocabulario.py --exportar
#    -> data\curaduria_vocabulario.xlsx

# b) El equipo decide en las celdas AMARILLAS (menús desplegables) y justifica cada decisión.

# c) Validar y aplicar:
.\.venv\Scripts\python.exe scripts\curaduria_vocabulario.py --importar
#    -> data\sinonimos_revision.json  (copia previa: sinonimos_revision.anterior.json)

# d) Correr las fases 2 a 7 (sección 3, paso 5)
```

| Hoja | Para qué sirve | Opciones |
|---|---|---|
| Unificaciones | Revisar variantes que el algoritmo unió | `aceptar` / `no unificar` / `cambiar` (indicar el término nuevo) |
| Siglas ambiguas | Elegir la expansión correcta de una sigla con varias | `ninguna` / `1`–`4` |
| Sinónimos nuevos | Equivalencias que el algoritmo no detectó | variante → término canónico |
| Métodos | Ajustar el vocabulario de métodos | `automatico` / `incluir` / `excluir` |
| Términos interés | Términos que el equipo quiere seguir siempre | Se informan en `output/reports/terminos_de_interes.csv` **sin cambiar ningún análisis** |
| Exclusiones | Términos a quitar del vocabulario (p. ej. etiquetas de indexación) | término + justificación |
| Vocabulario | Solo consulta: todos los términos con su frecuencia | — |

**Criterios para decidir:**
- **Sinónimo:** solo si los dos términos significan lo mismo **en este campo**. Si se duda, no unifique.
- **No unificar:** cuando la regla automática juntó dos cosas distintas (p. ej. un plural que en el campo tiene otro sentido).
- **Incluir un método:** solo si es un método del campo que el corpus nombra poco. Su inclusión queda registrada como decisión del equipo.
- **Términos de interés:** para seguir un concepto de la pregunta de investigación sin forzar su aparición en redes ni mapas.
- **Toda decisión distinta de la automática exige justificación.** Sin ella, `--importar` rechaza la planilla y no cambia nada. Las justificaciones quedan en `sinonimos_revision.json` y pueden citarse en el método del paper.

---

## 6. Validación externa (antes de publicar)

Se hace sobre la **corrida registrada** cuyos resultados van al paper.

```powershell
# 6.1 bibliometrix (recalcula indicadores y clusters con otra implementación)
& "C:\Program Files\R\R-4.5.1\bin\Rscript.exe" scripts\validacion_bibliometrix.R
#    -> validacion_externa\bibliometrix\
```

**6.2 VOSviewer (paso manual).** Para cada red (`bibliographic_coupling`, `cocitation`, `keyword_cooccurrence`, `coauthorship_countries`, `coauthorship_authors`, `coupling_journals`, `cocitation_journals`):

1. *Create → Create a map based on network data*. Cargue `output\vosviewer\<red>_map.txt` y `<red>_network.txt`.
2. *Analysis*: **Normalization method = LinLog/modularity**, *Resolution* 1.00, *Min. cluster size* 1, *Merge small clusters* marcado. Pulse **Update clustering** y luego **Update layout**. Sin *Update clustering*, VOSviewer conserva los clusters del pipeline y la validación no sirve.
3. *Save → Save map file* como `validacion_externa\vosviewer\<red>_map_linLog.txt`.
4. Si cambia algún parámetro, anótelo en `validacion_externa\vosviewer\parametros.txt`.

```powershell
# 6.3 Concordancia pipeline - bibliometrix - VOSviewer
.\.venv\Scripts\python.exe scripts\concordancia_externa.py
#    -> validacion_externa\informe_concordancia.md

# 6.4 Piezas de la Fase 2h (ráfagas de Kleinberg y sigma) contra una referencia independiente
.\.venv\Scripts\python.exe scripts\validacion_citespace.py
#    -> validacion_externa\citespace\informe_validacion_citespace.md
```

**Cómo leer `informe_concordancia.md`:**
- **ARI y NMI** miden cuánto coinciden dos particiones (1 = idénticas; cerca de 0 = como al azar).
- No hay un umbral fijado a mano. Cada valor se compara con la **estabilidad interna**, que es el ARI de la misma herramienta corrida con 20 semillas distintas:
  - si la concordancia entre herramientas se acerca a esa estabilidad, la estructura **no depende del software**;
  - si queda cerca del azar, es un efecto del algoritmo o de cómo se armó la red.
- Si una red figura como **«VOSviewer desactualizado»**, la red cambió desde que se agrupó en VOSviewer: repita 6.2 para esa red.
- La red de palabras clave de bibliometrix se construye con el mismo umbral que el pipeline (al menos 2 co-ocurrencias). El informe muestra también la versión sin umbral, y la justificación con las cifras de esa corrida.

---

## 7. Qué se puede publicar (estado al 2026-11-22)

| Resultado | Cómo usarlo |
|---|---|
| Descriptivos, producción, países (MCP), fuentes | Directo (concuerdan con bibliometrix) |
| Indicadores con intervalo de confianza (5b), Leiden (2i) | Directo, con su IC |
| RPYS (2g) | Directo. Los picos con alta *concentración* son raíces históricas; los de baja concentración son picos de volumen. Los últimos 3 años están incompletos |
| Estructura de co-citación y de países | Directo (concordancia alta en las tres herramientas) |
| Estructura de palabras clave y de acoplamiento | **Con su estabilidad**: reportar el ARI entre semillas y la concordancia externa |
| Diagrama estratégico (Callon, 3b) | Directo. Con 4 a 8 temas, un tema cerca de la mediana puede cambiar de cuadrante |
| Tendencia de tópicos (2e) | Directo (año mediano, Q1–Q3). Advertir si el último año está incompleto |
| Métodos y vacíos declarados (2b, 2f) | Como descripción de lo que declara el corpus, no como evaluación de calidad |
| Fase 2h | Como **«análisis al estilo CiteSpace, implementación propia»**, citando los métodos (Kleinberg, 2002; Freeman, 1977; Chen et al., 2009). **No** citar CiteSpace como herramienta |
| Lotka | Solo descriptivo; con pocos autores prolíficos el ajuste informa poco |
| Fase 7 (Knowledge Map) | **Solo la tabla de indicadores.** El *impact score*, la «triangulación» y los «cross-cutting themes» no tienen sustento todavía |
| Fase 6 (RDM) | No publicar (no validada) |
| `output/figures/vosviewer_static/` | Son figuras **al estilo** VOSviewer hechas por el pipeline. Para figuras «hechas con VOSviewer», use las de la sección 6.2 |

**Declarar siempre en el método del paper:**
- análisis limitado a corpus en inglés;
- solo palabras clave de autor;
- umbral de 2 co-ocurrencias;
- Louvain con semilla 42;
- normalización del vocabulario aprendida del corpus, con las decisiones del equipo, si las hubo;
- en el export para bibliometrix, solo entran al campo CR las referencias resueltas en OpenAlex: bibliometrix no ve todas.

---

## 8. Solución de problemas

| Síntoma | Causa | Qué hacer |
|---|---|---|
| `Corrida detenida: Ningun articulo tiene ...` (código de salida 3) | Falta por completo un insumo crítico en `consolidado.csv` | Reexportar con la columna indicada. Si es intencional, agregar `--permitir-insumos-incompletos` |
| `El checkpoint pertenece a OTRA revision` | Se cambió `consolidado.csv` sin reset | `main.py --new-review` y repetir desde la Fase 0 |
| Muchos `429` de Semantic Scholar | Límite de la API sin clave | Agregar `S2_API_KEY` al `.env` central, o `--skip-s2` |
| `knowledge_base.json contiene contenido de dominio` | Se agregaron sinónimos de un tema a la base genérica | Pasarlos a `sinonimos_revision.json` (o a la planilla) |
| `--importar` dice «La planilla tiene errores» | Falta una justificación, hay un ciclo de sinónimos o una decisión contradictoria | Corregir lo que lista el mensaje; no se modificó nada |
| `VOSviewer desactualizado` en el informe | La red cambió después de agruparla en VOSviewer | Repetir 6.2 para esa red |
| `uv add` falla con `UnicodeDecodeError` | Windows lee archivos como cp1252 | `$env:PYTHONUTF8 = "1"; uv add ...` |
| La fecha de una carpeta de `output/` no cambia | Windows solo cambia la fecha de una carpeta cuando se crean o borran archivos, no cuando se sobrescriben | Mirar la fecha de cada archivo |

---

## 9. Referencia rápida

| Acción | Comando |
|---|---|
| Ver todas las opciones | `.\.venv\Scripts\python.exe main.py --help` |
| Reset para otra revisión | `.\.venv\Scripts\python.exe main.py --new-review` |
| Borrar solo `output/` (conserva datos y cachés) | `.\.venv\Scripts\python.exe main.py --clean-output` |
| Repetir los análisis sin volver a consultar APIs | `.\.venv\Scripts\python.exe main.py --phases 2 3 4 5 7 --run-id <id>` |
| Correr también la RDM | agregar `--enable-rdm` |
| Revisar que no haya nombres sin definir en el código | `uv run pyflakes src main.py config.py` |

**Dónde están los resultados:**
- `output/reports/`: descriptivo (`descriptive_report.md`), resúmenes (`abstract_analysis_report.md`), indicadores (`supplementary_indicators_report.md`), Callon (`strategic_thematic_report.md`), Excel maestro (`bibliometric_report.xlsx`), normalización, métodos y términos de interés.
- `output/sota_analysis/sota_gap_report.md`: estado del arte y vacíos.
- `output/spectroscopy/spectroscopy_report.md`: RPYS.
- `output/citespace/citespace_report.md`: análisis al estilo CiteSpace.
- `output/leiden/leiden_summary.md`: indicadores de Leiden.
- `output/reproducibility/bootstrap_ci.md`: intervalos de confianza.
- `output/knowledge_map/knowledge_map_report.md`: tabla de indicadores (ver sección 7).
- `output/bibliometrix_export.csv`, `.xlsx` y `.bib`, `output/vosviewer/`: exportaciones para otras herramientas. En **biblioshiny**, cargue **`bibliometrix_export.xlsx`** con *Load bibliometrix file* (no acepta `.csv` en ese menú). Para *Thematic Evolution*: campo *Author's Keywords*, al menos un corte de años y frecuencia mínima baja. El *Historiograph* usa las citas locales entre artículos del corpus (`AU` y `J9` van en formato WoS para que se emparejen).
- `output/figures/` (PNG) y `output/plotly/` (HTML): figuras.
- `validacion_externa/`: validación con bibliometrix, VOSviewer y la Fase 2h.
