# Decisiones de Diseño — Sistema Bibliométrico Determinista (`biblio1403`)
Actualizado: 2026-11-02

## 1. Propósito y filosofía arquitectónica

El sistema es un **pipeline de enriquecimiento y análisis bibliométrico/cienciométrico de frontera 100 % determinista y agnóstico al dominio**. Recibe cualquier corpus bibliográfico en la forma de `consolidado.csv` y produce indicadores, redes, pruebas estadísticas no paramétricas y reportes reproducibles bit a bit.

### Principios rectores

1. **Cero dependencias de LLMs y cero alucinaciones**:
   - Se eliminaron todas las fases y submódulos basados en LLMs o heurísticas cualitativas externas (antiguas Fases 0b, 2e, 4/4b, 5c, 7/7a, 9 y las capas narrativas Layer B de las Fases 2g y 8).
   - Todo número, tabla, red y hallazgo proviene exclusivamente de algoritmos deterministas auditables sobre los metadatos enriquecidos y las referencias resueltas.

2. **Agnosticismo total de dominio y de entrada (`consolidado.csv`)**:
   - El cargador (`src/data_loader.py`) detecta automáticamente el delimitador (`,`, `;`, `\t`), soporta esquemas de exportación de Zotero, Scopus, Web of Science y tablas consolidadas, deduplica defensivamente por DOI y por título normalizado, y genera claves canónicas legibles `ApellidoAño[sufijo]`.
   - La huella criptográfica (`data/review_fingerprint.json`) vincula `data/enriched_articles.json` directamente con `(csv_path, csv_sha256)`. Si el archivo CSV cambia, el sistema impide mezclar estados entre revisiones distintas y solicita `python main.py --new-review`.

3. **Jerarquía temática CWTS Leiden (`topics` de OpenAlex) frente a conceptos MAG heredados**:
   - OpenAlex deprecó los antiguos `concepts` basados en Microsoft Academic Graph (propensos a homónimos ruidosos como `Context (archaeology)`, `Test (biology)` o `Paleontology`).
   - Todos los módulos temáticos, de espectroscopía (Fase 2h), de evolución de impacto (Fase 6), de vacíos estructurales (Fase 2g) y de redes de palabras clave (Fase 2i/3) priorizan `topics` (`domain -> field -> subfield -> topic`) y aplican el filtro `is_noisy_legacy_concept()` (`src/utils.py`) sobre cualquier etiqueta de respaldo.

4. **Canonicalización transversal de metadatos**:
   - **Idiomas**: `normalize_language()` (`src/utils.py`) traduce códigos ISO-639 (`en`, `es`, `pt`, `tr`, etc.) y variantes a nombres canónicos en inglés (`English`, `Spanish`, `Portuguese`, `Turkish`).
   - **Revistas**: `canonicalize_corpus_journals()` (`src/utils.py`) limpia sufijos editoriales redundantes (`(Switzerland)`), unifica puntuación (`&` vs `and`, `:` vs `-`) y consolida títulos que comparten el mismo ISSN.
   - **Acceso Abierto**: `get_canonical_oa_status()` (`src/utils.py`) aplica una regla única en todo el pipeline: el estatus `diamond` (declarado por OpenAlex o verificado por DOAJ + APC = 0) prevalece sobre el `gold` genérico de Unpaywall.

5. **Cobertura íntegra de literatura citada (`references_resolved`)**:
   - La Fase 1i (`src/reference_resolver.py`) y los scripts de rescate (`scripts/recuperar_referencias_s2.py` y `scripts/importar_referencias_manuales.py`) almacenan cada referencia citada resuelta en `references_resolved` con su año, primer autor, DOI, `topics` y citas.
   - Tanto la espectroscopía RPYS (Fase 2h) como el Índice de Price y la semivida de citación (Fase 2f y Fase 6/6b) leen `references_resolved` primero, garantizando que las referencias rescatadas fuera de OpenAlex entren efectivamente en todos los cálculos.

---

## 2. Entradas que persisten entre revisiones (`--new-review`)

Únicamente dos archivos se consideran insumos portables al ejecutar `python main.py --new-review`:
1. `consolidado.csv` (o la ruta indicada por `--csv`) — el corpus de entrada.
2. `data/scimagojr.csv` — tabla local de cuartiles e índice SJR de SCImago (opcional, usada en Fase 1e).

Todo el resto de `data/` (incluido `data/enriched_articles.json` y las cachés de API) y `output/` es estado derivado de la revisión activa.

---

## 3. Comandos de referencia

```bash
# Limpiar estado derivado antes de cambiar de corpus
.venv\Scripts\python.exe main.py --new-review-yes

# Ejecutar enriquecimiento + análisis completo desde cero
.venv\Scripts\python.exe main.py --phases 0 1 2 3 4 5 6 7 --run-id corrida_completa

# Re-ejecutar todas las fases analíticas usando el checkpoint enriquecido
.venv\Scripts\python.exe main.py --phases 2 3 4 5 6 7 --skip-bert --skip-ref-resolution --run-id reanalisis

# Recuperar referencias faltantes vía Semantic Scholar y luego re-ejecutar análisis
.venv\Scripts\python.exe scripts/recuperar_referencias_s2.py --aplicar
.venv\Scripts\python.exe main.py --phases 2 3 4 5 6 7 --skip-bert --skip-ref-resolution --run-id post_s2
```
