# Criterios de cribado por título, resumen y año — INSTRUMENTOS_ING v3.1

Aplicar a todos los registros de corpus_1128.csv. Fuentes permitidas: UID como identificador, Ano (año bibliográfico suministrado), Titulo y Resumen. No consultar otros archivos, metadatos, artículos o votos. El año no se deduce del copyright ni del UID. NR es dato ausente, no un incumplimiento.

## Columnas de criterio y códigos

Evaluar en orden A1, A2, A3, A4, A5, A6, A7. El primer NO demostrado determina el criterio y código primario. La tabla explicita los pares para MAPA.md; no deducir otros códigos ni usar códigos de texto completo por analogía.

| Columna | Comprobación | Código si incumple positivamente |
|---|---|---|
| A1 | Tipo documental admisible: artículo original, early access original o ponencia completa indexada y revisada por pares; si no consta en las fuentes suministradas, INCIERTO. | EX_DOC |
| A2 | Ano dentro de 2014–2026 inclusive. Si es NR o hay conflicto esencial no resuelto, INCIERTO. | EX_YEAR_LANG |
| A3 | Idioma del artículo inglés, español o portugués. El idioma del resumen no confirma por sí solo el del artículo. Si no consta, INCIERTO. | EX_YEAR_LANG |
| A4 | Participación empírica de estudiantes universitarios de pregrado o posgrado en ingeniería, incluso muestras mixtas. | EX_POP |
| A5 | Competencia técnica o transversal declarada y medida en personas, según el alcance del protocolo. | EX_CONSTRUCT |
| A6 | Instrumento de medida de competencias humanas identificable: cuestionario, escala, inventario, rúbrica, prueba situacional, autoinforme o tarea de desempeño; no solo artefacto técnico. | EX_INSTRUMENT |
| A7 | Evaluación empírica original de una propiedad de medición de ese instrumento, con procedimiento o resultado identificable. | EX_ONLY_USE; EX_NO_ORIGINAL_PROPERTY |

EX_ONLY_USE en A7: uso meramente sustantivo positivamente demostrado de instrumento existente, sin evaluación original pertinente. EX_NO_ORIGINAL_PROPERTY en A7: ausencia positivamente demostrada de evaluación empírica original, no simple silencio del resumen. No usar EX_INSUFFICIENT_REPORT en título/resumen ni excluir por un resumen truncado o datos faltantes.

## Valores controlados de A1–A7

- SI: condición explícitamente satisfecha; sentido del contrato: satisface.
- NO: incumplimiento demostrado; sentido: contradice. Requiere evidencia del registro.
- INCIERTO: falta información o existe ambigüedad relevante; sentido: silencio. No equivale a NO.
- NO_EVALUADO: únicamente en criterios posteriores al primer NO demostrado, por detención temprana de la evaluación; sentido: no_evaluado. No usar antes del primer NO ni cuando no exista incumplimiento. Este marcador es de ejecución, no un criterio nuevo de elegibilidad.

Sin NO, completar A1–A7 con SI o INCIERTO. Si aparece NO, conservar los criterios anteriores y marcar NO_EVALUADO en los posteriores. No ejecutar descarte léxico previo, puntuaciones de exclusión ni inferencias por falta de información.

## Decisión y campos de registro

DECISION usa exactamente INCLUDE, UNCERTAIN o EXCLUDE:
- INCLUDE: A1–A7 SI y ninguna exclusión; sentido incluye.
- UNCERTAIN: no hay NO, hay al menos un INCIERTO y la interpretación elegible es plausible; sentido duda.
- EXCLUDE: al menos un NO demostrado; sentido excluye. Prevalece el primer NO en el orden A1–A7.

Criterio_aplicado: para EXCLUDE escribir el primer A# con NO; para UNCERTAIN indicar los A# inciertos; para INCLUDE escribir A1-A7. Es texto de registro, no otra columna con sentido.
Motivo: en EXCLUDE copiar exactamente uno de los códigos de su criterio primario (EX_DOC, EX_YEAR_LANG, EX_POP, EX_CONSTRUCT, EX_INSTRUMENT, EX_ONLY_USE, EX_NO_ORIGINAL_PROPERTY). En INCLUDE o UNCERTAIN dejar Motivo vacío: no inventar códigos de exclusión para dudas.
Evidencia: fragmento literal de Titulo o Resumen, o valor literal de Ano cuando este sea decisivo. No parafrasear, traducir ni fabricar frases; si se usan segmentos separados, conservarlos literalmente y separarlos con puntos suspensivos. Para silencio, citar lo disponible que sustenta plausibilidad y explicar la ausencia en Justificacion.
Justificacion: explicación concisa de la decisión, el criterio demostrado o los datos faltantes. No afirmar resultados de partes no suministradas del artículo.

En sentidos incluir solo A1–A7 y DECISION. No asignar sentidos a Criterio_aplicado, Motivo, Evidencia ni Justificacion.

## Aclaraciones de aplicación

TI, STEM, computación o formación vocacional/politécnica no confirman ni excluyen automáticamente ingeniería. Mantener A4 INCIERTO si la composición es ambigua y existe plausibilidad; no admitir esas poblaciones automáticamente en texto completo.

Engagement cognitivo, estrategias de aprendizaje, autorregulación, gestión del conocimiento e indicadores grupales no son automáticamente competencias elegibles ni exclusiones. Mantener A5 incierto si su relación con una competencia humana es plausible y no está clara; no ampliar el alcance a todo atributo psicológico o resultado de grupo.

SEM, asociaciones, comparaciones de grupos, un objetivo sustantivo o una escala establecida no demuestran por sí solos ausencia de evaluación original. Tampoco toda asociación o validación de un modelo causal demuestra una propiedad del instrumento. Para A7 SI debe identificarse una evaluación original pertinente; si solo se afirma «validado» o «tested for validity and reliability» sin procedimiento/resultado identificable, A7 INCIERTO. Excluir por solo uso únicamente si está demostrado, no por silencio.

Una rúbrica puede tener evidencia pertinente sin análisis factorial o alfa. Utilidad diagnóstica, notas, mejora de rendimiento o propuestas de mejora no establecen automáticamente una propiedad ni prueban que sea imposible su presencia. No exigir una técnica única ni resultados favorables para incluir. No inferir contenido de una parte truncada o ausente.

INCLUDE y UNCERTAIN determinan candidatos a texto completo. La localización y la aplicación de FT01–FT06 se realizan después del cribado global y su integración, no durante esta ejecución.
