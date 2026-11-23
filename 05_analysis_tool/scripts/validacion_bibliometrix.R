# Validacion externa con bibliometrix: recalcula con una implementacion independiente
# los indicadores y estructuras que produce el pipeline, y los deja como CSV para
# compararlos (scripts de concordancia aparte). No genera figuras: solo tablas.
#
# Uso (desde la raiz del proyecto, despues de la Fase 4):
#   Rscript scripts/validacion_bibliometrix.R [csv] [carpeta_salida]
# Por defecto: output/bibliometrix_export.csv -> validacion_externa/bibliometrix/
#
# Independencia: bibliometrix recalcula por su cuenta AU_CO (desde C1) y AU1_CO (desde RP);
# los valores que trae el export del pipeline se conservan como *_PIPELINE para compararlos.

suppressPackageStartupMessages({
  library(bibliometrix)
  library(igraph)
})

args <- commandArgs(trailingOnly = TRUE)
csv_path <- if (length(args) >= 1) args[1] else file.path("output", "bibliometrix_export.csv")
out_dir  <- if (length(args) >= 2) args[2] else file.path("validacion_externa", "bibliometrix")

if (!file.exists(csv_path)) {
  stop(sprintf("No existe %s. Ejecute primero: python main.py --phases 4", csv_path))
}
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

SEMILLA <- 42
SEMILLAS_ESTABILIDAD <- 1:20   # para medir la estabilidad interna de Louvain
K_TOP <- 20

log_path <- file.path(out_dir, "validacion_bibliometrix.log")
log_con <- file(log_path, open = "wt", encoding = "UTF-8")
logm <- function(...) {
  msg <- paste0(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " ", paste0(...))
  writeLines(msg, log_con); flush(log_con); message(msg)
}
guardar <- function(df, nombre) {
  p <- file.path(out_dir, nombre)
  write.csv(df, p, row.names = FALSE, fileEncoding = "UTF-8", na = "")
  logm("  guardado: ", nombre, " (", NROW(df), " filas)")
}
paso <- function(nombre, expr) {
  logm(nombre)
  tryCatch(expr, error = function(e) logm("  ERROR en ", nombre, ": ", conditionMessage(e)))
}

# --- Procedencia ---------------------------------------------------------------
logm("CSV: ", normalizePath(csv_path), "  md5=", unname(tools::md5sum(csv_path)))
logm("bibliometrix ", as.character(packageVersion("bibliometrix")),
     " | igraph ", as.character(packageVersion("igraph")), " | ", R.version.string)

# --- 1. Carga -----------------------------------------------------------------
M <- read.csv(csv_path, stringsAsFactors = FALSE, fileEncoding = "UTF-8-BOM",
              check.names = FALSE)
for (col in colnames(M)) if (is.character(M[[col]])) M[[col]][is.na(M[[col]])] <- ""
M$DB[M$DB == ""] <- "ISI"
M$DT[M$DT == ""] <- "ARTICLE"
for (col in c("DE", "ID", "CR", "C1", "AU")) {
  if (col %in% colnames(M)) M[[col]] <- gsub("\\s*;\\s*", ";", M[[col]])
}
for (col in c("DE", "ID", "CR")) M[[col]][M[[col]] == ""] <- NA

# Paises: conservar los del pipeline y dejar que bibliometrix los derive por su cuenta
if ("AU_CO" %in% colnames(M))  { M$AU_CO_PIPELINE  <- M$AU_CO;  M$AU_CO  <- NULL }
if ("AU1_CO" %in% colnames(M)) { M$AU1_CO_PIPELINE <- M$AU1_CO; M$AU1_CO <- NULL }

# SR = KEY del pipeline: identificador unico que permite emparejar nodos articulo
M$SR_ORIGINAL <- M$SR
M$SR <- M$KEY
rownames(M) <- M$SR
class(M) <- c("bibliometrixDB", "data.frame")

M <- metaTagExtraction(M, Field = "AU_CO", sep = ";")
M <- metaTagExtraction(M, Field = "AU1_CO", sep = ";")
logm("Registros: ", nrow(M), " | anos ", min(M$PY, na.rm = TRUE), "-", max(M$PY, na.rm = TRUE),
     " | con CR: ", sum(!is.na(M$CR)), " | con DE: ", sum(!is.na(M$DE)))

col_o_vacio <- function(x) if (is.null(x)) rep("", nrow(M)) else x
guardar(data.frame(KEY = M$KEY,
                   AU_CO_bibliometrix = col_o_vacio(M$AU_CO),
                   AU_CO_PIPELINE = col_o_vacio(M$AU_CO_PIPELINE),
                   AU1_CO_bibliometrix = col_o_vacio(M$AU1_CO),
                   AU1_CO_PIPELINE = col_o_vacio(M$AU1_CO_PIPELINE)),
        "00_paises_por_registro.csv")

# --- 2. Analisis descriptivo ---------------------------------------------------
results <- NULL
paso("2. biblioAnalysis + summary", {
  results <- biblioAnalysis(M, sep = ";")
  S <- summary(results, k = K_TOP, pause = FALSE, verbose = FALSE)
  guardar(S$MainInformationDF,  "01_informacion_principal.csv")
  guardar(S$AnnualProduction,   "02_produccion_anual.csv")
  guardar(S$MostRelSources,     "03_fuentes_top.csv")
  guardar(S$MostProdAuthors,    "04_autores_top.csv")
  guardar(S$MostCitedPapers,    "05_documentos_mas_citados.csv")
  guardar(S$MostProdCountries,  "06_paises_scp_mcp.csv")
  guardar(S$TCperCountries,     "06b_citas_por_pais.csv")
  guardar(data.frame(indicador = "Annual growth rate %", valor = S$AnnualGrowthRate),
          "02b_tasa_crecimiento.csv")
})

paso("3. Bradford", guardar(bradford(M)$table, "07_bradford.csv"))
paso("4. Lotka", {
  L <- lotka(M)   # bibliometrix >= 5 recibe M, no el resultado de biblioAnalysis
  guardar(L$AuthorProd, "08_lotka_productividad.csv")
  guardar(data.frame(beta = L$Beta, C = L$C, R2 = L$R2,
                     p_value_ks = if (!is.null(L$p.value)) L$p.value else NA),
          "08b_lotka_ajuste.csv")
})

# --- 3. Estructura conceptual -------------------------------------------------
for (campo in c("DE", "ID")) {
  paso(paste0("5. Mapa tematico (Callon) sobre ", campo), {
    set.seed(SEMILLA)
    tm <- thematicMap(M, field = campo, n = 250, minfreq = 5, stemming = FALSE,
                      size = 0.5, n.labels = 3, repel = FALSE)
    cl <- tm$clusters
    cl <- cl[, !vapply(cl, is.list, logical(1)), drop = FALSE]
    guardar(cl, sprintf("09_mapa_tematico_%s_clusters.csv", campo))
    guardar(tm$words, sprintf("09_mapa_tematico_%s_palabras.csv", campo))
  })
}

paso("6. Tendencia de topicos (ID)", {
  tt <- fieldByYear(M, field = "ID", min.freq = 5, n.items = 5, graph = FALSE)
  guardar(tt$df, "10_tendencia_topicos.csv")
})
# DE (palabras clave de autor): es el campo mas cercano a lo que usa el pipeline (manual_tags)
paso("6b. Tendencia de topicos (DE)", {
  tt <- fieldByYear(M, field = "DE", min.freq = 5, n.items = 5, graph = FALSE)
  guardar(tt$df, "10_tendencia_topicos_DE.csv")
})

paso("7. RPYS", {
  r <- rpys(M, sep = ";", graph = FALSE)
  guardar(r$rpysTable, "11_rpys_por_ano.csv")
  guardar(r$CR, "11b_rpys_referencias.csv")
})

# --- 4. Redes y clusters ------------------------------------------------------
# Para cada red, sobre la matriz que construye bibliometrix (independiente del pipeline):
#  (a) Louvain de igraph con semilla fija -> particion de referencia;
#  (b) Louvain con 20 semillas -> estabilidad interna, la referencia contra la cual
#      se juzga la concordancia externa (no un umbral fijado a mano);
#  (c) networkPlot de bibliometrix con cluster='louvain', si no falla (falla en redes
#      pequenas y densas como la de paises; entonces se registra y se sigue).
# min_peso: enlaces con peso < min_peso se eliminan (co-citacion: 2, como el pipeline).
clusters_red <- function(net_matrix, nombre, min_peso = 1) {
  W <- as.matrix(net_matrix)
  diag(W) <- 0
  W[W < min_peso] <- 0
  g <- graph_from_adjacency_matrix(W, mode = "undirected", weighted = TRUE, diag = FALSE)
  g <- delete_vertices(g, which(degree(g) == 0))
  logm("  red ", nombre, ": ", vcount(g), " nodos, ", ecount(g), " enlaces (min_peso=", min_peso, ")")

  set.seed(SEMILLA)
  ref <- cluster_louvain(g)
  guardar(data.frame(nodo = V(g)$name, cluster = as.integer(membership(ref)),
                     grado = degree(g), fuerza = strength(g)),
          sprintf("12_%s_clusters_igraph.csv", nombre))
  logm("  modularidad (semilla ", SEMILLA, "): ", round(modularity(ref), 4),
       " | clusters: ", length(unique(membership(ref))))

  tab <- data.frame(nodo = V(g)$name)
  for (s in SEMILLAS_ESTABILIDAD) {
    set.seed(s)
    tab[[paste0("semilla_", s)]] <- as.integer(membership(cluster_louvain(g)))
  }
  guardar(tab, sprintf("12_%s_louvain_semillas.csv", nombre))

  tryCatch({
    Wn <- W[V(g)$name, V(g)$name, drop = FALSE]
    set.seed(SEMILLA)
    np <- networkPlot(Wn, n = nrow(Wn), cluster = "louvain", type = "auto",
                      remove.isolates = TRUE, verbose = FALSE)
    guardar(np$cluster_res, sprintf("12_%s_clusters_networkPlot.csv", nombre))
  }, error = function(e) logm("  networkPlot no disponible para ", nombre, ": ", conditionMessage(e)))
}

paso("8. Acoplamiento bibliografico (articulos)", {
  clusters_red(biblioNetwork(M, analysis = "coupling", network = "references", sep = ";"),
               "acoplamiento")
})
paso("9. Co-citacion (referencias)", {
  clusters_red(biblioNetwork(M, analysis = "co-citation", network = "references", sep = ";"),
               "cocitacion", min_peso = 2)
})
paso("10. Co-ocurrencia de palabras clave de autor (DE)", {
  NM_DE <- biblioNetwork(M, analysis = "co-occurrences", network = "author_keywords", sep = ";")
  # Misma construccion que el pipeline: al menos 2 co-ocurrencias (comparacion entre redes iguales)
  clusters_red(NM_DE, "coocurrencia_DE", min_peso = 2)
  # Sensibilidad: la red por defecto de bibliometrix (sin umbral; domina la co-ocurrencia unica)
  clusters_red(NM_DE, "coocurrencia_DE_umbral1", min_peso = 1)
})
paso("11. Colaboracion entre paises", {
  clusters_red(biblioNetwork(M, analysis = "collaboration", network = "countries", sep = ";"),
               "colaboracion_paises")
})

# --- Parametros y entorno -----------------------------------------------------
writeLines(c(
  paste0("csv=", normalizePath(csv_path)),
  paste0("csv_md5=", unname(tools::md5sum(csv_path))),
  paste0("semilla=", SEMILLA),
  paste0("semillas_estabilidad=", paste(range(SEMILLAS_ESTABILIDAD), collapse = "-")),
  paste0("k_top=", K_TOP),
  "thematicMap: n=250, minfreq=5, stemming=FALSE, size=0.5",
  "fieldByYear: field=ID y field=DE, min.freq=5, n.items=5",
  "thematicMap: cluster='louvain' (valor por defecto en bibliometrix 5.2.1), sobre las n=250 palabras mas frecuentes",
  "clusters: igraph::cluster_louvain sobre la matriz de biblioNetwork; co-citacion y co-ocurrencia DE con min_peso=2 (como el pipeline); DE con min_peso=1 como sensibilidad",
  "networkPlot: cluster=louvain, remove.isolates=TRUE (adicional, si no falla)",
  "SR reemplazado por KEY del pipeline; AU_CO y AU1_CO recalculados por bibliometrix"
), file.path(out_dir, "parametros.txt"), useBytes = TRUE)
capture.output(sessionInfo(), file = file.path(out_dir, "sessionInfo.txt"))
logm("Listo. Salidas en ", normalizePath(out_dir))
close(log_con)
