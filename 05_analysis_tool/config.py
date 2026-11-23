"""Configuracion central del sistema de analisis bibliometrico y cienciometrico."""

import logging
import os
from pathlib import Path
from autoshared import cargar_entorno

ENTORNO = cargar_entorno(Path(__file__).parent)

# --- Rutas Base ---
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
CSV_PATH = BASE_DIR / "consolidado.csv"

# --- Caches y Estado en data/ ---
ENRICHED_PATH = DATA_DIR / "enriched_articles.json"
OPENALEX_CACHE_DIR = DATA_DIR / "openalex_cache"
OPENALEX_META_CACHE_DIR = DATA_DIR / "openalex_meta_cache"
CROSSREF_CACHE_DIR = DATA_DIR / "crossref_cache"
CROSSREF_EVENTS_CACHE_DIR = DATA_DIR / "crossref_events_cache"
S2_CACHE_DIR = DATA_DIR / "s2_cache"
ALTMETRIC_CACHE_DIR = DATA_DIR / "altmetric_cache"
OPENCITATIONS_CACHE_DIR = DATA_DIR / "opencitations_cache"
UNPAYWALL_CACHE_DIR = DATA_DIR / "unpaywall_cache"
GAP_FILLER_CACHE_DIR = DATA_DIR / "gap_filler_cache"
DISAMBIGUATION_CACHE_DIR = DATA_DIR / "disambiguation_cache"
SCIMAGO_DIR = BASE_DIR / "scimago"
SCIMAGO_CSV_PATH = DATA_DIR / "scimagojr.csv"
KNOWLEDGE_BASE_PATH = DATA_DIR / "knowledge_base.json"          # recursos genericos (cualquier revision)
SINONIMOS_REVISION_PATH = DATA_DIR / "sinonimos_revision.json"  # opcional, propio de cada revision

# --- Salidas en output/ ---
FIGURES_DIR = OUTPUT_DIR / "figures"
NETWORKS_DIR = OUTPUT_DIR / "networks"
REPORTS_DIR = OUTPUT_DIR / "reports"
BIBLIOMETRIX_CSV = OUTPUT_DIR / "bibliometrix_export.csv"
BIBLIOMETRIX_BIB = OUTPUT_DIR / "bibliometrix_export.bib"
PLOTLY_DIR = OUTPUT_DIR / "plotly"
VOSVIEWER_DIR = OUTPUT_DIR / "vosviewer"
SOTA_DIR = OUTPUT_DIR / "sota_analysis"
STRATIFIED_SUBDIR = "by_item_type"
OA_CORPUS_DIR = OUTPUT_DIR / "oa_corpus"
SPECTROSCOPY_DIR = OUTPUT_DIR / "spectroscopy"
IMPACT_EVOLUTION_DIR = OUTPUT_DIR / "impact_evolution"
CITESPACE_DIR = OUTPUT_DIR / "citespace"
LEIDEN_DIR = OUTPUT_DIR / "leiden"
RDM_DIR = OUTPUT_DIR / "rdm"
KNOWLEDGE_MAP_DIR = OUTPUT_DIR / "knowledge_map"
LOGS_DIR = OUTPUT_DIR / "logs"


def ensure_dirs() -> None:
    """Crea los directorios base requeridos si no existen."""
    for d in [
        DATA_DIR,
        OPENALEX_CACHE_DIR,
        OPENALEX_META_CACHE_DIR,
        CROSSREF_CACHE_DIR,
        CROSSREF_EVENTS_CACHE_DIR,
        S2_CACHE_DIR,
        ALTMETRIC_CACHE_DIR,
        OPENCITATIONS_CACHE_DIR,
        UNPAYWALL_CACHE_DIR,
        GAP_FILLER_CACHE_DIR,
        DISAMBIGUATION_CACHE_DIR,
        FIGURES_DIR,
        NETWORKS_DIR,
        REPORTS_DIR,
        PLOTLY_DIR,
        VOSVIEWER_DIR,
        SOTA_DIR,
        OA_CORPUS_DIR,
        SPECTROSCOPY_DIR,
        IMPACT_EVOLUTION_DIR,
        CITESPACE_DIR,
        LEIDEN_DIR,
        RDM_DIR,
        KNOWLEDGE_MAP_DIR,
        LOGS_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


def clean_output() -> int:
    """Elimina todos los archivos dentro de OUTPUT_DIR, preservando directorios."""
    logger = logging.getLogger("bibliometrico.config")
    if not OUTPUT_DIR.exists():
        return 0
    count = 0
    for f in OUTPUT_DIR.rglob("*"):
        if f.is_file():
            f.unlink()
            count += 1
    logger.info(f"Limpiado: output/ ({count} archivos)")
    return count


def clean_dirs() -> int:
    """Elimina archivos dentro de DATA_DIR y OUTPUT_DIR, preservando directorios."""
    logger = logging.getLogger("bibliometrico.config")
    total = 0
    for root_dir in (DATA_DIR, OUTPUT_DIR):
        if not root_dir.exists():
            continue
        count = 0
        for f in root_dir.rglob("*"):
            if f.is_file():
                f.unlink()
                count += 1
        total += count
        logger.info(f"Limpiado: {root_dir.name}/ ({count} archivos)")
    return total


def _is_preserved_data_input(f: Path) -> bool:
    """Preserva archivos de entrada y base de conocimiento en la raiz de data/ (.csv, .xlsx, .xls, .ris, knowledge_base.json)."""
    if f.resolve() in (SCIMAGO_CSV_PATH.resolve(), KNOWLEDGE_BASE_PATH.resolve(), SINONIMOS_REVISION_PATH.resolve()):
        return True
    if f.parent.resolve() == DATA_DIR.resolve() and f.suffix.lower() in (".csv", ".xlsx", ".xls", ".ris"):
        return True
    return False


def clean_review() -> int:
    """Reset ENTRE REVISIONES: borra todo el estado derivado en data/ y output/
    preservando archivos de entrada (.csv, .xlsx, .ris) en data/ y consolidado.csv en la raíz.
    """
    logger = logging.getLogger("bibliometrico.config")
    total = 0
    for root_dir in (DATA_DIR, OUTPUT_DIR):
        if not root_dir.exists():
            continue
        count = 0
        for f in root_dir.rglob("*"):
            if not f.is_file():
                continue
            if root_dir == DATA_DIR and _is_preserved_data_input(f):
                continue
            f.unlink()
            count += 1
        total += count
        logger.info(
            f"Reset revision: {root_dir.name}/ ({count} archivos eliminados)"
        )
    logger.info(
        "Reset revision completado. Preservados: insumos (.csv, .xlsx, .ris) en data/ y consolidado.csv."
    )
    _warn_preserved_review_inputs(logger)
    return total


def _warn_preserved_review_inputs(logger: logging.Logger) -> None:
    """Avisa de los insumos conservados que dependen de la revision y podrian ser de la anterior."""
    if not DATA_DIR.exists():
        return
    preserved = sorted(
        f for f in DATA_DIR.iterdir()
        if f.is_file() and _is_preserved_data_input(f)
        and f.resolve() not in (SCIMAGO_CSV_PATH.resolve(), KNOWLEDGE_BASE_PATH.resolve())
    )
    if not preserved:
        return
    logger.warning(
        "Se conservaron %d insumos de data/ que dependen de la revision y pueden ser de la ANTERIOR: %s. "
        "Reemplazalos por los de la revision nueva o borralos antes de la fase 0 "
        "(sinonimos_revision.json: vaciarlo o revisarlo; knowledge_base.json es generico y se conserva).",
        len(preserved), ", ".join(f.name for f in preserved),
    )


def count_review_state_files() -> int:
    """Cuenta los archivos que clean_review() eliminaria (para confirmar)."""
    n = 0
    for root_dir in (DATA_DIR, OUTPUT_DIR):
        if not root_dir.exists():
            continue
        for f in root_dir.rglob("*"):
            if not f.is_file():
                continue
            if root_dir == DATA_DIR and _is_preserved_data_input(f):
                continue
            n += 1
    return n


# --- API Keys & Credentials (Bibliometric Sources Only) ---
OPENALEX_EMAIL = os.getenv("OPENALEX_EMAIL", "your.email@example.org") or "your.email@example.org"
OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY", "")
ALTMETRIC_API_KEY = os.getenv("ALTMETRIC_API_KEY", "")
OPENCITATIONS_TOKEN = os.getenv("OPENCITATIONS_TOKEN", "")
UNPAYWALL_EMAIL = os.getenv("UNPAYWALL_EMAIL", OPENALEX_EMAIL) or OPENALEX_EMAIL

# --- CSV Parsing ---
CSV_SEPARATOR = ";"
CSV_ENCODING = "utf-8-sig"

# --- OpenAlex ---
OPENALEX_BASE_URL = "https://api.openalex.org"
OPENALEX_RATE_LIMIT = 10  # requests per second (polite pool)
OPENALEX_FUZZY_THRESHOLD = 85  # minimum rapidfuzz score for title match

# --- Crossref ---
CROSSREF_BASE_URL = "https://api.crossref.org"
CROSSREF_EMAIL = os.getenv("CROSSREF_EMAIL", OPENALEX_EMAIL)
CROSSREF_PLUS_API_TOKEN = os.getenv("CROSSREF_PLUS_API_TOKEN", "")
CROSSREF_RATE_LIMIT = 10
CROSSREF_FUZZY_THRESHOLD = 85

# --- Semantic Scholar ---
S2_BASE_URL = "https://api.semanticscholar.org/graph/v1"
S2_API_KEY = os.getenv("S2_API_KEY", "")
S2_RATE_LIMIT = 0.85  # < 1 req/s estricto acumulativo en todos los endpoints (intervalo >= 1.18s post-respuesta)
S2_FUZZY_THRESHOLD = 85

# --- Altmetric ---
ALTMETRIC_BASE_URL = "https://api.altmetric.com/v1"
ALTMETRIC_RATE_LIMIT = 1  # 1 req/s (free tier)

# --- OpenCitations ---
OPENCITATIONS_BASE_URL = "https://opencitations.net/index/api/v2"
OPENCITATIONS_META_URL = "https://api.opencitations.net/meta/v1"
OPENCITATIONS_RATE_LIMIT = 3  # 3 req/s

# --- Unpaywall ---
UNPAYWALL_BASE_URL = "https://api.unpaywall.org/v2"
UNPAYWALL_RATE_LIMIT = 10  # 10 req/s

# --- Metadata Gap-Filler (Phase 1k) ---
EUROPE_PMC_BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest"
EUROPE_PMC_RATE_LIMIT = 3  # 3 req/s (conservative)
DOAJ_BASE_URL = "https://doaj.org/api"
DOAJ_RATE_LIMIT = 2  # 2 req/s
NLM_CATALOG_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NLM_RATE_LIMIT = 3  # 3 req/s (NCBI without API key)
ISSN_PORTAL_BASE_URL = "https://portal.issn.org/resource/ISSN"
ISSN_RATE_LIMIT = 1  # 1 req/s (conservative)
GAP_FILLER_FUZZY_THRESHOLD = 85

# --- Cue Words for Future Research & Limitations (Algorithmic Gap Detection) ---
FUTURE_RESEARCH_CUES = [
    "future research", "future work", "future studies", "further research",
    "further investigation", "further studies", "remains to be",
    "should be explored", "could be investigated", "warrants further",
    "promising direction", "open question", "yet to be determined",
    "needs further", "deserves further", "future direction",
]

LIMITATION_CUES = [
    "limitation", "limitations", "limited by", "shortcoming",
    "drawback", "constraint", "restricted to", "confined to",
    "weakness", "caveat", "however, this study", "despite these",
    "small sample", "limited sample", "generalizability",
    "should be interpreted with caution",
]

# --- BERT / SciBERT ---
BERT_MODEL_NAME = "allenai/scibert_scivocab_uncased"
BERT_BATCH_SIZE = 32

# --- WordCloud ---
WORDCLOUD_MAX_WORDS = 200

# --- Interactive Charts ---
CHART_TOP_N = 15

# --- Bibliometrix Export Column Mapping ---
BIBLIOMETRIX_COLUMNS = {
    "AU": "authors",          # Authors (semicolon separated)
    "TI": "title",            # Title
    "SO": "publication_title", # Source
    "JI": "journal_abbrev",   # Journal Abbreviation
    "DT": "item_type",        # Document Type
    "DE": "manual_tags",      # Author Keywords
    "ID": "automatic_tags",   # Keywords Plus
    "AB": "abstract",         # Abstract
    "C1": "affiliations",     # Author Affiliations
    "RP": "corresponding",    # Corresponding Author
    "CR": "cited_references", # Cited References
    "TC": "cited_by_count",   # Times Cited
    "PY": "publication_year", # Year
    "VL": "volume",           # Volume
    "IS": "issue",            # Issue
    "PP": "pages",            # Pages
    "DI": "doi",              # DOI
    "SN": "issn",             # ISSN
    "LA": "language",         # Language
    "OA": "open_access",      # Open Access status
    "URL": "url",             # URL
}
