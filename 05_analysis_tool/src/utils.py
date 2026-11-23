"""Utilidades compartidas: rate limiter, cache en disco, logging, helpers."""
from __future__ import annotations

import json
import hashlib
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from filelock import FileLock

logger = logging.getLogger("bibliometrico")


_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str = "INFO") -> None:
    """Configura logging global (solo consola)."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=_LOG_FORMAT,
        datefmt=_LOG_DATEFMT,
    )
    for noisy_logger in ("kaleido", "choreographer", "urllib3"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def add_file_logging(log_file: Path, level: str = "INFO") -> Optional[Path]:
    """Adjunta un FileHandler al root logger (procedencia: log por corrida).

    Se llama DESPUES de setup_logging y DESPUES de las ramas de
    limpieza/reset de main.py: si se abriera el archivo antes, el unlink
    de clean_dirs()/clean_review() fallaria en Windows (archivo en uso).
    Idempotente: no duplica handlers para el mismo destino.

    Devuelve la ruta efectiva, o None si no se pudo crear (el pipeline no
    debe abortar por un fallo de logging).
    """
    try:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        root = logging.getLogger()
        target = str(log_file.resolve())
        for h in root.handlers:
            if isinstance(h, logging.FileHandler) and \
                    getattr(h, "_bibliometrico_target", None) == target:
                return log_file
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(getattr(logging, level.upper(), logging.INFO))
        fh.setFormatter(logging.Formatter(_LOG_FORMAT, _LOG_DATEFMT))
        fh._bibliometrico_target = target  # type: ignore[attr-defined]
        root.addHandler(fh)
        return log_file
    except OSError as e:
        logger.warning("No se pudo crear el log de corrida %s: %s", log_file, e)
        return None


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, calls_per_second: float):
        self.min_interval = 1.0 / calls_per_second
        self._last_call = 0.0

    def wait(self) -> None:
        now = time.time()
        elapsed = now - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call = time.time()

    def mark_done(self) -> None:
        """Actualiza la marca de tiempo al finalizar una respuesta HTTP para evitar ráfagas por latencia."""
        self._last_call = time.time()


class DiskCache:
    """Cache JSON en disco, keyed por string, con cache en memoria para lecturas rapidas."""

    _MEM_CACHE: dict[tuple[str, str], Any] = {}

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self._dir_key = str(cache_dir.resolve())
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key_to_path(self, key: str) -> Path:
        safe = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{safe}.json"

    def _lock_path(self, path: Path) -> Path:
        return path.with_suffix(path.suffix + ".lock")

    def get(self, key: str) -> Optional[Any]:
        mem_key = (self._dir_key, key)
        if mem_key in self._MEM_CACHE:
            return self._MEM_CACHE[mem_key]
        path = self._key_to_path(key)
        if path.exists():
            # set() usa os.replace() atomico; lectura directa evita crear miles de .lock en NTFS
            try:
                val = json.loads(path.read_text(encoding="utf-8"))
                self._MEM_CACHE[mem_key] = val
                return val
            except (json.JSONDecodeError, OSError):
                lock = FileLock(str(self._lock_path(path)))
                try:
                    with lock.acquire(timeout=5):
                        val = json.loads(path.read_text(encoding="utf-8"))
                        self._MEM_CACHE[mem_key] = val
                        return val
                except Exception:
                    return None
        return None

    def set(self, key: str, value: Any) -> None:
        self._MEM_CACHE[(self._dir_key, key)] = value
        path = self._key_to_path(key)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        lock = FileLock(str(self._lock_path(path)))
        data = json.dumps(value, ensure_ascii=False, indent=2)
        try:
            with lock.acquire(timeout=10):
                tmp_path.write_text(data, encoding="utf-8")
                os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    def has(self, key: str) -> bool:
        if (self._dir_key, key) in self._MEM_CACHE:
            return True
        return self._key_to_path(key).exists()


def create_session(retries: int = 3, backoff: float = 0.5) -> requests.Session:
    """Crea una sesion requests con reintentos y backoff exponencial."""
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def safe_request(
    url: str,
    session: Optional[requests.Session] = None,
    rate_limiter: Optional[RateLimiter] = None,
    params: Optional[dict] = None,
    timeout: int = 30,
    headers: Optional[dict] = None,
    quiet_statuses: tuple = (),
) -> Optional[dict]:
    """GET request seguro con rate limiting y manejo de errores.

    quiet_statuses: codigos HTTP esperados/benignos (p.ej. ``(404,)`` cuando
    un recurso "no encontrado" es normal). Se registran a DEBUG en vez de
    WARNING para no inundar la consola. El default ``()`` no cambia el
    comportamiento de los demas llamadores.
    """
    if rate_limiter:
        rate_limiter.wait()
    sess = session or create_session()
    try:
        resp = sess.get(url, params=params, timeout=timeout, headers=headers)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        if status in quiet_statuses:
            logger.debug(f"Request {status} (esperado) for {url}")
        else:
            logger.warning(f"Request failed for {url}: {e}")
        return None
    except json.JSONDecodeError:
        logger.warning(f"Invalid JSON from {url}")
        return None
    finally:
        if rate_limiter:
            rate_limiter.mark_done()


def normalize_doi(doi: str) -> str:
    """Normaliza un DOI eliminando prefijos URL."""
    if not doi:
        return ""
    doi = doi.strip()
    for prefix in ["https://doi.org/", "http://doi.org/", "http://dx.doi.org/",
                    "https://dx.doi.org/", "doi:"]:
        if doi.lower().startswith(prefix.lower()):
            doi = doi[len(prefix):]
    return doi.strip().lower()


def normalize_author_name(name: str) -> str:
    """Normaliza nombre de autor: 'Apellido, I.' format."""
    name = name.strip()
    if not name:
        return ""
    # Remove extra spaces
    name = re.sub(r"\s+", " ", name)
    return name


def parse_author_string(author_str: str) -> list[str]:
    """Parsea string de autores del formato Zotero CSV.
    Formato: 'Apellido1, N1.; Apellido2, N2.'
    """
    if not author_str:
        return []
    # Split by '; ' (semicolon + space) which separates authors in Zotero
    authors = []
    for author in author_str.split("; "):
        author = author.strip().strip('"')
        if author:
            authors.append(normalize_author_name(author))
    return authors


def extract_year(date_str: str) -> Optional[int]:
    """Extrae año de un string de fecha."""
    if not date_str:
        return None
    match = re.search(r"\b(19|20)\d{2}\b", str(date_str))
    return int(match.group()) if match else None


def save_json(data: Any, path: Path) -> None:
    """Guarda datos como JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info(f"Saved: {path}")


def load_json(path: Path) -> Optional[Any]:
    """Carga datos desde JSON."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Error loading {path}: {e}")
        return None


# ======================================================================
# OpenAlex label resolver (centralizado)
# ======================================================================

_label_cache: dict[str, str] = {}


def resolve_openalex_label(
    node_id: str,
    articles: Optional[list[dict]] = None,
    cache: Optional["DiskCache"] = None,
) -> str:
    """Resolve an OpenAlex work ID or opaque reference label to 'Author (Year) — Title'.

    Resolution chain:
    1. Match OpenAlex ID against articles in corpus (openalex_id field)
    2. Extract DOI from composite labels and match against articles
    3. Match DOI against referenced_works_detailed in articles
    4. Read from OpenAlex disk cache
    5. Return cleaned-up fallback (DOI or OA:Wxxxx, never a raw URL)
    """
    if not node_id:
        return node_id

    # Extract OpenAlex ID and DOI from composite labels
    oa_id = ""
    doi = ""
    if "openalex.org/" in node_id:
        oa_id = node_id.strip().rstrip("/")
    elif "openalex:" in node_id.lower():
        oa_match = re.search(r'openalex[:/]+(W\d+)', node_id, re.IGNORECASE)
        if oa_match:
            oa_id = f"https://openalex.org/{oa_match.group(1)}"
        doi_match = re.search(r'(10\.\d{4,}/[^\s]+)', node_id)
        if doi_match:
            doi = doi_match.group(1).rstrip("}")
    elif node_id.startswith("10."):
        doi = node_id.strip()

    # Try matching against corpus articles
    if articles:
        for a in articles:
            a_oa = a.get("openalex_id", "") or a.get("openalex", {}).get("id", "")
            a_doi = a.get("doi", "")

            if oa_id and a_oa and oa_id.lower().rstrip("/") == a_oa.lower().rstrip("/"):
                first_author = (a.get("authors", ["Unknown"])[0] if isinstance(a.get("authors"), list)
                                else str(a.get("authors", "Unknown")).split(",")[0].split(" and ")[0])
                year = a.get("year", a.get("publication_year", "n.d."))
                title = (a.get("title", "") or "")[:80]
                return f"{first_author} ({year}) — {title}"

            if doi and a_doi and doi.lower() == a_doi.lower():
                first_author = (a.get("authors", ["Unknown"])[0] if isinstance(a.get("authors"), list)
                                else str(a.get("authors", "Unknown")).split(",")[0].split(" and ")[0])
                year = a.get("year", a.get("publication_year", "n.d."))
                title = (a.get("title", "") or "")[:80]
                return f"{first_author} ({year}) — {title}"

    # Try referenced_works_detailed (Crossref refs)
    if articles and doi:
        for a in articles:
            for ref in a.get("referenced_works_detailed", []):
                if ref.get("doi", "").lower() == doi.lower():
                    author = ref.get("author", "Unknown")
                    if " and " in str(author):
                        author = str(author).split(" and ")[0]
                    year = ref.get("year", "n.d.")
                    title = ref.get("article_title", "") or ref.get("title", "")
                    if author or title:
                        return f"{author} ({year}) — {title[:80]}"

    # Try OpenAlex disk cache as last resort
    try:
        if cache is None:
            from config import OPENALEX_CACHE_DIR
            cache = DiskCache(OPENALEX_CACHE_DIR)
        cache_key = oa_id.split("/")[-1] if oa_id else ""
        if cache_key:
            for prefix in ("work:", ""):
                cached = cache.get(f"{prefix}{cache_key}")
                if cached and isinstance(cached, dict):
                    author = (cached.get("authorships", [{}])[0]
                              .get("author", {}).get("display_name", "Unknown")
                              if cached.get("authorships") else "Unknown")
                    year = cached.get("publication_year", "n.d.")
                    title = (cached.get("title", "") or "")[:80]
                    if title:
                        return f"{author} ({year}) — {title}"
    except Exception:
        pass

    # Fallback: clean up the label (remove URLs, keep DOI if available)
    if doi:
        return f"[DOI: {doi}]"
    cleaned = node_id.replace("https://openalex.org/", "OA:")
    if "omid:" in cleaned:
        doi_match = re.search(r'(10\.\d{4,}/[^\s]+)', cleaned)
        if doi_match:
            return f"[DOI: {doi_match.group(1)}]"
    return cleaned


def resolve_openalex_label_cached(
    node_id: str,
    articles: Optional[list[dict]] = None,
    cache: Optional["DiskCache"] = None,
) -> str:
    """Wrapper con cache en memoria para resolve_openalex_label."""
    if node_id in _label_cache:
        return _label_cache[node_id]
    resolved = resolve_openalex_label(node_id, articles, cache)
    _label_cache[node_id] = resolved
    return resolved


# ======================================================================
# Normalizadores Canonicos (Idioma, Revistas, Open Access, Topics)
# ======================================================================

_LANGUAGE_MAP = {
    "en": "English", "eng": "English", "english": "English", "ingles": "English", "inglés": "English",
    "es": "Spanish", "spa": "Spanish", "spanish": "Spanish", "espanol": "Spanish", "español": "Spanish", "castellano": "Spanish",
    "pt": "Portuguese", "por": "Portuguese", "portuguese": "Portuguese", "portugues": "Portuguese", "português": "Portuguese",
    "fr": "French", "fre": "French", "fra": "French", "french": "French", "francais": "French", "français": "French",
    "de": "German", "ger": "German", "deu": "German", "german": "German", "deutsch": "German",
    "it": "Italian", "ita": "Italian", "italian": "Italian", "italiano": "Italian",
    "zh": "Chinese", "chi": "Chinese", "zho": "Chinese", "chinese": "Chinese",
    "ru": "Russian", "rus": "Russian", "russian": "Russian",
    "tr": "Turkish", "tur": "Turkish", "turkish": "Turkish",
    "id": "Indonesian", "ind": "Indonesian", "indonesian": "Indonesian",
    "ja": "Japanese", "jpn": "Japanese", "japanese": "Japanese",
    "ko": "Korean", "kor": "Korean", "korean": "Korean",
    "ar": "Arabic", "ara": "Arabic", "arabic": "Arabic",
    "nl": "Dutch", "dut": "Dutch", "nld": "Dutch", "dutch": "Dutch",
    "pl": "Polish", "pol": "Polish", "polish": "Polish",
    "uk": "Ukrainian", "ukr": "Ukrainian", "ukrainian": "Ukrainian",
    "ca": "Catalan", "cat": "Catalan", "catalan": "Catalan",
    "ms": "Malay", "msa": "Malay", "may": "Malay", "malay": "Malay",
}


def normalize_language(raw: str) -> str:
    """Normaliza codigos ISO (en, eng, es, pt...) y nombres de idioma a nombre canonico en ingles."""
    if not raw:
        return ""
    s = str(raw).strip()
    if not s or s.lower() in ("nan", "none", "null", "und", "unknown"):
        return ""
    # Si viene multiple ("English; Spanish" o "en-US"), tomar el primero/base
    first = re.split(r"[;,/|]", s)[0].strip()
    base = first.split("-")[0].split("_")[0].strip().lower()
    if base in _LANGUAGE_MAP:
        return _LANGUAGE_MAP[base]
    low = first.lower()
    if low in _LANGUAGE_MAP:
        return _LANGUAGE_MAP[low]
    return first.capitalize()


def normalize_issn(raw: str) -> str:
    """Extrae el primer ISSN normalizado (XXXX-XXXX) de una cadena."""
    if not raw:
        return ""
    m = re.search(r"\b(\d{4})-?(\d{3}[\dXx])\b", str(raw))
    if m:
        return f"{m.group(1)}-{m.group(2).upper()}"
    return ""


def normalize_journal_name(name: str) -> str:
    """Limpia variaciones tipograficas comunes en nombres de revistas."""
    if not name:
        return ""
    s = str(name).strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return ""
    # Si tiene nombre bilingue separado por ' / ', conservar la primera parte si tiene >= 10 chars
    if " / " in s:
        first_part = s.split(" / ")[0].strip()
        if len(first_part) >= 10:
            s = first_part
    # Unificar ' - ' como ': ' cuando separa subtitulo o acronimo (ej. 'ZDM - Mathematics Education')
    s = re.sub(r"\s+[-–—]\s+", ": ", s)
    s = re.sub(r"\s+", " ", s).rstrip(".")
    return s


def _journal_match_key(name: str) -> str:
    """Clave alfanumerica insensible a puntuacion/mayusculas para agrupar la misma revista."""
    s = normalize_journal_name(name).lower()
    s = re.sub(r"\band\b", "&", s)
    s = re.sub(r"^the\s+", "", s)
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def canonicalize_corpus_journals(articles: list[dict]) -> int:
    """Unifica nombres de revista equivalentes en el corpus usando ISSN y clave normalizada.

    Returns:
        Numero de articulos cuyo publication_title fue normalizado/unificado.
    """
    from collections import Counter

    # Paso 1: normalizacion sintactica basica
    changed = 0
    for a in articles:
        raw = a.get("publication_title", "") or ""
        cleaned = normalize_journal_name(raw)
        if cleaned != raw:
            a["publication_title"] = cleaned
            changed += 1

    # Paso 2: agrupar por clave normalizada -> elegir la variante mas frecuente
    key_counts: dict[str, Counter] = {}
    for a in articles:
        j = a.get("publication_title", "") or ""
        if not j:
            continue
        k = _journal_match_key(j)
        if k:
            key_counts.setdefault(k, Counter())[j] += 1

    key_to_canonical = {
        k: cnt.most_common(1)[0][0]
        for k, cnt in key_counts.items()
        if len(cnt) >= 1
    }

    # Paso 3: agrupar por ISSN compartido -> si dos claves tienen el mismo ISSN, unificar al mas frecuente
    issn_counts: dict[str, Counter] = {}
    for a in articles:
        issn = normalize_issn(a.get("issn", ""))
        j = a.get("publication_title", "") or ""
        if issn and j:
            k = _journal_match_key(j)
            canon = key_to_canonical.get(k, j)
            issn_counts.setdefault(issn, Counter())[canon] += 1

    issn_to_canonical = {
        issn: cnt.most_common(1)[0][0]
        for issn, cnt in issn_counts.items()
        if len(cnt) > 1
    }

    for a in articles:
        j = a.get("publication_title", "") or ""
        if not j:
            continue
        k = _journal_match_key(j)
        new_j = key_to_canonical.get(k, j)
        issn = normalize_issn(a.get("issn", ""))
        if issn and issn in issn_to_canonical:
            new_j = issn_to_canonical[issn]
        if new_j != j:
            a["publication_title"] = new_j
            changed += 1

    return changed


def get_canonical_oa_status(article: dict) -> str:
    """Criterio unico para el color de Acceso Abierto (preserva diamond frente a gold)."""
    oa = article.get("open_access") or {}
    if not isinstance(oa, dict):
        oa = {}
    oa_ext = article.get("openalex") or {}
    upw = article.get("unpaywall") or {}

    raw_status = (oa.get("status") or oa_ext.get("oa_status") or "").strip().lower()
    raw_color = (oa.get("oa_color") or upw.get("oa_status") or "").strip().lower()

    # 1. Si OpenAlex, la fuente o SCImago declaro 'diamond', prevalece sobre el 'gold' de Unpaywall
    if raw_status == "diamond" or raw_color == "diamond":
        return "diamond"
    scimago = article.get("scimago") or {}
    if isinstance(scimago, dict) and str(scimago.get("open_access_diamond", "")).strip().lower() == "yes":
        if raw_status in ("gold", "diamond") or raw_color in ("gold", "diamond") or oa.get("is_oa") or oa_ext.get("is_oa"):
            return "diamond"

    # 2. Si es revista OA en DOAJ con APC listado = 0 y APC pagado = 0 -> diamond
    if (oa.get("journal_is_in_doaj") or upw.get("journal_is_in_doaj")) and (raw_status == "gold" or raw_color == "gold"):
        apc_list = oa_ext.get("apc_list") or {}
        apc_paid = oa_ext.get("apc_paid") or {}
        if apc_list.get("value", None) == 0 and not apc_paid.get("value_usd"):
            return "diamond"

    chosen = raw_status or raw_color
    if chosen in ("diamond", "gold", "green", "hybrid", "bronze", "closed"):
        return chosen
    if oa.get("is_oa") or oa_ext.get("is_oa") or upw.get("is_oa"):
        return chosen or "gold"
    return "closed" if chosen == "closed" else (chosen or "")


_NOISY_CONCEPT_NAMES = {
    "paleontology", "geology", "archaeology", "pathology",
    "internal medicine", "endocrinology", "surgery", "botany",
}
_NOISY_PAREN_RE = re.compile(
    r"\((?:archaeology|biology|physics|punctuation|linguistics|chemistry|"
    r"geology|botany|zoology|medicine|anatomy|astronomy|music|law|theology|"
    r"computing|philosophy|abstract data type|signal processing|topology)\)",
    re.IGNORECASE,
)


def is_noisy_legacy_concept(name: str) -> bool:
    """Detecta conceptos heredados ruidosos de OpenAlex MAG (homonimos con desambiguacion entre parentesis)."""
    if not name:
        return True
    s = name.strip()
    if _NOISY_PAREN_RE.search(s):
        return True
    if s.lower() in _NOISY_CONCEPT_NAMES:
        return True
    return False


# ======================================================================
# Base de Conocimiento Externa y Normalizacion Agnostica (data/knowledge_base.json)
# ======================================================================

_KB_CACHE: Optional[dict] = None
_UNICODE_HYPHENS_RE = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]")
_UNICODE_QUOTES_RE = re.compile(r"[\u2018\u2019\u201b\u02bc]")
_TRAILING_ACRONYM_PAREN_RE = re.compile(r"^(.{6,}?)\s*\([a-z0-9-]{2,8}\)$")
_INVARIANT_SINGULAR_SUFFIXES = (
    "ss", "us", "is", "os", "ics", "ness", "sis", "ais",
    "series", "species", "data", "media", "criteria", "analysis",
)


def load_knowledge_base(force_reload: bool = False) -> dict:
    """Carga la base de conocimiento desde data/knowledge_base.json de forma agnostica.
    Si el archivo no existe, devuelve estructuras vacias para que el nucleo opere
    unicamente con reglas morfologicas/estadisticas del corpus.
    """
    global _KB_CACHE
    if _KB_CACHE is not None and not force_reload:
        return _KB_CACHE

    from config import KNOWLEDGE_BASE_PATH, SINONIMOS_REVISION_PATH
    raw_kb: dict = {}
    if KNOWLEDGE_BASE_PATH.exists():
        try:
            raw_kb = json.loads(KNOWLEDGE_BASE_PATH.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"No se pudo cargar {KNOWLEDGE_BASE_PATH}: {e}")

    # knowledge_base.json debe ser generico. Si trae contenido de dominio, se avisa (y se usa,
    # para no cambiar resultados en silencio), indicando donde debe ir.
    _dominio = [k for k in ("keyword_synonyms", "prefix_canonical_rules", "institution_synonyms")
                if raw_kb.get(k)]
    if _dominio:
        logger.warning(
            f"{KNOWLEDGE_BASE_PATH.name} contiene contenido de dominio ({', '.join(_dominio)}). "
            f"Debe ser generico: mover esas entradas a {SINONIMOS_REVISION_PATH.name}."
        )

    # Sinonimos propios de la revision (opcional; vacio = normalizacion aprendida solo del corpus)
    rev: dict = {}
    if SINONIMOS_REVISION_PATH.exists():
        try:
            rev = json.loads(SINONIMOS_REVISION_PATH.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"No se pudo cargar {SINONIMOS_REVISION_PATH}: {e}")
    raw_kb = dict(raw_kb)
    raw_kb["keyword_synonyms"] = {**(raw_kb.get("keyword_synonyms") or {}), **(rev.get("keyword_synonyms") or {})}
    raw_kb["institution_synonyms"] = {**(raw_kb.get("institution_synonyms") or {}), **(rev.get("institution_synonyms") or {})}
    raw_kb["prefix_canonical_rules"] = list(raw_kb.get("prefix_canonical_rules") or []) + list(rev.get("prefix_canonical_rules") or [])
    raw_kb["keyword_stopwords"] = list(raw_kb.get("keyword_stopwords") or []) + list(rev.get("keyword_stopwords") or [])
    n_rev = sum(len(rev.get(k) or []) for k in ("keyword_synonyms", "institution_synonyms", "prefix_canonical_rules",
                                                 "keyword_stopwords", "no_unificar", "metodos_incluidos",
                                                 "metodos_excluidos", "terminos_de_interes"))
    _norm_rev = lambda x: _UNICODE_HYPHENS_RE.sub("-", str(x).strip().lower())
    no_unificar = {_norm_rev(x) for x in (rev.get("no_unificar") or []) if str(x).strip()}
    terminos_interes = [_norm_rev(x) for x in (rev.get("terminos_de_interes") or []) if str(x).strip()]
    if n_rev:
        logger.info(f"Reglas propias de la revision cargadas desde {SINONIMOS_REVISION_PATH.name}: {n_rev}")

    mesh_subs = {str(x).strip().lower() for x in raw_kb.get("mesh_subheadings", []) if str(x).strip()}
    stopwords = {
        _UNICODE_HYPHENS_RE.sub("-", str(x).strip().lower())
        for x in raw_kb.get("keyword_stopwords", [])
        if str(x).strip()
    }
    kw_syns = {
        _UNICODE_HYPHENS_RE.sub("-", str(k).strip().lower()): _UNICODE_HYPHENS_RE.sub("-", str(v).strip().lower())
        for k, v in (raw_kb.get("keyword_synonyms", {}) or {}).items()
        if str(k).strip() and str(v).strip()
    }
    prefix_rules = []
    for rule in raw_kb.get("prefix_canonical_rules", []) or []:
        if isinstance(rule, dict) and rule.get("prefix") and rule.get("canonical"):
            prefix_rules.append({
                "prefix": str(rule["prefix"]).strip().lower(),
                "canonical": str(rule["canonical"]).strip().lower(),
            })
    inst_syns = {
        _UNICODE_QUOTES_RE.sub("'", _UNICODE_HYPHENS_RE.sub("-", str(k).strip().lower())): str(v).strip()
        for k, v in (raw_kb.get("institution_synonyms", {}) or {}).items()
        if str(k).strip() and str(v).strip()
    }
    journal_acronyms = {
        str(x).strip().upper(): str(x).strip()
        for x in (raw_kb.get("journal_acronyms", []) or [])
        if str(x).strip()
    }
    country_iso = {
        str(k).strip().upper(): str(v).strip()
        for k, v in (raw_kb.get("country_iso_names", {}) or {}).items()
        if str(k).strip() and str(v).strip()
    }

    _KB_CACHE = {
        "mesh_subheadings": mesh_subs,
        "keyword_stopwords": stopwords,
        "keyword_synonyms": kw_syns,
        "prefix_canonical_rules": prefix_rules,
        "institution_synonyms": inst_syns,
        "journal_acronyms": journal_acronyms,
        "country_iso_names": country_iso,
        "n_reglas_revision": n_rev,
        # Decisiones del equipo (planilla de curaduria -> sinonimos_revision.json)
        "no_unificar": no_unificar,
        "terminos_de_interes": terminos_interes,
    }
    return _KB_CACHE


def singularize_term_agnostic(term: str) -> str:
    """Singularizador morfologico general (no dependiente de dominio) sobre el nucleo final del termino."""
    if not term or len(term) <= 3:
        return term
    words = term.split(" ")
    last = words[-1]
    if len(last) <= 3 or any(last.endswith(sfx) for sfx in _INVARIANT_SINGULAR_SUFFIXES):
        return term
    if last.endswith("ies") and len(last) > 4 and last[-4] not in "aeiou":
        words[-1] = last[:-3] + "y"
        return " ".join(words)
    if last.endswith(("yses", "eses")) and len(last) > 5:  # analyses -> analysis, hypotheses -> hypothesis
        words[-1] = last[:-2] + "is"
        return " ".join(words)
    if last.endswith(("sses", "xes", "zes", "ches", "shes")) and len(last) > 4:   # classes -> class
        words[-1] = last[:-2]
        return " ".join(words)
    if last.endswith("ses") and len(last) > 4:            # courses -> course, purposes -> purpose
        words[-1] = last[:-1]
        return " ".join(words)
    if last.endswith("s") and not last.endswith(("ss", "us", "is", "os")):
        words[-1] = last[:-1]
        return " ".join(words)
    return term


def clean_keyword_syntax(raw: str) -> str:
    """Limpia artefactos sintacticos de indexacion (marcadores '*' MeSH, calificadores '/sub', guiones Unicode)."""
    if not raw or not isinstance(raw, str):
        return ""
    kb = load_knowledge_base()
    s = _UNICODE_QUOTES_RE.sub("'", _UNICODE_HYPHENS_RE.sub("-", raw.strip()))
    # Quitar asteriscos de MeSH Major Topic (al inicio o despues de '/')
    s = s.lstrip("*+ ").replace("/*", "/")
    s = re.sub(r"\s+", " ", s).strip().lower()
    if not s:
        return ""

    # Si ya coincide exactamente con un sinonimo en KB antes de recortar '/', devolverlo
    if s in kb["keyword_synonyms"]:
        return kb["keyword_synonyms"][s]

    # Recortar calificadores MeSH ('termino/subencabezado' -> 'termino', 'nucleo, adjetivo/subencabezado' -> 'nucleo, adjetivo')
    if "/" in s:
        parts = [p.strip().lstrip("*") for p in s.split("/") if p.strip()]
        if len(parts) >= 2:
            sub = parts[-1]
            if sub in kb["mesh_subheadings"] or len(parts[0]) >= 3:
                s = parts[0]

    # Quitar acronimo redundante al final entre parentesis si la frase base es suficientemente larga
    if s not in kb["keyword_synonyms"]:
        m_acr = _TRAILING_ACRONYM_PAREN_RE.match(s)
        if m_acr:
            s = m_acr.group(1).strip()

    return s


def canonicalize_keyword(raw: str, corpus_map: Optional[dict[str, str]] = None) -> str:
    """Normaliza un termino aplicando limpieza sintactica, base de conocimiento externa y mapa de corpus."""
    s = clean_keyword_syntax(raw)
    if not s:
        return ""
    kb = load_knowledge_base()
    syns = kb["keyword_synonyms"]

    if s in syns:
        s = syns[s]
    for rule in kb["prefix_canonical_rules"]:
        if s.startswith(rule["prefix"]):
            s = rule["canonical"]
            break

    if corpus_map and s in corpus_map:
        s = corpus_map[s]

    # Verificar singularizacion morfologica contra sinonimos
    sing = singularize_term_agnostic(s)
    if sing in syns:
        s = syns[sing]
    elif corpus_map and sing in corpus_map:
        s = corpus_map[sing]

    if s in syns:
        s = syns[s]
    return s


def is_keyword_stopword(term: str) -> bool:
    """Evalua si un termino pertenece a la lista de stopwords bibliograficos/epidemiologicos de la KB."""
    if not term:
        return True
    kb = load_knowledge_base()
    s = clean_keyword_syntax(term)
    if not s:
        return True
    if s in kb["keyword_stopwords"]:
        return True
    sing = singularize_term_agnostic(s)
    if sing in kb["keyword_stopwords"]:
        return True
    return False


def _iter_raw_article_keywords(article: dict) -> list[str]:
    """Palabras clave que forman el vocabulario comun: SOLO las de autor (campo 'author_keywords').

    Decision de metodo (2026-11-22, opcion A): las palabras automaticas (Index Keywords, Keywords Plus,
    OpenAlex, MeSH) las asigna una maquina o un indizador y metian ruido ('generative grammar',
    'active learning (machine learning)'). Es la convencion de bibliometrix para DE. Un articulo sin
    palabras clave de autor no aporta a los analisis de palabras clave.
    Si el checkpoint no tiene 'author_keywords' (anterior al cambio), se usa 'manual_tags' y se avisa.
    """
    if "author_keywords" in article:
        return [kw for kw in (article.get("author_keywords") or []) if isinstance(kw, str) and kw.strip()]
    global _AVISO_SIN_AUTHOR_KW
    if not _AVISO_SIN_AUTHOR_KW:
        logger.warning("Articulos sin campo 'author_keywords': se usa 'manual_tags' (puede mezclar palabras "
                       "no de autor). Cargue el checkpoint con main.py para completarlo desde consolidado.csv.")
        _AVISO_SIN_AUTHOR_KW = True
    return [kw for kw in (article.get("manual_tags") or []) if isinstance(kw, str) and kw.strip()]


_AVISO_SIN_AUTHOR_KW = False


_SH_PAREN_RE = re.compile(r"([^()]{2,300}?)\s*\(\s*([^()]{2,10}?)\s*\)")


def _sh_long_form(sf: str, candidate: str) -> Optional[str]:
    """Schwartz y Hearst (2003), 'A simple algorithm for identifying abbreviation definitions in
    biomedical text': recorre la sigla de derecha a izquierda buscando cada caracter en la frase
    candidata; el primero debe iniciar una palabra. Devuelve la forma larga mas corta compatible."""
    s_idx, l_idx = len(sf) - 1, len(candidate) - 1
    while s_idx >= 0:
        c = sf[s_idx].lower()
        if not c.isalnum():
            s_idx -= 1
            continue
        while l_idx >= 0 and (
            candidate[l_idx].lower() != c
            or (s_idx == 0 and l_idx > 0 and candidate[l_idx - 1].isalnum())
        ):
            l_idx -= 1
        if l_idx < 0:
            return None
        l_idx -= 1
        s_idx -= 1
    start = candidate.rfind(" ", 0, l_idx + 1) + 1
    return candidate[start:].strip()


def learn_corpus_acronyms(articles: list[dict]) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Aprende pares sigla -> forma larga del PROPIO corpus (palabras clave, titulos y resumenes) con
    Schwartz y Hearst (2003). Solo se aceptan siglas con UNA unica forma larga en todo el corpus
    (tras normalizar sintaxis y singular/plural); las ambiguas se devuelven aparte para el reporte.
    No usa ningun vocabulario externo: es agnostico al dominio."""
    from collections import Counter, defaultdict

    vistos: dict[str, Counter] = defaultdict(Counter)
    for a in articles:
        textos = [str(a.get("title", "") or ""), str(a.get("abstract", "") or "")]
        textos += [str(k) for k in _iter_raw_article_keywords(a)]
        for texto in textos:
            if "(" not in texto:
                continue
            for m in _SH_PAREN_RE.finditer(texto):
                sf = m.group(2).strip()
                if not (2 <= len(sf) <= 10) or not sf[0].isalnum() or not any(ch.isalpha() for ch in sf):
                    continue
                if len(sf.split()) > 2:
                    continue
                palabras = m.group(1).split()
                ventana = min(len(sf) + 5, 2 * len(sf))
                cand = " ".join(palabras[-ventana:])
                lf = _sh_long_form(sf, cand)
                if not lf or len(lf) <= len(sf) or len(lf.split()) < 2 or sf.lower() in lf.lower().split():
                    continue
                lf_norm = singularize_term_agnostic(
                    re.sub(r"\s+", " ", _UNICODE_HYPHENS_RE.sub("-", lf.lower())).strip(" ,.;:-\"'“”‘’")
                )
                sf_norm = _UNICODE_HYPHENS_RE.sub("-", sf.lower())
                if len(sf) > 2 and sf.endswith("s") and sf[:-1].isupper():
                    sf_norm = sf_norm[:-1]          # plural de sigla: 'LLMs' -> 'llm'
                if lf_norm:
                    vistos[sf_norm][lf_norm] += 1
    def _esqueleto(lf: str, conocidas: dict[str, str]) -> str:
        # Misma expansion escrita distinto: sin puntuacion, sin 'and'/'&', cada palabra en singular
        # y con las siglas ya aprendidas expandidas ('generative ai' = 'generative artificial intelligence').
        toks = re.sub(r"[^a-z0-9 ]+", " ", lf.replace("&", " ")).split()
        toks = " ".join(conocidas.get(t, t) for t in toks).split()
        return " ".join(singularize_term_agnostic(t) for t in toks if t != "and")

    def _resolver(conocidas: dict[str, str]):
        unicos: dict[str, str] = {}
        ambiguos: dict[str, list[str]] = {}
        for sf, c in vistos.items():
            grupos = {}
            for lf, n in c.items():
                grupos.setdefault(_esqueleto(lf, conocidas), Counter())[lf] += n
            if len(grupos) == 1:
                # una sola expansion (con variantes de escritura): la mas frecuente, desempate alfabetico
                formas = next(iter(grupos.values()))
                unicos[sf] = sorted(formas.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
            else:
                ambiguos[sf] = sorted(c)
        return unicos, ambiguos

    # Dos pasadas: la segunda usa las siglas inequivocas de la primera para comparar expansiones
    unicos, _ = _resolver({})
    return _resolver({k: v for k, v in unicos.items() if " " not in k})


def build_corpus_keyword_normalizer(
    articles: list[dict],
    trace: Optional[dict] = None,
) -> dict[str, str]:
    """Construye dinamicamente un diccionario de unificacion (variante -> canonico) a partir del corpus.

    Reglas, todas agnosticas al dominio salvo la 1 (opcional y vacia por defecto):
    1. Sinonimos propios de la revision (data/sinonimos_revision.json).
    2. Inversiones MeSH ('sustantivo, adjetivo' -> 'adjetivo sustantivo') cuando existen en el corpus.
    3. Variantes con guion vs espacio ('palabra-compuesta' <-> 'palabra compuesta') hacia la mas frecuente.
    4. Variantes singular/plural ('terminos' <-> 'termino').
    5. Siglas aprendidas del corpus (Schwartz y Hearst, 2003): 'ai' -> 'artificial intelligence' y, si
       el resultado existe en el corpus, 'generative ai' -> 'generative artificial intelligence'.

    Si se pasa `trace`, se llena con {variante: {"canonico", "regla"}} para el reporte de normalizacion.
    """
    from collections import Counter

    kb = load_knowledge_base()
    syns = kb["keyword_synonyms"]
    freq = Counter()
    regla: dict[str, str] = {}

    def _marcar(var: str, nombre: str) -> None:
        regla.setdefault(var, nombre)

    for a in articles:
        seen_in_art = set()
        for raw in _iter_raw_article_keywords(a):
            c = clean_keyword_syntax(raw)
            if not c:
                continue
            if c in syns:
                _marcar(c, "sinonimo de la revision")
                c = syns[c]
            for rule in kb["prefix_canonical_rules"]:
                if c.startswith(rule["prefix"]):
                    _marcar(c, "prefijo de la revision")
                    c = rule["canonical"]
                    break
            if not is_keyword_stopword(c):
                seen_in_art.add(c)
        for c in seen_in_art:
            freq[c] += 1

    corpus_map: dict[str, str] = {}
    vocab = set(freq.keys())

    # 1. Inversiones con coma ('nucleo, adjetivo' -> 'adjetivo nucleo')
    for term in list(vocab):
        if "," in term and " and " not in term:
            parts = [p.strip() for p in term.split(",") if p.strip()]
            if len(parts) == 2:
                inverted = f"{parts[1]} {parts[0]}"
                if inverted in syns:
                    corpus_map[term] = syns[inverted]
                    _marcar(term, "inversion con coma + sinonimo de la revision")
                elif inverted in vocab:
                    corpus_map[term] = inverted
                    _marcar(term, "inversion con coma")

    # 2. Guion vs espacio ('palabra-compuesta' <-> 'palabra compuesta')
    for term in sorted(vocab):
        if "-" in term:
            spaced = term.replace("-", " ")
            if spaced in vocab and spaced != term:
                target = spaced if freq[spaced] >= freq[term] else term
                corpus_map[term] = target
                corpus_map[spaced] = target
                _marcar(term if target == spaced else spaced, "guion / espacio")

    # 3. Singular / plural dentro del vocabulario del corpus
    for term in sorted(vocab):
        mapped = corpus_map.get(term, term)
        sing = singularize_term_agnostic(mapped)
        if sing != mapped:
            if sing in syns:
                corpus_map[term] = syns[sing]
                corpus_map[mapped] = syns[sing]
                _marcar(term, "singular + sinonimo de la revision")
            elif sing in vocab:
                # Preferir la forma singular canonica en redes cientificas (estandar VOSviewer)
                corpus_map[term] = sing
                corpus_map[mapped] = sing
                _marcar(term, "singular / plural")

    # 4. Siglas aprendidas del corpus (Schwartz y Hearst, 2003), solo las inequivocas
    acr, _ = learn_corpus_acronyms(articles)
    for sf, lf in sorted(acr.items()):
        if sf in vocab or singularize_term_agnostic(sf) in vocab:
            for var in {sf, singularize_term_agnostic(sf)} & vocab:
                if var != lf:
                    corpus_map[var] = lf
                    _marcar(var, f"sigla aprendida del corpus ({sf} = {lf})")
    for term in sorted(vocab):
        toks = term.split(" ")
        if len(toks) < 2:
            continue
        nuevos = [acr.get(t, t) for t in toks]
        if nuevos != toks:
            candidato = " ".join(nuevos)
            # Solo si la forma expandida existe en el corpus: evita crear terminos que nadie uso
            if candidato in vocab or singularize_term_agnostic(candidato) in vocab:
                corpus_map[term] = candidato if candidato in vocab else singularize_term_agnostic(candidato)
                _marcar(term, "sigla aprendida del corpus, dentro de un termino")

    # Decision del equipo: variantes que NO se unifican (anula cualquier regla automatica)
    for var in kb.get("no_unificar", set()):
        if var in corpus_map:
            del corpus_map[var]
            regla[var] = "no unificar (decision del equipo)"

    # Resolver transitividad (A -> B -> C)
    for k in list(corpus_map.keys()):
        target = corpus_map[k]
        visited = {k}
        while target in corpus_map and corpus_map[target] != target and target not in visited:
            visited.add(target)
            target = corpus_map[target]
        if target in syns:
            target = syns[target]
        corpus_map[k] = target

    if trace is not None:
        for var, canon in corpus_map.items():
            if var != canon:
                trace[var] = {"canonico": canon, "regla": regla.get(var, "transitividad"),
                              "frecuencia_variante": int(freq.get(var, 0))}
        for var, nombre in regla.items():
            if var not in corpus_map and var in syns:
                trace[var] = {"canonico": syns[var], "regla": nombre, "frecuencia_variante": 0}
            elif nombre.startswith("no unificar"):
                trace[var] = {"canonico": var, "regla": nombre, "frecuencia_variante": int(freq.get(var, 0))}
    return corpus_map


# ----------------------------------------------------------------------
# Vocabulario comun de palabras clave para todas las fases
# ----------------------------------------------------------------------
_VOCABULARIO: dict = {"clave": None, "mapa": None}


def _huella_corpus(articles: list[dict]) -> tuple:
    return (len(articles), tuple(str(a.get("key", "")) for a in articles))


def prepare_corpus_vocabulary(articles: list[dict], write_report: bool = True) -> dict[str, str]:
    """Calcula UNA vez el mapa de normalizacion del corpus y lo deja disponible para todas las fases
    (article_keywords). Escribe output/reports/normalizacion_palabras_clave.csv con cada equivalencia
    aplicada y la regla que la produjo, para revisarla y declararla en el manuscrito."""
    trace: dict = {}
    mapa = build_corpus_keyword_normalizer(articles, trace=trace)
    _VOCABULARIO["clave"] = _huella_corpus(articles)
    _VOCABULARIO["mapa"] = mapa
    if write_report:
        try:
            import csv
            from config import REPORTS_DIR
            acr, amb = learn_corpus_acronyms(articles)
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            path = REPORTS_DIR / "normalizacion_palabras_clave.csv"
            with open(path, "w", encoding="utf-8", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["variante", "canonico", "regla", "frecuencia_variante"])
                for var in sorted(trace):
                    t = trace[var]
                    w.writerow([var, t["canonico"], t["regla"], t["frecuencia_variante"]])
                for sf in sorted(amb):
                    w.writerow([sf, "", "sigla AMBIGUA, no se aplica: " + " | ".join(amb[sf]), ""])
            from collections import Counter as _C
            por_regla = _C(t["regla"].split(" (")[0] for t in trace.values())
            logger.info(
                f"Vocabulario de palabras clave: {len(trace)} variantes unificadas "
                f"({dict(por_regla)}); {len(acr)} siglas inequivocas aprendidas, {len(amb)} ambiguas. "
                f"Reporte: {path}"
            )
        except Exception as e:
            logger.warning(f"No se pudo escribir el reporte de normalizacion: {e}")
    return mapa


def article_keywords(article: dict, articles: Optional[list[dict]] = None) -> list[str]:
    """Palabras clave normalizadas de un articulo con el vocabulario comun del corpus.

    Todas las fases que agrupan, cuentan o fechan palabras clave deben usar esta funcion para que
    'palabra clave' signifique lo mismo en todo el pipeline. Si se pasa `articles` y el vocabulario
    vigente es de otro corpus, se recalcula."""
    from src.modelo_corpus import article_entry
    entrada = article_entry(article)
    if entrada is not None:
        return list(entrada["palabras_clave"])
    if articles is not None and _VOCABULARIO["clave"] != _huella_corpus(articles):
        prepare_corpus_vocabulary(articles, write_report=False)
    if _VOCABULARIO["mapa"] is None:
        logger.warning("Vocabulario del corpus no preparado: se normaliza sin reglas del corpus "
                       "(llame a prepare_corpus_vocabulary(articles) al inicio).")
    return extract_normalized_article_keywords(article, corpus_map=_VOCABULARIO["mapa"])


def extract_normalized_article_keywords(
    article: dict,
    corpus_map: Optional[dict[str, str]] = None,
    drop_stopwords: bool = True,
) -> list[str]:
    """Extrae y normaliza las palabras clave de un articulo sin duplicados ni stopwords."""
    result = set()
    for raw in _iter_raw_article_keywords(article):
        canon = canonicalize_keyword(raw, corpus_map=corpus_map)
        if not canon:
            continue
        if drop_stopwords and is_keyword_stopword(canon):
            continue
        result.add(canon)
    return sorted(result)


def normalize_author_display_name(name: str) -> str:
    """Unifica variantes tipograficas en nombres de autores (guiones Unicode, ALL CAPS, espacios)."""
    if not name or not isinstance(name, str):
        return ""
    s = _UNICODE_QUOTES_RE.sub("'", _UNICODE_HYPHENS_RE.sub("-", name.strip()))
    s = re.sub(r"\s+", " ", s)
    if not s:
        return ""
    # Si algun token de >=3 letras esta completamente en mayusculas o minusculas, aplicar title-case
    tokens = s.split(" ")
    fixed_tokens = []
    for tok in tokens:
        # Preservar iniciales con puntos (ej. 'K.A.S.H.' o 'S.D.R.Y.L')
        if "." in tok and len(re.sub(r"[^A-Za-z]", "", tok)) <= 6:
            fixed_tokens.append(tok.upper())
            continue
        alpha_only = re.sub(r"[^A-Za-zÀ-ÿ]", "", tok)
        if len(alpha_only) >= 3 and (alpha_only.isupper() or alpha_only.islower()):
            fixed_tokens.append("-".join(p.capitalize() for p in tok.split("-")))
        else:
            fixed_tokens.append(tok)
    return " ".join(fixed_tokens)


def normalize_institution_name(name: str, corpus_inst_map: Optional[dict[str, str]] = None) -> str:
    """Limpia nombres de instituciones (sufijos de pais OpenAlex, apostrofos Unicode y sinonimos de KB)."""
    if not name or not isinstance(name, str):
        return ""
    kb = load_knowledge_base()
    s = _UNICODE_QUOTES_RE.sub("'", _UNICODE_HYPHENS_RE.sub("-", name.strip()))
    # Quitar sufijo de pais entre parentesis al final, ej. 'Nombre Institucion (Pais)'
    s = re.sub(r"\s*\([A-Z][A-Za-z\s,.&-]{2,30}\)\s*$", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return ""
    low = s.lower()
    if low in kb["institution_synonyms"]:
        s = kb["institution_synonyms"][low]
        low = s.lower()
    if corpus_inst_map and low in corpus_inst_map:
        return corpus_inst_map[low]
    return s


def build_corpus_institution_normalizer(articles: list[dict]) -> dict[str, str]:
    """Detecta y unifica automaticamente variantes 'University of X' <-> 'X University' y mayusculas en el corpus."""
    from collections import Counter

    kb = load_knowledge_base()
    counts = Counter()
    display_forms: dict[str, Counter] = {}

    for a in articles:
        for auth in (a.get("openalex", {}) or {}).get("authorships", []) or []:
            for inst in auth.get("institutions", []) or []:
                raw = inst.get("name", "")
                cleaned = normalize_institution_name(raw)
                if cleaned:
                    low = cleaned.lower()
                    counts[low] += 1
                    display_forms.setdefault(low, Counter())[cleaned] += 1

    inst_map: dict[str, str] = {
        low: cnt.most_common(1)[0][0]
        for low, cnt in display_forms.items()
    }

    # Unificar 'university of X' <-> 'X university' cuando ambos aparecen en el corpus
    for low in list(counts.keys()):
        if low.startswith("university of "):
            x = low[len("university of "):].strip()
            alt = f"{x} university"
            if alt in counts:
                if low in kb["institution_synonyms"]:
                    canon = kb["institution_synonyms"][low]
                elif alt in kb["institution_synonyms"]:
                    canon = kb["institution_synonyms"][alt]
                else:
                    winner = low if counts[low] >= counts[alt] else alt
                    canon = inst_map[winner]
                inst_map[low] = canon
                inst_map[alt] = canon

    return inst_map


def format_journal_display_name(name: str) -> str:
    """Formatea el nombre de revista eliminando subtitulos redundantes y corrigiendo ALL-CAPS."""
    s = normalize_journal_name(name)
    if not s:
        return ""
    kb = load_knowledge_base()
    acronyms = kb["journal_acronyms"]

    # Recortar subtitulos editoriales largos (' : official journal of...', ': an international journal...')
    s = re.sub(
        r"\s*:\s*(?:the\s+|an?\s+)?(?:official\s+(?:journal|publication)|journal\s+of\s+the\s+american).*$",
        "",
        s,
        flags=re.IGNORECASE,
    )
    # Quitar desambiguacion geografica de Scopus/OpenAlex en revistas si el nombre base es claro
    # salvo cuando sea parte esencial, pero normalizar ALL-CAPS o all-lowercase
    alpha_chars = [c for c in s if c.isalpha()]
    is_all_caps = len(alpha_chars) >= 4 and all(c.isupper() for c in alpha_chars)
    is_all_lower = len(alpha_chars) >= 4 and all(c.islower() for c in alpha_chars)

    if is_all_caps or is_all_lower:
        minor_words = {"of", "in", "and", "for", "on", "the", "to", "with", "at", "by", "from", "or"}
        words = s.split(" ")
        formatted = []
        for idx, w in enumerate(words):
            clean_w = re.sub(r"[^A-Za-z0-9-]", "", w).upper()
            if clean_w in acronyms:
                # Reemplazar preservando signos de puntuacion adyacentes
                prefix = re.match(r"^[^A-Za-z0-9]*", w).group(0)
                suffix = re.search(r"[^A-Za-z0-9]*$", w).group(0)
                formatted.append(f"{prefix}{acronyms[clean_w]}{suffix}")
            elif idx > 0 and w.lower() in minor_words:
                formatted.append(w.lower())
            else:
                formatted.append(w.capitalize())
        s = " ".join(formatted)

    return s


def disambiguate_citation_labels(id_to_label: dict[str, str]) -> dict[str, str]:
    """Desambigua etiquetas 'Autor (Año)' colisionantes añadiendo sufijos a, b, c ('Firth (2017a)', 'Firth (2017b)')."""
    from collections import defaultdict

    by_label: dict[str, list[str]] = defaultdict(list)
    for node_id, lbl in id_to_label.items():
        if lbl:
            by_label[lbl].append(node_id)

    result = dict(id_to_label)
    year_paren_re = re.compile(r"^(.+\(\d{4})(\))$")

    for lbl, node_ids in by_label.items():
        if len(node_ids) <= 1:
            continue
        sorted_ids = sorted(node_ids, key=str)
        for idx, nid in enumerate(sorted_ids):
            # Generar sufijo a, b, ..., z, aa, ab...
            if idx < 26:
                sfx = chr(ord("a") + idx)
            else:
                sfx = chr(ord("a") + (idx // 26) - 1) + chr(ord("a") + (idx % 26))
            m = year_paren_re.match(lbl)
            if m:
                result[nid] = f"{m.group(1)}{sfx}{m.group(2)}"
            else:
                result[nid] = f"{lbl} [{sfx}]"

    return result


def _kill_child_headless_chrome() -> None:
    """Termina subprocesos headless de Chrome huerfanos hijos del proceso actual en Windows."""
    if os.name != "nt":
        return
    try:
        import subprocess
        pid = os.getpid()
        cmd = (
            f"Get-CimInstance Win32_Process -Filter \"Name = 'chrome.exe' AND ParentProcessId = {pid}\" "
            "| ForEach-Object { taskkill /F /T /PID $_.ProcessId 2>$null }"
        )
        subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except Exception:
        pass


def safe_write_plotly_png(
    fig: Any,
    png_path: Path | str,
    width: Optional[int] = None,
    height: Optional[int] = None,
    scale: int | float = 2,
    timeout: float = 25.0,
    log: Optional[logging.Logger] = None,
) -> bool:
    """Exporta una figura Plotly a PNG de forma segura con timeout duro y sin depender de CDN externo.

    Evita que Kaleido/Choreographer bloquee el pipeline en Windows esperando a MathJax CDN
    o al cierre del subproceso headless de Chrome.
    """
    import threading

    active_logger = log or logger
    target_path = Path(png_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    err_holder: list[BaseException] = []

    def _export_worker() -> None:
        try:
            import kaleido
            if hasattr(kaleido, "write_fig_sync"):
                layout = getattr(fig, "layout", None)
                eff_w = width or getattr(layout, "width", None) or 1200
                eff_h = height or getattr(layout, "height", None) or 800
                opts = {
                    "format": target_path.suffix.lstrip(".") or "png",
                    "width": int(eff_w),
                    "height": int(eff_h),
                    "scale": scale,
                }
                kaleido.write_fig_sync(
                    fig,
                    path=str(target_path),
                    opts=opts,
                    kopts={"mathjax": False, "timeout": timeout},
                )
            else:
                kwargs: dict[str, Any] = {"scale": scale}
                if width is not None:
                    kwargs["width"] = width
                if height is not None:
                    kwargs["height"] = height
                fig.write_image(str(target_path), **kwargs)
        except BaseException as exc:
            err_holder.append(exc)

    worker = threading.Thread(target=_export_worker, daemon=True)
    worker.start()
    worker.join(timeout=timeout + 5.0)

    if worker.is_alive():
        _kill_child_headless_chrome()
        if target_path.exists() and target_path.stat().st_size > 0:
            active_logger.info(f"Saved: {target_path}")
            return True
        active_logger.warning(
            f"PNG export timed out after {timeout:.0f}s ({target_path.name}); HTML preserved."
        )
        return False

    if err_holder:
        _kill_child_headless_chrome()
        # Si Kaleido escribio el PNG antes de fallar al cerrar Chrome, conservarlo como exito
        if target_path.exists() and target_path.stat().st_size > 0:
            active_logger.info(f"Saved: {target_path}")
            return True
        active_logger.warning(f"PNG export failed for {target_path.name}: {err_holder[0]}")
        return False

    if target_path.exists() and target_path.stat().st_size > 0:
        active_logger.info(f"Saved: {target_path}")
        return True
    return False




# ----------------------------------------------------------------------
# Betweenness con pesos de FUERZA (co-citacion, acoplamiento, co-ocurrencia)
# ----------------------------------------------------------------------
_BETWEENNESS_EXACTA_HASTA = 5000   # nodos; por encima se muestrea con semilla fija


def strength_betweenness(G, normalized: bool = True) -> dict:
    """Betweenness (Freeman, 1977) para redes cuyos pesos son FUERZA de vinculo.

    networkx interpreta `weight` como DISTANCIA: pasarle co-citaciones hace que los vinculos fuertes
    cuenten como caminos largos. Aqui la distancia de cada enlace es 1/peso (convencion de Newman, 2001;
    Opsahl et al., 2010 con alpha = 1). Calculo exacto (determinista) salvo en redes muy grandes, donde se
    muestrea con semilla fija para que el resultado no cambie entre corridas.
    """
    import networkx as nx
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    for u, v, d in G.edges(data=True):
        try:
            w = float(d.get("weight", 1) or 1)
        except (TypeError, ValueError):
            w = 1.0
        H.add_edge(u, v, _dist=(1.0 / w) if w > 0 else float("inf"))
    if H.number_of_nodes() > _BETWEENNESS_EXACTA_HASTA:
        return nx.betweenness_centrality(H, weight="_dist", normalized=normalized,
                                         k=min(1000, H.number_of_nodes()), seed=42)
    return nx.betweenness_centrality(H, weight="_dist", normalized=normalized)
