"""Fase 4: Exportacion e integracion - bibliometrix CSV/BibTeX, Excel, Markdown, R."""
from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path

import pandas as pd

from config import (
    BIBLIOMETRIX_CSV, BIBLIOMETRIX_BIB, BIBLIOMETRIX_COLUMNS,
    OUTPUT_DIR, REPORTS_DIR, OPENALEX_CACHE_DIR,
)
from src.utils import DiskCache, get_canonical_oa_status

logger = logging.getLogger("bibliometrico.export")

# Pre-compiled patterns
_RE_WORK_ID_FALLBACK = re.compile(r'^W\d+$')


# ---------------------------------------------------------------------------
# Document type normalization: Zotero/OpenAlex -> Bibliometrix DT
# ---------------------------------------------------------------------------

_ZOTERO_TO_DT: dict[str, str] = {
    # Zotero item_type values
    "journalarticle": "ARTICLE",
    "conferencepaper": "ARTICLE; PROCEEDINGS PAPER",
    "book": "BOOK",
    "booksection": "BOOK CHAPTER",
    "thesis": "THESIS",
    "report": "REPORT",
    "preprint": "ARTICLE; EARLY ACCESS",
    "manuscript": "ARTICLE",
    "presentation": "MEETING ABSTRACT",
    "letter": "LETTER",
    "review": "REVIEW",
    "editorial": "EDITORIAL MATERIAL",
    "note": "NOTE",
    "patent": "PATENT",
    "webpage": "ARTICLE",
    "document": "ARTICLE",
    # OpenAlex type values
    "article": "ARTICLE",
    "review-article": "REVIEW",
    "proceedings-article": "ARTICLE; PROCEEDINGS PAPER",
    "book-chapter": "BOOK CHAPTER",
    "dissertation": "THESIS",
    "dataset": "DATA PAPER",
    "paratext": "EDITORIAL MATERIAL",
    "erratum": "CORRECTION",
    "editorial": "EDITORIAL MATERIAL",
    "letter": "LETTER",
    "other": "ARTICLE",
}


def _normalize_dt(raw_type: str, openalex_type: str = "") -> str:
    """Normalize document type to Bibliometrix DT format.

    Tries Zotero item_type first, then OpenAlex type as fallback.
    Returns uppercase Bibliometrix-compatible DT string.
    """
    if not raw_type and not openalex_type:
        return "ARTICLE"

    # Try raw_type (Zotero) — normalize to lowercase, remove spaces
    key = (raw_type or "").strip().lower().replace(" ", "").replace("_", "")
    dt = _ZOTERO_TO_DT.get(key)
    if dt:
        return dt

    # Try OpenAlex type
    oa_key = (openalex_type or "").strip().lower()
    dt = _ZOTERO_TO_DT.get(oa_key)
    if dt:
        return dt

    # Fallback: uppercase the raw type if it looks already normalized
    if raw_type and raw_type.isupper() and len(raw_type) > 2:
        return raw_type

    return "ARTICLE"


# ---------------------------------------------------------------------------
# ISO-2 -> country name mapping for bibliometrix compatibility
# ---------------------------------------------------------------------------

_ISO2_TO_COUNTRY = {
    "AF": "AFGHANISTAN", "AL": "ALBANIA", "DZ": "ALGERIA", "AR": "ARGENTINA",
    "AM": "ARMENIA", "AU": "AUSTRALIA", "AT": "AUSTRIA", "AZ": "AZERBAIJAN",
    "BH": "BAHRAIN", "BD": "BANGLADESH", "BY": "BELARUS", "BE": "BELGIUM",
    "BJ": "BENIN", "BT": "BHUTAN", "BO": "BOLIVIA", "BA": "BOSNIA AND HERZEGOVINA",
    "BW": "BOTSWANA", "BR": "BRAZIL", "BN": "BRUNEI", "BG": "BULGARIA",
    "BF": "BURKINA FASO", "KH": "CAMBODIA", "CM": "CAMEROON", "CA": "CANADA",
    "CL": "CHILE", "CN": "CHINA", "CO": "COLOMBIA", "CR": "COSTA RICA",
    "HR": "CROATIA", "CU": "CUBA", "CY": "CYPRUS", "CZ": "CZECH REPUBLIC",
    "DK": "DENMARK", "DO": "DOMINICAN REPUBLIC", "EC": "ECUADOR", "EG": "EGYPT",
    "SV": "EL SALVADOR", "EE": "ESTONIA", "ET": "ETHIOPIA", "FI": "FINLAND",
    "FR": "FRANCE", "GA": "GABON", "GE": "GEORGIA", "DE": "GERMANY",
    "GH": "GHANA", "GR": "GREECE", "GT": "GUATEMALA", "HN": "HONDURAS",
    "HK": "HONG KONG", "HU": "HUNGARY", "IS": "ICELAND", "IN": "INDIA",
    "ID": "INDONESIA", "IR": "IRAN", "IQ": "IRAQ", "IE": "IRELAND",
    "IL": "ISRAEL", "IT": "ITALY", "JM": "JAMAICA", "JP": "JAPAN",
    "JO": "JORDAN", "KZ": "KAZAKHSTAN", "KE": "KENYA", "KW": "KUWAIT",
    "KG": "KYRGYZSTAN", "LA": "LAOS", "LV": "LATVIA", "LB": "LEBANON",
    "LY": "LIBYA", "LT": "LITHUANIA", "LU": "LUXEMBOURG", "MO": "MACAO",
    "MK": "NORTH MACEDONIA", "MG": "MADAGASCAR", "MW": "MALAWI", "MY": "MALAYSIA",
    "ML": "MALI", "MT": "MALTA", "MX": "MEXICO", "MD": "MOLDOVA",
    "MN": "MONGOLIA", "ME": "MONTENEGRO", "MA": "MOROCCO", "MZ": "MOZAMBIQUE",
    "MM": "MYANMAR", "NA": "NAMIBIA", "NP": "NEPAL", "NL": "NETHERLANDS",
    "NZ": "NEW ZEALAND", "NI": "NICARAGUA", "NE": "NIGER", "NG": "NIGERIA",
    "NO": "NORWAY", "OM": "OMAN", "PK": "PAKISTAN", "PS": "PALESTINE",
    "PA": "PANAMA", "PY": "PARAGUAY", "PE": "PERU", "PH": "PHILIPPINES",
    "PL": "POLAND", "PT": "PORTUGAL", "QA": "QATAR", "RO": "ROMANIA",
    "RU": "RUSSIA", "RW": "RWANDA", "SA": "SAUDI ARABIA", "SN": "SENEGAL",
    "RS": "SERBIA", "SG": "SINGAPORE", "SK": "SLOVAKIA", "SI": "SLOVENIA",
    "ZA": "SOUTH AFRICA", "KR": "SOUTH KOREA", "ES": "SPAIN", "LK": "SRI LANKA",
    "SD": "SUDAN", "SE": "SWEDEN", "CH": "SWITZERLAND", "SY": "SYRIA",
    "TW": "TAIWAN", "TZ": "TANZANIA", "TH": "THAILAND", "TN": "TUNISIA",
    "TR": "TURKEY", "UG": "UGANDA", "UA": "UKRAINE", "AE": "UNITED ARAB EMIRATES",
    "GB": "UNITED KINGDOM", "US": "USA", "UY": "URUGUAY", "UZ": "UZBEKISTAN",
    "VE": "VENEZUELA", "VN": "VIETNAM", "YE": "YEMEN", "ZM": "ZAMBIA",
    "ZW": "ZIMBABWE",
}


def _iso2_to_country_name(code: str) -> str:
    """Convierte ISO 3166-1 alpha-2 a nombre de pais (MAYUSCULAS).
    Bibliometrix necesita nombres completos, no codigos ISO.
    """
    if not code:
        return ""
    name = _ISO2_TO_COUNTRY.get(code.upper(), "")
    if name:
        return name
    # Fallback: try pycountry
    try:
        import pycountry
        c = pycountry.countries.get(alpha_2=code.upper())
        return c.name.upper() if c else code.upper()
    except ImportError:
        return code.upper()


# ---------------------------------------------------------------------------
# Helpers para formateo bibliometrix
# ---------------------------------------------------------------------------

def _to_ascii(text: str) -> str:
    """Transliterate text to ASCII. Greek/Cyrillic etc. get decomposed or dropped."""
    if not text:
        return ""
    # NFKD decomposition splits accented chars into base + combining mark
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_bytes = nfkd.encode("ascii", "ignore")
    result = ascii_bytes.decode("ascii").strip()
    # Collapse multiple spaces
    return re.sub(r"\s+", " ", result)


def _format_author_bibliometrix(name: str) -> str:
    """Convierte 'Apellido, N.I.' -> 'APELLIDO NI' (mayusculas, sin puntos).
    Si el formato es 'First Last' lo convierte tambien.
    Aplica _to_ascii para normalizar acentos y strip hyphens de iniciales.
    """
    name = _to_ascii(name.strip())
    if not name:
        return ""
    # Remove dots
    name = name.replace(".", "")
    if "," in name:
        parts = [p.strip() for p in name.split(",", 1)]
        last = parts[0].upper()
        given = parts[1] if len(parts) > 1 else ""
        # Iniciales de cada nombre y de cada segmento con guion ('Monica F.' -> 'MF', 'Seung-Jun' -> 'SJ')
        initials = "".join(seg[0].upper() for tok in given.split() for seg in tok.split("-") if seg)
        return f"{last} {initials}".strip()
    else:
        # "First Middle Last" -> "LAST FM"
        tokens = name.split()
        if len(tokens) >= 2:
            last = tokens[-1].upper()
            initials = "".join(t[0].upper() for t in tokens[:-1])
            return f"{last} {initials}"
        return name.upper()


def _normalize_journal_for_cr(journal: str) -> str:
    """Normaliza nombre de revista para matching entre SO y CR en bibliometrix.

    Reemplaza '&' por 'AND', elimina sufijos parentéticos como '(Switzerland)',
    y limpia espacios extra.
    """
    import re
    if not journal:
        return ""
    j = journal.replace("&", "AND")
    j = re.sub(r"\s*\([^)]*\)\s*$", "", j)  # Remove trailing (...)
    j = re.sub(r"\s{2,}", " ", j).strip()
    return j


def _author_display_to_cr(display_name: str) -> str:
    """Convierte nombre OpenAlex 'Given Middle Family' a formato CR 'FAMILY GM'.

    Maneja nombres con guiones (Seung-Jun → SJ, Li-Wei → LW).
    Esto coincide con el formato que bibliometrix genera en SR_FULL desde AU.
    """
    if not display_name:
        return "UNKNOWN"
    ascii_name = _to_ascii(display_name)
    parts = ascii_name.split()
    if not parts:
        parts = display_name.split()
    if not parts:
        return "UNKNOWN"
    if len(parts) == 1:
        return parts[0].upper()
    family = parts[-1].upper()
    # Build initials: for each given-name token, take first letter of each
    # hyphen-separated segment (e.g., "Seung-Jun" → "SJ", "Li-Wei" → "LW")
    init_chars = []
    for p in parts[:-1]:
        for segment in p.split("-"):
            if segment:
                init_chars.append(segment[0].upper())
    initials = "".join(init_chars)
    return f"{family} {initials}" if initials else family


def _resolve_cr_from_cache(ref_urls: list[str], cache: DiskCache,
                           mem_cache: dict = None,
                           refs_resolved: list[dict] | None = None,
                           ) -> list[str]:
    """Resuelve URLs de OpenAlex a formato 'LASTNAME IN, YEAR, JOURNAL, V##, P##, DOI ...'
    usando la cache en disco (no hace peticiones de red).
    Produce strings ASCII-safe en MAYUSCULAS para compatibilidad con bibliometrix.
    mem_cache: optional in-memory dict to avoid repeated disk lookups for same work_id.
    refs_resolved: optional pre-resolved references (from Phase 1i) with guaranteed year/author.
    """
    # Build lookup from references_resolved by work_id for fast access
    rr_by_id: dict[str, dict] = {}
    if refs_resolved:
        for rr in refs_resolved:
            if not isinstance(rr, dict):
                continue
            oa_id = rr.get("openalex_id", "")
            wid = oa_id.split("/")[-1] if "/" in oa_id else oa_id
            if wid:
                rr_by_id[wid] = rr

    resolved = []
    for ref_url in ref_urls:
        work_id = ref_url.split("/")[-1] if "/" in ref_url else ref_url

        # Check in-memory cache first (avoid disk I/O)
        if mem_cache is not None and work_id in mem_cache:
            cached = mem_cache[work_id]
        else:
            cached = cache.get(f"work:{work_id}")
            if not cached or not isinstance(cached, dict):
                cached = cache.get(f"doi:{work_id}")
            if mem_cache is not None:
                mem_cache[work_id] = cached

        # Pre-resolved reference data (guaranteed year/author from Phase 1i)
        rr = rr_by_id.get(work_id, {})

        if cached and isinstance(cached, dict):
            # Extraer primer autor en formato LASTNAME INITIALS (ej: KIM SJ)
            first_author = "UNKNOWN"
            authorships = cached.get("authorships", [])
            if authorships:
                display_name = authorships[0].get("author", {}).get("display_name", "")
                if not display_name:
                    display_name = authorships[0].get("name", "")
                if display_name:
                    first_author = _author_display_to_cr(display_name)

            # Fallback author from references_resolved
            if first_author == "UNKNOWN" and rr.get("first_author"):
                first_author = _author_display_to_cr(rr["first_author"])

            year = cached.get("publication_year", "")
            # Fallback year from references_resolved (Phase 1i)
            if not year and rr.get("year"):
                year = rr["year"]

            # Journal: primary_location -> source -> display_name
            journal = ""
            primary = cached.get("primary_location", {}) or {}
            source = primary.get("source", {}) or {}
            journal = source.get("display_name", "")
            if not journal:
                # Fallback: host_venue (older API)
                host = cached.get("host_venue", {}) or {}
                journal = host.get("display_name", "")
            if not journal:
                # Fallback: locations
                for loc in cached.get("locations", []) or []:
                    src = (loc or {}).get("source", {}) or {}
                    j = src.get("display_name", "")
                    if j:
                        journal = j
                        break
            journal = _normalize_journal_for_cr(_to_ascii(journal).upper()) if journal else ""

            # Volume y paginas del biblio
            biblio = cached.get("biblio", {}) or {}
            volume = biblio.get("volume", "")
            first_page = biblio.get("first_page", "")

            # Skip entries without a year — bibliometrix rpys() requires
            # parseable years in CR; entries like "UNKNOWN" crash seq().
            if not year:
                continue

            # DOI del trabajo referenciado (para matching local en histNetwork)
            ref_doi = cached.get("doi", "") or ""
            # OpenAlex devuelve "https://doi.org/10.xxx/yyy" — extraer el bare DOI
            if ref_doi:
                ref_doi = ref_doi.replace("https://doi.org/", "").replace("http://doi.org/", "").strip()

            # Construir referencia (all ASCII uppercase)
            parts_cr = [first_author]
            parts_cr.append(str(year))
            if journal:
                parts_cr.append(journal)
            if volume:
                parts_cr.append(f"V{volume}")
            if first_page:
                parts_cr.append(f"P{first_page}")
            if ref_doi:
                parts_cr.append(f"DOI {ref_doi.upper()}")

            ref_str = ", ".join(parts_cr)
        elif rr and rr.get("year"):
            # No cache entry but we have references_resolved data with year
            fa = _author_display_to_cr(rr.get("first_author", "")) if rr.get("first_author") else "UNKNOWN"
            parts_cr = [fa, str(rr["year"])]
            ref_doi = (rr.get("doi") or "").replace("https://doi.org/", "").replace("http://doi.org/", "").strip()
            if ref_doi:
                parts_cr.append(f"DOI {ref_doi.upper()}")
            ref_str = ", ".join(parts_cr)
        else:
            # No year available — skip to avoid breaking bibliometrix rpys()
            continue

        resolved.append(ref_str)

    return resolved


def _format_c1(authorships: list[dict]) -> str:
    """Formato C1 estilo WoS: '[AUTOR] Institucion, PAIS.; [AUTOR2] Inst2, PAIS2.'.
    Con DB=ISI, bibliometrix (AU_CO) solo reconoce un pais seguido de punto ('USA.');
    sin el punto por afiliacion detectaba unicamente el pais de la ULTIMA afiliacion.
    El pais debe ser el nombre completo en mayusculas (CHINA, USA, etc.).
    """
    parts = []
    for auth in authorships:
        name = auth.get("name", "")
        if not name:
            continue
        name_upper = _format_author_bibliometrix(name)
        institutions = auth.get("institutions", [])
        if institutions:
            for inst in institutions:
                inst_name = inst.get("name", "")
                country_code = inst.get("country", "")
                country_name = _iso2_to_country_name(country_code)
                entry = f"[{name_upper}] {inst_name}"
                if country_name:
                    entry += f", {country_name}."
                parts.append(entry)
        else:
            # Sin institucion pero con pais en OpenAlex (authorships[].countries): se escribe el pais
            # para que bibliometrix no pierda el articulo en AU_CO/SCP/MCP (antes: 49 de 347 sin pais).
            country_names = [n for n in (_iso2_to_country_name(c) for c in auth.get("countries", []) or []) if n]
            if country_names:
                parts.append(f"[{name_upper}] {country_names[0]}.")
            else:
                parts.append(f"[{name_upper}]")
    return "; ".join(parts)


def _format_rp(authorships: list[dict]) -> str:
    """Extrae autor correspondiente de authorships (is_corresponding=True o position=first)."""
    def _rp_entry(auth: dict) -> str:
        name = _format_author_bibliometrix(auth.get("name", ""))
        institutions = auth.get("institutions", [])
        if institutions:
            inst = institutions[0]
            country_name = _iso2_to_country_name(inst.get("country", ""))
            entry = f"{name}, {inst.get('name', '')}, {country_name}".rstrip(", ")
            # Punto final estilo WoS: bibliometrix deriva AU1_CO de RP buscando 'PAIS.'
            return f"{entry}." if country_name else entry
        country_names = [n for n in (_iso2_to_country_name(c) for c in auth.get("countries", []) or []) if n]
        return f"{name}, {country_names[0]}." if country_names else name

    # Buscar is_corresponding primero
    for auth in authorships:
        if auth.get("is_corresponding", False):
            return _rp_entry(auth)

    # Fallback: primer autor
    if authorships:
        return _rp_entry(authorships[0])
    return ""


def _corresponding_author_country(authorships: list[dict]) -> str:
    """Pais del autor de correspondencia (mismo autor que RP); si no hay, el del primer autor.

    Es la semantica de AU1_CO en bibliometrix (metaTagExtraction 'AU1_CO' lo deriva de RP),
    que es la que usa para clasificar SCP/MCP. Antes se escribian aqui TODOS los paises.
    """
    if not authorships:
        return ""
    chosen = next((a for a in authorships if a.get("is_corresponding", False)), authorships[0])
    codes = list(chosen.get("countries", []) or [])
    if not codes:
        codes = [i.get("country", "") for i in chosen.get("institutions", []) if i.get("country")]
    for c in codes:
        name = _iso2_to_country_name(c)
        if name:
            return name
    return ""


def _extract_author_countries(authorships: list[dict]) -> str:
    """Extrae paises unicos de autores como nombres completos: 'CHINA;USA;GERMANY'."""
    countries = []
    seen = set()
    for auth in authorships:
        for c in auth.get("countries", []):
            if c and c not in seen:
                seen.add(c)
                name = _iso2_to_country_name(c)
                if name:
                    countries.append(name)
    return ";".join(countries)


def _extract_subject_categories(article: dict) -> str:
    """Extrae subject categories de topics/concepts de OpenAlex."""
    openalex = article.get("openalex", {})
    categories = set()

    # De topics: usar field
    for topic in openalex.get("topics", []):
        field = topic.get("field", "")
        if field:
            categories.add(field)
        subfield = topic.get("subfield", "")
        if subfield:
            categories.add(subfield)

    # De concepts: nivel 0 y 1
    for concept in openalex.get("concepts", article.get("concepts", [])):
        level = concept.get("level", 99)
        if level <= 1 and concept.get("name"):
            categories.add(concept["name"])

    return "; ".join(sorted(categories)[:5])


def _extract_keywords_plus(article: dict) -> str:
    """Genera Keywords Plus (ID) desde OpenAlex concepts nivel 1-2.

    Bibliometrix usa ID para Keywords Plus — palabras clave derivadas
    automaticamente. Los concepts de OpenAlex nivel 1-2 cumplen esta funcion.
    """
    openalex = article.get("openalex", {})
    concepts = openalex.get("concepts", [])
    kw = []
    for c in concepts:
        level = c.get("level", 99)
        score = c.get("score", 0) or 0
        name = c.get("display_name", "") or c.get("name", "")
        if 1 <= level <= 2 and float(score) >= 0.3 and name:
            kw.append(name.upper())
    return "; ".join(kw[:10])


def _extract_wos_categories(article: dict) -> str:
    """Genera WC (Web of Science Categories) desde OpenAlex topics.

    Usa el campo 'field' de topics como equivalente a WC.
    """
    openalex = article.get("openalex", {})
    fields = set()
    for topic in openalex.get("topics", []):
        field = topic.get("field", "")
        if field:
            fields.add(field.upper())
    if not fields:
        # Fallback: concepts nivel 0 (disciplinas amplias)
        for c in openalex.get("concepts", []):
            if c.get("level", 99) == 0 and c.get("name"):
                name = c.get("display_name", "") or c.get("name", "")
                fields.add(name.upper())
    return "; ".join(sorted(fields)[:5])


def _clean_language(lang: str) -> str:
    """Limpia valores de idioma: 'nan' -> '' y normaliza."""
    if not lang or lang == "nan" or lang == "None":
        return ""
    return lang.strip()


# ---------------------------------------------------------------------------
# Generadores de exportacion
# ---------------------------------------------------------------------------

def articles_to_bibliometrix_df(articles: list[dict]) -> pd.DataFrame:
    """Convierte articulos a DataFrame formato bibliometrix con campos completos."""
    cache = DiskCache(OPENALEX_CACHE_DIR)
    mem_cache = {}  # In-memory cache to avoid repeated disk lookups
    rows = []
    cr_resolved_count = 0
    cr_fallback_count = 0

    for a in articles:
        openalex = a.get("openalex", {})
        authorships = openalex.get("authorships", [])

        # AU en formato WoS ('APELLIDO INICIALES'), con la MISMA fuente y funcion que el CR (OpenAlex
        # display_name -> _author_display_to_cr). Asi el SR de un articulo coincide con como lo citan
        # otros articulos del corpus y bibliometrix encuentra las citas locales (Historiograph).
        oa_names = [au.get("name", "") for au in authorships if au.get("name")]
        if oa_names:
            authors_formatted = [_author_display_to_cr(n) for n in oa_names]
        else:
            authors_formatted = [_format_author_bibliometrix(n) for n in a.get("authors", [])]
        au_str = ";".join(x for x in authors_formatted if x)
        # J9: revista con la misma fuente y normalizacion que el CR (bibliometrix arma el SR con J9)
        j9_raw = openalex.get("source_display_name") or a.get("publication_title", "")
        j9_str = _normalize_journal_for_cr(_to_ascii(j9_raw).upper()) if j9_raw else ""

        # CR: referencias citadas en formato legible
        ref_urls = a.get("referenced_works", [])
        refs_resolved = a.get("references_resolved", [])
        cr_list = _resolve_cr_from_cache(
            ref_urls, cache, mem_cache=mem_cache,
            refs_resolved=refs_resolved,
        )
        # Contar resueltos vs fallback
        for cr in cr_list:
            if _RE_WORK_ID_FALLBACK.match(cr):
                cr_fallback_count += 1
            else:
                cr_resolved_count += 1
        cr_str = "; ".join(cr_list)

        # C1: afiliaciones con formato autor-institucion-pais
        c1_str = _format_c1(authorships)

        # RP: autor correspondiente
        rp_str = _format_rp(authorships)

        # AU_CO: todos los paises de los autores; AU1_CO: pais del autor de correspondencia
        au_co_str = _extract_author_countries(authorships)
        au1_co_str = _corresponding_author_country(authorships)

        # SC: subject categories
        sc_str = _extract_subject_categories(a)

        # Keywords — filter empty, use "; " separator for bibliometrix
        # DE = palabras clave de AUTOR (convencion WoS/bibliometrix); antes salia de manual_tags (mezcla)
        manual_kws = [kw.strip() for kw in (a["author_keywords"] if "author_keywords" in a else a.get("manual_tags", [])) if kw.strip()]
        auto_kws = [kw.strip() for kw in a.get("automatic_tags", []) if kw.strip()]
        manual_tags = "; ".join(manual_kws).upper()
        auto_tags = "; ".join(auto_kws).upper()

        # ID fallback: si automatic_tags vacio, generar desde OpenAlex concepts
        if not auto_tags:
            auto_tags = _extract_keywords_plus(a)

        # WC: Web of Science Categories desde OpenAlex topics/concepts
        wc_str = _extract_wos_categories(a)

        # Open access (canonico: diamond cuando APC=0/DOAJ sin APC)
        oa_status = get_canonical_oa_status(a)

        # SO uppercase + normalized for histNetwork matching (CR has uppercase journals)
        so_raw = a.get("publication_title", "")
        so_upper = _normalize_journal_for_cr(so_raw.upper()) if so_raw else ""

        # SR: mismo formato que arma bibliometrix (primer autor de AU, ano, J9) y que usa el CR
        sr_parts = [authors_formatted[0] if authors_formatted else "UNKNOWN", str(a.get("publication_year", ""))]
        if j9_str:
            sr_parts.append(j9_str)
        sr_str = ", ".join(sr_parts)

        row = {
            "AU": au_str,
            "TI": a.get("title", ""),
            "SO": so_upper,
            "JI": a.get("journal_abbrev", ""),
            "J9": j9_str,
            "DT": _normalize_dt(
                a.get("item_type", ""),
                openalex_type=openalex.get("type", ""),
            ),
            "DE": manual_tags,
            "ID": auto_tags,
            "AB": a.get("abstract", ""),
            "C1": c1_str,
            "RP": rp_str,
            "CR": cr_str,
            "TC": a.get("cited_by_count", 0),
            "PY": int(a["publication_year"]) if a.get("publication_year") else "",
            "VL": a.get("volume", ""),
            "IS": a.get("issue", ""),
            "PP": a.get("pages", ""),
            "DI": a.get("doi", ""),
            "SN": a.get("issn", ""),
            "LA": _clean_language(a.get("language", "")),
            "OA": oa_status,
            "URL": a.get("url", ""),
            "DB": "ISI",
            "SR": sr_str,
            "AU_CO": au_co_str,
            "AU1_CO": au1_co_str,
            "WC": wc_str,
            "SC": sc_str,
            # Campos extra
            "KEY": a.get("key", ""),
            "NR": len(ref_urls),
        }
        rows.append(row)

    logger.info(
        f"CR resolution: {cr_resolved_count} resolved, "
        f"{cr_fallback_count} fallback IDs"
    )
    return pd.DataFrame(rows)


def export_bibliometrix_csv(articles: list[dict], output_path: Path = BIBLIOMETRIX_CSV) -> None:
    """Exporta CSV para R/bibliometrix."""
    df = articles_to_bibliometrix_df(articles)
    # Cast numeric columns to object before fillna to avoid FutureWarning
    for col in df.columns:
        if df[col].dtype != object:
            df[col] = df[col].astype(object)
    df.fillna("", inplace=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"Bibliometrix CSV: {output_path} ({len(df)} articles)")
    # biblioshiny ('Load bibliometrix file') solo acepta .xlsx/.RData, no .csv: se escribe tambien el .xlsx
    xlsx_path = output_path.with_suffix(".xlsx")
    largo_max = int(df.astype(str).apply(lambda col: col.str.len().max()).max()) if len(df) else 0
    if largo_max > 32767:
        logger.warning(f"No se escribe {xlsx_path.name}: una celda supera el limite de Excel (32767 caracteres).")
    else:
        df.to_excel(xlsx_path, index=False)
        logger.info(f"Bibliometrix XLSX (para biblioshiny > Load bibliometrix file): {xlsx_path}")


def export_bibtex(articles: list[dict], output_path: Path = BIBLIOMETRIX_BIB) -> None:
    """Exporta BibTeX."""
    entries = []
    for a in articles:
        key = a.get("key", "unknown")
        authors_bib = " and ".join(a.get("authors", []))
        doi = a.get("doi", "")

        entry = f"""@article{{{key},
  author = {{{authors_bib}}},
  title = {{{a.get('title', '')}}},
  journal = {{{a.get('publication_title', '')}}},
  year = {{{a.get('publication_year', '')}}},
  volume = {{{a.get('volume', '')}}},
  number = {{{a.get('issue', '')}}},
  pages = {{{a.get('pages', '')}}},
  doi = {{{doi}}},
  issn = {{{a.get('issn', '')}}},
  abstract = {{{a.get('abstract', '')[:500]}}},
  keywords = {{{'; '.join(a.get('manual_tags', []))}}},
  language = {{{a.get('language', '')}}}
}}"""
        entries.append(entry)

    output_path.write_text("\n\n".join(entries), encoding="utf-8")
    logger.info(f"BibTeX: {output_path} ({len(entries)} entries)")


def export_excel_report(articles: list[dict], output_dir: Path = REPORTS_DIR) -> None:
    """Exporta Excel multi-hoja con resumen completo."""
    output_path = output_dir / "bibliometric_report.xlsx"

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # Hoja 1: Resumen de articulos
        summary_rows = []
        for a in articles:
            oa = a.get("openalex", {})
            pt = oa.get("primary_topic", {}) or {}
            perc = oa.get("cited_by_percentile_year", {}) or {}
            apc = oa.get("apc_paid", {}) or {}
            s2 = a.get("semantic_scholar", {}) or {}
            ext = s2.get("external_ids", {}) or {}

            # Format SDGs
            sdgs = oa.get("sustainable_development_goals", [])
            sdg_str = "; ".join(
                f"{s.get('display_name', '')} ({s.get('score', 0):.2f})"
                for s in sdgs[:5]
            ) if sdgs else ""

            # Format grants
            grants = a.get("funding", [])
            grants_str = "; ".join(
                f.get("name", "") for f in grants if f.get("name")
            ) if grants else ""

            # Format OpenAlex keywords
            oa_kw = oa.get("keywords", [])
            oa_kw_str = "; ".join(
                k.get("keyword", "") for k in oa_kw if k.get("keyword")
            ) if oa_kw else ""

            # Format MeSH terms
            mesh = a.get("mesh_terms", [])
            mesh_str = "; ".join(mesh) if isinstance(mesh, list) else ""

            summary_rows.append({
                "Key": a.get("key", ""),
                "Authors": "; ".join(a.get("authors", [])),
                "Full_Author_Names": a.get("full_author_names", ""),
                "Year": a.get("publication_year", ""),
                "Month": a.get("month", ""),
                "Title": a.get("title", ""),
                "Subtitle": a.get("subtitle", ""),
                "Journal": a.get("publication_title", ""),
                "Journal_Abbrev": a.get("journal_abbrev", ""),
                "Volume": a.get("volume", ""),
                "Issue": a.get("issue", ""),
                "Pages": a.get("pages", ""),
                "Article_Number": a.get("article_number", ""),
                "DOI": a.get("doi", ""),
                "ISBN": a.get("isbn", ""),
                "ISSN": a.get("issn", ""),
                "Publisher": a.get("publisher", ""),
                "Booktitle": a.get("booktitle", ""),
                "Editor": a.get("editor", ""),
                "Language": a.get("language", ""),
                "Document_Type": a.get("item_type", ""),
                "Study_Design": "; ".join(a.get("study_design", []) or []),
                "Citations": a.get("cited_by_count", 0),
                "Citation_Percentile": perc.get("min", perc.get("value", "")),
                "Self_Citation_Rate": a.get("self_citation_rate", ""),
                "Citation_Velocity": a.get("citation_velocity", ""),
                "Num_References": len(a.get("referenced_works", [])),
                "Num_Keywords": len(a.get("manual_tags", [])),
                "Keywords_Author": "; ".join(a.get("manual_tags", [])) if isinstance(a.get("manual_tags"), list) else "",
                "Keywords_OpenAlex": oa_kw_str,
                "MeSH_Terms": mesh_str,
                "Primary_Topic": pt.get("name", ""),
                "Primary_Subfield": pt.get("subfield", ""),
                "Primary_Field": pt.get("field", ""),
                "Primary_Domain": pt.get("domain", ""),
                "SDGs": sdg_str,
                "Grants": grants_str,
                "Funding_Text": a.get("funding_text", ""),
                "Funding_Agency": a.get("funding_agency", ""),
                "APC_Paid_USD": apc.get("value_usd", ""),
                "Open_Access": get_canonical_oa_status(a),
                "OA_URL": a.get("open_access", {}).get("oa_url", "") if isinstance(a.get("open_access"), dict) else "",
                "Source_DB": a.get("source_db", ""),
                "OpenAlex_ID": oa.get("openalex_id", ""),
                "PMID": a.get("pmid", ""),
                "PMC_ID": a.get("pmc_id", ""),
                "ERIC_ID": a.get("eric_id", ""),
                "WoS_Accession": a.get("wos_accession", ""),
                "S2_Paper_ID": s2.get("paper_id", ""),
                "MAG_ID": ext.get("mag", ""),
                "ArXiv_ID": ext.get("arxiv", ""),
                "SJR": a.get("sjr", ""),
                "SJR_Quartile": a.get("sjr_quartile", ""),
                "Altmetric_Score": a.get("altmetric_score", ""),
                "Event_Total": a.get("event_total", ""),
                "Event_Wikipedia": a.get("event_wikipedia", ""),
                "Event_Twitter": a.get("event_twitter", ""),
                "Event_Reddit": a.get("event_reddit", ""),
                "TLDR": a.get("tldr", ""),
                "Institution": a.get("institution", ""),
                "Place_of_Publication": a.get("place_of_publication", ""),
                "Archive": "; ".join(a.get("crossref", {}).get("archive", [])) if isinstance(a.get("crossref", {}).get("archive"), list) else "",
            })
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Articles", index=False)

        # Hoja 2: Keywords
        kw_rows = []
        for a in articles:
            for kw in a.get("manual_tags", []):
                kw_rows.append({"Key": a["key"], "Keyword": kw, "Type": "Author"})
            for kw in a.get("automatic_tags", []):
                kw_rows.append({"Key": a["key"], "Keyword": kw, "Type": "Automatic"})
            for kw_obj in a.get("openalex", {}).get("keywords", []):
                kw_text = kw_obj.get("keyword", "")
                if kw_text:
                    kw_rows.append({"Key": a["key"], "Keyword": kw_text, "Type": "OpenAlex"})
            for mesh in a.get("mesh_terms", []):
                if mesh:
                    kw_rows.append({"Key": a["key"], "Keyword": mesh, "Type": "MeSH"})
        if kw_rows:
            pd.DataFrame(kw_rows).to_excel(writer, sheet_name="Keywords", index=False)

        # Hoja 3: Afiliaciones
        aff_rows = []
        for a in articles:
            for auth in a.get("openalex", {}).get("authorships", []):
                for inst in auth.get("institutions", []):
                    aff_rows.append({
                        "Article_Key": a["key"],
                        "Author": auth.get("name", ""),
                        "Institution": inst.get("name", ""),
                        "Country": inst.get("country", ""),
                    })
        if aff_rows:
            pd.DataFrame(aff_rows).to_excel(writer, sheet_name="Affiliations", index=False)

        # Hoja 4: Future research & Limitations
        fl_rows = []
        for a in articles:
            for s in a.get("future_research_sentences", []):
                fl_rows.append({"Key": a["key"], "Type": "Future_Research", "Sentence": s[:300]})
            for s in a.get("limitation_sentences", []):
                fl_rows.append({"Key": a["key"], "Type": "Limitation", "Sentence": s[:300]})
        if fl_rows:
            pd.DataFrame(fl_rows).to_excel(writer, sheet_name="Future_Limitations", index=False)

        # Hoja 6: Bridge Articles (from Phase 3 structural holes analysis)
        try:
            import json
            from config import NETWORKS_DIR
            bridge_path = NETWORKS_DIR / "bridge_articles_report.json"
            if bridge_path.exists():
                bridge_data = json.loads(bridge_path.read_text(encoding="utf-8"))
                bridge_rows = []
                for ba in bridge_data.get("bridge_articles", []):
                    bridge_rows.append({
                        "Key": ba.get("key", ""),
                        "Title": ba.get("title", ""),
                        "Year": ba.get("year", ""),
                        "Citations": ba.get("citations", 0),
                        "Betweenness": ba.get("betweenness", 0),
                        "Community": ba.get("community", ""),
                        "Communities_Bridged": ba.get("n_communities_bridged", 0),
                        "Connected_Communities": str(ba.get("connected_communities", [])),
                        "Interdisciplinarity": ba.get("interdisciplinarity_score", 0),
                        "Burt_Constraint": ba.get("burt_constraint", ""),
                    })
                if bridge_rows:
                    pd.DataFrame(bridge_rows).to_excel(
                        writer, sheet_name="Bridge_Articles", index=False
                    )
        except Exception as e:
            logger.debug(f"Bridge articles sheet skipped: {e}")

        # Hoja 7: Funding (grants from all sources)
        fund_rows = []
        for a in articles:
            for f in a.get("funding", []):
                fund_rows.append({
                    "Article_Key": a["key"],
                    "Funder": f.get("name", ""),
                    "Award_ID": f.get("award_id", ""),
                    "Source": f.get("source", "crossref"),
                })
        if fund_rows:
            pd.DataFrame(fund_rows).to_excel(writer, sheet_name="Funding", index=False)

        # Hoja 8: SDGs (Sustainable Development Goals)
        sdg_rows = []
        for a in articles:
            for sdg in a.get("openalex", {}).get("sustainable_development_goals", []):
                sdg_rows.append({
                    "Article_Key": a["key"],
                    "SDG": sdg.get("display_name", ""),
                    "SDG_ID": sdg.get("id", ""),
                    "Score": sdg.get("score", 0),
                })
        if sdg_rows:
            pd.DataFrame(sdg_rows).to_excel(writer, sheet_name="SDGs", index=False)

        # Hoja 9: External IDs (cross-reference identifiers)
        eid_rows = []
        for a in articles:
            ids = {}
            ids["Article_Key"] = a["key"]
            ids["DOI"] = a.get("doi", "")
            ids["OpenAlex_ID"] = a.get("openalex", {}).get("openalex_id", "")
            ids["PMID"] = a.get("pmid", "")
            ids["PMC_ID"] = a.get("pmc_id", "")
            ids["ERIC_ID"] = a.get("eric_id", "")
            ids["WoS_Accession"] = a.get("wos_accession", "")
            s2_ext = a.get("semantic_scholar", {}).get("external_ids", {}) or {}
            ids["S2_Paper_ID"] = a.get("semantic_scholar", {}).get("paper_id", "")
            ids["MAG_ID"] = s2_ext.get("mag", "")
            ids["DBLP_ID"] = s2_ext.get("dblp", "")
            ids["ArXiv_ID"] = s2_ext.get("arxiv", "")
            ids["ACL_ID"] = s2_ext.get("acl", "")
            ids["S2_Corpus_ID"] = s2_ext.get("corpus_id", "")
            # Only add if has at least one non-DOI ID
            if any(v for k, v in ids.items() if k not in ("Article_Key", "DOI")):
                eid_rows.append(ids)
        if eid_rows:
            pd.DataFrame(eid_rows).to_excel(writer, sheet_name="External_IDs", index=False)

    logger.info(f"Excel report: {output_path}")


def generate_integrated_report(articles: list[dict], output_dir: Path = REPORTS_DIR) -> None:
    """Genera reporte Markdown integrado."""
    lines = [
        "# Integrated Bibliometric Analysis Report",
        f"\n**Corpus size**: {len(articles)} articles\n",
    ]

    # Resumen general
    years = [a["publication_year"] for a in articles if a.get("publication_year")]
    total_cites = sum(a.get("cited_by_count", 0) for a in articles)
    with_oa = sum(1 for a in articles if get_canonical_oa_status(a) != "closed")

    lines.append("## Summary Statistics")
    lines.append(f"- Year range: {min(years)}-{max(years)}" if years else "")
    lines.append(f"- Total citations: {total_cites}")
    lines.append(f"- Mean citations: {total_cites / len(articles):.1f}")
    lines.append(f"- Open Access articles: {with_oa}")
    lines.append("")

    # Top cited
    lines.append("## Most Cited Articles")
    sorted_articles = sorted(articles, key=lambda x: x.get("cited_by_count", 0), reverse=True)
    for a in sorted_articles[:10]:
        lines.append(f"- **{a.get('cited_by_count', 0)}** cites: {a['title'][:80]} ({a.get('publication_year', '')})")
    lines.append("")

    # Future research
    future_all = [(a["key"], s) for a in articles for s in a.get("future_research_sentences", [])]
    if future_all:
        lines.append(f"## Future Research Directions ({len(future_all)} sentences)")
        for key, sentence in future_all[:20]:
            lines.append(f"- [{key}] {sentence[:200]}")
        lines.append("")

    # Limitations
    limit_all = [(a["key"], s) for a in articles for s in a.get("limitation_sentences", [])]
    if limit_all:
        lines.append(f"## Reported Limitations ({len(limit_all)} sentences)")
        for key, sentence in limit_all[:20]:
            lines.append(f"- [{key}] {sentence[:200]}")
        lines.append("")

    report_path = output_dir / "integrated_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Integrated report: {report_path}")


def generate_r_analysis_script(output_dir: Path = OUTPUT_DIR) -> None:
    """Genera script R comprehensivo para bibliometrix."""
    script = r'''# =============================================================================
# Bibliometrix Comprehensive Analysis Script
# Generated by the Bibliometric Analysis Pipeline
# =============================================================================
# Install if needed: install.packages("bibliometrix")

library(bibliometrix)

# --- Configuration ---
# Resolve paths relative to this script's location (output/)
script_dir <- tryCatch(
  dirname(sys.frame(1)$ofile),
  error = function(e) {
    if (requireNamespace("rstudioapi", quietly = TRUE) &&
        rstudioapi::isAvailable()) {
      dirname(rstudioapi::getSourceEditorContext()$path)
    } else {
      getwd()
    }
  }
)
csv_path <- file.path(script_dir, "bibliometrix_export.csv")
output_dir <- file.path(script_dir, "figures")
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

if (!file.exists(csv_path)) {
  alt <- file.path(getwd(), "output", "bibliometrix_export.csv")
  if (file.exists(alt)) {
    csv_path <- alt
    output_dir <- file.path(getwd(), "output", "figures")
    dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  } else {
    stop("bibliometrix_export.csv not found. Run 'python main.py --phases 4' first.")
  }
}

cat("=== Bibliometrix Comprehensive Analysis ===\n")
cat("CSV:", csv_path, "\n")
cat("Output:", output_dir, "\n\n")

# --- 1. Load & Preprocess Data ---
cat("1. Loading data...\n")
M <- read.csv(csv_path, stringsAsFactors = FALSE)

# Replace NA with empty strings in ALL character columns
for (col in colnames(M)) {
  if (is.character(M[[col]])) {
    M[[col]][is.na(M[[col]])] <- ""
  }
}

# Ensure required fields
if (!"DB" %in% colnames(M)) M$DB <- "ISI"
M$DB[M$DB == ""] <- "ISI"
M$DT[M$DT == "" | is.na(M$DT)] <- "ARTICLE"

# Remove auxiliary columns that collide with bibliometrix internals
if ("AU1_CO" %in% colnames(M)) M$AU1_CO <- NULL

# Mark truly empty DE/ID as NA so strsplit doesn't produce phantom "" keywords
M$DE[M$DE == ""] <- NA
M$ID[M$ID == ""] <- NA
M$CR[M$CR == ""] <- NA

# Normalize separators: ensure "; " (semicolon + space)
for (col in c("DE", "ID", "CR", "C1")) {
  if (col %in% colnames(M)) {
    M[[col]] <- gsub("\\s*;\\s*", "; ", M[[col]])
  }
}

# Extract AU_CO from C1 (country = last token after final comma)
M <- metaTagExtraction(M, Field = "AU_CO", sep = ";")
if ("AU_CO" %in% colnames(M)) {
  M$AU_CO[is.na(M$AU_CO)] <- ""
}

# Also extract first author field (AU_UN = university, SR = short reference)
tryCatch({
  M <- metaTagExtraction(M, Field = "AU_UN", sep = ";")
  M <- metaTagExtraction(M, Field = "SR", sep = ";")
}, error = function(e) {})

cat("  Articles loaded:", nrow(M), "\n")
cat("  Year range:", min(M$PY, na.rm=TRUE), "-", max(M$PY, na.rm=TRUE), "\n")
cat("  Articles with keywords (DE):", sum(!is.na(M$DE)), "/", nrow(M), "\n")
cat("  Articles with references (CR):", sum(!is.na(M$CR)), "/", nrow(M), "\n\n")

# --- 2. Basic Bibliometric Analysis ---
cat("2. Running biblioAnalysis...\n")
tryCatch({
  results <- biblioAnalysis(M, sep = ";")
  pdf(file.path(output_dir, "01_basic_analysis.pdf"), width = 12, height = 8)
  S <- summary(results, k = 15, pause = FALSE)
  plot(x = results, k = 15, pause = FALSE)
  dev.off()
  cat("  Saved: 01_basic_analysis.pdf\n\n")
}, error = function(e) cat("  Error in basic analysis:", e$message, "\n\n"))

# --- 3. Co-citation Network ---
cat("3. Building co-citation network...\n")
tryCatch({
  NetMatrix_cocit <- biblioNetwork(M, analysis = "co-citation",
                                    network = "references", sep = ";")
  if (!is.null(NetMatrix_cocit) && nrow(NetMatrix_cocit) > 3) {
    n_plot <- min(30, nrow(NetMatrix_cocit))
    pdf(file.path(output_dir, "02_cocitation_network.pdf"), width = 14, height = 12)
    net_cocit <- networkPlot(NetMatrix_cocit, n = n_plot, type = "fruchterman",
                              Title = "Co-Citation Network",
                              size.cex = TRUE, size = 15,
                              label.cex = TRUE, labelsize = 1.2,
                              cluster = "louvain", edges.min = 2)
    dev.off()
    cat("  Saved: 02_cocitation_network.pdf\n")
  } else {
    cat("  Co-citation matrix too small, skipping.\n")
  }
}, error = function(e) cat("  Error in co-citation:", e$message, "\n"))

# --- 4. Keyword Co-occurrence Network ---
cat("4. Building keyword co-occurrence network...\n")
tryCatch({
  # Only use articles that have keywords
  M_kw <- M[!is.na(M$DE) & M$DE != "", ]
  if (nrow(M_kw) >= 5) {
    NetMatrix_kw <- biblioNetwork(M_kw, analysis = "co-occurrences",
                                   network = "keywords", sep = ";")
    if (!is.null(NetMatrix_kw) && nrow(NetMatrix_kw) > 3) {
      n_plot <- min(30, nrow(NetMatrix_kw))
      pdf(file.path(output_dir, "03_keyword_cooccurrence.pdf"), width = 14, height = 12)
      net_kw <- networkPlot(NetMatrix_kw, n = n_plot, type = "fruchterman",
                             Title = "Keyword Co-occurrence Network",
                             size.cex = TRUE, size = 12,
                             label.cex = TRUE, labelsize = 1.0,
                             cluster = "louvain", edges.min = 2)
      dev.off()
      cat("  Saved: 03_keyword_cooccurrence.pdf\n")
    } else {
      cat("  Keyword matrix too small.\n")
    }
  } else {
    cat("  Not enough articles with keywords.\n")
  }
}, error = function(e) cat("  Error in keyword network:", e$message, "\n"))

# --- 5. Bibliographic Coupling ---
cat("5. Building bibliographic coupling network...\n")
tryCatch({
  NetMatrix_coup <- biblioNetwork(M, analysis = "coupling",
                                   network = "references", sep = ";")
  if (!is.null(NetMatrix_coup) && nrow(NetMatrix_coup) > 3) {
    n_plot <- min(30, nrow(NetMatrix_coup))
    pdf(file.path(output_dir, "04_bibliographic_coupling.pdf"), width = 14, height = 12)
    net_coup <- networkPlot(NetMatrix_coup, n = n_plot, type = "fruchterman",
                             Title = "Bibliographic Coupling Network",
                             size.cex = TRUE, size = 15, cluster = "louvain")
    dev.off()
    cat("  Saved: 04_bibliographic_coupling.pdf\n")
  } else {
    cat("  Coupling matrix too small.\n")
  }
}, error = function(e) cat("  Error in coupling:", e$message, "\n"))

# --- 6. Thematic Map (Strategic Diagram) ---
cat("6. Creating thematic map...\n")
tryCatch({
  M_tm <- M[!is.na(M$DE) & M$DE != "", ]
  if (nrow(M_tm) >= 10) {
    Map <- thematicMap(M_tm, field = "DE", n = 200, minfreq = 3,
                       stemming = FALSE, size = 0.5, repel = TRUE)
    pdf(file.path(output_dir, "05_thematic_map.pdf"), width = 14, height = 10)
    print(Map$map)
    dev.off()
    cat("  Saved: 05_thematic_map.pdf\n")
  } else {
    cat("  Not enough articles with keywords for thematic map.\n")
  }
}, error = function(e) cat("  Error in thematic map:", e$message, "\n"))

# --- 7. Thematic Evolution ---
cat("7. Creating thematic evolution...\n")
tryCatch({
  M_te <- M[!is.na(M$DE) & M$DE != "", ]
  years <- sort(unique(M_te$PY))
  if (length(years) >= 4) {
    # Build explicit year breaks ensuring uniqueness
    yr_min <- min(years)
    yr_max <- max(years)
    yr_mid <- floor(median(years))
    # Ensure 3 distinct breaks for 2 time slices
    cuts <- unique(c(yr_min, yr_mid, yr_max))
    if (length(cuts) >= 3) {
      nexus <- thematicEvolution(M_te, field = "DE", years = cuts,
                                  n = 200, minFreq = 2)
      pdf(file.path(output_dir, "06_thematic_evolution.pdf"), width = 16, height = 10)
      plotThematicEvolution(nexus$Nodes, nexus$Edges)
      dev.off()
      cat("  Saved: 06_thematic_evolution.pdf\n")
    } else {
      cat("  Not enough distinct year breaks.\n")
    }
  } else {
    cat("  Not enough year range for thematic evolution.\n")
  }
}, error = function(e) cat("  Error in thematic evolution:", e$message, "\n"))

# --- 8. Historiograph ---
cat("8. Creating historiograph...\n")
tryCatch({
  histResults <- histNetwork(M, min.citations = 1, sep = ";")
  if (!is.null(histResults$NetMatrix) &&
      is.matrix(histResults$NetMatrix) &&
      nrow(histResults$NetMatrix) > 1) {
    pdf(file.path(output_dir, "07_historiograph.pdf"), width = 16, height = 10)
    net_hist <- histPlot(histResults, n = 25, size = 5, labelsize = 3)
    dev.off()
    cat("  Saved: 07_historiograph.pdf\n")
  } else {
    cat("  Historiograph: not enough citation matches in corpus.\n")
    cat("  (histNetwork needs CR entries to match 'AUTHOR, YEAR, JOURNAL' of corpus articles)\n")
  }
}, error = function(e) cat("  Error in historiograph:", e$message, "\n"))

# --- 9. Country Collaboration Network ---
cat("9. Building country collaboration network...\n")
tryCatch({
  NetMatrix_country <- biblioNetwork(M, analysis = "collaboration",
                                      network = "countries", sep = ";")
  if (!is.null(NetMatrix_country) && nrow(NetMatrix_country) > 2) {
    n_plot <- min(30, nrow(NetMatrix_country))
    pdf(file.path(output_dir, "08_country_collaboration.pdf"), width = 14, height = 12)
    net_country <- networkPlot(NetMatrix_country, n = n_plot, type = "circle",
                                Title = "Country Collaboration Network",
                                size.cex = TRUE, size = 15,
                                cluster = "louvain",
                                label.cex = TRUE, labelsize = 1.5)
    dev.off()
    cat("  Saved: 08_country_collaboration.pdf\n")
  } else {
    cat("  Country network too small.\n")
  }
}, error = function(e) cat("  Error in country network:", e$message, "\n"))

# --- 10. Conceptual Structure (MCA) ---
cat("10. Conceptual structure analysis...\n")
tryCatch({
  M_cs <- M[!is.na(M$DE) & M$DE != "", ]
  if (nrow(M_cs) >= 10) {
    pdf(file.path(output_dir, "09_conceptual_structure.pdf"), width = 14, height = 10)
    CS <- conceptualStructure(M_cs, field = "DE", method = "MCA",
                               minDegree = 3, clust = "auto",
                               k.max = 8, stemming = FALSE,
                               labelsize = 10, documents = 15)
    dev.off()
    cat("  Saved: 09_conceptual_structure.pdf\n")
  } else {
    cat("  Not enough articles with keywords for conceptual structure.\n")
  }
}, error = function(e) {
  dev.off()
  cat("  Error in conceptual structure:", e$message, "\n")
})

# --- 11. Three Fields Plot (Sankey AU-DE-SO) ---
cat("11. Creating three fields plot...\n")
tryCatch({
  M_tf <- M[!is.na(M$DE) & M$DE != "", ]
  if (nrow(M_tf) >= 10) {
    pdf(file.path(output_dir, "10_three_fields_plot.pdf"), width = 16, height = 10)
    threeFieldsPlot(M_tf, fields = c("AU", "DE", "SO"), n = c(15, 15, 15))
    dev.off()
    cat("  Saved: 10_three_fields_plot.pdf\n")
  } else {
    cat("  Not enough articles with keywords for three fields plot.\n")
  }
}, error = function(e) {
  dev.off()
  cat("  Error in three fields plot:", e$message, "\n")
})

# --- 12. Author Collaboration Network ---
cat("12. Building author collaboration network...\n")
tryCatch({
  NetMatrix_auth <- biblioNetwork(M, analysis = "collaboration",
                                   network = "authors", sep = ";")
  if (!is.null(NetMatrix_auth) && nrow(NetMatrix_auth) > 3) {
    n_plot <- min(30, nrow(NetMatrix_auth))
    pdf(file.path(output_dir, "11_author_collaboration.pdf"), width = 14, height = 12)
    net_auth <- networkPlot(NetMatrix_auth, n = n_plot, type = "fruchterman",
                             Title = "Author Collaboration Network",
                             size.cex = TRUE, size = 12,
                             cluster = "louvain",
                             label.cex = TRUE, labelsize = 1.0)
    dev.off()
    cat("  Saved: 11_author_collaboration.pdf\n")
  } else {
    cat("  Author network too small.\n")
  }
}, error = function(e) cat("  Error in author network:", e$message, "\n"))

# --- 12b. Institution Collaboration Network ---
cat("12b. Building institution collaboration network...\n")
tryCatch({
  NetMatrix_inst <- biblioNetwork(M, analysis = "collaboration",
                                   network = "universities", sep = ";")
  if (!is.null(NetMatrix_inst) && nrow(NetMatrix_inst) > 3) {
    n_plot <- min(30, nrow(NetMatrix_inst))
    pdf(file.path(output_dir, "12b_institution_collaboration.pdf"), width = 14, height = 12)
    net_inst <- networkPlot(NetMatrix_inst, n = n_plot, type = "fruchterman",
                             Title = "Institution Collaboration Network",
                             size.cex = TRUE, size = 12,
                             cluster = "louvain",
                             label.cex = TRUE, labelsize = 0.9)
    dev.off()
    cat("  Saved: 12b_institution_collaboration.pdf\n")
  } else {
    cat("  Institution network too small.\n")
  }
}, error = function(e) cat("  Error in institution network:", e$message, "\n"))

# --- 12c. Bibliographic Coupling by Sources (Journals) ---
cat("12c. Building bibliographic coupling by sources...\n")
tryCatch({
  NetMatrix_coup_src <- biblioNetwork(M, analysis = "coupling",
                                       network = "sources", sep = ";")
  if (!is.null(NetMatrix_coup_src) && nrow(NetMatrix_coup_src) > 3) {
    n_plot <- min(30, nrow(NetMatrix_coup_src))
    pdf(file.path(output_dir, "12c_coupling_sources.pdf"), width = 14, height = 12)
    net_coup_src <- networkPlot(NetMatrix_coup_src, n = n_plot, type = "fruchterman",
                                 Title = "Bibliographic Coupling (Sources/Journals)",
                                 size.cex = TRUE, size = 15, cluster = "louvain")
    dev.off()
    cat("  Saved: 12c_coupling_sources.pdf\n")
  } else {
    cat("  Coupling by sources matrix too small.\n")
  }
}, error = function(e) cat("  Error in coupling by sources:", e$message, "\n"))

# --- 12d. Bibliographic Coupling by Countries ---
cat("12d. Building bibliographic coupling by countries...\n")
tryCatch({
  NetMatrix_coup_co <- biblioNetwork(M, analysis = "coupling",
                                      network = "countries", sep = ";")
  if (!is.null(NetMatrix_coup_co) && nrow(NetMatrix_coup_co) > 2) {
    n_plot <- min(30, nrow(NetMatrix_coup_co))
    pdf(file.path(output_dir, "12d_coupling_countries.pdf"), width = 14, height = 12)
    net_coup_co <- networkPlot(NetMatrix_coup_co, n = n_plot, type = "circle",
                                Title = "Bibliographic Coupling (Countries)",
                                size.cex = TRUE, size = 15, cluster = "louvain",
                                label.cex = TRUE, labelsize = 1.3)
    dev.off()
    cat("  Saved: 12d_coupling_countries.pdf\n")
  } else {
    cat("  Coupling by countries matrix too small.\n")
  }
}, error = function(e) cat("  Error in coupling by countries:", e$message, "\n"))

# --- 12e. Co-citation by Sources (Journals) ---
cat("12e. Building co-citation by sources...\n")
tryCatch({
  NetMatrix_cocit_src <- biblioNetwork(M, analysis = "co-citation",
                                        network = "sources", sep = ";")
  if (!is.null(NetMatrix_cocit_src) && nrow(NetMatrix_cocit_src) > 3) {
    n_plot <- min(30, nrow(NetMatrix_cocit_src))
    pdf(file.path(output_dir, "12e_cocitation_sources.pdf"), width = 14, height = 12)
    net_cocit_src <- networkPlot(NetMatrix_cocit_src, n = n_plot, type = "fruchterman",
                                  Title = "Co-citation Network (Sources/Journals)",
                                  size.cex = TRUE, size = 15,
                                  label.cex = TRUE, labelsize = 1.1,
                                  cluster = "louvain", edges.min = 2)
    dev.off()
    cat("  Saved: 12e_cocitation_sources.pdf\n")
  } else {
    cat("  Co-citation by sources matrix too small.\n")
  }
}, error = function(e) cat("  Error in co-citation by sources:", e$message, "\n"))

# --- 13. Most Cited References ---
cat("13. Most cited references...\n")
tryCatch({
  CR <- citations(M, field = "article", sep = ";")
  if (!is.null(CR) && is.data.frame(CR) && nrow(CR) > 0 &&
      "Freq" %in% colnames(CR) && any(CR$Freq > 0)) {
    n_top <- min(20, nrow(CR))
    pdf(file.path(output_dir, "12_most_cited_references.pdf"), width = 14, height = 8)
    par(mar = c(5, 18, 4, 2))
    barplot(CR$Freq[1:n_top],
            names.arg = substr(CR$Cited[1:n_top], 1, 45),
            horiz = TRUE, las = 1, cex.names = 0.65,
            main = "Most Cited References in the Corpus",
            xlab = "Citations", col = "#4e79a7")
    dev.off()
    cat("  Saved: 12_most_cited_references.pdf\n")
  } else {
    cat("  No valid citation data for barplot.\n")
  }
}, error = function(e) cat("  Error in cited references:", e$message, "\n"))

# --- Summary ---
cat("\n=== Analysis Complete ===\n")
cat("Output files saved to:", output_dir, "\n")
cat("Generated PDFs:\n")
pdfs <- list.files(output_dir, pattern = "\\.pdf$", full.names = FALSE)
for (p in pdfs) cat("  -", p, "\n")

# --- Optional: Launch Biblioshiny ---
# Uncomment the line below to launch the interactive Biblioshiny app:
# biblioshiny()

cat("\nTo launch interactive analysis, run: biblioshiny()\n")
'''
    script_path = output_dir / "bibliometrix_analysis.R"
    script_path.write_text(script, encoding="utf-8")
    logger.info(f"R analysis script: {script_path}")


def run_phase4(articles: list[dict]) -> None:
    """Ejecuta la Fase 4 completa."""
    logger.info("=== FASE 4: Exportacion e Integracion ===")

    export_bibliometrix_csv(articles)
    export_bibtex(articles)
    export_excel_report(articles)
    generate_integrated_report(articles)
    generate_r_analysis_script()

    logger.info("Fase 4 completada.")

run_phase5 = run_phase4
