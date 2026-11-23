"""Fase 0: Carga y parseo del CSV Zotero."""
from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path
from typing import Optional

import pandas as pd

from config import CSV_PATH, CSV_SEPARATOR, CSV_ENCODING, BASE_DIR
from src.utils import (
    parse_author_string,
    normalize_doi,
    normalize_language,
    normalize_journal_name,
    canonicalize_corpus_journals,
)

logger = logging.getLogger("bibliometrico.data_loader")


def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia sufijos _x/_y de merges de pandas en nombres de columna.

    Si una columna con sufijo (ej. 'Publication Year_y') no tiene contraparte
    sin sufijo, se renombra al nombre base ('Publication Year').
    """
    import re
    renames = {}
    for col in df.columns:
        m = re.match(r'^(.+?)(_x|_y)$', col)
        if m:
            base = m.group(1)
            # Solo renombrar si la columna base no existe ya
            if base not in df.columns:
                renames[col] = base
    if renames:
        logger.info(f"Columnas renombradas (limpieza de merge): {renames}")
        df = df.rename(columns=renames)
    return df


# Mapeo: nombre en CSV consolidado -> nombre Zotero esperado por el pipeline
_CONSOLIDADO_COL_MAP = {
    "key": "Key",
    "Authors": "Author",
    "Publication_Year": "Publication Year",
    "Year": "Publication Year",
    "Abstract": "Abstract Note",
    "Journal_Name": "Publication Title",
    "Source_Title": "Publication Title",
    "Document_Type": "Item Type",
    "URL": "Url",
    "Link": "Url",
    "Keywords": "Manual Tags",
    "Author_Keywords": "Manual Tags",
    "Index_Keywords": "Automatic Tags",
    "Journal_ISO": "Journal Abbreviation",
    "Times_Cited": "Times Cited",
    "Cited_By": "Times Cited",
    "Funding_Text": "Funding Text",
    "Funding": "Funding Text",
    "Funding_Agency": "Funding Agency",
    "Source_DB": "Source DB",
    "Merged_From": "Merged From",
    "Richness_Score": "Richness Score",
    "WoS_Accession": "WoS Accession",
    "WoS_Index": "WoS Index",
    "ERIC_ID": "ERIC ID",
    "Database_ID": "Database ID",
    "Article_Number": "Article Number",
    "Booktitle": "Book Title",
    "Full_Author_Names": "Full Author Names",
    "Author_Full_Names": "Full Author Names",
    "Place_of_Publication": "Place of Publication",
    "MeSH_Terms": "MeSH Terms",
    "PMC_ID": "PMC ID",
    "Open_Access": "Open Access",
    "References": "References",
    "CR": "References",
    "Cited_References": "References",
    "Cited References": "References",
}


# Mapeo del VEHICULO de publicacion a categorias canónicas (claves en minúsculas).
# item_type responde "donde se publico", no "como se estudio": el diseño del
# estudio (RCT, observacional) viaja aparte en study_design, porque PubMed mete
# ambos ejes en el mismo campo ("Journal Article; Randomized Controlled Trial").
_DOC_TYPE_MAP = {
    "jour": "Journal Article",
    "article": "Journal Article",
    "journal article": "Journal Article",
    "journal articles": "Journal Article",
    "journalarticle": "Journal Article",
    "conference paper": "Conference Paper",
    "conference proceeding": "Conference Paper",
    "proceedings paper": "Conference Paper",
    "inproceedings": "Conference Paper",
    "conferencepaper": "Conference Paper",
    "speeches/meeting papers": "Conference Paper",
    "book chapter": "Book Chapter",
    "book section": "Book Chapter",
    "booksection": "Book Chapter",
    "incollection": "Book Chapter",
    "reports - research": "Other",
}


# Diseños de estudio reconocidos (lista blanca). Los terminos de financiamiento
# y de soporte ("Research Support, Non-U.S. Gov't") o de formato
# ("Video-Audio Media") NO son diseños y quedan fuera a proposito.
_STUDY_DESIGN_MAP = {
    "randomized controlled trial": "Randomized Controlled Trial",
    "controlled clinical trial": "Controlled Clinical Trial",
    "pragmatic clinical trial": "Pragmatic Clinical Trial",
    "equivalence trial": "Equivalence Trial",
    "clinical trial": "Clinical Trial",
    "clinical study": "Clinical Study",
    "observational study": "Observational Study",
    "validation study": "Validation Study",
    "comparative study": "Comparative Study",
    "multicenter study": "Multicenter Study",
    "evaluation study": "Evaluation Study",
    "case reports": "Case Report",
    "meta-analysis": "Meta-Analysis",
    "systematic review": "Systematic Review",
    "scoping review": "Scoping Review",
    "review": "Review",
}


def normalize_item_type(raw: str) -> str:
    """Normaliza el tipo de documento al vehiculo de publicacion canónico.

    Acepta campos compuestos separados por ';' (estilo PubMed) y devuelve el
    primer vehiculo reconocido. Si no reconoce ninguno, devuelve el valor
    original sin tocar: no se inventa una categoria.
    """
    s = str(raw or "").strip()
    if not s:
        return ""
    directo = _DOC_TYPE_MAP.get(s.lower())
    if directo:
        return directo
    for tok in s.split(";"):
        v = _DOC_TYPE_MAP.get(tok.strip().lower())
        if v:
            return v
    return s


def extract_study_design(raw: str) -> list[str]:
    """Extrae los diseños de estudio declarados en un campo de tipo compuesto."""
    s = str(raw or "").strip()
    if not s:
        return []
    designs: list[str] = []
    for tok in s.split(";"):
        d = _STUDY_DESIGN_MAP.get(tok.strip().lower())
        if d and d not in designs:
            designs.append(d)
    return designs


def _log_type_token_coverage(df: pd.DataFrame) -> None:
    """Reporta los tokens de 'Item Type' que no son vehiculo ni diseño conocido.

    Evita que el colapso de tipos descarte informacion en silencio.
    """
    if "Item Type" not in df.columns:
        return
    desconocidos: Counter = Counter()
    for raw in df["Item Type"]:
        if pd.isna(raw):
            continue
        for tok in str(raw).split(";"):
            t = tok.strip()
            if not t:
                continue
            tl = t.lower()
            if tl in _DOC_TYPE_MAP or tl in _STUDY_DESIGN_MAP:
                continue
            if tl.startswith("research support") or tl in _TYPE_TOKENS_IGNORADOS:
                continue
            desconocidos[t] += 1
    if desconocidos:
        logger.warning(
            "Tokens de 'Item Type' sin clasificar (ni vehiculo ni diseño): %s",
            dict(desconocidos.most_common(10)),
        )


# Tokens de 'Item Type' que no aportan ni vehiculo ni diseño y se ignoran
# sin advertencia (formato, idioma, soporte editorial).
_TYPE_TOKENS_IGNORADOS = {
    "video-audio media",
    "english abstract",
    "historical article",
    "introductory journal article",
    "published erratum",
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres de columna del CSV consolidado al formato Zotero."""
    # Las palabras clave DE AUTOR se conservan aparte: 'Keywords' y 'Author_Keywords' se mapean ambas a
    # 'Manual Tags' y gana la primera, que en los consolidados es una mezcla de autor + Keywords Plus.
    if "Author_Keywords" in df.columns and "__author_keywords" not in df.columns:
        df["__author_keywords"] = df["Author_Keywords"]
    elif "Author_Keywords" not in df.columns and "__author_keywords" not in df.columns:
        logging.getLogger("bibliometrico.data_loader").warning(
            "El CSV no trae la columna Author_Keywords: los articulos quedaran sin palabras clave de autor y los "
            "analisis de palabras clave (fases 2e, 2h, 3, 3b, 7) no tendran datos. Exporte consolidado.csv con esa columna.")
    renames = {}
    assigned = set(df.columns)
    for src, dst in _CONSOLIDADO_COL_MAP.items():
        if src in df.columns and dst not in assigned:
            renames[src] = dst
            assigned.add(dst)
    if renames:
        logger.info(f"Columnas mapeadas (consolidado -> Zotero): {renames}")
        df = df.rename(columns=renames)

    # Sintetizar columna "Pages" desde Page_Start / Page_End si no existe
    if "Pages" not in df.columns and ("Page_Start" in df.columns or "Page_End" in df.columns):
        def _merge_pages(row):
            ps = row.get("Page_Start", "")
            pe = row.get("Page_End", "")
            ps = str(ps).strip() if pd.notna(ps) else ""
            pe = str(pe).strip() if pd.notna(pe) else ""
            if ps and pe and ps != pe:
                return f"{ps}-{pe}"
            return ps or pe
        df["Pages"] = df.apply(_merge_pages, axis=1)
        logger.info("Columna 'Pages' sintetizada desde Page_Start/Page_End")

    return df


def load_csv(csv_path: Path = CSV_PATH) -> pd.DataFrame:
    """Carga el CSV (Zotero o consolidado) con autodeteccion de separador."""
    logger.info(f"Cargando CSV: {csv_path}")
    df = pd.read_csv(csv_path, sep=CSV_SEPARATOR, encoding=CSV_ENCODING)
    if len(df.columns) <= 1:
        for alt_sep in (",", "\t", "|"):
            df_try = pd.read_csv(csv_path, sep=alt_sep, encoding=CSV_ENCODING)
            if len(df_try.columns) > 1:
                logger.info(f"Separador autodetectado: {repr(alt_sep)}")
                df = df_try
                break
    df = _clean_column_names(df)
    df = _normalize_columns(df)
    logger.info(f"Articulos cargados: {len(df)}")
    return df


def parse_tags(tag_string: str) -> list[str]:
    """Parsea tags separados por '; '."""
    if not tag_string or pd.isna(tag_string):
        return []
    return [t.strip() for t in str(tag_string).split("; ") if t.strip()]


def parse_extra_field(extra: str) -> dict:
    """Extrae informacion del campo Extra de Zotero.
    Ejemplo: '<p>Export Date: 21 January 2026; Cited By: 1; Correspondence Address: ...'
    """
    info = {}
    if not extra or pd.isna(extra):
        return info
    extra = str(extra)
    # Remove HTML tags
    import re
    extra_clean = re.sub(r"<[^>]+>", "", extra)

    # Cited By
    match = re.search(r"Cited By:\s*(\d+)", extra_clean)
    if match:
        info["scopus_cited_by"] = int(match.group(1))

    # Correspondence Address
    match = re.search(r"Correspondence Address:\s*(.+?)(?:;|$)", extra_clean)
    if match:
        info["correspondence"] = match.group(1).strip()

    # Export Date
    match = re.search(r"Export Date:\s*(.+?)(?:;|$)", extra_clean)
    if match:
        info["export_date"] = match.group(1).strip()

    return info


def find_pdf_path(file_attachments: str) -> Optional[str]:
    """Extrae y normaliza la ruta del PDF del campo File Attachments.
    - Soporta multiples adjuntos separados por ';'
    - Limpia comillas envolventes
    - Resuelve rutas relativas respecto a BASE_DIR
    - Devuelve ruta en formato POSIX (portable)
    """
    if not file_attachments or pd.isna(file_attachments):
        return None
    path_str = str(file_attachments).strip().strip('"').strip("'")
    candidates = []
    parts = [p.strip() for p in path_str.split(";") if p.strip()]
    for part in parts:
        if part.lower().endswith(".pdf"):
            candidates.append(part)
    # Si no se separó por ';', probar la cadena completa
    if not candidates and path_str.lower().endswith(".pdf"):
        candidates = [path_str]

    for cand in candidates:
        p = Path(cand)
        if not p.is_absolute():
            p = (BASE_DIR / p).resolve()
        # Devolver primera que exista; si ninguna existe, devolver la primera normalizada
        if p.exists():
            return p.as_posix()
    return Path(candidates[0]).as_posix() if candidates else None


def _extract_first_surname(authors_raw) -> str:
    """Extrae el apellido del primer autor del string de autores.

    Soporta formatos:
    - "Apellido, Nombre; ..."        (Zotero/Scopus)
    - "Nombre M. Apellido; ..."      (directo)
    - "Nombre Medio Apellido; ..."   (espacio)
    """
    if not authors_raw or pd.isna(authors_raw):
        return "Unknown"
    raw = str(authors_raw).strip().strip('"')
    if not raw or raw == "nan":
        return "Unknown"

    first_author = raw.split(";")[0].strip()
    if not first_author:
        return "Unknown"

    if "," in first_author:
        # "Apellido, Nombre" → tomar antes de la coma
        surname = first_author.split(",")[0].strip()
    else:
        # "Nombre M. Apellido" → tomar ultima palabra
        parts = first_author.split()
        surname = parts[-1].strip() if parts else "Unknown"

    # Limpiar: solo letras (preservar guiones como parte del apellido)
    surname = re.sub(r"[^a-zA-ZÀ-ÿ\-]", "", surname)
    if not surname:
        return "Unknown"

    # Capitalizar primera letra, respetar el resto
    return surname[0].upper() + surname[1:] if len(surname) > 1 else surname.upper()


def _generate_unique_keys(df: pd.DataFrame) -> list[str]:
    """Genera claves unicas Apellido+Año para cada fila del DataFrame.

    Si la columna 'Key' existe y tiene valores, los preserva.
    Para filas sin key, genera Surname+Year (ej: Elmotri2025, Huang2023).
    Duplicados se resuelven con sufijo: Tang2024a, Tang2024b, ...

    Returns:
        Lista de keys en el mismo orden que las filas del DataFrame.
    """
    has_key_col = "Key" in df.columns

    # Determinar columna de autores y año (ya normalizadas por _normalize_columns)
    author_col = "Author" if "Author" in df.columns else "Authors"
    year_col = None
    for col in ("Publication Year", "Year", "Publication_Year"):
        if col in df.columns:
            year_col = col
            break

    # Fase 1: generar key base para cada fila
    base_keys = []
    has_existing_key = []  # True si la fila ya tenia Key valido

    for _, row in df.iterrows():
        # Si Key existe y es valido, usarlo
        if has_key_col:
            existing = str(row.get("Key", "")).strip()
            if existing and existing != "nan":
                base_keys.append(existing)
                has_existing_key.append(True)
                continue

        has_existing_key.append(False)

        # Extraer apellido
        surname = _extract_first_surname(row.get(author_col, ""))

        # Extraer año
        year = "XXXX"
        if year_col:
            yr_val = row.get(year_col)
            if pd.notna(yr_val):
                try:
                    year = str(int(float(yr_val)))
                except (ValueError, TypeError):
                    pass

        base_keys.append(f"{surname}{year}")

    # Fase 2: resolver duplicados con sufijos a, b, c, ...
    counts = Counter(base_keys)
    suffix_tracker: dict[str, int] = {}
    final_keys = []

    for i, bk in enumerate(base_keys):
        if has_existing_key[i]:
            # Key original de Zotero — no modificar
            final_keys.append(bk)
        elif counts[bk] == 1:
            # Unico — sin sufijo
            final_keys.append(bk)
        else:
            # Duplicado — agregar sufijo a, b, c, ...
            idx = suffix_tracker.get(bk, 0)
            if idx < 26:
                suffix = chr(ord("a") + idx)
            else:
                suffix = f"_{idx}"
            suffix_tracker[bk] = idx + 1
            final_keys.append(f"{bk}{suffix}")

    # Verificar unicidad final (defensivo)
    seen = set()
    for i, k in enumerate(final_keys):
        if k in seen:
            final_keys[i] = f"{k}_{i}"
        seen.add(final_keys[i])

    n_generated = sum(1 for e in has_existing_key if not e)
    if n_generated:
        logger.info(f"Keys auto-generadas: {n_generated}/{len(final_keys)} "
                     f"(formato: Apellido+Año)")
        # Mostrar algunos ejemplos
        examples = [k for k, e in zip(final_keys, has_existing_key) if not e][:5]
        logger.info(f"  Ejemplos: {examples}")

    return final_keys


def _parse_raw_references(refs_value) -> list[str]:
    """Parsea la columna References / CR del CSV a lista de strings de referencia.

    Soporta separadores comunes: '; ' (Scopus/WoS), '|', newlines.
    Cada referencia se limpia de espacios extra.
    """
    if not refs_value or pd.isna(refs_value):
        return []
    raw = str(refs_value).strip()
    if not raw or raw == "nan":
        return []
    # Separar por '; ' primero (formato WoS/Scopus), luego '|', luego newline
    if "; " in raw:
        parts = raw.split("; ")
    elif "|" in raw:
        parts = raw.split("|")
    elif "\n" in raw:
        parts = raw.split("\n")
    else:
        parts = [raw]
    return [p.strip() for p in parts if p.strip()]


_REF_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s;,)\]]+", re.IGNORECASE)
_REF_WOS_RE = re.compile(
    r"^([A-Za-zÀ-ÿ'\-\s\.]+?),\s*((?:18|19|20)\d{2})\s*,\s*([^,]+)"
)
_REF_APA_RE = re.compile(
    r"^([A-Za-zÀ-ÿ'\-]+(?:,\s*[A-Z]\.)?[^()]*?)\(\s*((?:18|19|20)\d{2})[a-z]?\s*\)\.\s*([^.]+?)\.\s*([^,.;]+)"
)
_REF_SCOPUS_RE = re.compile(
    r"^([A-Za-zÀ-ÿ'\-]+(?:\s+[A-Z]\.)+)(?:,\s*[A-Za-zÀ-ÿ'\-]+(?:\s+[A-Z]\.)+)*,\s*(.+?),\s*([^,]+?),\s*(?:pp\.|vol\.|\(\s*(?:18|19|20)\d{2}\s*\))"
)
_REF_YEAR_RE = re.compile(r"\b((?:17|18|19|20)\d{2})\b")


def _parse_structured_reference(ref_str: str) -> Optional[dict]:
    """Parsea una cadena de referencia nativa (WoS CR, Scopus References o APA/IEEE)
    en un diccionario estructurado compatible con RPYS, CiteSpace y Acoplamiento Bibliografico.
    """
    if not ref_str or not ref_str.strip():
        return None
    s = ref_str.strip()
    import hashlib

    doi = ""
    m_doi = _REF_DOI_RE.search(s)
    if m_doi:
        doi = normalize_doi(m_doi.group(0).rstrip(".)]"))

    first_author = ""
    year: Optional[int] = None
    title = ""
    journal = ""

    # 1. Intentar formato WoS CR: "Apellido Iniciales, 2018, JOURNAL, V10, P25, DOI ..."
    m_wos = _REF_WOS_RE.match(s)
    if m_wos:
        first_author = m_wos.group(1).strip()
        year = int(m_wos.group(2))
        journal = m_wos.group(3).strip()
        title = f"{first_author} ({year}) {journal}"
    else:
        # 2. Intentar formato APA: "Autor, A. (2018). Titulo. Revista, ..."
        m_apa = _REF_APA_RE.match(s)
        if m_apa:
            first_author = m_apa.group(1).split(",")[0].strip()
            year = int(m_apa.group(2))
            title = m_apa.group(3).strip()
            journal = m_apa.group(4).strip()
        else:
            # 3. Intentar formato Scopus: "Autor A., Titulo, Revista, (2018)"
            m_scop = _REF_SCOPUS_RE.match(s)
            if m_scop:
                first_author = m_scop.group(1).strip()
                title = m_scop.group(2).strip()
                journal = m_scop.group(3).strip()

    # Fallback para año y primer autor si no coincidio el patron completo
    if year is None:
        m_yr = _REF_YEAR_RE.search(s)
        if m_yr:
            year = int(m_yr.group(1))
    if not first_author:
        head = s.split(",")[0].split("(")[0].strip()
        if head and len(head) <= 45 and not head[0].isdigit():
            first_author = head
    if not title:
        title = s[:140]

    ref_id = doi if doi else f"cr:{hashlib.md5(s.lower().encode('utf-8')).hexdigest()[:12]}"
    return {
        "id": ref_id,
        "doi": doi,
        "first_author": first_author or "Unknown",
        "authors": [first_author] if first_author else [],
        "publication_year": year,
        "year": year,
        "title": title,
        "journal": journal,
        "raw": s,
        "source": "csv_cr",
    }


def _parse_open_access_csv(oa_value) -> dict | None:
    """Parsea la columna Open_Access del CSV a dict compatible con el pipeline.

    Valores esperados: 'Gold', 'Green', 'Bronze', 'Hybrid', 'Yes', 'No', etc.
    Retorna dict con campo 'status' o None si vacio.
    """
    if not oa_value or pd.isna(oa_value):
        return None
    raw = str(oa_value).strip().lower()
    if not raw or raw == "nan":
        return None
    # Normalizar valores comunes
    oa_map = {
        "gold": "gold", "green": "green", "bronze": "bronze",
        "hybrid": "hybrid", "diamond": "diamond",
        "yes": "gold", "true": "gold", "1": "gold",
        "no": "closed", "false": "closed", "0": "closed",
        "closed": "closed",
    }
    status = oa_map.get(raw, raw)
    is_oa = status != "closed"
    return {"is_oa": is_oa, "oa_status": status, "source": "csv"}


def dataframe_to_articles(df: pd.DataFrame) -> list[dict]:
    """Convierte DataFrame a lista de dicts (estructura central de datos)."""
    _log_type_token_coverage(df)

    # Pre-computar keys unicas para todo el DataFrame
    unique_keys = _generate_unique_keys(df)

    articles = []
    for idx, (_, row) in enumerate(df.iterrows()):
        doi = normalize_doi(str(row.get("DOI", "")))
        extra_info = parse_extra_field(row.get("Extra", ""))
        pdf_path = find_pdf_path(row.get("File Attachments", ""))

        # Times Cited: prefer explicit column, fallback to Extra field
        times_cited = 0
        if pd.notna(row.get("Times Cited")):
            try:
                times_cited = int(row["Times Cited"])
            except (ValueError, TypeError):
                pass
        if times_cited == 0:
            times_cited = extra_info.get("scopus_cited_by", 0)

        article = {
            "key": unique_keys[idx],
            "item_type": normalize_item_type(str(row.get("Item Type", ""))),
            "study_design": extract_study_design(str(row.get("Item Type", ""))),
            "publication_year": int(row["Publication Year"]) if pd.notna(row.get("Publication Year")) else None,
            "authors_raw": str(row.get("Author", "")),
            "authors": parse_author_string(str(row.get("Author", ""))),
            "title": str(row.get("Title", "")).strip(),
            "publication_title": normalize_journal_name(str(row.get("Publication Title", ""))),
            "issn": str(row.get("ISSN", "")).strip() if pd.notna(row.get("ISSN")) else "",
            "doi": doi,
            "url": str(row.get("Url", "")),
            "abstract": str(row.get("Abstract Note", "")),
            "pages": str(row.get("Pages", "")),
            "num_pages": int(row["Num Pages"]) if pd.notna(row.get("Num Pages")) else None,
            "issue": str(row.get("Issue", "")),
            "volume": str(row.get("Volume", "")),
            "journal_abbrev": str(row.get("Journal Abbreviation", "")),
            "publisher": str(row.get("Publisher", "")),
            "language": normalize_language(str(row.get("Language", ""))) if pd.notna(row.get("Language")) else "",
            "manual_tags": parse_tags(row.get("Manual Tags", "")),
            # Solo palabras clave de autor (columna Author_Keywords); [] si el registro no las trae
            "author_keywords": parse_tags(row.get("__author_keywords", "")) if pd.notna(row.get("__author_keywords")) else [],
            "automatic_tags": parse_tags(row.get("Automatic Tags", "")),
            "all_tags": list(set(
                parse_tags(row.get("Manual Tags", ""))
                + parse_tags(row.get("Automatic Tags", ""))
            )),
            "pdf_path": pdf_path,
            "extra": extra_info,
            "source_db": str(row.get("Source DB", "")).strip() if pd.notna(row.get("Source DB")) else "",
            "merged_from": str(row.get("Merged From", "")).strip() if pd.notna(row.get("Merged From")) else "",
            "richness_score": float(row["Richness Score"]) if pd.notna(row.get("Richness Score")) else None,
            "wos_accession": str(row.get("WoS Accession", "")) if pd.notna(row.get("WoS Accession")) else "",
            "wos_index": str(row.get("WoS Index", "")) if pd.notna(row.get("WoS Index")) else "",
            "eric_id": str(row.get("ERIC ID", "")) if pd.notna(row.get("ERIC ID")) else "",
            "database_id": str(row.get("Database ID", "")) if pd.notna(row.get("Database ID")) else "",
            "isbn": str(row.get("ISBN", "")) if pd.notna(row.get("ISBN")) else "",
            "booktitle": str(row.get("Book Title", "")) if pd.notna(row.get("Book Title")) else "",
            "article_number": str(row.get("Article Number", "")) if pd.notna(row.get("Article Number")) else "",
            "month": str(row.get("Month", "")) if pd.notna(row.get("Month")) else "",
            "editor": str(row.get("Editor", "")) if pd.notna(row.get("Editor")) else "",
            "institution": str(row.get("Institution", "")) if pd.notna(row.get("Institution")) else "",
            "pmid": str(row.get("PMID", "")) if pd.notna(row.get("PMID")) else "",
            "pmc_id": str(row.get("PMC ID", "")) if pd.notna(row.get("PMC ID")) else "",
            "mesh_terms": parse_tags(row.get("MeSH Terms", "")),
            "full_author_names": str(row.get("Full Author Names", "")) if pd.notna(row.get("Full Author Names")) else "",
            "place_of_publication": str(row.get("Place of Publication", "")) if pd.notna(row.get("Place of Publication")) else "",
            "funding_text": str(row.get("Funding Text", "")) if pd.notna(row.get("Funding Text")) else "",
            "funding_agency": str(row.get("Funding Agency", "")) if pd.notna(row.get("Funding Agency")) else "",
            # Campos a enriquecer en fases posteriores
            "openalex": {},
            "crossref": {},
            "semantic_scholar": {},
            "cited_by_count": times_cited,
            "referenced_works": [],
            "referenced_works_detailed": [],
            "referenced_works_raw": _parse_raw_references(row.get("References", "")),
            "concepts": [],
            "topics": [],
            "affiliations_raw": str(row.get("Affiliations", "")) if pd.notna(row.get("Affiliations")) else "",
            "affiliations": [],
            "open_access": _parse_open_access_csv(row.get("Open Access", "")) or {},
            "funding": [],
            "licenses": [],
            "author_orcids": [],
            "tldr": "",
            "influential_citation_count": 0,
            "citation_intents": {},
            "s2_fields_of_study": [],
            "specter_embedding": None,
            # Enrichment: OpenAlex Meta (Phase 1d)
            "journal_metrics": {},
            "author_metrics": [],
            # Enrichment: SCImago (Phase 1e)
            "scimago": {},
            "sjr": None,
            "sjr_quartile": None,
            # Enrichment: Altmetric (Phase 1f)
            "altmetric": {},
            "altmetric_score": None,
            "social_media_mentions": {},
            "policy_citations": 0,
            # Enrichment: Crossref Event Data (Phase 1l)
            "event_total": 0,
            "event_by_source": {},
            "event_timeline": {},
            # Enrichment: OpenCitations (Phase 1g)
            "opencitations": {},
            "self_citation_rate": None,
            "citation_velocity": None,
            # Enrichment: Unpaywall (Phase 1h) — extends open_access
        }

        # Seed affiliations from Institution column if affiliations_raw is empty
        if article["institution"] and not article["affiliations_raw"]:
            article["affiliations_raw"] = article["institution"]

        # Seed funding[] from CSV columns (Crossref will merge/extend later)
        _fa = article["funding_agency"]
        _ft = article["funding_text"]
        if _fa:
            for agency in _fa.split(";"):
                agency = agency.strip()
                if agency:
                    article["funding"].append({
                        "name": agency,
                        "source": "csv",
                    })
        if _ft and not article["funding"]:
            # If no structured agency but there is funding text, store as single entry
            article["funding"].append({
                "name": _ft[:200],
                "source": "csv_text",
            })

        # Seed referenced_works y referenced_works_detailed desde References / CR nativas del CSV
        if article["referenced_works_raw"]:
            seen_ref_ids: set[str] = set()
            for ref_str in article["referenced_works_raw"]:
                parsed_ref = _parse_structured_reference(ref_str)
                if not parsed_ref:
                    continue
                rid = parsed_ref["id"]
                if rid not in seen_ref_ids:
                    seen_ref_ids.add(rid)
                    article["referenced_works"].append(rid)
                    article["referenced_works_detailed"].append(parsed_ref)

        articles.append(article)

    # Deduplicacion defensiva por DOI o titulo normalizado
    deduped = []
    seen_dois: set[str] = set()
    seen_titles: set[str] = set()
    n_dup = 0
    for a in articles:
        d = (a.get("doi") or "").strip().lower()
        t_norm = re.sub(r"[^a-z0-9]+", "", (a.get("title") or "").lower())
        if d and d != "nan" and d in seen_dois:
            logger.warning(f"Duplicado por DOI omitido: [{a.get('key')}] DOI={d}")
            n_dup += 1
            continue
        if len(t_norm) >= 25 and t_norm in seen_titles and not d:
            logger.warning(f"Duplicado por titulo omitido: [{a.get('key')}] {a.get('title', '')[:60]}")
            n_dup += 1
            continue
        if d and d != "nan":
            seen_dois.add(d)
        if len(t_norm) >= 25:
            seen_titles.add(t_norm)
        deduped.append(a)

    if n_dup:
        logger.info(f"Deduplicacion automatica: {n_dup} registros duplicados eliminados")
    articles = deduped

    # Unificar variantes de nombres de revista en el corpus
    n_j_norm = canonicalize_corpus_journals(articles)
    if n_j_norm:
        logger.info(f"Nombres de revista normalizados/unificados en {n_j_norm} articulos")

    logger.info(f"Articulos parseados: {len(articles)}")
    logger.info(
        "Tipo de documento (vehiculo): %s",
        dict(Counter(a["item_type"] for a in articles).most_common()),
    )
    _n_dis = sum(1 for a in articles if a.get("study_design"))
    if _n_dis:
        logger.info(
            "Diseño de estudio detectado en %d articulos: %s",
            _n_dis,
            dict(Counter(d for a in articles for d in a.get("study_design", [])).most_common()),
        )
    return articles


def _is_doi_key(key: str) -> bool:
    """Detecta si una key parece un DOI deformado (ej: 10_1234_xxx)."""
    return bool(key and re.match(r"^10[_.]", key))


def rekey_articles(articles: list[dict]) -> dict[str, str]:
    """Re-genera keys Apellido+Año para articulos ya cargados (in-place).

    Util para migrar enriched_articles.json con keys DOI-based
    a formato legible sin perder datos de enriquecimiento.

    Returns:
        Mapeo {old_key: new_key} para trazabilidad.
        Solo incluye keys que cambiaron.
    """
    n = len(articles)

    # Fase 1: generar key base para cada articulo
    base_keys = []
    for a in articles:
        surname = _extract_first_surname(a.get("authors_raw", ""))
        year = str(a["publication_year"]) if a.get("publication_year") else "XXXX"
        base_keys.append(f"{surname}{year}")

    # Fase 2: resolver duplicados con sufijos a, b, c, ...
    counts = Counter(base_keys)
    suffix_tracker: dict[str, int] = {}
    new_keys = []

    for bk in base_keys:
        if counts[bk] == 1:
            new_keys.append(bk)
        else:
            idx = suffix_tracker.get(bk, 0)
            suffix = chr(ord("a") + idx) if idx < 26 else f"_{idx}"
            suffix_tracker[bk] = idx + 1
            new_keys.append(f"{bk}{suffix}")

    # Verificar unicidad final (defensivo)
    seen: set[str] = set()
    for i, k in enumerate(new_keys):
        if k in seen:
            new_keys[i] = f"{k}_{i}"
        seen.add(new_keys[i])

    # Aplicar y registrar cambios
    key_map: dict[str, str] = {}
    for i, a in enumerate(articles):
        old_key = a["key"]
        nk = new_keys[i]
        if old_key != nk:
            key_map[old_key] = nk
            a["key"] = nk

    if key_map:
        logger.info(f"Re-keyed {len(key_map)}/{n} articulos (Apellido+Año)")
        examples = list(key_map.items())[:5]
        for old, new in examples:
            logger.info(f"  {old[:45]:45s} -> {new}")
    else:
        logger.info("Keys ya estan en formato Apellido+Año — sin cambios")

    return key_map


def migrate_keys_in_outputs(key_map: dict[str, str]) -> int:
    """Migra keys viejas en archivos JSON de output usando el mapeo."""
    import glob

    output_dir = BASE_DIR / "output"
    patterns = [
        str(output_dir / "sota_analysis" / "**" / "*.json"),
        str(output_dir / "knowledge_map" / "**" / "*.json"),
        str(output_dir / "rdm" / "**" / "*.json"),
    ]

    files_modified = 0
    for pattern in patterns:
        for filepath in glob.glob(pattern, recursive=True):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()

                original = content
                for old_key, new_key in key_map.items():
                    # Reemplazar como valor JSON: "old_key" -> "new_key"
                    content = content.replace(f'"{old_key}"', f'"{new_key}"')

                if content != original:
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(content)
                    files_modified += 1
                    logger.debug(f"  Migrated keys in: {filepath}")
            except Exception as e:
                logger.warning(f"  Could not migrate {filepath}: {e}")

    if files_modified:
        logger.info(f"Keys migradas en {files_modified} archivos de output/")
    return files_modified


def run_phase0(csv_path: Path = CSV_PATH) -> list[dict]:
    """Ejecuta la Fase 0 completa."""
    logger.info("=== FASE 0: Carga y parseo del CSV ===")
    df = load_csv(csv_path)
    articles = dataframe_to_articles(df)

    # Resumen
    years = [a["publication_year"] for a in articles if a["publication_year"]]
    dois = [a for a in articles if a["doi"]]
    pdfs = [a for a in articles if a["pdf_path"]]

    logger.info(f"  Total articulos: {len(articles)}")
    logger.info(f"  Rango de años: {min(years)}-{max(years)}" if years else "  Sin años")
    logger.info(f"  Con DOI: {len(dois)}")
    logger.info(f"  Con PDF: {len(pdfs)}")
    logger.info(f"  Revistas unicas: {len(set(a['publication_title'] for a in articles))}")

    return articles


if __name__ == "__main__":
    from src.utils import setup_logging
    setup_logging()
    articles = run_phase0()
    for a in articles[:3]:
        print(f"  [{a['key']}] {a['authors'][0] if a['authors'] else 'N/A'} ({a['publication_year']}) - {a['title'][:60]}...")
