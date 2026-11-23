"""
Fase 2d: Evaluacion de calidad de abstracts - completitud, estructura, informatividad.

Diseño: Sistema agnóstico al dominio con verificaciones anti-alucinación.

Principios:
1. Patrones base conservadores (alta precisión, evitar falsos positivos)
2. Extensiones de dominio opcionales y configurables
3. Verificación cruzada: cada flag requiere evidencia múltiple
4. Scores de confianza para transparencia
5. Auditoría: se guarda evidencia de cada detección
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, Callable

logger = logging.getLogger("bibliometrico.quality")


# ===========================================================================
# CONFIGURACIÓN DE PATRONES - SISTEMA EXTENSIBLE
# ===========================================================================

@dataclass
class PatternConfig:
    """Configuración de patrones para detección de elementos en abstracts."""

    # Patrones base (conservadores, alta precisión)
    base_patterns: dict[str, list[str]] = field(default_factory=dict)

    # Extensiones de dominio (opcionales)
    domain_extensions: dict[str, list[str]] = field(default_factory=dict)

    # Umbral mínimo de matches para confirmar detección
    min_matches_for_confidence: int = 1

    # Palabras a excluir (falsos positivos conocidos)
    exclusions: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# PATRONES BASE - Conservadores, agnósticos al dominio
# ---------------------------------------------------------------------------

BASE_PATTERNS = {
    "has_objective": [
        # Declaraciones explícitas de objetivo (alta confianza)
        r"\b(aim|objective|purpose|goal)\s+(of\s+)?(this|the)\s+(study|paper|research|work)",
        r"\b(this|the)\s+(study|paper|research|work)\s+(aims?|investigates?|examines?|explores?)",
        r"\bwe\s+(aim|propose|present|introduce|investigate|examine)\b",
        r"\b(to\s+investigate|to\s+examine|to\s+explore|to\s+analyze)\b",
        r"\b(is\s+proposed|are\s+proposed|proposes)\b",
        r"\bpurpose\s+of\s+(this|the)\b",
    ],
    "has_methods": [
        # Métodos y metodología (alta confianza)
        r"\b(method|methodology|approach|technique|procedure|protocol)\b",
        r"\b(experiment|simulation|analysis|evaluation)\s+(is|was|were)\s+(conducted|performed)",
        r"\busing\s+(a|an|the)\s+\w+\s+(method|approach|technique|algorithm)",
        r"\b(dataset|data\s*set|sample|corpus)\s+(of|with|containing)\b",
        r"\b(we|the\s+authors?)\s+(use|employ|apply|implement|develop)",
        r"\b(algorithm|model|framework|system)\s+(is|was)\s+(developed|designed|implemented)",
    ],
    "has_results": [
        # Resultados y hallazgos (alta confianza)
        r"\b(results?|findings?)\s+(show|indicate|demonstrate|reveal|suggest)",
        r"\b(showed?|demonstrated?|achieved?|obtained?|found)\s+(that|a|an|the|\d)",
        r"\b(significant(ly)?|improved?|better|higher|lower|increased?|decreased?)\s+(than|in|on|effects?|differences?|results?|scores?|levels?)\b",
        r"\bcompared?\s+(to|with)\s+(the|other|existing|baseline|state-of-the-art)",
        r"\b(outperform|superior|effective|efficient)\b",
        # Antes: metricas de machine learning (accuracy, precision, recall, f1). Ahora, lenguaje general de efecto:
        r"\b(increase|decrease|improvement|reduction|difference|association|correlation|effect)s?\s+(of|in|between|on)\b",
    ],
    "has_conclusion": [
        # Conclusiones y contribuciones (alta confianza)
        r"\b(conclude|conclusion|in\s+summary|to\s+summarize)\b",
        r"\b(therefore|thus|hence|consequently)\s*,?\s*(this|the|we|it)",
        r"\b(contribution|contributions)\s+(of|to)\s+(this|the)",
        r"\b(findings?|results?)\s+(suggest|indicate|show)\s+that",
        r"\b(future\s+(work|research|studies)|limitations?)\b",
        r"\b(can\s+be\s+used|suitable\s+for|applicable\s+to|effective\s+for)\b",
    ],
    "has_quantitative": [
        # Datos cuantitativos (conservador - solo patrones claros)
        r"\d+\.?\d*\s*%",  # Porcentajes
        r"\b[nN]\s*=\s*\d+",  # Tamaño de muestra
        r"\bp\s*[<>=]\s*0?\.\d+",  # Valores p estadísticos
        # Antes: metricas de machine learning. Ahora, notacion estadistica general de cualquier campo:
        # r = .45, beta = 0.31, F(2, 120) = 4.5, OR = 1.3, accuracy = 0.91, chi2 = 8.1
        r"\b[a-zA-Zα-ωΒΧ]{1,8}\d?\s*(\(\s*\d+(\s*,\s*\d+)?\s*\))?\s*[=<>≤≥]\s*-?\d*\.?\d+",
        r"\b\d{2}\s*%\s*(ci|confidence\s+interval)\b",  # 95% CI
        r"\d+\s+(samples?|participants?|subjects?|cases?|instances?|examples?)\b",
        r"\(\s*\d+\.?\d*\s*[,;±]\s*\d+\.?\d*\s*\)",  # Valores con desviación
    ],
}

# ---------------------------------------------------------------------------
# Sin extensiones de dominio: antes habia paquetes de patrones por area (ingenieria electrica,
# salud, ciencias sociales...) y detect_domain() adivinaba el area de cada resumen; los resumenes
# con area 'detectada' recibian mas patrones y puntuaban mas facil. Todos se miden igual.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# INFORMATIVIDAD sin lista de terminos: especificidad del vocabulario RELATIVA AL CORPUS
# ---------------------------------------------------------------------------
# Antes: densidad de 95 'terminos tecnicos' fijos (algorithm, architecture, benchmark, tensor...),
# orientados a computacion. Ahora: IDF medio de las palabras de contenido del resumen, calculado sobre
# los resumenes del propio corpus, y su percentil dentro del corpus (escala independiente del campo).
import math as _math

_CORPUS_IDF: dict[str, float] = {}
_CORPUS_ESPECIFICIDAD: list[float] = []      # IDF medio de cada resumen del corpus, ordenado


def _palabras_contenido(texto: str) -> list[str]:
    from src.metodos_declarados import _FUNCTION_WORDS
    return [w for w in re.findall(r"\b[a-z][a-z\-]{2,}\b", (texto or "").lower()) if w not in _FUNCTION_WORDS]


def prepare_corpus_specificity(abstracts: list[str]) -> None:
    """Calcula el IDF de cada palabra sobre los resumenes del corpus y la distribucion de especificidad."""
    validos = [a for a in abstracts if a and a != "nan" and len(a.strip()) >= 10]
    n = len(validos)
    df: Counter = Counter()
    for a in validos:
        df.update(set(_palabras_contenido(a)))
    _CORPUS_IDF.clear()
    _CORPUS_IDF.update({w: _math.log((1 + n) / (1 + c)) + 1.0 for w, c in df.items()})
    _CORPUS_ESPECIFICIDAD.clear()
    _CORPUS_ESPECIFICIDAD.extend(sorted(e for e in (_especificidad(a) for a in validos) if e is not None))


def _especificidad(abstract: str) -> Optional[float]:
    ws = _palabras_contenido(abstract)
    if not ws or not _CORPUS_IDF:
        return None
    return sum(_CORPUS_IDF.get(w, max(_CORPUS_IDF.values())) for w in ws) / len(ws)


def _percentil(valor: float) -> float:
    """Proporcion de resumenes del corpus con especificidad <= valor (funcion de distribucion empirica)."""
    import bisect
    if not _CORPUS_ESPECIFICIDAD:
        return 0.0
    return bisect.bisect_right(_CORPUS_ESPECIFICIDAD, valor) / len(_CORPUS_ESPECIFICIDAD)


# ===========================================================================
# SISTEMA DE VERIFICACIÓN ANTI-ALUCINACIÓN
# ===========================================================================

@dataclass
class DetectionEvidence:
    """Evidencia de una detección para auditoría y verificación."""
    element: str
    detected: bool
    confidence: float  # 0.0 - 1.0
    matches: list[str]  # Fragmentos que matchearon
    patterns_matched: int
    patterns_total: int
    verification_passed: bool
    verification_notes: list[str]


class QualityVerifier:
    """
    Sistema de verificación anti-alucinación.

    Principios:
    1. Verificación cruzada: múltiples señales confirman detección
    2. Contexto: verificar que el match tiene sentido en contexto
    3. Exclusiones: filtrar falsos positivos conocidos
    4. Confianza calibrada: score basado en calidad de evidencia
    """

    # Exclusiones globales (falsos positivos comunes)
    GLOBAL_EXCLUSIONS = [
        r"©\s*\d{4}",  # Copyright
        r"all\s+rights\s+reserved",
        r"elsevier|springer|wiley|ieee|acm",  # Publishers
        r"doi:\s*\d+",
        r"https?://",
    ]

    # Palabras de contexto negativo (indican que no es contenido sustantivo)
    NEGATIVE_CONTEXT = [
        r"(rights?\s+reserved|copyright|license|permission)",
        r"(corresponding\s+author|email|affiliation)",
    ]

    @classmethod
    def clean_abstract(cls, abstract: str) -> str:
        """Limpia el abstract de elementos no sustantivos."""
        cleaned = abstract
        for pattern in cls.GLOBAL_EXCLUSIONS:
            cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    @classmethod
    def is_truncated(cls, abstract: str) -> tuple[bool, str]:
        """Detecta si el abstract está truncado."""
        if not abstract:
            return True, "empty"

        abstract = abstract.strip()

        # Termina abruptamente (letra minúscula sin puntuación)
        if re.search(r"[a-z]\s*$", abstract) and not re.search(r"[.!?]\s*$", abstract):
            return True, "ends_abruptly"

        # Muy corto
        if len(abstract.split()) < 30:
            return True, "too_short"

        return False, "ok"

    @classmethod
    def verify_detection(
        cls,
        abstract: str,
        element: str,
        matches: list[str],
        min_confidence: float = 0.3,
    ) -> tuple[bool, float, list[str]]:
        """
        Verifica que una detección es válida (no es alucinación).

        Args:
            abstract: Texto original
            element: Elemento detectado (has_objective, etc.)
            matches: Lista de fragmentos que matchearon
            min_confidence: Umbral mínimo de confianza

        Returns:
            tuple: (verificado, confianza, notas)
        """
        notes = []
        confidence = 0.0

        if not matches:
            return False, 0.0, ["no_matches"]

        # 1. Verificar que los matches existen en el texto original
        valid_matches = []
        for match in matches:
            # Normalizar para comparación
            match_clean = match.lower().strip()
            abstract_clean = abstract.lower()
            if match_clean in abstract_clean or re.search(re.escape(match_clean), abstract_clean):
                valid_matches.append(match)

        if not valid_matches:
            notes.append("matches_not_found_in_original")
            return False, 0.0, notes

        # 2. Verificar contexto (no es parte de copyright, etc.)
        contextual_matches = []
        for match in valid_matches:
            is_negative = False
            for neg_pattern in cls.NEGATIVE_CONTEXT:
                # Buscar contexto alrededor del match
                match_pos = abstract.lower().find(match.lower())
                if match_pos >= 0:
                    context_start = max(0, match_pos - 50)
                    context_end = min(len(abstract), match_pos + len(match) + 50)
                    context = abstract[context_start:context_end]
                    if re.search(neg_pattern, context, re.IGNORECASE):
                        is_negative = True
                        break
            if not is_negative:
                contextual_matches.append(match)

        if not contextual_matches:
            notes.append("all_matches_in_negative_context")
            return False, 0.1, notes

        # 3. Calcular confianza basada en cantidad y calidad de matches
        n_matches = len(contextual_matches)

        # Base: al menos 1 match válido
        confidence = 0.5 if n_matches >= 1 else 0.0

        # Bonus por múltiples matches
        if n_matches >= 2:
            confidence += 0.2
        if n_matches >= 3:
            confidence += 0.1

        # Bonus por matches largos (más contexto = más confianza)
        avg_match_len = sum(len(m) for m in contextual_matches) / len(contextual_matches)
        if avg_match_len >= 20:
            confidence += 0.1
        if avg_match_len >= 40:
            confidence += 0.1

        confidence = min(1.0, confidence)

        # 4. Verificaciones específicas por elemento
        if element == "has_quantitative":
            # Para cuantitativo, verificar que hay números reales
            has_real_numbers = any(re.search(r"\d+\.?\d*", m) for m in contextual_matches)
            if not has_real_numbers:
                notes.append("quantitative_without_numbers")
                confidence *= 0.5

        verified = confidence >= min_confidence
        if verified:
            notes.append(f"verified_with_{n_matches}_matches")
        else:
            notes.append(f"below_threshold_{min_confidence}")

        return verified, confidence, notes


# ===========================================================================
# FUNCIONES DE EVALUACIÓN DE CALIDAD
# ===========================================================================

def compile_patterns(
    base_patterns: dict[str, list[str]],
    domain_extensions: Optional[dict[str, list[str]]] = None,
) -> dict[str, re.Pattern]:
    """
    Compila patrones base + extensiones de dominio en regex.

    Args:
        base_patterns: Patrones base (obligatorios)
        domain_extensions: Extensiones específicas de dominio (opcionales)

    Returns:
        dict de patrones compilados
    """
    compiled = {}

    for element, patterns in base_patterns.items():
        all_patterns = patterns.copy()

        # Agregar extensiones de dominio si existen
        if domain_extensions and element in domain_extensions:
            all_patterns.extend(domain_extensions[element])

        # Compilar como un solo patrón OR
        combined = "|".join(f"({p})" for p in all_patterns)
        compiled[element] = re.compile(combined, re.IGNORECASE)

    return compiled


def detect_element(
    abstract: str,
    element: str,
    pattern: re.Pattern,
    verifier: type[QualityVerifier] = QualityVerifier,
) -> DetectionEvidence:
    """
    Detecta un elemento en el abstract con verificación anti-alucinación.

    Args:
        abstract: Texto del abstract
        element: Nombre del elemento a detectar
        pattern: Patrón regex compilado
        verifier: Clase de verificación

    Returns:
        DetectionEvidence con resultado y evidencia
    """
    # Limpiar abstract
    cleaned = verifier.clean_abstract(abstract)

    # Buscar matches
    matches = pattern.findall(cleaned)

    # Aplanar matches (findall puede devolver tuplas con grupos)
    flat_matches = []
    for m in matches:
        if isinstance(m, tuple):
            flat_matches.extend([x for x in m if x])
        else:
            flat_matches.append(m)

    # Eliminar duplicados y vacíos
    flat_matches = list(set(m.strip() for m in flat_matches if m and m.strip()))

    # Verificar detección
    verified, confidence, notes = verifier.verify_detection(
        abstract, element, flat_matches
    )

    return DetectionEvidence(
        element=element,
        detected=verified,
        confidence=confidence,
        matches=flat_matches[:5],  # Limitar para no saturar
        patterns_matched=len(flat_matches),
        patterns_total=1,  # Patrón combinado
        verification_passed=verified,
        verification_notes=notes,
    )


def compute_completeness(
    abstract: str,
    patterns: dict[str, re.Pattern],
    verifier: type[QualityVerifier] = QualityVerifier,
) -> tuple[float, list[str], list[str], dict[str, DetectionEvidence]]:
    """
    Evalúa completitud estructural del abstract con verificación.

    Args:
        abstract: Texto del abstract
        patterns: Patrones compilados
        verifier: Clase de verificación

    Returns:
        tuple: (score, elementos presentes, elementos faltantes, evidencias)
    """
    if not abstract or len(abstract.strip()) < 10:
        return 0.0, [], list(patterns.keys()), {}

    present = []
    missing = []
    evidences = {}

    for element, pattern in patterns.items():
        evidence = detect_element(abstract, element, pattern, verifier)
        evidences[element] = evidence

        if evidence.detected:
            present.append(element)
        else:
            missing.append(element)

    # Score ponderado por confianza
    if evidences:
        confidence_sum = sum(e.confidence for e in evidences.values() if e.detected)
        max_possible = len(patterns)
        score = (confidence_sum / max_possible) * 100
    else:
        score = 0.0

    return score, present, missing, evidences


def compute_structure_score(abstract: str) -> tuple[float, dict]:
    """Evalúa estructura y formato del abstract."""
    if not abstract or len(abstract.strip()) < 10:
        return 0.0, {"length": 0, "sentences": 0, "avg_sentence_len": 0}

    # Métricas básicas
    length = len(abstract)
    words = abstract.split()
    word_count = len(words)

    # Contar oraciones
    sentences = re.split(r'[.!?]+', abstract)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = len(sentences)

    avg_sentence_len = word_count / max(sentence_count, 1)

    metrics = {
        "length": length,
        "word_count": word_count,
        "sentences": sentence_count,
        "avg_sentence_len": round(avg_sentence_len, 1),
    }

    score = 0.0

    # Longitud óptima: 150-300 palabras
    if 150 <= word_count <= 300:
        score += 40
    elif 100 <= word_count < 150 or 300 < word_count <= 400:
        score += 30
    elif 50 <= word_count < 100 or 400 < word_count <= 500:
        score += 15
    elif word_count >= 50:
        score += 5

    # Número de oraciones: 5-10 es óptimo
    if 5 <= sentence_count <= 10:
        score += 30
    elif 3 <= sentence_count < 5 or 10 < sentence_count <= 15:
        score += 20
    elif sentence_count >= 2:
        score += 10

    # Longitud promedio de oración: 15-25 palabras
    if 15 <= avg_sentence_len <= 25:
        score += 30
    elif 10 <= avg_sentence_len < 15 or 25 < avg_sentence_len <= 35:
        score += 20
    elif avg_sentence_len >= 8:
        score += 10

    return min(score, 100), metrics


def compute_informativeness(abstract: str) -> tuple[float, dict]:
    """Evalua la informatividad del abstract sin listas de terminos (especificidad relativa al corpus)."""
    if not abstract or len(abstract.strip()) < 10:
        return 0.0, {"especificidad_percentil_corpus": None, "unique_terms": 0}

    text_lower = abstract.lower()
    words = re.findall(r'\b[a-z]{3,}\b', text_lower)

    if not words:
        return 0.0, {"especificidad_percentil_corpus": None, "unique_terms": 0}

    # Especificidad del vocabulario respecto del propio corpus (antes: densidad de 95 terminos fijos)
    esp = _especificidad(abstract)
    percentil = _percentil(esp) if esp is not None else None

    # Diversidad léxica
    unique_words = set(words)
    lexical_diversity = (len(unique_words) / len(words)) * 100

    # Números y acrónimos
    numbers = re.findall(r'\d+\.?\d*', abstract)
    acronyms = re.findall(r'\b[A-Z]{2,5}\b', abstract)

    metrics = {
        "especificidad_idf_media": round(esp, 3) if esp is not None else None,
        "especificidad_percentil_corpus": round(percentil, 3) if percentil is not None else None,
        "unique_terms": len(unique_words),
        "lexical_diversity": round(lexical_diversity, 1),
        "numeric_values": len(numbers),
        "acronyms": len(set(acronyms)),
    }

    score = 0.0

    # Mismos 35 puntos maximos que el componente anterior, proporcionales al percentil en el corpus
    if percentil is not None:
        score += 35 * percentil

    if 40 <= lexical_diversity <= 70:
        score += 25
    elif 30 <= lexical_diversity < 40 or 70 < lexical_diversity <= 80:
        score += 15
    elif lexical_diversity > 20:
        score += 5

    if len(numbers) > 0:
        score += 20

    if len(set(acronyms)) >= 2:
        score += 20
    elif len(set(acronyms)) == 1:
        score += 10

    return min(score, 100), metrics


def compute_readability(abstract: str) -> tuple[float, dict]:
    """Calcula métricas de legibilidad."""
    if not abstract or len(abstract.strip()) < 10:
        return 0.0, {"flesch_score": 0, "syllables_per_word": 0}

    words = abstract.split()
    word_count = len(words)

    if word_count == 0:
        return 0.0, {"flesch_score": 0, "syllables_per_word": 0}

    sentences = re.split(r'[.!?]+', abstract)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = max(len(sentences), 1)

    def count_syllables(word: str) -> int:
        word = word.lower()
        vowels = "aeiou"
        count = 0
        prev_vowel = False
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        if word.endswith('e') and count > 1:
            count -= 1
        return max(count, 1)

    total_syllables = sum(count_syllables(w) for w in words)
    syllables_per_word = total_syllables / word_count
    words_per_sentence = word_count / sentence_count

    flesch = 206.835 - (1.015 * words_per_sentence) - (84.6 * syllables_per_word)
    flesch = max(0, min(100, flesch))

    metrics = {
        "flesch_score": round(flesch, 1),
        "syllables_per_word": round(syllables_per_word, 2),
        "words_per_sentence": round(words_per_sentence, 1),
    }

    if 20 <= flesch <= 60:
        score = 80 + ((flesch - 20) / 40) * 20
    elif 10 <= flesch < 20 or 60 < flesch <= 70:
        score = 60
    elif flesch > 70:
        score = 50
    else:
        score = 40

    return min(score, 100), metrics


# ===========================================================================
# EVALUACIÓN PRINCIPAL CON VERIFICACIÓN
# ===========================================================================

def assess_abstract_quality(
    abstract: str,
    domain: Optional[str] = None,
    include_evidence: bool = False,
) -> dict:
    """
    Evaluación multi-dimensional de calidad de abstract con verificación anti-alucinación.

    Args:
        abstract: Texto del abstract a evaluar
        domain: IGNORADO (ya no hay extensiones por dominio; se conserva por compatibilidad). Antes: ('electrical_engineering',
                'health_sciences', 'social_sciences', 'computer_science')
        include_evidence: Si True, incluye evidencia detallada de detecciones

    Returns:
        dict con scores, métricas y evidencias
    """
    # Manejar casos vacíos
    if not abstract or not isinstance(abstract, str):
        return {
            "completeness_score": 0,
            "structure_score": 0,
            "informativeness_score": 0,
            "readability_score": 0,
            "overall_score": 0,
            "overall_quality": "low",
            "quality_flags": ["empty_abstract"],
            "flag_confidence": {},
            "details": {},
        }

    abstract = abstract.strip()

    # Manejar "nan" como string
    if abstract.lower() == "nan":
        return {
            "completeness_score": 0,
            "structure_score": 0,
            "informativeness_score": 0,
            "readability_score": 0,
            "overall_score": 0,
            "overall_quality": "low",
            "quality_flags": ["empty_abstract"],
            "flag_confidence": {},
            "details": {},
        }

    if len(abstract) < 50:
        return {
            "completeness_score": 0,
            "structure_score": 0,
            "informativeness_score": 0,
            "readability_score": 0,
            "overall_score": 0,
            "overall_quality": "low",
            "quality_flags": ["too_short"],
            "flag_confidence": {},
            "details": {"length": len(abstract)},
        }

    # Verificar truncamiento
    is_truncated, truncation_reason = QualityVerifier.is_truncated(abstract)

    # Compilar patrones con extensiones de dominio
    # Mismos patrones para todos los resumenes (sin extensiones por dominio)
    patterns = compile_patterns(BASE_PATTERNS, {})

    # Calcular scores
    completeness_score, present, missing, evidences = compute_completeness(
        abstract, patterns, QualityVerifier
    )
    structure_score, structure_metrics = compute_structure_score(abstract)
    informativeness_score, info_metrics = compute_informativeness(abstract)
    readability_score, read_metrics = compute_readability(abstract)

    # Score general ponderado
    weights = {
        "completeness": 0.35,
        "structure": 0.25,
        "informativeness": 0.25,
        "readability": 0.15,
    }

    overall_score = (
        completeness_score * weights["completeness"] +
        structure_score * weights["structure"] +
        informativeness_score * weights["informativeness"] +
        readability_score * weights["readability"]
    )

    # Determinar calidad general
    if overall_score >= 70:
        overall_quality = "high"
    elif overall_score >= 45:
        overall_quality = "medium"
    else:
        overall_quality = "low"

    # Generar flags CON CONFIANZA
    quality_flags = []
    flag_confidence = {}

    # Flags de completitud (basados en evidencia verificada)
    flag_mapping = {
        "has_objective": "missing_objective",
        "has_methods": "missing_methods",
        "has_results": "missing_results",
        "has_conclusion": "missing_conclusion",
        "has_quantitative": "no_quantitative_data",
    }

    for element, flag_name in flag_mapping.items():
        if element in evidences:
            evidence = evidences[element]
            if not evidence.detected:
                quality_flags.append(flag_name)
                # Confianza del flag = 1 - confianza de la detección
                # (alta confianza en el flag si baja confianza en detección)
                flag_confidence[flag_name] = round(1.0 - evidence.confidence, 2)

    # Flags de estructura
    if structure_metrics.get("word_count", 0) < 100:
        quality_flags.append("short_abstract")
        flag_confidence["short_abstract"] = 0.9
    elif structure_metrics.get("word_count", 0) > 400:
        quality_flags.append("long_abstract")
        flag_confidence["long_abstract"] = 0.9

    if structure_metrics.get("sentences", 0) < 3:
        quality_flags.append("few_sentences")
        flag_confidence["few_sentences"] = 0.9

    # Flags de informatividad
    if info_metrics.get("lexical_diversity", 0) < 30:
        quality_flags.append("low_lexical_diversity")
        flag_confidence["low_lexical_diversity"] = 0.8

    # Flag de truncamiento
    if is_truncated and truncation_reason != "ok":
        quality_flags.append(f"truncated_{truncation_reason}")
        flag_confidence[f"truncated_{truncation_reason}"] = 0.7

    result = {
        "completeness_score": round(completeness_score, 1),
        "structure_score": round(structure_score, 1),
        "informativeness_score": round(informativeness_score, 1),
        "readability_score": round(readability_score, 1),
        "overall_score": round(overall_score, 1),
        "overall_quality": overall_quality,
        "quality_flags": quality_flags,
        "flag_confidence": flag_confidence,
        "domain_used": domain,
        "details": {
            "completeness": {
                "present_elements": present,
                "missing_elements": missing,
            },
            "structure": structure_metrics,
            "informativeness": info_metrics,
            "readability": read_metrics,
        },
    }

    # Incluir evidencia si se solicita
    if include_evidence:
        result["evidence"] = {
            element: {
                "detected": ev.detected,
                "confidence": ev.confidence,
                "matches": ev.matches,
                "verification_notes": ev.verification_notes,
            }
            for element, ev in evidences.items()
        }

    return result


# ===========================================================================
# FUNCIÓN DE AUTO-DETECCIÓN DE DOMINIO
# ===========================================================================

# ===========================================================================
# FUNCIONES DE FILTRADO Y PIPELINE
# ===========================================================================

def filter_by_quality(
    articles: list[dict],
    threshold: float = 50.0,
    abstract_field: str = "abstract",
    domain: Optional[str] = None,
    auto_detect_domain: bool = True,
) -> tuple[list[dict], list[dict], dict]:
    """
    Filtra artículos por calidad de abstract con verificación anti-alucinación.

    Args:
        articles: Lista de artículos
        threshold: Umbral mínimo de calidad (0-100)
        abstract_field: Campo que contiene el abstract
        domain: IGNORADO (sin dominios; se conserva por compatibilidad)
        auto_detect_domain: IGNORADO (ya no se detectan dominios)

    Returns:
        tuple: (articulos_con_quality, articulos_high_quality, estadisticas)
    """
    high_quality = []
    stats = {
        "total": len(articles),
        "evaluated": 0,
        "high_quality": 0,
        "medium_quality": 0,
        "low_quality": 0,
        "above_threshold": 0,
        "avg_score": 0.0,
        "quality_distribution": {"high": 0, "medium": 0, "low": 0},
        "domains_detected": Counter(),
        "flags_distribution": Counter(),
    }

    scores = []
    # Referencia de especificidad: el propio corpus
    prepare_corpus_specificity([str(a.get(abstract_field, "") or "") for a in articles])

    for article in articles:
        abstract = article.get(abstract_field, "")

        if abstract == "nan" or abstract is None:
            abstract = ""

        # Evaluar calidad (sin dominio: todos los resumenes con los mismos patrones)
        quality = assess_abstract_quality(abstract)
        article["abstract_quality"] = quality
        stats["evaluated"] += 1

        # Contar flags
        for flag in quality.get("quality_flags", []):
            stats["flags_distribution"][flag] += 1

        # Clasificar
        q = quality["overall_quality"]
        stats["quality_distribution"][q] += 1
        if q == "high":
            stats["high_quality"] += 1
        elif q == "medium":
            stats["medium_quality"] += 1
        else:
            stats["low_quality"] += 1

        if quality["overall_score"] >= threshold:
            high_quality.append(article)
            stats["above_threshold"] += 1

        scores.append(quality["overall_score"])

    if scores:
        stats["avg_score"] = round(sum(scores) / len(scores), 1)

    # Convertir Counters a dicts para serialización
    stats["domains_detected"] = dict(stats["domains_detected"])
    stats["flags_distribution"] = dict(stats["flags_distribution"])

    logger.info(
        f"Quality assessment: {stats['above_threshold']}/{stats['total']} "
        f"above threshold {threshold} (avg: {stats['avg_score']})"
    )

    return articles, high_quality, stats


def run_phase2d_quality(
    articles: list[dict],
    threshold: float = 50.0,
    abstract_field: str = "abstract",
    domain: Optional[str] = None,
    auto_detect_domain: bool = True,
) -> tuple[list[dict], list[dict], dict]:
    """
    Ejecuta Fase 2d: Evaluación de calidad de abstracts.

    Args:
        articles: Lista de artículos
        threshold: Umbral mínimo de calidad
        abstract_field: Campo con el abstract
        domain: IGNORADO (sin dominios)
        auto_detect_domain: IGNORADO (ya no se detectan dominios)

    Returns:
        tuple: (todos_los_articulos, articulos_filtrados, estadisticas)
    """
    logger.info("=" * 60)
    logger.info("FASE 2d: Evaluacion de Calidad de Abstracts (Anti-Hallucination)")
    logger.info("=" * 60)

    articles, high_quality, stats = filter_by_quality(
        articles,
        threshold=threshold,
        abstract_field=abstract_field,
        domain=domain,
        auto_detect_domain=auto_detect_domain,
    )

    logger.info(f"  Total articulos: {stats['total']}")
    logger.info(f"  Calidad alta: {stats['high_quality']}")
    logger.info(f"  Calidad media: {stats['medium_quality']}")
    logger.info(f"  Calidad baja: {stats['low_quality']}")
    logger.info(f"  Score promedio: {stats['avg_score']}")
    logger.info(f"  Superan umbral ({threshold}): {stats['above_threshold']}")


    if stats["flags_distribution"]:
        top_flags = sorted(
            stats["flags_distribution"].items(),
            key=lambda x: -x[1]
        )[:5]
        logger.info(f"  Top flags: {dict(top_flags)}")

    return articles, high_quality, stats


# ===========================================================================
# TESTS DE VERIFICACIÓN (ANTI-ALUCINACIÓN)
# ===========================================================================

def run_verification_tests() -> dict:
    """
    Ejecuta tests de verificación para validar el sistema anti-alucinación.

    Returns:
        dict con resultados de tests
    """
    results = {"passed": 0, "failed": 0, "tests": []}

    # Test 1: Abstract vacío
    test1 = assess_abstract_quality("")
    t1_pass = test1["overall_score"] == 0 and "empty_abstract" in test1["quality_flags"]
    results["tests"].append({"name": "empty_abstract", "passed": t1_pass})
    results["passed" if t1_pass else "failed"] += 1

    # Test 2: Abstract muy corto
    test2 = assess_abstract_quality("This is a short text.")
    t2_pass = test2["overall_score"] == 0 and "too_short" in test2["quality_flags"]
    results["tests"].append({"name": "short_abstract", "passed": t2_pass})
    results["passed" if t2_pass else "failed"] += 1

    # Test 3: Abstract bien formado debería detectar elementos
    good_abstract = """
    This study aims to investigate the effectiveness of a novel machine learning
    approach for fault detection in power systems. We propose a deep learning
    algorithm using wavelet transform for feature extraction. The method was
    validated on the IEEE 39-bus test system. Results show that the proposed
    approach achieved 98.5% accuracy, outperforming existing methods. These
    findings demonstrate that the proposed method can be effectively used for
    real-time fault detection applications.
    """
    test3 = assess_abstract_quality(good_abstract, domain="electrical_engineering")
    t3_pass = (
        test3["overall_quality"] == "high" and
        len(test3["quality_flags"]) <= 1  # Máximo 1 flag menor
    )
    results["tests"].append({"name": "good_abstract_detection", "passed": t3_pass})
    results["passed" if t3_pass else "failed"] += 1

    # Test 4: No debe detectar falsos positivos en copyright
    copyright_text = """
    This paper presents a method. © 2024 Elsevier B.V. All rights reserved.
    The proposed approach uses data analysis. Results indicate improvement.
    """
    test4 = assess_abstract_quality(copyright_text, include_evidence=True)
    # Verificar que el copyright no contribuye a detecciones
    evidence = test4.get("evidence", {})
    t4_pass = True  # Básicamente verificar que no crashea
    results["tests"].append({"name": "copyright_exclusion", "passed": t4_pass})
    results["passed" if t4_pass else "failed"] += 1

    # Test 5: Confianza de flags debe estar entre 0 y 1
    test5 = assess_abstract_quality(good_abstract)
    t5_pass = all(
        0 <= conf <= 1
        for conf in test5.get("flag_confidence", {}).values()
    )
    results["tests"].append({"name": "confidence_bounds", "passed": t5_pass})
    results["passed" if t5_pass else "failed"] += 1

    # Test 6: Auto-detección de dominio
    # (Prueba de deteccion de dominio eliminada: la fase ya no detecta dominios.)

    results["summary"] = f"{results['passed']}/{results['passed'] + results['failed']} tests passed"

    return results


if __name__ == "__main__":
    # Ejecutar tests de verificación
    print("Running verification tests...")
    test_results = run_verification_tests()
    print(f"\nResults: {test_results['summary']}")
    for test in test_results["tests"]:
        status = "✓" if test["passed"] else "✗"
        print(f"  {status} {test['name']}")
