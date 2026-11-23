"""Fase 2b: Analisis profundo de abstracts - topic modeling, gaps, metricas."""
from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import FIGURES_DIR, REPORTS_DIR, BASE_DIR

logger = logging.getLogger("bibliometrico.abstracts")

# Pre-compiled performance metric patterns (avoid per-abstract recompilation)
_COMPILED_PERF_PATTERNS = [
    (re.compile(r"(?i)(accuracy|acc)\s*(?:of\s*)?(\d+\.?\d*)\s*%"), "accuracy"),
    (re.compile(r"(?i)(f1[- ]?score?|f1)\s*(?:of\s*)?(\d+\.?\d*)"), "f1_score"),
    (re.compile(r"(?i)(auc|auroc)\s*(?:of\s*)?(\d+\.?\d*)"), "auc"),
    (re.compile(r"(?i)(sensitivity|recall)\s*(?:of\s*)?(\d+\.?\d*)\s*%?"), "sensitivity"),
    (re.compile(r"(?i)(specificity)\s*(?:of\s*)?(\d+\.?\d*)\s*%?"), "specificity"),
    (re.compile(r"(?i)(precision)\s*(?:of\s*)?(\d+\.?\d*)"), "precision"),
]

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight"})

# ---------------------------------------------------------------------------
# Preprocesamiento de texto
# ---------------------------------------------------------------------------

# Stopwords minimas (sin dependencia de NLTK)
STOPWORDS = set("""
a about above after again against all am an and any are aren't as at be because
been before being below between both but by can't cannot could couldn't did didn't
do does doesn't doing don't down during each few for from further get got had hadn't
has hasn't have haven't having he her here hers herself him himself his how how's
i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most
mustn't my myself no nor not of off on once only or other ought our ours ourselves
out over own same shan't she she'd she'll she's should shouldn't so some such than
that that's the their theirs them themselves then there there's these they they'd
they'll they're they've this those through to too under until up very was wasn't
we we'd we'll we're we've were weren't what what's when when's where where's which
while who who's whom why why's with won't would wouldn't you you'd you'll you're
you've your yours yourself yourselves also using used use based however may one two
three four five new proposed method study results show can will using paper article
approach work propose present data analysis model models performance system
""".split())


def tokenize(text: str) -> list[str]:
    """Tokeniza y limpia texto."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    tokens = text.split()
    return [t for t in tokens if len(t) > 2 and t not in STOPWORDS]


def get_ngrams(tokens: list[str], n: int = 2) -> list[str]:
    """Genera n-gramas."""
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


# ---------------------------------------------------------------------------
# TF-IDF manual (sin sklearn)
# ---------------------------------------------------------------------------

def compute_tfidf(corpus_tokens: list[list[str]], top_n: int = 30) -> list[tuple[str, float]]:
    """Calcula TF-IDF simple sobre un corpus tokenizado."""
    n_docs = len(corpus_tokens)
    if n_docs == 0:
        return []

    # Document frequency
    df = Counter()
    for tokens in corpus_tokens:
        for term in set(tokens):
            df[term] += 1

    # TF-IDF global (suma de TF-IDF por termino)
    tfidf_scores = Counter()
    for tokens in corpus_tokens:
        tf = Counter(tokens)
        n_tokens = len(tokens) if tokens else 1
        for term, count in tf.items():
            tf_val = count / n_tokens
            idf_val = np.log(n_docs / (1 + df[term])) + 1
            tfidf_scores[term] += tf_val * idf_val

    return tfidf_scores.most_common(top_n)


# ---------------------------------------------------------------------------
# Topic Modeling (LDA simplificado con NMF via numpy)
# ---------------------------------------------------------------------------

def simple_topic_modeling(
    articles: list[dict], n_topics: int = 5, top_words: int = 10
) -> dict:
    """Topic modeling basico usando NMF sobre TF-IDF.
    Evita dependencia de sklearn usando numpy directamente.
    """
    # Construir vocabulario
    corpus_tokens = []
    valid_articles = []
    for a in articles:
        abstract = a.get("abstract", "")
        if not abstract or abstract == "nan" or len(abstract) < 50:
            continue
        tokens = tokenize(abstract)
        if tokens:
            corpus_tokens.append(tokens)
            valid_articles.append(a)

    if len(corpus_tokens) < n_topics:
        logger.warning(f"Not enough articles ({len(corpus_tokens)}) for {n_topics} topics")
        return {}

    # Vocabulario (top 500 terminos por DF)
    df = Counter()
    for tokens in corpus_tokens:
        for t in set(tokens):
            df[t] += 1

    # Filtrar: al menos en 2 docs, no mas del 80%
    min_df = 2
    max_df = int(len(corpus_tokens) * 0.8)
    vocab = [t for t, c in df.most_common(500) if min_df <= c <= max_df]
    vocab_idx = {t: i for i, t in enumerate(vocab)}

    if len(vocab) < n_topics * top_words:
        logger.warning(f"Vocabulary too small ({len(vocab)}) for topic modeling")
        return {}

    # Construir matriz TF-IDF
    n_docs = len(corpus_tokens)
    n_terms = len(vocab)
    tfidf_matrix = np.zeros((n_docs, n_terms))

    for i, tokens in enumerate(corpus_tokens):
        tf = Counter(tokens)
        n_tok = len(tokens) if tokens else 1
        for term, count in tf.items():
            if term in vocab_idx:
                j = vocab_idx[term]
                tf_val = count / n_tok
                idf_val = np.log(n_docs / (1 + df[term])) + 1
                tfidf_matrix[i, j] = tf_val * idf_val

    # NMF simple via iteracion multiplicativa
    try:
        # Reduce iterations for large corpora to avoid excessive computation
        nmf_iter = 100 if len(corpus_tokens) > 500 else 200
        W, H = _nmf(tfidf_matrix, n_components=n_topics, max_iter=nmf_iter)
    except Exception as e:
        logger.warning(f"NMF failed: {e}")
        return {}

    # Extraer topics
    topics = {}
    for k in range(n_topics):
        top_indices = H[k].argsort()[-top_words:][::-1]
        topic_words = [(vocab[j], round(float(H[k, j]), 4)) for j in top_indices]
        topics[f"topic_{k}"] = topic_words

    # Asignar topic dominante a cada articulo
    topic_assignments = W.argmax(axis=1)
    for i, a in enumerate(valid_articles):
        a["dominant_topic"] = int(topic_assignments[i])
        a["topic_weights"] = [round(float(w), 4) for w in W[i]]

    # Conteo por topic
    topic_counts = Counter(int(t) for t in topic_assignments)

    logger.info(f"Topics extracted: {n_topics}, Vocabulary: {len(vocab)}")
    return {
        "topics": topics,
        "topic_counts": dict(topic_counts),
        "n_articles_analyzed": len(valid_articles),
        "vocabulary_size": len(vocab),
    }


def _nmf(V: np.ndarray, n_components: int, max_iter: int = 200) -> tuple:
    """Non-negative Matrix Factorization (NMF) via actualizacion multiplicativa."""
    n, m = V.shape
    np.random.seed(42)
    W = np.random.rand(n, n_components) + 0.1
    H = np.random.rand(n_components, m) + 0.1
    eps = 1e-10

    for _ in range(max_iter):
        # Actualizar H
        numerator = W.T @ V
        denominator = W.T @ W @ H + eps
        H *= numerator / denominator

        # Actualizar W
        numerator = V @ H.T
        denominator = W @ H @ H.T + eps
        W *= numerator / denominator

    return W, H


# ---------------------------------------------------------------------------
# Extraccion de metricas de rendimiento de abstracts
# ---------------------------------------------------------------------------

def extract_performance_metrics(abstract: str) -> list[dict]:
    """Extrae metricas de rendimiento reportadas en abstracts.
    Ej: 'accuracy of 99.4%', 'F1 score of 0.92', 'AUC of 0.98'
    """
    if not abstract or abstract == "nan":
        return []

    metrics = []

    for cpat, metric_name in _COMPILED_PERF_PATTERNS:
        for match in cpat.finditer(abstract):
            value = float(match.group(2))
            # Normalizar: si >1 asumir porcentaje
            if value > 1 and metric_name in ("f1_score", "auc", "precision"):
                value = value / 100.0
            metrics.append({
                "metric": metric_name,
                "value": value,
                "raw_match": match.group(0),
            })

    return metrics


# ---------------------------------------------------------------------------
# Extraccion de metodos y datasets mencionados
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Metodos: SIN lista de metodos. Se usan los metodos que el propio corpus declara
# (src/metodos_declarados.py, el mismo detector que la Fase 2f). La lista fija anterior
# (_DEFAULT_METHOD_PATTERNS: experimental, survey, anova, rasch, machine learning...) se elimino.
# domain_patterns.json (opcional, manual) sigue aceptando 'datasets' y 'applications'.
# ---------------------------------------------------------------------------
_METODOS_CORPUS: set[str] = set()


def _compile_word_boundary_patterns(patterns: list[str]) -> list[tuple[str, re.Pattern]]:
    """Compile patterns with word boundaries to prevent substring false positives."""
    compiled = []
    for p in patterns:
        p_clean = p.strip().lower()
        if not p_clean:
            continue
        canonical = p_clean
        regex = re.compile(r"(?i)\b" + re.escape(p_clean) + r"\b")
        compiled.append((canonical, regex))
    return compiled


def _load_domain_patterns() -> tuple[list[str], list[str], list[str]]:
    """Carga patrones de dominio con prioridad:
    1. domain_patterns.json (override manual)
    2. Defaults cientificos multidisciplinares agnosticos
    """
    import json

    patterns_path = BASE_DIR / "domain_patterns.json"
    if patterns_path.exists():
        try:
            with open(patterns_path, encoding="utf-8") as f:
                data = json.load(f)
            if data.get("methods"):
                logger.warning(
                    "domain_patterns.json trae 'methods': se ignora. Los metodos se leen de lo que "
                    "declara el corpus (src/metodos_declarados.py)."
                )
            methods: list[str] = []
            datasets = data.get("datasets", [])
            applications = data.get("applications", [])
            logger.info(
                f"Domain patterns (manual): "
                f"{len(methods)} methods, {len(datasets)} datasets, {len(applications)} applications"
            )
            return methods, datasets, applications
        except Exception as e:
            logger.warning(f"Error loading domain_patterns.json: {e}")

    return [], [], []


METHOD_PATTERNS, DATASET_PATTERNS, APPLICATION_PATTERNS = _load_domain_patterns()
_COMPILED_DATASETS = _compile_word_boundary_patterns(DATASET_PATTERNS)
_COMPILED_APPLICATIONS = _compile_word_boundary_patterns(APPLICATION_PATTERNS)


def extract_methods_datasets(abstract: str, article: Optional[dict] = None) -> dict:
    """Extrae metodos (con word-boundaries), temas OpenAlex y conceptos de un articulo."""
    text_parts = []
    if abstract and abstract != "nan":
        text_parts.append(str(abstract))
    if article and article.get("title"):
        text_parts.append(str(article.get("title", "")))
    text = " ".join(text_parts)

    if not text.strip():
        return {"methods": [], "datasets": [], "applications": []}

    from src.metodos_declarados import article_methods
    if not _METODOS_CORPUS:                       # llamada fuera de run_phase2b: vocabulario del modelo
        from src.modelo_corpus import get_model
        _METODOS_CORPUS.update(get_model()["metodos"]["vocabulario"])
    methods = set(article_methods(article or {"title": "", "abstract": text}, _METODOS_CORPUS))

    if _COMPILED_DATASETS:
        datasets = {
            canonical for canonical, regex in _COMPILED_DATASETS
            if regex.search(text)
        }
    elif article:
        # Data-driven themes from OpenAlex CWTS Leiden topics
        datasets = set()
        pt = article.get("openalex", {}).get("primary_topic", {}) or {}
        if pt.get("name"):
            datasets.add(pt["name"])
        elif pt.get("subfield"):
            datasets.add(pt["subfield"])
        for t in (article.get("topics") or article.get("openalex", {}).get("topics", []))[:2]:
            t_name = (t.get("display_name") or t.get("name") or "").strip()
            if t_name:
                datasets.add(t_name)
    else:
        datasets = set()

    if _COMPILED_APPLICATIONS:
        applications = {
            canonical for canonical, regex in _COMPILED_APPLICATIONS
            if regex.search(text)
        }
    elif article:
        # Data-driven key concepts from author keywords / manual_tags
        applications = set()
        for kw in (article.get("manual_tags") or article.get("automatic_tags") or [])[:5]:
            if isinstance(kw, str) and len(kw.strip()) > 2:
                applications.add(kw.strip().lower())
    else:
        applications = set()

    return {
        "methods": sorted(methods),
        "datasets": sorted(datasets),
        "applications": sorted(applications),
    }


# ---------------------------------------------------------------------------
# Deteccion de gaps desde abstracts
# ---------------------------------------------------------------------------

def extract_gaps_from_abstracts(articles: list[dict]) -> list[dict]:
    """Vacios declarados en los resumenes: detector unico compartido con la Fase 2f
    (src/vacios_declarados.py). La lista propia de 34 frases de esta fase se elimino."""
    from src.vacios_declarados import extract_declared_gaps
    gaps = extract_declared_gaps(articles)
    logger.info(f"Knowledge gaps extracted: {len(gaps)} from {len(articles)} articles")
    return gaps


# ---------------------------------------------------------------------------
# Visualizaciones
# ---------------------------------------------------------------------------

def plot_topics(topic_data: dict, output_dir: Path = FIGURES_DIR) -> None:
    """Visualiza topics."""
    topics = topic_data.get("topics", {})
    if not topics:
        return

    n_topics = len(topics)
    fig, axes = plt.subplots(1, n_topics, figsize=(5 * n_topics, 6))
    if n_topics == 1:
        axes = [axes]

    for idx, (topic_name, words) in enumerate(topics.items()):
        ax = axes[idx]
        word_labels = [w for w, _ in words[:8]]
        word_scores = [s for _, s in words[:8]]
        colors = sns.color_palette("deep", n_colors=8)
        ax.barh(range(len(word_labels)), word_scores, color=colors)
        ax.set_yticks(range(len(word_labels)))
        ax.set_yticklabels(word_labels, fontsize=9)
        ax.set_title(f"Topic {idx}", fontsize=11)
        ax.invert_yaxis()

    plt.suptitle("Topic Modeling (NMF on Abstracts)", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "topics_nmf.png")
    plt.close(fig)
    logger.info("Saved: topics_nmf.png")


def plot_topic_evolution(articles: list[dict], output_dir: Path = FIGURES_DIR) -> None:
    """Evolucion temporal de topics."""
    data = [(a.get("publication_year"), a.get("dominant_topic"))
            for a in articles if a.get("dominant_topic") is not None and a.get("publication_year")]
    if not data:
        return

    df = pd.DataFrame(data, columns=["year", "topic"])
    ct = pd.crosstab(df["year"], df["topic"])

    fig, ax = plt.subplots(figsize=(10, 5))
    ct.plot(kind="bar", stacked=True, ax=ax, colormap="Set2")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Articles")
    ax.set_title("Topic Evolution Over Time")
    ax.legend(title="Topic", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(output_dir / "topic_evolution.png")
    plt.close(fig)
    logger.info("Saved: topic_evolution.png")


def plot_methods_datasets(articles: list[dict], output_dir: Path = FIGURES_DIR) -> None:
    """Visualiza frecuencia de metodos y datasets."""
    method_counts = Counter()
    dataset_counts = Counter()
    app_counts = Counter()

    for a in articles:
        md = a.get("extracted_methods_datasets", {})
        for m in md.get("methods", []):
            method_counts[m] += 1
        for d in md.get("datasets", []):
            dataset_counts[d] += 1
        for ap in md.get("applications", []):
            app_counts[ap] += 1

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Methods
    if method_counts:
        top_m = method_counts.most_common(15)
        axes[0].barh([m for m, _ in top_m], [c for _, c in top_m],
                     color=sns.color_palette("deep")[0])
        axes[0].set_title("Methods Mentioned")
        axes[0].invert_yaxis()

    # Datasets
    if dataset_counts:
        top_d = dataset_counts.most_common(15)
        axes[1].barh([d for d, _ in top_d], [c for _, c in top_d],
                     color=sns.color_palette("deep")[1])
        axes[1].set_title("Themes Detected")
        axes[1].invert_yaxis()

    # Applications / Concepts
    if app_counts:
        top_a = app_counts.most_common(15)
        axes[2].barh([a for a, _ in top_a], [c for _, c in top_a],
                     color=sns.color_palette("deep")[2])
        axes[2].set_title("Key Concepts")
        axes[2].invert_yaxis()

    plt.suptitle("Methods, Themes & Concepts in Corpus", fontsize=13)
    plt.tight_layout()
    fig.savefig(output_dir / "methods_datasets_apps.png")
    plt.close(fig)
    logger.info("Saved: methods_datasets_apps.png")


def plot_performance_summary(articles: list[dict], output_dir: Path = FIGURES_DIR) -> None:
    """Visualiza metricas de rendimiento extraidas."""
    all_metrics = []
    for a in articles:
        for m in a.get("performance_metrics", []):
            all_metrics.append({
                "key": a["key"],
                "year": a.get("publication_year"),
                "metric": m["metric"],
                "value": m["value"],
            })

    if not all_metrics:
        logger.info("No performance metrics to plot")
        return

    df = pd.DataFrame(all_metrics)

    # Box plot por tipo de metrica
    fig, ax = plt.subplots(figsize=(10, 5))
    metric_order = ["accuracy", "f1_score", "auc", "sensitivity", "specificity", "precision"]
    present = [m for m in metric_order if m in df["metric"].values]

    if present:
        df_filtered = df[df["metric"].isin(present)]
        sns.boxplot(data=df_filtered, x="metric", y="value", hue="metric", ax=ax,
                    order=present, hue_order=present, palette="deep", legend=False)
        ax.set_title("Reported Performance Metrics in Abstracts")
        ax.set_xlabel("Metric")
        ax.set_ylabel("Value")
        # Agregar conteo
        for i, m in enumerate(present):
            n = len(df_filtered[df_filtered["metric"] == m])
            ax.text(i, ax.get_ylim()[1], f"n={n}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_dir / "performance_metrics.png")
    plt.close(fig)
    logger.info("Saved: performance_metrics.png")


def plot_method_application_heatmap(articles: list[dict], output_dir: Path = FIGURES_DIR) -> None:
    """Heatmap metodo vs aplicacion para detectar vacios."""
    pairs = Counter()
    for a in articles:
        md = a.get("extracted_methods_datasets", {})
        methods = md.get("methods", [])
        apps = md.get("applications", [])
        for m in methods:
            for ap in apps:
                pairs[(m, ap)] += 1

    if not pairs:
        return

    # Top methods y apps
    method_counts = Counter()
    app_counts = Counter()
    for (m, ap), c in pairs.items():
        method_counts[m] += c
        app_counts[ap] += c

    top_methods = [m for m, _ in method_counts.most_common(12)]
    top_apps = [a for a, _ in app_counts.most_common(10)]

    # Crear matriz
    matrix = np.zeros((len(top_methods), len(top_apps)))
    for i, m in enumerate(top_methods):
        for j, ap in enumerate(top_apps):
            matrix[i, j] = pairs.get((m, ap), 0)

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(matrix, xticklabels=top_apps, yticklabels=top_methods,
                annot=True, fmt=".0f", cmap="YlOrRd", ax=ax,
                linewidths=0.5, cbar_kws={"label": "Number of Articles"})
    ax.set_title("Method x Concept Matrix (gaps = empty cells)")
    ax.set_xlabel("Concept")
    ax.set_ylabel("Method")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)

    plt.tight_layout()
    fig.savefig(output_dir / "method_application_heatmap.png")
    plt.close(fig)
    logger.info("Saved: method_application_heatmap.png")


# ---------------------------------------------------------------------------
# Analisis cruzados: metodos x año, metodos x pais, keywords x año, etc.
# ---------------------------------------------------------------------------

def _get_countries(article: dict) -> list[str]:
    """Paises de un articulo: definicion unica del modelo del corpus (autores UNION instituciones, ISO2)."""
    from src.modelo_corpus import article_countries
    return article_countries(article)


def _get_first_author_last_name(article: dict) -> str:
    """Extrae apellido del primer autor."""
    authors = article.get("authors", [])
    if not authors:
        return "Unknown"
    return authors[0].split(",")[0].strip()


def plot_methods_by_year(articles: list[dict], output_dir: Path = FIGURES_DIR) -> None:
    """Heatmap: metodos por año - muestra evolucion metodologica."""
    rows = []
    for a in articles:
        year = a.get("publication_year")
        if not year:
            continue
        for m in a.get("extracted_methods_datasets", {}).get("methods", []):
            rows.append({"year": year, "method": m})

    if not rows:
        return

    df = pd.DataFrame(rows)
    ct = pd.crosstab(df["method"], df["year"])

    # Filtrar metodos con al menos 3 apariciones
    ct = ct[ct.sum(axis=1) >= 3]
    ct = ct.loc[ct.sum(axis=1).sort_values(ascending=False).index]

    if ct.empty:
        return

    fig, ax = plt.subplots(figsize=(14, max(6, len(ct) * 0.45)))
    sns.heatmap(ct, annot=True, fmt="d", cmap="YlGnBu", ax=ax,
                linewidths=0.5, cbar_kws={"label": "Articles"})
    ax.set_title("Methods Evolution by Year", fontsize=13, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("")
    plt.yticks(fontsize=9)
    plt.tight_layout()
    fig.savefig(output_dir / "methods_by_year.png", dpi=150)
    plt.close(fig)
    logger.info("Saved: methods_by_year.png")


def plot_methods_by_country(articles: list[dict], output_dir: Path = FIGURES_DIR) -> None:
    """Heatmap: metodos por pais."""
    rows = []
    for a in articles:
        countries = _get_countries(a)
        for m in a.get("extracted_methods_datasets", {}).get("methods", []):
            for c in countries:
                rows.append({"country": c, "method": m})

    if not rows:
        return

    df = pd.DataFrame(rows)
    ct = pd.crosstab(df["method"], df["country"])

    # Filtrar
    top_countries = ct.sum(axis=0).sort_values(ascending=False).head(12).index
    ct = ct[top_countries]
    ct = ct[ct.sum(axis=1) >= 3]
    ct = ct.loc[ct.sum(axis=1).sort_values(ascending=False).index]

    if ct.empty:
        return

    fig, ax = plt.subplots(figsize=(14, max(6, len(ct) * 0.45)))
    sns.heatmap(ct, annot=True, fmt="d", cmap="OrRd", ax=ax,
                linewidths=0.5, cbar_kws={"label": "Articles"})
    ax.set_title("Methods by Country", fontsize=13, fontweight="bold")
    ax.set_xlabel("Country Code")
    ax.set_ylabel("")
    plt.yticks(fontsize=9)
    plt.tight_layout()
    fig.savefig(output_dir / "methods_by_country.png", dpi=150)
    plt.close(fig)
    logger.info("Saved: methods_by_country.png")


def plot_datasets_by_year(articles: list[dict], output_dir: Path = FIGURES_DIR) -> None:
    """Heatmap: datasets por año."""
    rows = []
    for a in articles:
        year = a.get("publication_year")
        if not year:
            continue
        for d in a.get("extracted_methods_datasets", {}).get("datasets", []):
            rows.append({"year": year, "dataset": d})

    if not rows:
        return

    df = pd.DataFrame(rows)
    ct = pd.crosstab(df["dataset"], df["year"])
    ct = ct[ct.sum(axis=1) >= 2]
    ct = ct.loc[ct.sum(axis=1).sort_values(ascending=False).index]

    if ct.empty:
        return

    fig, ax = plt.subplots(figsize=(14, max(5, len(ct) * 0.5)))
    sns.heatmap(ct, annot=True, fmt="d", cmap="PuBu", ax=ax,
                linewidths=0.5, cbar_kws={"label": "Articles"})
    ax.set_title("Themes Over Time", fontsize=13, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("")
    plt.yticks(fontsize=9)
    plt.tight_layout()
    fig.savefig(output_dir / "datasets_by_year.png", dpi=150)
    plt.close(fig)
    logger.info("Saved: datasets_by_year.png")


def plot_journals_by_topic(articles: list[dict], output_dir: Path = FIGURES_DIR) -> None:
    """Heatmap: revistas por topic dominante."""
    rows = []
    for a in articles:
        topic = a.get("dominant_topic")
        journal = a.get("publication_title", "")
        if topic is not None and journal and journal != "nan":
            rows.append({"journal": journal[:35], "topic": f"Topic {topic}"})

    if not rows:
        return

    df = pd.DataFrame(rows)
    ct = pd.crosstab(df["journal"], df["topic"])

    # Top revistas
    ct = ct[ct.sum(axis=1) >= 2]
    ct = ct.loc[ct.sum(axis=1).sort_values(ascending=False).head(20).index]

    if ct.empty:
        return

    fig, ax = plt.subplots(figsize=(10, max(5, len(ct) * 0.4)))
    sns.heatmap(ct, annot=True, fmt="d", cmap="YlOrBr", ax=ax,
                linewidths=0.5, cbar_kws={"label": "Articles"})
    ax.set_title("Journals by Dominant Topic", fontsize=13, fontweight="bold")
    ax.set_xlabel("Topic")
    ax.set_ylabel("")
    plt.yticks(fontsize=8)
    plt.tight_layout()
    fig.savefig(output_dir / "journals_by_topic.png", dpi=150)
    plt.close(fig)
    logger.info("Saved: journals_by_topic.png")


def plot_author_method_specialization(articles: list[dict], output_dir: Path = FIGURES_DIR) -> None:
    """Sankey-like: autores mas productivos y sus metodos preferidos."""
    author_methods = Counter()
    author_count = Counter()

    for a in articles:
        methods = a.get("extracted_methods_datasets", {}).get("methods", [])
        for auth in a.get("authors", []):
            last_name = auth.split(",")[0].strip()
            author_count[last_name] += 1
            for m in methods:
                author_methods[(last_name, m)] += 1

    # Top autores (>=2 articulos)
    top_authors = [a for a, c in author_count.most_common(15) if c >= 2]
    # Top metodos
    method_totals = Counter()
    for (a, m), c in author_methods.items():
        if a in top_authors:
            method_totals[m] += c
    top_methods = [m for m, _ in method_totals.most_common(10)]

    if not top_authors or not top_methods:
        return

    matrix = np.zeros((len(top_authors), len(top_methods)))
    for i, a in enumerate(top_authors):
        for j, m in enumerate(top_methods):
            matrix[i, j] = author_methods.get((a, m), 0)

    fig, ax = plt.subplots(figsize=(12, max(5, len(top_authors) * 0.45)))
    sns.heatmap(matrix, xticklabels=top_methods, yticklabels=top_authors,
                annot=True, fmt=".0f", cmap="BuPu", ax=ax,
                linewidths=0.5, cbar_kws={"label": "Articles"})
    ax.set_title("Author Specialization by Method", fontsize=13, fontweight="bold")
    ax.set_xlabel("Method")
    ax.set_ylabel("Author")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    fig.savefig(output_dir / "author_method_specialization.png", dpi=150)
    plt.close(fig)
    logger.info("Saved: author_method_specialization.png")


def plot_keyword_trends(articles: list[dict], output_dir: Path = FIGURES_DIR) -> None:
    """Evolucion temporal de keywords principales."""
    rows = []
    for a in articles:
        year = a.get("publication_year")
        if not year:
            continue
        for kw in a.get("manual_tags", []):
            rows.append({"year": year, "keyword": kw.lower()})

    if not rows:
        return

    df = pd.DataFrame(rows)
    # Top keywords globales
    top_kw = df["keyword"].value_counts().head(12).index.tolist()
    df_filtered = df[df["keyword"].isin(top_kw)]
    ct = pd.crosstab(df_filtered["keyword"], df_filtered["year"])

    fig, ax = plt.subplots(figsize=(12, 6))
    ct.T.plot(kind="line", ax=ax, marker="o", linewidth=2)
    ax.set_xlabel("Year")
    ax.set_ylabel("Frequency")
    ax.set_title("Keyword Trends Over Time", fontsize=13, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7, title="Keyword")
    plt.tight_layout()
    fig.savefig(output_dir / "keyword_trends.png", dpi=150)
    plt.close(fig)
    logger.info("Saved: keyword_trends.png")


def plot_country_collaboration_heatmap(articles: list[dict], output_dir: Path = FIGURES_DIR) -> None:
    """Heatmap de co-autoria entre paises."""
    pairs = Counter()
    for a in articles:
        countries = sorted(set(_get_countries(a)))
        for i in range(len(countries)):
            for j in range(i + 1, len(countries)):
                pairs[(countries[i], countries[j])] += 1

    if not pairs:
        return

    # Todos los paises involucrados en colaboraciones
    all_countries = set()
    for (c1, c2), _ in pairs.items():
        all_countries.add(c1)
        all_countries.add(c2)

    # Ordenar por frecuencia total
    country_freq = Counter()
    for (c1, c2), n in pairs.items():
        country_freq[c1] += n
        country_freq[c2] += n
    top_countries = [c for c, _ in country_freq.most_common(15)]

    matrix = np.zeros((len(top_countries), len(top_countries)))
    for i, c1 in enumerate(top_countries):
        for j, c2 in enumerate(top_countries):
            if i != j:
                pair = tuple(sorted([c1, c2]))
                matrix[i, j] = pairs.get(pair, 0)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix, xticklabels=top_countries, yticklabels=top_countries,
                annot=True, fmt=".0f", cmap="Greens", ax=ax,
                linewidths=0.5, cbar_kws={"label": "Co-authored Articles"})
    ax.set_title("International Collaboration Heatmap", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / "country_collaboration.png", dpi=150)
    plt.close(fig)
    logger.info("Saved: country_collaboration.png")


def plot_performance_by_method(articles: list[dict], output_dir: Path = FIGURES_DIR) -> None:
    """Compara metricas de rendimiento por metodo (accuracy)."""
    rows = []
    for a in articles:
        methods = a.get("extracted_methods_datasets", {}).get("methods", [])
        for m_info in a.get("performance_metrics", []):
            if m_info["metric"] == "accuracy" and m_info["value"] <= 100:
                for method in methods:
                    rows.append({"method": method, "accuracy": m_info["value"]})

    if len(rows) < 5:
        return

    df = pd.DataFrame(rows)
    # Metodos con al menos 3 data points
    method_counts = df["method"].value_counts()
    valid_methods = method_counts[method_counts >= 3].index.tolist()
    df = df[df["method"].isin(valid_methods)]

    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    order = df.groupby("method")["accuracy"].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x="method", y="accuracy", ax=ax, order=order,
                hue="method", palette="Set2", legend=False)
    sns.stripplot(data=df, x="method", y="accuracy", ax=ax, order=order,
                  color="black", alpha=0.4, size=4)
    ax.set_title("Reported Accuracy by Method", fontsize=13, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Accuracy (%)")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.tight_layout()
    fig.savefig(output_dir / "performance_by_method.png", dpi=150)
    plt.close(fig)
    logger.info("Saved: performance_by_method.png")


# ---------------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------------

def generate_abstract_report(
    articles: list[dict], topic_data: dict, gaps: list[dict],
    output_dir: Path = REPORTS_DIR,
) -> None:
    """Genera reporte Markdown del analisis de abstracts."""
    lines = ["# Abstract Analysis Report\n"]

    # Topic modeling
    if topic_data:
        lines.append(f"## Topic Modeling ({topic_data.get('n_articles_analyzed', '?')} abstracts)")
        for topic_name, words in topic_data.get("topics", {}).items():
            word_str = ", ".join(f"{w} ({s:.3f})" for w, s in words[:8])
            lines.append(f"- **{topic_name}**: {word_str}")
        lines.append("")

    # Methods & Datasets
    method_counts = Counter()
    dataset_counts = Counter()
    for a in articles:
        md = a.get("extracted_methods_datasets", {})
        for m in md.get("methods", []):
            method_counts[m] += 1
        for d in md.get("datasets", []):
            dataset_counts[d] += 1

    lines.append("## Most Used Methods")
    for m, c in method_counts.most_common(15):
        lines.append(f"- {m}: {c} articles")
    lines.append("")

    lines.append("## Most Frequent Themes")
    for d, c in dataset_counts.most_common(15):
        lines.append(f"- {d}: {c} articles")
    lines.append("")

    # Performance metrics summary
    all_metrics = [m for a in articles for m in a.get("performance_metrics", [])]
    if all_metrics:
        lines.append(f"## Performance Metrics ({len(all_metrics)} extractions)")
        by_type = {}
        for m in all_metrics:
            by_type.setdefault(m["metric"], []).append(m["value"])
        for metric_name, values in sorted(by_type.items()):
            arr = np.array(values)
            lines.append(
                f"- **{metric_name}**: n={len(arr)}, "
                f"mean={arr.mean():.3f}, std={arr.std():.3f}, "
                f"min={arr.min():.3f}, max={arr.max():.3f}"
            )
        lines.append("")

    # Knowledge gaps
    if gaps:
        lines.append(f"## Knowledge Gaps Detected ({len(gaps)} sentences)")
        for g in gaps[:30]:
            lines.append(f"- [{g['key']}] ({g['year']}) *\"{g['sentence'][:150]}...\"*")
        lines.append("")

    report_path = output_dir / "abstract_analysis_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Report saved: {report_path}")


# ---------------------------------------------------------------------------
# Fase principal
# ---------------------------------------------------------------------------

def run_phase2b(articles: list[dict]) -> dict:
    """Ejecuta Fase 2b: Analisis profundo de abstracts."""
    logger.info("=== FASE 2b: Analisis de Abstracts ===")

    # 1. Extraer metodos (declarados por el corpus, sin lista), datasets, aplicaciones
    from src.modelo_corpus import get_model
    _METODOS_CORPUS.clear()
    _METODOS_CORPUS.update(get_model(articles)["metodos"]["vocabulario"])   # vocabulario del modelo del corpus
    for a in articles:
        a["extracted_methods_datasets"] = extract_methods_datasets(a.get("abstract", ""), article=a)
        a["performance_metrics"] = extract_performance_metrics(a.get("abstract", ""))

    # Estadisticas rapidas
    with_methods = sum(1 for a in articles if a["extracted_methods_datasets"]["methods"])
    with_datasets = sum(1 for a in articles if a["extracted_methods_datasets"]["datasets"])
    with_metrics = sum(1 for a in articles if a["performance_metrics"])
    logger.info(f"  Methods detected: {with_methods}/{len(articles)}")
    logger.info(f"  Datasets detected: {with_datasets}/{len(articles)}")
    logger.info(f"  Performance metrics: {with_metrics}/{len(articles)}")

    # 2. Topic modeling
    topic_data = simple_topic_modeling(articles, n_topics=5, top_words=10)

    # 3. Deteccion de gaps
    gaps = extract_gaps_from_abstracts(articles)

    # 4. TF-IDF global
    corpus_tokens = [tokenize(a.get("abstract", "")) for a in articles
                     if a.get("abstract") and a["abstract"] != "nan"]
    tfidf_top = compute_tfidf(corpus_tokens, top_n=30)

    # 5. Bigrams
    all_bigrams = Counter()
    for tokens in corpus_tokens:
        for bg in get_ngrams(tokens, 2):
            all_bigrams[bg] += 1
    top_bigrams = all_bigrams.most_common(20)

    # 6. Visualizaciones basicas
    plot_topics(topic_data)
    plot_topic_evolution(articles)
    plot_methods_datasets(articles)
    plot_performance_summary(articles)
    plot_method_application_heatmap(articles)

    # 7. Analisis cruzados
    plot_methods_by_year(articles)
    plot_methods_by_country(articles)
    plot_datasets_by_year(articles)
    plot_journals_by_topic(articles)
    plot_author_method_specialization(articles)
    plot_keyword_trends(articles)
    plot_country_collaboration_heatmap(articles)
    plot_performance_by_method(articles)

    # TF-IDF plot
    if tfidf_top:
        fig, ax = plt.subplots(figsize=(10, 7))
        terms = [t for t, _ in tfidf_top[:20]]
        scores = [s for _, s in tfidf_top[:20]]
        ax.barh(range(len(terms)), scores, color=sns.color_palette("deep")[3])
        ax.set_yticks(range(len(terms)))
        ax.set_yticklabels(terms, fontsize=9)
        ax.set_title("Top TF-IDF Terms in Abstracts")
        ax.invert_yaxis()
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "tfidf_abstracts.png")
        plt.close(fig)
        logger.info("Saved: tfidf_abstracts.png")

    # Bigrams plot
    if top_bigrams:
        fig, ax = plt.subplots(figsize=(10, 6))
        bgs = [b for b, _ in top_bigrams]
        counts = [c for _, c in top_bigrams]
        ax.barh(range(len(bgs)), counts, color=sns.color_palette("deep")[4])
        ax.set_yticks(range(len(bgs)))
        ax.set_yticklabels(bgs, fontsize=9)
        ax.set_title("Top Bigrams in Abstracts")
        ax.invert_yaxis()
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "bigrams_abstracts.png")
        plt.close(fig)
        logger.info("Saved: bigrams_abstracts.png")

    # 8. Reporte
    generate_abstract_report(articles, topic_data, gaps)

    results = {
        "topics": topic_data,
        "gaps": gaps,
        "tfidf_top": tfidf_top,
        "top_bigrams": top_bigrams,
    }

    logger.info("Fase 2b completada.")
    return results
