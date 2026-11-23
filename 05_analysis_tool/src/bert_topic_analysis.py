"""BERT/SciBERT topic modeling, keyword extraction, and semantic similarity.

Optional module - requires sentence-transformers, umap-learn, hdbscan.
Gracefully degrades if dependencies are not installed.
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from config import PLOTLY_DIR, FIGURES_DIR

logger = logging.getLogger("bibliometrico.bert")

# Check optional dependencies
BERT_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    BERT_AVAILABLE = True
except ImportError:
    pass

UMAP_AVAILABLE = False
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    pass

HDBSCAN_AVAILABLE = False
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    pass

# Default config
BERT_MODEL_NAME = "allenai/scibert_scivocab_uncased"
BERT_BATCH_SIZE = 32


def _save_plotly_figure(fig, output_dir: Path, name: str) -> None:
    """Save Plotly figure reusing the shared helper."""
    from src.advanced_visualizations import _save_plotly_figure as _save
    _save(fig, output_dir, name)


def compute_embeddings(
    articles: list[dict],
    model_name: str = BERT_MODEL_NAME,
    batch_size: int = BERT_BATCH_SIZE,
) -> np.ndarray:
    """Compute sentence embeddings for article abstracts/titles.

    Args:
        articles: List of enriched articles.
        model_name: HuggingFace model name.
        batch_size: Batch size for encoding.

    Returns:
        Numpy array of shape (n_articles, embedding_dim).
    """
    if not BERT_AVAILABLE:
        raise ImportError("sentence-transformers not installed")

    texts = []
    for a in articles:
        abstract = a.get("abstract", "")
        title = a.get("title", "")
        if abstract and abstract != "nan":
            texts.append(f"{title}. {abstract}")
        elif title:
            texts.append(title)
        else:
            texts.append("")

    logger.info(f"Computing embeddings with {model_name} for {len(texts)} articles...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    logger.info(f"Embeddings shape: {embeddings.shape}")
    # Cache model on function for reuse (avoids reloading 500MB+ model)
    compute_embeddings._cached_model = model
    compute_embeddings._cached_model_name = model_name
    return embeddings


def bert_topic_modeling(
    articles: list[dict],
    embeddings: np.ndarray,
    min_cluster_size: int = 5,
) -> dict[int, list[str]]:
    """Perform topic modeling using UMAP + HDBSCAN + c-TF-IDF.

    Args:
        articles: List of enriched articles.
        embeddings: Precomputed embeddings.
        min_cluster_size: Minimum cluster size for HDBSCAN.

    Returns:
        Dict mapping topic_id -> list of top keywords.
    """
    if not UMAP_AVAILABLE:
        raise ImportError("umap-learn not installed")
    if not HDBSCAN_AVAILABLE:
        raise ImportError("hdbscan not installed")

    import umap as umap_module
    import hdbscan as hdbscan_module

    # Dimensionality reduction
    logger.info("Running UMAP dimensionality reduction...")
    reducer = umap_module.UMAP(
        n_components=10, n_neighbors=15, min_dist=0.0,
        metric="cosine", random_state=42,
    )
    reduced = reducer.fit_transform(embeddings)

    # Clustering
    logger.info("Running HDBSCAN clustering...")
    clusterer = hdbscan_module.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=3,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(reduced)

    # Assign topics to articles
    for i, a in enumerate(articles):
        a["bert_topic"] = int(labels[i])

    n_topics = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = sum(1 for l in labels if l == -1)
    logger.info(f"Topics found: {n_topics}, noise points: {n_noise}")

    topic_keywords = compute_ctfidf_topic_keywords(articles, labels)
    return topic_keywords


def compute_ctfidf_topic_keywords(
    articles: list[dict],
    labels: np.ndarray | list[int],
) -> dict[int, list[str]]:
    """True c-TF-IDF (Grootendorst 2022): w_{t,c} = tf_{t,c} * ln(1 + A / cf_t)."""
    import json
    import math
    import re
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    from config import REPORTS_DIR

    _extra_stops = {
        "using", "based", "via", "towards", "toward", "through", "among",
        "study", "studies", "analysis", "approach", "approaches", "method",
        "methods", "model", "models", "system", "systems", "data", "results",
        "effect", "effects", "impact", "role", "case", "new", "use", "used",
    }
    _stopwords = set(ENGLISH_STOP_WORDS) | _extra_stops

    def _clean_term(raw: str) -> str:
        s = re.sub(r"(\w)\?s\b", r"\1's", raw.lower().strip())
        s = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", s)
        s = re.sub(r"\s+", " ", s)
        return s

    topic_keywords: dict[int, list[str]] = {}
    cluster_term_counts: dict[int, Counter] = defaultdict(Counter)

    for i, a in enumerate(articles):
        topic = int(labels[i])
        if topic == -1:
            continue
        # Multi-word author/OpenAlex keywords get priority weight (x2)
        for kw in (a.get("manual_tags") or []) + (a.get("automatic_tags") or []):
            ct = _clean_term(str(kw))
            if len(ct) > 2 and ct not in _stopwords and not ct.isdigit():
                cluster_term_counts[topic][ct] += 2
        # Title words (cleaned, stopword-filtered)
        title_clean = re.sub(r"(\w)\?s\b", r"\1s", str(a.get("title", "")).lower())
        for tok in re.findall(r"[a-z][a-z0-9\-]{2,}", title_clean):
            if tok not in _stopwords:
                cluster_term_counts[topic][tok] += 1

    if cluster_term_counts:
        global_cf: Counter = Counter()
        total_words_all = 0
        for c_counts in cluster_term_counts.values():
            for term, cnt in c_counts.items():
                global_cf[term] += cnt
                total_words_all += cnt
        avg_words_per_class = max(total_words_all / len(cluster_term_counts), 1.0)

        for topic, c_counts in sorted(cluster_term_counts.items()):
            class_total = max(sum(c_counts.values()), 1)
            scored = []
            for term, cnt in c_counts.items():
                tf = cnt / class_total
                idf = math.log(1.0 + avg_words_per_class / max(global_cf[term], 1))
                scored.append((tf * idf, cnt, term))
            scored.sort(key=lambda x: (-x[0], -x[1], x[2]))
            topic_keywords[topic] = [term for _, _, term in scored[:10]]

    for i, a in enumerate(articles):
        t = int(labels[i])
        a["bert_topic_keywords"] = topic_keywords.get(t, [])[:5] if t >= 0 else []

    try:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        (REPORTS_DIR / "bert_topics_ctfidf.json").write_text(
            json.dumps({str(k): v for k, v in topic_keywords.items()}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        pass

    return topic_keywords


def extract_keywords_bert(
    articles: list[dict],
    embeddings: np.ndarray,
    top_k: int = 5,
) -> None:
    """Extract keywords per article using cosine similarity with embeddings.

    Sets 'bert_keywords' on each article.
    """
    if not BERT_AVAILABLE:
        raise ImportError("sentence-transformers not installed")

    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    # Collect all unique keywords
    all_keywords = set()
    for a in articles:
        for kw in a.get("manual_tags", []):
            all_keywords.add(kw.lower().strip())
        for kw in a.get("automatic_tags", []):
            all_keywords.add(kw.lower().strip())

    kw_list = sorted(all_keywords)
    if not kw_list:
        logger.warning("No keywords found for BERT keyword extraction")
        return

    logger.info(f"Encoding {len(kw_list)} keywords for BERT keyword extraction...")
    # Reuse cached model from compute_embeddings to avoid 500MB+ reload
    if (hasattr(compute_embeddings, '_cached_model')
            and getattr(compute_embeddings, '_cached_model_name', '') == BERT_MODEL_NAME):
        model = compute_embeddings._cached_model
    else:
        model = SentenceTransformer(BERT_MODEL_NAME)
    kw_embeddings = model.encode(kw_list, batch_size=BERT_BATCH_SIZE)

    # For each article, find top-k most similar keywords
    for i, a in enumerate(articles):
        sim = cosine_similarity([embeddings[i]], kw_embeddings)[0]
        top_indices = np.argsort(sim)[-top_k:][::-1]
        a["bert_keywords"] = [kw_list[idx] for idx in top_indices]


def compute_semantic_similarity(
    articles: list[dict],
    embeddings: np.ndarray,
    top_k: int = 5,
) -> None:
    """Compute top-k semantically similar articles for each article.

    Sets 'similar_articles' on each article as list of (key, score) tuples.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    n = len(articles)
    if n > 1000:
        # For large datasets, compute similarity in batches to avoid n^2 memory
        logger.info(f"Computing semantic similarity in batches ({n} articles)...")
        batch_size = 200
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_sim = cosine_similarity(embeddings[start:end], embeddings)
            for i_local in range(end - start):
                i_global = start + i_local
                sims = batch_sim[i_local].copy()
                sims[i_global] = -1
                top_indices = np.argsort(sims)[-top_k:][::-1]
                articles[i_global]["similar_articles"] = [
                    (articles[idx].get("key", str(idx)), round(float(sims[idx]), 4))
                    for idx in top_indices
                ]
    else:
        logger.info("Computing semantic similarity matrix...")
        sim_matrix = cosine_similarity(embeddings)
        for i, a in enumerate(articles):
            sims = sim_matrix[i].copy()
            sims[i] = -1
            top_indices = np.argsort(sims)[-top_k:][::-1]
            a["similar_articles"] = [
                (articles[idx].get("key", str(idx)), round(float(sims[idx]), 4))
                for idx in top_indices
            ]


def plot_bert_topics_2d(
    articles: list[dict],
    embeddings: np.ndarray,
    output_dir: Path = PLOTLY_DIR,
) -> None:
    """Create UMAP 2D scatter plot colored by BERT topic.

    Args:
        articles: List with 'bert_topic' set.
        embeddings: Precomputed embeddings.
        output_dir: Output directory.
    """
    if not UMAP_AVAILABLE:
        logger.warning("umap-learn not installed, skipping 2D topic plot")
        return

    import plotly.graph_objects as go
    import umap as umap_module

    logger.info("Computing 2D UMAP projection for topic visualization...")
    reducer_2d = umap_module.UMAP(
        n_components=2, n_neighbors=15, min_dist=0.1,
        metric="cosine", random_state=42,
    )
    coords_2d = reducer_2d.fit_transform(embeddings)

    topics = [a.get("bert_topic", -1) for a in articles]
    unique_topics = sorted(set(topics))

    _COLORS = [
        "#e41a1c", "#377eb8", "#4daf4a", "#ff7f00", "#984ea3",
        "#a65628", "#f781bf", "#66c2a5", "#fc8d62", "#8da0cb",
        "#e78ac3", "#a6d854", "#ffd92f", "#e5c494", "#1b9e77",
        "#d95f02", "#7570b3", "#e7298a", "#999999", "#b3b3b3",
    ]

    fig = go.Figure()
    for topic in unique_topics:
        mask = [i for i, t in enumerate(topics) if t == topic]
        if not mask:
            continue
        xs = [coords_2d[i, 0] for i in mask]
        ys = [coords_2d[i, 1] for i in mask]
        hovers = []
        for i in mask:
            a = articles[i]
            first_auth = a["authors"][0].split(",")[0] if a.get("authors") else "?"
            yr = a.get("publication_year", "?")
            title = a.get("title", "")[:60]
            hovers.append(f"<b>{first_auth} ({yr})</b><br>{title}<br>Topic: {topic}")

        color = "#cccccc" if topic == -1 else _COLORS[topic % len(_COLORS)]
        name = "Noise" if topic == -1 else f"Topic {topic}"

        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers",
            marker=dict(size=6, color=color, opacity=0.7 if topic != -1 else 0.3,
                        line=dict(width=0.5, color="white")),
            hovertext=hovers, hoverinfo="text",
            name=name,
        ))

    fig.update_layout(
        title="BERT Topic Modeling (UMAP 2D Projection)",
        xaxis=dict(title="UMAP 1", showgrid=False),
        yaxis=dict(title="UMAP 2", showgrid=False),
        width=1400, height=1000, paper_bgcolor="white", plot_bgcolor="#fafafa",
        hovermode="closest",
        legend=dict(font=dict(size=10)),
    )
    _save_plotly_figure(fig, output_dir, "bert_topics_2d")


def run_bert_analysis(
    articles: list[dict],
    model_name: str = BERT_MODEL_NAME,
    skip: bool = False,
) -> None:
    """Entry point for BERT analysis.

    Sets bert_topic, bert_keywords, similar_articles on each article.
    """
    if skip:
        logger.info("BERT analysis skipped (--skip-bert)")
        return

    if not BERT_AVAILABLE:
        logger.warning(
            "BERT analysis skipped: sentence-transformers not installed. "
            "Install with: pip install sentence-transformers"
        )
        return

    logger.info("=== BERT/SciBERT Analysis ===")

    try:
        embeddings = compute_embeddings(articles, model_name=model_name)
    except Exception as e:
        logger.error(f"Failed to compute embeddings: {e}")
        return

    # Topic modeling
    if UMAP_AVAILABLE and HDBSCAN_AVAILABLE:
        try:
            topic_keywords = bert_topic_modeling(articles, embeddings)
            for topic_id, keywords in sorted(topic_keywords.items()):
                logger.info(f"  Topic {topic_id}: {', '.join(keywords[:5])}")
        except Exception as e:
            logger.error(f"BERT topic modeling failed: {e}")
    else:
        logger.warning("UMAP or HDBSCAN not installed, skipping topic modeling")

    # Keyword extraction
    try:
        extract_keywords_bert(articles, embeddings)
    except Exception as e:
        logger.error(f"BERT keyword extraction failed: {e}")

    # Semantic similarity
    try:
        compute_semantic_similarity(articles, embeddings)
    except Exception as e:
        logger.error(f"Semantic similarity failed: {e}")

    # 2D topic visualization
    try:
        plot_bert_topics_2d(articles, embeddings)
    except Exception as e:
        logger.error(f"BERT topic 2D plot failed: {e}")

    logger.info("BERT analysis completed.")
