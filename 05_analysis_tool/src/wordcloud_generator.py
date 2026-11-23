"""WordCloud generation from article text fields using NLTK stopwords."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import FIGURES_DIR

logger = logging.getLogger("bibliometrico.wordcloud")

# Default max words
WORDCLOUD_MAX_WORDS = 200


def _get_nltk_stopwords() -> set[str]:
    """Get NLTK English stopwords with fallback to a minimal set."""
    try:
        import nltk
        try:
            from nltk.corpus import stopwords
            return set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            from nltk.corpus import stopwords
            return set(stopwords.words("english"))
    except ImportError:
        logger.warning("NLTK not installed, using minimal stopword list")
        return set(
            "a about above after again against all am an and any are as at be because "
            "been before being below between both but by can could did do does doing "
            "down during each few for from further get got had has have having he her "
            "here hers herself him himself his how i if in into is it its itself just "
            "me more most my myself no nor not now of off on once only or other our ours "
            "ourselves out over own s same she should so some such t than that the their "
            "theirs them themselves then there these they this those through to too under "
            "until up very was we were what when where which while who whom why will with "
            "would you your yours yourself yourselves".split()
        )


def extract_text_for_wordcloud(articles: list[dict], source: str) -> str:
    """Extract concatenated text from articles for a given source type.

    Args:
        articles: List of enriched articles.
        source: One of 'abstracts', 'titles', 'author_keywords', 'keywords_plus'.

    Returns:
        Concatenated text string.
    """
    parts = []
    for a in articles:
        if source == "abstracts":
            text = a.get("abstract", "")
            if text and text != "nan":
                parts.append(text)
        elif source == "titles":
            text = a.get("title", "")
            if text and text != "nan":
                parts.append(text)
        elif source == "author_keywords":
            for kw in (a["author_keywords"] if "author_keywords" in a else a.get("manual_tags", [])):
                if kw.strip():
                    parts.append(kw.strip())
        elif source == "keywords_plus":
            for kw in a.get("automatic_tags", []):
                if kw.strip():
                    parts.append(kw.strip())

    return " ".join(parts)


def generate_wordcloud(
    articles: list[dict],
    source: str,
    output_dir: Path = FIGURES_DIR,
    max_words: int = WORDCLOUD_MAX_WORDS,
    width: int = 1600,
    height: int = 800,
) -> None:
    """Create and save a WordCloud PNG for the given source type.

    Args:
        articles: List of enriched articles.
        source: One of 'abstracts', 'titles', 'author_keywords', 'keywords_plus'.
        output_dir: Directory to save the PNG.
        max_words: Maximum number of words in the cloud.
        width: Image width in pixels.
        height: Image height in pixels.
    """
    try:
        from wordcloud import WordCloud
    except ImportError:
        logger.warning("wordcloud package not installed, skipping wordcloud generation")
        return

    text = extract_text_for_wordcloud(articles, source)
    if not text.strip():
        logger.warning(f"No text found for wordcloud source='{source}'")
        return

    stopwords = _get_nltk_stopwords()

    wc = WordCloud(
        width=width,
        height=height,
        max_words=max_words,
        stopwords=stopwords,
        background_color="white",
        colormap="viridis",
        collocations=False,
        min_font_size=8,
    )
    wc.generate(text)

    fig, ax = plt.subplots(figsize=(width / 100, height / 100))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"WordCloud: {source.replace('_', ' ').title()}", fontsize=16, pad=10)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"wordcloud_{source}.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out_path}")


def run_wordclouds(
    articles: list[dict],
    sources: list[str] | None = None,
) -> None:
    """Generate wordclouds for all requested source types.

    Args:
        articles: List of enriched articles.
        sources: List of source types. Defaults to all four types.
    """
    if sources is None:
        sources = ["abstracts", "titles", "author_keywords", "keywords_plus"]

    logger.info(f"=== WordCloud Generation ({len(sources)} sources) ===")

    for source in sources:
        try:
            generate_wordcloud(articles, source)
        except Exception as e:
            logger.error(f"WordCloud for '{source}' failed: {e}")

    logger.info("WordCloud generation completed.")
