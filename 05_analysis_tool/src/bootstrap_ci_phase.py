"""Phase 5b — Non-parametric bootstrap confidence intervals for headline indicators.

This module wraps the analytical pipeline's headline aggregate statistics with a
non-parametric percentile bootstrap so that the Results section can report each
point estimate together with its 95 % CI. The same per-article sources used by
the rest of the pipeline are reused (the OpenAlex DiskCache for FWCI, the
enriched corpus for MCP / APC / population / MNCS), which guarantees that the
bootstrap is consistent with what Phase 5 reports as the point estimate.

Indicators:
    1. MCP rate (% of classifiable articles with >= 2 distinct countries).
    2. Median reported participant sample size.
    3. APC vs. no-APC mean citation counts (difference + Mann-Whitney U).
    4. Field-Weighted Citation Impact (mean / median / % > 1).
    5. Cross-check: MNCS mean and PP(top 10 % of field).

Outputs:
    output/reproducibility/bootstrap_ci.json
    output/reproducibility/bootstrap_ci.md
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from config import OPENALEX_CACHE_DIR, OUTPUT_DIR
from src.utils import DiskCache, normalize_doi, save_json

logger = logging.getLogger("bibliometrico.bootstrap_ci")

REPRODUCIBILITY_DIR = OUTPUT_DIR / "reproducibility"


# ---------------------------------------------------------------------------
# Generic bootstrap helpers
# ---------------------------------------------------------------------------

def _percentile_ci(samples: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    lo = float(np.percentile(samples, 100 * alpha / 2))
    hi = float(np.percentile(samples, 100 * (1 - alpha / 2)))
    return lo, hi


def _bootstrap_stat(values: np.ndarray, statistic, B: int, rng: np.random.Generator) -> np.ndarray:
    n = len(values)
    out = np.empty(B, dtype=float)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        out[b] = statistic(values[idx])
    return out


def _bootstrap_two_sample_diff(
    a: np.ndarray, b: np.ndarray, statistic, B: int, rng: np.random.Generator
) -> np.ndarray:
    out = np.empty(B, dtype=float)
    na, nb = len(a), len(b)
    for k in range(B):
        ai = rng.integers(0, na, size=na)
        bi = rng.integers(0, nb, size=nb)
        out[k] = statistic(a[ai]) - statistic(b[bi])
    return out


def _mann_whitney_u(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return float("nan"), float("nan")
    combined = np.concatenate([a, b])
    ranks = np.empty_like(combined)
    order = np.argsort(combined, kind="mergesort")
    ranks[order] = np.arange(1, len(combined) + 1)
    sorted_vals = combined[order]
    i = 0
    while i < len(sorted_vals):
        j = i + 1
        while j < len(sorted_vals) and sorted_vals[j] == sorted_vals[i]:
            j += 1
        if j - i > 1:
            avg = (ranks[order[i:j]].sum()) / (j - i)
            ranks[order[i:j]] = avg
        i = j
    Ra = ranks[:na].sum()
    U_a = Ra - na * (na + 1) / 2.0
    U_b = na * nb - U_a
    U = min(U_a, U_b)
    mu_U = na * nb / 2.0
    sigma_U = math.sqrt(na * nb * (na + nb + 1) / 12.0)
    if sigma_U == 0:
        return float(U), float("nan")
    z = (U - mu_U) / sigma_U
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    return float(U), float(p)


# ---------------------------------------------------------------------------
# Per-article extractors
# ---------------------------------------------------------------------------

def _article_countries(article: dict) -> set:
    """Paises de un articulo: definicion unica del modelo del corpus (antes solo paises de los autores)."""
    from src.modelo_corpus import article_countries
    return set(article_countries(article))


def _article_price_index(article: dict) -> Optional[float]:
    """Per-article Price Index (% of references aged 0-5 years at citation time)."""
    art_year = article.get("publication_year")
    if not art_year or not isinstance(art_year, int):
        return None
    ages = []
    for ref in article.get("references_resolved") or []:
        if not isinstance(ref, dict):
            continue
        ry = ref.get("year")
        if ry and isinstance(ry, (int, str)):
            try:
                ry_int = int(ry)
                if 1900 <= ry_int <= art_year:
                    ages.append(art_year - ry_int)
            except (ValueError, TypeError):
                pass
    if not ages:
        for ref in article.get("referenced_works_detailed") or []:
            if not isinstance(ref, dict):
                continue
            ry = ref.get("year")
            if ry and isinstance(ry, (int, str)):
                try:
                    ry_int = int(ry)
                    if 1900 <= ry_int <= art_year:
                        ages.append(art_year - ry_int)
                except (ValueError, TypeError):
                    pass
    if len(ages) < 3:
        return None
    recent = sum(1 for a in ages if 0 <= a <= 5)
    return float(recent) / float(len(ages))


def _article_apc(article: dict) -> Optional[float]:
    oa = article.get("openalex") or {}
    apc_paid = oa.get("apc_paid")
    if isinstance(apc_paid, dict):
        val = apc_paid.get("value_usd") or apc_paid.get("value")
        if val:
            return float(val)
    jm = article.get("journal_metrics") or {}
    if jm.get("apc_usd"):
        return float(jm["apc_usd"])
    return None


def _article_mncs(article: dict) -> Optional[float]:
    m = article.get("mncs")
    field = article.get("mncs_field")
    if not isinstance(m, (int, float)) or math.isnan(float(m)):
        return None
    if not field or str(field).strip().lower() in {"unknown", "none", ""}:
        return None
    return float(m)


def _article_in_top10(article: dict) -> Optional[bool]:
    v = article.get("in_top10_field")
    return v if isinstance(v, bool) else None


def _lookup_fwci_from_openalex_cache(articles: List[dict]) -> List[Tuple[str, float]]:
    """Same source as src/impact_evolution_analysis.py:_lookup_fwci_from_cache."""
    cache = DiskCache(OPENALEX_CACHE_DIR)
    out: List[Tuple[str, float]] = []
    for a in articles:
        doi = a.get("doi", "")
        if not doi:
            continue
        ndoi = normalize_doi(doi)
        if not ndoi:
            continue
        cached = cache.get(f"doi:{ndoi}")
        if not isinstance(cached, dict):
            continue
        fwci = cached.get("fwci")
        if isinstance(fwci, (int, float)) and not math.isnan(float(fwci)):
            out.append((a.get("key", ""), float(fwci)))
    return out


# ---------------------------------------------------------------------------
# Indicator-specific bootstraps
# ---------------------------------------------------------------------------

@dataclass
class CIResult:
    indicator: str
    point: float
    ci_lo: float
    ci_hi: float
    effective_n: int
    extra: dict


def _bootstrap_mcp(articles, B, rng) -> CIResult:
    flags = []
    for a in articles:
        c = _article_countries(a)
        if not c:
            continue
        flags.append(1 if len(c) >= 2 else 0)
    arr = np.array(flags, dtype=float)
    n = len(arr)
    if n == 0:
        return CIResult("MCP_rate", float("nan"), float("nan"), float("nan"), 0, {})
    samples = _bootstrap_stat(arr, np.mean, B, rng)
    lo, hi = _percentile_ci(samples)
    return CIResult(
        indicator="MCP_rate",
        point=float(arr.mean()),
        ci_lo=lo, ci_hi=hi, effective_n=n,
        extra={"scp_count": int((arr == 0).sum()), "mcp_count": int((arr == 1).sum())},
    )


def _bootstrap_price_index(articles, B, rng) -> CIResult:
    vals = [_article_price_index(a) for a in articles]
    arr = np.array([v for v in vals if v is not None], dtype=float)
    n = len(arr)
    if n == 0:
        return CIResult("price_index_mean", float("nan"), float("nan"), float("nan"), 0, {})
    samples = _bootstrap_stat(arr, np.mean, B, rng)
    lo, hi = _percentile_ci(samples)
    return CIResult(
        indicator="price_index_mean",
        point=float(arr.mean()),
        ci_lo=lo, ci_hi=hi, effective_n=n,
        extra={"median": float(np.median(arr)), "min": float(arr.min()), "max": float(arr.max())},
    )


def _bootstrap_citations(articles, B, rng) -> dict:
    cites = [float(a.get("cited_by_count", 0) or 0) for a in articles if a.get("cited_by_count") is not None]
    arr = np.array(cites, dtype=float)
    n = len(arr)
    if n == 0:
        return {"effective_n": 0}
    s_mean = _bootstrap_stat(arr, np.mean, B, rng)
    s_med = _bootstrap_stat(arr, np.median, B, rng)
    lo_m, hi_m = _percentile_ci(s_mean)
    lo_md, hi_md = _percentile_ci(s_med)
    return {
        "effective_n": int(n),
        "citations_mean": {"point": float(arr.mean()), "ci_lo": lo_m, "ci_hi": hi_m},
        "citations_median": {"point": float(np.median(arr)), "ci_lo": lo_md, "ci_hi": hi_md},
    }


def _bootstrap_apc(articles, B, rng) -> dict:
    apc_cites, no_apc_cites = [], []
    for a in articles:
        cites = a.get("cited_by_count")
        if cites is None:
            continue
        apc = _article_apc(a)
        if apc and apc > 0:
            apc_cites.append(float(cites))
        else:
            no_apc_cites.append(float(cites))
    a_arr = np.array(apc_cites)
    b_arr = np.array(no_apc_cites)
    if len(a_arr) == 0 or len(b_arr) == 0:
        return {"effective_n_apc": len(a_arr), "effective_n_no_apc": len(b_arr)}
    diff_samples = _bootstrap_two_sample_diff(a_arr, b_arr, np.mean, B, rng)
    lo, hi = _percentile_ci(diff_samples)
    apc_samples = _bootstrap_stat(a_arr, np.mean, B, rng)
    no_apc_samples = _bootstrap_stat(b_arr, np.mean, B, rng)
    lo_a, hi_a = _percentile_ci(apc_samples)
    lo_b, hi_b = _percentile_ci(no_apc_samples)
    U, p_val = _mann_whitney_u(a_arr, b_arr)
    return {
        "indicator": "APC_vs_noAPC_mean_citations",
        "apc_mean": {"point": float(a_arr.mean()), "ci_lo": lo_a, "ci_hi": hi_a, "n": int(len(a_arr))},
        "no_apc_mean": {"point": float(b_arr.mean()), "ci_lo": lo_b, "ci_hi": hi_b, "n": int(len(b_arr))},
        "diff_mean": {"point": float(a_arr.mean() - b_arr.mean()), "ci_lo": lo, "ci_hi": hi},
        "ratio_mean": float(a_arr.mean() / b_arr.mean()) if b_arr.mean() > 0 else None,
        "mann_whitney_U": U,
        "mann_whitney_p_two_sided": p_val,
        "effective_n_apc": int(len(a_arr)),
        "effective_n_no_apc": int(len(b_arr)),
    }


def _bootstrap_fwci(articles, B, rng) -> dict:
    pairs = _lookup_fwci_from_openalex_cache(articles)
    vals = np.array([v for _, v in pairs], dtype=float)
    out: dict = {"effective_n": int(len(vals))}
    if len(vals) == 0:
        return out
    above1 = (vals > 1.0).astype(float)
    s_mean = _bootstrap_stat(vals, np.mean, B, rng)
    s_med = _bootstrap_stat(vals, np.median, B, rng)
    s_above = _bootstrap_stat(above1, np.mean, B, rng)
    lo_m, hi_m = _percentile_ci(s_mean)
    lo_md, hi_md = _percentile_ci(s_med)
    lo_a, hi_a = _percentile_ci(s_above)
    out["FWCI_mean"] = {"point": float(vals.mean()), "ci_lo": lo_m, "ci_hi": hi_m}
    out["FWCI_median"] = {"point": float(np.median(vals)), "ci_lo": lo_md, "ci_hi": hi_md}
    out["pct_above_world_avg"] = {
        "point": float(above1.mean()), "ci_lo": lo_a, "ci_hi": hi_a,
        "n_above1": int(above1.sum()),
    }
    return out


def _bootstrap_mncs_and_top10(articles, B, rng) -> dict:
    mncs_values, top10_flags = [], []
    for a in articles:
        m = _article_mncs(a)
        t = _article_in_top10(a)
        if m is not None:
            mncs_values.append(m)
        if t is not None:
            top10_flags.append(1 if t else 0)
    mncs_arr = np.array(mncs_values)
    top10_arr = np.array(top10_flags, dtype=float)
    out = {}
    if len(mncs_arr) > 0:
        s = _bootstrap_stat(mncs_arr, np.mean, B, rng)
        lo, hi = _percentile_ci(s)
        out["MNCS_mean"] = {
            "point": float(mncs_arr.mean()), "ci_lo": lo, "ci_hi": hi,
            "effective_n": int(len(mncs_arr)),
        }
    if len(top10_arr) > 0:
        s = _bootstrap_stat(top10_arr, np.mean, B, rng)
        lo, hi = _percentile_ci(s)
        out["PP_top10_field"] = {
            "point": float(top10_arr.mean()), "ci_lo": lo, "ci_hi": hi,
            "effective_n": int(len(top10_arr)),
            "n_top10": int(top10_arr.sum()),
        }
    return out


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _fmt_ci(point, lo, hi, as_percent=False):
    if as_percent:
        return f"{point*100:.2f}% [95 % CI {lo*100:.2f}, {hi*100:.2f}]"
    return f"{point:.4g} [95 % CI {lo:.4g}, {hi:.4g}]"


def _write_markdown(result: dict, path: Path) -> None:
    lines = [
        "# Bootstrap confidence intervals for headline bibliometric indicators",
        "",
        f"- B = {result['config']['B']} replicates",
        f"- Random seed = {result['config']['seed']}",
        f"- CI method: percentile (2.5 %–97.5 %)",
        f"- Corpus N = {result['corpus_size']}",
        "",
        "## 1. MCP rate (multi-country publications)",
        "",
    ]
    m = result["mcp_rate"]
    if m["effective_n"] == 0:
        lines.append(
            "- Not computable: no article in the corpus has a country affiliation "
            "that can be classified as single- or multi-country."
        )
    else:
        lines.append(
            f"- Effective N = {m['effective_n']} classifiable articles "
            f"(SCP = {m['extra']['scp_count']}, MCP = {m['extra']['mcp_count']})"
        )
        lines.append(f"- MCP rate = {_fmt_ci(m['point'], m['ci_lo'], m['ci_hi'], as_percent=True)}")
    lines.append("")
    lines.append("## 2. Price Index (References Aged 0–5 Years at Citation Time)")
    lines.append("")
    pi = result["price_index"]
    if pi["effective_n"] == 0:
        lines.append("- Not computable: insufficient resolved references with publication years.")
    else:
        lines.append(f"- Effective N = {pi['effective_n']} articles with >= 3 resolved references")
        lines.append(
            f"- Mean Price Index = {_fmt_ci(pi['point'], pi['ci_lo'], pi['ci_hi'], as_percent=True)} "
            f"(median = {pi['extra']['median']*100:.1f}%)"
        )
    lines.append("")
    lines.append("## 2b. Citation Impact per Article")
    lines.append("")
    ci_cites = result.get("citations", {})
    if ci_cites.get("effective_n", 0) > 0:
        cm = ci_cites["citations_mean"]
        cmd = ci_cites["citations_median"]
        lines.append(f"- Effective N = {ci_cites['effective_n']} articles")
        lines.append(f"- Mean citations = {_fmt_ci(cm['point'], cm['ci_lo'], cm['ci_hi'])}")
        lines.append(f"- Median citations = {_fmt_ci(cmd['point'], cmd['ci_lo'], cmd['ci_hi'])}")
    lines.append("")
    lines.append("## 3. APC vs. no-APC mean citation counts")
    lines.append("")
    a = result["apc_difference"]
    if "indicator" in a:
        lines.append(
            f"- N APC-paying = {a['effective_n_apc']}, N no-APC = {a['effective_n_no_apc']}"
        )
        lines.append(
            f"- APC-paying mean cites = {_fmt_ci(a['apc_mean']['point'], a['apc_mean']['ci_lo'], a['apc_mean']['ci_hi'])}"
        )
        lines.append(
            f"- No-APC mean cites = {_fmt_ci(a['no_apc_mean']['point'], a['no_apc_mean']['ci_lo'], a['no_apc_mean']['ci_hi'])}"
        )
        lines.append(
            f"- Difference (APC − no-APC) = {_fmt_ci(a['diff_mean']['point'], a['diff_mean']['ci_lo'], a['diff_mean']['ci_hi'])}"
        )
        if a.get("ratio_mean") is not None:
            lines.append(f"- Ratio (APC / no-APC mean) = {a['ratio_mean']:.3f}")
        lines.append(
            f"- Mann–Whitney U = {a['mann_whitney_U']:.1f}, two-sided p = {a['mann_whitney_p_two_sided']:.4f}"
        )
    else:
        lines.append("- Insufficient data for APC vs. no-APC comparison")
    lines.append("")
    lines.append("## 4. Field-Weighted Citation Impact (FWCI)")
    lines.append("")
    fw = result["fwci"]
    lines.append(f"- Effective N = {fw.get('effective_n', 0)} articles with FWCI in the OpenAlex cache")
    if not fw.get("FWCI_mean"):
        lines.append("- Not computable: no article has an FWCI value.")
    if fw.get("FWCI_mean"):
        m = fw["FWCI_mean"]; med = fw["FWCI_median"]; pa = fw["pct_above_world_avg"]
        lines.append(f"- FWCI mean = {_fmt_ci(m['point'], m['ci_lo'], m['ci_hi'])}")
        lines.append(f"- FWCI median = {_fmt_ci(med['point'], med['ci_lo'], med['ci_hi'])}")
        lines.append(
            f"- % above world average (FWCI > 1) = "
            f"{_fmt_ci(pa['point'], pa['ci_lo'], pa['ci_hi'], as_percent=True)} "
            f"({pa['n_above1']} of {fw['effective_n']} articles)"
        )
    lines.append("")
    lines.append("## 5. Cross-check — Leiden-style MNCS and PP(top 10 %)")
    lines.append("")
    mw = result["mncs_top10"]
    if not mw:
        lines.append(
            "- Not computable: no article has the field-level baseline citation counts "
            "that MNCS and PP(top 10 %) require."
        )
    if "MNCS_mean" in mw:
        lines.append(
            f"- MNCS mean = {_fmt_ci(mw['MNCS_mean']['point'], mw['MNCS_mean']['ci_lo'], mw['MNCS_mean']['ci_hi'])} "
            f"(N = {mw['MNCS_mean']['effective_n']})"
        )
    if "PP_top10_field" in mw:
        pp = mw["PP_top10_field"]
        lines.append(
            f"- PP(top 10 % of field) = {_fmt_ci(pp['point'], pp['ci_lo'], pp['ci_hi'], as_percent=True)} "
            f"({pp['n_top10']} of {pp['effective_n']} articles)"
        )
    lines.append("")
    lines.append("---")
    lines.append("_Generated by src/bootstrap_ci_phase.py (Phase 5b)._")
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Public entry point (called by main.py as Phase 5b)
# ---------------------------------------------------------------------------

def run_phase5b(
    articles: List[dict],
    B: int = 10000,
    seed: int = 42,
) -> dict:
    """Bootstrap CIs for the four headline indicators. Idempotent and side-effect-safe.

    Returns the result dict; also persists JSON + Markdown to output/reproducibility/.
    """
    logger.info("Phase 5b: bootstrap CIs (B=%d, seed=%d) on %d articles", B, seed, len(articles))
    rng = np.random.default_rng(seed)

    mcp = _bootstrap_mcp(articles, B, rng)
    price_idx = _bootstrap_price_index(articles, B, rng)
    cites_ci = _bootstrap_citations(articles, B, rng)
    apc = _bootstrap_apc(articles, B, rng)
    fwci = _bootstrap_fwci(articles, B, rng)
    mncs = _bootstrap_mncs_and_top10(articles, B, rng)

    result = {
        "config": {"B": B, "seed": seed, "ci_method": "percentile_95"},
        "corpus_size": len(articles),
        "mcp_rate": asdict(mcp),
        "price_index": asdict(price_idx),
        "citations": cites_ci,
        "apc_difference": apc,
        "fwci": fwci,
        "mncs_top10": mncs,
    }

    REPRODUCIBILITY_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPRODUCIBILITY_DIR / "bootstrap_ci.json"
    md_path = REPRODUCIBILITY_DIR / "bootstrap_ci.md"
    save_json(result, json_path)
    _write_markdown(result, md_path)
    logger.info(
        "Phase 5b: MCP %.2f%% (CI %.2f–%.2f, n=%d); FWCI mean %.3f (n=%d); APC p=%.3f",
        mcp.point * 100, mcp.ci_lo * 100, mcp.ci_hi * 100, mcp.effective_n,
        fwci.get("FWCI_mean", {}).get("point", float("nan")), fwci.get("effective_n", 0),
        apc.get("mann_whitney_p_two_sided", float("nan")),
    )
    return result

run_phase6b = run_phase5b
