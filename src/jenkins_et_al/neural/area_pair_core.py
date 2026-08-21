"""Shared core for the two area-pair correlation notebooks.

Ported from `notebooks/neural/noise_correlations_area_pair_analysis.ipynb`
and `notebooks/neural/spontaneous_correlations_area_pair_analysis.ipynb` --
both independently define byte-identical versions of every function in this
module (group-data loading/filtering, the nMLF-mask relabeling, correlation
and empirical-p-value helpers, and the heatmap-building set). Extracted once
here.

**The per-fish/per-area-pair analysis loop is also shared** (`run_area_pair_analysis`
below), via a `load_fish_activity` callback -- checked line-by-line against
both notebooks' own main loops (noise-correlation cell 12, spontaneous cell
13) and confirmed identical from "assign group labels" onward (area-pair
iteration, null resampling, empirical p-values, summary-record building).
What genuinely differs between the two notebooks is only how each neuron's
one activity vector gets built beforehand -- residual computation from a
scalar response-vector file (wide pivot) vs. a raw-fluorescence window
sliced and concatenated across trials (array-per-cell pivot then
concatenated). That step is each notebook's own `load_fish_activity`
closure in `noise_correlations.py`/`spontaneous_correlations.py`; this
module never reads raw neural data itself.

Config values that were notebook-global (`MIN_PAIRS_THRESHOLD`,
`USE_CUSTOM_AXES`/`X_AREAS`/`Y_AREAS`, `compare_same_group`, `xlabel`/
`ylabel`/`color_x`/`color_y`, `FDR_ALPHA`) are now explicit function
parameters -- a refactor-shape change, not a logic change; every notebook's
own default value is preserved as each function's default where one existed.

**Confirmed inconsistency, not unified**: `pval_to_stars` exists in three
different forms across the two notebooks -- the "official" one used by every
heatmap (3-tier: `***`/`**`/`*`/`ns`, thresholds 0.001/0.01/0.05), and two
different local redefinitions inside each notebook's own (broken, see
`reduce_to_per_fish_means` below) intra-vs-inter section (4-tier, thresholds
1e-4/1e-3/1e-2/5e-2, differing only in the top-tier label: `"****"` in the
noise-correlation notebook vs. `"p < 0.001"` in the spontaneous one). Kept as
three separate functions (`pval_to_stars`, `pval_to_stars_detailed_noise`,
`pval_to_stars_detailed_spontaneous`) rather than merged, since the
originals genuinely disagree and the difference was never surfaced as a
question needing a single answer.
"""
from __future__ import annotations

from ast import literal_eval
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import combine_pvalues
from statsmodels.stats.multitest import multipletests


def load_nmlf_mask(mask_path):
    """Load nMLF brain area mask from a TIFF file, or None if not found."""
    from skimage import io

    if mask_path.exists():
        return io.imread(str(mask_path))
    return None


def is_within_3d_mask(coordinates, mask) -> bool:
    """Return True if a (z, y, x) coordinate falls inside a 3-D binary mask."""
    if mask is None:
        return False
    coordinates = np.array(coordinates, dtype=int)
    if coordinates.shape != (3,):
        return False
    z, y, x = coordinates
    depth, height, width = mask.shape
    if (0 <= z < depth) and (0 <= y < height) and (0 <= x < width):
        return mask[z, y, x] > 0
    return False


def load_group_data(
    group_path,
    nmlf_mask,
    filter_method: str = "regression",
    fdr_threshold: float | None = None,
    corr_threshold: float | None = None,
    sparseness_col: str | None = None,
    sparseness_percentile: float | None = None,
    sparseness_mode: str = "high",
) -> pd.DataFrame:
    """Load and filter a neuron-group CSV.

    filter_method:
        'regression' -> filter by p_fdr and optional correlation
        'sparseness' -> filter by top/bottom sparseness percentile
    """
    df = pd.read_csv(group_path)

    if "coords" in df.columns:
        df = df.dropna(subset=["coords"])
        df["coords_tuple"] = df["coords"].apply(literal_eval)
    elif "coords_serialised" in df.columns:
        df = df.dropna(subset=["coords_serialised"])
        df["coords_tuple"] = df["coords_serialised"].apply(literal_eval)

    if nmlf_mask is not None:
        df["area"] = df.apply(
            lambda row: "nMLF"
            if is_within_3d_mask(row["coords_tuple"], nmlf_mask)
            else row["area"],
            axis=1,
        )

    if filter_method == "regression":
        if fdr_threshold is not None:
            df = df[df["p_fdr"] < fdr_threshold]
        if corr_threshold is not None:
            df = df[df["correlation"] >= corr_threshold]
    elif filter_method == "sparseness":
        if sparseness_col is None:
            raise ValueError("sparseness_col must be provided for sparseness filtering")
        if sparseness_percentile is None:
            raise ValueError("sparseness_percentile must be provided for sparseness filtering")

        if sparseness_mode == "high":
            cutoff = np.percentile(df[sparseness_col].dropna(), 100 - sparseness_percentile)
            df = df[df[sparseness_col] >= cutoff]
        elif sparseness_mode == "low":
            cutoff = np.percentile(df[sparseness_col].dropna(), sparseness_percentile)
            df = df[df[sparseness_col] <= cutoff]
        else:
            raise ValueError("sparseness_mode must be 'high' or 'low'")
    else:
        raise ValueError("filter_method must be 'regression' or 'sparseness'")
    return df


def compute_pairwise_correlation(
    mat1: np.ndarray, mat2: np.ndarray, same_group: bool = False
) -> np.ndarray:
    """Compute pairwise Pearson correlations between rows of mat1 and mat2.

    When same_group is True and mat1 == mat2, the diagonal (self-correlations)
    is removed.
    """
    mat1, mat2 = np.asarray(mat1, dtype=float), np.asarray(mat2, dtype=float)
    c = np.corrcoef(mat1, mat2)[: mat1.shape[0], mat1.shape[0] :]
    if same_group and c.shape[0] == c.shape[1]:
        c = c[~np.eye(c.shape[0], dtype=bool)]
    return c.ravel()


def fisher_z_mean(r_values) -> float:
    """Fisher-Z mean: transform to arctanh, average, transform back."""
    r = np.asarray(r_values, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return np.nan
    return float(np.tanh(np.mean(np.arctanh(np.clip(r, -0.9999, 0.9999)))))


def summarize_corrs(r_values, pctl_hi: float = 95, pctl_lo: float = 5):
    """Return (Fisher-z mean, pctl_hi percentile, pctl_lo percentile)."""
    r = np.asarray(r_values, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return np.nan, np.nan, np.nan
    return fisher_z_mean(r), float(np.percentile(r, pctl_hi)), float(np.percentile(r, pctl_lo))


def _clean_null(null_dist, obs):
    """Shared validation for empirical p-value functions."""
    null = np.asarray(null_dist, dtype=float)
    null = null[np.isfinite(null)]
    valid = null.size > 0 and np.isfinite(obs)
    return null, valid


def empirical_p_two_sided(obs: float, null_dist) -> float:
    """Two-sided empirical p-value with +1 continuity correction."""
    null, valid = _clean_null(null_dist, obs)
    if not valid:
        return np.nan
    n = null.size
    p_lo = (np.sum(null <= obs) + 1) / (n + 1)
    p_hi = (np.sum(null >= obs) + 1) / (n + 1)
    return float(min(2 * min(p_lo, p_hi), 1.0))


def empirical_p_greater(obs: float, null_dist) -> float:
    """One-sided empirical p-value: P(null >= obs)."""
    null, valid = _clean_null(null_dist, obs)
    if not valid:
        return np.nan
    return float((np.sum(null >= obs) + 1) / (null.size + 1))


def empirical_p_less(obs: float, null_dist) -> float:
    """One-sided empirical p-value: P(null <= obs)."""
    null, valid = _clean_null(null_dist, obs)
    if not valid:
        return np.nan
    return float((np.sum(null <= obs) + 1) / (null.size + 1))


def _sample_null(pool: pd.DataFrame, n: int, seed: int, activity_col: str) -> np.ndarray:
    """Sample n rows from pool and return a stacked activity matrix."""
    samp = pool.sample(n=n, replace=False, random_state=seed)
    return np.stack(samp[activity_col].values)


def run_area_pair_analysis(
    fish_list: list[str],
    load_fish_activity: Callable[[str], pd.DataFrame],
    brain_areas: list[str],
    compare_same_group: bool,
    use_group1_for_both: bool,
    use_group2_for_both: bool,
    min_neurons_per_group: int,
    n_random_samples: int,
    pctl_hi: float = 95,
    pctl_lo: float = 5,
) -> tuple[pd.DataFrame, dict, dict]:
    """Run the shared per-fish/per-area-pair analysis loop.

    `load_fish_activity(fish)` must return a DataFrame with one row per
    neuron, columns `area`/`group`/`activity` (`activity` a 1D array per
    neuron, `group` already assigned to `"G1"`/`"G2"`/`"OTHER"` -- each
    notebook's own closure does this exactly as its original did, group
    assignment before pivoting, group1_df/group2_df captured by closure, not
    passed in here) -- see module docstring for what differs between the two
    notebooks upstream of this point (both converge to this shape).

    Returns:
        (summary_df, area_pair_corr_values, random_pair_corr_values) --
        matching both notebooks' own top-level result variables.
    """
    from itertools import combinations, product

    area_pairs = (
        list(combinations(brain_areas, 2)) + [(a, a) for a in brain_areas]
        if compare_same_group
        else list(product(brain_areas, brain_areas))
    )

    area_pair_corr_values: dict[tuple[str, str], dict[str, list[float]]] = {}
    random_pair_corr_values: dict[tuple[str, str], dict[str, list[float]]] = {}
    area_pair_summary_records = []

    for fish in fish_list:
        try:
            activity_df = load_fish_activity(fish)

            for area1, area2 in area_pairs:
                if use_group1_for_both:
                    g1 = activity_df[(activity_df["group"] == "G1") & (activity_df["area"] == area1)]
                    g2 = activity_df[(activity_df["group"] == "G1") & (activity_df["area"] == area2)]
                    same_group_flag = area1 == area2
                elif use_group2_for_both:
                    g1 = activity_df[(activity_df["group"] == "G2") & (activity_df["area"] == area1)]
                    g2 = activity_df[(activity_df["group"] == "G2") & (activity_df["area"] == area2)]
                    same_group_flag = area1 == area2
                else:
                    g1 = activity_df[(activity_df["group"] == "G1") & (activity_df["area"] == area1)]
                    g2 = activity_df[(activity_df["group"] == "G2") & (activity_df["area"] == area2)]
                    same_group_flag = False

                r1_pool = activity_df[(activity_df["group"] == "OTHER") & (activity_df["area"] == area1)]
                r2_pool = activity_df[(activity_df["group"] == "OTHER") & (activity_df["area"] == area2)]

                n1, n2, nr1, nr2 = len(g1), len(g2), len(r1_pool), len(r2_pool)
                if n1 < min_neurons_per_group or n2 < min_neurons_per_group or nr1 < n1 or nr2 < n2:
                    continue

                mat1 = np.stack(g1["activity"].values)
                mat2 = np.stack(g2["activity"].values)
                flat_corr_real = compute_pairwise_correlation(
                    mat1, mat1 if same_group_flag else mat2, same_group=same_group_flag
                )
                real_mean_r, real_r95, real_r5 = summarize_corrs(flat_corr_real, pctl_hi, pctl_lo)
                area_pair_corr_values.setdefault((area1, area2), {})[fish] = flat_corr_real.tolist()

                if same_group_flag:
                    mat_r = _sample_null(r1_pool, n1, seed=42, activity_col="activity")
                    corr_r_desc = compute_pairwise_correlation(mat_r, mat_r, same_group=True)
                else:
                    mat_r1 = _sample_null(r1_pool, n1, seed=42, activity_col="activity")
                    mat_r2 = _sample_null(r2_pool, n2, seed=42042, activity_col="activity")
                    corr_r_desc = compute_pairwise_correlation(mat_r1, mat_r2, same_group=False)
                random_pair_corr_values.setdefault((area1, area2), {})[fish] = corr_r_desc.tolist()

                null_mean = np.empty(n_random_samples)
                null_r95 = np.empty(n_random_samples)
                null_r5 = np.empty(n_random_samples)
                for s in range(n_random_samples):
                    if same_group_flag:
                        mat_r = _sample_null(r1_pool, n1, seed=s, activity_col="activity")
                        corr_null = compute_pairwise_correlation(mat_r, mat_r, same_group=True)
                    else:
                        mat_r1 = _sample_null(r1_pool, n1, seed=s, activity_col="activity")
                        mat_r2 = _sample_null(r2_pool, n2, seed=s + 10_000, activity_col="activity")
                        corr_null = compute_pairwise_correlation(mat_r1, mat_r2, same_group=False)
                    null_mean[s], null_r95[s], null_r5[s] = summarize_corrs(corr_null, pctl_hi, pctl_lo)

                area_pair_summary_records.append({
                    "fish_id": fish, "area1": area1, "area2": area2,
                    "n1": int(n1), "n2": int(n2),
                    "n_pairs_real": int(len(flat_corr_real)),
                    "real_mean_r": real_mean_r, "real_r95": real_r95, "real_r5": real_r5,
                    "p_mean": empirical_p_two_sided(real_mean_r, null_mean),
                    "p_r95": empirical_p_greater(real_r95, null_r95),
                    "p_r5": empirical_p_less(real_r5, null_r5),
                    "rand_mean_mean": float(np.nanmean(null_mean)),
                    "rand_mean_r95": float(np.nanmean(null_r95)),
                    "rand_mean_r5": float(np.nanmean(null_r5)),
                })
        except Exception as e:
            print(f"  Error processing {fish}: {e}")
            continue

    summary_df = pd.DataFrame(area_pair_summary_records)
    return summary_df, area_pair_corr_values, random_pair_corr_values


def pval_to_stars(p: float) -> str:
    """Convert a p-value to star notation (the "official" 3-tier version
    used by every heatmap in both notebooks)."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def pval_to_stars_detailed_noise(p: float) -> str:
    """4-tier variant local to the noise-correlation notebook's intra-vs-inter
    section -- see module docstring."""
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "ns"


def pval_to_stars_detailed_spontaneous(p: float) -> str:
    """4-tier variant local to the spontaneous-correlation notebook's
    intra-vs-inter section -- see module docstring."""
    if p < 1e-4:
        return "p < 0.001"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "ns"


def filter_by_min_pairs(df: pd.DataFrame, min_pairs_threshold: int) -> pd.DataFrame:
    """Keep only area pairs whose mean neuron-pair count >= min_pairs_threshold."""
    mean_counts = (
        df.groupby(["area1", "area2"])["n_pairs_real"].mean().reset_index(name="mean_n_pairs")
    )
    valid = mean_counts[mean_counts["mean_n_pairs"] >= min_pairs_threshold]
    return df.merge(valid[["area1", "area2"]], on=["area1", "area2"], how="inner")


def build_heatmap_matrix(
    df: pd.DataFrame,
    value_col: str,
    area_shorthand: dict[str, str],
    y_areas: list[str],
    x_areas: list[str],
    compare_same_group: bool,
    agg_fn: Callable | None = None,
) -> pd.DataFrame:
    """Pivot a summary_df column into a matrix, reindexed onto (y_areas, x_areas)
    and renamed to shorthand. Symmetrized only if within-group and square."""
    agg_fn = agg_fn or fisher_z_mean

    mat = (
        df.groupby(["area1", "area2"])[value_col]
        .apply(agg_fn)
        .reset_index()
        .pivot(index="area1", columns="area2", values=value_col)
        .reindex(index=y_areas, columns=x_areas)
    )
    mat = mat.rename(index=area_shorthand, columns=area_shorthand)

    if compare_same_group and (mat.shape[0] == mat.shape[1]):
        upper = np.triu(mat.values)
        mat.values[:, :] = upper + upper.T - np.diag(np.diag(upper))

    return mat


def fisher_combine_pvalues(df: pd.DataFrame, p_col: str, fdr_alpha: float) -> pd.DataFrame:
    """Fisher-combine per-fish p-values for each area pair, then apply BH-FDR.

    Returns a DataFrame with columns: area1, area2, n_fish, p_fisher, p_fdr, sig.
    """
    rows = []
    for (a1, a2), g in df.groupby(["area1", "area2"], sort=False):
        pvals = g[p_col].dropna().values
        p_fish = float(combine_pvalues(pvals, method="fisher")[1]) if len(pvals) else np.nan
        rows.append(
            {"area1": a1, "area2": a2, "n_fish": int(g["fish_id"].nunique()), "p_fisher": p_fish}
        )

    out = pd.DataFrame(rows)
    out["p_fdr"] = np.nan
    m = out["p_fisher"].notna()
    if m.sum() > 0:
        _, q, _, _ = multipletests(out.loc[m, "p_fisher"], method="fdr_bh")
        out.loc[m, "p_fdr"] = q
    out["sig"] = out["p_fdr"] < fdr_alpha
    return out


def build_sig_matrix(
    masked_matrix: pd.DataFrame,
    per_area_p: pd.DataFrame,
    area_shorthand: dict[str, str],
    compare_same_group: bool,
) -> pd.DataFrame:
    """Build a boolean significance matrix aligned to masked_matrix index/columns."""
    sig = pd.DataFrame(False, index=masked_matrix.index, columns=masked_matrix.columns)
    tmp = per_area_p.copy()
    tmp["a1s"] = tmp["area1"].map(area_shorthand)
    tmp["a2s"] = tmp["area2"].map(area_shorthand)
    for _, row in tmp.iterrows():
        if bool(row.get("sig", False)) and pd.notna(row["a1s"]) and pd.notna(row["a2s"]):
            sig.loc[row["a1s"], row["a2s"]] = True
    if compare_same_group:
        sig = sig | sig.T
    return sig


def add_significance_stars(ax, masked_matrix: pd.DataFrame, sig_matrix: pd.DataFrame) -> None:
    """Overlay significance stars on a seaborn heatmap axes object."""
    for yi, y in enumerate(masked_matrix.index):
        for xi, x in enumerate(masked_matrix.columns):
            if pd.notna(masked_matrix.loc[y, x]) and bool(sig_matrix.loc[y, x]):
                ax.text(xi + 0.5, yi + 0.5, "*", ha="center", va="center_baseline",
                        fontsize=16, fontweight="bold", color="black")


def reduce_to_per_fish_means(
    area_pair_values: dict[tuple[str, str], dict[str, list[float]]],
    brain_areas: list[str],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Reduce raw per-(area-pair, fish) correlation lists to one intra-area
    mean and one inter-area mean per fish.

    Confirmed missing from both original notebooks' "intra vs inter"
    section, which references `fish_intra_means`/`fish_inter_means` without
    ever computing them -- reconstructed at the user's request, following
    the same per-fish-mean-reduction pattern already used (and working) in
    each notebook's own "real vs random" distribution cell.

    Returns:
        (fish_used, fish_intra_means, fish_inter_means)
    """
    fish_used: list[str] = []
    fish_intra_means: list[float] = []
    fish_inter_means: list[float] = []

    all_fish = sorted({f for pair in area_pair_values.values() for f in pair.keys()})
    for fish in all_fish:
        intra_vals: list[float] = []
        inter_vals: list[float] = []
        for (area1, area2), fish_dict in area_pair_values.items():
            if area1 not in brain_areas or area2 not in brain_areas:
                continue
            if fish not in fish_dict:
                continue
            vals = np.asarray(fish_dict[fish], dtype=float)
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                continue
            if area1 == area2:
                intra_vals.extend(vals)
            else:
                inter_vals.extend(vals)

        if len(intra_vals) > 0 and len(inter_vals) > 0:
            fish_used.append(fish)
            fish_intra_means.append(float(np.mean(intra_vals)))
            fish_inter_means.append(float(np.mean(inter_vals)))

    return fish_used, np.array(fish_intra_means, dtype=float), np.array(fish_inter_means, dtype=float)


def per_fish_ks_distributions(
    area_pair_values: dict[tuple[str, str], dict[str, list[float]]],
    fish_list: list[str],
    brain_areas: list[str],
) -> pd.DataFrame:
    """Per-fish KS test between that fish's pooled intra-area and inter-area
    correlation distributions.

    Ported from the spontaneous-correlation notebook's cell 18 (the one
    notebook where this per-fish KS loop was actually written out) -- reused
    for the noise-correlation notebook too, which never had this loop at
    all, only the (broken) downstream cell that assumed its output existed.
    """
    from scipy.stats import ks_2samp

    records = []
    for fish in fish_list:
        fish_intra: list[float] = []
        fish_inter: list[float] = []
        for (area1, area2), fish_dict in area_pair_values.items():
            if area1 not in brain_areas or area2 not in brain_areas:
                continue
            if fish not in fish_dict:
                continue
            vals = np.asarray(fish_dict[fish], dtype=float)
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                continue
            if area1 == area2:
                fish_intra.extend(vals)
            else:
                fish_inter.extend(vals)

        fish_intra_arr = np.asarray(fish_intra, dtype=float)
        fish_inter_arr = np.asarray(fish_inter, dtype=float)
        if len(fish_intra_arr) > 0 and len(fish_inter_arr) > 0:
            ks_stat, ks_p = ks_2samp(fish_intra_arr, fish_inter_arr)
            records.append(
                {
                    "fish_id": fish,
                    "n_intra_pairs": len(fish_intra_arr),
                    "n_inter_pairs": len(fish_inter_arr),
                    "ks_D": ks_stat,
                    "ks_p": ks_p,
                    "intra_mean": float(np.mean(fish_intra_arr)),
                    "inter_mean": float(np.mean(fish_inter_arr)),
                    "intra_median": float(np.median(fish_intra_arr)),
                    "inter_median": float(np.median(fish_inter_arr)),
                }
            )
    return pd.DataFrame(records)


# ── Plotting ─────────────────────────────────────────────────────────────
#
# Ported from both notebooks' Visualisation section (8.1-8.7) -- byte-
# identical between the two apart from labels/colors/value ranges, which
# are now explicit parameters instead of notebook globals. Every function
# returns a `Figure` without calling `plt.show()`, consistent with every
# other plotting function in this project.


def style_heatmap(ax, cbar_label: str, xlabel: str, ylabel: str, color_x: str, color_y: str) -> None:
    """Apply shared styling to a seaborn heatmap axes."""
    ax.set_facecolor("grey")
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=26)
    cbar.set_label(cbar_label, fontsize=26)
    ax.tick_params(axis="both", which="both", length=0)
    ax.set_xlabel(xlabel, color=color_x, fontsize=26)
    ax.set_ylabel(ylabel, color=color_y, fontsize=26)
    plt.xticks(rotation=90, fontsize=26)
    plt.yticks(rotation=0, fontsize=26)


def plot_distribution_real_vs_random(
    area_pair_corr_values: dict,
    random_pair_corr_values: dict,
    real_label: str,
    dist_color: str,
    xlabel: str = "Correlation",
) -> plt.Figure | None:
    """Distribution: pooled real vs. pooled random-vs-random correlations,
    with pooled Mann-Whitney/KS tests and a per-fish paired Wilcoxon test."""
    from scipy.stats import ks_2samp, mannwhitneyu, wilcoxon

    real_vals = [v for pair in area_pair_corr_values.values() for vals in pair.values() for v in vals]
    rand_vals = [v for pair in random_pair_corr_values.values() for vals in pair.values() for v in vals]
    if not real_vals or not rand_vals:
        return None

    u_stat, u_p = mannwhitneyu(real_vals, rand_vals, alternative="two-sided")
    ks_stat, ks_p = ks_2samp(real_vals, rand_vals)

    common_fish = sorted(
        set(f for pair in area_pair_corr_values.values() for f in pair.keys())
        & set(f for pair in random_pair_corr_values.values() for f in pair.keys())
    )
    fish_real_means, fish_rand_means = [], []
    for fish in common_fish:
        fish_real = np.array(
            [v for pair in area_pair_corr_values.values() if fish in pair for v in pair[fish]], dtype=float
        )
        fish_rand = np.array(
            [v for pair in random_pair_corr_values.values() if fish in pair for v in pair[fish]], dtype=float
        )
        fish_real = fish_real[~np.isnan(fish_real)]
        fish_rand = fish_rand[~np.isnan(fish_rand)]
        if len(fish_real) > 0 and len(fish_rand) > 0:
            fish_real_means.append(np.mean(fish_real))
            fish_rand_means.append(np.mean(fish_rand))
    w_stat, w_p = (
        wilcoxon(fish_real_means, fish_rand_means, alternative="two-sided")
        if fish_real_means
        else (np.nan, np.nan)
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(real_vals, bins=50, alpha=0.7, density=True, label=real_label, color=dist_color)
    ax.hist(rand_vals, bins=50, histtype="step", linewidth=2.5, density=True, label="Random vs Random", color="black")
    ax.axvline(np.mean(real_vals), color=dist_color, linestyle="--", linewidth=2,
               label=f"Mean {real_label}: {np.mean(real_vals):.3f} n = {len(real_vals)}")
    ax.axvline(np.mean(rand_vals), color="black", linestyle="--", linewidth=2,
               label=f"Mean Random: {np.mean(rand_vals):.3f} n = {len(rand_vals)}")
    ax.set_xlabel(xlabel, fontsize=26)
    ax.set_ylabel("Density", fontsize=26)
    ax.set_title(
        f"MWU: {pval_to_stars(u_p)}   |   KS: {pval_to_stars(ks_p)}   |   Fish Wilcoxon: {pval_to_stars(w_p)}",
        fontsize=16,
    )
    ax.legend(fontsize=14, loc="upper left")
    ax.tick_params(axis="both", labelsize=26)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(1.8)
    plt.tight_layout()
    return fig


def plot_neuron_pair_counts_heatmap(
    summary_df: pd.DataFrame,
    area_shorthand: dict[str, str],
    y_areas: list[str],
    x_areas: list[str],
    compare_same_group: bool,
    xlabel: str,
    ylabel: str,
    color_x: str,
    color_y: str,
    min_pairs_threshold: int,
) -> plt.Figure | None:
    """Mean neuron-pair count per area pair (masked below `min_pairs_threshold`)."""
    if summary_df is None or len(summary_df) == 0:
        return None
    mat = build_heatmap_matrix(
        summary_df, "n_pairs_real", area_shorthand, y_areas, x_areas, compare_same_group, agg_fn="mean"
    )
    masked = mat.mask(mat < min_pairs_threshold)
    max_val = np.nanmax(masked.values)
    vmax_rnd = int(np.ceil(max_val / 100.0) * 100)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(masked, cmap="mako_r", annot=False, square=True, linewidths=0.5,
                vmin=0, vmax=vmax_rnd, cbar_kws={"shrink": 0.95}, ax=ax)
    style_heatmap(ax, "Pairs\n(/ fish)", xlabel, ylabel, color_x, color_y)
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0, vmax_rnd])
    cbar.set_ticklabels([0, vmax_rnd])
    ax.set_title("Neuron Pairs per Area", fontsize=20, pad=20)
    plt.tight_layout()
    return fig


def plot_mean_correlation_heatmap(
    summary_df: pd.DataFrame,
    value_col: str,
    area_shorthand: dict[str, str],
    y_areas: list[str],
    x_areas: list[str],
    compare_same_group: bool,
    min_pairs_threshold: int,
    vmin: float,
    vmax: float,
    cbar_label: str,
    title: str,
    xlabel: str | None = None,
    ylabel: str | None = None,
    color_x: str | None = None,
    color_y: str | None = None,
    is_null: bool = False,
) -> plt.Figure | None:
    """Mean correlation heatmap -- reused for both the "Real" (`value_col='real_mean_r'`)
    and "Null" (`value_col='rand_mean_mean'`) versions."""
    if summary_df is None or len(summary_df) == 0:
        return None
    df_plot = filter_by_min_pairs(summary_df, min_pairs_threshold)
    mat = build_heatmap_matrix(df_plot, value_col, area_shorthand, y_areas, x_areas, compare_same_group)
    masked = mat.mask(mat == 0)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(masked, vmin=vmin, vmax=vmax, cmap="bwr", annot=False, square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.95}, ax=ax)
    if is_null:
        ax.set_facecolor("grey")
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=26)
        cbar.set_label(cbar_label, fontsize=26)
        ax.tick_params(axis="both", which="both", length=0)
        ax.set_xlabel("Rnd", color="gray", fontsize=26)
        ax.set_ylabel("Rnd", color="gray", fontsize=26)
        plt.xticks(rotation=90, fontsize=26)
        plt.yticks(rotation=0, fontsize=26)
    else:
        style_heatmap(ax, cbar_label, xlabel, ylabel, color_x, color_y)
    ax.set_title(title, fontsize=20, pad=20)
    plt.tight_layout()
    return fig


def plot_delta_heatmap(
    summary_df: pd.DataFrame,
    real_col: str,
    null_col: str,
    p_col: str,
    area_shorthand: dict[str, str],
    y_areas: list[str],
    x_areas: list[str],
    compare_same_group: bool,
    min_pairs_threshold: int,
    fdr_alpha: float,
    vmin: float,
    vmax: float,
    cbar_label: str,
    title: str,
    xlabel: str,
    ylabel: str,
    color_x: str,
    color_y: str,
    direction_gate: str | None = None,
) -> plt.Figure | None:
    """Delta (real - null) heatmap with significance stars -- reused for the
    Δ-mean, Δ-95th-percentile, and Δ-5th-percentile figures.

    `direction_gate`: None, `">0"`, or `"<0"` -- the 95th/5th-percentile
    versions only show a star when the delta has the expected sign (ported
    from each notebook's own "direction gate" comment).
    """
    if summary_df is None or len(summary_df) == 0:
        return None
    df = filter_by_min_pairs(summary_df.copy(), min_pairs_threshold)
    df["_diff"] = df[real_col] - df[null_col]
    if len(df) == 0:
        return None

    mat = build_heatmap_matrix(df, "_diff", area_shorthand, y_areas, x_areas, compare_same_group, agg_fn="mean")
    masked = mat.mask(mat == 0)
    per_area_p = fisher_combine_pvalues(df, p_col, fdr_alpha)
    sig_matrix = build_sig_matrix(masked, per_area_p, area_shorthand, compare_same_group)
    if direction_gate == ">0":
        sig_matrix = sig_matrix & (masked > 0)
    elif direction_gate == "<0":
        sig_matrix = sig_matrix & (masked < 0)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(masked, vmin=vmin, vmax=vmax, cmap="bwr", annot=False, square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.95}, ax=ax)
    style_heatmap(ax, cbar_label, xlabel, ylabel, color_x, color_y)
    ax.set_title(title, fontsize=18 if direction_gate is None else 16, pad=30 if direction_gate is None else 20)
    add_significance_stars(ax, masked, sig_matrix)
    plt.tight_layout()
    return fig


def plot_intra_vs_inter_distribution(
    intra_vals: np.ndarray,
    inter_vals: np.ndarray,
    fish_intra_means: np.ndarray,
    fish_inter_means: np.ndarray,
    ks_d_values: np.ndarray,
    pooled_mwu_p: float,
    pooled_ks_stat: float,
    pooled_ks_p: float,
    per_fish_wilcoxon_p: float,
    ks_d_wilcoxon_p: float,
    dist_color: str,
    title: str,
    xlabel: str = "Correlation",
    pval_to_stars_fn: Callable[[float], str] | None = None,
) -> plt.Figure:
    """Intra-area vs. inter-area correlation distribution -- the reconstructed
    "intra vs inter" section, see `reduce_to_per_fish_means`."""
    stars = pval_to_stars_fn or pval_to_stars
    intra_color = dist_color
    inter_color = "darkgreen"

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(intra_vals, bins=50, alpha=0.55, density=True, label="Intra-area", color=intra_color)
    ax.hist(inter_vals, bins=50, histtype="step", linewidth=2.5, density=True, label="Inter-area", color=inter_color)
    ax.axvline(np.mean(intra_vals), color=intra_color, linestyle="--", linewidth=2)
    ax.axvline(np.mean(inter_vals), color=inter_color, linestyle="--", linewidth=2)

    stats_text = (
        "Pooled pairs\n"
        f"Intra n = {len(intra_vals)}\n"
        f"Inter n = {len(inter_vals)}\n"
        f"MWU: p = {pooled_mwu_p:.2e} ({stars(pooled_mwu_p)})\n"
        f"KS: D = {pooled_ks_stat:.3f}, p = {pooled_ks_p:.2e} ({stars(pooled_ks_p)})\n"
        f"intra mean = {np.mean(intra_vals):.4f}\n"
        f"inter mean = {np.mean(inter_vals):.4f}\n\n"
        "Per-fish means\n"
        f"Intra = {np.mean(fish_intra_means):.3f} ± "
        f"{np.std(fish_intra_means, ddof=1) / np.sqrt(len(fish_intra_means)):.3f}\n"
        f"Inter = {np.mean(fish_inter_means):.3f} ± "
        f"{np.std(fish_inter_means, ddof=1) / np.sqrt(len(fish_inter_means)):.3f}\n"
        f"Wilcoxon: p = {per_fish_wilcoxon_p:.3e} ({stars(per_fish_wilcoxon_p)})\n\n"
        "Per-fish KS D\n"
        f"mean D = {np.mean(ks_d_values):.3f}\n"
        f"median D = {np.median(ks_d_values):.3f}\n"
        f"Wilcoxon vs 0: p = {ks_d_wilcoxon_p:.3e} ({stars(ks_d_wilcoxon_p)})"
    )
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, ha="right", va="top", fontsize=12,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.7", alpha=0.95))

    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel("Density", fontsize=18)
    ax.set_title(title, fontsize=16)
    ax.legend(fontsize=13, loc="upper left", frameon=False)
    ax.tick_params(axis="both", labelsize=14)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(1.5)
    plt.tight_layout()
    return fig
