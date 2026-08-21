"""Valence-neuron counts, fish contributions, area distributions, and
overlap analyses.

Ported from `notebooks/neural/neuron_counts_and_overlaps.ipynb`; logic
unchanged. Data loading moved to
`jenkins_et_al.neural.valence_bhv_core.load_valence_bhv_dataframe`. Cell
28's neuron-group selection is byte-identical to
`cross_regressor_neuron_correlations.py`'s cell 9 -- reused directly rather
than redefined.

**Confirmed harmless, not a bug**: cell 11's `stim_df_fb['fish_id'] = ...`
assignment (`filter_to_brain_areas` below) triggers a pandas
`SettingWithCopyWarning` in the original notebook (mutating a
boolean-indexed slice without `.copy()`) -- checked against real data that
this does *not* silently corrupt the source `stim_df`'s own `fish_id`
values; only the slice itself is affected, as intended. `.copy()` added
here to silence the warning cleanly -- no behavior change, confirmed by the
same check.

**Confirmed harmless in practice, not a bug**: `compute_area_fish_counts`'s
`.loc[area_order]` reindex (cell 11) would raise `KeyError` if any area in
`BRAIN_AREAS` had <=10 total VENs after filtering -- confirmed against real
data that all 13 areas clear that bar, so this doesn't fire, but it's
fragile if that ever changes. Left as `.loc` (not `.reindex`) to match the
original's exact behavior.

**Confirmed title/threshold mismatch, fixed at the user's request**: cell
14's plot title said "(>20 neurons)" but its own filter was `total_sig > 10`
-- a copy-paste leftover from cell 20's similar figure, which does use
`>20` and whose title correctly says so. Title text now says "(>10
neurons)", matching the actual filter applied to this figure specifically.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib_venn import venn2_unweighted

from jenkins_et_al.neural.cross_regressor_neuron_correlations import select_neuron_groups

#: Source: cell 9. A different, broader set than area_pair_core's BRAIN_AREAS.
BRAIN_AREAS: tuple[str, ...] = (
    "olfactory_bulb", "olfactory_epithelium", "pallium", "subpallium", "preoptic",
    "prethalamus", "eminentia_thalami", "dorsal_thalamus", "dorsal_habenula",
    "ventral_habenula", "posterior_tuberculum", "rostral_hypothalamus",
    "intermediate_hypothalamus",
)


def _strip_fish_suffix(fish_id: pd.Series) -> pd.Series:
    return fish_id.astype(str).str.replace(r"_(fb|hb)$", "", regex=True)


def filter_to_brain_areas(stim_df: pd.DataFrame, brain_areas=BRAIN_AREAS) -> pd.DataFrame:
    """Restrict to `brain_areas` and strip the `_fb`/`_hb` fish-ID suffix.

    Source: cells 9-10 (`BRAIN_AREAS`, `stim_df_fb`) + cell 11 lines 1-6
    (fish-ID suffix stripping).
    """
    stim_df_fb = stim_df[stim_df["area"].isin(brain_areas)].copy()
    stim_df_fb["fish_id"] = _strip_fish_suffix(stim_df_fb["fish_id"])
    return stim_df_fb


def compute_area_fish_counts(
    stim_df_fb: pd.DataFrame,
    area_shorthand_dict: dict[str, str],
    area_order=BRAIN_AREAS,
    min_total_vens: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """VEN (positive- or negative-significant) counts per area per fish,
    and the corresponding within-area fraction.

    Source: cell 11 lines 7-45.

    Returns:
        (area_fish_counts, area_fish_frac) -- both indexed by shorthand area
        name, columns = fish IDs.
    """
    vens = stim_df_fb[(stim_df_fb["pos_pvalue"] < 0.05) | (stim_df_fb["neg_pvalue"] < 0.05)].copy()
    area_fish_counts = vens.groupby(["area", "fish_id"]).size().unstack(fill_value=0)

    area_total_vens = area_fish_counts.sum(axis=1)
    area_fish_counts = area_fish_counts.loc[area_total_vens > min_total_vens]

    area_fish_frac = area_fish_counts.div(area_fish_counts.sum(axis=1), axis=0)
    area_fish_frac = area_fish_frac.loc[list(area_order)]
    area_fish_counts = area_fish_counts.loc[list(area_order)]

    area_fish_frac = area_fish_frac.rename(index=area_shorthand_dict)
    area_fish_counts = area_fish_counts.rename(index=area_shorthand_dict)
    area_fish_frac = area_fish_frac[~area_fish_frac.index.astype(str).str.contains("_")]
    area_fish_counts = area_fish_counts.loc[area_fish_frac.index]

    return area_fish_counts, area_fish_frac


def plot_fish_contribution_stacked_bar(area_fish_counts: pd.DataFrame, area_fish_frac: pd.DataFrame) -> plt.Figure:
    """100%-stacked bar of each fish's fractional contribution to each
    area's VEN count.

    Source: cell 11 lines 46-.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    area_fish_frac.plot(kind="bar", stacked=True, ax=ax, width=0.85)

    ax.set_ylabel("Fraction of VENs in area", fontsize=18)
    ax.set_xlabel("Brain area", fontsize=18)
    ax.set_title("Fish contribution to VENs within each brain area", fontsize=20)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", labelrotation=45, labelsize=18)
    ax.tick_params(axis="y", labelsize=18)

    area_sizes = area_fish_counts.sum(axis=1)
    ax.set_xticklabels([f"{area} (n={area_sizes.loc[area]})" for area in area_fish_frac.index])

    ax.legend(title="Fish ID", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def summarize_fish_per_area(area_fish_counts: pd.DataFrame) -> pd.DataFrame:
    """Total VENs and number of contributing fish, per area.

    Source: cell 12.
    """
    fish_per_area = (area_fish_counts > 0).sum(axis=1)
    return pd.DataFrame({
        "Total VENs": area_fish_counts.sum(axis=1),
        "Fish contributing": fish_per_area,
    }).sort_values("Total VENs", ascending=False)


def compute_fish_valence_fractions(stim_df: pd.DataFrame) -> pd.DataFrame:
    """Per-fish positive/negative/all-valence significant-neuron counts and
    each fish's fraction of the dataset-wide total.

    Source: cell 13 lines 1-27. Mutates `stim_df['fish_id']` in place
    (stripping the `_fb`/`_hb` suffix), matching the original -- this is a
    real, intentional side effect on the shared dataframe, not scoped to a
    copy, since later cells (e.g. `filter_to_brain_areas`'s own stripping)
    are idempotent against an already-stripped fish_id.
    """
    stim_df["fish_id"] = _strip_fish_suffix(stim_df["fish_id"])

    positive_counts = stim_df[stim_df["pos_pvalue"] < 0.05].groupby("fish_id").size()
    negative_counts = stim_df[stim_df["neg_pvalue"] < 0.05].groupby("fish_id").size()
    all_valence_counts = stim_df[
        (stim_df["pos_pvalue"] < 0.05) | (stim_df["neg_pvalue"] < 0.05)
    ].groupby("fish_id").size()

    fish_summary_df = pd.DataFrame({
        "positive": positive_counts, "negative": negative_counts, "all_valence": all_valence_counts,
    }).fillna(0)
    fish_summary_df["positive_frac"] = fish_summary_df["positive"] / fish_summary_df["positive"].sum()
    fish_summary_df["negative_frac"] = fish_summary_df["negative"] / fish_summary_df["negative"].sum()
    fish_summary_df["all_valence_frac"] = fish_summary_df["all_valence"] / fish_summary_df["all_valence"].sum()
    return fish_summary_df


def plot_fish_valence_fractions(fish_summary_df: pd.DataFrame) -> plt.Figure:
    """Per-fish positive/negative/all-valence fraction bar chart.

    Source: cell 13 lines 28-.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(fish_summary_df))
    width = 0.25

    ax.bar(x - width, fish_summary_df["positive_frac"], width, color="green", label="Positive")
    ax.bar(x, fish_summary_df["negative_frac"], width, color="magenta", label="Negative")
    ax.bar(x + width, fish_summary_df["all_valence_frac"], width, color="black", label="All valence")

    ax.set_xticks(x)
    ax.set_xticklabels(fish_summary_df.index, rotation=45, ha="right")
    ax.set_ylabel("Fraction of neurons")
    ax.set_xlabel("Fish ID")
    ax.legend(frameon=False)
    plt.tight_layout()
    return fig


def compute_area_pos_neg_summary(
    stim_df_fb: pd.DataFrame, area_shorthand_dict: dict[str, str], min_total_sig: int = 10,
) -> pd.DataFrame:
    """Per-area positive/negative significant-neuron counts.

    Source: cell 14 lines 1-16.
    """
    positive_counts = stim_df_fb[stim_df_fb["pos_pvalue"] < 0.05].groupby("area").size()
    negative_counts = stim_df_fb[stim_df_fb["neg_pvalue"] < 0.05].groupby("area").size()

    summary_df = pd.DataFrame({"positive": positive_counts, "negative": negative_counts}).fillna(0)
    summary_df["total_sig"] = summary_df["positive"] + summary_df["negative"]
    summary_df = summary_df[summary_df["total_sig"] > min_total_sig]

    summary_df = summary_df.rename(index=area_shorthand_dict)
    summary_df = summary_df[~summary_df.index.str.contains("_")]
    return summary_df


def plot_pos_vs_neg_scatter(summary_df: pd.DataFrame, min_total_sig: int, random_state: int | None = None) -> plt.Figure:
    """Scatter of per-area negative vs. positive significant-neuron counts,
    with a unity reference line and jittered area labels.

    Source: cell 14 lines 17-. Title now shows `min_total_sig` (see module
    docstring for the confirmed >20-vs->10 title/threshold mismatch fix).
    """
    rng = np.random.default_rng(random_state)
    x = summary_df["negative"].values
    y = summary_df["positive"].values
    labels = summary_df.index.tolist()

    jitter_strength = 0.5
    jitter_x = rng.uniform(-jitter_strength, jitter_strength, size=len(x))
    jitter_y = rng.uniform(-jitter_strength, jitter_strength, size=len(y))

    fig, ax = plt.subplots(figsize=(9, 8))
    plt.scatter(x, y, s=100, color="black")
    for i in range(len(labels)):
        plt.text(x[i] + jitter_x[i], y[i] + jitter_y[i], labels[i], fontsize=18)

    min_val, max_val = min(min(x), min(y)), max(max(x), max(y))
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="gray", linewidth=1.5)

    plt.xlabel("Negative VENs count", fontsize=28, color="magenta")
    plt.ylabel("Positive VENs count", fontsize=28, color="green")
    plt.xticks([0, 50, 100, 150, 200], fontsize=25)
    plt.yticks([0, 50, 100, 150, 200], fontsize=25)
    plt.xlim(-5, 205)
    plt.ylim(-5, 205)
    plt.title(f"Significant Neuron Counts per Brain Area (>{min_total_sig} neurons)", fontsize=20, pad=20)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    plt.tight_layout()
    return fig


def compute_valence_bhv_overlap(stim_df: pd.DataFrame, bhv_corr_threshold: float = 0.3) -> dict[str, set]:
    """Valence-neuron vs. behaviour-neuron identity sets, for the Venn
    diagram of their overlap.

    Source: cell 16 lines 1-11.
    """
    valence_mask = (stim_df["pos_pvalue"] < 0.05) | (stim_df["neg_pvalue"] < 0.05)
    bhv_mask = (stim_df["bhv_pvalue"] < 0.05) & (stim_df["bhv_correlation"] >= bhv_corr_threshold)

    valence_neurons = set(stim_df[valence_mask][["fish_id", "neuron_id"]].apply(tuple, axis=1))
    bhv_neurons = set(stim_df[bhv_mask][["fish_id", "neuron_id"]].apply(tuple, axis=1))

    return {
        "only_valence": valence_neurons - bhv_neurons,
        "only_bhv": bhv_neurons - valence_neurons,
        "both": valence_neurons & bhv_neurons,
    }


def plot_valence_bhv_venn(overlap: dict[str, set]) -> plt.Figure:
    """Venn diagram: significant valence neurons vs. significant behaviour
    (S-BCN) neurons.

    Source: cell 16 lines 12-.
    """
    fig = plt.figure(figsize=(6, 6))
    venn = venn2_unweighted(
        subsets=(len(overlap["only_valence"]), len(overlap["only_bhv"]), len(overlap["both"])),
        set_labels=("VENs", "BCNs"),
    )
    for region, color in zip(["10", "01", "11"], ["purple", "blue", "gray"]):
        patch = venn.get_patch_by_id(region)
        if patch:
            patch.set_edgecolor(color)
            patch.set_linewidth(5)
            patch.set_facecolor("none")
    for region_id in ["10", "01", "11"]:
        label = venn.get_label_by_id(region_id)
        if label:
            label.set_fontsize(22)
    venn.set_labels[0].set_fontsize(25)
    venn.set_labels[1].set_fontsize(25)
    plt.title("Overlap between Sig. Valence and BCN Neurons", fontsize=20)
    return fig


def compute_bcn_overlap(stim_df: pd.DataFrame, neg_threshold: float = 0.3, pos_threshold: float = -0.1) -> dict[str, set]:
    """Two behaviour-neuron definitions (by negative- vs. positive-valence
    correlation) and their overlap.

    Source: cell 17 lines 1-13.
    """
    bhv_mask_1 = (stim_df["bhv_pvalue"] < 0.05) & (stim_df["neg_correlation"] >= neg_threshold)
    bhv_mask_2 = (stim_df["bhv_pvalue"] < 0.05) & (stim_df["pos_correlation"] <= pos_threshold)

    bhv_neurons1 = set(stim_df[bhv_mask_1][["fish_id", "neuron_id"]].apply(tuple, axis=1))
    bhv_neurons2 = set(stim_df[bhv_mask_2][["fish_id", "neuron_id"]].apply(tuple, axis=1))

    return {
        "bhv_1": bhv_neurons1 - bhv_neurons2,
        "bhv_2": bhv_neurons2 - bhv_neurons1,
        "both": bhv_neurons1 & bhv_neurons2,
    }


def plot_bcn_overlap_venn(overlap: dict[str, set]) -> plt.Figure:
    """Venn diagram: L-BCNs defined by negative-correlation threshold vs.
    positive-correlation threshold.

    Source: cell 17 lines 14-.
    """
    fig = plt.figure(figsize=(6, 6))
    venn = venn2_unweighted(
        subsets=(len(overlap["bhv_1"]), len(overlap["bhv_2"]), len(overlap["both"])),
        set_labels=("L-BCNs NEG ≥ 0.3", "L-BCNs POS ≤ -0.1"),
    )
    for region, color in zip(["10", "01", "11"], ["red", "blue", "grey"]):
        patch = venn.get_patch_by_id(region)
        if patch:
            patch.set_edgecolor(color)
            patch.set_linewidth(5)
            patch.set_facecolor("none")
    for region_id in ["10", "01", "11"]:
        label = venn.get_label_by_id(region_id)
        if label:
            label.set_fontsize(22)
    venn.set_labels[0].set_fontsize(25)
    venn.set_labels[1].set_fontsize(25)
    plt.title("Overlap between L-BCNs", fontsize=20)
    return fig


def get_overlap_neuron_info(stim_df: pd.DataFrame, both: set) -> pd.DataFrame:
    """Full rows for neurons in the `both` overlap set.

    Source: cell 18.
    """
    both_df = pd.DataFrame(list(both), columns=["fish_id", "neuron_id"])
    return stim_df.merge(both_df, on=["fish_id", "neuron_id"], how="inner")


def plot_overlap_neurons_by_area(both_neurons_info: pd.DataFrame) -> plt.Figure:
    """Bar chart of overlap-neuron counts per brain area.

    Source: cell 19.
    """
    area_counts = both_neurons_info["area"].value_counts()
    fig = plt.figure(figsize=(6, 4))
    plt.bar(area_counts.index, area_counts.values, color="gray", edgecolor="black")
    plt.ylabel("Number of neurons")
    plt.xlabel("Brain area")
    plt.title("Overlap neurons by area")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


def compute_area_norm_counts(
    stim_df: pd.DataFrame, area_shorthand_dict: dict[str, str],
    bhv_corr_threshold: float = 0.3, min_valence: int = 20,
) -> pd.DataFrame:
    """Per-area valence and behaviour neuron counts, normalized to fractions
    of the dataset-wide total of each.

    Source: cell 20 lines 1-20.
    """
    valence_mask = (stim_df["pos_pvalue"] < 0.05) | (stim_df["neg_pvalue"] < 0.05)
    valence_counts = stim_df[valence_mask].groupby("area").size()

    bhv_mask = (stim_df["bhv_pvalue"] < 0.05) & (stim_df["bhv_correlation"] >= bhv_corr_threshold)
    bhv_counts = stim_df[bhv_mask].groupby("area").size()

    summary_df = pd.DataFrame({"valence": valence_counts, "bhv": bhv_counts}).fillna(0)
    summary_df = summary_df[summary_df["valence"] > min_valence]
    summary_df = summary_df.rename(index=area_shorthand_dict)
    summary_df = summary_df[~summary_df.index.str.contains("_")]

    summary_df["valence_norm"] = summary_df["valence"] / summary_df["valence"].sum()
    summary_df["bhv_norm"] = summary_df["bhv"] / summary_df["bhv"].sum()
    return summary_df


def plot_norm_counts_scatter(summary_df: pd.DataFrame, min_valence: int, random_state: int | None = None) -> plt.Figure:
    """Scatter of normalized per-area behaviour-neuron vs. valence-neuron
    counts, with a unity reference line and jittered area labels.

    Source: cell 20 lines 21-.
    """
    rng = np.random.default_rng(random_state)
    x = summary_df["bhv_norm"].values
    y = summary_df["valence_norm"].values
    labels = summary_df.index.tolist()

    jitter_strength = 0.001
    jitter_x = rng.uniform(-jitter_strength, jitter_strength, size=len(x))
    jitter_y = rng.uniform(-jitter_strength, jitter_strength, size=len(y))

    fig, ax = plt.subplots(figsize=(10, 9))
    plt.scatter(x, y, s=50, color="black")
    for i in range(len(labels)):
        plt.text(x[i] + jitter_x[i], y[i] + jitter_y[i], labels[i], fontsize=20)

    min_val, max_val = min(min(x), min(y)), max(max(x), max(y))
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="gray")

    plt.xlabel("Normalized BHV Neuron Count", fontsize=28, color="black")
    plt.ylabel("Normalized VEN Count", fontsize=28, color="black")
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.xlim(-0.03, 0.35)
    plt.ylim(-0.03, 0.25)
    ax.set_aspect("equal", adjustable="box")
    plt.title(f"Norm. neuron counts per area (>{min_valence} VENs)", fontsize=20, pad=20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    plt.tight_layout()
    return fig


def select_bhv_valence_neurons(stim_df: pd.DataFrame, bhv_threshold: float = 0.3) -> pd.DataFrame:
    """Neurons significant for behaviour (correlation below `bhv_threshold`,
    p<0.05) AND significant for either valence direction.

    Source: cell 23.
    """
    bhv_significant = stim_df[(stim_df["bhv_correlation"] < bhv_threshold) & (stim_df["bhv_pvalue"] < 0.05)]
    valence_significant = stim_df[(stim_df["pos_pvalue"] < 0.05) | (stim_df["neg_pvalue"] < 0.05)]
    return stim_df.loc[bhv_significant.index.intersection(valence_significant.index)]


def summarize_bhv_valence_overlap(bhv_valence_neurons: pd.DataFrame) -> dict[str, int]:
    """Counts of bhv+negative-valence and bhv+positive-valence neurons
    within the bhv-and-valence-significant set.

    Source: cell 24.
    """
    bhv_neg_neurons = bhv_valence_neurons[bhv_valence_neurons["neg_pvalue"] < 0.05]
    bhv_pos_neurons = bhv_valence_neurons[bhv_valence_neurons["pos_pvalue"] < 0.05]
    return {
        "total": len(bhv_valence_neurons),
        "bhv_negative": len(bhv_neg_neurons),
        "bhv_positive": len(bhv_pos_neurons),
    }


def save_bhv_valence_neurons(
    bhv_valence_neurons: pd.DataFrame,
    output_path: Path = Path("/Volumes/LaCie/larval_HuC/imaging/7dpf_anti_bhv_valence_neurons.csv"),
) -> None:
    """Write the bhv+valence-significant neuron table to CSV.

    Source: cell 25. `output_path` is a parameter (was hardcoded) so golden
    capture doesn't write to the real production path -- default preserved
    for production parity.
    """
    bhv_valence_neurons.to_csv(output_path)


def count_high_bhv_high_neg(high_bhv_neurons: pd.DataFrame) -> int:
    """Count of high-bhv-correlation neurons that are also
    negative-valence-significant.

    Source: cells 28+30 -- `high_bhv_neurons` itself comes from
    `cross_regressor_neuron_correlations.select_neuron_groups` (identical
    to this notebook's own cell 28).
    """
    return len(high_bhv_neurons[high_bhv_neurons["neg_pvalue"] < 0.05])
