"""Cross-regressor neuron correlations: behaviour-neurons vs valence
regressors, and valence-neurons vs the behaviour regressor.

Ported from `notebooks/neural/cross_regressor_neuron_correlations.ipynb`;
logic unchanged. Data loading moved to
`jenkins_et_al.neural.valence_bhv_core.load_valence_bhv_dataframe`.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon


def select_neuron_groups(stim_df: pd.DataFrame, r_threshold: float = 0.3) -> dict[str, pd.DataFrame]:
    """Select high-behaviour-correlation, positive-valence, and
    negative-valence neuron subsets.

    Source: cell 9 (and identically cell 28 of `neuron_counts_and_overlaps.ipynb`).
    """
    high_bhv_neurons = stim_df[
        (stim_df["bhv_correlation"] >= r_threshold) & (stim_df["bhv_pvalue"] < 0.05)
    ]
    high_neg_neurons = stim_df[stim_df["neg_pvalue"] < 0.05]
    high_pos_neurons = stim_df[stim_df["pos_pvalue"] < 0.05]
    return {"high_bhv": high_bhv_neurons, "high_neg": high_neg_neurons, "high_pos": high_pos_neurons}


def _style_violin(parts, colors: list[str]) -> None:
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor("none")
        pc.set_edgecolor(colors[i])
        pc.set_linewidth(8)
    for key in ["cmins", "cmaxes", "cbars"]:
        parts[key].set_color("black")
        parts[key].set_linewidth(2.5)
    parts["cmeans"].set_color("#ff7f0e")
    parts["cmeans"].set_linewidth(2.5)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(2.5)


def plot_bhv_neurons_neg_vs_pos_correlation(high_bhv_neurons: pd.DataFrame) -> tuple[plt.Figure, float, float]:
    """Among high-behaviour-correlation neurons, compare their negative vs.
    positive valence-regressor correlation (paired Wilcoxon).

    Source: cell 10.
    """
    bhv_corr = high_bhv_neurons["bhv_correlation"].dropna()
    neg_corr = high_bhv_neurons["neg_correlation"].dropna()
    pos_corr = high_bhv_neurons["pos_correlation"].dropna()

    paired_index = bhv_corr.index.intersection(neg_corr.index).intersection(pos_corr.index)
    neg_corr_aligned = neg_corr.loc[paired_index]
    pos_corr_aligned = pos_corr.loc[paired_index]

    if len(neg_corr_aligned) == 0 or len(pos_corr_aligned) == 0:
        raise ValueError("Error: One of the datasets is empty. Check data filtering and alignment.")
    wilcoxon_stat, p_value = wilcoxon(neg_corr_aligned, pos_corr_aligned)

    fig = plt.figure(figsize=(5.5, 5.5))
    parts = plt.violinplot([neg_corr_aligned, pos_corr_aligned], showmeans=True, showextrema=True, showmedians=True)
    _style_violin(parts, ["magenta", "green"])

    plt.xticks([1, 2], ["Negative", "Positive"], fontsize=16)
    plt.text(1.6, 1, f"p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}", fontsize=26, ha="center")
    plt.ylabel("Correlation (r)", fontsize=28)
    plt.tick_params(labelsize=28)
    plt.tick_params(axis="both", which="major", length=8, width=2)
    plt.tick_params(axis="both", which="minor", length=4, width=1)
    plt.ylim(-0.5, 1)
    plt.yticks([-0.5, 0, 0.5, 1])

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    plt.tight_layout()
    return fig, wilcoxon_stat, p_value


def plot_valence_neurons_bhv_correlation(high_pos_neurons: pd.DataFrame, high_neg_neurons: pd.DataFrame) -> tuple[plt.Figure, float, float]:
    """Compare behaviour-regressor correlation between positive-valence and
    negative-valence neurons (unpaired Mann-Whitney U).

    Source: cell 13.
    """
    bhv_corr_pos = high_pos_neurons["bhv_correlation"].dropna()
    bhv_corr_neg = high_neg_neurons["bhv_correlation"].dropna()
    u_stat, p_value = mannwhitneyu(bhv_corr_pos, bhv_corr_neg, alternative="two-sided")

    fig = plt.figure(figsize=(5, 6))
    parts = plt.violinplot([bhv_corr_neg, bhv_corr_pos], showmeans=True, showextrema=True, showmedians=True)
    _style_violin(parts, ["magenta", "green"])

    plt.xticks([1, 2], ["-VEN", "+VEN"], fontsize=16)
    plt.text(1.5, 1, f"p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}", fontsize=20, ha="center")
    plt.ylabel("BHV Correlation (r)", fontsize=28)
    plt.tick_params(labelsize=28)
    plt.tick_params(axis="both", which="major", length=8, width=2)
    plt.tick_params(axis="both", which="minor", length=4, width=1)
    plt.ylim(-0.5, 1)
    plt.yticks([-0.5, 0, 0.5, 1])

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    plt.tight_layout()
    return fig, u_stat, p_value
