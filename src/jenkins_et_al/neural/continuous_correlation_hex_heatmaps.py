"""Continuous valence-score / behaviour-correlation hexbin heatmaps.

Ported from `notebooks/neural/continuous_correlation_hex_heatmaps.ipynb`;
logic unchanged. Data loading moved to
`jenkins_et_al.neural.valence_bhv_core.load_valence_bhv_dataframe` -- see
that module's docstring.

For one brain area, restricts to neurons significant for at least one of
positive/negative/behaviour correlation, computes a continuous
`valence_score = pos_correlation - neg_correlation`, and plots three hexbin
heatmaps against `bhv_correlation` (turning-behaviour correlation), each
with its own Spearman rho/p.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr


def filter_significant_for_area(stim_df: pd.DataFrame, area: str) -> pd.DataFrame:
    """Restrict to one area, neurons significant for pos/neg/bhv, with all
    three correlation columns present.

    Source: cell 12, lines 1-14.
    """
    df = stim_df.copy()
    df = df[(df["pos_pvalue"] < 0.05) | (df["neg_pvalue"] < 0.05) | (df["bhv_pvalue"] < 0.05)]
    df = df[df["area"] == area]
    cols = ["bhv_correlation", "pos_correlation", "neg_correlation"]
    df = df.dropna(subset=cols)
    df["valence_score"] = df["pos_correlation"] - df["neg_correlation"]
    return df


def compute_spearman_correlations(df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Spearman rho/p for valence-score, positive, and negative correlation
    each against behaviour correlation.

    Source: cell 12, lines 15-20.
    """
    rho_val, p_val = spearmanr(df["valence_score"], df["bhv_correlation"])
    rho_pos, p_pos = spearmanr(df["pos_correlation"], df["bhv_correlation"])
    rho_neg, p_neg = spearmanr(df["neg_correlation"], df["bhv_correlation"])
    return {"valence": (rho_val, p_val), "positive": (rho_pos, p_pos), "negative": (rho_neg, p_neg)}


def plot_hexbin(
    x,
    y,
    xlabel: str,
    ylabel: str,
    title: str,
    savepath: Path | None = None,
    gridsize: int = 60,
    saturation_frac: float = 0.2,
) -> plt.Figure:
    """Hexbin heatmap with a saturated colormap (clim capped at
    `saturation_frac` of the max bin count) and dashed zero-reference lines.

    Source: cell 12's `plot_hexbin` helper.
    """
    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)

    hb = ax.hexbin(x, y, gridsize=gridsize, mincnt=1, cmap="viridis", extent=(-0.8, 0.8, -0.8, 0.8))
    max_count = hb.get_array().max()
    hb.set_clim(vmin=1, vmax=max_count * saturation_frac)

    ax.axvline(0, color="grey", linestyle="--", linewidth=1)
    ax.axhline(0, color="grey", linestyle="--", linewidth=1)
    ax.set_xlim(-0.8, 0.8)
    ax.set_ylim(-0.8, 0.8)
    ax.set_box_aspect(1)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_title(title, fontsize=18)
    ax.set_xticks([-0.5, 0, 0.5])
    ax.set_yticks([-0.8, -0.4, 0, 0.4, 0.8])
    ax.tick_params(axis="both", labelsize=20)

    cbar = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Neuron count", fontsize=18)
    cbar.ax.tick_params(labelsize=18)

    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight")
    return fig


def plot_valence_vs_behaviour_hexbins(
    df: pd.DataFrame,
    rhos: dict[str, tuple[float, float]],
    area: str,
    saturation_frac: float = 0.3,
    figure_output_dir: Path | None = None,
) -> dict[str, plt.Figure]:
    """The three hexbin figures: valence score, positive correlation, and
    negative correlation, each vs. behaviour correlation.

    Source: cell 12's three `plot_hexbin(...)` calls.
    """
    rho_val, _ = rhos["valence"]
    rho_pos, _ = rhos["positive"]
    rho_neg, _ = rhos["negative"]

    def savepath(name: str) -> Path | None:
        if figure_output_dir is None:
            return None
        return figure_output_dir / f"{area}_{name}_vs_L-BCN_corr_heatmap_sig.svg"

    figs = {}
    figs["valence_score"] = plot_hexbin(
        df["bhv_correlation"], df["valence_score"],
        xlabel="Turning behaviour correlation", ylabel="Valence score: pos - neg",
        title=f"{area} Valence score vs Turning behaviour\nSpearman rho = {rho_val:.3f}",
        savepath=savepath("valence_score"), saturation_frac=saturation_frac,
    )
    figs["positive"] = plot_hexbin(
        df["bhv_correlation"], df["pos_correlation"],
        xlabel="Turning behaviour correlation", ylabel="Positive-valence correlation",
        title=f"{area} Positive valence vs Turning behaviour\nSpearman rho = {rho_pos:.3f}",
        savepath=savepath("pos_valence"), saturation_frac=saturation_frac,
    )
    figs["negative"] = plot_hexbin(
        df["bhv_correlation"], df["neg_correlation"],
        xlabel="Turning behaviour correlation", ylabel="Negative-valence correlation",
        title=f"{area} Negative valence vs Turning behaviour\nSpearman rho = {rho_neg:.3f}",
        savepath=savepath("neg_valence"), saturation_frac=saturation_frac,
    )
    return figs
