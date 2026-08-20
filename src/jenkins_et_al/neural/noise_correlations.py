"""Noise correlation area-pair analysis.

Ported from `notebooks/neural/noise_correlations_area_pair_analysis.ipynb`
(replaces the notebook previously tracked as item #8 in PORTING_PLAN.md,
which the user has since replaced entirely with this one). Shared
utilities and the per-fish/per-area-pair analysis loop moved to
`jenkins_et_al.neural.area_pair_core` -- see that module's docstring.

**Noise correlations** quantify shared trial-to-trial variability between
neurons after removing stimulus-evoked responses: per-neuron-per-stimulus
mean response is subtracted (`compute_residuals`) from a scalar response
value per (neuron, stimulus, trial) -- not a full timeseries -- read from a
whole-brain "response vector" file. This is the real difference from
`spontaneous_correlations.py`, which correlates raw fluorescence windows
instead.

**Confirmed bug, reconstructed at the user's request**: the notebook's own
"intra vs inter" distribution section references `fish_intra_means`/
`fish_inter_means` (and, in this notebook specifically, also `ks_D_values`/
`fish_used` and an unimported `smf` for `smf.mixedlm`) that are never
defined anywhere in the notebook -- `NameError` if run fresh. Execution
counts confirm it's orphaned: those cells are numbered 42/43 while
everything else in the notebook is 75-98. Reconstructed here using
`area_pair_core.reduce_to_per_fish_means`/`per_fish_ks_distributions`,
following the per-fish-mean-reduction pattern already working elsewhere in
the same notebook (the "real vs random" distribution cell). The mixed-effects
model (`smf.mixedlm`) is not reconstructed -- unlike the per-fish-mean
reduction, there's no working sibling cell elsewhere in either notebook to
infer its intended form from, so it's left out rather than guessed at.
"""
from __future__ import annotations

from ast import literal_eval
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from jenkins_et_al.neural.area_pair_core import (
    is_within_3d_mask,
    run_area_pair_analysis,
    reduce_to_per_fish_means,
    per_fish_ks_distributions,
)

#: Source: cell 4.
FISH_LIST: tuple[str, ...] = (
    "230713_fb", "230714_fb", "230720_fb", "230727_fb", "230728_fb",
    "230810_fb", "230811_fb", "230817_fb", "230818_fb", "230919_fb",
)
BRAIN_AREAS: tuple[str, ...] = (
    "olfactory_bulb", "pallium", "subpallium", "preoptic",
    "posterior_tuberculum", "intermediate_hypothalamus", "nMLF",
    "tegmentum", "nucleus_isthmi",
)
NEGATIVE_STIMULI: tuple[str, ...] = ("ade", "cad_2.5mm", "qui_2.5mm")
POSITIVE_STIMULI: tuple[str, ...] = ("fex_1", "kw", "pro_2.5mm")

N_RANDOM_SAMPLES = 1000
FDR_ALPHA = 0.05
MIN_PAIRS_THRESHOLD = 100
MIN_NEURONS_PER_GROUP = 2
PCTL_HI = 95
PCTL_LO = 5


def compute_residuals(fish_df: pd.DataFrame) -> pd.DataFrame:
    """Subtract per-neuron-per-stimulus mean response from `resp`."""
    fish_df = fish_df.copy()
    fish_df["residual"] = fish_df.groupby(["neuron_id", "stimulus"])["resp"].transform(
        lambda x: x - x.mean()
    )
    return fish_df


def make_load_fish_activity(
    fish_data_path: Path,
    group1_df: pd.DataFrame,
    group2_df: pd.DataFrame,
    nmlf_mask,
    stimulus_filter: list[str] | None = None,
    behavior_filter: str | None = None,
    bhv_summary: pd.DataFrame | None = None,
):
    """Build the `load_fish_activity` closure for `run_area_pair_analysis`.

    Ported from cell 12's per-fish preprocessing (residual computation,
    area relabeling, optional stimulus/behaviour filtering, group
    assignment, pivot to one row per neuron).
    """

    def load_fish_activity(fish: str) -> pd.DataFrame:
        fish_df = pd.read_hdf(fish_data_path, where=f"fish_id == '{fish}'")

        if stimulus_filter is not None:
            fish_df = fish_df[fish_df["stimulus"].isin(stimulus_filter)]

        fish_df = compute_residuals(fish_df)

        fish_df["coords"] = fish_df["coords_serialised"]
        fish_df["coords_tuple"] = fish_df["coords"].apply(literal_eval)
        if nmlf_mask is not None:
            fish_df["area"] = fish_df.apply(
                lambda row: "nMLF"
                if is_within_3d_mask(row["coords_tuple"], nmlf_mask)
                else row["area"],
                axis=1,
            )

        if behavior_filter is not None and bhv_summary is not None:
            fish_df["trial_number"] = fish_df["trial_number"].astype(int)
            fish_df = pd.merge(
                fish_df,
                bhv_summary[["fish_id", "stimulus", "trial_number", "bhv_change"]],
                on=["fish_id", "stimulus", "trial_number"],
                how="inner",
            )
            fish_df = fish_df[fish_df["bhv_change"] == behavior_filter]

        g1_neurons = group1_df[group1_df["fish_id"] == fish]["neuron_id"]
        g2_neurons = group2_df[group2_df["fish_id"] == fish]["neuron_id"]
        fish_df["group"] = "OTHER"
        fish_df.loc[fish_df["neuron_id"].isin(g1_neurons), "group"] = "G1"
        fish_df.loc[
            fish_df["neuron_id"].isin(g2_neurons) & (fish_df["group"] != "G1"), "group"
        ] = "G2"

        pivot_df = fish_df.pivot_table(
            index=["coords", "area", "group"],
            columns=["stimulus", "trial_number"],
            values="residual",
        ).dropna(axis=0, how="all").reset_index()

        drop_cols = ["coords", "area", "group"]
        pivot_df["activity"] = pivot_df.drop(columns=drop_cols).apply(
            lambda row: row.values.astype(float), axis=1
        )
        return pivot_df[["coords", "area", "group", "activity"]]

    return load_fish_activity


def run_noise_correlation_analysis(
    fish_list: list[str],
    fish_data_path: Path,
    group1_df: pd.DataFrame,
    group2_df: pd.DataFrame,
    brain_areas: list[str],
    compare_same_group: bool,
    use_group1_for_both: bool,
    use_group2_for_both: bool,
    nmlf_mask=None,
    stimulus_filter: list[str] | None = None,
    behavior_filter: str | None = None,
    bhv_summary: pd.DataFrame | None = None,
    min_neurons_per_group: int = MIN_NEURONS_PER_GROUP,
    n_random_samples: int = N_RANDOM_SAMPLES,
    pctl_hi: float = PCTL_HI,
    pctl_lo: float = PCTL_LO,
) -> tuple[pd.DataFrame, dict, dict]:
    """Run the full noise-correlation area-pair analysis.

    Returns:
        (summary_df, area_pair_corr_values, random_pair_corr_values)
    """
    load_fish_activity = make_load_fish_activity(
        fish_data_path, group1_df, group2_df, nmlf_mask,
        stimulus_filter, behavior_filter, bhv_summary,
    )
    return run_area_pair_analysis(
        fish_list=fish_list,
        load_fish_activity=load_fish_activity,
        brain_areas=brain_areas,
        compare_same_group=compare_same_group,
        use_group1_for_both=use_group1_for_both,
        use_group2_for_both=use_group2_for_both,
        min_neurons_per_group=min_neurons_per_group,
        n_random_samples=n_random_samples,
        pctl_hi=pctl_hi,
        pctl_lo=pctl_lo,
    )


def intra_vs_inter_summary(
    area_pair_corr_values: dict, brain_areas: list[str],
) -> dict[str, Any]:
    """Reconstructed "intra vs inter" summary -- see module docstring."""
    from scipy.stats import ks_2samp, mannwhitneyu, wilcoxon

    intra_corrs: list[float] = []
    inter_corrs: list[float] = []
    for (area1, area2), fish_dict in area_pair_corr_values.items():
        if area1 not in brain_areas or area2 not in brain_areas:
            continue
        for _fish, corr_values in fish_dict.items():
            if not corr_values:
                continue
            (intra_corrs if area1 == area2 else inter_corrs).extend(corr_values)

    intra_vals = np.array(intra_corrs, dtype=float)
    inter_vals = np.array(inter_corrs, dtype=float)
    intra_vals = intra_vals[~np.isnan(intra_vals)]
    inter_vals = inter_vals[~np.isnan(inter_vals)]

    fish_used, fish_intra_means, fish_inter_means = reduce_to_per_fish_means(
        area_pair_corr_values, brain_areas
    )
    ks_df = per_fish_ks_distributions(area_pair_corr_values, fish_used, brain_areas)

    u_stat, u_p = mannwhitneyu(intra_vals, inter_vals, alternative="two-sided")
    ks_stat, ks_p = ks_2samp(intra_vals, inter_vals)
    w_stat, w_p = (
        wilcoxon(fish_intra_means, fish_inter_means, alternative="two-sided")
        if len(fish_intra_means) > 0
        else (np.nan, np.nan)
    )
    ks_d_values = ks_df["ks_D"].to_numpy() if not ks_df.empty else np.array([])
    ksd_w_stat, ksd_w_p = wilcoxon(ks_d_values) if len(ks_d_values) > 0 else (np.nan, np.nan)

    return {
        "intra_vals": intra_vals, "inter_vals": inter_vals,
        "fish_used": fish_used,
        "fish_intra_means": fish_intra_means, "fish_inter_means": fish_inter_means,
        "ks_df": ks_df,
        "pooled_mwu_stat": u_stat, "pooled_mwu_p": u_p,
        "pooled_ks_stat": ks_stat, "pooled_ks_p": ks_p,
        "per_fish_wilcoxon_stat": w_stat, "per_fish_wilcoxon_p": w_p,
        "ks_d_wilcoxon_stat": ksd_w_stat, "ks_d_wilcoxon_p": ksd_w_p,
    }
