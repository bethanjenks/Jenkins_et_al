"""Spontaneous correlation area-pair analysis.

Ported from `notebooks/neural/spontaneous_correlations_area_pair_analysis.ipynb`.
Shared utilities and the per-fish/per-area-pair analysis loop moved to
`jenkins_et_al.neural.area_pair_core` -- see that module's docstring.

**Spontaneous correlations** quantify shared activity between neurons during
a pre-stimulus baseline window, independent of any evoked response: a
configurable frame window (`spont_start:spont_end`) is sliced from each
trial's raw fluorescence trace (`trace_serialised`, a full timeseries, not a
scalar) and concatenated across all stimulus x trial combinations into one
long vector per neuron. This is the real difference from
`noise_correlations.py`, which subtracts a per-stimulus mean from a scalar
response value instead.

**`eval()` replaced with `literal_eval()`**: the notebook's own cell 13
parses `coords` via bare `eval()`, inconsistent with `load_group_data`'s own
`literal_eval()` two cells earlier and every other coordinate-parsing site
in this codebase. Functionally equivalent for this data (coordinate
tuples/lists), `literal_eval` used here for safety -- not a behavior change.

**Confirmed bug, reconstructed at the user's request**: same missing
`fish_intra_means`/`fish_inter_means` issue as `noise_correlations.py` --
see that module's docstring. This notebook's own version is more complete
(it does build a proper per-fish KS-test loop, `ks_records`, before failing
to reduce to flat mean lists), and its per-fish KS loop is what
`area_pair_core.per_fish_ks_distributions` is ported from.

**Visualisation section headers mislabeled, not a logic bug**: the
notebook's own "8.2" through "8.7" markdown headers under Visualisation are
shifted relative to their actual cell content (e.g. the cell under "8.2 Mean
Correlation Heatmap (Real)" is actually the real-vs-random distribution
plot). Read top-to-bottom the code itself is coherent and produces
correctly-named output files -- only the markdown titles are out of sync.
Not reproduced here since this module organizes by function name, not
notebook cell position.
"""
from __future__ import annotations

import json
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

#: Source: cell 5.
FISH_LIST: tuple[str, ...] = (
    "230713_fb", "230714_fb", "230720_fb", "230727_fb", "230728_fb",
    "230810_fb", "230811_fb", "230817_fb", "230818_fb", "230919_fb",
)
BRAIN_AREAS: tuple[str, ...] = (
    "olfactory_bulb", "pallium", "subpallium", "preoptic",
    "posterior_tuberculum", "intermediate_hypothalamus", "nMLF",
    "tegmentum", "nucleus_isthmi",
)
SPONT_START = 80
SPONT_END = 150

N_RANDOM_SAMPLES = 1000
FDR_ALPHA = 0.05
MIN_PAIRS_THRESHOLD = 100
MIN_NEURONS_PER_GROUP = 2
PCTL_HI = 95
PCTL_LO = 5


def make_load_fish_activity(
    fish_data_path: Path,
    group1_df: pd.DataFrame,
    group2_df: pd.DataFrame,
    nmlf_mask,
    spont_start: int = SPONT_START,
    spont_end: int = SPONT_END,
):
    """Build the `load_fish_activity` closure for `run_area_pair_analysis`.

    Ported from cell 13's per-fish preprocessing (spontaneous-window
    extraction and concatenation, area relabeling, group assignment, pivot
    to one row per neuron).
    """

    def load_fish_activity(fish: str) -> pd.DataFrame:
        fish_df = pd.read_hdf(fish_data_path, where=f"fish_id == '{fish}'")

        fish_df = fish_df.copy()
        fish_df["trace"] = fish_df["trace_serialised"].apply(lambda x: np.asarray(json.loads(x)))
        fish_df["coords"] = fish_df["coords_serialised"]
        fish_df["coords_tuple"] = fish_df["coords"].apply(literal_eval)
        if nmlf_mask is not None:
            fish_df["area"] = fish_df.apply(
                lambda row: "nMLF"
                if is_within_3d_mask(row["coords_tuple"], nmlf_mask)
                else row["area"],
                axis=1,
            )

        fish_df["spontaneous_activity"] = fish_df["trace"].apply(
            lambda x: x[spont_start:spont_end]
        )

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
            values="spontaneous_activity",
            aggfunc="first",
        )
        pivot_df["activity"] = pivot_df.apply(
            lambda row: np.concatenate([np.asarray(x) for x in row.dropna()]), axis=1
        )
        return pivot_df[["activity"]].reset_index()

    return load_fish_activity


def run_spontaneous_correlation_analysis(
    fish_list: list[str],
    fish_data_path: Path,
    group1_df: pd.DataFrame,
    group2_df: pd.DataFrame,
    brain_areas: list[str],
    compare_same_group: bool,
    use_group1_for_both: bool,
    use_group2_for_both: bool,
    nmlf_mask=None,
    spont_start: int = SPONT_START,
    spont_end: int = SPONT_END,
    min_neurons_per_group: int = MIN_NEURONS_PER_GROUP,
    n_random_samples: int = N_RANDOM_SAMPLES,
    pctl_hi: float = PCTL_HI,
    pctl_lo: float = PCTL_LO,
) -> tuple[pd.DataFrame, dict, dict]:
    """Run the full spontaneous-correlation area-pair analysis.

    Returns:
        (summary_df, area_pair_corr_values, random_pair_corr_values)
    """
    load_fish_activity = make_load_fish_activity(
        fish_data_path, group1_df, group2_df, nmlf_mask, spont_start, spont_end,
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
