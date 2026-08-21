"""Shared core for the three valence/behaviour-correlation notebooks.

Ported from `notebooks/neural/continuous_correlation_hex_heatmaps.ipynb`,
`cross_regressor_neuron_correlations.ipynb`, and
`neuron_counts_and_overlaps.ipynb` -- all three share a byte-identical
preamble (cells 0-6): load the area-shorthand dict, load the nMLF mask,
load and merge the positive/negative/behaviour correlation CSVs into one
`stim_df`. Each notebook's own docstring notes it was "split from
`Compare_Val_BHV_neurons.ipynb`", which explains the shared preamble.

Reuses `jenkins_et_al.neural.area_pair_core.load_nmlf_mask`/
`is_within_3d_mask` rather than a second copy -- confirmed byte-identical to
what these three notebooks each independently redefine.

**`eval()` replaced with `literal_eval()`**: the notebooks' own coordinate-
parsing line uses bare `eval()`, the same issue already fixed in
`spontaneous_correlations.py` -- functionally equivalent for this data,
changed for safety, not a behavior change.

**Confirmed-harmless leftover columns, reproduced as-is**: `neg`'s and
`bhv`'s own `Unnamed: 0`/`Unnamed: 0.1`/`p_value` columns are never dropped
(unlike `pos`'s, which are) -- the notebook's own commented-out
`# neg.drop(...)` line shows this was intended but never finished. The
merge produces suffixed junk columns (`Unnamed: 0.1_x`, `p_value_y`, etc.)
that no downstream cell in any of the three notebooks reads. Reproduced
faithfully (not cleaned up) so `load_valence_bhv_dataframe`'s output column
set matches the original notebooks' `stim_df` exactly, confirmed against
real data (200116 rows, same column list).
"""
from __future__ import annotations

from ast import literal_eval
from pathlib import Path

import pandas as pd

from jenkins_et_al.neural.area_pair_core import is_within_3d_mask


def load_valence_bhv_dataframe(
    pos_path: Path,
    neg_path: Path,
    bhv_path: Path,
    nmlf_mask=None,
) -> pd.DataFrame:
    """Load and merge the positive/negative/behaviour correlation CSVs into
    one per-neuron DataFrame, with nMLF-mask area relabeling.

    Source: cells 3-6, all three notebooks (identical).
    """
    pos = pd.read_csv(pos_path)
    pos.rename(columns={"correlation": "pos_correlation", "p_fdr": "pos_pvalue"}, inplace=True)
    pos.drop(columns=["Unnamed: 0", "p_value", "Unnamed: 0.1"], inplace=True)

    neg = pd.read_csv(neg_path)
    neg.rename(columns={"correlation": "neg_correlation", "p_fdr": "neg_pvalue"}, inplace=True)

    bhv = pd.read_csv(bhv_path)
    bhv.rename(columns={"correlation": "bhv_correlation", "p_fdr": "bhv_pvalue"}, inplace=True)

    stim_df = bhv.merge(pos, on=["fish_id", "neuron_id", "area", "coords"], how="outer").merge(
        neg, on=["fish_id", "neuron_id", "area", "coords"], how="outer"
    )

    order = ["fish_id", "neuron_id", "area", "coords"]
    all_columns = order + [col for col in stim_df.columns if col not in order]
    stim_df = stim_df[all_columns]

    stim_df["coords_tuple"] = stim_df["coords"].apply(literal_eval)
    if nmlf_mask is not None:
        stim_df["area"] = stim_df.apply(
            lambda row: "nMLF" if is_within_3d_mask(row["coords_tuple"], nmlf_mask) else row["area"],
            axis=1,
        )
    return stim_df
