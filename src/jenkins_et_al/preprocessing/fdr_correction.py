"""Per-fish BH-FDR correction for neuron-correlation CSVs.

Ported from `notebooks/preprocessing/fdr_correction.ipynb`; logic unchanged.
This is the missing step behind the `..._corrected.csv` files referenced
throughout the rest of this project (e.g. the positive/negative valence
correlation CSVs consumed by `regression/`, `neural/area_pair_core.py`,
`neural/valence_bhv_core.py`) -- it's a general utility applied once per raw
correlation CSV, not tied to one specific stimulus combination, despite the
notebook's own saved state showing only one example input file.

**Confirmed bug, fixed**: as saved, this notebook cannot run past its main
cell -- `multipletests` is called but never imported (the notebook's own
import cell has only `numpy`/`pandas`). Confirmed against real data:
`NameError: name 'multipletests' is not defined`. Fixed by importing it,
matching how it's already imported and used identically elsewhere in this
project (`neural/area_pair_core.py`'s `fisher_combine_pvalues`).

**`eval()` replaced with `literal_eval()`**: the notebook's own `coords`
parsing uses bare `eval()`, the same issue already fixed in
`spontaneous_correlations.py`/`valence_bhv_core.py`. Confirmed the
eval-then-restringify round-trip through `coords.apply(eval)` followed by
`to_csv` reproduces the exact original string for real data -- functionally
inert either way, changed for safety, not a behavior change.
"""
from __future__ import annotations

from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


def load_correlations(csv_path: Path) -> pd.DataFrame:
    """Load a raw correlation CSV and parse its `coords` column.

    Source: cell 1.
    """
    data = pd.read_csv(csv_path)
    data["coords"] = data["coords"].apply(literal_eval)
    return data


def apply_per_fish_fdr_correction(data: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Apply BH-FDR correction to `p_value`, separately per fish, adding
    a `p_fdr` column.

    Source: cells 2-3.
    """
    fdr_results = []
    for fish_id in data["fish_id"].unique():
        fish_data = data[data["fish_id"] == fish_id].copy()
        pvals = fish_data["p_value"].values

        if len(pvals) > 0:
            _, pvals_corrected, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
            fish_data["p_fdr"] = pvals_corrected
        else:
            fish_data["p_fdr"] = np.nan

        fdr_results.append(fish_data)

    return pd.concat(fdr_results, ignore_index=True)


def save_corrected_correlations(data_fdr: pd.DataFrame, output_path: Path) -> None:
    """Write the FDR-corrected table to CSV.

    Source: cell 4.
    """
    data_fdr.to_csv(output_path)
