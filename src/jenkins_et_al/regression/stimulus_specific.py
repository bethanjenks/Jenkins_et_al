"""Stimulus-regressor / neural-activity correlation analysis.

Ported from `scripts/run_stimulus_specific_regression.py`; logic unchanged.
Shared core moved to `jenkins_et_al.regression.core` -- see that module's
docstring. Unlike `bhv_vigour`/`straight_bout`/`turning_bout`, this script
doesn't build its own behavioral regressor -- it correlates neural traces
against a pre-computed, pickled per-stimulus regressor dict.

One of two scripts (with `bhv_vigour`) that restrict neural data to a
stimulus subset (`NEURO_STIMULI`, six stimuli -- no `ph4.5`, unlike
`bhv_vigour`'s seven) rather than the full 15-stimulus battery.
"""
from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from jenkins_et_al.regression.core import (
    compute_correlations_and_pvalues_fast,
    concatenate_neural_data,
    create_significance_results_df,
    load_neural_data,
)

warnings.filterwarnings("ignore")

#: Source: module-level constants in the original script.
FISH_LIST: tuple[int, ...] = (
    230713, 230714, 230720, 230727, 230728,
    230810, 230811, 230817, 230818, 230919,
)
NEURO_STIMULI: tuple[str, ...] = (
    "ade", "cad_2.5mm", "fex_1", "kw", "pro_2.5mm", "qui_2.5mm",
)

NUM_SHUFFLES: int = 1000
TRIAL_LENGTH: int = 300
RANDOM_SEED: int = 42


def process_one_dataset(
    neural_data: pd.DataFrame,
    regressor: np.ndarray,
    neuro_stimuli: list[str],
    dataset_name: str,
    num_shuffles: int,
    trial_length: int,
    random_seed: int,
) -> pd.DataFrame:
    """Process one dataset (fb or hb) for one fish."""
    filtered = neural_data[neural_data["stimulus"].isin(neuro_stimuli)].copy()
    key_data = concatenate_neural_data(filtered) if not filtered.empty else pd.DataFrame()
    if key_data.empty:
        return pd.DataFrame()

    key_data = compute_correlations_and_pvalues_fast(
        key_data=key_data,
        regressor=regressor,
        num_shuffles=num_shuffles,
        trial_length=trial_length,
        random_seed=random_seed,
        two_tailed=False,
    )
    return create_significance_results_df(key_data)


def run_stimulus_specific_regression(
    fish_list: list[int],
    neuro_stimuli: list[str],
    fb_data_path: Path,
    hb_data_path: Path,
    regressor_path: Path,
    num_shuffles: int = NUM_SHUFFLES,
    trial_length: int = TRIAL_LENGTH,
    random_seed: int = RANDOM_SEED,
) -> dict[str, pd.DataFrame]:
    """Run correlation analysis for every pickled stimulus regressor, across all fish.

    Returns:
        Mapping of stimulus name to its results DataFrame (one entry per key
        in the pickled regressor dict, matching one output CSV per stimulus
        in the original script).
    """
    with regressor_path.open("rb") as fh:
        stim_regressors: dict[str, np.ndarray] = pickle.load(fh)

    results_by_stimulus: dict[str, pd.DataFrame] = {}

    for stim, regressor in stim_regressors.items():
        all_corr: list[pd.DataFrame] = []

        for fish in fish_list:
            fish_str = str(fish)
            for path, end, label in [
                (fb_data_path, "_fb", "forebrain"),
                (hb_data_path, "_hb", "hindbrain"),
            ]:
                try:
                    neural = load_neural_data(path, fish_str, end=end)
                    result = process_one_dataset(
                        neural_data=neural,
                        regressor=regressor,
                        neuro_stimuli=neuro_stimuli,
                        dataset_name=label,
                        num_shuffles=num_shuffles,
                        trial_length=trial_length,
                        random_seed=random_seed,
                    )
                    if not result.empty:
                        all_corr.append(result)
                except Exception as e:
                    print(f"    {label.capitalize()} failed: {e}")

        if all_corr:
            results_by_stimulus[stim] = pd.concat(all_corr, ignore_index=True)

    return results_by_stimulus
