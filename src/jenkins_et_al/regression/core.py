"""Shared core for the four neural-behavioral regression scripts.

Ported from `scripts/run_bhv_vigour_regression.py`,
`run_stimulus_specific_regression.py`, `run_straight_bout_regression.py`, and
`run_turning_bout_regression.py` -- all four independently define near-
identical versions of every function in this module (calcium kernel,
concatenation, z-scoring, trial-shuffle permutation test, neural data
loading), confirmed via direct comparison during this port. Extracted once
here; each script's own module keeps only what's actually specific to it
(behavioral-regressor construction).

**`create_significance_results_df`'s `coords` column, normalized at the
user's request**: two of the four originals (`run_bhv_vigour_regression.py`,
`run_stimulus_specific_regression.py`) parse `coords_serialised` via
`ast.literal_eval` before writing it out as `coords`; the other two
(`run_straight_bout_regression.py`, `run_turning_bout_regression.py`) don't --
their output `coords` column is the raw unparsed string. Same field, two
different formats depending which script wrote the CSV. The user chose to
normalize all four to always parse, rather than preserve the split
bug-for-bug -- this is a deliberate deviation from two of the four originals,
not a transcription error.

**Stimulus filtering is intentionally *not* unified**: `run_bhv_vigour_regression.py`
and `run_stimulus_specific_regression.py` filter neural/behavioral data down
to a specific stimulus subset before pivoting; `run_straight_bout_regression.py`
and `run_turning_bout_regression.py` don't filter at all, deliberately
correlating against the full 15-stimulus x 3-trial battery (45 trials per
neuron, matching `wholebrain_population_analysis.ipynb`'s `TRIALS_PER_NEURON`
constant) -- confirmed against real data that both the neural HDF5 and
behavioral CSV span the same 45-trial universe for the two unfiltered
scripts, so this isn't a length-mismatch bug, just a real difference in
analysis scope between the two script pairs. `load_neural_data` and
`concatenate_neural_data` below take an optional `neuro_stimuli` filter so
each caller can opt in or out, matching its own original.
"""
from __future__ import annotations

import json
from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import convolve


def calcium_indicator_kernel(t: np.ndarray, tau_rise: float, tau_decay: float) -> np.ndarray:
    """Generate a normalised calcium indicator kernel."""
    kernel = np.exp(-t / tau_decay) - np.exp(-t / tau_rise)
    total = np.sum(kernel)
    if total == 0:
        raise ValueError("Calcium kernel sum is zero.")
    return kernel / total


def convolve_trace(trace: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve trace with calcium kernel, preserving original length."""
    return convolve(trace, kernel, mode="full")[: len(trace)]


def concatenate_row(row: pd.Series) -> np.ndarray:
    """Concatenate array-like entries in a row, ignoring missing values."""
    arrays = [x for x in row.values if isinstance(x, np.ndarray)]
    return np.concatenate(arrays) if arrays else np.array([])


def load_neural_data(
    file_path: Path,
    fish: str,
    end: str,
    neuro_stimuli: list[str] | None = None,
) -> pd.DataFrame:
    """Load neural data for one fish from HDF5.

    `neuro_stimuli`, when given, filters to that stimulus subset -- see
    module docstring for which of the four original scripts do this.
    """
    fish_id = f"{fish}{end}"
    try:
        neural_data = pd.read_hdf(file_path, where=f'fish_id == "{fish_id}"')
    except Exception:
        neural_data = pd.read_hdf(file_path)
        neural_data = neural_data[neural_data["fish_id"] == fish_id].copy()

    if neural_data.empty:
        raise ValueError(f"No neural data found for fish_id {fish_id}")

    neural_data = neural_data.copy()
    neural_data["img_trace"] = neural_data["trace_serialised"].apply(
        lambda x: np.asarray(json.loads(x))
    )
    if neuro_stimuli is not None and "stimulus" in neural_data.columns:
        neural_data = neural_data[neural_data["stimulus"].isin(neuro_stimuli)].copy()
    return neural_data


def concatenate_neural_data(neural_data: pd.DataFrame) -> pd.DataFrame:
    """Concatenate neural traces across all stimuli and trials per neuron."""
    pivoted = neural_data.pivot_table(
        index=["fish_id", "neuron_id", "coords_serialised", "area"],
        columns=["stimulus", "trial_number"],
        values="img_trace",
        aggfunc="first",
    )
    pivoted["concat_trace"] = pivoted.apply(concatenate_row, axis=1)
    pivoted = pivoted.reset_index()
    pivoted = pivoted[pivoted["concat_trace"].apply(lambda x: len(x) > 0)].copy()

    if pivoted.empty:
        raise ValueError("No valid concatenated neural traces found.")
    return pivoted


def zscore_rows(X: np.ndarray) -> np.ndarray:
    """Z-score each row of a 2-D array."""
    X = np.asarray(X, dtype=float)
    std = np.nanstd(X, axis=1, keepdims=True)
    std[std == 0] = np.nan
    return (X - np.nanmean(X, axis=1, keepdims=True)) / std


def zscore_1d(x: np.ndarray) -> np.ndarray:
    """Z-score a 1-D array."""
    x = np.asarray(x, dtype=float)
    std = np.nanstd(x)
    if std == 0:
        return np.full_like(x, np.nan)
    return (x - np.nanmean(x)) / std


def shuffle_trials(
    regressor: np.ndarray,
    trial_length: int,
    num_shuffles: int,
    random_seed: int,
) -> np.ndarray:
    """Generate shuffled regressors by permuting trial order.

    Returns:
        Array of shape ``(num_shuffles, len(regressor))``.
    """
    regressor = np.asarray(regressor, dtype=float)
    if len(regressor) % trial_length != 0:
        raise ValueError(
            f"Regressor length ({len(regressor)}) must be divisible by "
            f"trial_length ({trial_length})."
        )
    rng = np.random.default_rng(random_seed)
    num_trials = len(regressor) // trial_length
    trials = regressor.reshape(num_trials, trial_length)
    shuffled = np.empty((num_shuffles, len(regressor)), dtype=float)
    for i in range(num_shuffles):
        shuffled[i] = trials[rng.permutation(num_trials)].reshape(-1)
    return shuffled


def compute_correlations_and_pvalues_fast(
    key_data: pd.DataFrame,
    regressor: np.ndarray,
    num_shuffles: int,
    trial_length: int,
    random_seed: int,
    two_tailed: bool = False,
) -> pd.DataFrame:
    """Compute real correlations and shuffle-test p-values for all neurons at once."""
    key_data = key_data.copy().reset_index(drop=True)

    if key_data.empty:
        key_data["regressor_corr"] = []
        key_data["p_value"] = []
        return key_data

    X = np.vstack(key_data["concat_trace"].values).astype(float)
    regressor = np.asarray(regressor, dtype=float)

    if X.shape[1] != len(regressor):
        raise ValueError(
            f"Neural trace length ({X.shape[1]}) does not match "
            f"regressor length ({len(regressor)})."
        )

    Xz = zscore_rows(X)
    rz = zscore_1d(regressor)
    real_corr = np.nanmean(Xz * rz[None, :], axis=1)

    Rz = zscore_rows(shuffle_trials(regressor, trial_length, num_shuffles, random_seed))
    shuffle_corrs = (Xz @ Rz.T) / Xz.shape[1]

    if two_tailed:
        p_values = np.mean(np.abs(shuffle_corrs) >= np.abs(real_corr[:, None]), axis=1)
    else:
        p_values = np.mean(shuffle_corrs >= real_corr[:, None], axis=1)

    key_data["regressor_corr"] = real_corr
    key_data["p_value"] = p_values
    return key_data


def create_significance_results_df(neuro_data: pd.DataFrame) -> pd.DataFrame:
    """Build a clean results DataFrame, with `coords` always parsed.

    See module docstring: two of the four originals skip the `literal_eval`
    parse: this always applies it, at the user's request.
    """
    if neuro_data.empty:
        return pd.DataFrame(
            columns=["correlation", "coords", "area", "fish_id", "neuron_id", "p_value"]
        )

    neuro_data = neuro_data.copy()
    neuro_data["coords"] = neuro_data["coords_serialised"].apply(literal_eval)
    return pd.DataFrame({
        "correlation": neuro_data["regressor_corr"],
        "coords": neuro_data["coords"],
        "area": neuro_data["area"],
        "fish_id": neuro_data["fish_id"],
        "neuron_id": neuro_data["neuron_id"],
        "p_value": neuro_data["p_value"],
    })
