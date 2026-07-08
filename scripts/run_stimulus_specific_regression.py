"""
Stimulus-Regressor Correlation Analysis

Calculates correlations between neural traces and stimulus regressors,
with significance testing via trial-shuffle permutation test.

This vectorized version:
1. Computes correlations for all neurons at once
2. Computes shuffled correlations for all neurons at once
3. Avoids neuron-by-neuron / shuffle-by-shuffle Python loops
"""

from __future__ import annotations

import argparse
import json
import pickle
import warnings
from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ── Defaults (overridable via CLI) ────────────────────────────────────────────

_FISH_LIST: list[int] = [
    230713, 230714, 230720, 230727, 230728,
    230810, 230811, 230817, 230818, 230919,
]
_NEURO_STIMULI: list[str] = [
    "ade", "cad_2.5mm", "fex_1", "kw", "pro_2.5mm", "qui_2.5mm",
]


# ── Helper functions ───────────────────────────────────────────────────────────

def concatenate_row(row: pd.Series) -> np.ndarray:
    """Concatenate array-like entries in a row, ignoring missing values."""
    arrays = [x for x in row.values if isinstance(x, np.ndarray)]
    return np.concatenate(arrays) if arrays else np.array([])


def load_neural_data(file_path: Path, fish: str, end: str) -> pd.DataFrame:
    """Load neural data for a specific fish from an HDF5 file."""
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
    return neural_data


def filter_and_concatenate(
    neural_data: pd.DataFrame,
    neuro_stimuli: list[str],
) -> pd.DataFrame:
    """Filter for specific stimuli and concatenate trial traces per neuron."""
    filtered = neural_data[neural_data["stimulus"].isin(neuro_stimuli)].copy()

    if filtered.empty:
        return pd.DataFrame(
            columns=["fish_id", "neuron_id", "coords_serialised", "area", "concat_trace"]
        )

    pivoted = filtered.pivot_table(
        index=["fish_id", "neuron_id", "coords_serialised", "area"],
        columns=["stimulus", "trial_number"],
        values="img_trace",
        aggfunc="first",
    )
    pivoted["concat_trace"] = pivoted.apply(concatenate_row, axis=1)
    pivoted = pivoted.reset_index()
    return pivoted[pivoted["concat_trace"].apply(lambda x: len(x) > 0)].copy()


# ── Vectorised correlation + shuffle test ─────────────────────────────────────

def zscore_rows(X: np.ndarray) -> np.ndarray:
    """Z-score each row of a 2-D array."""
    X = np.asarray(X, dtype=float)
    mean = np.nanmean(X, axis=1, keepdims=True)
    std = np.nanstd(X, axis=1, keepdims=True)
    std[std == 0] = np.nan
    return (X - mean) / std


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
            f"trial_length ({trial_length})"
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

    Rz = zscore_rows(
        shuffle_trials(regressor, trial_length, num_shuffles, random_seed)
    )
    shuffle_corrs = (Xz @ Rz.T) / Xz.shape[1]

    if two_tailed:
        p_values = np.mean(np.abs(shuffle_corrs) >= np.abs(real_corr[:, None]), axis=1)
    else:
        p_values = np.mean(shuffle_corrs >= real_corr[:, None], axis=1)

    key_data["regressor_corr"] = real_corr
    key_data["p_value"] = p_values
    return key_data


def create_significance_results_df(neuro_data: pd.DataFrame) -> pd.DataFrame:
    """Build a clean results DataFrame with correlation and significance."""
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


# ── Main pipeline ──────────────────────────────────────────────────────────────

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
    print(f"    Processing {dataset_name}...")
    key_data = filter_and_concatenate(neural_data, neuro_stimuli)
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


def main(
    fish_list: list[int],
    neuro_stimuli: list[str],
    fb_data_path: Path,
    hb_data_path: Path,
    regressor_path: Path,
    output_dir: Path,
    num_shuffles: int,
    trial_length: int,
    random_seed: int,
) -> None:
    """Run correlation analysis for all stimuli across all fish."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with regressor_path.open("rb") as fh:
        stim_regressors: dict[str, np.ndarray] = pickle.load(fh)

    for stim, regressor in stim_regressors.items():
        print(f"Processing stimulus: {stim}")
        all_corr: list[pd.DataFrame] = []

        for fish in fish_list:
            fish_str = str(fish)
            print(f"  Fish: {fish_str}")

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

        if not all_corr:
            print(f"  No results for stimulus {stim}\n")
            continue

        out_path = output_dir / f"7dpf_{stim}_correlations_25s_1000.csv"
        pd.concat(all_corr, ignore_index=True).to_csv(out_path, index=False)
        print(f"  Saved: {out_path}\n")


# ── CLI entry point ────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fb-data", type=Path,
                   default=Path("/Volumes/LaCie/larval_HuC/imaging/7dpf_fb_fish_dfs_final.h5"),
                   help="Path to forebrain HDF5 file")
    p.add_argument("--hb-data", type=Path,
                   default=Path("/Volumes/LaCie/larval_HuC/imaging/7dpf_hb_fish_dfs_final.h5"),
                   help="Path to hindbrain HDF5 file")
    p.add_argument("--regressors", type=Path,
                   default=Path("/Volumes/LaCie/larval_HuC/combination_regressors_no_HCl_25s.pkl"),
                   help="Path to pickled stimulus regressors")
    p.add_argument("--output-dir", type=Path,
                   default=Path("/Volumes/LaCie/larval_HuC/imaging"),
                   help="Output directory for CSV results")
    p.add_argument("--num-shuffles", type=int, default=1000)
    p.add_argument("--trial-length", type=int, default=300)
    p.add_argument("--random-seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        fish_list=_FISH_LIST,
        neuro_stimuli=_NEURO_STIMULI,
        fb_data_path=args.fb_data,
        hb_data_path=args.hb_data,
        regressor_path=args.regressors,
        output_dir=args.output_dir,
        num_shuffles=args.num_shuffles,
        trial_length=args.trial_length,
        random_seed=args.random_seed,
    )
    print("Analysis complete!")
