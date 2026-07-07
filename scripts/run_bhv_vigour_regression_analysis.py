"""
Behavioral-Regressor Correlation Analysis

Calculates correlations between neural activity and behavioral vigor,
with significance testing via trial-shuffle permutation testing.

Behavioral vigor is computed from tail angle traces using running standard
deviation (includes all movement: swim bouts + postural adjustments),
then convolved with a calcium indicator kernel to match neural dynamics.

This version is optimized by:
1. Computing correlations for all neurons at once using matrix operations
2. Computing shuffled correlations for all neurons at once
3. Avoiding neuron-by-neuron / shuffle-by-shuffle Python loops
"""

from __future__ import annotations

import argparse
import json
import warnings
from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, convolve, filtfilt

warnings.filterwarnings("ignore")


# ── Defaults (overridable via CLI) ────────────────────────────────────────────

_FISH_LIST: list[int] = [
    230713, 230714, 230720, 230727, 230728,
    230810, 230811, 230817, 230818, 230919,
]
_NEURO_STIMULI: list[str] = [
    "ade", "cad_2.5mm", "fex_1", "kw", "ph4.5", "pro_2.5mm", "qui_2.5mm",
]

# Behavioral processing defaults
_ORIGINAL_FREQ: float = 60.85   # Hz
_TARGET_FREQ: float = 3.0       # Hz
_WINDOW_SIZE_MS: int = 100      # ms for running std

# Calcium indicator kernel defaults
_SAMPLING_RATE: float = 3.0     # Hz
_TAU_RISE: float = 1.44 / _SAMPLING_RATE
_TAU_DECAY: float = 5.05 / _SAMPLING_RATE


# ── Calcium indicator kernel ───────────────────────────────────────────────────

def calcium_indicator_kernel(
    t: np.ndarray,
    tau_rise: float,
    tau_decay: float,
) -> np.ndarray:
    """Generate a normalised calcium indicator kernel."""
    kernel = np.exp(-t / tau_decay) - np.exp(-t / tau_rise)
    total = np.sum(kernel)
    if total == 0:
        raise ValueError("Calcium kernel sum is zero.")
    return kernel / total


def convolve_trace(trace: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve trace with calcium kernel, preserving original length."""
    return convolve(trace, kernel, mode="full")[: len(trace)]


# ── Behavioral preprocessing ──────────────────────────────────────────────────

def concatenate_row(row: pd.Series) -> np.ndarray:
    """Concatenate array-like entries in a row, ignoring missing values."""
    arrays = [x for x in row.values if isinstance(x, np.ndarray)]
    return np.concatenate(arrays) if arrays else np.array([])


def downsample_trace(
    trace: np.ndarray,
    original_freq: float,
    target_freq: float,
) -> np.ndarray:
    """Downsample a trace with an anti-aliasing Butterworth filter."""
    decimation_factor = int(original_freq / target_freq)
    if decimation_factor <= 1:
        raise ValueError("Target frequency must be lower than original frequency.")
    b, a = butter(4, (target_freq / 2.0) / (original_freq / 2.0))
    return filtfilt(b, a, trace)[::decimation_factor]


def running_std(trace: np.ndarray, window_size_samples: int) -> np.ndarray:
    """Running standard deviation over a trace."""
    return (
        pd.Series(np.asarray(trace, dtype=float))
        .rolling(window=window_size_samples)
        .std()
        .fillna(0)
        .to_numpy()
    )


def load_and_process_behavior(
    data: pd.DataFrame,
    original_freq: float,
    target_freq: float,
    window_size_samples: int,
    kernel: np.ndarray,
    neuro_stimuli: list[str],
) -> np.ndarray:
    """Build a concatenated convolved vigor regressor for one fish."""
    data = data.copy()
    data["bhv_trace"] = data["angles"].apply(lambda x: np.asarray(json.loads(x)))
    data["bhv_vigour"] = data["bhv_trace"].apply(
        lambda t: running_std(t, window_size_samples)
    )
    data["downsampled_vigour"] = data["bhv_vigour"].apply(
        lambda t: downsample_trace(t, original_freq, target_freq)
    )
    data["convolved_vigour"] = data["downsampled_vigour"].apply(
        lambda t: convolve_trace(np.abs(t), kernel)
    )

    if "stimulus" in data.columns:
        data = data[data["stimulus"].isin(neuro_stimuli)].copy()

    vigor_df = data.pivot_table(
        index="fish_id",
        columns=["stimulus", "trial_number"],
        values="convolved_vigour",
        aggfunc="first",
    )
    vigor_df["concat_trace"] = vigor_df.apply(concatenate_row, axis=1)

    if vigor_df.empty:
        raise ValueError("No behavioral vigor regressor created for this fish.")

    regressor = vigor_df["concat_trace"].iloc[0]
    if len(regressor) == 0:
        raise ValueError("Concatenated behavioral vigor regressor is empty.")
    return regressor


# ── Neural data functions ──────────────────────────────────────────────────────

def load_neural_data(
    file_path: Path,
    fish: str,
    end: str,
    neuro_stimuli: list[str],
) -> pd.DataFrame:
    """Load neural data for one fish from HDF5."""
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
    if "stimulus" in neural_data.columns:
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


# ── Vectorised correlation + shuffle test ─────────────────────────────────────

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
    """Build a clean results DataFrame."""
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
    neural_path: Path,
    fish: str,
    end: str,
    regressor: np.ndarray,
    dataset_name: str,
    neuro_stimuli: list[str],
    num_shuffles: int,
    trial_length: int,
    random_seed: int,
) -> pd.DataFrame:
    """Process one dataset (fb or hb) for one fish."""
    print(f"  Loading {dataset_name} neural data...")
    neural_data = load_neural_data(neural_path, fish, end=end, neuro_stimuli=neuro_stimuli)
    print(f"  Concatenating {dataset_name} neural traces...")
    key_data = concatenate_neural_data(neural_data)
    print(f"  Computing {dataset_name} correlations and p-values...")
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
    fb_bhv_path: Path,
    hb_bhv_path: Path,
    fb_neural_path: Path,
    hb_neural_path: Path,
    output_path: Path,
    original_freq: float,
    target_freq: float,
    window_size_ms: int,
    tau_rise: float,
    tau_decay: float,
    num_shuffles: int,
    trial_length: int,
    random_seed: int,
) -> None:
    """Run behavioral vigor–neural correlation analysis across all fish."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    window_size_samples = int(original_freq * (window_size_ms / 1000.0))

    print("Loading behavioral data...")
    fb_data = pd.read_csv(fb_bhv_path)
    hb_data = pd.read_csv(hb_bhv_path)

    print("Generating calcium indicator kernel...")
    t = np.arange(0, 10, 1 / _SAMPLING_RATE)
    kernel = calcium_indicator_kernel(t, tau_rise, tau_decay)

    all_corr: list[pd.DataFrame] = []

    for fish in fish_list:
        print(f"\nProcessing fish: {fish}")
        fish_fb = fb_data[fb_data["fish_id"] == fish].copy()
        fish_hb = hb_data[hb_data["fish_id"] == fish].copy()

        if fish_fb.empty:
            print(f"  Skipping {fish}: no forebrain behavior data.")
            continue
        if fish_hb.empty:
            print(f"  Skipping {fish}: no hindbrain behavior data.")
            continue

        print("  Processing forebrain behavior...")
        fb_vigour = load_and_process_behavior(
            fish_fb, original_freq, target_freq, window_size_samples, kernel, neuro_stimuli
        )
        print("  Processing hindbrain behavior...")
        hb_vigour = load_and_process_behavior(
            fish_hb, original_freq, target_freq, window_size_samples, kernel, neuro_stimuli
        )

        for path, end, label, vigour in [
            (fb_neural_path, "_fb", "forebrain", fb_vigour),
            (hb_neural_path, "_hb", "hindbrain", hb_vigour),
        ]:
            try:
                result = process_one_dataset(
                    neural_path=path,
                    fish=str(fish),
                    end=end,
                    regressor=vigour,
                    dataset_name=label,
                    neuro_stimuli=neuro_stimuli,
                    num_shuffles=num_shuffles,
                    trial_length=trial_length,
                    random_seed=random_seed,
                )
                if not result.empty:
                    all_corr.append(result)
            except Exception as e:
                print(f"  {label.capitalize()} failed for fish {fish}: {e}")

        if all_corr:
            pd.concat(all_corr, ignore_index=True).to_csv(output_path, index=False)
            print(f"  Saved progress to: {output_path}")

    if not all_corr:
        raise ValueError("No results generated.")

    pd.concat(all_corr, ignore_index=True).to_csv(output_path, index=False)
    print(f"\nFinal results saved to: {output_path}")


# ── CLI entry point ────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fb-bhv", type=Path,
                   default=Path("/Volumes/LaCie/larval_HuC/behavior/7dpf_fb_behavior_traces.csv"))
    p.add_argument("--hb-bhv", type=Path,
                   default=Path("/Volumes/LaCie/larval_HuC/behavior/7dpf_hb_behavior_traces.csv"))
    p.add_argument("--fb-neural", type=Path,
                   default=Path("/Volumes/LaCie/larval_HuC/imaging/7dpf_fb_fish_dfs_final.h5"))
    p.add_argument("--hb-neural", type=Path,
                   default=Path("/Volumes/LaCie/larval_HuC/imaging/7dpf_hb_fish_dfs_final.h5"))
    p.add_argument("--output", type=Path,
                   default=Path("/Volumes/LaCie/larval_HuC/imaging/7dpf_bhv_vigor_all_correlations_1000.csv"))
    p.add_argument("--original-freq", type=float, default=_ORIGINAL_FREQ)
    p.add_argument("--target-freq", type=float, default=_TARGET_FREQ)
    p.add_argument("--window-size-ms", type=int, default=_WINDOW_SIZE_MS)
    p.add_argument("--tau-rise", type=float, default=_TAU_RISE)
    p.add_argument("--tau-decay", type=float, default=_TAU_DECAY)
    p.add_argument("--num-shuffles", type=int, default=1000)
    p.add_argument("--trial-length", type=int, default=300)
    p.add_argument("--random-seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        fish_list=_FISH_LIST,
        neuro_stimuli=_NEURO_STIMULI,
        fb_bhv_path=args.fb_bhv,
        hb_bhv_path=args.hb_bhv,
        fb_neural_path=args.fb_neural,
        hb_neural_path=args.hb_neural,
        output_path=args.output,
        original_freq=args.original_freq,
        target_freq=args.target_freq,
        window_size_ms=args.window_size_ms,
        tau_rise=args.tau_rise,
        tau_decay=args.tau_decay,
        num_shuffles=args.num_shuffles,
        trial_length=args.trial_length,
        random_seed=args.random_seed,
    )
    print("\nAnalysis complete!")
