"""
Asymmetric Bout-Regressor Correlation Analysis

Calculates correlations between neural activity and ASYMMETRIC swim bouts,
with significance testing via trial-shuffle permutation testing.

Asymmetric bouts are classified using the circular mean of tail angles during
the first 70 ms of each bout. A binary regressor marks frames during asymmetric
turning bouts (1 = asymmetric bout, 0 = otherwise), which is then downsampled
to the neural sampling rate and convolved with a calcium indicator kernel.

This version is optimized by:
1. Computing correlations for all neurons at once using matrix operations
2. Computing shuffled correlations for all neurons at once
3. Avoiding neuron-by-neuron / shuffle-by-shuffle Python loops
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import convolve
from scipy.stats import circmean

warnings.filterwarnings("ignore")


# ── Defaults (overridable via CLI) ────────────────────────────────────────────

_FISH_LIST: list[int] = [
    230713, 230714, 230720, 230727, 230728,
    230810, 230811, 230817, 230818, 230919,
]

# Behavioral processing defaults
_ORIGINAL_FREQ: float = 60.85   # Hz
_TARGET_FREQ: float = 3.0       # Hz
_BOUT_DURATION_MS: int = 70     # ms for laterality classification
_LEFT_THRESHOLD: float = -0.08  # radians
_RIGHT_THRESHOLD: float = 0.08  # radians
_EXPECTED_FRAMES_PER_TRIAL: int = 300

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


# ── Bout classification ────────────────────────────────────────────────────────

def concatenate_row(row: pd.Series) -> np.ndarray:
    """Concatenate array-like entries in a row, ignoring missing values."""
    arrays = [x for x in row.values if isinstance(x, np.ndarray)]
    return np.concatenate(arrays) if arrays else np.array([])


def classify_asymm_bouts(
    row: pd.Series,
    left_threshold: float,
    right_threshold: float,
    sampling_rate: float,
    duration_ms: int,
) -> list[bool]:
    """Classify each bout as asymmetric (True) or symmetric (False).

    Uses the circular mean of tail angles during the first *duration_ms* ms
    of each bout to determine laterality.
    """
    bouts: list[list[int]] = row["bout_indx"]
    tail_angle: np.ndarray = row["bhv_trace"]
    duration_frames = max(1, int((duration_ms / 1000) * sampling_rate))

    flags: list[bool] = []
    for bout in bouts:
        if not bout:
            flags.append(False)
            continue
        segment = np.asarray(tail_angle[bout[0]: min(bout[0] + duration_frames, len(tail_angle))])
        if len(segment) == 0:
            flags.append(False)
            continue
        lat = circmean(segment, high=np.pi, low=-np.pi)
        flags.append(lat < left_threshold or lat > right_threshold)

    return flags


def create_asymmetric_bout_regressor(
    total_frames: int,
    bouts: list[list[int]],
    asymmetric_flags: list[bool],
) -> np.ndarray:
    """Create a binary regressor marking frames during asymmetric bouts."""
    regressor = np.zeros(total_frames, dtype=float)
    for bout, is_asymmetric in zip(bouts, asymmetric_flags):
        if is_asymmetric and bout:
            start = max(0, int(bout[0]))
            end = min(total_frames - 1, int(bout[-1]))
            regressor[start: end + 1] = 1.0
    return regressor


def downsample_binary_regressor(
    regressor: np.ndarray,
    original_freq: float,
    target_freq: float,
    expected_frames: int,
) -> np.ndarray:
    """Downsample a binary regressor using max-pooling within each window.

    Note:
        Uses an integer approximation to the decimation factor
        (60.85 / 3 ≈ 20), matching the original notebook workflow.
    """
    decimation_factor = int(original_freq / target_freq)
    if decimation_factor <= 1:
        raise ValueError("Target frequency must be lower than original frequency.")

    downsampled = np.maximum.reduceat(
        regressor, np.arange(0, len(regressor), decimation_factor)
    )

    if len(downsampled) > expected_frames:
        downsampled = downsampled[:expected_frames]
    elif len(downsampled) < expected_frames:
        downsampled = np.pad(
            downsampled, (0, expected_frames - len(downsampled)), mode="constant"
        )
    return downsampled


# ── Data processing ────────────────────────────────────────────────────────────

def load_and_process_behavior(
    data: pd.DataFrame,
    kernel: np.ndarray,
    original_freq: float,
    target_freq: float,
    bout_duration_ms: int,
    left_threshold: float,
    right_threshold: float,
    expected_frames: int,
) -> np.ndarray:
    """Build a concatenated convolved asymmetric-bout regressor for one fish."""
    data = data.copy()
    data["bhv_trace"] = data["angles"].apply(lambda x: np.asarray(json.loads(x)))
    data["bout_indx"] = data["bouts"].apply(json.loads)

    data["asymmetric_bout"] = data.apply(
        classify_asymm_bouts,
        axis=1,
        left_threshold=left_threshold,
        right_threshold=right_threshold,
        sampling_rate=original_freq,
        duration_ms=bout_duration_ms,
    )
    data["asymmetric_regressor"] = data.apply(
        lambda row: create_asymmetric_bout_regressor(
            len(row["bhv_trace"]), row["bout_indx"], row["asymmetric_bout"]
        ),
        axis=1,
    )
    data["downsampled_asymmetric_regressor"] = data["asymmetric_regressor"].apply(
        lambda t: downsample_binary_regressor(t, original_freq, target_freq, expected_frames)
    )
    data["convolved_regressor"] = data["downsampled_asymmetric_regressor"].apply(
        lambda t: convolve_trace(t, kernel)
    )

    reg_df = data.pivot_table(
        index="fish_id",
        columns=["stimulus", "trial_number"],
        values="convolved_regressor",
        aggfunc="first",
    )
    reg_df["concat_trace"] = reg_df.apply(concatenate_row, axis=1)

    if reg_df.empty:
        raise ValueError("No behavioral regressor created for this fish.")
    regressor = reg_df["concat_trace"].iloc[0]
    if len(regressor) == 0:
        raise ValueError("Concatenated behavioral regressor is empty.")
    return regressor


def load_neural_data(file_path: Path, fish: str, end: str) -> pd.DataFrame:
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
    return pd.DataFrame({
        "correlation": neuro_data["regressor_corr"],
        "coords": neuro_data["coords_serialised"],
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
    num_shuffles: int,
    trial_length: int,
    random_seed: int,
) -> pd.DataFrame:
    """Process one dataset (fb or hb) for one fish."""
    print(f"  Loading {dataset_name} neural data...")
    neural_data = load_neural_data(neural_path, fish, end=end)
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
    fb_bhv_path: Path,
    hb_bhv_path: Path,
    fb_neural_path: Path,
    hb_neural_path: Path,
    output_path: Path,
    original_freq: float,
    target_freq: float,
    bout_duration_ms: int,
    left_threshold: float,
    right_threshold: float,
    expected_frames: int,
    tau_rise: float,
    tau_decay: float,
    num_shuffles: int,
    trial_length: int,
    random_seed: int,
) -> None:
    """Run asymmetric bout–neural correlation analysis across all fish."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
        fb_regressor = load_and_process_behavior(
            fish_fb, kernel, original_freq, target_freq,
            bout_duration_ms, left_threshold, right_threshold, expected_frames,
        )
        print("  Processing hindbrain behavior...")
        hb_regressor = load_and_process_behavior(
            fish_hb, kernel, original_freq, target_freq,
            bout_duration_ms, left_threshold, right_threshold, expected_frames,
        )

        for path, end, label, reg in [
            (fb_neural_path, "_fb", "forebrain", fb_regressor),
            (hb_neural_path, "_hb", "hindbrain", hb_regressor),
        ]:
            try:
                result = process_one_dataset(
                    neural_path=path,
                    fish=str(fish),
                    end=end,
                    regressor=reg,
                    dataset_name=label,
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
                   default=Path("/Volumes/LaCie/larval_HuC/imaging/7dpf_new_vector_asymm_bout_all_correlations_1000.csv"))
    p.add_argument("--original-freq", type=float, default=_ORIGINAL_FREQ)
    p.add_argument("--target-freq", type=float, default=_TARGET_FREQ)
    p.add_argument("--bout-duration-ms", type=int, default=_BOUT_DURATION_MS)
    p.add_argument("--left-threshold", type=float, default=_LEFT_THRESHOLD)
    p.add_argument("--right-threshold", type=float, default=_RIGHT_THRESHOLD)
    p.add_argument("--expected-frames", type=int, default=_EXPECTED_FRAMES_PER_TRIAL)
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
        fb_bhv_path=args.fb_bhv,
        hb_bhv_path=args.hb_bhv,
        fb_neural_path=args.fb_neural,
        hb_neural_path=args.hb_neural,
        output_path=args.output,
        original_freq=args.original_freq,
        target_freq=args.target_freq,
        bout_duration_ms=args.bout_duration_ms,
        left_threshold=args.left_threshold,
        right_threshold=args.right_threshold,
        expected_frames=args.expected_frames,
        tau_rise=args.tau_rise,
        tau_decay=args.tau_decay,
        num_shuffles=args.num_shuffles,
        trial_length=args.trial_length,
        random_seed=args.random_seed,
    )
    print("\nAnalysis complete!")
