"""Straight-bout / neural-activity correlation analysis.

Ported from `scripts/run_straight_bout_regression.py`; logic unchanged.
Shared core moved to `jenkins_et_al.regression.core` -- see that module's
docstring, including why this script's lack of stimulus filtering (unlike
`bhv_vigour`/`stimulus_specific`) is a deliberate difference in analysis
scope, not a bug.

Straight bouts are classified via the circular mean of tail angles during the
first `BOUT_DURATION_MS` of each bout. A binary regressor marks frames during
straight bouts (1 = straight, 0 = otherwise), downsampled to the neural
sampling rate with max-pooling (appropriate for a binary signal, unlike
`bhv_vigour`'s anti-aliasing Butterworth filter for a continuous one), then
convolved with the same calcium indicator kernel.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import circmean

from jenkins_et_al.regression.core import (
    calcium_indicator_kernel,
    compute_correlations_and_pvalues_fast,
    concatenate_neural_data,
    concatenate_row,
    convolve_trace,
    create_significance_results_df,
    load_neural_data,
)

warnings.filterwarnings("ignore")

#: Source: module-level constants in the original script.
FISH_LIST: tuple[int, ...] = (
    230713, 230714, 230720, 230727, 230728,
    230810, 230811, 230817, 230818, 230919,
)

ORIGINAL_FREQ: float = 60.85     # Hz
TARGET_FREQ: float = 3.0         # Hz
BOUT_DURATION_MS: int = 70       # ms used for straight/turn classification
LEFT_THRESHOLD: float = -0.08    # radians
RIGHT_THRESHOLD: float = 0.08    # radians
EXPECTED_FRAMES_PER_TRIAL: int = 300

SAMPLING_RATE: float = 3.0       # Hz
TAU_RISE: float = 1.44 / SAMPLING_RATE
TAU_DECAY: float = 5.05 / SAMPLING_RATE

NUM_SHUFFLES: int = 1000
TRIAL_LENGTH: int = 300
RANDOM_SEED: int = 42


def classify_straight_bouts(
    row: pd.Series,
    left_threshold: float,
    right_threshold: float,
    sampling_rate: float,
    duration_ms: int,
) -> list[bool]:
    """Classify each bout as straight (True) or turning/asymmetric (False).

    Assumes `row["bout_indx"]` is a list of lists, each inner list holding
    all frame indices belonging to one bout.
    """
    straight_flags = []
    bouts = row["bout_indx"]
    tail_angle = row["bhv_trace"]
    duration_frames = max(1, int((duration_ms / 1000) * sampling_rate))

    for bout in bouts:
        if len(bout) < 1:
            straight_flags.append(False)
            continue

        bout_start = bout[0]
        bout_end = min(bout_start + duration_frames, len(tail_angle))
        bout_angles = np.asarray(tail_angle[bout_start:bout_end])

        if len(bout_angles) == 0:
            straight_flags.append(False)
            continue

        laterality_index = circmean(bout_angles, high=np.pi, low=-np.pi)
        straight_flags.append(left_threshold <= laterality_index <= right_threshold)

    return straight_flags


def create_straight_bout_regressor(
    total_frames: int,
    bouts: list[list[int]],
    straight_flags: list[bool],
) -> np.ndarray:
    """Create a binary regressor marking frames during straight bouts."""
    regressor = np.zeros(total_frames, dtype=float)
    for bout, is_straight in zip(bouts, straight_flags):
        if is_straight and len(bout) > 0:
            bout_start = max(0, int(bout[0]))
            bout_end = min(total_frames - 1, int(bout[-1]))
            regressor[bout_start : bout_end + 1] = 1
    return regressor


def downsample_binary_regressor(
    regressor: np.ndarray,
    original_freq: float,
    target_freq: float,
    expected_frames: int,
) -> np.ndarray:
    """Downsample a binary regressor using max-pooling within each window.

    Uses an integer approximation to the decimation factor (60.85 / 3 ~= 20)
    -- preserves the original workflow and ensures exactly `expected_frames`
    output.
    """
    decimation_factor = int(original_freq / target_freq)
    if decimation_factor <= 1:
        raise ValueError("Target frequency must be lower than original frequency.")

    downsampled = np.maximum.reduceat(regressor, np.arange(0, len(regressor), decimation_factor))

    if len(downsampled) > expected_frames:
        downsampled = downsampled[:expected_frames]
    elif len(downsampled) < expected_frames:
        downsampled = np.pad(downsampled, (0, expected_frames - len(downsampled)), mode="constant")
    return downsampled


def load_and_process_data(
    data: pd.DataFrame,
    kernel: np.ndarray,
    original_freq: float = ORIGINAL_FREQ,
    target_freq: float = TARGET_FREQ,
    bout_duration_ms: int = BOUT_DURATION_MS,
    left_threshold: float = LEFT_THRESHOLD,
    right_threshold: float = RIGHT_THRESHOLD,
    expected_frames: int = EXPECTED_FRAMES_PER_TRIAL,
) -> np.ndarray:
    """Build one concatenated convolved straight-bout regressor for one fish."""
    data = data.copy()
    data["bhv_trace"] = data["angles"].apply(lambda x: np.asarray(json.loads(x)))
    data["bout_indx"] = data["bouts"].apply(json.loads)

    data["straight_bout"] = data.apply(
        classify_straight_bouts,
        axis=1,
        left_threshold=left_threshold,
        right_threshold=right_threshold,
        sampling_rate=original_freq,
        duration_ms=bout_duration_ms,
    )
    data["straight_regressor"] = data.apply(
        lambda row: create_straight_bout_regressor(
            total_frames=len(row["bhv_trace"]),
            bouts=row["bout_indx"],
            straight_flags=row["straight_bout"],
        ),
        axis=1,
    )
    data["downsampled_straight_regressor"] = data["straight_regressor"].apply(
        lambda t: downsample_binary_regressor(t, original_freq, target_freq, expected_frames)
    )
    data["convolved_regressor"] = data["downsampled_straight_regressor"].apply(
        lambda t: convolve_trace(t, kernel)
    )

    regressor_df = data.pivot_table(
        index="fish_id",
        columns=["stimulus", "trial_number"],
        values="convolved_regressor",
        aggfunc="first",
    )
    regressor_df["concat_trace"] = regressor_df.apply(concatenate_row, axis=1)

    if regressor_df.shape[0] == 0:
        raise ValueError("No behavioral regressor created for this fish.")

    regressor = regressor_df["concat_trace"].iloc[0]
    if len(regressor) == 0:
        raise ValueError("Concatenated behavioral regressor is empty.")
    return regressor


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
    neural_data = load_neural_data(neural_path, fish, end=end)
    key_data = concatenate_neural_data(neural_data)
    key_data = compute_correlations_and_pvalues_fast(
        key_data=key_data,
        regressor=regressor,
        num_shuffles=num_shuffles,
        trial_length=trial_length,
        random_seed=random_seed,
        two_tailed=False,
    )
    return create_significance_results_df(key_data)


def run_straight_bout_regression(
    fish_list: list[int],
    fb_bhv_path: Path,
    hb_bhv_path: Path,
    fb_neural_path: Path,
    hb_neural_path: Path,
    num_shuffles: int = NUM_SHUFFLES,
    trial_length: int = TRIAL_LENGTH,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Run straight-bout-neural correlation analysis across all fish."""
    fb_data = pd.read_csv(fb_bhv_path)
    hb_data = pd.read_csv(hb_bhv_path)

    t = np.arange(0, 10, 1 / SAMPLING_RATE)
    kernel = calcium_indicator_kernel(t, TAU_RISE, TAU_DECAY)

    all_corr: list[pd.DataFrame] = []

    for fish in fish_list:
        fish_fb = fb_data[fb_data["fish_id"] == fish].copy()
        fish_hb = hb_data[hb_data["fish_id"] == fish].copy()

        if fish_fb.empty or fish_hb.empty:
            continue

        fb_regressor = load_and_process_data(fish_fb, kernel)
        hb_regressor = load_and_process_data(fish_hb, kernel)

        for path, end, label, regressor in [
            (fb_neural_path, "_fb", "forebrain", fb_regressor),
            (hb_neural_path, "_hb", "hindbrain", hb_regressor),
        ]:
            try:
                result = process_one_dataset(
                    neural_path=path,
                    fish=str(fish),
                    end=end,
                    regressor=regressor,
                    dataset_name=label,
                    num_shuffles=num_shuffles,
                    trial_length=trial_length,
                    random_seed=random_seed,
                )
                if not result.empty:
                    all_corr.append(result)
            except Exception as e:
                print(f"  {label.capitalize()} failed for fish {fish}: {e}")

    if not all_corr:
        raise ValueError("No results generated.")
    return pd.concat(all_corr, ignore_index=True)
