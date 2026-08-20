"""Asymmetric (turning) bout / neural-activity correlation analysis.

Ported from `scripts/run_turning_bout_regression.py`; logic unchanged.
Shared core moved to `jenkins_et_al.regression.core` -- see that module's
docstring, including why this script's lack of stimulus filtering is a
deliberate difference in analysis scope from `bhv_vigour`/`stimulus_specific`,
not a bug.

Asymmetric (turning) bouts are classified via the same circular-mean
laterality test as `straight_bout`, but with the comparison inverted: a bout
is asymmetric when its laterality falls *outside* the threshold band, rather
than inside it. Kept as this module's own `classify_asymm_bouts` rather than
sharing `straight_bout.classify_straight_bouts` with a flag, since the two
were independently verified in this same port and unifying them isn't forced
per this project's "don't generalize until each site is independently
verified" rule -- flagged here as a candidate for a future unification pass.
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
BOUT_DURATION_MS: int = 70       # ms for laterality classification
LEFT_THRESHOLD: float = -0.08    # radians
RIGHT_THRESHOLD: float = 0.08    # radians
EXPECTED_FRAMES_PER_TRIAL: int = 300

SAMPLING_RATE: float = 3.0       # Hz
TAU_RISE: float = 1.44 / SAMPLING_RATE
TAU_DECAY: float = 5.05 / SAMPLING_RATE

NUM_SHUFFLES: int = 1000
TRIAL_LENGTH: int = 300
RANDOM_SEED: int = 42


def classify_asymm_bouts(
    row: pd.Series,
    left_threshold: float,
    right_threshold: float,
    sampling_rate: float,
    duration_ms: int,
) -> list[bool]:
    """Classify each bout as asymmetric (True) or symmetric (False).

    Uses the circular mean of tail angles during the first `duration_ms` of
    each bout to determine laterality.
    """
    bouts: list[list[int]] = row["bout_indx"]
    tail_angle: np.ndarray = row["bhv_trace"]
    duration_frames = max(1, int((duration_ms / 1000) * sampling_rate))

    flags: list[bool] = []
    for bout in bouts:
        if not bout:
            flags.append(False)
            continue
        segment = np.asarray(tail_angle[bout[0] : min(bout[0] + duration_frames, len(tail_angle))])
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
            regressor[start : end + 1] = 1.0
    return regressor


def downsample_binary_regressor(
    regressor: np.ndarray,
    original_freq: float,
    target_freq: float,
    expected_frames: int,
) -> np.ndarray:
    """Downsample a binary regressor using max-pooling within each window.

    Uses an integer approximation to the decimation factor (60.85 / 3 ~= 20),
    matching the original script.
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


def load_and_process_behavior(
    data: pd.DataFrame,
    kernel: np.ndarray,
    original_freq: float = ORIGINAL_FREQ,
    target_freq: float = TARGET_FREQ,
    bout_duration_ms: int = BOUT_DURATION_MS,
    left_threshold: float = LEFT_THRESHOLD,
    right_threshold: float = RIGHT_THRESHOLD,
    expected_frames: int = EXPECTED_FRAMES_PER_TRIAL,
) -> np.ndarray:
    """Build one concatenated convolved asymmetric-bout regressor for one fish."""
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


def run_turning_bout_regression(
    fish_list: list[int],
    fb_bhv_path: Path,
    hb_bhv_path: Path,
    fb_neural_path: Path,
    hb_neural_path: Path,
    num_shuffles: int = NUM_SHUFFLES,
    trial_length: int = TRIAL_LENGTH,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Run asymmetric-bout-neural correlation analysis across all fish."""
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

        fb_regressor = load_and_process_behavior(fish_fb, kernel)
        hb_regressor = load_and_process_behavior(fish_hb, kernel)

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
