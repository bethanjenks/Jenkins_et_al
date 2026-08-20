"""Behavioral-vigor / neural-activity correlation analysis.

Ported from `scripts/run_bhv_vigour_regression.py`; logic unchanged. Shared
z-score/shuffle/correlation/neural-loading core moved to
`jenkins_et_al.regression.core` -- see that module's docstring for what's
shared across all four regression scripts and the one deliberate deviation
(`coords` always parsed).

Behavioral vigor is computed from tail-angle traces via running standard
deviation (all movement: swim bouts + postural adjustments), downsampled with
an anti-aliasing Butterworth filter, then convolved with a calcium indicator
kernel to match neural dynamics. Neural/behavioral data are both filtered to
`_NEURO_STIMULI` before pivoting -- one of two scripts (with
`stimulus_specific`) that restrict to a stimulus subset rather than the full
15-stimulus battery.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

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
NEURO_STIMULI: tuple[str, ...] = (
    "ade", "cad_2.5mm", "fex_1", "kw", "ph4.5", "pro_2.5mm", "qui_2.5mm",
)

ORIGINAL_FREQ: float = 60.85  # Hz
TARGET_FREQ: float = 3.0      # Hz
WINDOW_SIZE_MS: int = 100     # ms for running std

SAMPLING_RATE: float = 3.0    # Hz
TAU_RISE: float = 1.44 / SAMPLING_RATE
TAU_DECAY: float = 5.05 / SAMPLING_RATE

NUM_SHUFFLES: int = 1000
TRIAL_LENGTH: int = 300
RANDOM_SEED: int = 42


def downsample_trace(trace: np.ndarray, original_freq: float, target_freq: float) -> np.ndarray:
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
    data["bhv_vigour"] = data["bhv_trace"].apply(lambda t: running_std(t, window_size_samples))
    data["downsampled_vigour"] = data["bhv_vigour"].apply(
        lambda t: downsample_trace(t, original_freq, target_freq)
    )
    data["convolved_vigour"] = data["downsampled_vigour"].apply(
        lambda t: convolve_trace(np.abs(t), kernel)
    )

    if "stimulus" in data.columns:
        # Confirmed bug, fixed here: the real behavioral CSV spells this
        # stimulus "fex1" (no underscore), but `neuro_stimuli` uses the
        # neural-side spelling "fex_1" (see module docstring). As saved, this
        # filter drops "fex1" entirely, shortening the behavioral regressor
        # relative to the neural trace and raising a length-mismatch error
        # for every fish -- the original script could never complete a real
        # run. Aliasing here, not upstream in the raw data, is the fix.
        stimulus = data["stimulus"].replace({"fex1": "fex_1"})
        data = data[stimulus.isin(neuro_stimuli)].copy()

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
    neural_data = load_neural_data(neural_path, fish, end=end, neuro_stimuli=neuro_stimuli)
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


def run_bhv_vigour_regression(
    fish_list: list[int],
    neuro_stimuli: list[str],
    fb_bhv_path: Path,
    hb_bhv_path: Path,
    fb_neural_path: Path,
    hb_neural_path: Path,
    original_freq: float = ORIGINAL_FREQ,
    target_freq: float = TARGET_FREQ,
    window_size_ms: int = WINDOW_SIZE_MS,
    tau_rise: float = TAU_RISE,
    tau_decay: float = TAU_DECAY,
    num_shuffles: int = NUM_SHUFFLES,
    trial_length: int = TRIAL_LENGTH,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Run behavioral vigor-neural correlation analysis across all fish."""
    window_size_samples = int(original_freq * (window_size_ms / 1000.0))

    fb_data = pd.read_csv(fb_bhv_path)
    hb_data = pd.read_csv(hb_bhv_path)

    t = np.arange(0, 10, 1 / SAMPLING_RATE)
    kernel = calcium_indicator_kernel(t, tau_rise, tau_decay)

    all_corr: list[pd.DataFrame] = []

    for fish in fish_list:
        fish_fb = fb_data[fb_data["fish_id"] == fish].copy()
        fish_hb = hb_data[hb_data["fish_id"] == fish].copy()

        if fish_fb.empty or fish_hb.empty:
            continue

        fb_vigour = load_and_process_behavior(
            fish_fb, original_freq, target_freq, window_size_samples, kernel, neuro_stimuli
        )
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

    if not all_corr:
        raise ValueError("No results generated.")
    return pd.concat(all_corr, ignore_index=True)
