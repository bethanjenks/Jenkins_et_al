"""Build a head-fixed behaviour reorientation dataframe from tail-angle traces.

Ported from notebooks/preprocessing/process_headfixed_BHV.ipynb (cells 2, 27,
29, 41); logic unchanged from the notebook, only moved into a module and given
a docstring/type-hint pass per the /tdd REFACTOR checklist. The notebook also
defines `find_bouts_array_adjusted` (an angle-diff bout finder), but nothing
in the notebook's own pipeline calls it -- only `find_bouts` (running-std bout
finder) feeds the saved dataframe -- so it was not ported.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def running_std(trace: np.ndarray, window_size_samples: int) -> np.ndarray:
    """Running standard deviation of `trace`, with NaNs (window warm-up) replaced by 0."""
    trace = np.asarray(trace)
    return pd.Series(trace).rolling(window=window_size_samples).std().fillna(0).to_numpy()


def find_bouts(
    angles: np.ndarray,
    fs: float,
    min_length: float = 0.05,
    stepsize: int = 10,
) -> list[list[int]]:
    """Find bout frame ranges from a tail-angle trace using a running-std threshold.

    Parameters
    ----------
    angles : numpy.ndarray
        Tail-tip angle for each frame of a continuously tracked trial.
    fs : float
        Sampling frequency (frames per second).
    min_length : float, optional
        Minimum bout duration in seconds.
    stepsize : int, optional
        Maximum frame gap allowed within a single bout.

    Returns
    -------
    list of list of int
        Frame indices for each detected bout, extended by one frame on either side.
    """
    def find_contiguous(data, minsize):
        runs = np.split(data, np.where(np.diff(data) > stepsize)[0] + 1)
        return [list(run) for run in runs if len(run) >= minsize]

    bhv_vigour = running_std(angles, 50)

    threshold = np.std(angles)
    above_threshold = np.where(bhv_vigour > threshold)[0]

    movement_indices = find_contiguous(above_threshold, minsize=int(min_length * fs))

    adjusted_movement_indices = []
    for movement_index in movement_indices:
        start_frame = max(0, movement_index[0] - 1)
        end_frame = min(len(angles) - 1, movement_index[-1] + 1)
        adjusted_movement_indices.append(np.arange(start_frame, end_frame + 1).tolist())

    return adjusted_movement_indices


def get_stimulus_name(file: str) -> str:
    """Extract the raw stimulus token from a `*_angles*` trial filename.

    Assumes the `movie_<date>_<n>_<stimulus>_<trial>_angles.npy` naming
    convention used by the source data; edit if trial files are named
    differently.
    """
    parts = file.split("movie_")[1].split("_angles.npy")
    return "_".join(parts[0].split("_")[1:])


def fix_stim_name(stim: str) -> str:
    """Rewrite a behaviour-side stimulus code to match the imaging-side naming.

    e.g. "cad_2.5" -> "cad_2.5mm" (concentration in mM), "cad_25" -> "cad_25um".
    """
    if "_" in stim and "." in stim:
        return stim + "mm"
    if "_" in stim:
        return stim + "um"
    return stim


def _load_trial(file: Path, fish_id: str, trace_length: int, sampling_frequency: float) -> dict:
    stimulus = get_stimulus_name(str(file))
    stim, trial_number = "_".join(stimulus.split("_")[:-1]), int(stimulus.split("_")[-1])
    stim = fix_stim_name(stim)

    trace = np.load(file)[:trace_length]
    if len(trace) < trace_length:
        trace = np.concatenate([trace, np.zeros(trace_length - len(trace))])

    angles = (trace - np.mean(trace)).tolist()
    bouts = find_bouts(angles, sampling_frequency, min_length=0.05)

    return {
        "angles": angles,
        "vigour": running_std(angles, 50),
        "bouts": bouts,
        "fish_id": fish_id,
        "stimulus": stim,
        "trial_number": str(trial_number),
    }


def build_bhv_dataframe(
    folders_path: Path,
    sampling_frequency: float = 60.85,
    trace_length: int = 6000,
) -> pd.DataFrame:
    """Build the per-trial reorientation dataframe from one fish-id folder tree.

    `folders_path` holds one subfolder per fish id, each containing
    `*_angles*.npy` tail-angle trace files. Each trace is cut/zero-padded to
    `trace_length` frames (default 6000, matching imaging trial length),
    normalised to zero mean, and bout frames are located with `find_bouts`.
    """
    bhv_df = pd.DataFrame()

    for data_dir in folders_path.iterdir():
        fish_id = data_dir.name
        fish_rows = [
            _load_trial(file, fish_id, trace_length, sampling_frequency)
            for file in data_dir.glob("*_angles*")
        ]

        fish_df = pd.DataFrame(fish_rows)
        bhv_df = pd.concat([bhv_df, fish_df], ignore_index=True)

    return bhv_df


def output_csv_path(folders_path: Path) -> Path:
    """Default output path for `build_bhv_dataframe`'s result: sibling to `folders_path`."""
    return folders_path.parent / f"{folders_path.name}_reorientation_df.csv"


def process_headfixed_bhv(
    folders_path: Path,
    sampling_frequency: float = 60.85,
    trace_length: int = 6000,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Build the reorientation dataframe from `folders_path` and save it to CSV."""
    bhv_df = build_bhv_dataframe(folders_path, sampling_frequency, trace_length)
    csv_path = output_path if output_path is not None else output_csv_path(folders_path)
    bhv_df.to_csv(csv_path)
    return bhv_df
