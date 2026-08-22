"""Per-area mean traces and positive-vs-negative valence comparison.

Ported from `notebooks/neural/wholebrain_avg_traces_PCA_raster.ipynb` cells
0-11. This notebook replaces the earlier, paused
`wholebrain_population_analysis.ipynb` (deleted by the user) -- despite
loading hindbrain data (cell 5) and joining it into `stim_dff` (cell 6),
every actual analysis cell in this notebook operates on `stim_fb`
(forebrain only) or a `stim_df = stim_fb` alias; `stim_dff` is never read
again after being built. The hindbrain load/join is therefore not ported --
it's dead code in this version, confirmed by checking every later cell.

**Confirmed bug, fixed at the user's request**: cell 8's neuron count
(`len(area_df) / 45`) uses a divisor from a different dataset shape --
`area_df` only ever contains the 7-stimulus-filtered load from cell 4 (21
trials/neuron: 7 stimuli x 3 trials), confirmed uniform on real data, never
45 (the full 15-stimulus-battery trial count used elsewhere in this
project, e.g. `wholebrain_avg_traces_PCA_raster`'s own predecessor). There's
no unfiltered load anywhere in this notebook to justify /45; fixed to /21,
the divisor consistent with what's actually loaded.

**Confirmed bug, fixed at the user's request**: cell 11's `positive_stimuli`/
`negative_stimuli` mostly don't match anything in `all_mean_traces` (built
from the 7 stimuli in `KEY_STIMULI`) -- `'fex_2'` was never loaded (should
be `'fex_1'`), `'pro_2.5mM'`/`'cad_2.5mM'` don't match the lowercase forms
actually used (`'pro_2.5mm'`/`'cad_2.5mm'`), and `'qui'` has no match at all
(should be `'qui_2.5mm'`). As saved, the "positive vs negative" comparison
is actually 1-stimulus-vs-2-stimuli, not 3-vs-4. Fixed to the stimuli that
were actually intended and loaded.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

#: Source: cell 3.
KEY_STIMULI: tuple[str, ...] = ("ade", "cad_2.5mm", "kw", "fex_1", "qui_2.5mm", "pro_2.5mm", "ph4.5")
STIMULUS_COLORS: tuple[str, ...] = ("plum", "mediumorchid", "forestgreen", "limegreen", "deeppink", "olive", "fuchsia")
COLOR_MAP: dict[str, str] = dict(zip(KEY_STIMULI, STIMULUS_COLORS))

#: Source: cell 9.
LABEL_MAPPING: dict[str, str] = {
    "ade": "Ade", "cad_2.5mm": "Cad", "kw": "Kin", "fex_1": "Fex",
    "qui_2.5mm": "Qun", "pro_2.5mm": "Pro", "ph4.5": "HCl",
}

#: Source: cell 11, fixed -- see module docstring.
POSITIVE_STIMULI: tuple[str, ...] = ("fex_1", "kw", "pro_2.5mm")
NEGATIVE_STIMULI: tuple[str, ...] = ("ade", "cad_2.5mm", "ph4.5", "qui_2.5mm")

SAMPLING_FREQUENCY = 3  # Hz
SMOOTHING_SIGMA = 2
START_TIME = 35  # seconds
TRIALS_PER_NEURON = 21  # 7 stimuli x 3 trials -- see module docstring


def load_forebrain_data(file_path: Path, stimuli=KEY_STIMULI) -> pd.DataFrame:
    """Load and preprocess forebrain data, filtered to `stimuli`.

    Source: cell 4.
    """
    stim_fb = pd.read_hdf(file_path, where="stimulus in stimuli")
    stim_fb["img_trace"] = stim_fb.apply(lambda x: np.asarray(json.loads(x["trace_serialised"])), axis=1)
    stim_fb.rename(columns={"coords_serialised": "coords"}, inplace=True)
    stim_fb["neuron_id"] = stim_fb["fish_id"] + stim_fb["neuron_id"]
    return stim_fb


def count_neurons_in_area(area_df: pd.DataFrame, trials_per_neuron: int = TRIALS_PER_NEURON) -> int:
    """Number of distinct neurons in an area-filtered DataFrame.

    Source: cell 8.
    """
    return int(len(area_df) / trials_per_neuron)


def calculate_area_traces(area_df: pd.DataFrame, stimuli=KEY_STIMULI) -> dict[str, dict[str, np.ndarray]]:
    """Mean/SEM/STD trace per stimulus for one area.

    Source: cell 9 lines 1-11.
    """
    all_mean_traces: dict[str, np.ndarray] = {}
    all_sem_traces: dict[str, np.ndarray] = {}
    all_std_traces: dict[str, np.ndarray] = {}

    for stimulus in stimuli:
        neuron_traces = area_df[area_df["stimulus"] == stimulus]["img_trace"]
        if len(neuron_traces) == 0:
            continue
        stacked_traces = np.stack(neuron_traces.values)
        mean_trace = np.nanmean(stacked_traces, axis=0)
        std_trace = np.std(stacked_traces, axis=0)
        sem_trace = std_trace / np.sqrt(stacked_traces.shape[0])

        all_mean_traces[stimulus] = mean_trace
        all_sem_traces[stimulus] = sem_trace
        all_std_traces[stimulus] = std_trace

    return {"mean": all_mean_traces, "sem": all_sem_traces, "std": all_std_traces}


def plot_area_traces(
    traces: dict[str, dict[str, np.ndarray]],
    color_map: dict[str, str] = COLOR_MAP,
    label_map: dict[str, str] = LABEL_MAPPING,
    start_time: float = START_TIME,
    smoothing_sigma: float = SMOOTHING_SIGMA,
    sampling_frequency: float = SAMPLING_FREQUENCY,
    gradient_extent: tuple[float, float] = (43, 63),
    scale_bar_x: tuple[float, float] = (90, 110),
) -> plt.Figure:
    """Mean traces with SEM + 95% CI shading, a stimulus-period gradient, and
    scale bars (no axes) -- the notebook's own combined single-panel style.

    Source: cell 9 lines 12-.
    """
    time_interval = 1 / sampling_frequency
    fig = plt.figure(figsize=(7, 5))

    for stimulus in traces["mean"]:
        mean_trace = np.array(traces["mean"][stimulus])
        sem_trace = np.array(traces["sem"][stimulus])

        trace_length = len(mean_trace)
        time_seconds = np.arange(0, trace_length * time_interval, time_interval)
        start_index = np.searchsorted(time_seconds, start_time)
        time_seconds = time_seconds[start_index:]

        smoothed_mean = gaussian_filter1d(mean_trace, sigma=smoothing_sigma)[start_index:]
        smoothed_sem = gaussian_filter1d(sem_trace, sigma=smoothing_sigma)[start_index:]
        ci_trace = 1.96 * smoothed_sem

        plt.plot(time_seconds, smoothed_mean, label=label_map.get(stimulus, stimulus),
                  color=color_map.get(stimulus, "gray"), linewidth=2)
        plt.fill_between(time_seconds, smoothed_mean - smoothed_sem, smoothed_mean + smoothed_sem,
                          alpha=0.2, color=color_map.get(stimulus, "gray"))
        plt.fill_between(time_seconds, smoothed_mean - ci_trace, smoothed_mean + ci_trace,
                          alpha=0.1, color=color_map.get(stimulus, "gray"))

    ax = plt.gca()

    gradient = np.ones((2, 256, 4))
    gradient[:, :, :3] *= 0.5
    gradient[:, :, 3] = np.concatenate([np.full(128, 0.3), np.linspace(0.3, 0, 128)])
    extent = [gradient_extent[0], gradient_extent[1], ax.get_ylim()[0], ax.get_ylim()[1]]
    plt.imshow(gradient, extent=extent, origin="lower", aspect="auto", zorder=-1)

    y_max = ax.get_ylim()[1]
    plt.hlines(y=y_max, xmin=gradient_extent[0], xmax=gradient_extent[0] + 10, colors="black", linewidth=5)

    plt.legend(loc="upper left", bbox_to_anchor=(0.9, 1.2), fontsize=20, frameon=False)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)

    scale_bar_y = ax.get_ylim()[0] + 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    scale_bar_height = 0.25 * ax.get_ylim()[1]
    rounded_scale_bar_height = round(scale_bar_height * 20) / 20

    plt.plot(scale_bar_x, [scale_bar_y, scale_bar_y], "k", linewidth=3)
    plt.text(scale_bar_x[0] + 10, scale_bar_y - 0.06, "20 s", fontsize=24, ha="center", va="top")
    plt.plot([scale_bar_x[1], scale_bar_x[1]], [scale_bar_y, scale_bar_y + scale_bar_height], "k", linewidth=3)
    plt.text(scale_bar_x[1] + 2, scale_bar_y + rounded_scale_bar_height / 2,
              f"{rounded_scale_bar_height:.2f} SD", fontsize=24, ha="left", va="center")

    plt.title(" ", ha="right", fontsize=24, pad=20)
    plt.tight_layout()
    return fig


def compute_group_average(
    all_mean_traces: dict[str, np.ndarray],
    all_sem_traces: dict[str, np.ndarray],
    stimuli_list,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and SEM averaged across a group of stimuli.

    Source: cell 11's `compute_group_average`.
    """
    traces = [np.array(all_mean_traces[stim]) for stim in stimuli_list if stim in all_mean_traces]
    sems = [np.array(all_sem_traces[stim]) for stim in stimuli_list if stim in all_sem_traces]
    mean_trace = np.nanmean(np.stack(traces), axis=0)
    sem_trace = np.nanstd(np.stack(sems), axis=0) / np.sqrt(len(traces))
    return mean_trace, sem_trace


def plot_valence_comparison(
    mean_positive: np.ndarray, sem_positive: np.ndarray,
    mean_negative: np.ndarray, sem_negative: np.ndarray,
    start_time: float = START_TIME,
    smoothing_sigma: float = SMOOTHING_SIGMA,
    sampling_frequency: float = SAMPLING_FREQUENCY,
    gradient_extent: tuple[float, float] = (53, 73),
    scale_bar_x: tuple[float, float] = (90, 110),
    color_positive: str = "green",
    color_negative: str = "magenta",
) -> plt.Figure:
    """Positive-vs-negative valence-stimuli average trace comparison.

    Source: cell 11's plotting section.
    """
    time_interval = 1 / sampling_frequency
    trace_length = len(mean_positive)
    time_seconds = np.arange(0, trace_length * time_interval, time_interval)
    start_index = np.searchsorted(time_seconds, start_time)
    time_seconds = time_seconds[start_index:]

    mean_positive = gaussian_filter1d(mean_positive, sigma=smoothing_sigma)[start_index:]
    sem_positive = gaussian_filter1d(sem_positive, sigma=smoothing_sigma)[start_index:]
    mean_negative = gaussian_filter1d(mean_negative, sigma=smoothing_sigma)[start_index:]
    sem_negative = gaussian_filter1d(sem_negative, sigma=smoothing_sigma)[start_index:]

    fig = plt.figure(figsize=(7, 5))
    plt.plot(time_seconds, mean_positive, color=color_positive, linewidth=2, label="Positive Stimuli")
    plt.fill_between(time_seconds, mean_positive - sem_positive, mean_positive + sem_positive,
                      color=color_positive, alpha=0.2)
    plt.plot(time_seconds, mean_negative, color=color_negative, linewidth=2, label="Negative Stimuli")
    plt.fill_between(time_seconds, mean_negative - sem_negative, mean_negative + sem_negative,
                      color=color_negative, alpha=0.2)

    ax = plt.gca()
    gradient = np.ones((2, 256, 4))
    gradient[:, :, :3] *= 0.5
    gradient[:, :, 3] = np.concatenate([np.full(128, 0.3), np.linspace(0.3, 0, 128)])
    extent = [gradient_extent[0], gradient_extent[1], ax.get_ylim()[0], ax.get_ylim()[1]]
    plt.imshow(gradient, extent=extent, origin="lower", aspect="auto", zorder=-1)

    y_max = ax.get_ylim()[1]
    plt.hlines(y=y_max, xmin=gradient_extent[0], xmax=gradient_extent[0] + 10, colors="black", linewidth=4)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)

    scale_bar_y = ax.get_ylim()[0] + 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    scale_bar_height = 0.25 * ax.get_ylim()[1]
    rounded_scale_bar_height = round(scale_bar_height * 20) / 20

    plt.plot(scale_bar_x, [scale_bar_y, scale_bar_y], "k", linewidth=3)
    plt.text(scale_bar_x[0] + 10, scale_bar_y - 0.02, "20 s", fontsize=24, ha="center", va="top")
    plt.plot([scale_bar_x[1], scale_bar_x[1]], [scale_bar_y, scale_bar_y + rounded_scale_bar_height], "k", linewidth=3)
    plt.text(scale_bar_x[1] + 2, scale_bar_y + rounded_scale_bar_height / 2,
              f"{rounded_scale_bar_height:.2f} SD", fontsize=24, ha="left", va="center")

    plt.title("", ha="right", fontsize=20, pad=20)
    plt.tight_layout()
    return fig
