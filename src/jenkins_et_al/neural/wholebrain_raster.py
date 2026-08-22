"""Randomly sampled forebrain neuron raster plot.

Ported from `notebooks/neural/wholebrain_avg_traces_PCA_raster.ipynb`
cells 13-16; logic unchanged. Operates on `stim_fb` only (`stim_df =
stim_fb` in the notebook's own cell 13) -- see `wholebrain_traces.py`'s
module docstring for why hindbrain data isn't part of this notebook's
actual analysis despite being loaded.

**Confirmed dead code, not reproduced**: cell 15 computes
`norm = mcolors.Normalize(vmin=heatmap_data.min() + 1, ...)` then
immediately overwrites it with a second `norm = mcolors.Normalize(vmin=-5,
...)` on the next line -- only the second value is ever used. Not ported.

**Confirmed threshold filters nothing at this scale, not a new finding**:
`RASTER_RESPONSE_THRESHOLD = 0.5` passes 100% of forebrain neurons
(86155/86155) on real data -- consistent with the same finding already
documented for the (now-deleted) predecessor notebook at whole-brain scale.
"""
from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jenkins_et_al.neural.wholebrain_traces import LABEL_MAPPING, SAMPLING_FREQUENCY

#: Source: cell 15.
RASTER_STIMULUS_ORDER: tuple[str, ...] = ("fex_1", "kw", "pro_2.5mm", "ade", "cad_2.5mm", "ph4.5", "qui_2.5mm")
RASTER_START_FRAME = 60
RASTER_RESPONSE_THRESHOLD = 0.5
RASTER_N_NEURONS = 100
RASTER_RANDOM_STATE = 42


def prepare_raster_data(
    stim_df: pd.DataFrame,
    stimulus_order=RASTER_STIMULUS_ORDER,
    start_frame: int = RASTER_START_FRAME,
    response_threshold: float = RASTER_RESPONSE_THRESHOLD,
    n_neurons: int = RASTER_N_NEURONS,
    random_state: int = RASTER_RANDOM_STATE,
) -> tuple[np.ndarray, list[tuple[int, str]]]:
    """Build the (n_neurons, total_frames) heatmap matrix and stimulus label
    positions.

    Source: cell 15 lines 1-.
    """
    average_traces = (
        stim_df.groupby(["neuron_id", "stimulus"])["img_trace"]
        .apply(lambda traces: np.mean(np.stack(traces), axis=0))
        .reset_index()
    )
    average_traces["stimulus"] = pd.Categorical(
        average_traces["stimulus"], categories=stimulus_order, ordered=True
    )
    average_traces = average_traces.sort_values(by=["neuron_id", "stimulus"])

    concatenated_averages = (
        average_traces.groupby("neuron_id")
        .apply(lambda df: np.concatenate([trace[start_frame:] for trace in df.sort_values("stimulus")["img_trace"]]))
        .reset_index(name="img_trace")
    )
    concatenated_averages["response_strength"] = concatenated_averages["img_trace"].apply(np.max)

    filtered_neurons = concatenated_averages[concatenated_averages["response_strength"] > response_threshold]
    if len(filtered_neurons) >= n_neurons:
        random_sample = filtered_neurons.sample(n=n_neurons, random_state=random_state)
    else:
        random_sample = filtered_neurons

    heatmap_data = np.vstack(random_sample["img_trace"].to_numpy())

    stimulus_positions: list[tuple[int, str]] = []
    current_position = 0
    for stim in stimulus_order:
        stim_traces = average_traces[average_traces["stimulus"] == stim]["img_trace"]
        if len(stim_traces) == 0:
            continue
        stim_length = len(stim_traces.iloc[0][start_frame:])
        label_position = current_position + stim_length // 2
        stimulus_positions.append((label_position, stim))
        current_position += stim_length

    return heatmap_data, stimulus_positions


def plot_raster_heatmap(
    heatmap_data: np.ndarray,
    stimulus_positions: list[tuple[int, str]],
    label_map: dict[str, str] = LABEL_MAPPING,
    vmin: float = -5,
    vmax: float | None = None,
    cmap: str = "rainbow",
    frame_marker_interval: int = 240,
    sampling_frequency: float = SAMPLING_FREQUENCY,
) -> plt.Figure:
    """Raster heatmap of sampled neuron responses across concatenated
    stimulus traces.

    Source: cell 15 lines -end.
    """
    if vmax is None:
        vmax = heatmap_data.max()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=(8, 6))
    plt.imshow(heatmap_data, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")

    for pos, stim in stimulus_positions:
        plt.text(pos, heatmap_data.shape[0] - 103, label_map.get(stim, stim), fontsize=20, ha="center", color="black")

    for x in range(0, heatmap_data.shape[1], frame_marker_interval):
        plt.axvline(x=x, color="white", linestyle="--", linewidth=2)

    time_interval = 1 / sampling_frequency
    total_duration = heatmap_data.shape[1] * time_interval
    x_ticks_in_seconds = np.arange(0, total_duration, 10)
    x_ticks_in_frames = (x_ticks_in_seconds / time_interval).astype(int)

    plt.xticks(x_ticks_in_frames)
    plt.gca().xaxis.set_ticks_position("top")
    plt.gca().set_xticklabels([])
    plt.gca().tick_params(axis="x", which="both", length=3, width=0.5)
    plt.yticks([0, 49, 99], ["1", "50", "100"], fontsize=26)
    plt.ylabel("Forebrain Neurons", fontsize=28)

    cbar = plt.colorbar(shrink=0.5)
    cbar.set_label("Z-score", fontsize=28)
    cbar.ax.tick_params(labelsize=28)

    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    return fig
