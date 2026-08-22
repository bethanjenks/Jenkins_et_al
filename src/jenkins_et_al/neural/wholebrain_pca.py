"""Per-area population PCA trajectories.

Ported from `notebooks/neural/wholebrain_avg_traces_PCA_raster.ipynb`
cells 18-22. This is the replacement for the reshape bug found in the
(now-deleted) predecessor notebook, `wholebrain_population_analysis.ipynb`
-- confirmed genuinely fixed here: `perform_pca` builds its PCA input via
`np.hstack` over per-stimulus frame-window slices then transposes
(`.T`), which correctly produces one row per (stimulus, frame) timepoint
and one column per neuron -- not the old version's plain `.reshape()`,
which silently scrambled neuron/frame axes (see `PORTING_PLAN.md`'s note on
that notebook for the full account). Also new here, not present in the
predecessor: per-neuron z-scoring before PCA, a restricted response window
(frames 90:200, not the full trial), and `n_components=6`/`sigma=4`
(previously 10/2).

**Confirmed bug, fixed at the user's request**: cell 18 computes
`area_df = stim_fb[stim_fb['area']=='pallium']` (13453 neurons on real
data), but cell 19's `average_traces` groups by `stim_df` (=`stim_fb`, all
forebrain areas, 86155 neurons) -- `area_df` is never actually used, so the
saved notebook's PCA silently runs on the whole forebrain despite the
commented-out save paths (`..._pallium_PCs_2d_plots.svg`) indicating
pallium-only was intended. `prepare_pca_data` below takes the
already-area-filtered DataFrame directly, fixing this at the call site.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA

from jenkins_et_al.neural.wholebrain_traces import KEY_STIMULI, LABEL_MAPPING, STIMULUS_COLORS

N_COMPONENTS = 6
SMOOTHING_SIGMA = 4
RESPONSE_WINDOW = (90, 200)  # frame slice, source: cell 19's perform_pca


def prepare_pca_data(area_df: pd.DataFrame, stimuli=KEY_STIMULI) -> tuple[np.ndarray, np.ndarray]:
    """Build the (stimuli, neurons, frames) array for one area.

    `area_df` must already be filtered to the area of interest -- see
    module docstring for the confirmed area_df/stim_df mismatch this fixes.

    Source: cell 19 lines 1-.
    """
    average_traces = (
        area_df.groupby(["neuron_id", "stimulus"])["img_trace"]
        .apply(lambda traces: np.mean(np.stack(traces), axis=0))
        .reset_index()
    )

    unique_neurons = average_traces["neuron_id"].unique()
    num_neurons = len(unique_neurons)
    num_stimuli = len(stimuli)
    num_frames = len(average_traces["img_trace"].iloc[0])

    data_array = np.zeros((num_stimuli, num_neurons, num_frames))
    neuron_index_map = {neuron: idx for idx, neuron in enumerate(unique_neurons)}

    for i, stim in enumerate(stimuli):
        stim_data = average_traces[average_traces["stimulus"] == stim]
        for _, row in stim_data.iterrows():
            neuron_idx = neuron_index_map[row["neuron_id"]]
            data_array[i, neuron_idx, :] = row["img_trace"]

    return data_array, unique_neurons


def perform_pca(
    data_array: np.ndarray,
    n_components: int = N_COMPONENTS,
    sigma: float = SMOOTHING_SIGMA,
    response_window: tuple[int, int] = RESPONSE_WINDOW,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit PCA on z-scored, response-window-restricted traces (stacked
    across stimuli by `hstack`, then transposed so rows are (stimulus,
    frame) timepoints and columns are neurons), then smooth each PC's
    per-stimulus trajectory.

    Source: cell 19's `perform_pca`.

    Returns:
        (smoothed_pca_data, explained_variance, weights) -- shapes
        (n_stimuli, n_response_frames, n_components), (n_components,),
        (n_components, n_neurons).
    """
    num_stims = data_array.shape[0]
    r0, r1 = response_window
    data_2d = np.hstack([data_array[i, :, r0:r1] for i in range(num_stims)])

    mean = np.average(data_2d, axis=0)
    sd = np.std(data_2d, axis=0)
    z_data_2d = (data_2d - mean) / sd

    pca = PCA(n_components=n_components)
    pca_data_2d = pca.fit_transform(z_data_2d.T)
    explained_variance = pca.explained_variance_ratio_ * 100
    weights = pca.components_

    num_observations = pca_data_2d.shape[0] // num_stims
    smoothed_pca_data = np.array([
        gaussian_filter1d(pca_data_2d[i * num_observations:(i + 1) * num_observations, :n_components], sigma, axis=0)
        for i in range(num_stims)
    ])

    return smoothed_pca_data, explained_variance, weights


def plot_pca_2d_panels(
    smoothed_pca_data: np.ndarray,
    explained_variance: np.ndarray,
    stimuli=KEY_STIMULI,
    colors=STIMULUS_COLORS,
    label_map: dict[str, str] = LABEL_MAPPING,
    pc_pairs=((0, 1), (1, 2), (2, 3)),
) -> plt.Figure:
    """PC-pair trajectory panels, one subplot per pair.

    Source: cell 20.
    """
    fig, axes = plt.subplots(1, len(pc_pairs), figsize=(20, 6), sharey=False)

    for idx, (pc_x, pc_y) in enumerate(pc_pairs):
        ax = axes[idx]
        for i, stimulus in enumerate(stimuli):
            pc_x_data = smoothed_pca_data[i, :, pc_x]
            pc_y_data = smoothed_pca_data[i, :, pc_y]
            ax.plot(pc_x_data, pc_y_data, label=label_map.get(stimulus, stimulus) if idx == 0 else None,
                     color=colors[i], linewidth=6)

        ax.set_title("", fontsize=20)
        ax.set_xlabel(f"PC{pc_x + 1} ({explained_variance[pc_x]:.2f}% Var. expl.)", fontsize=28)
        ax.set_ylabel(f"PC{pc_y + 1}  ({explained_variance[pc_y]:.2f}% Var. expl.)", fontsize=28)
        ax.tick_params(axis="both", which="major", labelsize=26)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.legend(loc="center right", bbox_to_anchor=(1.07, 0.8), fontsize=18)
    fig.tight_layout()
    return fig


def plot_pca_timecourse(
    smoothed_pca_data: np.ndarray,
    stimuli=KEY_STIMULI,
    colors=STIMULUS_COLORS,
) -> list[plt.Figure]:
    """Each PC plotted as a function of time, one figure per PC.

    Source: cell 21.
    """
    num_stimuli, time_points, num_pcs = smoothed_pca_data.shape
    figures = []

    for pc in range(num_pcs):
        fig = plt.figure(figsize=(6, 3))
        pc_all_stimuli = smoothed_pca_data[:, :, pc]
        for i in range(num_stimuli):
            plt.plot(range(time_points), pc_all_stimuli[i, :], label=stimuli[i], color=colors[i])

        plt.title(f"Principal Component {pc + 1} as a Function of Time for All Stimuli")
        plt.xlabel("Time (frames)")
        plt.ylabel(f"PC{pc + 1} Value")
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax = plt.gca()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        figures.append(fig)

    return figures


def plot_pca_3d(
    smoothed_pca_data: np.ndarray,
    explained_variance: np.ndarray,
    stimuli=KEY_STIMULI,
    colors=STIMULUS_COLORS,
    label_map: dict[str, str] = LABEL_MAPPING,
) -> plt.Figure:
    """3D PC1-PC2-PC3 trajectory plot.

    Source: cell 22.
    """
    num_stimuli = smoothed_pca_data.shape[0]
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for i in range(num_stimuli):
        pc1 = smoothed_pca_data[i, :, 0]
        pc2 = smoothed_pca_data[i, :, 1]
        pc3 = smoothed_pca_data[i, :, 2]
        ax.plot(pc1, pc2, pc3, label=label_map.get(stimuli[i], stimuli[i]), color=colors[i], linewidth=2.5)

    ax.set_xlabel(f"PC1 ({explained_variance[0]:.2f}% variance)", fontsize=16)
    ax.set_ylabel(f"PC2 ({explained_variance[1]:.2f}% variance)", fontsize=16)
    ax.set_zlabel(f"PC3 ({explained_variance[2]:.2f}% variance)", fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.legend(loc="center left", bbox_to_anchor=(1.1, 0.5), fontsize=16)
    return fig
