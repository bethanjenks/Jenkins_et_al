"""Position heatmaps and single-fish trajectory plots for free-swim data.

Ported from notebooks/behaviour_free_swim/fish_position_plots.ipynb; logic is
unchanged from the notebook, only moved into a module, split into
loading / compute / plot functions, and given a docstring/type-hint pass per
the /tdd REFACTOR checklist. The notebook has no cell ids (older nbformat),
so docstrings trace back to cell index instead.

Loading and the movement-filter reuse `jenkins_et_al.behaviour.free_swim`'s
already-ported/verified `load_dataframes_from_folder`,
`is_fish_significantly_moving`, and `total_distance_travelled` rather than
redefining them -- the notebook's own copies of those three are identical in
behaviour (one unused dead-code line in `is_fish_significantly_moving` aside).

The notebook produces two figures from a single stimulus's data
(`adenosine_2.5mM`, loaded from its `stim`/`base` subfolders -- not
`stim`/`control`, unlike `free_swim.load_stimulus_data`):

- a stim-vs-control difference position heatmap, with marginal difference
  histograms (`compute_difference_heatmap` + `plot_difference_heatmap`,
  cells 6-7)
- a single fish's trajectory coloured by time (`prepare_fish_trajectory` +
  `plot_fish_trajectory`, cell 9)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure
from scipy.ndimage import gaussian_filter

from jenkins_et_al.behaviour.free_swim import (
    is_fish_significantly_moving,
    load_dataframes_from_folder,
)

# ---------------------------------------------------------------------------
# Constants (source: cells 1, 3, 6, 9)
# ---------------------------------------------------------------------------

#: Movement/tracking filter thresholds used by this notebook; identical to
#: `free_swim`'s defaults (`MOVEMENT_THRESHOLD_CM`, `TRACKING_THRESHOLD_PCT`).
MOVEMENT_THRESHOLD_CM = 60.0
TRACKING_THRESHOLD_PCT = 90.0

#: Default heatmap parameters (source: cell 6, `create_difference_heatmap`).
DEFAULT_ARENA_SIZE = (2, 1)
DEFAULT_HEATMAP_BINS = (11, 6)
DEFAULT_HEATMAP_SIGMA = 1

#: Default trajectory-plot axis limits (source: cell 9).
DEFAULT_TRAJECTORY_XLIM = (0.1, 14)
DEFAULT_TRAJECTORY_YLIM = (0.6, 6.8)


# ---------------------------------------------------------------------------
# Loading (source: cells 1, 3)
# ---------------------------------------------------------------------------

def load_dataframes_from_subfolders(parent_dir: Path) -> list[pd.DataFrame]:
    """Load every per-fish tracking CSV one level under `parent_dir`'s subfolders."""
    dataframes = []
    for subfolder in parent_dir.iterdir():
        if subfolder.is_dir():
            dataframes.extend(load_dataframes_from_folder(subfolder))
    return dataframes


def filter_significantly_moving(
    dfs: list[pd.DataFrame],
    movement_threshold: float = MOVEMENT_THRESHOLD_CM,
    tracking_threshold: float = TRACKING_THRESHOLD_PCT,
) -> list[pd.DataFrame]:
    """Keep only fish passing `is_fish_significantly_moving`."""
    return [
        df for df in dfs
        if is_fish_significantly_moving(df, movement_threshold, tracking_threshold)
    ]


# ---------------------------------------------------------------------------
# Difference heatmap (source: cell 6)
# ---------------------------------------------------------------------------

def get_all_x_y(list_dfs: list[pd.DataFrame]) -> tuple[list[float], list[float]]:
    """Pool x/y over each df's second half, tracked (`mask` == True) frames only."""
    all_x = []
    all_y = []

    for df in list_dfs:
        half_len = len(df) // 2
        filtered_df = df[half_len:]
        all_x.extend(filtered_df.loc[filtered_df["mask"], "x"])
        all_y.extend(filtered_df.loc[filtered_df["mask"], "y"])

    return all_x, all_y


def normalize_to_range(
    values: list[float], new_min: float = -1, new_max: float = 1
) -> list[float]:
    """Min-max rescale `values` into `[new_min, new_max]`."""
    old_min = min(values)
    old_max = max(values)

    normalized_0_1 = [(x - old_min) / (old_max - old_min) for x in values]
    return [new_min + (x * (new_max - new_min)) for x in normalized_0_1]


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """Rescale `heatmap` so its values sum to 1."""
    return heatmap / np.sum(heatmap)


@dataclass
class DifferenceHeatmapData:
    """Data feeding `plot_difference_heatmap` (source: cell 6)."""

    stim_heatmap: np.ndarray
    control_heatmap: np.ndarray
    difference_heatmap: np.ndarray
    xedges: np.ndarray
    yedges: np.ndarray
    diff_x_hist: np.ndarray
    diff_y_hist: np.ndarray


def compute_difference_heatmap(
    stim_df_list: list[pd.DataFrame],
    control_df_list: list[pd.DataFrame],
    arena_size: tuple[float, float] = DEFAULT_ARENA_SIZE,
    bins: tuple[int, int] = DEFAULT_HEATMAP_BINS,
    sigma: float = DEFAULT_HEATMAP_SIGMA,
) -> DifferenceHeatmapData:
    """Gaussian-smoothed stim-vs-control position-density difference, plus
    marginal difference histograms, over a list of per-fish dataframes each.
    """
    stim_x_positions, stim_y_positions = get_all_x_y(stim_df_list)
    stim_normalised_x = normalize_to_range(stim_x_positions, new_min=0, new_max=arena_size[0])
    stim_normalised_y = normalize_to_range(stim_y_positions, new_min=0, new_max=arena_size[1])

    stim_heatmap, xedges, yedges = np.histogram2d(
        stim_normalised_x, stim_normalised_y, bins=bins,
        range=[[0, arena_size[0]], [0, arena_size[1]]],
    )
    stim_heatmap = gaussian_filter(normalize_heatmap(stim_heatmap), sigma=sigma)

    control_x_positions, control_y_positions = get_all_x_y(control_df_list)
    control_normalised_x = normalize_to_range(control_x_positions, new_min=0, new_max=arena_size[0])
    control_normalised_y = normalize_to_range(control_y_positions, new_min=0, new_max=arena_size[1])

    control_heatmap, _, _ = np.histogram2d(
        control_normalised_x, control_normalised_y, bins=bins,
        range=[[0, arena_size[0]], [0, arena_size[1]]],
    )
    control_heatmap = gaussian_filter(normalize_heatmap(control_heatmap), sigma=sigma)

    difference_heatmap = stim_heatmap - control_heatmap

    stim_x_hist, _ = np.histogram(stim_normalised_x, bins=xedges, range=[0, arena_size[0]])
    control_x_hist, _ = np.histogram(control_normalised_x, bins=xedges, range=[0, arena_size[0]])
    diff_x_hist = stim_x_hist - control_x_hist

    stim_y_hist, _ = np.histogram(stim_normalised_y, bins=yedges, range=[0, arena_size[1]])
    control_y_hist, _ = np.histogram(control_normalised_y, bins=yedges, range=[0, arena_size[1]])
    diff_y_hist = stim_y_hist - control_y_hist

    return DifferenceHeatmapData(
        stim_heatmap=stim_heatmap,
        control_heatmap=control_heatmap,
        difference_heatmap=difference_heatmap,
        xedges=xedges,
        yedges=yedges,
        diff_x_hist=diff_x_hist,
        diff_y_hist=diff_y_hist,
    )


def plot_difference_heatmap(
    title: str,
    data: DifferenceHeatmapData,
    arena_size: tuple[float, float] = DEFAULT_ARENA_SIZE,
) -> Figure:
    """Render `data` as a difference heatmap with marginal difference histograms."""
    fig = plt.figure(figsize=(6, 3))
    ax_main = fig.add_subplot(111)
    ax_main.set_title(title, fontsize=22, weight="bold", x=0.1, y=1.15)

    norm = TwoSlopeNorm(vmin=-0.02, vcenter=0, vmax=0.02)
    cax = ax_main.imshow(
        data.difference_heatmap.T, extent=[0, arena_size[0], 0, arena_size[1]],
        origin="lower", aspect="auto", cmap="coolwarm", norm=norm, interpolation="nearest",
    )
    ax_main.set_xlabel(r"$X_{\mathrm{norm}}$", fontsize=26)
    ax_main.set_ylabel(r"$Y_{\mathrm{norm}}$", fontsize=26)
    ax_main.tick_params(axis="both", labelsize=26)

    bbox = ax_main.get_position()
    ax_x_hist = fig.add_axes([bbox.x0, bbox.y1 + 0.01, bbox.width, 0.09])
    ax_y_hist = fig.add_axes([bbox.x1 + 0.01, bbox.y0, 0.06, bbox.height])

    ax_x_hist.margins(x=0)
    ax_y_hist.margins(y=0)

    ax_x_hist.axis("off")
    ax_y_hist.axis("off")

    ax_x_hist.bar(range(len(data.diff_x_hist)), data.diff_x_hist, width=1, color="darkblue", alpha=0.6)
    ax_y_hist.barh(range(len(data.diff_y_hist)), data.diff_y_hist, height=1, color="darkblue", alpha=0.6)

    ax_colorbar = fig.add_axes([bbox.x1 + 0.1, bbox.y0, 0.02, bbox.height])
    colorbar = plt.colorbar(cax, cax=ax_colorbar)
    colorbar.ax.tick_params(labelsize=26)
    colorbar.set_label(r"Δ Density", fontsize=26, labelpad=10)
    colorbar.set_ticks([-0.02, 0, 0.02])

    return fig


def create_difference_heatmap(
    title: str,
    stim_df_list: list[pd.DataFrame],
    control_df_list: list[pd.DataFrame],
    arena_size: tuple[float, float] = DEFAULT_ARENA_SIZE,
    bins: tuple[int, int] = DEFAULT_HEATMAP_BINS,
    sigma: float = DEFAULT_HEATMAP_SIGMA,
) -> Figure:
    """Compute and plot the stim-vs-control difference heatmap in one call."""
    data = compute_difference_heatmap(stim_df_list, control_df_list, arena_size, bins, sigma)
    return plot_difference_heatmap(title, data, arena_size)


# ---------------------------------------------------------------------------
# Single-fish trajectory (source: cell 9)
# ---------------------------------------------------------------------------

def prepare_fish_trajectory(fish_df: pd.DataFrame) -> pd.DataFrame:
    """Interpolate a fish's x/y/time and add a `time_normalized` column.

    Returns a copy; unlike the notebook cell, does not mutate `fish_df`.
    """
    fish = fish_df.copy()

    fish["x"] = fish["x"].replace([np.inf, -np.inf], np.nan)
    fish["y"] = fish["y"].replace([np.inf, -np.inf], np.nan)
    fish["time"] = fish["time"].replace([np.inf, -np.inf], np.nan)

    fish["x"] = fish["x"].interpolate(method="linear", limit_direction="both")
    fish["y"] = fish["y"].interpolate(method="linear", limit_direction="both")
    fish["time"] = fish["time"].interpolate(method="linear", limit_direction="both")

    fish["time_normalized"] = (fish["time"] - fish["time"].min()) / (
        fish["time"].max() - fish["time"].min()
    )

    return fish


def plot_fish_trajectory(
    fish_df: pd.DataFrame,
    xlim: tuple[float, float] = DEFAULT_TRAJECTORY_XLIM,
    ylim: tuple[float, float] = DEFAULT_TRAJECTORY_YLIM,
) -> Figure:
    """Plot a single fish's trajectory, coloured by normalized time.

    `fish_df` must already have a `time_normalized` column, e.g. from
    `prepare_fish_trajectory`.
    """
    cmap = mcolors.LinearSegmentedColormap.from_list("pink_magenta", ["pink", "deeppink"])

    fig, ax = plt.subplots(figsize=(10, 5))

    points = np.array([fish_df["x"], fish_df["y"]]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    time_normalized = fish_df["time_normalized"]
    norm = plt.Normalize(time_normalized.min(), time_normalized.max())
    lc = mcoll.LineCollection(segments, cmap=cmap, norm=norm, linewidth=2.5, alpha=0.9)
    lc.set_array(time_normalized)

    ax.add_collection(lc)
    ax.scatter(
        fish_df["x"], fish_df["y"], c=time_normalized, cmap=cmap, s=1, alpha=0.8, edgecolors="none"
    )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_yticks([])
    ax.set_xticks([])

    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["right"].set_linewidth(2)
    ax.spines["top"].set_linewidth(2)

    ax.text(0.5, -0.1, "1 cm", ha="center", va="center", fontsize=20, color="black")
    ax.plot([0.1, 1.1], [0.3, 0.3], color="black", linewidth=3, clip_on=False)

    cbar = plt.colorbar(lc, ax=ax, shrink=0.9)
    cbar.set_label(r"Normalized Time", fontsize=28, labelpad=10)
    cbar.ax.tick_params(labelsize=24)
    cbar.set_ticks([0, 0.5, 1.0])

    fig.tight_layout()
    return fig
