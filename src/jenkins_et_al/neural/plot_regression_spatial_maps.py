"""Brain-projection spatial maps of valence-correlated neurons, colored by
correlation strength (continuous colormap, split into one colormap per
valence sign) -- the simplest consumer of the same positive/negative
valence-correlation CSVs as `functional_clustering_valence_neurons.py` (#11).
Also an optional second variant restricted to neurons whose behaviour-vigor
correlation is *also* significant.

Ported from notebooks/neural/plot_regression_spatial_maps.ipynb; logic
unchanged, split into loading/processing/plotting functions per the /tdd
REFACTOR checklist. Cell 12 (the notebook's last cell) is empty -- not
reproduced.

**Not unified with the categorical brain-projection pattern** (`plotting.
plot_neuron_scatter_on_brain`'s categorical siblings,
`functional_clustering_concentration_response.plot_response_conditions_2view`
and `functional_clustering_valence_neurons.plot_brain_spatial`), despite
both of those modules' docstrings earmarking this notebook as their third
site to unify. Checked directly: this notebook's own `scatter_plots` is a
*different*, fourth pattern -- a continuous colormap (by correlation
strength), but a different colormap for each of the two valence groups
(green for positive, magenta/pink for negative), not one flat color per
category like the other two. It also isn't the general single-continuous-
colormap `plot_neuron_scatter_on_brain` either, since that function takes
only one colormap for its whole scatter and has no per-subgroup split.
Unifying would mean changing `plot_neuron_scatter_on_brain`'s signature (to
add a group/colormap-per-group option) or calling it twice onto the same
axes (which it doesn't support) -- both out of scope here, so this pattern
stays its own function, correcting those two docstrings' assumption.

**Confirmed correct, checked against #11's fixed bug**: `functional_clustering_valence_neurons.plot_brain_spatial`
had a real sagittal-panel axis-swap bug (`coords[:,0]/coords[:,1]` needed to
be `coords[:,1]/coords[:,0]`). This notebook's `scatter_plots` already uses
the correct order for both panels (`(coords[:, 1], proj.shape[0] -
coords[:, 2])` for the dorsal/horizontal panel, `(coords[:, 1], coords[:, 0])`
for the sagittal panel, matching `plotting.plot_neuron_scatter_on_brain`'s
verified convention) -- not the same bug, no fix needed here.

**Confirmed discrepancy, inert on real data**: cell 9's own first comment
says assigning `valence` for a BHV-significant neuron that's significant for
*both* positive and negative correlation "will assign 1", but the actual
code executes the negative assignment (`valence = 0`) *after* the positive
one, so on any row significant for both, negative overwrites positive --
the comment and the code disagree. Confirmed via golden capture against the
real dataset: zero neurons are significant for both directions
(`n neurons significant for BOTH pos and neg: 0`), so this has no effect on
the frozen analysis's actual output -- reproduced bug-for-bug (negative wins
ties) and exposed as `tie_break` below rather than silently resolved,
per the user's request to make this an explicit rule instead of a silent
default.

**Parametrized at the user's request** (2026-08-11): `filter_significant`'s
`p_value_column`/`significance_alpha`/`threshold`/`use_significance` were
module-level `CONFIG` constants in the notebook (cell 1); `plot_valence_corr`/
`scatter_plots`'s `cmap_neg`/`cmap_pos`/`vmax` (colormaps accept any
matplotlib-recognized colormap, not just the notebook's own two custom
gradients), `point_size`/`point_alpha`, and the colorbar labels/fontsize were
all hardcoded in the notebook's `scatter_plots`/`plot_valence_corr`. All are
now parameters with the notebook's own values as defaults. `vmin` was
already effectively a parameter in the notebook (threaded through from the
call site's `THRESHOLD`, defaulting to `0` inside `plot_valence_corr` when
`None`) -- kept the same way here.

**Refactor-shape change**: `plot_valence_corr` no longer calls `plt.show()`
before returning `fig` -- a notebook-inline-display side effect, dropped
here consistent with every other ported plotting function in this project,
which all just return the figure and let the caller display/save it.
"""
from __future__ import annotations

from ast import literal_eval
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PathCollection
from matplotlib.colors import Colormap, LinearSegmentedColormap
from skimage import io

#: Source: cell 2. Any matplotlib-recognized colormap (name or `Colormap`)
#: can be passed instead via `plot_valence_corr`'s/`scatter_plots`'s
#: `cmap_neg`/`cmap_pos` parameters -- these are just the notebook's own
#: default two-color gradients.
DEFAULT_CMAP_NEG = LinearSegmentedColormap.from_list("neg_cmap", ["pink", "magenta"])
DEFAULT_CMAP_POS = LinearSegmentedColormap.from_list("pos_cmap", ["lightgreen", "darkgreen"])

#: Source: cell 1.
P_VALUE_COLUMN = "p_fdr"
SIGNIFICANCE_ALPHA = 0.05


# ---------------------------------------------------------------------------
# Loading (source: cell 3)
# ---------------------------------------------------------------------------

def load_valence_data(pos_path: str | Path, neg_path: str | Path) -> pd.DataFrame:
    """Load positive/negative valence-correlation CSVs into one labeled, coordinate-parsed table.

    Adds a `valence` column (1 = positive, 0 = negative), drops
    `area == "not_assigned"` rows, and parses `coords` from its serialized
    string form via `ast.literal_eval`.
    """
    pos = pd.read_csv(pos_path)
    neg = pd.read_csv(neg_path)

    pos = pos.copy()
    neg = neg.copy()
    pos["valence"] = 1
    neg["valence"] = 0

    valence = pd.concat([pos, neg], ignore_index=True)
    valence = valence[valence["area"] != "not_assigned"].copy()
    valence["coords"] = valence["coords"].apply(literal_eval)

    return valence


def load_ref_brain(ref_brain_path: str | Path) -> np.ndarray:
    """Load the reference brain volume."""
    return io.imread(ref_brain_path)


# ---------------------------------------------------------------------------
# Processing: significance/threshold filtering (source: cell 4)
# ---------------------------------------------------------------------------

def filter_significant(
    valence_df: pd.DataFrame,
    p_value_column: str = P_VALUE_COLUMN,
    significance_alpha: float = SIGNIFICANCE_ALPHA,
    threshold: float | None = None,
    use_significance: bool = True,
) -> pd.DataFrame:
    """Restrict to significant (`p_value_column < significance_alpha`) and, optionally, `|correlation| >= threshold` rows.

    Raises `ValueError` if `use_significance` is `True` and `p_value_column`
    isn't present. Source: cell 4.
    """
    plot_df = valence_df.copy()

    if use_significance:
        if p_value_column not in plot_df.columns:
            raise ValueError(
                f"Expected a '{p_value_column}' column for significance filtering. "
                f"Available columns: {list(plot_df.columns)}"
            )
        plot_df = plot_df[plot_df[p_value_column] < significance_alpha].copy()

    if threshold is not None:
        plot_df = plot_df[np.abs(plot_df["correlation"]) >= threshold].copy()

    return plot_df


# ---------------------------------------------------------------------------
# Processing: optional behaviour-vigor-correlated subset (source: cells 7-10)
# ---------------------------------------------------------------------------

def load_bhv_merged_data(bhv_path: str | Path, pos_path: str | Path, neg_path: str | Path) -> pd.DataFrame:
    """Merge behaviour-vigor correlations with each neuron's positive/negative valence correlation.

    Renamed, left-joined on `(fish_id, neuron_id)`: `bhv_correlation`/
    `bhv_pfdr` (from `bhv_path`), `pos_correlation`/`pos_pfdr` (from
    `pos_path`), `neg_correlation`/`neg_pfdr` (from `neg_path`). `coords` is
    parsed via `literal_eval`; `area == "not_assigned"` rows are dropped if
    an `area` column is present.

    Source: cell 7.
    """
    bhv_data = pd.read_csv(bhv_path).rename(columns={"correlation": "bhv_correlation", "p_fdr": "bhv_pfdr"})
    pos_data = pd.read_csv(pos_path)
    neg_data = pd.read_csv(neg_path)
    pos_data["valence"] = 1
    neg_data["valence"] = 0
    pos_data = pos_data.rename(columns={"correlation": "pos_correlation", "p_fdr": "pos_pfdr"})
    neg_data = neg_data.rename(columns={"correlation": "neg_correlation", "p_fdr": "neg_pfdr"})

    merged = bhv_data.merge(pos_data[["fish_id", "neuron_id", "pos_correlation", "pos_pfdr"]], on=["fish_id", "neuron_id"], how="left")
    merged = merged.merge(neg_data[["fish_id", "neuron_id", "neg_correlation", "neg_pfdr"]], on=["fish_id", "neuron_id"], how="left")
    merged["coords"] = merged["coords"].apply(literal_eval)

    if "area" in merged.columns:
        merged = merged[merged["area"] != "not_assigned"].copy()

    return merged


def select_significant_bhv_neurons(merged: pd.DataFrame, significance_alpha: float = SIGNIFICANCE_ALPHA) -> pd.DataFrame:
    """Neurons significant for behaviour vigor AND for at least one of positive/negative valence.

    Source: cell 8.
    """
    return merged[
        (merged["bhv_pfdr"] < significance_alpha)
        & ((merged["pos_pfdr"] < significance_alpha) | (merged["neg_pfdr"] < significance_alpha))
    ]


def assign_bhv_valence(
    significant_neurons: pd.DataFrame, significance_alpha: float = SIGNIFICANCE_ALPHA, tie_break: str = "negative",
) -> pd.DataFrame:
    """Label each neuron `valence=1` (positive-significant) or `valence=0` (negative-significant).

    `tie_break` controls which label wins for a neuron significant in *both*
    directions: `"negative"` (default, matches the notebook's actual
    executed behavior -- see module docstring's "Confirmed discrepancy" note,
    inert on the real dataset) or `"positive"`.

    Source: cell 9.
    """
    significant_neurons = significant_neurons.copy()
    significant_neurons["valence"] = None

    first, second = (1, 0) if tie_break == "negative" else (0, 1)
    first_col, second_col = ("pos_pfdr", "neg_pfdr") if tie_break == "negative" else ("neg_pfdr", "pos_pfdr")

    significant_neurons.loc[significant_neurons[first_col] < significance_alpha, "valence"] = first
    significant_neurons.loc[significant_neurons[second_col] < significance_alpha, "valence"] = second

    return significant_neurons


def select_bhv_correlation(significant_neurons: pd.DataFrame) -> pd.DataFrame:
    """Set `correlation` to `pos_correlation` where `valence == 1`, else `neg_correlation`.

    Source: cell 10.
    """
    correlation_data = significant_neurons.copy()
    correlation_data["correlation"] = np.where(
        correlation_data["valence"] == 1, correlation_data["pos_correlation"], correlation_data["neg_correlation"],
    )
    return correlation_data


# ---------------------------------------------------------------------------
# Plotting (source: cell 2)
# ---------------------------------------------------------------------------

def scatter_plots(
    axs: list[plt.Axes],
    ref_brain: np.ndarray,
    coords: np.ndarray,
    regressor_corr: np.ndarray,
    valence: np.ndarray,
    vmin: float | None = None,
    vmax: float = 0.7,
    cmap_neg: str | Colormap = DEFAULT_CMAP_NEG,
    cmap_pos: str | Colormap = DEFAULT_CMAP_POS,
    point_size: float = 10,
    point_alpha: float = 0.9,
) -> list[PathCollection]:
    """Dorsal (`axs[0]`) + sagittal (`axs[1]`) neuron scatter, colored by `regressor_corr` with `cmap_pos`/`cmap_neg` split by `valence`.

    Source: cell 2 (`scatter_plots`).
    """
    dimensions = [(0, "upper"), (2, "lower")]
    coord_indices = [(2, 1), (1, 0)]
    scatter_list = []

    for ax, (dim, origin), (x_idx, y_idx) in zip(axs, dimensions, coord_indices):
        proj = np.nanmean(ref_brain, axis=dim)

        if dim == 0:
            proj = np.rot90(proj, k=1)
            ax.imshow(proj, cmap="gray_r", alpha=0.3, origin="lower")
            plot_x = coords[:, y_idx]
            plot_y = proj.shape[0] - coords[:, x_idx]
        else:
            ax.imshow(proj, cmap="gray_r", alpha=0.3, origin=origin)
            plot_x = coords[:, x_idx]
            plot_y = coords[:, y_idx]

        for val in [0, 1]:
            cmap = cmap_pos if val == 1 else cmap_neg
            mask = valence == val
            scatter = ax.scatter(
                plot_x[mask], plot_y[mask], c=regressor_corr[mask], cmap=cmap,
                s=point_size, edgecolor="none", alpha=point_alpha, vmin=vmin, vmax=vmax,
            )
            scatter_list.append(scatter)

        ax.axis("off")
        ax.set_aspect("equal")

    return scatter_list


def plot_valence_corr(
    ref_brain: np.ndarray,
    correlation_data: pd.DataFrame,
    vmin: float | None = None,
    vmax: float = 0.7,
    cmap_neg: str | Colormap = DEFAULT_CMAP_NEG,
    cmap_pos: str | Colormap = DEFAULT_CMAP_POS,
    point_size: float = 10,
    point_alpha: float = 0.9,
    neg_colorbar_label: str = "Negative r value",
    pos_colorbar_label: str = "Positive r value",
    colorbar_label_fontsize: float = 20,
) -> plt.Figure:
    """Dorsal + sagittal brain-projection scatter of `correlation_data`, colored by correlation strength.

    `correlation_data` needs `coords`, `correlation`, and `valence` columns.
    `vmin` defaults to `0` when not given (matches the notebook's own
    default, which passed `THRESHOLD` -- usually `None` -- straight through).

    Source: cell 2 (`plot_valence_corr`).
    """
    fig = plt.figure(figsize=(6, 9))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
    gs.update(hspace=-0.6)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    coords = np.vstack(correlation_data["coords"])
    regressor_corr = correlation_data["correlation"].values
    valence = correlation_data["valence"].values

    vmin = 0 if vmin is None else vmin

    scatter_list = scatter_plots(
        [ax1, ax2], ref_brain, coords, regressor_corr, valence, vmin=vmin, vmax=vmax,
        cmap_neg=cmap_neg, cmap_pos=cmap_pos, point_size=point_size, point_alpha=point_alpha,
    )

    cbar_ax1 = fig.add_axes([1.08, 0.4, 0.02, 0.3])
    cbar_ax2 = fig.add_axes([0.88, 0.4, 0.02, 0.3])

    cbar1 = fig.colorbar(scatter_list[0], cax=cbar_ax1)
    cbar1.set_label(neg_colorbar_label, fontsize=colorbar_label_fontsize)
    cbar1.ax.tick_params(labelsize=colorbar_label_fontsize)

    cbar2 = fig.colorbar(scatter_list[1], cax=cbar_ax2)
    cbar2.set_label(pos_colorbar_label, fontsize=colorbar_label_fontsize)
    cbar2.ax.tick_params(labelsize=colorbar_label_fontsize)

    return fig
