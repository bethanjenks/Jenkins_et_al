"""General-purpose figure functions shared across neural notebooks.

Holds two patterns reused across multiple already-ported notebooks:

- The whole-brain neuron scatter projection: two stacked views
  (dorsal/horizontal on top, sagittal below) of a reference brain volume
  with neurons overlaid as a colormapped scatter. First ported from
  notebooks/neural/lifetime_sparseness.ipynb's `plot_sparseness_brain`
  (cell 27), which colored neurons by lifetime sparseness with a custom
  purple-white-orange colormap. The same horizontal+sagittal projection
  pattern (`np.rot90` dorsal view, `np.nanmean` sagittal view, coordinate
  transform `rotated_x = coords[:, 1]`, `rotated_y = height - coords[:, 2]`)
  recurs with a different colormap/value per notebook in
  notebooks/neural/build_brain_map.ipynb,
  notebooks/neural/functional_clustering_valence_neurons.ipynb, and
  notebooks/neural/plot_regression_spatial_maps.ipynb -- those are not yet
  ported/verified, so this function is deliberately generalized only along
  the axis lifetime_sparseness itself needs (continuous value + colormap),
  not unified with those other notebooks' categorical-label variants. Reuse
  this function when porting them, extending its parameters as needed, then
  re-verify lifetime_sparseness's own golden capture still matches.

- The per-brain-area boxplot-vs-a-reference-area: colored boxes per area,
  jittered per-fish points, significance stars vs. one reference area.
  Unified (2026-08-04, at the user's request) from two independently-ported,
  near-identical copies -- `lifetime_sparseness.plot_sparseness_boxplot` and
  `template_matching_classification.plot_classification_accuracy_boxplot` --
  after noticing the latter's forebrain-area call rendered visibly wider
  boxes than lifetime_sparseness's own boxplot for the *same 12 forebrain
  areas*, because both functions used a fixed figsize regardless of area
  count (`(11, 6)` vs `(8, 6)`) rather than scaling with `len(area_order)`.
  `plot_area_comparison_boxplot`'s defaults match lifetime_sparseness's
  original numbers (the older, already-shipped style); its box/line
  recoloring uses the more robust x-position-based matching from
  template_matching_classification's own (already-fixed) approach, since
  that one isn't just a style choice -- it doesn't assume a fixed
  Line2D-per-box count, which the older approach silently relied on.
  `notebooks/neural/response_latency.ipynb` (not yet ported) reuses this
  same pattern almost verbatim -- reuse this function there too, then
  re-verify.
"""
from __future__ import annotations

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_neuron_scatter_on_brain(
    ref_brain: np.ndarray,
    coords: np.ndarray,
    values: np.ndarray,
    cmap,
    *,
    colorbar_label: str = "",
    point_size: float = 10,
    point_alpha: float = 0.5,
    figsize: tuple[float, float] = (6, 8.2),
    scale_bar_y_offset: float = 20,
    scale_bar_xmin: float = 20,
    scale_bar_xmax: float = 120,
) -> plt.Figure:
    """Dorsal + sagittal brain projections with a colormapped neuron scatter.

    `ref_brain` is a 3D (z, y, x) reference brain volume; `coords` an
    (n_neurons, 3) array of (z, y, x) neuron positions; `values` the
    (n_neurons,) array each neuron is colored by via `cmap`. The
    `scale_bar_*` args position the dorsal-view scale bar (y-offset from the
    projection's top edge, x start/end), in the same pixel units as `coords`.
    """
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
    gs.update(hspace=-0.5)

    ax_horizontal = fig.add_subplot(gs[0])
    ax_sagittal = fig.add_subplot(gs[1])

    coords = np.asarray(coords)
    values = np.asarray(values)

    # Dorsal (horizontal) projection: mean along z, rotated 90 deg CCW.
    proj_horizontal = np.nanmean(ref_brain, axis=0)
    proj_horizontal = np.rot90(proj_horizontal, k=1)
    ax_horizontal.imshow(proj_horizontal, cmap="gray_r", alpha=0.5, origin="lower")

    rotated_x = coords[:, 1]
    rotated_y = proj_horizontal.shape[0] - coords[:, 2]

    ax_horizontal.scatter(
        rotated_x, rotated_y, c=values, cmap=cmap,
        s=point_size, edgecolor="none", alpha=point_alpha,
    )
    ax_horizontal.axis("off")
    ax_horizontal.set_aspect("equal")
    ax_horizontal.set_xlim(0, proj_horizontal.shape[1])
    ax_horizontal.set_ylim(0, proj_horizontal.shape[0])
    ax_horizontal.hlines(
        y=proj_horizontal.shape[0] - scale_bar_y_offset,
        xmin=scale_bar_xmin, xmax=scale_bar_xmax,
        color="black", linewidth=3,
    )

    # Sagittal projection: mean along x.
    proj_sagittal = np.nanmean(ref_brain, axis=2)
    ax_sagittal.imshow(proj_sagittal, cmap="gray_r", alpha=0.5, origin="lower")
    ax_sagittal.scatter(
        coords[:, 1], coords[:, 0], c=values, cmap=cmap,
        s=point_size, edgecolor="none", alpha=point_alpha,
    )
    ax_sagittal.axis("off")
    ax_sagittal.set_aspect("equal")
    ax_sagittal.set_xlim(0, proj_sagittal.shape[1])
    ax_sagittal.set_ylim(0, proj_sagittal.shape[0])

    cbar_ax = fig.add_axes([0.88, 0.4, 0.02, 0.3])
    cbar = fig.colorbar(ax_horizontal.collections[0], cax=cbar_ax)
    cbar.set_label(colorbar_label, fontsize=23)
    cbar.ax.tick_params(labelsize=23)

    return fig


def plot_area_comparison_boxplot(
    data: pd.DataFrame,
    value_col: str,
    areas_to_plot: tuple[str, ...],
    area_order: tuple[str, ...],
    pvals: dict[str, float],
    palette: dict[str, str],
    *,
    reference_area: str | None = None,
    ylabel: str = "",
    ylim: tuple[float, float] = (0.0, 1.0),
    yticks: list[float] | None = None,
    significance_y: float | None = None,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Per-brain-area boxplot with jittered per-fish points and significance stars vs. `reference_area`.

    `data` needs `"area"` and `value_col` columns. `figsize` defaults to
    `(max(6.0, 0.6 * len(area_order)), 6.0)` -- scaling box width with area
    count avoids the same style looking cramped or overly wide when reused
    across area groups of very different sizes (see module docstring).
    `significance_y` defaults to `ylim[1]` when not given.
    """
    import seaborn as sns

    if figsize is None:
        figsize = (max(6.0, 0.6 * len(area_order)), 6.0)
    if significance_y is None:
        significance_y = ylim[1]

    plot_data = data[data["area"].isin(areas_to_plot)].copy()

    fig = plt.figure(figsize=figsize, facecolor="white")
    ax = plt.gca()
    ax.set_facecolor("white")

    ax = sns.boxplot(
        data=plot_data, x="area", y=value_col, order=area_order,
        color="white", width=0.7, showcaps=False, showfliers=False,
        boxprops={"edgecolor": "black", "linewidth": 3},
        whiskerprops={"color": "black", "linewidth": 3},
        medianprops={"color": "black", "linewidth": 3},
    )

    plotted_areas = [area for area in area_order if area in plot_data["area"].unique()]

    for i, area in enumerate(plotted_areas):
        box = ax.patches[i]
        box.set_edgecolor(palette[area])
        box.set_linewidth(2.7)
        box.set_facecolor("none")

    # Recolor lines by x-position rather than assuming a fixed line count per
    # box -- robust to seaborn versions that draw a different number of
    # Line2D objects per box.
    for line in ax.lines:
        xdata = line.get_xdata()
        if len(xdata) == 0:
            continue

        x_center = int(round(np.nanmean(xdata)))
        if 0 <= x_center < len(plotted_areas):
            color = palette[plotted_areas[x_center]]
            line.set_color(color)
            line.set_linewidth(2.5)

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=24, rotation=45, ha="center")
    for label, area in zip(ax.get_xticklabels(), plotted_areas):
        label.set_color(palette[area])

    rng = np.random.default_rng(42)
    for i, area in enumerate(plotted_areas):
        area_data = plot_data.loc[plot_data["area"] == area, value_col].values
        jittered_x = rng.normal(loc=i, scale=0.1, size=len(area_data))
        color = palette[area]
        ax.scatter(
            jittered_x, area_data, facecolors="white", edgecolors=color,
            s=55, alpha=0.6, linewidth=1.5, zorder=3,
        )

    for i, area in enumerate(area_order):
        if reference_area is not None and area == reference_area:
            continue

        p_val = pvals.get(area, 1.0)
        if p_val < 0.001:
            significance = "***"
        elif p_val < 0.01:
            significance = "**"
        elif p_val < 0.05:
            significance = "*"
        else:
            significance = "ns"

        if area in plotted_areas:
            ax.text(i, significance_y, significance, ha="center", va="bottom", fontsize=18, color="black", weight="bold")

    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=28)
    ax.set_ylim(ylim)
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.tick_params(axis="both", labelsize=28)
    ax.tick_params(axis="x", width=2)
    ax.tick_params(axis="y", width=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)

    plt.tight_layout()
    return fig
