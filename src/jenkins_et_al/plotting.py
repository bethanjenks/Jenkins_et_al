"""General-purpose figure functions shared across neural notebooks.

Holds three patterns reused across multiple already-ported notebooks:

- The whole-brain neuron scatter projection: two stacked views
  (dorsal/horizontal on top, sagittal below) of a reference brain volume
  with neurons overlaid as a colormapped scatter. First ported from
  notebooks/neural/lifetime_sparseness.ipynb's `plot_sparseness_brain`
  (cell 27), which colored neurons by lifetime sparseness with a custom
  purple-white-orange colormap. The same horizontal+sagittal projection
  pattern (`np.rot90` dorsal view, `np.nanmean` sagittal view, coordinate
  transform `rotated_x = coords[:, 1]`, `rotated_y = height - coords[:, 2]`)
  recurs with a different colormap/value in
  notebooks/neural/functional_clustering_valence_neurons.ipynb (cell 17) and
  notebooks/neural/plot_regression_spatial_maps.ipynb (cell 2) -- those are
  not yet ported/verified, so this function is deliberately generalized only
  along the axis lifetime_sparseness itself needs (continuous value +
  colormap), not unified with those other notebooks' categorical-label
  variants. Reuse this function when porting them, extending its parameters
  as needed, then re-verify lifetime_sparseness's own golden capture still
  matches.
  **Correction (port #5, build_brain_map)**: an earlier version of this
  docstring also listed `build_brain_map.ipynb` here. Checked directly while
  porting it: that notebook never does a per-neuron scatter -- it fills/
  outlines whole *area* contours at one A-P slice instead (see
  `plot_brain_area_map` below). The grouping was an unverified guess made
  while porting #6; removed now that #5 has actually been read.

- The single-slice brain-area contour map: per-area 2D contours (extracted
  from 3D area masks via erosion + Gaussian smoothing + marching squares) at
  one A-P slice of a reference brain volume, either as plain black outlines
  (`area_values=None`) or filled by a colormapped per-area value with an
  optional significance-highlighted outline. Ported from
  notebooks/neural/build_brain_map.ipynb (cells 5 and 9, which are this same
  function called with different arguments -- see
  `jenkins_et_al.neural.build_brain_map` for the two thin wrappers that
  reproduce each cell's exact original defaults).

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
from adjustText import adjust_text
from matplotlib.colors import Colormap, Normalize
from matplotlib.ticker import FuncFormatter
from scipy.ndimage import gaussian_filter
from skimage import measure, morphology


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


def extract_area_contours(
    mask: np.ndarray,
    slice_offset: int,
    *,
    erosion_radius: int = 1,
    smoothing_sigma: float = 3,
    contour_level: float = 0.5,
) -> list[np.ndarray]:
    """2D contours of one area's 3D mask at one A-P slice.

    Pipeline: take the slice at index ``mask.shape[2] // 2 - slice_offset``
    along the mask's last axis, binary-erode it, Gaussian-smooth it, then
    extract contours via marching squares (`skimage.measure.find_contours`).
    Returns a list of `(n_points, 2)` `(row, col)` coordinate arrays -- one
    per disconnected contour found at that slice (usually one, sometimes
    zero, occasionally more). Used by `plot_brain_area_map`; factored out
    so contour data can be checked directly in tests without re-deriving it
    from a rendered figure.
    """
    slice_index = mask.shape[2] // 2
    area_2d = mask[:, :, slice_index - slice_offset]

    eroded = morphology.binary_erosion(area_2d, morphology.disk(erosion_radius))
    smoothed = gaussian_filter(eroded.astype(float), sigma=smoothing_sigma)
    return measure.find_contours(smoothed, level=contour_level)


def plot_brain_area_map(
    ref_brain: np.ndarray,
    areas: dict[str, np.ndarray],
    area_shorthand: dict[str, str],
    *,
    slice_offset: int = 68,
    area_values: pd.DataFrame | None = None,
    area_name_col: str = "area_names",
    value_col: str = "values",
    pvalue_col: str | None = "p_value",
    default_p_value: float = 1.0,
    show_significance: bool = True,
    significance_threshold: float = 0.05,
    significance_color: str = "orange",
    significance_linewidth: float = 3,
    nonsignificance_linewidth: float = 0.5,
    cmap: str | Colormap = "viridis",
    vmin: float = 0.0,
    vmax: float = 1.0,
    nan_color: str = "gray",
    colorbar_label: str = "",
    colorbar_ticks: list[float] | None = None,
    colorbar_fontsize: float = 30,
    background_alpha: float = 0.5,
    fill_alpha: float = 0.8,
    figsize: tuple[float, float] = (20, 15),
    label_fontsize: float = 15,
    erosion_radius: int = 1,
    smoothing_sigma: float = 3,
    contour_level: float = 0.5,
) -> plt.Figure:
    """Single-slice map of brain-area contours, optionally filled/colored by a per-area value.

    Source: notebooks/neural/build_brain_map.ipynb, cells 5 (outline-only,
    ``area_values=None``) and 9 (value-filled). Both cells are this same
    function called with different arguments; see
    ``jenkins_et_al.neural.build_brain_map.plot_area_outline_map`` and
    ``.plot_area_value_map`` for thin wrappers reproducing each cell's exact
    original parameter values.

    **Data this function needs:**

    - ``ref_brain``: a 3D ``(z, y, x)`` reference brain image volume (e.g.
      loaded from a ``.tif`` stack). Only used for a faint background
      projection (``np.nanmean(ref_brain, axis=2)``) -- it is not the source
      of the area contours themselves.
    - ``areas``: a dict mapping brain-area name -> a 3D binary/label mask
      array with the *same shape as* ``ref_brain``. One coronal-ish slice
      (index ``mask.shape[2] // 2 - slice_offset`` along the last axis) is
      taken per area and turned into a 2D contour via binary erosion,
      Gaussian smoothing, then marching-squares contour extraction
      (``skimage.measure.find_contours``). Iterated in dict order.
    - ``area_shorthand``: dict mapping the same area names used as
      ``areas`` keys to a short display label (e.g. ``"olfactory_bulb"`` ->
      ``"OB"``). An area missing from this dict is labeled with its raw
      (full) name instead.
    - ``area_values`` (optional): a long-format `pandas.DataFrame` with one
      row per brain area, giving the value (and optionally the p-value) to
      color/annotate that area with. Required columns:

      - ``area_name_col`` (default ``"area_names"``): brain-area name,
        matching a key of ``areas``.
      - ``value_col`` (default ``"values"``): the numeric value that
        ``cmap``/``vmin``/``vmax`` map to a fill color. Any area present in
        ``areas`` but absent from this dataframe is filled with
        ``nan_color`` instead.
      - ``pvalue_col`` (default ``"p_value"``, optional): only read when
        ``show_significance=True``; drives the significance-highlighted
        outline (see below). Pass ``pvalue_col=None`` for analyses that
        don't compute a p-value per area -- areas then always get the
        ``nonsignificance_linewidth``/black outline.

      Leave ``area_values=None`` entirely for an outline-only map with no
      fill and no colorbar (cell 5's plot).

    Color/significance behavior:

    - ``cmap``/``vmin``/``vmax`` control the fill colormap and its value
      range -- pass any named colormap or `matplotlib.colors.Colormap`
      instance, and the min/max data values it should span. A resolved copy
      of ``cmap`` is used internally (via ``.copy()``) so ``nan_color`` is
      never set on a colormap instance shared with other callers.
    - Every contour always gets a plain 1px black outline. When
      ``show_significance=True`` *and* ``area_values`` is given, a second,
      thicker outline is drawn on top: ``significance_color`` (default
      orange) at ``significance_linewidth`` if that area's p-value is below
      ``significance_threshold``, else black at ``nonsignificance_linewidth``.
      Set ``show_significance=False`` to skip this second outline entirely
      (e.g. for analyses with no significance test).

    Returns the `matplotlib.figure.Figure`; a colorbar is added only when
    ``area_values`` is given.
    """
    resolved_cmap = plt.get_cmap(cmap).copy() if isinstance(cmap, str) else cmap.copy()
    resolved_cmap.set_bad(color=nan_color)
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(np.nanmean(ref_brain, 2), cmap="gray_r", alpha=background_alpha, origin="lower")
    ax.axis("off")
    text_objects = []

    for brain_area, mask in areas.items():
        contours = extract_area_contours(
            mask, slice_offset,
            erosion_radius=erosion_radius, smoothing_sigma=smoothing_sigma, contour_level=contour_level,
        )

        shorthand_name = area_shorthand.get(brain_area, brain_area)

        area_color = None
        p_value = default_p_value
        if area_values is not None:
            matches = area_values[area_values[area_name_col] == brain_area]
            if len(matches) > 0:
                area_value = matches[value_col].values[0]
                if pvalue_col is not None:
                    p_value = matches[pvalue_col].values[0]
            else:
                area_value = np.nan
            area_color = resolved_cmap(norm(area_value))

        for contour in contours:
            x, y = contour[:, 1], contour[:, 0]

            if area_color is not None:
                ax.fill(x, y, color=area_color, alpha=fill_alpha)

            text_objects.append(
                ax.text(np.mean(x), np.mean(y), shorthand_name, color="black",
                        fontsize=label_fontsize, ha="center", va="center")
            )
            ax.plot(x, y, color="black", linewidth=1)

            if area_values is not None and show_significance and pvalue_col is not None:
                significant = p_value < significance_threshold
                ax.plot(
                    x, y,
                    color=significance_color if significant else "black",
                    linewidth=significance_linewidth if significant else nonsignificance_linewidth,
                )

    adjust_text(text_objects, ax=ax, expand_text=(1.2, 1.2))

    if area_values is not None:
        sm = plt.cm.ScalarMappable(cmap=resolved_cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.3, alpha=fill_alpha)
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))
        if colorbar_ticks is not None:
            cbar.set_ticks(colorbar_ticks)
        cbar.set_label(colorbar_label, rotation=270, labelpad=50, fontsize=colorbar_fontsize)
        cbar.ax.tick_params(labelsize=colorbar_fontsize)

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
