"""General-purpose figure functions shared across neural notebooks.

Currently holds the whole-brain neuron scatter projection: two stacked
views (dorsal/horizontal on top, sagittal below) of a reference brain
volume with neurons overlaid as a colormapped scatter. First ported from
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
"""
from __future__ import annotations

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


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
