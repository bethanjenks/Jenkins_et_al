"""Unit tests for jenkins_et_al.plotting against small synthetic inputs
(no dependency on the real data drive)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from jenkins_et_al.plotting import plot_neuron_scatter_on_brain


def test_plot_neuron_scatter_on_brain_returns_figure_with_two_axes_and_colorbar():
    rng = np.random.default_rng(0)
    ref_brain = rng.random((10, 20, 30))
    coords = rng.integers(0, 10, size=(50, 3)).astype(float)
    values = rng.random(50)

    fig = plot_neuron_scatter_on_brain(ref_brain, coords, values, cmap="viridis", colorbar_label="Test value")

    assert isinstance(fig, plt.Figure)
    # 2 projection axes + 1 colorbar axis
    assert len(fig.axes) == 3
    assert fig.axes[0].collections[0].get_alpha() == 0.5

    plt.close(fig)


def test_plot_neuron_scatter_on_brain_accepts_a_custom_colormap_object():
    import matplotlib.colors as mcolors

    rng = np.random.default_rng(1)
    ref_brain = rng.random((5, 10, 15))
    coords = rng.integers(0, 5, size=(20, 3)).astype(float)
    values = rng.random(20)

    custom_cmap = mcolors.LinearSegmentedColormap.from_list("test_cmap", ["#9b00e8", "white", "#ff7f0e"])
    fig = plot_neuron_scatter_on_brain(ref_brain, coords, values, custom_cmap)

    assert isinstance(fig, plt.Figure)
    plt.close(fig)
