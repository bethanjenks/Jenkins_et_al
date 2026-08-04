"""Unit tests for jenkins_et_al.plotting against small synthetic inputs
(no dependency on the real data drive)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from jenkins_et_al.plotting import (
    extract_area_contours,
    plot_area_comparison_boxplot,
    plot_brain_area_map,
    plot_neuron_scatter_on_brain,
)


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


def test_plot_area_comparison_boxplot_returns_figure_with_scaled_default_figsize():
    data = pd.DataFrame({
        "area": ["a", "a", "a", "b", "b", "b"],
        "value": [0.1, 0.2, 0.3, 0.7, 0.8, 0.9],
    })

    fig = plot_area_comparison_boxplot(
        data, "value", ("a", "b"), ("a", "b"), pvals={"b": 0.001},
        palette={"a": "black", "b": "red"}, reference_area="a", ylabel="Value",
    )

    assert isinstance(fig, plt.Figure)
    # default figsize scales with area count: max(6.0, 0.6 * 2) == 6.0
    assert fig.get_size_inches()[0] == pytest.approx(6.0)
    plt.close(fig)


def test_plot_area_comparison_boxplot_figsize_scales_with_more_areas():
    areas = tuple(f"area{i}" for i in range(20))
    data = pd.DataFrame({"area": [a for a in areas for _ in range(3)], "value": np.tile([0.1, 0.5, 0.9], 20)})
    palette = {a: "black" for a in areas}

    fig = plot_area_comparison_boxplot(data, "value", areas, areas, pvals={}, palette=palette)

    assert fig.get_size_inches()[0] == pytest.approx(0.6 * 20)
    plt.close(fig)


def _make_single_area_mask(shape: tuple[int, int, int] = (20, 20, 5)) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    mask[5:15, 5:15, :] = 1
    return mask


def test_extract_area_contours_finds_one_contour_for_a_simple_blob():
    contours = extract_area_contours(_make_single_area_mask(), slice_offset=0)

    assert len(contours) == 1
    rows, cols = contours[0][:, 0], contours[0][:, 1]
    assert rows.min() >= 0 and rows.max() <= 19
    assert cols.min() >= 0 and cols.max() <= 19


def test_extract_area_contours_returns_empty_list_for_an_empty_slice():
    mask = np.zeros((20, 20, 5), dtype=np.uint8)
    assert extract_area_contours(mask, slice_offset=0) == []


def test_plot_brain_area_map_outline_only_has_no_colorbar_axis():
    ref_brain = np.zeros((20, 20, 5))
    areas = {"a": _make_single_area_mask()}

    fig = plot_brain_area_map(ref_brain, areas, {"a": "A"}, slice_offset=0, area_values=None)

    assert len(fig.axes) == 1
    plt.close(fig)


def test_plot_brain_area_map_with_values_adds_a_colorbar_axis():
    ref_brain = np.zeros((20, 20, 5))
    areas = {"a": _make_single_area_mask()}
    area_values = pd.DataFrame({"area_names": ["a"], "values": [0.5], "p_value": [1.0]})

    fig = plot_brain_area_map(ref_brain, areas, {"a": "A"}, slice_offset=0, area_values=area_values)

    assert len(fig.axes) == 2
    plt.close(fig)


def test_plot_brain_area_map_uses_the_given_colormap_and_value_range():
    from matplotlib.colors import Normalize

    ref_brain = np.zeros((20, 20, 5))
    areas = {"a": _make_single_area_mask()}
    area_values = pd.DataFrame({"area_names": ["a"], "values": [4.0], "p_value": [1.0]})

    fig = plot_brain_area_map(
        ref_brain, areas, {"a": "A"}, slice_offset=0,
        area_values=area_values, cmap="magma", vmin=0.0, vmax=10.0, fill_alpha=1.0,
    )

    expected_color = plt.get_cmap("magma")(Normalize(vmin=0.0, vmax=10.0)(4.0))
    actual_color = fig.axes[0].patches[0].get_facecolor()
    assert np.allclose(actual_color, expected_color)
    plt.close(fig)


def test_plot_brain_area_map_show_significance_toggle_adds_or_omits_outline():
    ref_brain = np.zeros((20, 20, 5))
    areas = {"a": _make_single_area_mask()}
    area_values = pd.DataFrame({"area_names": ["a"], "values": [0.5], "p_value": [0.001]})

    fig_on = plot_brain_area_map(
        ref_brain, areas, {"a": "A"}, slice_offset=0, area_values=area_values, show_significance=True,
    )
    fig_off = plot_brain_area_map(
        ref_brain, areas, {"a": "A"}, slice_offset=0, area_values=area_values, show_significance=False,
    )

    assert "orange" in {line.get_color() for line in fig_on.axes[0].lines}
    assert "orange" not in {line.get_color() for line in fig_off.axes[0].lines}
    plt.close(fig_on)
    plt.close(fig_off)


def test_plot_area_comparison_boxplot_accepts_explicit_figsize_override():
    data = pd.DataFrame({"area": ["a", "a", "b", "b"], "value": [0.1, 0.2, 0.8, 0.9]})

    fig = plot_area_comparison_boxplot(
        data, "value", ("a", "b"), ("a", "b"), pvals={}, palette={"a": "black", "b": "red"}, figsize=(8, 6),
    )

    assert fig.get_size_inches()[0] == pytest.approx(8.0)
    plt.close(fig)
