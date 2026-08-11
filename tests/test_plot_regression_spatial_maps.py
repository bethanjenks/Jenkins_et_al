"""Unit tests for jenkins_et_al.neural.plot_regression_spatial_maps against
small synthetic inputs (no dependency on the real data drive).

The full pipeline against real data is golden-verified separately in
tests/golden/test_plot_regression_spatial_maps.py.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from jenkins_et_al.neural import plot_regression_spatial_maps as prsm


def test_load_valence_data_labels_and_concatenates(tmp_path):
    pos_path = tmp_path / "pos.csv"
    neg_path = tmp_path / "neg.csv"
    pd.DataFrame({"coords": ["[1.0, 2.0, 3.0]"], "area": ["pallium"], "correlation": [0.5], "p_fdr": [0.01]}).to_csv(pos_path, index=False)
    pd.DataFrame({"coords": ["[4.0, 5.0, 6.0]"], "area": ["not_assigned"], "correlation": [-0.5], "p_fdr": [0.02]}).to_csv(neg_path, index=False)

    result = prsm.load_valence_data(pos_path, neg_path)

    # "not_assigned" area row (from neg) is dropped.
    assert len(result) == 1
    assert result.iloc[0]["valence"] == 1
    assert result.iloc[0]["coords"] == [1.0, 2.0, 3.0]


def test_filter_significant_applies_alpha_and_threshold():
    df = pd.DataFrame({"correlation": [0.9, 0.1, 0.9], "p_fdr": [0.01, 0.01, 0.9]})

    sig_only = prsm.filter_significant(df, significance_alpha=0.05)
    assert len(sig_only) == 2

    sig_and_threshold = prsm.filter_significant(df, significance_alpha=0.05, threshold=0.5)
    assert len(sig_and_threshold) == 1


def test_filter_significant_raises_when_p_value_column_missing():
    df = pd.DataFrame({"correlation": [0.5]})
    with pytest.raises(ValueError, match="Expected a"):
        prsm.filter_significant(df, p_value_column="p_fdr")


def test_filter_significant_skips_alpha_when_use_significance_false():
    df = pd.DataFrame({"correlation": [0.9, 0.01], "p_fdr": [0.9, 0.9]})
    result = prsm.filter_significant(df, use_significance=False, threshold=0.5)
    assert len(result) == 1


def _make_bhv_csvs(tmp_path):
    bhv_path = tmp_path / "bhv.csv"
    pos_path = tmp_path / "pos.csv"
    neg_path = tmp_path / "neg.csv"

    pd.DataFrame({
        "fish_id": ["f1", "f2", "f3"], "neuron_id": ["n1", "n2", "n3"],
        "coords": ["[1.0, 1.0, 1.0]", "[2.0, 2.0, 2.0]", "[3.0, 3.0, 3.0]"],
        "correlation": [0.6, 0.7, 0.8], "p_fdr": [0.01, 0.01, 0.9],
    }).to_csv(bhv_path, index=False)

    pd.DataFrame({
        "fish_id": ["f1", "f2", "f3"], "neuron_id": ["n1", "n2", "n3"],
        "correlation": [0.5, 0.05, 0.5], "p_fdr": [0.01, 0.9, 0.01],
    }).to_csv(pos_path, index=False)

    pd.DataFrame({
        "fish_id": ["f1", "f2", "f3"], "neuron_id": ["n1", "n2", "n3"],
        "correlation": [-0.5, -0.6, -0.05], "p_fdr": [0.01, 0.01, 0.9],
    }).to_csv(neg_path, index=False)

    return bhv_path, pos_path, neg_path


def test_load_bhv_merged_data_renames_and_joins(tmp_path):
    bhv_path, pos_path, neg_path = _make_bhv_csvs(tmp_path)
    merged = prsm.load_bhv_merged_data(bhv_path, pos_path, neg_path)

    assert len(merged) == 3
    assert {"bhv_correlation", "bhv_pfdr", "pos_correlation", "pos_pfdr", "neg_correlation", "neg_pfdr"}.issubset(merged.columns)
    assert merged.iloc[0]["coords"] == [1.0, 1.0, 1.0]


def test_select_significant_bhv_neurons_requires_bhv_and_one_valence_direction(tmp_path):
    bhv_path, pos_path, neg_path = _make_bhv_csvs(tmp_path)
    merged = prsm.load_bhv_merged_data(bhv_path, pos_path, neg_path)

    significant = prsm.select_significant_bhv_neurons(merged)

    # f1: bhv sig, pos sig and neg sig -> included.
    # f2: bhv sig, pos not sig but neg sig -> included.
    # f3: bhv not sig -> excluded even though pos is sig.
    assert set(significant["fish_id"]) == {"f1", "f2"}


def test_assign_bhv_valence_negative_wins_ties_by_default():
    df = pd.DataFrame({"pos_pfdr": [0.01, 0.9, 0.01], "neg_pfdr": [0.01, 0.01, 0.9]})

    result = prsm.assign_bhv_valence(df)

    assert result["valence"].tolist() == [0, 0, 1]


def test_assign_bhv_valence_positive_tie_break():
    df = pd.DataFrame({"pos_pfdr": [0.01, 0.9, 0.01], "neg_pfdr": [0.01, 0.01, 0.9]})

    result = prsm.assign_bhv_valence(df, tie_break="positive")

    assert result["valence"].tolist() == [1, 0, 1]


def test_select_bhv_correlation_picks_pos_or_neg_by_valence():
    df = pd.DataFrame({"valence": [1, 0], "pos_correlation": [0.5, 0.6], "neg_correlation": [-0.5, -0.6]})

    result = prsm.select_bhv_correlation(df)

    assert result["correlation"].tolist() == [0.5, -0.6]


def _make_ref_brain_and_coords():
    ref_brain = np.random.default_rng(0).normal(size=(20, 40, 30))
    coords = np.array([[10.0, 15.0, 5.0], [12.0, 20.0, 8.0], [8.0, 10.0, 3.0]])
    regressor_corr = np.array([0.5, -0.4, 0.3])
    valence = np.array([1, 0, 1])
    return ref_brain, coords, regressor_corr, valence


def test_scatter_plots_returns_one_scatter_per_axis_per_valence():
    ref_brain, coords, regressor_corr, valence = _make_ref_brain_and_coords()
    fig, (ax1, ax2) = plt.subplots(2, 1)

    scatter_list = prsm.scatter_plots([ax1, ax2], ref_brain, coords, regressor_corr, valence)

    assert len(scatter_list) == 4  # 2 axes x 2 valence groups
    plt.close(fig)


def test_scatter_plots_horizontal_panel_uses_y_then_flipped_z():
    """Regression check against the axis-swap bug class found in #11's
    plot_brain_spatial -- confirms this notebook's own convention is correct."""
    ref_brain, coords, regressor_corr, valence = _make_ref_brain_and_coords()
    valence = np.array([1, 1, 1])  # single group -> single scatter per axis, easier to inspect
    fig, (ax1, ax2) = plt.subplots(2, 1)

    prsm.scatter_plots([ax1, ax2], ref_brain, coords, regressor_corr, valence)

    proj_horizontal = np.rot90(np.nanmean(ref_brain, axis=0), k=1)
    offsets = ax1.collections[1].get_offsets()  # collections[0] is the empty valence=0 scatter
    expected_x = coords[:, 1]
    expected_y = proj_horizontal.shape[0] - coords[:, 2]
    assert np.allclose(offsets[:, 0], expected_x)
    assert np.allclose(offsets[:, 1], expected_y)
    plt.close(fig)


def test_scatter_plots_sagittal_panel_uses_y_then_z():
    ref_brain, coords, regressor_corr, valence = _make_ref_brain_and_coords()
    valence = np.array([1, 1, 1])
    fig, (ax1, ax2) = plt.subplots(2, 1)

    prsm.scatter_plots([ax1, ax2], ref_brain, coords, regressor_corr, valence)

    offsets = ax2.collections[1].get_offsets()  # collections[0] is the empty valence=0 scatter
    assert np.allclose(offsets[:, 0], coords[:, 1])
    assert np.allclose(offsets[:, 1], coords[:, 0])
    plt.close(fig)


def test_plot_valence_corr_renders():
    ref_brain = np.random.default_rng(0).normal(size=(20, 40, 30))
    correlation_data = pd.DataFrame({
        "coords": [[10.0, 15.0, 5.0], [12.0, 20.0, 8.0]],
        "correlation": [0.5, -0.4],
        "valence": [1, 0],
    })

    fig = prsm.plot_valence_corr(ref_brain, correlation_data)

    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_valence_corr_accepts_named_matplotlib_colormap():
    ref_brain = np.random.default_rng(0).normal(size=(20, 40, 30))
    correlation_data = pd.DataFrame({
        "coords": [[10.0, 15.0, 5.0], [12.0, 20.0, 8.0]],
        "correlation": [0.5, -0.4],
        "valence": [1, 0],
    })

    fig = prsm.plot_valence_corr(ref_brain, correlation_data, cmap_neg="RdBu", cmap_pos="viridis")

    assert isinstance(fig, plt.Figure)
    plt.close(fig)
