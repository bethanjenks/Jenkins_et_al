"""Regression test for jenkins_et_al.neural.plot_regression_spatial_maps
against a golden capture of notebooks/neural/plot_regression_spatial_maps.ipynb.

golden/plot_regression_spatial_maps/ was produced by running the notebook's
own cell source fresh top-to-bottom (capture_golden_plot_regression_spatial_maps.py,
run once during this port).

Confirms the data feeding both figures: the significance/threshold-filtered
`plot_df` (cell 4) and the BHV-restricted `correlation_data` (cells 7-10),
including the tie-break behavior for neurons significant in both valence
directions (none exist in the real dataset -- see module docstring).
Rendered figures are not re-asserted pixel-exact, per the golden-refactor
skill's "verify the data, not pixels" rule.

Skipped entirely if the source data drive isn't mounted.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from jenkins_et_al.neural import plot_regression_spatial_maps as prsm

POS_PATH = Path("/Volumes/LaCie/larval_HuC/imaging/7dpf_positive_correlations_25s_no_HCl_1000_corrected.csv")
NEG_PATH = Path("/Volumes/LaCie/larval_HuC/imaging/7dpf_negative_correlations_25s_no_HCl_1000_corrected.csv")
REF_BRAIN_PATH = Path("/Volumes/LaCie/area_masks/T_AVG_HuC.tif")
BHV_PATH = Path("/Volumes/LaCie/larval_HuC/imaging/7dpf_bhv_vigor_correlations_1000_corrected.csv")
GOLDEN_DIR = Path(__file__).resolve().parent.parent.parent / "golden" / "plot_regression_spatial_maps"

pytestmark = pytest.mark.skipif(
    not POS_PATH.exists() or not GOLDEN_DIR.exists(),
    reason="source data drive or golden capture not available on this machine",
)


@pytest.fixture(scope="module")
def valence_and_brain():
    valence = prsm.load_valence_data(POS_PATH, NEG_PATH)
    ref_brain = prsm.load_ref_brain(REF_BRAIN_PATH)
    return valence, ref_brain


def test_ref_brain_shape_matches_golden(valence_and_brain):
    _, ref_brain = valence_and_brain
    golden_shape = tuple(np.load(GOLDEN_DIR / "ref_brain_shape.npy"))
    assert ref_brain.shape == golden_shape


def test_plot_df_matches_golden(valence_and_brain):
    valence, _ = valence_and_brain
    plot_df = prsm.filter_significant(valence)

    golden = pd.read_csv(GOLDEN_DIR / "plot_df.csv")

    assert len(plot_df) == len(golden)
    assert (plot_df["valence"].value_counts().sort_index().to_numpy() == golden["valence"].value_counts().sort_index().to_numpy()).all()
    assert np.allclose(
        np.sort(plot_df["correlation"].to_numpy()), np.sort(golden["correlation"].to_numpy()),
    )
    assert plot_df["coords"].apply(str).sort_values().reset_index(drop=True).equals(
        golden["coords"].astype(str).sort_values().reset_index(drop=True)
    )


def test_bhv_correlation_data_matches_golden():
    merged = prsm.load_bhv_merged_data(BHV_PATH, POS_PATH, NEG_PATH)
    significant = prsm.select_significant_bhv_neurons(merged)
    labeled = prsm.assign_bhv_valence(significant)
    correlation_data = prsm.select_bhv_correlation(labeled)

    golden = pd.read_csv(GOLDEN_DIR / "correlation_data.csv")

    assert len(correlation_data) == len(golden)
    assert (
        correlation_data["valence"].astype(int).value_counts().sort_index().to_numpy()
        == golden["valence"].astype(int).value_counts().sort_index().to_numpy()
    ).all()
    assert np.allclose(
        np.sort(correlation_data["correlation"].to_numpy()), np.sort(golden["correlation"].to_numpy()),
    )


def test_no_neurons_significant_in_both_valence_directions():
    """Confirms the tie-break discrepancy (see module docstring) is inert on real data."""
    merged = prsm.load_bhv_merged_data(BHV_PATH, POS_PATH, NEG_PATH)
    significant = prsm.select_significant_bhv_neurons(merged)
    both_sig = significant[(significant["pos_pfdr"] < 0.05) & (significant["neg_pfdr"] < 0.05)]
    assert len(both_sig) == 0


def test_plots_render(valence_and_brain):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valence, ref_brain = valence_and_brain
    plot_df = prsm.filter_significant(valence)

    fig1 = prsm.plot_valence_corr(ref_brain, plot_df)
    assert isinstance(fig1, plt.Figure)
    plt.close(fig1)

    merged = prsm.load_bhv_merged_data(BHV_PATH, POS_PATH, NEG_PATH)
    significant = prsm.select_significant_bhv_neurons(merged)
    labeled = prsm.assign_bhv_valence(significant)
    correlation_data = prsm.select_bhv_correlation(labeled)

    fig2 = prsm.plot_valence_corr(ref_brain, correlation_data)
    assert isinstance(fig2, plt.Figure)
    plt.close(fig2)
