"""Regression test for jenkins_et_al.neural.lda_svm_spatial_segregation
against a golden capture of notebooks/neural/lda_svm_spatial_segregation.ipynb.

golden/lda_svm_spatial_segregation/ was produced by running the notebook's own
cell source fresh top-to-bottom against real data (capture_golden.py, run once
during this port). The notebook's saved outputs already had sequential
execution counts with no gaps, but per the golden-refactor skill's "run from a
clean kernel" rule, a fresh run was captured rather than trusted on
appearance -- it matched the saved outputs exactly, no bugs found.

Confirms the per-area LDA/SVM/permutation-test results table
(`spatial_method_results.csv`/`spatial_permutation_pvalues.csv`, the
notebook's own two saved outputs) for the default `AREAS = 'olfactory_bulb'`,
`N_PERMUTATIONS = 1000` configuration -- fully deterministic (StratifiedKFold
shuffling and the permutation shuffle both seeded via `RANDOM_STATE = 42`).
Also confirms the area-filtered neuron tables and the full-data LDA vector
feeding the "quick plots" section. Rendered figures are not re-asserted
pixel-exact, per the golden-refactor skill's "verify the data, not pixels"
rule.

Skipped entirely if the source data drive isn't mounted.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from jenkins_et_al.neural import lda_svm_spatial_segregation as lss

POS_PATH = Path("/Volumes/LaCie/larval_HuC/imaging/7dpf_positive_correlations_25s_no_HCl_1000_corrected.csv")
NEG_PATH = Path("/Volumes/LaCie/larval_HuC/imaging/7dpf_negative_correlations_25s_no_HCl_1000_corrected.csv")
GOLDEN_DIR = Path(__file__).resolve().parent.parent.parent / "golden" / "lda_svm_spatial_segregation"

P_FDR_THRESH = 0.05
MIN_PER_CLASS = 20
CV_FOLDS = 5
RANDOM_STATE = 42
N_PERMUTATIONS = 1000

pytestmark = pytest.mark.skipif(
    not POS_PATH.exists() or not GOLDEN_DIR.exists(),
    reason="source data drive or golden capture not available on this machine",
)


@pytest.fixture(scope="module")
def valence_df():
    return lss.load_valence_table(POS_PATH, NEG_PATH)


def test_valence_table_matches_golden(valence_df):
    golden = json.loads((GOLDEN_DIR / "valence_shape.json").read_text())
    assert list(valence_df.shape) == golden["shape"]
    assert valence_df["area"].nunique() == golden["n_areas"]


def test_areas_to_run_matches_golden(valence_df):
    golden = json.loads((GOLDEN_DIR / "areas_to_run.json").read_text())
    areas_to_run = lss.select_areas_to_run(
        valence_df, "olfactory_bulb", p_fdr_thresh=P_FDR_THRESH, min_per_class=MIN_PER_CLASS,
    )
    assert areas_to_run == golden


@pytest.fixture(scope="module")
def area_results(valence_df):
    areas_to_run = lss.select_areas_to_run(
        valence_df, "olfactory_bulb", p_fdr_thresh=P_FDR_THRESH, min_per_class=MIN_PER_CLASS,
    )
    return lss.run_area_analysis(
        valence_df, areas_to_run,
        p_fdr_thresh=P_FDR_THRESH, min_per_class=MIN_PER_CLASS,
        cv_folds=CV_FOLDS, random_state=RANDOM_STATE, n_permutations=N_PERMUTATIONS,
    )


def test_results_df_matches_golden(area_results):
    results_df, _ = area_results
    golden = pd.read_csv(GOLDEN_DIR / "results_df.csv")

    assert list(results_df["area"]) == list(golden["area"])
    assert list(results_df["method"]) == list(golden["method"])
    assert list(results_df["n_pos"]) == list(golden["n_pos"])
    assert list(results_df["n_neg"]) == list(golden["n_neg"])

    numeric_cols = [
        "accuracy_mean", "accuracy_std", "auc_mean", "auc_std", "fisher_ratio",
        "lda_vector_x", "lda_vector_y", "lda_vector_z",
        "angle_to_x", "angle_to_y", "angle_to_z", "n_total",
    ]
    for col in numeric_cols:
        port_vals = pd.to_numeric(results_df[col], errors="coerce").to_numpy(dtype=float)
        golden_vals = pd.to_numeric(golden[col], errors="coerce").to_numpy(dtype=float)
        assert np.allclose(port_vals, golden_vals, rtol=1e-10, atol=1e-12, equal_nan=True)


def test_perm_df_matches_golden(area_results):
    _, perm_df = area_results
    golden = pd.read_csv(GOLDEN_DIR / "perm_df.csv")

    assert list(perm_df["area"]) == list(golden["area"])
    assert list(perm_df["method"]) == list(golden["method"])
    assert np.allclose(perm_df["observed_auc"].to_numpy(), golden["observed_auc"].to_numpy(), rtol=1e-10)
    assert np.allclose(perm_df["p_value"].to_numpy(), golden["p_value"].to_numpy(), rtol=1e-10)


def test_val_neurons_match_golden(valence_df):
    val_neurons, nonval_neurons = lss.filter_area(valence_df, "olfactory_bulb", p_fdr_thresh=P_FDR_THRESH)

    golden_val = pd.read_csv(GOLDEN_DIR / "val_neurons.csv")
    golden_nonval = pd.read_csv(GOLDEN_DIR / "nonval_neurons.csv")

    assert len(val_neurons) == len(golden_val)
    assert len(nonval_neurons) == len(golden_nonval)
    assert (val_neurons["valence"].value_counts().sort_index().to_numpy()
            == golden_val["valence"].value_counts().sort_index().to_numpy()).all()


def test_lda_row_matches_golden(area_results):
    results_df, _ = area_results
    golden = json.loads((GOLDEN_DIR / "lda_row.json").read_text())

    lda_row = results_df[(results_df["area"] == "olfactory_bulb") & (results_df["method"] == "lda")].iloc[0]
    assert np.isclose(lda_row["lda_vector_x"], golden["lda_vector_x"], rtol=1e-10)
    assert np.isclose(lda_row["lda_vector_y"], golden["lda_vector_y"], rtol=1e-10)
    assert np.isclose(lda_row["lda_vector_z"], golden["lda_vector_z"], rtol=1e-10)
    assert np.isclose(lda_row["fisher_ratio"], golden["fisher_ratio"], rtol=1e-10)
    assert np.isclose(lda_row["accuracy_mean"], golden["accuracy_mean"], rtol=1e-10)
    assert np.isclose(lda_row["auc_mean"], golden["auc_mean"], rtol=1e-10)


def test_quick_plots_render(valence_df, area_results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results_df, _ = area_results
    val_neurons, nonval_neurons = lss.filter_area(valence_df, "olfactory_bulb", p_fdr_thresh=P_FDR_THRESH)
    lda_row = results_df[(results_df["area"] == "olfactory_bulb") & (results_df["method"] == "lda")].iloc[0]
    lda_vec = np.array([lda_row["lda_vector_x"], lda_row["lda_vector_y"], lda_row["lda_vector_z"]])

    fig1 = lss.plot_method_comparison(results_df, "olfactory_bulb")
    fig2 = lss.plot_basic_3d_scatter(val_neurons, nonval_neurons, plot_axes=("x", "y", "z"))
    fig3 = lss.plot_lda_axis_plane_3d(val_neurons, lda_vec, plot_axes=("x", "y", "z"))
    fig4 = lss.plot_lda_projection_histogram(val_neurons, title="LDA Projection — olfactory_bulb")

    slice_data = lss.compute_svm_decision_boundary_slice(val_neurons, balance="downsample")
    fig5 = lss.plot_svm_decision_boundary_slice(slice_data)

    for fig in (fig1, fig2, fig3, fig4, fig5):
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
