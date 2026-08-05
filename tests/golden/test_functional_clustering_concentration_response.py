"""Regression test for jenkins_et_al.neural.functional_clustering_concentration_response
against a golden capture of notebooks/neural/functional_clustering_concentration_response.ipynb.

golden/functional_clustering_concentration_response/ was produced by executing
the (now bug-fixed, see module docstring) notebook's own cell source fresh
top-to-bottom against real data for `CURRENT_ODORANT = "quinine"`, plus an
independent from-scratch reimplementation of the same pipeline
(capture_golden.py, run once during this port and cross-checked against the
notebook's own printed cluster sizes/fractions -- see PORTING_PLAN.md).

Confirms the data underlying all five figures, per the golden-refactor
skill's "verify the data, not pixels" rule: `pivot_df` (cluster assignments,
response-type names, per-concentration responses), `fractions_df`, and the
per-neuron coordinates + response-type labels feeding the brain-projection
plot. Rendered PNGs were also visually signed off by the user (see
PORTING_PLAN.md) but aren't re-asserted here as pixel-exact -- rendering
varies with backend/DPI even when the underlying data is identical.

Skipped entirely if the source data drive isn't mounted, since the
underlying response-vector HDF5 and regression CSV are not part of this
repository.
"""
from __future__ import annotations

from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from jenkins_et_al.neural import functional_clustering_concentration_response as fccr

REF_BRAIN_PATH = Path("/Volumes/LaCie/area_masks/T_AVG_HuC.tif")
REGRESSION_DATA_PATH = Path(
    "/Volumes/LaCie/larval_HuC/imaging/regression_results/7dpf_combined_stimulus_response_sustained_25s_correlations.csv"
)
RESPONSE_VECTOR_PATH = Path("/Volumes/LaCie/larval_HuC/imaging/7dpf_wb_population_general_response_vector.h5")
GOLDEN_DIR = Path(__file__).resolve().parent.parent.parent / "golden" / "functional_clustering_concentration_response"

pytestmark = pytest.mark.skipif(
    not REF_BRAIN_PATH.exists() or not GOLDEN_DIR.exists(),
    reason="source data drive or golden capture not available on this machine",
)


@pytest.fixture(scope="module")
def clustered_pivot_df():
    _, regress, data_resp = fccr.load_data(REF_BRAIN_PATH, REGRESSION_DATA_PATH, RESPONSE_VECTOR_PATH)
    odorant_config = fccr.ODORANT_CONCENTRATIONS["quinine"]

    pivot_df, averaged_responses = fccr.prepare_concentration_data(data_resp, odorant_config, regress)
    pivot_df, _, _ = fccr.perform_clustering(pivot_df, odorant_config["stimuli"])
    pivot_df = fccr.assign_cluster_names(pivot_df, fccr.QUININE_CLUSTER_NAME_MAP)
    return fccr.merge_coordinates(pivot_df, averaged_responses)


def test_pivot_df_matches_golden(clustered_pivot_df):
    golden = pd.read_csv(GOLDEN_DIR / "pivot_df.csv")
    port = clustered_pivot_df.drop(columns=["coords"]).reset_index(drop=True)
    golden = golden.reset_index(drop=True)

    assert list(port.columns) == list(golden.columns)
    assert port["fish_id"].tolist() == golden["fish_id"].tolist()
    assert port["neuron_id"].tolist() == golden["neuron_id"].tolist()
    assert (port["cluster"].values == golden["cluster"].values).all()
    assert (port["response_condition"].astype(str).values == golden["response_condition"].values).all()
    for stim in fccr.ODORANT_CONCENTRATIONS["quinine"]["stimuli"]:
        assert np.allclose(port[stim].values, golden[stim].values)


def test_cluster_sizes_match_golden(clustered_pivot_df):
    # Source: notebook's own printed cluster sizes after the odorant-switch fix.
    assert clustered_pivot_df["cluster"].value_counts().sort_index().to_dict() == {
        0: 16407, 1: 403, 2: 2932, 3: 2784,
    }


def test_response_fractions_match_golden(clustered_pivot_df):
    golden = pd.read_csv(GOLDEN_DIR / "fractions_df.csv").set_index("response_condition").sort_index()

    fractions_df = fccr.calculate_response_fractions(clustered_pivot_df, fccr.RESPONSE_CONDITION_ORDER)
    fractions_df = fractions_df.copy()
    fractions_df["response_condition"] = fractions_df["response_condition"].astype(str)
    port = fractions_df.set_index("response_condition").sort_index()

    assert (port["count"].values == golden["count"].values).all()
    assert np.allclose(port["fraction"].values, golden["fraction"].values)


def test_brain_plot_coords_and_labels_match_golden(clustered_pivot_df):
    golden = np.load(GOLDEN_DIR / "coords_and_labels.npz", allow_pickle=True)

    coords_list = [literal_eval(c) if isinstance(c, str) else c for c in clustered_pivot_df["coords"].values]
    coords_array = np.array(coords_list)
    response_array = clustered_pivot_df["response_condition"].values

    assert np.array_equal(coords_array, golden["coords"])
    assert np.array_equal(response_array, golden["response_condition"])


def test_brain_plot_renders(clustered_pivot_df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref_brain, _, _ = fccr.load_data(REF_BRAIN_PATH, REGRESSION_DATA_PATH, RESPONSE_VECTOR_PATH)
    coords_list = [
        literal_eval(c) if isinstance(c, str) else c for c in clustered_pivot_df["coords"].values
    ]
    coords_array = np.array(coords_list)
    response_array = clustered_pivot_df["response_condition"].values

    fig = fccr.plot_response_conditions_2view(coords_array, response_array, ref_brain, fccr.CLUSTER_COLORS)

    assert isinstance(fig, plt.Figure)
    plt.close(fig)
