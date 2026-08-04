"""Unit tests for jenkins_et_al.neural.template_matching_classification against
small synthetic inputs (no dependency on the real data drive).

The odor/valence classification loops and `permutation_test` use unseeded
randomness and are individually expensive against real data (1000
permutations x cross-validated classification, per area, per fish) -- per
the user's decision, these are verified here with sanity checks on synthetic
data, not a golden capture of the real ~200k-neuron dataset. The deterministic
boxplot-vs-OB stats/plotting path is golden-verified separately in
tests/golden/test_template_matching_classification.py against a real,
pre-existing accuracy CSV.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from jenkins_et_al.neural import template_matching_classification as tmc


# ---------------------------------------------------------------------------
# template_matching_cross_validation
# ---------------------------------------------------------------------------

def test_template_matching_cross_validation_separates_distinct_clusters():
    # Cosine distance is direction-sensitive, not magnitude-sensitive -- these
    # clusters must differ in direction (not just be offset near the origin,
    # where direction is unstable) to be cleanly separable.
    rng = np.random.default_rng(0)
    labels = ["a", "b"]
    X_a = rng.normal(loc=[5, 0, 0, 0, 0], scale=0.1, size=(20, 5))
    X_b = rng.normal(loc=[0, 5, 0, 0, 0], scale=0.1, size=(20, 5))
    X = np.vstack([X_a, X_b])
    y = np.array(["a"] * 20 + ["b"] * 20)

    result = tmc.template_matching_cross_validation(X, y, labels, n_splits=4)

    assert result["mean_score"] > 0.9
    assert result["confusion_matrix"].shape == (2, 2)


def test_template_matching_cross_validation_is_deterministic():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 4))
    y = np.array(["a", "b", "c"] * 10)

    result1 = tmc.template_matching_cross_validation(X, y, ["a", "b", "c"], n_splits=3)
    result2 = tmc.template_matching_cross_validation(X, y, ["a", "b", "c"], n_splits=3)

    assert result1["mean_score"] == result2["mean_score"]


# ---------------------------------------------------------------------------
# permutation_test (unseeded -- sanity checks only)
# ---------------------------------------------------------------------------

def test_permutation_test_gives_low_p_for_well_separated_classes():
    rng = np.random.default_rng(2)
    X_a = rng.normal(loc=0, scale=0.1, size=(15, 5))
    X_b = rng.normal(loc=10, scale=0.1, size=(15, 5))
    X = np.vstack([X_a, X_b])
    y = np.array(["a"] * 15 + ["b"] * 15)

    p = tmc.permutation_test(X, y, ["a", "b"], n_splits=3, n_permutations=50)

    assert p < 0.05


def test_permutation_test_gives_high_p_for_indistinguishable_classes():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(30, 5))
    y = np.array(["a", "b"] * 15)

    p = tmc.permutation_test(X, y, ["a", "b"], n_splits=3, n_permutations=50)

    assert p > 0.2


# ---------------------------------------------------------------------------
# jitter (unseeded -- shape/bounds sanity only)
# ---------------------------------------------------------------------------

def test_jitter_stays_within_amount_of_original_values():
    values = np.array([1.0, 2.0, 3.0])
    jittered = tmc.jitter(values, amount=0.1)

    assert jittered.shape == values.shape
    assert np.all(np.abs(jittered - values) <= 0.1)


# ---------------------------------------------------------------------------
# combine_fish_statistics (deterministic Fisher's + FDR)
# ---------------------------------------------------------------------------

def test_combine_fish_statistics_flags_a_consistently_significant_area():
    all_areas = {
        "area_sig": [("fish1", 0.9, 0.001), ("fish2", 0.85, 0.002), ("fish3", 0.88, 0.001)],
        "area_ns": [("fish1", 0.5, 0.8), ("fish2", 0.52, 0.9), ("fish3", 0.48, 0.7)],
    }

    result = tmc.combine_fish_statistics(all_areas, fdr_alpha=0.05)

    assert set(result["area"]) == {"area_sig", "area_ns"}
    sig_row = result[result["area"] == "area_sig"].iloc[0]
    ns_row = result[result["area"] == "area_ns"].iloc[0]
    assert sig_row["is_significant"]
    assert not ns_row["is_significant"]
    assert sig_row["fdr_corrected"] < ns_row["fdr_corrected"]


# ---------------------------------------------------------------------------
# reorganize_results_by_area
# ---------------------------------------------------------------------------

def test_reorganize_results_by_area_inverts_fish_to_area_mapping():
    all_results = {
        "fish1": {"OB": {"accuracy_score": 0.8, "p_value": 0.01}},
        "fish2": {"OB": {"accuracy_score": 0.7, "p_value": 0.02}, "dHb": {"accuracy_score": 0.3, "p_value": 0.5}},
    }

    result = tmc.reorganize_results_by_area(all_results)

    assert set(result.keys()) == {"OB", "dHb"}
    assert result["OB"] == [("fish1", 0.8, 0.01), ("fish2", 0.7, 0.02)]
    assert result["dHb"] == [("fish2", 0.3, 0.5)]


# ---------------------------------------------------------------------------
# average_confusion_matrices_by_area
# ---------------------------------------------------------------------------

def test_average_confusion_matrices_by_area_row_normalizes_then_averages():
    all_conf_matrices = {
        "fish1": {"OB": {"confusion_matrix": np.array([[8.0, 2.0], [4.0, 6.0]])}},
        "fish2": {"OB": {"confusion_matrix": np.array([[4.0, 4.0], [1.0, 9.0]])}},
    }

    area_cm_avg, area_cm_n_fish = tmc.average_confusion_matrices_by_area(("OB",), all_conf_matrices)

    expected = np.mean(np.stack([
        np.array([[0.8, 0.2], [0.4, 0.6]]),
        np.array([[0.5, 0.5], [0.1, 0.9]]),
    ]), axis=0)
    assert np.allclose(area_cm_avg["OB"], expected)
    assert area_cm_n_fish["OB"] == 2


def test_average_confusion_matrices_by_area_skips_areas_with_no_data():
    area_cm_avg, area_cm_n_fish = tmc.average_confusion_matrices_by_area(("missing",), {"fish1": {}})

    assert area_cm_avg == {}
    assert area_cm_n_fish == {}


# ---------------------------------------------------------------------------
# load_valence_neurons
# ---------------------------------------------------------------------------

def test_load_valence_neurons_unions_significant_pos_and_neg(tmp_path):
    pos_path = tmp_path / "pos.csv"
    neg_path = tmp_path / "neg.csv"
    pd.DataFrame({"fish_id": ["f1_", "f2_"], "neuron_id": ["n1", "n2"], "p_fdr": [0.01, 0.5]}).to_csv(pos_path, index=False)
    pd.DataFrame({"fish_id": ["f3_", "f4_"], "neuron_id": ["n3", "n4"], "p_fdr": [0.5, 0.02]}).to_csv(neg_path, index=False)

    neurons = tmc.load_valence_neurons(pos_path, neg_path, fdr_alpha=0.05)

    assert neurons == {"f1_n1", "f4_n4"}


# ---------------------------------------------------------------------------
# plot_area_accuracy_scatter (smoke test)
# ---------------------------------------------------------------------------

def test_plot_area_accuracy_scatter_returns_figure():
    combined_df = pd.DataFrame({
        "area": ["OB", "dHb"], "mean_accuracy": [0.8, 0.3], "std_accuracy": [0.05, 0.1],
    })
    fish_key = "f1_fb_f1_hb"
    all_areas = {"OB": [(fish_key, 0.8, 0.01)], "dHb": [(fish_key, 0.3, 0.5)]}

    fig = tmc.plot_area_accuracy_scatter(
        combined_df, all_areas, fish_ids=(("f1_fb", "f1_hb"),), chance_level=1 / 7, title="Test",
    )

    assert isinstance(fig, plt.Figure)
    plt.close(fig)


# ---------------------------------------------------------------------------
# compute_odor_classification_results / compute_valence_classification_results
# (smoke tests against a tiny synthetic HDF5 file)
# ---------------------------------------------------------------------------

def _make_synthetic_response_hdf5(path, fish_ids, stimuli, n_neurons=15, n_trials=3, area="test_area"):
    rng = np.random.default_rng(42)
    rows = []
    for fish_id in fish_ids:
        for neuron_id in range(n_neurons):
            for stim_idx, stim in enumerate(stimuli):
                for trial in range(1, n_trials + 1):
                    rows.append({
                        "fish_id": fish_id,
                        "neuron_id": str(neuron_id),
                        "coords_serialised": "[0, 0, 0]",
                        "area": area,
                        "stimulus": stim,
                        "trial_number": trial,
                        "resp": rng.normal(loc=stim_idx, scale=0.1),
                    })
    pd.DataFrame(rows).to_hdf(path, key="data", mode="w", format="table", data_columns=["fish_id", "stimulus"])


def test_compute_odor_classification_results_runs_end_to_end(tmp_path):
    h5_path = tmp_path / "synthetic.h5"
    fish_ids = (("fishA_fb", "fishA_hb"),)
    stimuli = ("ade", "cad_2.5mm")
    _make_synthetic_response_hdf5(h5_path, ["fishA_fb", "fishA_hb"], stimuli)
    nmlf_mask = np.zeros((2, 2, 2))

    results, conf_matrices = tmc.compute_odor_classification_results(
        h5_path, nmlf_mask, fish_ids=fish_ids, ordered_stimuli=stimuli,
        min_neurons=5, n_folds=2, n_permutations=5,
    )

    fish_key = "fishA_fb_fishA_hb"
    assert fish_key in results
    assert "test_area" in results[fish_key]
    assert 0.0 <= results[fish_key]["test_area"]["accuracy_score"] <= 1.0
    assert 0.0 <= results[fish_key]["test_area"]["p_value"] <= 1.0
    assert conf_matrices[fish_key]["test_area"]["confusion_matrix"].shape == (2, 2)


def test_compute_valence_classification_results_runs_end_to_end(tmp_path):
    h5_path = tmp_path / "synthetic_valence.h5"
    fish_ids = (("fishA_fb", "fishA_hb"),)
    stimuli = ("ade", "fex_1")
    _make_synthetic_response_hdf5(h5_path, ["fishA_fb", "fishA_hb"], stimuli, n_neurons=10)
    nmlf_mask = np.zeros((2, 2, 2))
    valence_neurons = {f"fishA_fb{i}" for i in range(10)} | {f"fishA_hb{i}" for i in range(10)}

    results = tmc.compute_valence_classification_results(
        h5_path, nmlf_mask, valence_neurons, fish_ids=fish_ids,
        selected_stimuli=stimuli, valence_map={"ade": 0, "fex_1": 1},
        min_neurons=5, n_folds=2, n_permutations=5,
    )

    fish_key = "fishA_fb_fishA_hb"
    assert fish_key in results
    assert "test_area" in results[fish_key]
    assert 0.0 <= results[fish_key]["test_area"]["mean_score"] <= 1.0
