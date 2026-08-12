"""Unit tests for jenkins_et_al.neural.lda_svm_classification against small
synthetic inputs (no dependency on the real data drive).

Per the user's decision (2026-08-12), this port is NOT golden-captured: its
permutation test has no seed anywhere in the notebook (`sklearn.utils.shuffle(
y, random_state=None)`), so exact reproduction is impossible even in
principle, and a full real-data run (10 fish x ~36 areas x 1000 permutations
x 2 classifiers) was timed at ~2.5-3 hours -- the same call already made for
#9 (template_matching_classification)'s near-identical situation. Every
function is instead verified here against small deterministic synthetic data.
A real-data smoke test (a handful of areas, reduced permutations) lives in
tests/golden/test_lda_svm_classification.py, skipped if the drive isn't
mounted.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from jenkins_et_al.neural import lda_svm_classification as lsc


def _make_separable_classes(n_classes: int = 3, n_trials: int = 6, n_neurons: int = 8, seed: int = 0):
    """Neurons with a distinct per-class response pattern -- trivially classifiable.

    Each class gets its own random center in neuron-space (not a single scalar
    shift applied identically to every neuron, which would make all features
    collinear and confuse LinearSVC's optimization despite "obvious"
    separation).
    """
    rng = np.random.default_rng(seed)
    labels = [f"stim{i}" for i in range(n_classes)]
    class_centers = rng.normal(scale=10.0, size=(n_classes, n_neurons))

    X_rows, y_rows = [], []
    for i, label in enumerate(labels):
        trials = class_centers[i] + rng.normal(scale=0.5, size=(n_trials, n_neurons))
        X_rows.append(trials)
        y_rows.extend([label] * n_trials)
    return np.vstack(X_rows), np.array(y_rows)


def test_load_fish_stimulus_data_filters_and_parses_coords(tmp_path):
    h5_path = tmp_path / "data.h5"
    df = pd.DataFrame({
        "fish_id": ["fish1_fb", "fish1_fb", "fish1_hb", "fish1_fb"],
        "stimulus": ["ade", "kw", "ade", "not_in_key"],
        "area": ["a", "a", "b", "a"],
        "neuron_id": [1, 2, 1, 3],
        "trial_number": [1, 1, 1, 1],
        "resp": [0.1, 0.2, 0.3, 0.4],
        "coords_serialised": ["[1.0, 2.0, 3.0]"] * 4,
    })
    df.to_hdf(h5_path, key="data", format="table", data_columns=["fish_id"])

    out = lsc.load_fish_stimulus_data(h5_path, "fish1_fb", "fish1_hb", key_stimuli=("ade", "kw"))

    assert set(out["stimulus"]) == {"ade", "kw"}
    assert "coords_serialised" not in out.columns
    assert out["coords"].iloc[0].tolist() == [1.0, 2.0, 3.0]


def test_pivot_area_response_matrix_shapes_and_drops_incomplete_neurons():
    stimulus_df = pd.DataFrame({
        "fish_id": ["f1"] * 6,
        "area": ["a"] * 4 + ["b"] * 2,
        "neuron_id": [1, 1, 2, 2, 3, 3],
        "stimulus": ["s1", "s2", "s1", "s2", "s1", "s2"],
        "trial_number": [1, 1, 1, 1, 1, 1],
        "resp": [0.1, 0.2, 0.3, np.nan, 0.5, 0.6],
    })
    X, y = lsc.pivot_area_response_matrix(stimulus_df, "a")

    # Neuron 2 has a NaN trial and is dropped; only neuron 1 survives -> 1 feature (neuron), 2 samples (trials).
    assert X.shape == (2, 1)
    assert sorted(y) == ["s1", "s2"]


def test_run_lda_classification_separates_well_separated_classes():
    X, y = _make_separable_classes(n_classes=3, n_trials=6, n_neurons=8)
    result = lsc.run_lda_classification(X, y, n_pca=3, n_folds=3, random_state=0)

    assert result["mean_score"] > 0.9
    assert result["confusion_matrix"].shape == (3, 3)
    assert result["labels"] == ["stim0", "stim1", "stim2"]
    # Perfect full-data refit on well-separated classes -> diagonal confusion matrix.
    assert np.array_equal(np.diag(result["confusion_matrix"]), result["confusion_matrix"].sum(axis=1))


def test_run_svm_classification_separates_well_separated_classes():
    X, y = _make_separable_classes(n_classes=3, n_trials=6, n_neurons=8)
    result = lsc.run_svm_classification(X, y, n_folds=3, random_state=0)

    assert result["mean_score"] > 0.9
    assert result["confusion_matrix"].shape == (3, 3)


def test_permutation_test_classifier_gives_low_p_for_real_separation():
    X, y = _make_separable_classes(n_classes=3, n_trials=9, n_neurons=8)
    p = lsc.permutation_test_classifier(
        X, y, classifier_func=lsc.run_lda_classification, n_permutations=30,
        random_state=0, n_pca=3, n_folds=3,
    )
    assert p < 0.1


def test_permutation_test_classifier_has_no_plus_one_correction():
    # Constant-accuracy classifier: every permutation ties the observed score exactly,
    # so p must be exactly 1.0 (sum(null >= obs) / n_perm), not shifted by a +1 correction.
    def constant_classifier(X, y):
        return {"mean_score": 0.5}

    X, y = _make_separable_classes(n_classes=2, n_trials=4, n_neurons=2)
    p = lsc.permutation_test_classifier(X, y, classifier_func=constant_classifier, n_permutations=10)
    assert p == 1.0


def test_permutation_test_classifier_random_state_is_reproducible_but_varies_within_a_run():
    X, y = _make_separable_classes(n_classes=2, n_trials=6, n_neurons=4)

    seen_shuffles = []

    def recording_classifier(X, y_perm, **kwargs):
        seen_shuffles.append(tuple(y_perm))
        return {"mean_score": 0.5}

    lsc.permutation_test_classifier(X, y, classifier_func=recording_classifier, n_permutations=5, shuffle_random_state=42)
    first_run = list(seen_shuffles)

    seen_shuffles.clear()
    lsc.permutation_test_classifier(X, y, classifier_func=recording_classifier, n_permutations=5, shuffle_random_state=42)
    second_run = list(seen_shuffles)

    assert first_run == second_run  # reproducible given the same random_state
    assert len(set(first_run)) > 1  # but successive permutations within a run differ


def test_run_classification_pipeline_skips_small_areas_and_organizes_by_fish(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)

    def fake_load(neural_data_path, fb_id, hb_id, key_stimuli):
        rows = []
        for area, n_neurons in [("big", 6), ("small", 2)]:
            class_centers = rng.normal(scale=10.0, size=(2, n_neurons))  # one center per stimulus
            for trial in range(1, 5):
                for stim_idx, stim in enumerate(["s1", "s2"]):
                    trial_resp = class_centers[stim_idx] + rng.normal(scale=0.5, size=n_neurons)
                    for neuron, resp in enumerate(trial_resp):
                        rows.append({
                            "fish_id": fb_id, "area": area, "neuron_id": neuron,
                            "stimulus": stim, "trial_number": trial, "resp": resp,
                        })
        return pd.DataFrame(rows)

    monkeypatch.setattr(lsc, "load_fish_stimulus_data", fake_load)

    all_results, all_confusion_matrices = lsc.run_classification_pipeline(
        "unused.h5", [("f1_fb", "f1_hb")], key_stimuli=("s1", "s2"),
        classifier_func=lsc.run_lda_classification, classifier_kwargs={"n_pca": 2, "n_folds": 2, "random_state": 0},
        n_permutations=5, min_neurons=5, permutation_random_state=0,
    )

    fish_key = "f1_fb_f1_hb"
    assert set(all_results[fish_key].keys()) == {"big"}  # "small" (2 neurons) is below min_neurons=5
    assert 0.0 <= all_results[fish_key]["big"]["p_value"] <= 1.0
    assert "big" in all_confusion_matrices[fish_key]


def test_reorganize_by_area_groups_by_area_across_fish():
    all_results = {
        "fishA": {"area1": {"mean_score": 0.5, "p_value": 0.01, "n_neurons": 10}},
        "fishB": {"area1": {"mean_score": 0.6, "p_value": 0.02, "n_neurons": 12}, "area2": {"mean_score": 0.3, "p_value": 0.5, "n_neurons": 8}},
    }
    by_area = lsc.reorganize_by_area(all_results)

    assert set(by_area.keys()) == {"area1", "area2"}
    assert by_area["area1"] == [("fishA", 0.5, 0.01), ("fishB", 0.6, 0.02)]
    assert by_area["area2"] == [("fishB", 0.3, 0.5)]


def test_combine_fish_statistics_flags_significant_areas():
    all_areas = {
        "strong": [("f1", 0.8, 0.001), ("f2", 0.75, 0.002), ("f3", 0.82, 0.001)],
        "weak": [("f1", 0.2, 0.9), ("f2", 0.25, 0.8), ("f3", 0.18, 0.95)],
    }
    df = lsc.combine_fish_statistics(all_areas, fdr_alpha=0.05)

    assert set(df.columns) == {"area", "mean_accuracy", "std_accuracy", "combined_p", "fdr_corrected", "is_significant", "n_fish"}
    strong_row = df.set_index("area").loc["strong"]
    weak_row = df.set_index("area").loc["weak"]
    assert strong_row["is_significant"]
    assert not weak_row["is_significant"]
    assert strong_row["combined_p"] < weak_row["combined_p"]
    # Sorted ascending by fdr_corrected (most significant first).
    assert df.iloc[0]["area"] == "strong"


def test_combine_fish_statistics_preserves_bare_except_fallback_path(monkeypatch):
    # The notebook's own combine_fish_statistics wraps combine_pvalues in a bare
    # except, falling back to combined_p = 1.0 on any failure. scipy's current
    # combine_pvalues doesn't actually raise on the degenerate inputs available
    # here (it warns and returns NaN instead) -- exercise the fallback directly
    # by forcing combine_pvalues to raise, confirming the port's except still
    # catches it and substitutes 1.0 rather than propagating.
    def raising_combine_pvalues(p_values, method):
        raise ValueError("forced failure")

    monkeypatch.setattr(lsc, "combine_pvalues", raising_combine_pvalues)

    all_areas = {"area_a": [("f1", 0.5, 0.2), ("f2", 0.6, 0.3)]}
    df = lsc.combine_fish_statistics(all_areas, fdr_alpha=0.05)
    assert df.loc[df["area"] == "area_a", "combined_p"].iloc[0] == 1.0


def test_compare_methods_picks_the_higher_mean_accuracy():
    df_lda = pd.DataFrame({"mean_accuracy": [0.8, 0.6], "is_significant": [True, False]})
    df_svm = pd.DataFrame({"mean_accuracy": [0.5, 0.4], "is_significant": [False, False]})

    result = lsc.compare_methods(df_lda, df_svm)

    assert result["winner"] == "lda"
    assert result["lda_n_significant"] == 1
    assert result["svm_n_significant"] == 0


def test_aggregate_confusion_matrices_sums_and_averages():
    key_stimuli = ("s1", "s2")
    all_cm = {
        "fish1": {"a": {"confusion_matrix": np.array([[2.0, 0.0], [0.0, 2.0]])}},
        "fish2": {"a": {"confusion_matrix": np.array([[4.0, 0.0], [0.0, 0.0]])}},
    }
    avg, n = lsc.aggregate_confusion_matrices(all_cm, key_stimuli)

    assert n == 2
    assert np.allclose(avg, [[3.0, 0.0], [0.0, 1.0]])


def test_aggregate_confusion_matrices_per_fish_averages_across_areas():
    key_stimuli = ("s1", "s2")
    fish_conf_dict = {
        "a": {"confusion_matrix": np.array([[2.0, 0.0], [0.0, 2.0]])},
        "b": {"confusion_matrix": np.array([[0.0, 2.0], [2.0, 0.0]])},
    }
    avg, n_areas = lsc.aggregate_confusion_matrices_per_fish(fish_conf_dict, key_stimuli)

    assert n_areas == 2
    assert np.allclose(avg, [[1.0, 1.0], [1.0, 1.0]])


def test_filter_significant_confusion_matrices_restricts_to_named_areas():
    key_stimuli = ("s1", "s2")
    all_cm = {
        "fish1": {
            "sig_area": {"confusion_matrix": np.array([[2.0, 0.0], [0.0, 2.0]])},
            "nonsig_area": {"confusion_matrix": np.array([[0.0, 4.0], [4.0, 0.0]])},
        },
    }
    avg, n = lsc.filter_significant_confusion_matrices(all_cm, significant_areas=["sig_area"], key_stimuli=key_stimuli)

    assert n == 1
    assert np.allclose(avg, [[2.0, 0.0], [0.0, 2.0]])


def test_analyze_confusion_pairs_ranks_and_splits_by_valence():
    labels = ["pos1", "pos2", "neg1"]
    # Raw counts, row-normalized internally: pos1->pos2 (0.4) is the top confusion,
    # pos2->pos1 (0.3) is second; neg1's row has no off-diagonal confusion at all.
    cm = np.array([
        [5.0, 4.0, 1.0],
        [3.0, 6.0, 1.0],
        [0.0, 0.0, 10.0],
    ])
    result = lsc.analyze_confusion_pairs(cm, labels, attractive=["pos1", "pos2"], aversive=["neg1"])

    assert result["top_confusions"][0][:2] == ("pos1", "pos2")
    assert np.isclose(result["top_confusions"][0][2], 0.4)
    assert result["within_attr_avg"] > result["cross_avg"]
    assert result["within_aver_avg"] == 0  # only one aversive label -> no within-aversive pairs


def test_jitter_perturbs_values_within_amount():
    values = np.zeros(1000)
    jittered = lsc.jitter(values, amount=0.05)
    assert np.all(np.abs(jittered) <= 0.05)
    assert not np.allclose(jittered, 0.0)


def test_plotting_functions_render_figures():
    key_stimuli = ("s1", "s2", "s3")
    cm = np.array([[5.0, 1.0, 0.0], [1.0, 5.0, 0.0], [0.0, 0.0, 5.0]])

    fig1 = lsc.plot_aggregate_confusion_matrix(cm, key_stimuli, title="Aggregate")

    all_cm = {
        "f1_fb_f1_hb": {"a": {"confusion_matrix": cm}},
        "f2_fb_f2_hb": {"a": {"confusion_matrix": cm}},
    }
    fish_ids = [("f1_fb", "f1_hb"), ("f2_fb", "f2_hb")]
    fig2 = lsc.plot_per_fish_confusion_matrices(all_cm, fish_ids, key_stimuli, suptitle="Per fish")

    for fig in (fig1, fig2):
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
