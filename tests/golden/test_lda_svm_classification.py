"""Real-data smoke test for jenkins_et_al.neural.lda_svm_classification.

Per the user's decision (2026-08-12), this port is NOT golden-captured: the
notebook's own permutation test has no seed anywhere
(`sklearn.utils.shuffle(y, random_state=None)`), so exact reproduction is
impossible even in principle, and a full real-data run (10 fish x ~36 areas
x 1000 permutations x 2 classifiers) was timed at ~2.5-3 hours -- the same
call already made for #9 (template_matching_classification)'s near-identical
situation.

This instead runs the real pipeline end-to-end against real data for a small
subset (2 fish, every qualifying area, `n_permutations` reduced to 20) and
checks it completes without error and lands in a sane range -- not an exact
match to any reference. Every function is exactly verified against small
deterministic synthetic data in tests/test_lda_svm_classification.py.

Skipped entirely if the source data drive isn't mounted.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jenkins_et_al.neural import lda_svm_classification as lsc

NEURAL_DATA_PATH = Path("/Volumes/LaCie/larval_HuC/imaging/7dpf_wb_population_general_response_vector.h5")

KEY_STIMULI = ("ade", "pro_25um", "qui_25um", "cad_25um", "fex_3", "kw", "ph4.5")
ATTRACTIVE = ["fex_3", "kw", "pro_25um"]
AVERSIVE = ["ade", "cad_25um", "qui_25um", "ph4.5"]
SMOKE_FISH_IDS = [["230713_fb", "230713_hb"], ["230714_fb", "230714_hb"]]
N_PERMUTATIONS_SMOKE = 20

pytestmark = pytest.mark.skipif(
    not NEURAL_DATA_PATH.exists(), reason="source data drive not available on this machine",
)


@pytest.fixture(scope="module")
def lda_pipeline_result():
    return lsc.run_classification_pipeline(
        NEURAL_DATA_PATH, SMOKE_FISH_IDS, KEY_STIMULI,
        classifier_func=lsc.run_lda_classification,
        classifier_kwargs={"n_pca": 5, "n_folds": 3},
        n_permutations=N_PERMUTATIONS_SMOKE,
    )


@pytest.fixture(scope="module")
def svm_pipeline_result():
    return lsc.run_classification_pipeline(
        NEURAL_DATA_PATH, SMOKE_FISH_IDS, KEY_STIMULI,
        classifier_func=lsc.run_svm_classification,
        classifier_kwargs={"n_folds": 3},
        n_permutations=N_PERMUTATIONS_SMOKE,
    )


MIN_NEURONS_SMOKE = 5  # matches run_classification_pipeline's own default, used unchanged in the fixtures above.


def _check_pipeline_result_is_sane(all_results, all_confusion_matrices, n_stimuli):
    assert set(all_results.keys()) == {"230713_fb_230713_hb", "230714_fb_230714_hb"}

    for fish_key, fish_results in all_results.items():
        assert len(fish_results) > 0  # at least some areas qualify (>= min_neurons)
        for area, area_result in fish_results.items():
            assert 0.0 <= area_result["mean_score"] <= 1.0
            assert 0.0 <= area_result["p_value"] <= 1.0
            assert area_result["n_neurons"] >= MIN_NEURONS_SMOKE

            cm = all_confusion_matrices[fish_key][area]["confusion_matrix"]
            assert cm.shape == (n_stimuli, n_stimuli)
            assert cm.sum() > 0


def test_lda_pipeline_runs_end_to_end_on_real_data(lda_pipeline_result):
    all_results, all_confusion_matrices = lda_pipeline_result
    _check_pipeline_result_is_sane(all_results, all_confusion_matrices, n_stimuli=len(KEY_STIMULI))


def test_svm_pipeline_runs_end_to_end_on_real_data(svm_pipeline_result):
    all_results, all_confusion_matrices = svm_pipeline_result
    _check_pipeline_result_is_sane(all_results, all_confusion_matrices, n_stimuli=len(KEY_STIMULI))


def test_lda_mean_accuracy_beats_chance_on_real_data(lda_pipeline_result):
    all_results, _ = lda_pipeline_result
    all_areas = lsc.reorganize_by_area(all_results)
    df = lsc.combine_fish_statistics(all_areas, fdr_alpha=0.05)

    chance = 1 / len(KEY_STIMULI)
    assert df["mean_accuracy"].mean() > chance


def test_svm_mean_accuracy_beats_chance_on_real_data(svm_pipeline_result):
    all_results, _ = svm_pipeline_result
    all_areas = lsc.reorganize_by_area(all_results)
    df = lsc.combine_fish_statistics(all_areas, fdr_alpha=0.05)

    chance = 1 / len(KEY_STIMULI)
    assert df["mean_accuracy"].mean() > chance


def test_confusion_matrix_aggregation_and_analysis_run_on_real_data(lda_pipeline_result):
    _, all_confusion_matrices = lda_pipeline_result

    avg_cm, n_matrices = lsc.aggregate_confusion_matrices(all_confusion_matrices, KEY_STIMULI)
    assert n_matrices > 0
    assert avg_cm.sum() > 0

    analysis = lsc.analyze_confusion_pairs(avg_cm, list(KEY_STIMULI), ATTRACTIVE, AVERSIVE)
    assert len(analysis["top_confusions"]) == min(10, len(KEY_STIMULI) * (len(KEY_STIMULI) - 1))


def test_compare_methods_runs_on_real_data(lda_pipeline_result, svm_pipeline_result):
    lda_all_results, _ = lda_pipeline_result
    svm_all_results, _ = svm_pipeline_result

    df_lda = lsc.combine_fish_statistics(lsc.reorganize_by_area(lda_all_results), fdr_alpha=0.05)
    df_svm = lsc.combine_fish_statistics(lsc.reorganize_by_area(svm_all_results), fdr_alpha=0.05)

    comparison = lsc.compare_methods(df_lda, df_svm)
    assert comparison["winner"] in {"lda", "svm", "tie"}


def test_plotting_functions_render_on_real_data(lda_pipeline_result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, all_confusion_matrices = lda_pipeline_result
    avg_cm, _ = lsc.aggregate_confusion_matrices(all_confusion_matrices, KEY_STIMULI)

    fig1 = lsc.plot_aggregate_confusion_matrix(avg_cm, list(KEY_STIMULI), title="Smoke test")
    assert isinstance(fig1, plt.Figure)
    plt.close(fig1)
