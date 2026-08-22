"""Regression test for jenkins_et_al.neural.wholebrain_traces against a
golden capture derived from
notebooks/neural/wholebrain_avg_traces_PCA_raster.ipynb (forebrain only --
see module docstring in wholebrain_traces.py for why hindbrain isn't part
of this notebook's real analysis despite being loaded).

golden/wholebrain_traces/ was produced by running the ported module against
real forebrain data (pallium, 13453 neurons), after independently verifying
it against the notebook's own cell code (with the user-agreed neuron-count
divisor and stimulus-identifier fixes applied), copied verbatim and run
standalone on identical inputs.

Skipped entirely if the source data drive isn't mounted.
"""
from pathlib import Path

import json
import numpy as np
import pytest

from jenkins_et_al.neural.wholebrain_traces import (
    KEY_STIMULI,
    NEGATIVE_STIMULI,
    POSITIVE_STIMULI,
    calculate_area_traces,
    compute_group_average,
    count_neurons_in_area,
    load_forebrain_data,
)

FB_PATH = Path("/Volumes/LaCie/larval_HuC/imaging/7dpf_fb_fish_dfs_final.h5")
GOLDEN_DIR = Path(__file__).resolve().parent.parent.parent / "golden" / "wholebrain_traces"

pytestmark = pytest.mark.skipif(
    not FB_PATH.exists() or not GOLDEN_DIR.exists(),
    reason="source data drive or golden capture not available on this machine",
)


@pytest.fixture(scope="module")
def area_results():
    stim_fb = load_forebrain_data(FB_PATH)
    area_df = stim_fb[stim_fb["area"] == "pallium"]
    neuron_count = count_neurons_in_area(area_df)
    traces = calculate_area_traces(area_df)
    mean_pos, sem_pos = compute_group_average(traces["mean"], traces["sem"], POSITIVE_STIMULI)
    mean_neg, sem_neg = compute_group_average(traces["mean"], traces["sem"], NEGATIVE_STIMULI)
    return neuron_count, traces, (mean_pos, sem_pos, mean_neg, sem_neg)


def test_neuron_count_matches_golden(area_results):
    neuron_count, _, _ = area_results
    golden = json.loads((GOLDEN_DIR / "neuron_count.json").read_text())
    assert neuron_count == golden["neuron_count"]


def test_area_traces_match_golden(area_results):
    _, traces, _ = area_results
    golden = np.load(GOLDEN_DIR / "traces.npz")
    for stim in KEY_STIMULI:
        assert np.allclose(traces["mean"][stim], golden[f"mean_{stim}"], atol=1e-10)
        assert np.allclose(traces["sem"][stim], golden[f"sem_{stim}"], atol=1e-10)


def test_valence_comparison_matches_golden(area_results):
    _, _, (mean_pos, sem_pos, mean_neg, sem_neg) = area_results
    golden = np.load(GOLDEN_DIR / "traces.npz")
    assert np.allclose(mean_pos, golden["mean_pos"], atol=1e-10)
    assert np.allclose(sem_pos, golden["sem_pos"], atol=1e-10)
    assert np.allclose(mean_neg, golden["mean_neg"], atol=1e-10)
    assert np.allclose(sem_neg, golden["sem_neg"], atol=1e-10)
