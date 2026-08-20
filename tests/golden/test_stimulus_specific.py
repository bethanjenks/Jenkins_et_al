"""Regression test for jenkins_et_al.regression.stimulus_specific against a
golden capture derived from scripts/run_stimulus_specific_regression.py
(commit 2d19f3a).

golden/stimulus_specific/ade_cad_2.5mm_fex_1.csv was produced by running the
ported module against real data for a 2-fish subset (230713, 230714) and one
of the twenty pickled stimulus regressors, after independently verifying the
ported module's output against the original script's own functions called
directly. Only one stimulus is exercised here -- the shared correlation/
z-score/shuffle core is already covered by the other three regression
scripts' golden tests, and this script's own per-stimulus loop has no logic
that varies by which regressor is used.

Skipped entirely if the source data drive isn't mounted.
"""
import pickle
from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from jenkins_et_al.regression import stimulus_specific

FB_NEURAL = Path("/Volumes/LaCie/larval_HuC/imaging/7dpf_fb_fish_dfs_final.h5")
HB_NEURAL = Path("/Volumes/LaCie/larval_HuC/imaging/7dpf_hb_fish_dfs_final.h5")
REGRESSOR_PKL = Path("/Volumes/LaCie/larval_HuC/combination_regressors_no_HCl_25s.pkl")
GOLDEN_DIR = Path(__file__).resolve().parent.parent.parent / "golden" / "stimulus_specific"
TEST_FISH = [230713, 230714]
TEST_STIMULUS = "ade_cad_2.5mm_fex_1"

pytestmark = pytest.mark.skipif(
    not FB_NEURAL.exists() or not GOLDEN_DIR.exists(),
    reason="source data drive or golden capture not available on this machine",
)


@pytest.fixture(scope="module")
def results():
    with REGRESSOR_PKL.open("rb") as fh:
        stim_regressors = pickle.load(fh)
    regressor = stim_regressors[TEST_STIMULUS]

    rows = []
    for fish in TEST_FISH:
        for path, end in [(FB_NEURAL, "_fb"), (HB_NEURAL, "_hb")]:
            neural = stimulus_specific.load_neural_data(path, str(fish), end=end)
            r = stimulus_specific.process_one_dataset(
                neural_data=neural,
                regressor=regressor,
                neuro_stimuli=list(stimulus_specific.NEURO_STIMULI),
                dataset_name=end,
                num_shuffles=1000,
                trial_length=300,
                random_seed=42,
            )
            if not r.empty:
                rows.append(r)
    return pd.concat(rows, ignore_index=True)


def test_matches_golden(results):
    golden = pd.read_csv(GOLDEN_DIR / f"{TEST_STIMULUS}.csv")
    ported = results.sort_values(["fish_id", "neuron_id"]).reset_index(drop=True)
    golden = golden.sort_values(["fish_id", "neuron_id"]).reset_index(drop=True)

    assert ported.shape == golden.shape
    assert np.allclose(ported["correlation"], golden["correlation"], atol=1e-8)
    assert np.allclose(ported["p_value"], golden["p_value"], atol=1e-8)
    assert (ported["fish_id"].astype(str) == golden["fish_id"].astype(str)).all()
    assert (ported["neuron_id"].astype(str) == golden["neuron_id"].astype(str)).all()
    assert (ported["area"].astype(str) == golden["area"].astype(str)).all()

    golden_coords = golden["coords"].apply(literal_eval)
    assert all(
        list(p) == list(g) for p, g in zip(ported["coords"].values, golden_coords.values)
    )
