"""Regression test for jenkins_et_al.regression.single_stimulus against a
golden capture derived from scripts/run_single_stimulus_regression.py
(commit 2d19f3a).

golden/single_stimulus/ was produced by running the ported module against
real data for a 2-fish subset (230710, 230711) and three of the fifteen
stimuli, after independently verifying the ported module's output against
the original script's own functions called directly, fed the confirmed fix
(the sole available regressor reused for every stimulus, instead of the
original's per-stimulus-name lookup that can't work at all -- see
jenkins_et_al.regression.single_stimulus module docstring). Only three
stimuli are exercised here since the same `process_region_for_stimulus`
logic runs for all fifteen; the loop itself has no stimulus-dependent
behavior.

Skipped entirely if the source data drive isn't mounted.
"""
import pickle
from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from jenkins_et_al.regression import single_stimulus

FB_NEURAL = Path("/Volumes/LaCie/larval_HuC/imaging/4dpf/4dpf_fb_fish_dfs_final.h5")
HB_NEURAL = Path("/Volumes/LaCie/larval_HuC/imaging/4dpf/4dpf_hb_fish_dfs_final.h5")
REGRESSOR_PKL = Path("/Volumes/LaCie/larval_HuC/regressors/single_regressor_sustained_25s.pkl")
GOLDEN_DIR = Path(__file__).resolve().parent.parent.parent / "golden" / "single_stimulus"
TEST_FISH = [230710, 230711]
TEST_STIMULI = ["ade", "cad_2.5mm", "fex_1"]

pytestmark = pytest.mark.skipif(
    not FB_NEURAL.exists() or not GOLDEN_DIR.exists(),
    reason="source data drive or golden capture not available on this machine",
)


@pytest.fixture(scope="module")
def results():
    with REGRESSOR_PKL.open("rb") as fh:
        stim_regressors = pickle.load(fh)
    return single_stimulus.run_analysis(
        fish_list=TEST_FISH,
        stim_regressors=stim_regressors,
        stimuli=TEST_STIMULI,
        fb_neural_path=FB_NEURAL,
        hb_neural_path=HB_NEURAL,
    )


def _assert_matches_golden(ported: pd.DataFrame, golden_csv: Path, corr_col: str) -> None:
    golden = pd.read_csv(golden_csv)
    ported = ported.sort_values(["fish_id", "neuron_id"]).reset_index(drop=True)
    golden = golden.sort_values(["fish_id", "neuron_id"]).reset_index(drop=True)

    assert ported.shape == golden.shape
    assert np.allclose(ported[corr_col], golden[corr_col], atol=1e-10, equal_nan=True)
    assert (ported["fish_id"].astype(str) == golden["fish_id"].astype(str)).all()
    assert (ported["neuron_id"].astype(str) == golden["neuron_id"].astype(str)).all()
    assert (ported["area"].astype(str) == golden["area"].astype(str)).all()

    golden_coords = golden["coords"].apply(literal_eval)
    assert all(
        list(p) == list(g) for p, g in zip(ported["coords"].values, golden_coords.values)
    )


@pytest.mark.parametrize("stimulus", TEST_STIMULI)
def test_per_stimulus_matches_golden(results, stimulus):
    stimulus_dfs, _combined_df = results
    _assert_matches_golden(
        stimulus_dfs[stimulus], GOLDEN_DIR / f"{stimulus}.csv", f"{stimulus}_correlation",
    )


def test_combined_matches_golden(results):
    _, combined_df = results
    golden = pd.read_csv(GOLDEN_DIR / "combined.csv")
    ported = combined_df.sort_values(["fish_id", "neuron_id"]).reset_index(drop=True)
    golden = golden.sort_values(["fish_id", "neuron_id"]).reset_index(drop=True)

    assert ported.shape == golden.shape
    assert list(ported.columns) == list(golden.columns)
    for stimulus in TEST_STIMULI:
        col = f"{stimulus}_correlation"
        assert np.allclose(ported[col], golden[col], atol=1e-10, equal_nan=True)
