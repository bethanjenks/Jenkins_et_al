"""Regression test for jenkins_et_al.regression.bhv_vigour against a golden
capture derived from scripts/run_bhv_vigour_regression.py (commit 2d19f3a).

golden/bhv_vigour/results.csv was produced by running the ported module
against real data for a 2-fish subset (230713, 230714), after independently
verifying the ported module's output against the original script's own
functions called directly (not just re-run via this same module) -- see
PORTING_PLAN.md for the comparison methodology and the confirmed fex1/fex_1
spelling-mismatch bug fixed during this port (see
jenkins_et_al.regression.bhv_vigour module docstring).

Skipped entirely if the source data drive isn't mounted.
"""
from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from jenkins_et_al.regression import bhv_vigour

FB_BHV = Path("/Volumes/LaCie/larval_HuC/behavior/7dpf_fb_behavior_traces.csv")
HB_BHV = Path("/Volumes/LaCie/larval_HuC/behavior/7dpf_hb_behavior_traces.csv")
FB_NEURAL = Path("/Volumes/LaCie/larval_HuC/imaging/7dpf_fb_fish_dfs_final.h5")
HB_NEURAL = Path("/Volumes/LaCie/larval_HuC/imaging/7dpf_hb_fish_dfs_final.h5")
GOLDEN_DIR = Path(__file__).resolve().parent.parent.parent / "golden" / "bhv_vigour"
TEST_FISH = [230713, 230714]

pytestmark = pytest.mark.skipif(
    not FB_NEURAL.exists() or not GOLDEN_DIR.exists(),
    reason="source data drive or golden capture not available on this machine",
)


@pytest.fixture(scope="module")
def results():
    return bhv_vigour.run_bhv_vigour_regression(
        fish_list=TEST_FISH,
        neuro_stimuli=list(bhv_vigour.NEURO_STIMULI),
        fb_bhv_path=FB_BHV,
        hb_bhv_path=HB_BHV,
        fb_neural_path=FB_NEURAL,
        hb_neural_path=HB_NEURAL,
    )


def test_matches_golden(results):
    golden = pd.read_csv(GOLDEN_DIR / "results.csv")
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
