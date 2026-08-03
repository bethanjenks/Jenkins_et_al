"""Regression test for jenkins_et_al.behaviour.bout_frequency_psth against a
golden capture of notebooks/behaviour_head_fixed/bout_frequency_PSTH.ipynb
(commit 2d19f3a).

The golden capture (golden/bout_frequency_PSTH/) was produced by executing
the notebook's own cell source directly (fresh top-to-bottom run; confirmed
this completes without error -- the saved notebook's NameError on
`straight_bout_valence_analysis` is a stale out-of-order-execution artifact,
not a real bug), redirecting only mean/SEM/valence-results tables to CSVs
under golden/bout_frequency_PSTH/ (the notebook itself never saves these,
only plots/prints them).

Skipped entirely if the source data drive isn't mounted, since the underlying
behaviour-traces-posture CSVs are not part of this repository.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from jenkins_et_al.behaviour import bout_frequency_psth
from jenkins_et_al.config import STIMULUS_GROUPS

FB_PATH = Path("/Volumes/LaCie/larval_HuC/behavior/7dpf_fb_behavior_traces_posture.csv")
HB_PATH = Path("/Volumes/LaCie/larval_HuC/behavior/7dpf_hb_behavior_traces_posture.csv")
GOLDEN_DIR = Path(__file__).resolve().parent.parent.parent / "golden" / "bout_frequency_PSTH"

pytestmark = pytest.mark.skipif(
    not FB_PATH.exists() or not HB_PATH.exists() or not GOLDEN_DIR.exists(),
    reason="source data drive or golden capture not available on this machine",
)


@pytest.fixture(scope="module")
def results():
    return bout_frequency_psth.run_bout_frequency_psth_analysis(
        fb_path=FB_PATH,
        hb_path=HB_PATH,
        key_stimuli=STIMULUS_GROUPS["high"],
    )


@pytest.mark.parametrize("bout_type", ["all", "asymmetric", "straight"])
def test_psth_mean_matches_golden(results, bout_type):
    mean_df = pd.DataFrame(results[bout_type]["psth"]["mean_data"])
    mean_df.index.name = "time_bin"

    golden = pd.read_csv(GOLDEN_DIR / f"{bout_type}_bout_psth_mean.csv", index_col="time_bin")

    assert list(mean_df.columns) == list(golden.columns)
    assert np.allclose(
        mean_df.to_numpy(dtype=float), golden.to_numpy(dtype=float),
        rtol=1e-8, atol=1e-10, equal_nan=True,
    )


@pytest.mark.parametrize("bout_type", ["all", "asymmetric", "straight"])
def test_psth_sem_matches_golden(results, bout_type):
    sem_df = pd.DataFrame(results[bout_type]["psth"]["sem_data"])
    sem_df.index.name = "time_bin"

    golden = pd.read_csv(GOLDEN_DIR / f"{bout_type}_bout_psth_sem.csv", index_col="time_bin")

    assert list(sem_df.columns) == list(golden.columns)
    assert np.allclose(
        sem_df.to_numpy(dtype=float), golden.to_numpy(dtype=float),
        rtol=1e-8, atol=1e-10, equal_nan=True,
    )


@pytest.mark.parametrize("bout_type", ["all", "asymmetric", "straight"])
def test_valence_results_df_matches_golden(results, bout_type):
    ported = results[bout_type]["valence_results_df"]
    golden = pd.read_csv(GOLDEN_DIR / f"{bout_type}_bout_valence_trajectory.csv")

    assert list(ported.columns) == list(golden.columns)
    assert np.allclose(
        ported.to_numpy(dtype=float), golden.to_numpy(dtype=float),
        rtol=1e-8, atol=1e-10, equal_nan=True,
    )
