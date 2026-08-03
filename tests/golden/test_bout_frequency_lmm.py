"""Regression test for jenkins_et_al.behaviour.bout_frequency_lmm against a
golden capture of notebooks/behaviour_head_fixed/bout_frequency_LMM.ipynb
(commit 2d19f3a).

The golden capture (golden/bout_frequency_LMM/) was produced by executing the
notebook's own cell source directly (fresh top-to-bottom run), redirecting
only the output directory and the paired-dot-plot's hardcoded savefig path to
golden/bout_frequency_LMM/ instead of the notebook's own absolute paths
(under /Users/bethanjenkins/Documents/valence_paper_code/). No logic or data
change.

Skipped entirely if the source data drive isn't mounted, since the underlying
behaviour-traces-posture CSVs are not part of this repository.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from jenkins_et_al.behaviour import bout_frequency_lmm
from jenkins_et_al.config import STIMULUS_GROUPS

FB_PATH = Path("/Volumes/LaCie/larval_HuC/behavior/7dpf_fb_behavior_traces_posture.csv")
HB_PATH = Path("/Volumes/LaCie/larval_HuC/behavior/7dpf_hb_behavior_traces_posture.csv")
GOLDEN_DIR = Path(__file__).resolve().parent.parent.parent / "golden" / "bout_frequency_LMM"

pytestmark = pytest.mark.skipif(
    not FB_PATH.exists() or not HB_PATH.exists() or not GOLDEN_DIR.exists(),
    reason="source data drive or golden capture not available on this machine",
)


@pytest.fixture(scope="module")
def results():
    return bout_frequency_lmm.run_bout_frequency_analysis(
        fb_path=FB_PATH,
        hb_path=HB_PATH,
        key_stimuli=STIMULUS_GROUPS["low"],
    )


def _assert_matches_golden(ported: pd.DataFrame, golden_csv: Path) -> None:
    golden = pd.read_csv(golden_csv)

    assert list(ported.columns) == list(golden.columns)
    assert len(ported) == len(golden)

    numeric_cols = golden.select_dtypes(include=[np.number]).columns
    object_cols = [c for c in golden.columns if c not in numeric_cols]

    assert np.allclose(
        ported[numeric_cols].to_numpy(dtype=float),
        golden[numeric_cols].to_numpy(dtype=float),
        rtol=1e-8,
        atol=1e-10,
        equal_nan=True,
    )
    assert (
        ported[object_cols].astype(str).to_numpy() == golden[object_cols].astype(str).to_numpy()
    ).all()


def test_mixed_results_stimulus_matches_golden(results):
    _assert_matches_golden(
        results["mixed_results_stimulus"],
        GOLDEN_DIR / "all_bout_frequency_Low_concs_mixed_model_per_stimulus.csv",
    )


def test_mixed_results_valence_matches_golden(results):
    _assert_matches_golden(
        results["mixed_results_valence"],
        GOLDEN_DIR / "all_bout_frequency_Low_concs_mixed_model_valence.csv",
    )


def test_fish_plot_df_stimulus_matches_golden(results):
    _assert_matches_golden(
        results["fish_plot_df_stimulus"],
        GOLDEN_DIR / "all_bout_frequency_Low_concs_fish_plot_values_per_stimulus.csv",
    )


def test_fish_plot_df_valence_matches_golden(results):
    _assert_matches_golden(
        results["fish_plot_df_valence"],
        GOLDEN_DIR / "all_bout_frequency_Low_concs_fish_plot_values_valence.csv",
    )
