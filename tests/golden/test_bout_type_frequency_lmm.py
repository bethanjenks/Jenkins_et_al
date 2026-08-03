"""Regression test for jenkins_et_al.behaviour.bout_type_frequency_lmm against
a golden capture of notebooks/behaviour_head_fixed/bout_type_frequency_LMM.ipynb
(commit 2d19f3a).

The golden capture (golden/bout_type_frequency_LMM/) was produced by executing
the notebook's own cell source (fresh top-to-bottom run), redirecting output
paths to golden/bout_type_frequency_LMM/, with one bug fix: the notebook's
straight-bout valence cell references undefined columns
("pre_stim_straight_freq"/"post_stim_straight_freq") and raises a KeyError as
written -- the capture (and the port) use the columns that actually exist
("straight_bout_freq_prestim"/"straight_bout_freq_poststim") instead. See
bout_type_frequency_lmm.py's module docstring for the full explanation.

Skipped entirely if the source data drive isn't mounted, since the underlying
behaviour-traces-posture CSVs are not part of this repository.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from jenkins_et_al.behaviour import bout_type_frequency_lmm
from jenkins_et_al.config import STIMULUS_GROUPS, STIMULUS_LABELS

FB_PATH = Path("/Volumes/LaCie/larval_HuC/behavior/7dpf_fb_behavior_traces_posture.csv")
HB_PATH = Path("/Volumes/LaCie/larval_HuC/behavior/7dpf_hb_behavior_traces_posture.csv")
GOLDEN_DIR = Path(__file__).resolve().parent.parent.parent / "golden" / "bout_type_frequency_LMM"

pytestmark = pytest.mark.skipif(
    not FB_PATH.exists() or not HB_PATH.exists() or not GOLDEN_DIR.exists(),
    reason="source data drive or golden capture not available on this machine",
)


@pytest.fixture(scope="module")
def results():
    return bout_type_frequency_lmm.run_bout_type_frequency_analysis(
        fb_path=FB_PATH,
        hb_path=HB_PATH,
        key_stimuli=STIMULUS_GROUPS["high"],
        stimulus_labels=STIMULUS_LABELS,
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


def test_asym_mixed_results_matches_golden(results):
    _assert_matches_golden(
        results["asym_mixed_results"], GOLDEN_DIR / "asym_bout_Intermediate_concs_mixed_model_results.csv"
    )


def test_asym_fish_summary_matches_golden(results):
    _assert_matches_golden(
        results["asym_fish_summary"], GOLDEN_DIR / "asym_bout_fish_Intermediate_concs_summary.csv"
    )


def test_straight_mixed_results_matches_golden(results):
    _assert_matches_golden(
        results["straight_mixed_results"], GOLDEN_DIR / "straight_bout_mixed_model_Low_conc_results.csv"
    )


def test_straight_fish_summary_matches_golden(results):
    _assert_matches_golden(
        results["straight_fish_summary"], GOLDEN_DIR / "straight_bout_fish_Low_conc_summary.csv"
    )


def test_straight_valence_wilcoxon_matches_golden(results):
    _assert_matches_golden(
        results["straight_valence_wilcoxon"], GOLDEN_DIR / "straight_bout_valence_wilcoxon_results.csv"
    )


def test_asym_valence_wilcoxon_matches_golden(results):
    _assert_matches_golden(
        results["asym_valence_wilcoxon"], GOLDEN_DIR / "asym_bout_valence_wilcoxon_results.csv"
    )


def test_all_bout_mixed_results_matches_golden(results):
    _assert_matches_golden(
        results["all_bout_mixed_results"], GOLDEN_DIR / "all_bout_freq_high_concs_mixed_model_results.csv"
    )
