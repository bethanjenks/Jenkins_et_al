"""Regression test for jenkins_et_al.behaviour.reorientation_posture_lmm
against a golden capture of
notebooks/behaviour_head_fixed/reorientation_posture_LMM.ipynb (commit
2d19f3a).

The golden capture (golden/reorientation_posture_LMM/) was produced by
executing the notebook's own cell source directly, fresh top-to-bottom.
**This is not the same as the notebook's own saved/displayed output** -- the
saved notebook shows n_trials=5 for "Ade" vs. 6 for every other stimulus,
which is inconsistent with the uniform-6-trials-per-fish pattern confirmed
in every other notebook in this folder, and is a stale/out-of-order-execution
artifact rather than the code's true behavior. Per the skill's "run from a
clean kernel" rule, the fresh top-to-bottom run is the golden reference here,
not the notebook's stale display. See reorientation_posture_lmm.py's module
docstring for the full explanation, including the separate (inert,
diagnostic-only) "n fish per stimulus" bug that this port doesn't reproduce.

Skipped entirely if the source data drive isn't mounted, since the underlying
behaviour-traces-posture CSVs are not part of this repository.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from jenkins_et_al.behaviour import reorientation_posture_lmm
from jenkins_et_al.config import STIMULUS_GROUPS, STIMULUS_LABELS

FB_PATH = Path("/Volumes/LaCie/larval_HuC/behavior/7dpf_fb_behavior_traces_posture.csv")
HB_PATH = Path("/Volumes/LaCie/larval_HuC/behavior/7dpf_hb_behavior_traces_posture.csv")
GOLDEN_DIR = Path(__file__).resolve().parent.parent.parent / "golden" / "reorientation_posture_LMM"

pytestmark = pytest.mark.skipif(
    not FB_PATH.exists() or not HB_PATH.exists() or not GOLDEN_DIR.exists(),
    reason="source data drive or golden capture not available on this machine",
)


@pytest.fixture(scope="module")
def results():
    return reorientation_posture_lmm.run_reorientation_posture_analysis(
        fb_path=FB_PATH,
        hb_path=HB_PATH,
        key_stimuli=STIMULUS_GROUPS["high"],
        stimulus_labels=STIMULUS_LABELS,
    )


def _assert_matches_golden(ported: pd.DataFrame, golden_csv: Path, drop_cols: tuple[str, ...] = ()) -> None:
    golden = pd.read_csv(golden_csv)
    ported = ported.drop(columns=list(drop_cols), errors="ignore")
    golden = golden.drop(columns=list(drop_cols), errors="ignore")

    assert list(ported.columns) == list(golden.columns)
    assert len(ported) == len(golden)

    numeric_cols = golden.select_dtypes(include=[np.number]).columns
    object_cols = [c for c in golden.columns if c not in numeric_cols]

    assert np.allclose(
        ported[numeric_cols].to_numpy(dtype=float),
        golden[numeric_cols].to_numpy(dtype=float),
        rtol=1e-6,
        atol=1e-9,
        equal_nan=True,
    )
    assert (
        ported[object_cols].astype(str).to_numpy() == golden[object_cols].astype(str).to_numpy()
    ).all()


def test_lmm_results_df_matches_golden(results):
    _assert_matches_golden(results["lmm_results_df"], GOLDEN_DIR / "lmm_results_df.csv")


def test_per_fish_summary_df_matches_golden(results):
    _assert_matches_golden(
        results["per_fish_summary_df"], GOLDEN_DIR / "Posture_traces_concs_mixed_model_per_stimulus.csv"
    )


def test_effects_df_matches_golden(results):
    _assert_matches_golden(results["effects_df"], GOLDEN_DIR / "effects_df.csv")


def test_conc_results_df_matches_golden(results):
    _assert_matches_golden(results["conc_results_df"], GOLDEN_DIR / "conc_results_df.csv")


def test_conc_fish_plot_df_matches_golden(results):
    _assert_matches_golden(results["conc_fish_plot_df"], GOLDEN_DIR / "conc_fish_plot_df.csv")
