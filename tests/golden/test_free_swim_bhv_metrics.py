"""Regression test for jenkins_et_al.behaviour.free_swim against a golden capture
of notebooks/behaviour_free_swim/free_swim_bhv_metrics.ipynb (commit 2d19f3a).

The golden capture (golden/free_swim_bhv_metrics/) was produced by executing
the notebook's own cell source directly, redirected to write CSVs under
golden/ instead of the notebook's own hardcoded output location
(/Users/bethanjenkins/Documents/valence_paper_code/FiNA/, outside the repo).
It also persists the two intermediate tables (results_df, angle_results_df)
that the notebook itself never saves, for finer-grained comparison.

Skipped entirely if the source data drive isn't mounted, since the
underlying per-fish tracking CSVs are not part of this repository.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from jenkins_et_al.behaviour import free_swim

ROOT_DIR = Path("/Volumes/LaCie/free_swimming/7dpf_TRex")
GOLDEN_DIR = Path(__file__).resolve().parent.parent.parent / "golden" / "free_swim_bhv_metrics"

pytestmark = pytest.mark.skipif(
    not ROOT_DIR.exists() or not GOLDEN_DIR.exists(),
    reason="source data drive or golden capture not available on this machine",
)


@pytest.fixture(scope="module")
def stimulus_data():
    return free_swim.load_stimulus_data(ROOT_DIR)


@pytest.fixture(scope="module")
def results_df(stimulus_data):
    return free_swim.build_experiment_results_table(stimulus_data)


@pytest.fixture(scope="module")
def angle_results_df(stimulus_data):
    return free_swim.build_experiment_angle_results_table(stimulus_data)


@pytest.fixture(scope="module")
def fish_results_df(stimulus_data):
    return free_swim.build_fish_results_table(stimulus_data)


def _assert_matches_golden(ported: pd.DataFrame, golden_csv: Path) -> None:
    golden = pd.read_csv(golden_csv)

    assert list(ported.columns) == list(golden.columns)
    assert len(ported) == len(golden)

    numeric_cols = golden.select_dtypes(include=[np.number]).columns
    object_cols = [c for c in golden.columns if c not in numeric_cols]

    assert np.allclose(
        ported[numeric_cols].to_numpy(dtype=float),
        golden[numeric_cols].to_numpy(dtype=float),
        rtol=1e-10,
        atol=1e-12,
        equal_nan=True,
    )
    assert (ported[object_cols].to_numpy() == golden[object_cols].to_numpy()).all()


def test_experiment_results_table_matches_golden(results_df):
    _assert_matches_golden(results_df, GOLDEN_DIR / "results_df.csv")


def test_experiment_angle_results_table_matches_golden(angle_results_df):
    _assert_matches_golden(angle_results_df, GOLDEN_DIR / "angle_results_df.csv")


def test_merged_experiment_results_matches_golden(results_df, angle_results_df):
    merged = results_df.merge(
        angle_results_df,
        on=["stimulus", "pair_key", "experiment", "condition"],
        how="left",
    )
    _assert_matches_golden(merged, GOLDEN_DIR / "exp_results_with_angles_stim_side.csv")


def test_fish_results_table_matches_golden(fish_results_df):
    _assert_matches_golden(fish_results_df, GOLDEN_DIR / "fish_results_df_stim_side_allfish.csv")


def test_fish_results_row_count_matches_golden_experiment_n_fish(results_df, fish_results_df):
    per_experiment_n_fish = (
        fish_results_df.groupby(["stimulus", "pair_key", "experiment", "condition"])
        .size()
        .rename("n_fish_actual")
        .reset_index()
    )
    merged = results_df.merge(
        per_experiment_n_fish,
        on=["stimulus", "pair_key", "experiment", "condition"],
        how="left",
    )
    assert (merged["n_fish"] == merged["n_fish_actual"]).all()
