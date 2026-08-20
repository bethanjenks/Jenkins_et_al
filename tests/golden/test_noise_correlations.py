"""Regression test for jenkins_et_al.neural.noise_correlations against a
golden capture derived from
notebooks/neural/noise_correlations_area_pair_analysis.ipynb (this notebook
replaces the one previously tracked as PORTING_PLAN.md item #8, per the
user).

golden/noise_correlations/summary.csv was produced by running the ported
module against real data for a reduced 2-fish, 3-area subset (20 null
resamples instead of the notebook's own default 1000, for speed -- the
resampling logic itself is exercised in full, just fewer draws), after
independently verifying the ported module against the notebook's own cell
code, copied verbatim and run standalone on identical inputs. See
PORTING_PLAN.md for the comparison methodology and the confirmed
"intra vs inter" reconstruction (`intra_vs_inter_summary`).

Skipped entirely if the source data drive isn't mounted.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from jenkins_et_al.neural.area_pair_core import load_group_data
from jenkins_et_al.neural.noise_correlations import run_noise_correlation_analysis

DATA_DIR = Path("/Volumes/LaCie/larval_HuC/imaging")
FISH_DATA_PATH = DATA_DIR / "7dpf_wb_population_general_response_vector.h5"
GROUP1_PATH = DATA_DIR / "7dpf_negative_correlations_25s_no_HCl_1000_corrected.csv"
GROUP2_PATH = DATA_DIR / "7dpf_new_asymm_bout_all_correlations_1000_corrected.csv"
GOLDEN_DIR = Path(__file__).resolve().parent.parent.parent / "golden" / "noise_correlations"

FISH_LIST = ["230713_fb", "230714_fb"]
BRAIN_AREAS = ["olfactory_bulb", "pallium", "subpallium"]
N_RANDOM_SAMPLES = 20

pytestmark = pytest.mark.skipif(
    not FISH_DATA_PATH.exists() or not GOLDEN_DIR.exists(),
    reason="source data drive or golden capture not available on this machine",
)


@pytest.fixture(scope="module")
def summary_df():
    group1_df = load_group_data(GROUP1_PATH, None, "regression", 0.05, 0.3)
    group2_df = load_group_data(GROUP2_PATH, None, "regression", 0.05, 0.3)
    summary_df, _area_pair_corr_values, _random_pair_corr_values = run_noise_correlation_analysis(
        fish_list=FISH_LIST,
        fish_data_path=FISH_DATA_PATH,
        group1_df=group1_df,
        group2_df=group2_df,
        brain_areas=BRAIN_AREAS,
        compare_same_group=False,
        use_group1_for_both=False,
        use_group2_for_both=False,
        n_random_samples=N_RANDOM_SAMPLES,
    )
    return summary_df


def test_summary_matches_golden(summary_df):
    golden = pd.read_csv(GOLDEN_DIR / "summary.csv")
    ported = summary_df.sort_values(["fish_id", "area1", "area2"]).reset_index(drop=True)
    golden = golden.sort_values(["fish_id", "area1", "area2"]).reset_index(drop=True)

    assert ported.shape == golden.shape
    for col in ["n1", "n2", "n_pairs_real"]:
        assert (ported[col].values == golden[col].values).all()
    for col in [
        "real_mean_r", "real_r95", "real_r5",
        "rand_mean_mean", "rand_mean_r95", "rand_mean_r5",
        "p_mean", "p_r95", "p_r5",
    ]:
        assert np.allclose(ported[col], golden[col], atol=1e-10, equal_nan=True)
