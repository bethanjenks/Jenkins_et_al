"""Regression test for jenkins_et_al.neural.cross_regressor_neuron_correlations
against a golden capture derived from
notebooks/neural/cross_regressor_neuron_correlations.ipynb.

golden/cross_regressor_neuron_correlations/stats.json was produced by
running the ported module against real data, after independently verifying
it against the notebook's own cell code, copied verbatim and run standalone
on identical inputs.

Skipped entirely if the source data drive isn't mounted.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from jenkins_et_al.neural.area_pair_core import load_nmlf_mask
from jenkins_et_al.neural.cross_regressor_neuron_correlations import (
    plot_bhv_neurons_neg_vs_pos_correlation,
    plot_valence_neurons_bhv_correlation,
    select_neuron_groups,
)
from jenkins_et_al.neural.valence_bhv_core import load_valence_bhv_dataframe

DATA_DIR = Path("/Volumes/LaCie/larval_HuC/imaging")
POS_PATH = DATA_DIR / "7dpf_positive_correlations_25s_no_HCl_1000_corrected.csv"
NEG_PATH = DATA_DIR / "7dpf_negative_correlations_25s_no_HCl_1000_corrected.csv"
BHV_PATH = DATA_DIR / "7dpf_bhv_vigor_correlations_1000_corrected.csv"
NMLF_MASK_PATH = Path("/Volumes/LaCie/area_masks/area_masks_mapzebrain/nMLF.tiff")
GOLDEN_DIR = Path(__file__).resolve().parent.parent.parent / "golden" / "cross_regressor_neuron_correlations"

pytestmark = pytest.mark.skipif(
    not POS_PATH.exists() or not GOLDEN_DIR.exists(),
    reason="source data drive or golden capture not available on this machine",
)


@pytest.fixture(scope="module")
def stats():
    nmlf_mask = load_nmlf_mask(NMLF_MASK_PATH)
    stim_df = load_valence_bhv_dataframe(POS_PATH, NEG_PATH, BHV_PATH, nmlf_mask)
    groups = select_neuron_groups(stim_df)
    _, w_stat, w_p = plot_bhv_neurons_neg_vs_pos_correlation(groups["high_bhv"])
    _, u_stat, u_p = plot_valence_neurons_bhv_correlation(groups["high_pos"], groups["high_neg"])
    return {"wilcoxon_stat": w_stat, "wilcoxon_p": w_p, "mannwhitney_stat": u_stat, "mannwhitney_p": u_p}


def test_stats_match_golden(stats):
    golden = json.loads((GOLDEN_DIR / "stats.json").read_text())
    for key in ["wilcoxon_stat", "wilcoxon_p", "mannwhitney_stat", "mannwhitney_p"]:
        assert np.isclose(stats[key], golden[key], atol=1e-10)
