"""Regression test for jenkins_et_al.neural.continuous_correlation_hex_heatmaps
against a golden capture derived from
notebooks/neural/continuous_correlation_hex_heatmaps.ipynb (commit 2d19f3a
baseline notebook, split off later into this standalone file -- see module
docstring).

golden/continuous_correlation_hex_heatmaps/ was produced by running the
ported module against real data (whole-brain, 200116 neurons), after
independently verifying it against the notebook's own cell code, copied
verbatim and run standalone on identical inputs.

Skipped entirely if the source data drive isn't mounted.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from jenkins_et_al.neural.area_pair_core import load_nmlf_mask
from jenkins_et_al.neural.continuous_correlation_hex_heatmaps import (
    compute_spearman_correlations,
    filter_significant_for_area,
)
from jenkins_et_al.neural.valence_bhv_core import load_valence_bhv_dataframe

DATA_DIR = Path("/Volumes/LaCie/larval_HuC/imaging")
POS_PATH = DATA_DIR / "7dpf_positive_correlations_25s_no_HCl_1000_corrected.csv"
NEG_PATH = DATA_DIR / "7dpf_negative_correlations_25s_no_HCl_1000_corrected.csv"
BHV_PATH = DATA_DIR / "7dpf_bhv_vigor_correlations_1000_corrected.csv"
NMLF_MASK_PATH = Path("/Volumes/LaCie/area_masks/area_masks_mapzebrain/nMLF.tiff")
GOLDEN_DIR = Path(__file__).resolve().parent.parent.parent / "golden" / "continuous_correlation_hex_heatmaps"
AREA = "olfactory_bulb"

pytestmark = pytest.mark.skipif(
    not POS_PATH.exists() or not GOLDEN_DIR.exists(),
    reason="source data drive or golden capture not available on this machine",
)


@pytest.fixture(scope="module")
def rhos_and_n():
    nmlf_mask = load_nmlf_mask(NMLF_MASK_PATH)
    stim_df = load_valence_bhv_dataframe(POS_PATH, NEG_PATH, BHV_PATH, nmlf_mask)
    df = filter_significant_for_area(stim_df, AREA)
    return compute_spearman_correlations(df), len(df)


def test_spearman_correlations_match_golden(rhos_and_n):
    rhos, n = rhos_and_n
    golden_rhos = json.loads((GOLDEN_DIR / "rhos.json").read_text())
    golden_n = json.loads((GOLDEN_DIR / "n.json").read_text())["n"]

    assert n == golden_n
    for key in ["valence", "positive", "negative"]:
        assert np.allclose(rhos[key], golden_rhos[key], atol=1e-10)
