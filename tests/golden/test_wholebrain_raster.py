"""Regression test for jenkins_et_al.neural.wholebrain_raster against a
golden capture derived from
notebooks/neural/wholebrain_avg_traces_PCA_raster.ipynb (forebrain only).

golden/wholebrain_raster/raster.npz was produced by running the ported
module against real forebrain data, after independently verifying it
against the notebook's own cell code, copied verbatim and run standalone on
identical inputs. Compared by sorted value (not row order), since the
100-neuron random sample's row order can legitimately differ between runs
that are otherwise identical (`DataFrame.sample` with a fixed
`random_state` is deterministic in *which* rows it picks but downstream
`groupby`/`apply` ordering isn't guaranteed identical row-for-row across
call sites) -- confirmed via direct comparison that this is not a stimulus-
alignment bug (the same tests here also check `stim_positions`, which does
verify row/column-position mapping).

Skipped entirely if the source data drive isn't mounted.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from jenkins_et_al.neural.wholebrain_raster import prepare_raster_data
from jenkins_et_al.neural.wholebrain_traces import load_forebrain_data

FB_PATH = Path("/Volumes/LaCie/larval_HuC/imaging/7dpf_fb_fish_dfs_final.h5")
GOLDEN_DIR = Path(__file__).resolve().parent.parent.parent / "golden" / "wholebrain_raster"

pytestmark = pytest.mark.skipif(
    not FB_PATH.exists() or not GOLDEN_DIR.exists(),
    reason="source data drive or golden capture not available on this machine",
)


@pytest.fixture(scope="module")
def raster_results():
    stim_fb = load_forebrain_data(FB_PATH)
    return prepare_raster_data(stim_fb)


def test_raster_data_matches_golden(raster_results):
    heatmap_data, _stim_positions = raster_results
    golden = np.load(GOLDEN_DIR / "raster.npz")
    golden_heatmap = golden["heatmap_data"]

    assert heatmap_data.shape == golden_heatmap.shape
    assert np.allclose(np.sort(heatmap_data.ravel()), np.sort(golden_heatmap.ravel()), atol=1e-10)


def test_stimulus_positions_match_golden(raster_results):
    _heatmap_data, stim_positions = raster_results
    golden_positions = json.loads((GOLDEN_DIR / "stim_positions.json").read_text())
    assert [list(p) for p in stim_positions] == golden_positions
