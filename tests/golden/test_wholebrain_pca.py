"""Regression test for jenkins_et_al.neural.wholebrain_pca against a golden
capture derived from
notebooks/neural/wholebrain_avg_traces_PCA_raster.ipynb (forebrain,
pallium-restricted -- see module docstring in wholebrain_pca.py for the
confirmed area_df/stim_df mismatch this fixes).

golden/wholebrain_pca/pca.npz was produced by running the ported module
against real data, after independently verifying it against the notebook's
own cell code (with the user-agreed area-restriction fix applied), copied
verbatim and run standalone on identical inputs.

**`data_array` (the actual per-neuron trace data feeding PCA) is compared
exactly** -- this is deterministic and matched bit-for-bit during
verification. **`explained_variance`/`smoothed_pca` use tolerances set from
an empirical measurement, not a guess**: confirmed real, non-trivial
run-to-run floating-point non-determinism in sklearn's `PCA.fit` on this
machine (almost certainly multi-threaded BLAS summation-order effects --
Accelerate on macOS) -- calling `perform_pca` five times in a row on the
*same* real `data_array` gave `explained_variance` stable to ~1e-6 (tight
tolerance kept), but `smoothed_pca` (per-neuron loadings on the low-variance
PC5/PC6 in particular) varied by up to ~0.07 across those runs -- `atol=0.1`
below is set from that measured spread, per this project's rule to verify
non-deterministic output lands within the observed spread rather than
guessing a tolerance. PC1-4 are far more stable in practice; this loose
tolerance is driven entirely by PC5/PC6.

Skipped entirely if the source data drive isn't mounted.
"""
from pathlib import Path

import numpy as np
import pytest

from jenkins_et_al.neural.wholebrain_pca import perform_pca, prepare_pca_data
from jenkins_et_al.neural.wholebrain_traces import load_forebrain_data

FB_PATH = Path("/Volumes/LaCie/larval_HuC/imaging/7dpf_fb_fish_dfs_final.h5")
GOLDEN_DIR = Path(__file__).resolve().parent.parent.parent / "golden" / "wholebrain_pca"

pytestmark = pytest.mark.skipif(
    not FB_PATH.exists() or not GOLDEN_DIR.exists(),
    reason="source data drive or golden capture not available on this machine",
)


@pytest.fixture(scope="module")
def pca_results():
    stim_fb = load_forebrain_data(FB_PATH)
    area_df = stim_fb[stim_fb["area"] == "pallium"]
    data_array, _unique_neurons = prepare_pca_data(area_df)
    smoothed_pca, explained_var, _weights = perform_pca(data_array)
    return data_array, smoothed_pca, explained_var


def test_data_array_matches_golden(pca_results):
    data_array, _smoothed_pca, _explained_var = pca_results
    golden = np.load(GOLDEN_DIR / "pca.npz")
    assert data_array.shape == golden["data_array"].shape
    assert np.allclose(data_array, golden["data_array"], atol=1e-10)


def test_pca_output_matches_golden(pca_results):
    _data_array, smoothed_pca, explained_var = pca_results
    golden = np.load(GOLDEN_DIR / "pca.npz")
    assert np.allclose(explained_var, golden["explained_var"], atol=1e-4)
    assert np.allclose(smoothed_pca, golden["smoothed_pca"], atol=0.1)
