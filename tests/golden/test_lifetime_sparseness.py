"""Regression test for jenkins_et_al.neural.lifetime_sparseness against a
golden capture of notebooks/neural/lifetime_sparseness.ipynb (current file
content, commit 95133c5 -- ahead of the frozen 2d19f3a baseline; the user's
own Mann-Whitney -> Wilcoxon edit to this notebook, see module docstring).

The golden capture (golden/lifetime_sparseness/) was produced by executing
the notebook's own cell source directly, fresh top-to-bottom, redirecting its
output-file writes into that directory instead of overwriting the real
production files under /Volumes/LaCie.

Confirms two things beyond plain output-identity:
- The nMLF-mask reassignment (cell 9) has zero effect on any output here
  (resp_pop is pivoted before the reassignment runs) -- not reproduced in
  the port; see lifetime_sparseness.py's module docstring.
- The area-shorthand mapping (cell 16) mutates the same dataframe object
  later reused by the brain-plot neuron selection (cell 26), so neurons
  with no shorthand entry ("not_assigned", 11228 of 200116) are dropped
  from the brain-plot neuron pool too -- reproduced explicitly via
  `map_area_to_shorthand` feeding both consumers.

Skipped entirely if the source data drive isn't mounted, since the
underlying response-vector HDF5 is not part of this repository.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from jenkins_et_al.neural import lifetime_sparseness as ls

DATA_PATH = Path("/Volumes/LaCie/larval_HuC/imaging/7dpf_wb_population_general_response_vector.h5")
GOLDEN_DIR = Path(__file__).resolve().parent.parent.parent / "golden" / "lifetime_sparseness"

pytestmark = pytest.mark.skipif(
    not DATA_PATH.exists() or not GOLDEN_DIR.exists(),
    reason="source data drive or golden capture not available on this machine",
)


@pytest.fixture(scope="module")
def results():
    return ls.run_lifetime_sparseness_analysis(DATA_PATH)


def test_neuron_sparseness_values_match_golden(results):
    ported = results["neuron_sparseness_df"]["sparseness_values"].to_numpy().ravel()

    # values_for_brain_map has MultiIndex columns (per-trial response columns
    # unioned with scalar columns) -- to_csv writes a 2-row header, matching
    # the notebook's own OUTPUT_CSV_PATH write.
    golden = pd.read_csv(GOLDEN_DIR / "7dpf_area_sparseness_values.csv", header=[0, 1])
    golden_values = golden.xs("values", axis=1, level=0).iloc[:, 0].to_numpy()

    assert len(ported) == len(golden_values)
    assert np.allclose(ported, golden_values, rtol=1e-9, atol=1e-12)


def test_area_fish_sparseness_df_matches_golden(results):
    ported = results["area_fish_sparseness_df"]
    golden = pd.read_csv(GOLDEN_DIR / "lifetimes_parseness_per_fish_general_response.csv", dtype={"fish_id": str})

    assert list(ported.columns) == list(golden.columns)
    assert len(ported) == len(golden)
    assert (ported["area"].to_numpy() == golden["area"].to_numpy()).all()
    assert (ported["fish_id"].astype(str).to_numpy() == golden["fish_id"].to_numpy()).all()
    assert np.allclose(ported["sparseness_values"].to_numpy(), golden["sparseness_values"].to_numpy(), rtol=1e-9, atol=1e-12)
    assert np.allclose(
        ported["p_value"].to_numpy(dtype=float), golden["p_value"].to_numpy(dtype=float),
        rtol=1e-9, atol=1e-12, equal_nan=True,
    )


def test_summary_df_matches_golden(results):
    ported = results["summary_df"]
    golden = pd.read_csv(GOLDEN_DIR / "summary.csv")

    assert list(ported.columns) == list(golden.columns)
    assert (ported["area"].to_numpy() == golden["area"].to_numpy()).all()
    assert (ported["n_fish"].to_numpy() == golden["n_fish"].to_numpy()).all()
    assert np.allclose(ported["mean"].to_numpy(), golden["mean"].to_numpy(), rtol=1e-9, atol=1e-12)
    assert np.allclose(ported["std"].to_numpy(), golden["std"].to_numpy(), rtol=1e-9, atol=1e-12)
    assert np.allclose(
        ported["p_value"].to_numpy(dtype=float), golden["p_value"].to_numpy(dtype=float),
        rtol=1e-9, atol=1e-12, equal_nan=True,
    )


def test_pvals_match_golden(results):
    golden_pvals = pd.read_csv(GOLDEN_DIR / "pval_df.csv").set_index("area")["p_value"].to_dict()

    assert set(results["pvals"].keys()) == set(golden_pvals.keys())
    for area, p in results["pvals"].items():
        assert np.isclose(p, golden_pvals[area], rtol=1e-9, atol=1e-12)


def test_brain_plot_neuron_subset_matches_golden(results):
    """Confirms the area-shorthand-mapping/brain-plot coupling: same 7554 neurons, same coords/values."""
    subset = ls.select_extreme_neurons(results["mapped_neuron_df"])
    ported_coords = np.vstack(subset["coords"])
    ported_values = subset["sparseness_values"].to_numpy()

    golden_coords = np.load(GOLDEN_DIR / "brain_coords.npy")
    golden_values = np.load(GOLDEN_DIR / "brain_values.npy")

    assert ported_coords.shape == golden_coords.shape
    assert np.allclose(ported_coords, golden_coords)
    assert np.allclose(ported_values, golden_values, rtol=1e-9, atol=1e-12)
