"""Regression test for jenkins_et_al.preprocessing.fdr_correction against a
golden capture derived from notebooks/preprocessing/fdr_correction.ipynb.

golden/fdr_correction/data_fdr.csv was produced by running the ported
module against real data, after independently verifying it against the
notebook's own cell code (with the confirmed missing `multipletests` import
added -- see module docstring), copied verbatim and run standalone on an
identical input file.

Skipped entirely if the source data drive isn't mounted.
"""
from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from jenkins_et_al.preprocessing.fdr_correction import (
    apply_per_fish_fdr_correction,
    load_correlations,
)

INPUT_PATH = Path("/Volumes/LaCie/larval_HuC/imaging/7dpf_ade_kw_qui_2.5mm_correlations_25s_1000.csv")
GOLDEN_DIR = Path(__file__).resolve().parent.parent.parent / "golden" / "fdr_correction"

pytestmark = pytest.mark.skipif(
    not INPUT_PATH.exists() or not GOLDEN_DIR.exists(),
    reason="source data drive or golden capture not available on this machine",
)


@pytest.fixture(scope="module")
def data_fdr():
    data = load_correlations(INPUT_PATH)
    return apply_per_fish_fdr_correction(data)


def test_matches_golden(data_fdr):
    golden = pd.read_csv(GOLDEN_DIR / "data_fdr.csv")
    ported = data_fdr.sort_values(["fish_id", "neuron_id"]).reset_index(drop=True)
    golden = golden.sort_values(["fish_id", "neuron_id"]).reset_index(drop=True)

    assert ported.shape == golden.shape
    assert np.allclose(ported["p_fdr"], golden["p_fdr"], atol=1e-12, equal_nan=True)
    assert np.allclose(ported["p_value"], golden["p_value"], atol=1e-12, equal_nan=True)
    assert np.allclose(ported["correlation"], golden["correlation"], atol=1e-12, equal_nan=True)
    assert (ported["fish_id"].astype(str) == golden["fish_id"].astype(str)).all()
    assert (ported["area"].astype(str) == golden["area"].astype(str)).all()

    golden_coords = golden["coords"].apply(literal_eval)
    assert all(
        list(p) == list(g) for p, g in zip(ported["coords"].values, golden_coords.values)
    )
