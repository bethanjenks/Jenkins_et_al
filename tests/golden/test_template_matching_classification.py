"""Regression test for the deterministic boxplot-vs-OB stats path in
jenkins_et_al.neural.template_matching_classification.

notebooks/neural/template_matching_classification.ipynb has never been run to
completion as saved (every cell's execution_count is None) and contains two
confirmed blocking bugs -- see the module's docstring. There is no golden
capture of a full notebook run to compare against. Instead, this verifies the
ported `load_classification_accuracy_csv` -> `clean_and_average_accuracy` ->
`compute_area_pvalues_paired_wilcoxon` chain against a from-scratch,
independent reimplementation of the same cells
(golden/template_matching_classification/), run once during development and
cross-checked to match exactly (see PORTING_PLAN.md). Both boxplots (hindbrain
and forebrain, vs. OB) use this same Wilcoxon function -- the user asked for
Wilcoxon in place of the notebook's own unpaired Mann-Whitney U for the
forebrain figure; see the module docstring.

The real per-fish/area accuracy CSV this reads
(classification_accuracy_per_fish_early_resp.csv) lives outside the repo, at
the notebook's own original working directory -- not on /Volumes/LaCie like
this project's other data dependencies.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from jenkins_et_al.config import FOREBRAIN_AREA_COLORS
from jenkins_et_al.neural import template_matching_classification as tmc

CSV_PATH = Path("/Users/bethanjenkins/Documents/valence_paper_code/Sense/classification_accuracy_per_fish_early_resp.csv")
GOLDEN_DIR = Path(__file__).resolve().parent.parent.parent / "golden" / "template_matching_classification"

pytestmark = pytest.mark.skipif(
    not CSV_PATH.exists() or not GOLDEN_DIR.exists(),
    reason="source accuracy CSV or golden capture not available on this machine",
)


@pytest.fixture(scope="module")
def cleaned_df():
    return tmc.clean_and_average_accuracy(tmc.load_classification_accuracy_csv(CSV_PATH))


def test_cleaned_and_averaged_accuracy_matches_golden(cleaned_df):
    golden = pd.read_csv(GOLDEN_DIR / "cleaned_accuracy.csv")

    assert len(cleaned_df) == len(golden)
    merged = cleaned_df.merge(golden, on=["area", "fish_id"], suffixes=("_port", "_golden"))
    assert len(merged) == len(golden)
    assert np.allclose(
        merged["classification_accuracy_port"], merged["classification_accuracy_golden"], rtol=1e-9, atol=1e-12,
    )


def test_wilcoxon_pvalues_match_golden(cleaned_df):
    pvals, _ = tmc.compute_area_pvalues_paired_wilcoxon(cleaned_df, reference_area=tmc.REFERENCE_AREA)
    golden = pd.read_csv(GOLDEN_DIR / "wilcoxon_pvals.csv").set_index("area")["p_value"].to_dict()

    assert set(pvals.keys()) == set(golden.keys())
    for area, p in pvals.items():
        assert np.isclose(p, golden[area], rtol=1e-9, atol=1e-12)


def test_all_hindbrain_and_forebrain_areas_present(cleaned_df):
    """Confirms the forebrain-boxplot KeyError is real: every forebrain area is present
    in the data, so the original hardcoded (hindbrain-only) palette would have crashed."""
    present = set(cleaned_df["area"].unique())

    assert set(tmc.HINDBRAIN_AREA_ORDER).issubset(present)
    assert set(tmc.FOREBRAIN_AREA_ORDER).issubset(present)


def test_both_boxplots_render(cleaned_df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wilcoxon_pvals, _ = tmc.compute_area_pvalues_paired_wilcoxon(cleaned_df, reference_area=tmc.REFERENCE_AREA)

    fig_hind = tmc.plot_classification_accuracy_boxplot(
        cleaned_df, tmc.HINDBRAIN_AREA_ORDER, tmc.HINDBRAIN_AREA_ORDER, wilcoxon_pvals, reference_area=tmc.REFERENCE_AREA,
    )
    fig_fore = tmc.plot_classification_accuracy_boxplot(
        cleaned_df, tmc.FOREBRAIN_AREA_ORDER, tmc.FOREBRAIN_AREA_ORDER, wilcoxon_pvals,
        reference_area=tmc.REFERENCE_AREA, palette=FOREBRAIN_AREA_COLORS,
    )

    assert isinstance(fig_hind, plt.Figure)
    assert isinstance(fig_fore, plt.Figure)
    plt.close(fig_hind)
    plt.close(fig_fore)
