"""Regression test for jenkins_et_al.neural.response_latency against a
golden capture of notebooks/neural/response_latency.ipynb.

golden/response_latency/ was produced by running the notebook's own cell
source fresh top-to-bottom (capture_golden_response_latency.py, run once
during this port; OUTPUT_DIR redirected into this in-repo golden/ directory
instead of the notebook's own off-repo
/Users/bethanjenkins/Documents/valence_paper_code/FiNA/figures path, same
approach as port #2 process_TRex_output). 10 fish, forebrain + hindbrain
imaging sessions each (see module docstring's `FOREBRAIN_FISH`/
`HINDBRAIN_FISH`).

Confirms the data feeding every notebook output: per-fish x area onset
detection (`area_latency_df`), whole-region detection before and after the
fish-230811 manual override, the paired forebrain/hindbrain Wilcoxon
comparison, OB-relative latency + its per-area Wilcoxon tests, and the
summary table. Rendered figures are not re-asserted pixel-exact, per the
golden-refactor skill's "verify the data, not pixels" rule -- signed off
visually separately (see PORTING_PLAN.md).

Expensive: reruns onset detection for all 10 fish x 2 imaging sessions x
~34GB/45GB HDF5 files (~5-10 minutes). Skipped entirely if the source data
drive isn't mounted.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from jenkins_et_al.neural import response_latency as rl

FOREBRAIN_DATA_PATH = Path("/Volumes/LaCie/larval_HuC/imaging/7dpf_fb_fish_dfs_final.h5")
HINDBRAIN_DATA_PATH = Path("/Volumes/LaCie/larval_HuC/imaging/7dpf_hb_fish_dfs_final.h5")
GOLDEN_DIR = Path(__file__).resolve().parent.parent.parent / "golden" / "response_latency"

pytestmark = pytest.mark.skipif(
    not FOREBRAIN_DATA_PATH.exists() or not GOLDEN_DIR.exists(),
    reason="source data drive or golden capture not available on this machine",
)


@pytest.fixture(scope="module")
def area_latency_df():
    results = []
    for fish in rl.FOREBRAIN_FISH:
        results.extend(rl.process_fish_by_area(fish, FOREBRAIN_DATA_PATH, "forebrain", suffix="_fb"))
    return pd.DataFrame(results)


@pytest.fixture(scope="module")
def region_latency_df_pre_manual():
    results = [rl.process_whole_region(fish, FOREBRAIN_DATA_PATH, "forebrain", suffix="_fb") for fish in rl.FOREBRAIN_FISH]
    results += [rl.process_whole_region(fish, HINDBRAIN_DATA_PATH, "hindbrain", suffix="_hb") for fish in rl.HINDBRAIN_FISH]
    return pd.DataFrame(results)


@pytest.fixture(scope="module")
def region_latency_df(region_latency_df_pre_manual):
    return rl.apply_manual_override(
        region_latency_df_pre_manual, "230811", "region", "hindbrain", rl.FISH_230811_HINDBRAIN_MANUAL_LATENCY_S,
    )


def test_area_latency_df_matches_golden(area_latency_df):
    golden = pd.read_csv(GOLDEN_DIR / "area_latency_derivative_all_results.csv", dtype={"fish_id": str})
    port = area_latency_df.sort_values(["fish_id", "area"]).reset_index(drop=True)
    golden = golden.sort_values(["fish_id", "area"]).reset_index(drop=True)

    assert len(port) == len(golden)
    assert (port["status"].to_numpy() == golden["status"].to_numpy()).all()
    assert (port["n_traces"].to_numpy() == golden["n_traces"].to_numpy()).all()
    assert np.allclose(port["latency_s"].dropna().to_numpy(), golden["latency_s"].dropna().to_numpy())


def test_region_latency_df_matches_golden_pre_manual_override(region_latency_df_pre_manual):
    golden = pd.read_csv(
        GOLDEN_DIR / "whole_region_latency_derivative_results_PRE_MANUAL.csv", dtype={"fish_id": str},
    )
    port = region_latency_df_pre_manual.sort_values(["fish_id", "region"]).reset_index(drop=True)
    golden = golden.sort_values(["fish_id", "region"]).reset_index(drop=True)

    assert (port["status"].to_numpy() == golden["status"].to_numpy()).all()
    assert np.allclose(port["latency_s"].dropna().to_numpy(), golden["latency_s"].dropna().to_numpy())


def test_region_latency_df_matches_golden_after_manual_override(region_latency_df):
    golden = pd.read_csv(GOLDEN_DIR / "whole_region_latency_derivative_results.csv", dtype={"fish_id": str})
    port = region_latency_df.sort_values(["fish_id", "region"]).reset_index(drop=True)
    golden = golden.sort_values(["fish_id", "region"]).reset_index(drop=True)

    assert (port["status"].to_numpy() == golden["status"].to_numpy()).all()
    assert np.allclose(port["latency_s"].to_numpy(), golden["latency_s"].to_numpy())

    overridden = port[(port["fish_id"] == "230811") & (port["region"] == "hindbrain")].iloc[0]
    assert overridden["status"] == "manual"
    assert overridden["latency_s"] == 8.0


def test_compare_forebrain_hindbrain_matches_golden(region_latency_df):
    paired, stats = rl.compare_forebrain_hindbrain(region_latency_df)

    golden_paired = pd.read_csv(GOLDEN_DIR / "paired_forebrain_hindbrain_latency.csv", dtype={"fish_id": str}, index_col="fish_id")
    golden_stats = pd.read_csv(GOLDEN_DIR / "forebrain_hindbrain_wilcoxon_stats.csv")

    paired = paired.sort_index()
    golden_paired = golden_paired.sort_index()
    assert np.allclose(paired["forebrain"].to_numpy(), golden_paired["forebrain"].to_numpy())
    assert np.allclose(paired["hindbrain"].to_numpy(), golden_paired["hindbrain"].to_numpy())
    assert stats.loc[0, "n"] == golden_stats.loc[0, "n"]
    assert np.isclose(stats.loc[0, "statistic"], golden_stats.loc[0, "statistic"])
    assert np.isclose(stats.loc[0, "p_value"], golden_stats.loc[0, "p_value"])


def test_fb_hb_difference_stats_match_golden(region_latency_df):
    paired, _ = rl.compare_forebrain_hindbrain(region_latency_df)
    diff_stats = rl.compute_fb_hb_difference_stats(paired)

    golden = pd.read_csv(GOLDEN_DIR / "fb_hb_difference_stats.csv", index_col=0)
    assert np.isclose(diff_stats["mean"], golden.loc["mean"].iloc[0])
    assert np.isclose(diff_stats["std"], golden.loc["std"].iloc[0])
    assert np.isclose(diff_stats["median"], golden.loc["median"].iloc[0])


def test_relative_to_ob_matches_golden(area_latency_df):
    rel_df = rl.compute_relative_to_reference(area_latency_df)
    golden = pd.read_csv(GOLDEN_DIR / "OB_relative_latency_derivative.csv", dtype={"fish_id": str})

    port = rel_df.sort_values(["fish_id", "area"]).reset_index(drop=True)
    golden = golden.sort_values(["fish_id", "area"]).reset_index(drop=True)

    assert len(port) == len(golden)
    assert np.allclose(
        port["relative_latency_s"].dropna().to_numpy(), golden["relative_latency_s"].dropna().to_numpy(),
    )


def test_wilcoxon_relative_latency_matches_golden(area_latency_df):
    rel_df = rl.compute_relative_to_reference(area_latency_df)
    stats_df = rl.wilcoxon_relative_latency(rel_df)
    # wilcoxon_relative_latency itself has no `symbol` column (source cell 33
    # adds it as a separate top-level statement) -- checked via p_to_symbol
    # directly, and again through summarize_relative_latency's own `symbol`
    # column below.
    stats_df = stats_df.assign(symbol=stats_df["p_value"].apply(rl.p_to_symbol))

    golden = pd.read_csv(GOLDEN_DIR / "OB_relative_latency_wilcoxon_stats.csv").set_index("area")
    port = stats_df.set_index("area").reindex(golden.index)

    assert (port["n"].to_numpy() == golden["n"].to_numpy()).all()
    assert np.allclose(port["statistic"].to_numpy(), golden["statistic"].to_numpy())
    assert np.allclose(port["p_value"].to_numpy(), golden["p_value"].to_numpy())
    assert (port["symbol"].to_numpy() == golden["symbol"].to_numpy()).all()


def test_summary_relative_latency_matches_golden(area_latency_df):
    rel_df = rl.compute_relative_to_reference(area_latency_df)
    stats_df = rl.wilcoxon_relative_latency(rel_df)
    summary_df = rl.summarize_relative_latency(rel_df, stats_df)

    golden = pd.read_csv(GOLDEN_DIR / "summary_relative_to_ob.csv").set_index("area")
    # summary_df's "area" is an AREA_ORDER-ordered Categorical (see
    # summarize_relative_latency) -- align on the plain-string golden index
    # rather than sort_values, whose categorical vs. alphabetical semantics
    # would otherwise disagree.
    port = summary_df.astype({"area": str}).set_index("area").reindex(golden.index)

    assert np.allclose(port["mean"].to_numpy(), golden["mean"].to_numpy())
    assert np.allclose(port["std"].dropna().to_numpy(), golden["std"].dropna().to_numpy())
    assert (port["symbol"].to_numpy() == golden["symbol"].to_numpy()).all()


def test_failed_detections_match_golden(area_latency_df):
    failures = rl.get_failed_detections(area_latency_df)
    golden = pd.read_csv(GOLDEN_DIR / "failed_area_latency_cases.csv", dtype={"fish_id": str})

    assert len(failures) == len(golden)
    assert set(zip(failures["fish_id"], failures["area"])) == set(zip(golden["fish_id"], golden["area"]))


def test_plots_render(area_latency_df, region_latency_df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paired, stats = rl.compare_forebrain_hindbrain(region_latency_df)
    fig1 = rl.plot_forebrain_hindbrain_latency(paired, stats)
    assert isinstance(fig1, plt.Figure)
    plt.close(fig1)

    rel_df = rl.compute_relative_to_reference(area_latency_df)
    rel_stats = rl.wilcoxon_relative_latency(rel_df)
    fig2 = rl.plot_relative_latency(rel_df, rel_stats)
    assert isinstance(fig2, plt.Figure)
    plt.close(fig2)
