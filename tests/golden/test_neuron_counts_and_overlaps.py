"""Regression test for jenkins_et_al.neural.neuron_counts_and_overlaps
against a golden capture derived from
notebooks/neural/neuron_counts_and_overlaps.ipynb.

golden/neuron_counts_and_overlaps/ was produced by running the ported
module against real data, after independently verifying it against the
notebook's own cell code, copied verbatim and run standalone on identical
inputs -- including the call-order-dependent global `stim_df['fish_id']`
mutation (see module docstring in `neuron_counts_and_overlaps.py`), which
the fixture below reproduces by calling functions in the same order as the
notebook's own cells.

Skipped entirely if the source data drive isn't mounted.
"""
import json
from pathlib import Path

import pandas as pd
import pytest

from jenkins_et_al.neural.area_pair_core import load_nmlf_mask
from jenkins_et_al.neural.cross_regressor_neuron_correlations import select_neuron_groups
from jenkins_et_al.neural.neuron_counts_and_overlaps import (
    compute_area_fish_counts,
    compute_area_norm_counts,
    compute_area_pos_neg_summary,
    compute_bcn_overlap,
    compute_fish_valence_fractions,
    compute_valence_bhv_overlap,
    count_high_bhv_high_neg,
    filter_to_brain_areas,
    get_overlap_neuron_info,
    select_bhv_valence_neurons,
    summarize_bhv_valence_overlap,
    summarize_fish_per_area,
)
from jenkins_et_al.neural.valence_bhv_core import load_valence_bhv_dataframe

DATA_DIR = Path("/Volumes/LaCie/larval_HuC/imaging")
POS_PATH = DATA_DIR / "7dpf_positive_correlations_25s_no_HCl_1000_corrected.csv"
NEG_PATH = DATA_DIR / "7dpf_negative_correlations_25s_no_HCl_1000_corrected.csv"
BHV_PATH = DATA_DIR / "7dpf_bhv_vigor_correlations_1000_corrected.csv"
NMLF_MASK_PATH = Path("/Volumes/LaCie/area_masks/area_masks_mapzebrain/nMLF.tiff")
AREA_SHORTHAND_PATH = Path("/Volumes/LaCie/larval_HuC/imaging/brain_map_values/area_shorthand_dict.json")
GOLDEN_DIR = Path(__file__).resolve().parent.parent.parent / "golden" / "neuron_counts_and_overlaps"

pytestmark = pytest.mark.skipif(
    not POS_PATH.exists() or not GOLDEN_DIR.exists(),
    reason="source data drive or golden capture not available on this machine",
)


def _assert_frame_matches_golden(df: pd.DataFrame, golden_csv: Path) -> None:
    golden = pd.read_csv(golden_csv, index_col=0)
    golden.index = golden.index.astype(str)
    df = df.copy()
    df.index = df.index.astype(str)
    pd.testing.assert_frame_equal(df, golden, check_dtype=False, check_names=False, check_index_type=False)


@pytest.fixture(scope="module")
def results():
    nmlf_mask = load_nmlf_mask(NMLF_MASK_PATH)
    stim_df = load_valence_bhv_dataframe(POS_PATH, NEG_PATH, BHV_PATH, nmlf_mask)
    area_shorthand_dict = json.loads(AREA_SHORTHAND_PATH.read_text())

    stim_df_fb = filter_to_brain_areas(stim_df)
    area_fish_counts, area_fish_frac = compute_area_fish_counts(stim_df_fb, area_shorthand_dict)
    fish_per_area = summarize_fish_per_area(area_fish_counts)
    fish_valence_fractions = compute_fish_valence_fractions(stim_df)  # mutates stim_df['fish_id']
    area_pos_neg_summary = compute_area_pos_neg_summary(stim_df_fb, area_shorthand_dict)
    overlap = compute_valence_bhv_overlap(stim_df)
    bcn = compute_bcn_overlap(stim_df)
    overlap_info = get_overlap_neuron_info(stim_df, overlap["both"])
    area_norm_counts = compute_area_norm_counts(stim_df, area_shorthand_dict)
    bhv_valence_neurons = select_bhv_valence_neurons(stim_df)
    bhv_valence_summary = summarize_bhv_valence_overlap(bhv_valence_neurons)
    groups = select_neuron_groups(stim_df)
    high_bhv_high_neg_count = count_high_bhv_high_neg(groups["high_bhv"])

    return {
        "area_fish_counts": area_fish_counts,
        "area_fish_frac": area_fish_frac,
        "fish_per_area": fish_per_area,
        "fish_valence_fractions": fish_valence_fractions,
        "area_pos_neg_summary": area_pos_neg_summary,
        "area_norm_counts": area_norm_counts,
        "overlap_sizes": {
            "only_valence": len(overlap["only_valence"]),
            "only_bhv": len(overlap["only_bhv"]),
            "both": len(overlap["both"]),
            "bhv_1": len(bcn["bhv_1"]),
            "bhv_2": len(bcn["bhv_2"]),
            "bcn_both": len(bcn["both"]),
            "overlap_info_n": len(overlap_info),
            "bhv_valence_summary": bhv_valence_summary,
            "high_bhv_high_neg_count": high_bhv_high_neg_count,
        },
    }


@pytest.mark.parametrize("key,filename", [
    ("area_fish_counts", "area_fish_counts.csv"),
    ("area_fish_frac", "area_fish_frac.csv"),
    ("fish_per_area", "fish_per_area.csv"),
    ("fish_valence_fractions", "fish_valence_fractions.csv"),
    ("area_pos_neg_summary", "area_pos_neg_summary.csv"),
    ("area_norm_counts", "area_norm_counts.csv"),
])
def test_frame_matches_golden(results, key, filename):
    _assert_frame_matches_golden(results[key], GOLDEN_DIR / filename)


def test_overlap_sizes_match_golden(results):
    golden = json.loads((GOLDEN_DIR / "overlap_sizes.json").read_text())
    assert results["overlap_sizes"] == golden
