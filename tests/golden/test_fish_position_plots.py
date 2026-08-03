"""Regression test for jenkins_et_al.behaviour.fish_position_plots against a
golden capture of notebooks/behaviour_free_swim/fish_position_plots.ipynb
(source data as currently laid out under /Volumes/LaCie; notebook itself is
not pinned to a frozen commit since it postdates the 2d19f3a baseline).

The golden capture (golden/fish_position_plots/) was produced by executing
the notebook's own cell source directly (cells 0, 1, 3, 4, 6, 7, 9), saving
every array feeding its two figures as .npy alongside a rendered .png of
each -- the notebook itself never saves anything to disk (its two
`plt.savefig` calls are commented out).

Skipped entirely if the source data drive isn't mounted, since the
underlying per-fish tracking CSVs are not part of this repository.
"""
from pathlib import Path

import numpy as np
import pytest

from jenkins_et_al.behaviour import fish_position_plots as fpp

ROOT_DIR = Path("/Volumes/LaCie/free_swimming/7dpf_TRex/adenosine_2.5mM")
GOLDEN_DIR = Path(__file__).resolve().parent.parent.parent / "golden" / "fish_position_plots"

pytestmark = pytest.mark.skipif(
    not ROOT_DIR.exists() or not GOLDEN_DIR.exists(),
    reason="source data drive or golden capture not available on this machine",
)


def _assert_allclose_to_golden(name: str, arr: np.ndarray) -> None:
    golden = np.load(GOLDEN_DIR / f"{name}.npy")
    assert np.allclose(arr, golden, rtol=1e-10, atol=1e-12, equal_nan=True), f"{name} mismatch"


@pytest.fixture(scope="module")
def stim_fish():
    return fpp.load_dataframes_from_subfolders(ROOT_DIR / "stim")


@pytest.fixture(scope="module")
def control_fish():
    return fpp.load_dataframes_from_subfolders(ROOT_DIR / "base")


@pytest.fixture(scope="module")
def filtered_stim(stim_fish):
    return fpp.filter_significantly_moving(stim_fish)


@pytest.fixture(scope="module")
def filtered_control(control_fish):
    return fpp.filter_significantly_moving(control_fish)


@pytest.fixture(scope="module")
def difference_heatmap_data(filtered_stim, filtered_control):
    return fpp.compute_difference_heatmap(filtered_stim, filtered_control)


def test_fish_counts_match_golden(stim_fish, filtered_stim, control_fish, filtered_control):
    golden_counts = tuple(int(c) for c in np.load(GOLDEN_DIR / "counts.npy"))
    counts = (len(stim_fish), len(filtered_stim), len(control_fish), len(filtered_control))
    assert counts == golden_counts


@pytest.mark.parametrize(
    "field",
    ["stim_heatmap", "control_heatmap", "difference_heatmap", "xedges", "yedges", "diff_x_hist", "diff_y_hist"],
)
def test_difference_heatmap_data_matches_golden(difference_heatmap_data, field):
    _assert_allclose_to_golden(field, getattr(difference_heatmap_data, field))


@pytest.fixture(scope="module")
def fish59(filtered_stim):
    return fpp.prepare_fish_trajectory(filtered_stim[59])


def test_prepare_fish_trajectory_does_not_mutate_input(filtered_stim):
    fish_before = filtered_stim[59].copy()
    fpp.prepare_fish_trajectory(filtered_stim[59])
    assert filtered_stim[59].equals(fish_before)


@pytest.mark.parametrize("column, golden_name", [
    ("x", "fish59_x"),
    ("y", "fish59_y"),
    ("time", "fish59_time"),
    ("time_normalized", "fish59_time_normalized"),
])
def test_fish59_trajectory_matches_golden(fish59, column, golden_name):
    _assert_allclose_to_golden(golden_name, fish59[column].to_numpy())
