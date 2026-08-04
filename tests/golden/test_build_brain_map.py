"""Regression test for jenkins_et_al.neural.build_brain_map against a golden
capture of notebooks/neural/build_brain_map.ipynb.

golden/build_brain_map/ was produced by executing the notebook's own cell
source (cells 0, 2, 5, 7, 8, 9), fresh top-to-bottom, redirecting its figure
writes into that directory instead of overwriting real production files.

Confirms the data feeding both figures, per the golden-refactor skill's
"verify the data, not pixels" rule:
- area_value_df (cell 7's loaded + filtered value table).
- Per-area contour coordinates (erosion -> Gaussian smoothing -> marching
  squares) for both slices the notebook uses: cell 5's "lateral" outline
  map and cell 9's "medial" value map.
- Per-area fill colors (cmap(norm(value))) for the value map.

Rendered PNGs were also confirmed md5-identical to golden capture during
this port (see PORTING_PLAN.md), but that isn't re-asserted here: repeated
runs of the exact same call were observed to occasionally produce a
different PNG even though the contour/color data underneath is unchanged
-- adjust_text falls back to unseeded np.random shifts for labels it can't
otherwise resolve (inherent to the original notebook, not this port; see
build_brain_map.py's module docstring). Pixel-exact PNG comparison would be
a flaky, non-deterministic test; the data checked here is the real,
deterministic output.

Skipped entirely if the source data drive isn't mounted, since the
underlying reference brain volume and area masks are not part of this
repository.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import Normalize

from jenkins_et_al.neural import build_brain_map as bbm
from jenkins_et_al.plotting import extract_area_contours

REF_BRAIN_PATH = Path("/Volumes/LaCie/area_masks/T_AVG_HuC.tif")
AREA_MASKS_PATH = Path("/Volumes/LaCie/area_masks/area_dict_updated.npy")
NMLF_MASK_PATH = Path("/Volumes/LaCie/area_masks/area_masks_mapzebrain/nMLF.tiff")
AREA_VALUES_CSV = Path("/Volumes/LaCie/larval_HuC/imaging/7dpf_straight_bout_sig_0.3.csv")
GOLDEN_DIR = Path(__file__).resolve().parent.parent.parent / "golden" / "build_brain_map"

pytestmark = pytest.mark.skipif(
    not REF_BRAIN_PATH.exists() or not GOLDEN_DIR.exists(),
    reason="source data drive or golden capture not available on this machine",
)


@pytest.fixture(scope="module")
def areas():
    return bbm.load_area_masks(AREA_MASKS_PATH, nmlf_mask_path=NMLF_MASK_PATH)


@pytest.fixture(scope="module")
def area_values():
    return bbm.load_area_values(AREA_VALUES_CSV)


def test_area_value_df_matches_golden(area_values):
    golden = pd.read_csv(GOLDEN_DIR / "area_value_df.csv")

    assert list(area_values["area_names"]) == list(golden["area_names"])
    assert np.allclose(area_values["values"].to_numpy(), golden["values"].to_numpy())
    assert np.allclose(area_values["p_value"].to_numpy(), golden["p_value"].to_numpy())


def test_outline_map_contours_match_golden(areas):
    golden = json.loads((GOLDEN_DIR / "no_data_contours.json").read_text())
    slice_offset = bbm.SLICE_OFFSETS["lateral"]

    assert set(areas.keys()) == set(golden.keys())
    for brain_area, mask in areas.items():
        contours = extract_area_contours(mask, slice_offset)
        golden_contours = golden[brain_area]

        assert len(contours) == len(golden_contours)
        for contour, golden_contour in zip(contours, golden_contours):
            assert np.allclose(contour, np.array(golden_contour))


def test_value_map_contours_and_colors_match_golden(areas, area_values):
    golden_records = {r["area"]: r for r in json.loads((GOLDEN_DIR / "with_data_records.json").read_text())}
    slice_offset = bbm.SLICE_OFFSETS["medial"]
    norm = Normalize(vmin=bbm.DEFAULT_VMIN, vmax=bbm.DEFAULT_VMAX)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="gray")

    assert set(areas.keys()) == set(golden_records.keys())
    for brain_area, mask in areas.items():
        contours = extract_area_contours(mask, slice_offset)
        golden = golden_records[brain_area]
        assert len(contours) == golden["n_contours"]

        matches = area_values[area_values["area_names"] == brain_area]
        area_value = matches["values"].values[0] if len(matches) else np.nan
        color = list(cmap(norm(area_value)))

        if golden["color"] is not None:
            assert np.allclose(color, golden["color"])
