"""Whole-brain area map: per-area contour outlines, optionally colored by a
value with a significance-highlighted outline.

Ported from notebooks/neural/build_brain_map.ipynb; logic unchanged from the
notebook, only moved into a module and split into loading/plotting
functions. The two plots (cell 5, outline-only; cell 9, value-filled) are
now the same general `jenkins_et_al.plotting.plot_brain_area_map` called
with different arguments -- `plot_area_outline_map`/`plot_area_value_map`
below are thin wrappers pinning each cell's exact original parameter values
(figsize, alpha, slice offset, fontsize, colormap, vmin/vmax, colorbar
ticks); `plot_brain_area_map` itself exposes colormap, vmin/vmax, and the
p-value-driven significance outline as caller-facing options, since the
notebook's own comment on cell 7 notes the p-value column is "optional based
on the analysis" -- not every value table this plot might be given has one.

**Unseeded non-determinism, inherent to the original notebook, not this
port**: `adjust_text` (called by `plot_brain_area_map` to de-overlap area
labels) falls back to `np.random.rand`-based random shifts for labels it
can't otherwise resolve, with no seed set anywhere in the notebook. Running
the exact same call twice in one process was observed to occasionally
produce a different PNG (confirmed: contour geometry and fill colors were
still byte-identical across runs -- only a couple of tightly-packed labels'
pixel positions moved). The golden capture's rendered PNGs happened to be
md5-identical to this port's output on the runs checked, but that specific
byte-equality isn't a guarantee for every run -- `tests/golden/test_build_brain_map.py`
therefore asserts the underlying contour/color data (the real, deterministic
output), not rendered-image bytes. Not something to fix by seeding the
original notebook (out of scope per this skill's rules); flagged here so a
future re-run's differing PNG isn't mistaken for a regression.

`slice_position` on both plot wrappers below defaults to `"lateral"`
(offset 68) at the user's explicit request (2026-08-04) -- cell 5's own
figure already used this slice, but cell 9's original used `"medial"`
(offset 10); pass `slice_position="medial"` to reproduce cell 9's exact
original output (verified against golden capture in
tests/golden/test_build_brain_map.py).

Reuses `jenkins_et_al.config.AREA_SHORTHAND` (confirmed byte-identical to
the notebook's own runtime load of
`/Volumes/LaCie/larval_HuC/imaging/brain_map_values/area_shorthand_dict.json`,
cell 2) instead of loading the JSON file a second time.

No confirmed in-repo consumer notebook of this plot as of this port (per
PORTING_PLAN.md) -- it establishes the reference brain volume, area mask
dict, and shorthand dict that later neural notebooks
(`functional_clustering_valence_neurons.ipynb`,
`plot_regression_spatial_maps.ipynb`) also load. Those two notebooks were
checked while porting this one: both use the *different* per-neuron-scatter
pattern (`jenkins_et_al.plotting.plot_neuron_scatter_on_brain`), not this
area-contour-map pattern -- see that function's module docstring for the
correction to an earlier (unverified) note that had grouped all three
together.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from skimage import io

from jenkins_et_al.config import AREA_SHORTHAND
from jenkins_et_al.plotting import plot_brain_area_map

#: Source: cell 9's inline comment `10 ("medial") 68 ("lateral")` -- the
#: only two A-P slice offsets (from mid-volume) actually used anywhere in
#: this notebook: cell 5's outline map at "lateral", cell 9's value map at
#: "medial".
SLICE_OFFSETS: dict[str, int] = {"lateral": 68, "medial": 10}


def _resolve_slice_offset(slice_position: str) -> int:
    if slice_position not in SLICE_OFFSETS:
        raise ValueError(f"slice_position must be one of {tuple(SLICE_OFFSETS)}, got {slice_position!r}")
    return SLICE_OFFSETS[slice_position]

#: Source: cell 9 (`colors.Normalize(vmin=0, vmax=0.1)` and
#: `cbar.set_ticks([0, 0.1])`). Specific to the notebook's own example value
#: table (a fraction in [0, ~0.06]) -- pass different `vmin`/`vmax` to
#: `plot_area_value_map` for other analyses' value ranges.
DEFAULT_VMIN = 0.0
DEFAULT_VMAX = 0.1


# ---------------------------------------------------------------------------
# Load reference brain + area masks + values (source: cells 2, 7)
# ---------------------------------------------------------------------------

def load_reference_brain(ref_brain_path: Path) -> np.ndarray:
    """Load the reference brain image volume (e.g. a Mapzebrain average `.tif`)."""
    return io.imread(ref_brain_path)


def load_area_masks(area_masks_path: Path, nmlf_mask_path: Path | None = None) -> dict[str, np.ndarray]:
    """Load the area-name -> 3D mask dict, optionally adding an `"nMLF"` entry.

    `nmlf_mask_path`, if given, is loaded and inserted under the `"nMLF"`
    key -- matching the notebook's own extra load-and-merge step (cell 2),
    which adds this one area mask from a separate file.
    """
    areas = np.load(area_masks_path, allow_pickle=True).item()
    if nmlf_mask_path is not None:
        areas["nMLF"] = io.imread(nmlf_mask_path)
    return areas


def load_area_values(csv_path: Path) -> pd.DataFrame:
    """Load a per-area value table and drop its `"not_assigned"` row. Source: cell 7.

    Expects columns `"area_names"` and `"values"`, and optionally
    `"p_value"` -- see `jenkins_et_al.plotting.plot_brain_area_map`'s
    docstring for exactly what each column drives.
    """
    df = pd.read_csv(csv_path)
    return df[df["area_names"] != "not_assigned"]


# ---------------------------------------------------------------------------
# Plots (source: cells 5, 9) -- thin wrappers around the shared
# `plotting.plot_brain_area_map`, pinning each cell's original parameters.
# ---------------------------------------------------------------------------

def plot_area_outline_map(
    ref_brain: np.ndarray,
    areas: dict[str, np.ndarray],
    area_shorthand: dict[str, str] = AREA_SHORTHAND,
    slice_position: Literal["lateral", "medial"] = "lateral",
):
    """Plain black-outline area map, no fill or colorbar.

    Source: cell 5, which used the `"lateral"` slice -- also this
    function's default.
    """
    return plot_brain_area_map(
        ref_brain, areas, area_shorthand,
        slice_offset=_resolve_slice_offset(slice_position), area_values=None,
        background_alpha=0.5, figsize=(20, 15), label_fontsize=15,
    )


def plot_area_value_map(
    ref_brain: np.ndarray,
    areas: dict[str, np.ndarray],
    area_values: pd.DataFrame,
    area_shorthand: dict[str, str] = AREA_SHORTHAND,
    slice_position: Literal["lateral", "medial"] = "lateral",
    cmap: str = "viridis",
    vmin: float = DEFAULT_VMIN,
    vmax: float = DEFAULT_VMAX,
    show_significance: bool = True,
    colorbar_label: str = "Fraction",
):
    """Value-filled area map with a colorbar and optional significance-highlighted outlines.

    Source: cell 9, which used the `"medial"` slice -- this function's
    default is `"lateral"` instead, at the user's request, since that's the
    slice they expect to use for most calls. Pass `slice_position="medial"`
    to reproduce cell 9's exact original output. `cmap`, `vmin`/`vmax`, and
    `show_significance` are the notebook's own hardcoded choices (viridis;
    0-0.1; always-on p<0.05 orange outlines) turned into parameters -- see
    `jenkins_et_al.plotting.plot_brain_area_map` for what each controls and
    what `area_values` needs to provide.
    """
    return plot_brain_area_map(
        ref_brain, areas, area_shorthand,
        slice_offset=_resolve_slice_offset(slice_position), area_values=area_values,
        show_significance=show_significance,
        cmap=cmap, vmin=vmin, vmax=vmax,
        colorbar_label=colorbar_label, colorbar_ticks=[vmin, vmax],
        background_alpha=0.001, fill_alpha=0.8, figsize=(20, 20),
        label_fontsize=16, colorbar_fontsize=30,
    )
