"""Unit tests for jenkins_et_al.neural.build_brain_map against small
synthetic inputs (no dependency on the real data drive).

Covers the logic that isn't exercised by tests/golden/test_build_brain_map.py:
the slice_position validation/mapping and the not_assigned-row drop in
load_area_values. Plot-rendering behavior itself (colormap, vmin/vmax,
show_significance) is covered by tests/test_plotting.py against the shared
plot_brain_area_map.
"""
from __future__ import annotations

import io

import pandas as pd
import pytest

from jenkins_et_al.neural import build_brain_map as bbm


def test_slice_offsets_has_exactly_the_two_values_the_notebook_uses():
    assert bbm.SLICE_OFFSETS == {"lateral": 68, "medial": 10}


def test_plot_area_outline_map_defaults_to_lateral(monkeypatch):
    seen = {}

    def fake_plot_brain_area_map(ref_brain, areas, area_shorthand, *, slice_offset, **kwargs):
        seen["slice_offset"] = slice_offset
        return "figure"

    monkeypatch.setattr(bbm, "plot_brain_area_map", fake_plot_brain_area_map)
    bbm.plot_area_outline_map(ref_brain=None, areas={})

    assert seen["slice_offset"] == bbm.SLICE_OFFSETS["lateral"]


def test_plot_area_value_map_defaults_to_lateral(monkeypatch):
    seen = {}

    def fake_plot_brain_area_map(ref_brain, areas, area_shorthand, *, slice_offset, **kwargs):
        seen["slice_offset"] = slice_offset
        return "figure"

    monkeypatch.setattr(bbm, "plot_brain_area_map", fake_plot_brain_area_map)
    bbm.plot_area_value_map(ref_brain=None, areas={}, area_values=pd.DataFrame())

    assert seen["slice_offset"] == bbm.SLICE_OFFSETS["lateral"]


def test_plot_area_value_map_medial_reproduces_cell_9s_original_slice(monkeypatch):
    seen = {}

    def fake_plot_brain_area_map(ref_brain, areas, area_shorthand, *, slice_offset, **kwargs):
        seen["slice_offset"] = slice_offset
        return "figure"

    monkeypatch.setattr(bbm, "plot_brain_area_map", fake_plot_brain_area_map)
    bbm.plot_area_value_map(ref_brain=None, areas={}, area_values=pd.DataFrame(), slice_position="medial")

    assert seen["slice_offset"] == bbm.SLICE_OFFSETS["medial"]


def test_invalid_slice_position_raises_a_clear_error():
    with pytest.raises(ValueError, match="slice_position must be one of"):
        bbm.plot_area_outline_map(ref_brain=None, areas={}, slice_position="rostral")


def test_load_area_values_drops_not_assigned_row(tmp_path):
    csv_path = tmp_path / "area_values.csv"
    pd.DataFrame({
        "area_names": ["OB", "not_assigned", "Pal"],
        "values": [0.1, 0.9, 0.2],
        "p_value": [0.01, 1.0, 0.5],
    }).to_csv(csv_path, index=False)

    result = bbm.load_area_values(csv_path)

    assert "not_assigned" not in result["area_names"].to_numpy()
    assert len(result) == 2
