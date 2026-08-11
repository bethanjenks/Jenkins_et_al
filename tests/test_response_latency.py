"""Unit tests for jenkins_et_al.neural.response_latency against small
synthetic inputs (no dependency on the real data drive).

The full pipeline against real data is golden-verified separately in
tests/golden/test_response_latency.py.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from jenkins_et_al.neural import response_latency as rl


def _make_fish_df(area="olfactory_bulb", stimulus="ade", n_trials=3, trace_len=120, onset_frame=None):
    rng = np.random.default_rng(0)
    rows = []
    for trial in range(n_trials):
        trace = rng.normal(scale=0.01, size=trace_len)
        if onset_frame is not None:
            trace[onset_frame:] += 1.0
        rows.append({
            "fish_id": "230713_fb", "area": area, "stimulus": stimulus,
            "trace_serialised": json.dumps(trace.tolist()),
        })
    return pd.DataFrame(rows)


def test_parse_trace_handles_array_list_and_json_string():
    assert rl.parse_trace(np.array([1.0, 2.0])).tolist() == [1.0, 2.0]
    assert rl.parse_trace([1.0, 2.0]).tolist() == [1.0, 2.0]
    assert rl.parse_trace("[1.0, 2.0]").tolist() == [1.0, 2.0]


def test_parse_trace_handles_python_literal_string():
    assert rl.parse_trace("[1.0, 2.0, 3.0]").tolist() == [1.0, 2.0, 3.0]


def test_load_fish_df_finds_fish_by_suffixed_id(tmp_path):
    path = tmp_path / "fish.h5"
    df = pd.DataFrame({"fish_id": ["230713_fb", "230713_fb", "230714_fb"], "value": [1, 2, 3]})
    df.to_hdf(path, key="data", format="table", data_columns=["fish_id"])

    result = rl.load_fish_df(path, 230713, suffix="_fb")

    assert len(result) == 2
    assert (result["fish_id"] == "230713_fb").all()


def test_load_fish_df_raises_when_fish_not_found(tmp_path):
    path = tmp_path / "fish.h5"
    df = pd.DataFrame({"fish_id": ["230714_fb"], "value": [1]})
    df.to_hdf(path, key="data", format="table", data_columns=["fish_id"])

    with pytest.raises(ValueError, match="No data found"):
        rl.load_fish_df(path, 230713, suffix="_fb")


def test_make_mean_trace_averages_across_trials_for_one_area():
    fish_df = _make_fish_df(area="olfactory_bulb", n_trials=3, trace_len=50)
    fish_df = pd.concat([fish_df, _make_fish_df(area="pallium", n_trials=2, trace_len=50)], ignore_index=True)

    mean_trace, status, n_traces = rl.make_mean_trace(fish_df, area="olfactory_bulb", key_stimuli=("ade",))

    assert status == "ok"
    assert n_traces == 3
    assert mean_trace.shape == (50,)


def test_make_mean_trace_fails_when_area_has_no_key_stimulus_data():
    fish_df = _make_fish_df(area="olfactory_bulb", stimulus="ade")

    mean_trace, status, n_traces = rl.make_mean_trace(fish_df, area="pallium", key_stimuli=("ade",))

    assert mean_trace is None
    assert status == "failed_no_key_stimulus_data"
    assert n_traces == 0


def test_first_sustained_crossing_bool_single_frame():
    mask = np.array([False, False, True, True, False])
    assert rl.first_sustained_crossing_bool(mask, min_consecutive=1) == 2


def test_first_sustained_crossing_bool_requires_consecutive_frames():
    mask = np.array([True, False, True, True, True])
    assert rl.first_sustained_crossing_bool(mask, min_consecutive=3) == 2


def test_first_sustained_crossing_bool_returns_nan_when_never_true():
    assert np.isnan(rl.first_sustained_crossing_bool(np.array([False, False, False])))


def test_detect_onset_derivative_restricted_detects_a_clear_step():
    trace = np.zeros(150)
    trace[80:] = 1.0  # step well inside the search window

    detection = rl.detect_onset_derivative_restricted(trace)

    assert detection["status"] == "detected"
    # Gaussian smoothing (sigma=3) spreads the step's derivative response
    # slightly earlier than the true step location.
    assert detection["onset_idx"] == pytest.approx(80, abs=10)


def test_detect_onset_derivative_restricted_falls_back_for_a_slow_ramp():
    trace = np.zeros(150)
    ramp_start, ramp_end = 70, 140
    trace[ramp_start:ramp_end] = np.linspace(0, 1, ramp_end - ramp_start)
    trace[ramp_end:] = 1.0

    detection = rl.detect_onset_derivative_restricted(trace, derivative_threshold=10.0)

    assert detection["status"] == "detected_fallback"


def test_detect_onset_derivative_restricted_fails_on_empty_trace():
    detection = rl.detect_onset_derivative_restricted(np.array([]))
    assert detection["status"] == "failed_empty_trace"
    assert np.isnan(detection["latency_s"])


def test_process_fish_by_area_returns_one_row_per_area(monkeypatch):
    fish_df = pd.concat([
        _make_fish_df(area="olfactory_bulb", n_trials=3, trace_len=200, onset_frame=170),
        _make_fish_df(area="pallium", n_trials=3, trace_len=200, onset_frame=170),
    ], ignore_index=True)
    monkeypatch.setattr(rl, "load_fish_df", lambda *a, **k: fish_df)

    results = rl.process_fish_by_area(
        230713, "unused_path.h5", "forebrain", suffix="_fb", areas=("olfactory_bulb", "pallium"),
    )

    assert len(results) == 2
    assert {r["area"] for r in results} == {"olfactory_bulb", "pallium"}
    assert results[0]["fish_id"] == "230713"
    assert results[0]["dataset"] == "forebrain"


def test_process_whole_region_uses_all_areas(monkeypatch):
    fish_df = pd.concat([
        _make_fish_df(area="olfactory_bulb", n_trials=2, trace_len=200, onset_frame=170),
        _make_fish_df(area="pallium", n_trials=2, trace_len=200, onset_frame=170),
    ], ignore_index=True)
    monkeypatch.setattr(rl, "load_fish_df", lambda *a, **k: fish_df)

    result = rl.process_whole_region(230713, "unused_path.h5", "forebrain", suffix="_fb")

    assert result["fish_id"] == "230713"
    assert result["region"] == "forebrain"
    assert result["n_traces"] == 4


def test_apply_manual_override_sets_latency_and_status():
    df = pd.DataFrame({
        "fish_id": ["230713", "230811"], "region": ["hindbrain", "hindbrain"],
        "latency_s": [0.5, 0.667], "status": ["detected", "detected_fallback"],
    })

    result = rl.apply_manual_override(df, "230811", "region", "hindbrain", 8.0)

    assert result.loc[result["fish_id"] == "230811", "latency_s"].iloc[0] == 8.0
    assert result.loc[result["fish_id"] == "230811", "status"].iloc[0] == "manual"
    # Untouched row unaffected.
    assert result.loc[result["fish_id"] == "230713", "latency_s"].iloc[0] == 0.5


def _make_region_df():
    return pd.DataFrame({
        "fish_id": ["f1", "f1", "f2", "f2", "f3", "f3"],
        "region": ["forebrain", "hindbrain", "forebrain", "hindbrain", "forebrain", "hindbrain"],
        "latency_s": [1.0, 2.0, 1.5, 2.5, 2.0, 1.0],
        "status": ["detected"] * 6,
    })


def test_compare_forebrain_hindbrain_pairs_by_fish():
    paired, stats = rl.compare_forebrain_hindbrain(_make_region_df())

    assert list(paired.index) == ["f1", "f2", "f3"]
    assert paired.loc["f1", "forebrain"] == 1.0
    assert paired.loc["f1", "hindbrain"] == 2.0
    assert stats.loc[0, "n"] == 3


def test_compare_forebrain_hindbrain_excludes_fallback_status():
    df = _make_region_df()
    df.loc[df["fish_id"] == "f3", "status"] = "detected_fallback"

    paired, stats = rl.compare_forebrain_hindbrain(df)

    assert "f3" not in paired.index
    assert stats.loc[0, "n"] == 2


def test_compute_fb_hb_difference_stats():
    paired = pd.DataFrame({"forebrain": [1.0, 2.0], "hindbrain": [2.0, 4.0]})
    diff_stats = rl.compute_fb_hb_difference_stats(paired)
    assert diff_stats["mean"] == 1.5
    assert diff_stats["median"] == 1.5


def _make_area_df():
    rows = []
    for fish in ["f1", "f2", "f3"]:
        rows.append({"fish_id": fish, "area": "olfactory_bulb", "latency_s": 1.0, "status": "detected"})
        rows.append({"fish_id": fish, "area": "pallium", "latency_s": 2.0, "status": "detected"})
    return pd.DataFrame(rows)


def test_compute_relative_to_reference_subtracts_reference_area():
    rel_df = rl.compute_relative_to_reference(
        _make_area_df(), reference_area="olfactory_bulb", area_shorthand={"olfactory_bulb": "OB", "pallium": "Pal"},
    )

    ob_rows = rel_df[rel_df["area"] == "olfactory_bulb"]
    pal_rows = rel_df[rel_df["area"] == "pallium"]
    assert (ob_rows["relative_latency_s"] == 0.0).all()
    assert (pal_rows["relative_latency_s"] == 1.0).all()
    assert set(rel_df["area_label"]) == {"OB", "Pal"}


def test_compute_relative_to_reference_raises_when_reference_area_absent():
    with pytest.raises(ValueError, match="not found"):
        rl.compute_relative_to_reference(_make_area_df(), reference_area="missing_area")


def test_wilcoxon_relative_latency_skips_reference_area():
    rel_df = pd.DataFrame({
        "area": ["pallium"] * 6, "relative_latency_s": [1.0, 2.0, -1.0, 3.0, 1.5, 2.5],
    })
    stats_df = rl.wilcoxon_relative_latency(
        rel_df, reference_area="olfactory_bulb", area_order=("olfactory_bulb", "pallium"),
        area_shorthand={"pallium": "Pal"},
    )

    assert list(stats_df["area"]) == ["pallium"]
    assert stats_df.iloc[0]["n"] == 6


def test_p_to_symbol_thresholds():
    assert rl.p_to_symbol(0.0001) == "***"
    assert rl.p_to_symbol(0.005) == "**"
    assert rl.p_to_symbol(0.02) == "*"
    assert rl.p_to_symbol(0.5) == "ns"
    assert rl.p_to_symbol(np.nan) == "na"


def test_summarize_relative_latency_orders_by_area_order():
    rel_df = pd.DataFrame({
        "area": ["pallium", "pallium", "olfactory_bulb", "olfactory_bulb"],
        "area_label": ["Pal", "Pal", "OB", "OB"],
        "relative_latency_s": [1.0, 3.0, 0.0, 0.0],
    })
    stats_df = pd.DataFrame({"area": ["pallium"], "p_value": [0.03]})

    summary = rl.summarize_relative_latency(rel_df, stats_df, area_order=("olfactory_bulb", "pallium"))

    assert list(summary["area"]) == ["olfactory_bulb", "pallium"]
    assert summary.iloc[1]["mean"] == 2.0
    assert summary.iloc[1]["symbol"] == "*"


def test_summarize_relative_latency_symbol_falls_through_nan_to_ns():
    """Regression test for the cell-33-vs-cell-36 symbol inconsistency (see docstring):
    unlike p_to_symbol, summarize_relative_latency's own symbol has no NaN case."""
    rel_df = pd.DataFrame({
        "area": ["olfactory_bulb", "olfactory_bulb"], "area_label": ["OB", "OB"],
        "relative_latency_s": [0.0, 0.0],
    })
    stats_df = pd.DataFrame({"area": [], "p_value": []})  # reference area has no Wilcoxon row -> NaN p_value

    summary = rl.summarize_relative_latency(rel_df, stats_df, area_order=("olfactory_bulb",))

    assert np.isnan(summary.iloc[0]["p_value"])
    assert summary.iloc[0]["symbol"] == "ns"
    assert rl.p_to_symbol(summary.iloc[0]["p_value"]) == "na"  # p_to_symbol would say "na" instead


def test_get_failed_detections_excludes_only_fully_detected_rows():
    df = pd.DataFrame({"status": ["detected", "detected_fallback", "failed_no_valid_traces", "detected"]})
    failures = rl.get_failed_detections(df)
    assert len(failures) == 2


def test_plot_debug_trace_renders():
    trace = np.zeros(150)
    trace[80:] = 1.0
    detection = rl.detect_onset_derivative_restricted(trace)

    fig = rl.plot_debug_trace(trace, detection, 230713, "OB", "forebrain")

    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_forebrain_hindbrain_latency_renders():
    paired_region, stats = rl.compare_forebrain_hindbrain(_make_region_df())
    fig = rl.plot_forebrain_hindbrain_latency(paired_region, stats)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_relative_latency_renders():
    rel_df = rl.compute_relative_to_reference(
        _make_area_df(), reference_area="olfactory_bulb", area_shorthand={"olfactory_bulb": "OB", "pallium": "Pal"},
    )
    stats_df = rl.wilcoxon_relative_latency(
        rel_df, reference_area="olfactory_bulb", area_order=("olfactory_bulb", "pallium"),
        area_shorthand={"olfactory_bulb": "OB", "pallium": "Pal"},
    )

    fig = rl.plot_relative_latency(
        rel_df, stats_df, area_order=("olfactory_bulb", "pallium"), reference_area="olfactory_bulb",
        area_shorthand={"olfactory_bulb": "OB", "pallium": "Pal"}, palette={"OB": "black", "Pal": "blue"},
    )

    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_save_failure_debug_plots_writes_one_file_per_failure(tmp_path, monkeypatch):
    fish_df = _make_fish_df(area="pallium", n_trials=2, trace_len=200, onset_frame=None)
    monkeypatch.setattr(rl, "load_fish_df", lambda *a, **k: fish_df)

    area_latency_df = pd.DataFrame([{
        "fish_id": "230713", "dataset": "forebrain", "area": "pallium", "area_label": "Pal",
        "n_traces": 2, "onset_idx": np.nan, "latency_s": np.nan, "status": "failed_no_derivative_crossing",
    }])

    saved = rl.save_failure_debug_plots(area_latency_df, "fb.h5", "hb.h5", tmp_path)

    assert len(saved) == 1
    assert saved[0].exists()


def test_show_failure_debug_plot_opens_browser(monkeypatch, tmp_path):
    fish_df = _make_fish_df(area="pallium", n_trials=2, trace_len=200, onset_frame=None)
    monkeypatch.setattr(rl, "load_fish_df", lambda *a, **k: fish_df)

    opened = {}
    def fake_show_in_browser(fig, **kwargs):
        opened["called"] = True
        path = tmp_path / "shown.png"
        fig.savefig(path)
        return path
    monkeypatch.setattr(rl, "show_in_browser", fake_show_in_browser)

    area_latency_df = pd.DataFrame([{
        "fish_id": "230713", "dataset": "forebrain", "area": "pallium", "area_label": "Pal",
        "n_traces": 2, "onset_idx": np.nan, "latency_s": np.nan, "status": "failed_no_derivative_crossing",
    }])

    path = rl.show_failure_debug_plot("230713", "pallium", area_latency_df, "fb.h5", "hb.h5")

    assert opened["called"]
    assert path.exists()
