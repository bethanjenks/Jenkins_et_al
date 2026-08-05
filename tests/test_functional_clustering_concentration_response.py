"""Unit tests for jenkins_et_al.neural.functional_clustering_concentration_response
against small synthetic inputs (no dependency on the real data drive).

The full pipeline against real data is golden-verified separately in
tests/golden/test_functional_clustering_concentration_response.py.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from jenkins_et_al.neural import functional_clustering_concentration_response as fccr


def _make_response_data():
    stimuli = ["low", "mid", "high"]
    rows = []
    for fish in ["fish1", "fish2"]:
        for neuron in range(6):
            neuron_id = f"{fish}_n{neuron}"
            # Two trials per (fish, neuron, stimulus) to exercise the averaging step.
            for trial in range(2):
                for stim in stimuli:
                    rows.append({
                        "fish_id": fish,
                        "neuron_id": neuron_id,
                        "stimulus": stim,
                        "coords": (float(neuron), 0.0, 0.0),
                        "area": "area_a",
                        "resp": float(neuron) + 0.01 * trial,
                    })
    return pd.DataFrame(rows), stimuli


def test_prepare_concentration_data_averages_trials_and_pivots():
    data, stimuli = _make_response_data()
    odorant_config = {"stimuli": stimuli, "labels": stimuli, "short_name": "x"}

    pivot_df, averaged_responses = fccr.prepare_concentration_data(data, odorant_config, regression_data=None)

    assert set(pivot_df.columns) >= {"fish_id", "neuron_id", "area", *stimuli}
    assert len(pivot_df) == 12  # 2 fish x 6 neurons
    # Averaging over the two trials should collapse the +/- 0.01 jitter away from raw resp.
    assert np.allclose(pivot_df["low"].values, pivot_df["neuron_id"].apply(lambda n: float(n.split("n")[1])).values, atol=0.02)
    assert "coords" in averaged_responses.columns


def test_prepare_concentration_data_applies_correlation_filter():
    data, stimuli = _make_response_data()
    odorant_config = {"stimuli": stimuli, "labels": stimuli, "short_name": "x"}

    coords = data[["coords", "area"]].drop_duplicates()
    coords["high_correlation"] = [0.9 if i % 2 == 0 else 0.1 for i in range(len(coords))]
    regression_data = coords

    pivot_df, _ = fccr.prepare_concentration_data(data, odorant_config, regression_data=regression_data, correlation_threshold=0.3)

    assert len(pivot_df) < 12
    assert len(pivot_df) > 0


def test_perform_clustering_is_deterministic():
    rng = np.random.default_rng(0)
    group_a = rng.normal(loc=[0, 0, 0], scale=0.1, size=(10, 3))
    group_b = rng.normal(loc=[5, 5, 5], scale=0.1, size=(10, 3))
    values = np.vstack([group_a, group_b])
    pivot_df = pd.DataFrame(values, columns=["low", "mid", "high"])
    pivot_df["fish_id"] = "fish1"
    pivot_df["neuron_id"] = range(len(pivot_df))
    pivot_df["area"] = "area_a"

    result1, _, _ = fccr.perform_clustering(pivot_df.copy(), ["low", "mid", "high"], n_clusters=2)
    result2, _, _ = fccr.perform_clustering(pivot_df.copy(), ["low", "mid", "high"], n_clusters=2)

    assert result1["cluster"].tolist() == result2["cluster"].tolist()
    assert result1["cluster"].nunique() == 2


def test_assign_cluster_names_maps_ids_to_labels():
    pivot_df = pd.DataFrame({"cluster": [0, 0, 1, 1]})

    named = fccr.assign_cluster_names(pivot_df, {0: "Low-threshold", 1: "High-threshold"})

    assert named["response_condition"].tolist() == ["Low-threshold", "Low-threshold", "High-threshold", "High-threshold"]


def test_assign_cluster_names_raises_on_missing_cluster():
    pivot_df = pd.DataFrame({"cluster": [0, 1, 2]})

    with pytest.raises(ValueError, match="Missing clusters"):
        fccr.assign_cluster_names(pivot_df, {0: "a", 1: "b"})


def test_calculate_response_fractions_sums_to_one():
    pivot_df = pd.DataFrame({"response_condition": ["a", "a", "a", "b"]})

    fractions_df = fccr.calculate_response_fractions(pivot_df)

    assert np.isclose(fractions_df["fraction"].sum(), 1.0)
    assert fractions_df.set_index("response_condition").loc["a", "count"] == 3


def test_calculate_response_fractions_respects_cluster_order():
    pivot_df = pd.DataFrame({"response_condition": ["b", "a", "a", "b", "b"]})

    fractions_df = fccr.calculate_response_fractions(pivot_df, cluster_order=["a", "b"])

    assert fractions_df["response_condition"].tolist() == ["a", "b"]


def test_merge_coordinates_reattaches_coords_per_neuron():
    pivot_df = pd.DataFrame({"fish_id": ["f1", "f1"], "neuron_id": ["n1", "n2"]})
    averaged_responses = pd.DataFrame({
        "fish_id": ["f1", "f1", "f1", "f1"],
        "neuron_id": ["n1", "n1", "n2", "n2"],
        "coords": [(1, 1, 1)] * 2 + [(2, 2, 2)] * 2,
    })

    merged = fccr.merge_coordinates(pivot_df, averaged_responses)

    assert len(merged) == 2
    assert merged.loc[merged["neuron_id"] == "n1", "coords"].iloc[0] == (1, 1, 1)
    assert merged.loc[merged["neuron_id"] == "n2", "coords"].iloc[0] == (2, 2, 2)


def test_plot_concentration_response_curves_renders_both_modes():
    pivot_df = pd.DataFrame({
        "cluster": [0, 0, 1, 1],
        "response_condition": ["High-threshold", "High-threshold", "Low-threshold", "Low-threshold"],
        "low": [0.1, 0.2, 0.8, 0.9],
        "mid": [0.2, 0.3, 0.8, 0.85],
        "high": [0.9, 0.95, 0.8, 0.82],
    })

    fig_ids = fccr.plot_concentration_response_curves(pivot_df, ["low", "mid", "high"], ["Low", "Mid", "High"], use_cluster_ids=True)
    fig_named = fccr.plot_concentration_response_curves(
        pivot_df, ["low", "mid", "high"], ["Low", "Mid", "High"],
        cluster_color_map=fccr.CLUSTER_COLORS, use_cluster_ids=False,
    )

    assert isinstance(fig_ids, plt.Figure)
    assert isinstance(fig_named, plt.Figure)
    plt.close(fig_ids)
    plt.close(fig_named)


def test_plot_response_conditions_2view_renders():
    coords = np.array([[i, i, i] for i in range(20)], dtype=float)
    conditions = np.array(["High-threshold", "Low-threshold"] * 10)
    ref_brain = np.random.default_rng(0).normal(size=(10, 20, 20))

    fig = fccr.plot_response_conditions_2view(coords, conditions, ref_brain, fccr.CLUSTER_COLORS, downsampling=1)

    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_silhouette_analysis_returns_scores_for_each_n():
    rng = np.random.default_rng(1)
    scaled_data = np.vstack([
        rng.normal(loc=[0, 0], scale=0.1, size=(10, 2)),
        rng.normal(loc=[5, 5], scale=0.1, size=(10, 2)),
    ])

    fig, scores = fccr.plot_silhouette_analysis(scaled_data, cluster_range=range(2, 4))

    assert set(scores.keys()) == {2, 3}
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_show_cluster_id_curves_opens_a_png_in_the_browser(monkeypatch):
    opened = {}
    monkeypatch.setattr("webbrowser.open", lambda url: opened.setdefault("url", url))

    pivot_df = pd.DataFrame({
        "cluster": [0, 0, 1, 1],
        "low": [0.1, 0.2, 0.8, 0.9],
        "mid": [0.2, 0.3, 0.8, 0.85],
        "high": [0.9, 0.95, 0.8, 0.82],
    })

    path = fccr.show_cluster_id_curves(pivot_df, ["low", "mid", "high"], ["Low", "Mid", "High"])

    assert "url" in opened
    assert path.exists()
    assert path.suffix == ".png"
    path.unlink()


def test_show_silhouette_analysis_opens_a_png_in_the_browser(monkeypatch):
    opened = {}
    monkeypatch.setattr("webbrowser.open", lambda url: opened.setdefault("url", url))

    rng = np.random.default_rng(1)
    scaled_data = np.vstack([
        rng.normal(loc=[0, 0], scale=0.1, size=(10, 2)),
        rng.normal(loc=[5, 5], scale=0.1, size=(10, 2)),
    ])

    scores, path = fccr.show_silhouette_analysis(scaled_data, cluster_range=range(2, 4))

    assert set(scores.keys()) == {2, 3}
    assert "url" in opened
    assert path.exists()
    path.unlink()
