"""Unit tests for jenkins_et_al.neural.functional_clustering_valence_neurons
against small synthetic inputs (no dependency on the real data drive).

The full pipeline against real data is golden-verified separately in
tests/golden/test_functional_clustering_valence_neurons.py.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from jenkins_et_al.neural import functional_clustering_valence_neurons as fcvn


def _make_regression_tables():
    pos_regress = pd.DataFrame({
        "coords": ["[1.0, 1.0, 1.0]", "[2.0, 2.0, 2.0]", "[3.0, 3.0, 3.0]"],
        "correlation": [0.6, 0.7, 0.05],
        "p_fdr": [0.01, 0.02, 0.5],
        "valence": [1, 1, 1],
    })
    neg_regress = pd.DataFrame({
        "coords": ["[4.0, 4.0, 4.0]", "[5.0, 5.0, 5.0]"],
        "correlation": [-0.6, -0.01],
        "p_fdr": [0.03, 0.9],
        "valence": [0, 0],
    })
    return pos_regress, neg_regress


def test_select_neurons_by_mode_combined_concatenates_and_labels_valence():
    pos_regress, neg_regress = _make_regression_tables()

    valence, description = fcvn.select_neurons_by_mode(pos_regress, neg_regress, "combined")

    assert "positive + negative" in description.lower() or "combined" in description.lower()
    # Only p_fdr < 0.05 rows survive: 2 positive, 1 negative.
    assert len(valence) == 3
    assert (valence["valence"] == 1).sum() == 2
    assert (valence["valence"] == 0).sum() == 1


def test_select_neurons_by_mode_positive_only():
    pos_regress, neg_regress = _make_regression_tables()

    valence, description = fcvn.select_neurons_by_mode(pos_regress, neg_regress, "positive_only")

    assert len(valence) == 2
    assert (valence["valence"] == 1).all()
    assert "positive" in description.lower()


def test_select_neurons_by_mode_negative_only():
    pos_regress, neg_regress = _make_regression_tables()

    valence, description = fcvn.select_neurons_by_mode(pos_regress, neg_regress, "negative_only")

    assert len(valence) == 1
    assert (valence["valence"] == 0).all()
    assert "negative" in description.lower()


def test_merge_valence_with_responses_inner_joins_on_coords():
    valence = pd.DataFrame({
        "coords": ["[1.0, 1.0, 1.0]", "[2.0, 2.0, 2.0]"],
        "correlation": [0.6, 0.7],
        "p_fdr": [0.01, 0.02],
        "valence": [1, 1],
    })
    data_resp = pd.DataFrame({
        "coords": ["[1.0, 1.0, 1.0]", "[1.0, 1.0, 1.0]", "[9.0, 9.0, 9.0]"],
        "fish_id": ["f1", "f1", "f2"],
        "neuron_id": ["n1", "n1", "n9"],
        "area": ["a", "a", "b"],
        "stimulus": ["ade", "kw", "ade"],
        "trial_number": [1, 1, 1],
        "resp": [0.1, 0.2, 0.9],
    })

    merged = fcvn.merge_valence_with_responses(valence, data_resp)

    # Only coords "[1.0, 1.0, 1.0]" is present in both tables.
    assert len(merged) == 2
    assert (merged["coords"] == "[1.0, 1.0, 1.0]").all()
    assert set(merged["stimulus"]) == {"ade", "kw"}


def test_average_and_pivot_responses_averages_trials_and_drops_incomplete_neurons():
    stimuli = ["ade", "kw"]
    rows = []
    # Complete neuron: both stimuli, 2 trials each -> should survive and be averaged.
    for stim, base in [("ade", 0.2), ("kw", 0.8)]:
        for trial in (1, 2):
            rows.append({
                "fish_id": "f1", "neuron_id": "n1", "stimulus": stim,
                "coords": "[1.0, 1.0, 1.0]", "area": "a", "valence": 1,
                "resp": base + 0.01 * trial,
            })
    # Incomplete neuron: only "ade" -> should be dropped by dropna(subset=stimuli).
    rows.append({
        "fish_id": "f2", "neuron_id": "n2", "stimulus": "ade",
        "coords": "[2.0, 2.0, 2.0]", "area": "a", "valence": 0, "resp": 0.5,
    })
    merged_df = pd.DataFrame(rows)

    pivot_df = fcvn.average_and_pivot_responses(merged_df, stimuli)

    assert len(pivot_df) == 1
    assert pivot_df.iloc[0]["neuron_id"] == "n1"
    assert np.isclose(pivot_df.iloc[0]["ade"], 0.215)
    assert np.isclose(pivot_df.iloc[0]["kw"], 0.815)


def test_average_and_pivot_responses_does_not_oom_on_high_cardinality_coords():
    """Regression test for the observed=False pivot_table memory explosion
    (see module docstring) -- coords must be unique per neuron here, matching
    the real-data shape that triggered it."""
    stimuli = ["ade", "kw", "pro_2.5mm"]
    rows = []
    for i in range(200):
        for stim in stimuli:
            rows.append({
                "fish_id": f"fish_{i % 10}", "neuron_id": f"{i}_fb", "stimulus": stim,
                "coords": f"[{i}.0, {i}.0, {i}.0]", "area": f"area_{i % 5}",
                "valence": i % 2, "resp": float(i),
            })
    merged_df = pd.DataFrame(rows)

    pivot_df = fcvn.average_and_pivot_responses(merged_df, stimuli)

    assert len(pivot_df) == 200


def test_normalize_responses_scales_each_neuron_to_unit_range():
    pivot_df = pd.DataFrame({
        "fish_id": ["f1", "f2"], "neuron_id": ["n1", "n2"],
        "coords": ["[1.0, 1.0, 1.0]", "[2.0, 2.0, 2.0]"], "area": ["a", "b"], "valence": [1, 0],
        "ade": [0.0, 2.0], "kw": [10.0, 4.0],
    })

    scaled_df, scaled_responses = fcvn.normalize_responses(pivot_df, ["ade", "kw"])

    assert scaled_responses.shape == (2, 2)
    assert np.allclose(scaled_responses.min(axis=1), 0.0)
    assert np.allclose(scaled_responses.max(axis=1), 1.0)
    assert list(scaled_df.columns) == ["fish_id", "neuron_id", "coords", "area", "valence", "ade", "kw"]


def test_run_kmeans_clustering_is_deterministic_and_covers_k_range():
    rng = np.random.default_rng(0)
    data = np.vstack([
        rng.normal(loc=[0, 0], scale=0.1, size=(10, 2)),
        rng.normal(loc=[5, 5], scale=0.1, size=(10, 2)),
    ])

    result1 = fcvn.run_kmeans_clustering(data, k_range=[1, 2, 3])
    result2 = fcvn.run_kmeans_clustering(data, k_range=[1, 2, 3])

    assert set(result1.keys()) == {1, 2, 3}
    assert result1[1]["silhouette"] == 0.0
    assert result1[2]["labels"].tolist() == result2[2]["labels"].tolist()


def test_find_optimal_k_picks_highest_silhouette():
    clustering_results = {2: {"silhouette": 0.3}, 3: {"silhouette": 0.7}, 4: {"silhouette": 0.5}}
    assert fcvn.find_optimal_k(clustering_results) == 3


def test_compute_pca_returns_two_components_and_explained_variance():
    rng = np.random.default_rng(1)
    data = rng.normal(size=(20, 6))

    pca_result, explained_var = fcvn.compute_pca(data)

    assert pca_result.shape == (20, 2)
    assert explained_var.shape == (2,)
    assert (explained_var >= 0).all()


def test_add_pca_and_correlation_attaches_pc_and_correlation_by_coords():
    scaled_df = pd.DataFrame({"coords": ["[1.0, 1.0, 1.0]", "[2.0, 2.0, 2.0]"]})
    pca_result = np.array([[0.1, 0.2], [0.3, 0.4]])
    valence = pd.DataFrame({"coords": ["[1.0, 1.0, 1.0]", "[2.0, 2.0, 2.0]"], "correlation": [0.5, -0.5]})

    result = fcvn.add_pca_and_correlation(scaled_df, pca_result, valence)

    assert result["PC1"].tolist() == [0.1, 0.3]
    assert result["PC2"].tolist() == [0.2, 0.4]
    assert result["correlation"].tolist() == [0.5, -0.5]


def test_add_pca_and_correlation_skips_correlation_when_absent():
    scaled_df = pd.DataFrame({"coords": ["[1.0, 1.0, 1.0]"]})
    pca_result = np.array([[0.1, 0.2]])
    valence = pd.DataFrame({"coords": ["[1.0, 1.0, 1.0]"]})  # no 'correlation' column

    result = fcvn.add_pca_and_correlation(scaled_df, pca_result, valence)

    assert "correlation" not in result.columns


def test_add_cluster_labels():
    scaled_df = pd.DataFrame({"neuron_id": ["n1", "n2"]})
    result = fcvn.add_cluster_labels(scaled_df, np.array([0, 1]))
    assert result["cluster"].tolist() == [0, 1]


def test_characterize_clusters_computes_size_valence_and_mean_response():
    df = pd.DataFrame({
        "fish_id": ["f1", "f1", "f2"], "area": ["a", "a", "b"], "valence": [1, 0, 1],
        "ade": [0.2, 0.8, 0.5], "kw": [0.1, 0.9, 0.5],
    })
    labels = np.array([0, 0, 1])

    summary = fcvn.characterize_clusters(df, labels, ["ade", "kw"])

    assert summary["cluster"].tolist() == [0, 1]
    assert summary.loc[summary["cluster"] == 0, "n_neurons"].iloc[0] == 2
    assert summary.loc[summary["cluster"] == 0, "n_positive"].iloc[0] == 1
    assert np.isclose(summary.loc[summary["cluster"] == 0, "mean_ade"].iloc[0], 0.5)


def test_characterize_clusters_computes_sem_of_mean_response():
    df = pd.DataFrame({
        "fish_id": ["f1", "f1", "f2"], "area": ["a", "a", "b"], "valence": [1, 0, 1],
        "ade": [0.2, 0.8, 0.5], "kw": [0.1, 0.9, 0.5],
    })
    labels = np.array([0, 0, 1])

    summary = fcvn.characterize_clusters(df, labels, ["ade", "kw"])

    # cluster 0: ade = [0.2, 0.8] -> sample std (ddof=1) 0.42426, sem = std/sqrt(2)
    assert np.isclose(summary.loc[summary["cluster"] == 0, "sem_ade"].iloc[0], 0.3)
    # cluster 1: single neuron -> sem undefined (ddof=1 with n=1)
    assert np.isnan(summary.loc[summary["cluster"] == 1, "sem_ade"].iloc[0])


def test_assign_cluster_colors_by_response_prefers_attractive_or_aversive():
    cluster_summary = pd.DataFrame({
        "cluster": [0, 1, 2],
        "mean_fex_1": [0.9, 0.1, 0.05], "mean_kw": [0.9, 0.1, 0.05], "mean_pro_2.5mm": [0.9, 0.1, 0.05],
        "mean_ade": [0.1, 0.9, 0.05], "mean_cad_2.5mm": [0.1, 0.9, 0.05], "mean_qui_2.5mm": [0.1, 0.9, 0.05],
    })

    colors = fcvn.assign_cluster_colors_by_response(cluster_summary, fcvn.STIMULI, fcvn.ATTRACTIVE, fcvn.AVERSIVE)

    assert colors[0] in ("limegreen", "forestgreen", "green", "darkgreen", "seagreen")
    assert colors[1] in ("magenta", "deeppink", "hotpink", "orchid", "mediumvioletred")
    assert colors[2] in ("orange", "gold", "darkorange", "coral", "goldenrod")  # low response -> mixed


def test_describe_cluster_colors_returns_report_string(capsys):
    cluster_summary = pd.DataFrame({
        "cluster": [0], "mean_fex_1": [0.9], "mean_kw": [0.5], "mean_pro_2.5mm": [0.5],
        "mean_ade": [0.1], "mean_cad_2.5mm": [0.1], "mean_qui_2.5mm": [0.1],
    })

    report = fcvn.describe_cluster_colors(cluster_summary, {0: "green"}, fcvn.STIMULI, fcvn.ATTRACTIVE, fcvn.AVERSIVE)

    assert "Cluster 0: GREEN" in report
    captured = capsys.readouterr()
    assert "Cluster 0: GREEN" in captured.out


def test_compute_fish_type_splits_forebrain_and_hindbrain():
    scaled_df = pd.DataFrame({"fish_id": ["230713_fb", "230919_hb"]})
    result = fcvn.compute_fish_type(scaled_df)
    assert result["fish_type"].tolist() == ["Forebrain", "Hindbrain"]


def test_compute_cluster_composition_fractions_sum_to_one_per_fish_type():
    scaled_df = pd.DataFrame({
        "fish_type": ["Forebrain", "Forebrain", "Hindbrain"],
        "cluster": [0, 1, 0],
    })

    comp_pivot = fcvn.compute_cluster_composition(scaled_df)

    assert np.allclose(comp_pivot.sum(axis=1).to_numpy(), 1.0)


def test_parse_coordinates_parses_json_strings_to_float_array():
    scaled_df = pd.DataFrame({"coords": ["[1.0, 2.0, 3.0]", "[4.0, 5.0, 6.0]"]})

    coords = fcvn.parse_coordinates(scaled_df)

    assert coords.shape == (2, 3)
    assert np.allclose(coords, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


def test_save_results_writes_named_csvs(tmp_path):
    scaled_df = pd.DataFrame({"neuron_id": ["n1"], "cluster": [0]})
    cluster_summary = pd.DataFrame({"cluster": [0], "n_neurons": [1]})

    output_file, summary_file = fcvn.save_results(scaled_df, cluster_summary, tmp_path, "combined", 2)

    assert output_file.name == "valence_clustering_combined_k2.csv"
    assert summary_file.name == "valence_cluster_summary_combined_k2.csv"
    assert output_file.exists()
    assert summary_file.exists()
    assert pd.read_csv(output_file)["cluster"].tolist() == [0]


def _make_cluster_summary():
    return pd.DataFrame({
        "cluster": [0, 1], "n_neurons": [5, 3],
        "mean_fex_1": [0.8, 0.2], "mean_kw": [0.7, 0.3], "mean_pro_2.5mm": [0.75, 0.25],
        "mean_ade": [0.2, 0.8], "mean_cad_2.5mm": [0.1, 0.9], "mean_qui_2.5mm": [0.15, 0.85],
        "sem_fex_1": [0.05, 0.04], "sem_kw": [0.04, 0.03], "sem_pro_2.5mm": [0.06, 0.05],
        "sem_ade": [0.03, 0.05], "sem_cad_2.5mm": [0.02, 0.06], "sem_qui_2.5mm": [0.03, 0.04],
    })


def test_plot_pca_clusters_renders():
    pca_result = np.random.default_rng(0).normal(size=(10, 2))
    labels = np.array([0, 1] * 5)

    fig = fcvn.plot_pca_clusters(pca_result, labels, 2, 0.5, np.array([50.0, 20.0]))

    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_pca_all_k_renders():
    pca_result = np.random.default_rng(0).normal(size=(10, 2))
    clustering_results = {
        2: {"labels": np.array([0, 1] * 5), "silhouette": 0.5},
        3: {"labels": np.array([0, 1, 2] * 3 + [0]), "silhouette": 0.4},
    }

    fig = fcvn.plot_pca_all_k(clustering_results, pca_result, np.array([50.0, 20.0]), k_range=[2, 3], mode="combined")

    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_pca_by_valence_renders_when_columns_present():
    pca_result = np.random.default_rng(0).normal(size=(6, 2))
    df = pd.DataFrame({"valence": [1, 1, 1, 0, 0, 0], "correlation": [0.5, 0.6, 0.7, -0.5, -0.6, -0.7]})

    fig = fcvn.plot_pca_by_valence(pca_result, df, np.array([50.0, 20.0]))

    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_pca_by_valence_returns_none_when_columns_missing():
    pca_result = np.random.default_rng(0).normal(size=(2, 2))
    df = pd.DataFrame({"foo": [1, 2]})

    assert fcvn.plot_pca_by_valence(pca_result, df, np.array([50.0, 20.0])) is None


def test_plot_cluster_heatmap_renders():
    fig = fcvn.plot_cluster_heatmap(_make_cluster_summary(), fcvn.STIMULI)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_cluster_response_profiles_renders():
    cluster_summary = _make_cluster_summary()
    colors = fcvn.assign_cluster_colors_by_response(cluster_summary, fcvn.STIMULI, fcvn.ATTRACTIVE, fcvn.AVERSIVE)

    fig = fcvn.plot_cluster_response_profiles(cluster_summary, colors, fcvn.STIMULI, fcvn.ATTRACTIVE)

    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_cluster_response_profiles_draws_sem_error_bars():
    cluster_summary = _make_cluster_summary()
    colors = fcvn.assign_cluster_colors_by_response(cluster_summary, fcvn.STIMULI, fcvn.ATTRACTIVE, fcvn.AVERSIVE)

    fig = fcvn.plot_cluster_response_profiles(cluster_summary, colors, fcvn.STIMULI, fcvn.ATTRACTIVE)
    ax = fig.axes[0]

    assert len(ax.containers) == 2  # one errorbar container per cluster
    for container, (_, row) in zip(ax.containers, cluster_summary.iterrows()):
        expected_sem = [row[f"sem_{s}"] for s in fcvn.STIMULI]
        # ErrorbarContainer.lines = (data_line, caplines, barlinecols)
        drawn_yerr = container.lines[2][0].get_segments()
        drawn_half_heights = [(seg[1][1] - seg[0][1]) / 2 for seg in drawn_yerr]
        assert np.allclose(drawn_half_heights, expected_sem)

    plt.close(fig)


def test_plot_cluster_composition_renders():
    comp_pivot = pd.DataFrame({0: [0.6, 0.4], 1: [0.4, 0.6]}, index=["Forebrain", "Hindbrain"])
    fig = fcvn.plot_cluster_composition(comp_pivot, {0: "green", 1: "magenta"}, mode="combined")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_brain_spatial_renders():
    coords = np.random.default_rng(0).normal(loc=100, scale=20, size=(20, 3))
    labels = np.array([0, 1] * 10)
    ref_brain = np.random.default_rng(0).normal(size=(10, 50, 50))

    fig = fcvn.plot_brain_spatial(ref_brain, coords, labels, {0: "magenta", 1: "green"}, title="test")

    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_brain_spatial_sagittal_scatter_uses_y_then_z_axis_order():
    """Regression test for the sagittal-panel axis swap (see module docstring).

    proj_sagittal = mean(ref_brain, axis=2) has shape (z, y) in (row, col)
    order, so the sagittal scatter's (x, y) must be (coords[:, 1], coords[:, 0])
    -- i.e. (y, z) -- to land on the brain image at all, matching
    jenkins_et_al.plotting.plot_neuron_scatter_on_brain's convention.
    """
    coords = np.array([[10.0, 20.0, 30.0], [11.0, 21.0, 31.0]])  # (z, y, x)
    labels = np.array([0, 0])
    ref_brain = np.zeros((40, 40, 40))

    fig = fcvn.plot_brain_spatial(ref_brain, coords, labels, {0: "magenta"})
    ax_sagittal = fig.axes[1]

    offsets = ax_sagittal.collections[0].get_offsets()
    assert np.allclose(offsets, coords[:, [1, 0]])  # (y, z)
    plt.close(fig)
