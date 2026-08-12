"""Regression test for jenkins_et_al.neural.functional_clustering_valence_neurons
against a golden capture of notebooks/neural/functional_clustering_valence_neurons.ipynb.

golden/functional_clustering_valence_neurons/ was produced by an independent
from-scratch reimplementation of the notebook's own cell source
(capture_golden.py, run once during this port), for `MODE = "combined"` --
no prior golden capture existed, since the notebook itself has never been run
to completion (every cell past the data-loading fix has `execution_count:
null`). See module docstring for the two bugs this uncovered:

1. `average_and_pivot_responses`'s `pivot_table` call OOMs under the
   notebook's own `observed=False` default -- fixed with `observed=True`,
   proven equivalent post-`dropna`. Reproduced independently on synthetic
   data of the same shape (a structural pandas/Categorical issue, not
   data-dependent).
2. The "Extract coordinates" step never parses `coords` out of its
   JSON-string form -- `np.vstack` over strings silently produces an
   `(N, 1)` string array instead of `(N, 3)` floats. Fixed with `json.loads`
   (`parse_coordinates`), the same fix already used for the identical field
   in `lifetime_sparseness.ipynb`.

Golden capture applied both fixes before comparing, per the golden-refactor
skill's rule that bug-for-bug preservation isn't possible when there's no
completed run to preserve.

Confirms every data-level output feeding the notebook's figures: pivot_df,
normalized responses, PCA, k-means labels/silhouette per k, optimal k,
cluster summary, cluster colors, parsed brain coordinates, and
forebrain/hindbrain cluster composition. Rendered PNGs were also visually
signed off by the user (see PORTING_PLAN.md) but aren't re-asserted here as
pixel-exact -- rendering varies with backend/DPI even when the underlying
data is identical.

Skipped entirely if the source data drive isn't mounted, since the
underlying response-vector HDF5 and correlation CSVs are not part of this
repository.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from jenkins_et_al.neural import functional_clustering_valence_neurons as fcvn

POS_VALENCE_PATH = Path("/Volumes/LaCie/larval_HuC/imaging/7dpf_positive_correlations_25s_no_HCl_1000_corrected.csv")
NEG_VALENCE_PATH = Path("/Volumes/LaCie/larval_HuC/imaging/7dpf_negative_correlations_25s_no_HCl_1000_corrected.csv")
NEURAL_DATA_PATH = Path("/Volumes/LaCie/larval_HuC/imaging/7dpf_wb_population_general_response_vector.h5")
REF_BRAIN_PATH = Path("/Volumes/LaCie/area_masks/T_AVG_HuC.tif")
GOLDEN_DIR = Path(__file__).resolve().parent.parent.parent / "golden" / "functional_clustering_valence_neurons"

pytestmark = pytest.mark.skipif(
    not POS_VALENCE_PATH.exists() or not GOLDEN_DIR.exists(),
    reason="source data drive or golden capture not available on this machine",
)


@pytest.fixture(scope="module")
def pipeline_result():
    pos_regress, neg_regress, data_resp, ref_brain = fcvn.load_data(
        POS_VALENCE_PATH, NEG_VALENCE_PATH, NEURAL_DATA_PATH, REF_BRAIN_PATH,
    )
    valence, _ = fcvn.select_neurons_by_mode(pos_regress, neg_regress, "combined")
    merged_df = fcvn.merge_valence_with_responses(valence, data_resp)
    pivot_df = fcvn.average_and_pivot_responses(merged_df, fcvn.STIMULI)
    scaled_df, scaled_responses = fcvn.normalize_responses(pivot_df, fcvn.STIMULI)

    clustering_results = fcvn.run_kmeans_clustering(scaled_responses, fcvn.K_RANGE)
    optimal_k = fcvn.find_optimal_k(clustering_results)
    pca_result, explained_var = fcvn.compute_pca(scaled_responses)
    scaled_df = fcvn.add_pca_and_correlation(scaled_df, pca_result, valence)

    optimal_labels = clustering_results[optimal_k]["labels"]
    scaled_df = fcvn.add_cluster_labels(scaled_df, optimal_labels)
    cluster_summary = fcvn.characterize_clusters(scaled_df, optimal_labels, fcvn.STIMULI)
    cluster_colors = fcvn.assign_cluster_colors_by_response(cluster_summary, fcvn.STIMULI, fcvn.ATTRACTIVE, fcvn.AVERSIVE)

    coords = fcvn.parse_coordinates(scaled_df)
    scaled_df = fcvn.compute_fish_type(scaled_df)
    comp_pivot = fcvn.compute_cluster_composition(scaled_df)

    return {
        "pivot_df": pivot_df,
        "scaled_df": scaled_df,
        "scaled_responses": scaled_responses,
        "clustering_results": clustering_results,
        "optimal_k": optimal_k,
        "pca_result": pca_result,
        "explained_var": explained_var,
        "optimal_labels": optimal_labels,
        "cluster_summary": cluster_summary,
        "cluster_colors": cluster_colors,
        "coords": coords,
        "comp_pivot": comp_pivot,
        "ref_brain": ref_brain,
    }


def test_pivot_df_matches_golden(pipeline_result):
    port = pipeline_result["pivot_df"].sort_values(["fish_id", "neuron_id"]).reset_index(drop=True)
    golden = pd.read_csv(GOLDEN_DIR / "pivot_df.csv").sort_values(["fish_id", "neuron_id"]).reset_index(drop=True)

    assert list(port.columns) == list(golden.columns)
    assert len(port) == len(golden)
    assert (port["fish_id"].to_numpy() == golden["fish_id"].to_numpy()).all()
    assert (port["neuron_id"].to_numpy() == golden["neuron_id"].to_numpy()).all()
    assert (port["area"].to_numpy() == golden["area"].to_numpy()).all()
    assert (port["valence"].to_numpy() == golden["valence"].to_numpy()).all()
    assert (port["coords"].astype(str).to_numpy() == golden["coords"].astype(str).to_numpy()).all()
    for stim in fcvn.STIMULI:
        assert np.allclose(port[stim].to_numpy(), golden[stim].to_numpy())


def test_scaled_responses_match_golden(pipeline_result):
    golden = np.load(GOLDEN_DIR / "scaled_responses.npy")
    assert pipeline_result["scaled_responses"].shape == golden.shape
    assert np.allclose(pipeline_result["scaled_responses"], golden)


def test_pca_matches_golden(pipeline_result):
    golden_pca = np.load(GOLDEN_DIR / "pca_result.npy")
    golden_var = np.load(GOLDEN_DIR / "explained_var.npy")
    assert np.allclose(pipeline_result["pca_result"], golden_pca)
    assert np.allclose(pipeline_result["explained_var"], golden_var)


def test_silhouette_and_optimal_k_match_golden(pipeline_result):
    with open(GOLDEN_DIR / "silhouette_scores.json") as f:
        golden_silhouette = json.load(f)
    with open(GOLDEN_DIR / "optimal_k.json") as f:
        golden_optimal_k = json.load(f)["optimal_k"]

    for k, result in pipeline_result["clustering_results"].items():
        assert np.isclose(result["silhouette"], golden_silhouette[str(k)])
    assert pipeline_result["optimal_k"] == golden_optimal_k


def test_kmeans_labels_match_golden_for_every_k(pipeline_result):
    for k in fcvn.K_RANGE:
        golden_labels = np.load(GOLDEN_DIR / f"labels_k{k}.npy")
        assert np.array_equal(pipeline_result["clustering_results"][k]["labels"], golden_labels)


def test_cluster_colors_match_golden(pipeline_result):
    with open(GOLDEN_DIR / "cluster_colors.json") as f:
        golden_colors = {int(k): v for k, v in json.load(f).items()}
    assert pipeline_result["cluster_colors"] == golden_colors


def test_cluster_summary_matches_golden(pipeline_result):
    port = pipeline_result["cluster_summary"]
    golden = pd.read_csv(GOLDEN_DIR / "cluster_summary.csv")

    assert len(port) == len(golden)
    for col in ("n_neurons", "n_positive", "n_negative"):
        assert (port[col].to_numpy() == golden[col].to_numpy()).all()
    for stim in fcvn.STIMULI:
        assert np.allclose(port[f"mean_{stim}"].to_numpy(), golden[f"mean_{stim}"].to_numpy())


def test_cluster_sizes_match_golden(pipeline_result):
    # Source: independent golden reimplementation's own printed cluster sizes.
    assert dict(pipeline_result["cluster_summary"].set_index("cluster")["n_neurons"]) == {0: 1242, 1: 984}


def test_brain_coords_match_golden(pipeline_result):
    golden = np.load(GOLDEN_DIR / "coords_and_labels.npz")
    assert np.allclose(pipeline_result["coords"], golden["coords"])
    assert np.array_equal(pipeline_result["optimal_labels"], golden["labels"])


def test_ref_brain_shape_matches_golden(pipeline_result):
    golden_shape = tuple(np.load(GOLDEN_DIR / "ref_brain_shape.npy"))
    assert tuple(pipeline_result["ref_brain"].shape) == golden_shape


def test_cluster_composition_matches_golden(pipeline_result):
    golden = pd.read_csv(GOLDEN_DIR / "comp_pivot.csv", index_col=0)
    golden.columns = [int(c) for c in golden.columns]

    port = pipeline_result["comp_pivot"].sort_index(axis=1)
    golden = golden.sort_index(axis=1)
    assert np.allclose(port.to_numpy(), golden.to_numpy())


def test_brain_spatial_plot_renders(pipeline_result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = fcvn.plot_brain_spatial(
        pipeline_result["ref_brain"], pipeline_result["coords"], pipeline_result["optimal_labels"],
        pipeline_result["cluster_colors"],
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
