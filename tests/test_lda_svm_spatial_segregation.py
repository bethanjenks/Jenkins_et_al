"""Unit tests for jenkins_et_al.neural.lda_svm_spatial_segregation
against small synthetic inputs (no dependency on the real data drive).

The full pipeline against real data is golden-verified separately in
tests/golden/test_lda_svm_spatial_segregation.py.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from jenkins_et_al.neural import lda_svm_spatial_segregation as lss


def _make_separated_neurons(n_per_class: int = 30, area: str = "area_a", seed: int = 0) -> pd.DataFrame:
    """Positive/negative neurons well-separated along x, all significant."""
    rng = np.random.default_rng(seed)
    pos_coords = rng.normal(loc=[0, 0, 20], scale=1.0, size=(n_per_class, 3))
    neg_coords = rng.normal(loc=[0, 0, -20], scale=1.0, size=(n_per_class, 3))

    rows = []
    for z, y, x in pos_coords:
        rows.append({"coords": [float(z), float(y), float(x)], "area": area, "p_fdr": 0.001, "valence": 1})
    for z, y, x in neg_coords:
        rows.append({"coords": [float(z), float(y), float(x)], "area": area, "p_fdr": 0.001, "valence": 0})

    df = pd.DataFrame(rows)
    df["coords_parsed"] = df["coords"].apply(lss.parse_coords)
    df[["z", "y", "x"]] = pd.DataFrame(df["coords_parsed"].tolist(), index=df.index)
    return df


def test_parse_coords_handles_string_and_native_forms():
    assert lss.parse_coords("[1.0, 2.0, 3.0]") == (1.0, 2.0, 3.0)
    assert lss.parse_coords((1.0, 2.0, 3.0)) == (1.0, 2.0, 3.0)
    assert lss.parse_coords(np.array([1.0, 2.0, 3.0])) == (1.0, 2.0, 3.0)

    with pytest.raises(ValueError):
        lss.parse_coords("[1.0, 2.0]")
    with pytest.raises(ValueError):
        lss.parse_coords(123)


def test_load_valence_table_labels_and_unpacks_coords(tmp_path):
    pos_path = tmp_path / "pos.csv"
    neg_path = tmp_path / "neg.csv"
    pd.DataFrame({"coords": ["[1.0, 2.0, 3.0]"], "area": ["a"], "p_fdr": [0.01]}).to_csv(pos_path, index=False)
    pd.DataFrame({"coords": ["[4.0, 5.0, 6.0]"], "area": ["a"], "p_fdr": [0.5]}).to_csv(neg_path, index=False)

    df = lss.load_valence_table(pos_path, neg_path)

    assert list(df["valence"]) == [1, 0]
    # Unpacked as [z, y, x], matching the original notebook's convention.
    assert df.loc[0, ["z", "y", "x"]].tolist() == [1.0, 2.0, 3.0]
    assert df.loc[1, ["z", "y", "x"]].tolist() == [4.0, 5.0, 6.0]


def test_filter_area_splits_on_area_and_p_fdr_threshold():
    df = pd.DataFrame({
        "area": ["a", "a", "b"],
        "p_fdr": [0.01, 0.5, 0.01],
        "valence": [1, 0, 1],
    })
    valence_neurons, nonvalence_neurons = lss.filter_area(df, "a", p_fdr_thresh=0.05)

    assert len(valence_neurons) == 1
    assert len(nonvalence_neurons) == 1
    assert "b" not in valence_neurons["area"].values


@pytest.mark.parametrize("areas,expected", [
    ("area_a", ["area_a"]),
    (["area_a", "area_b"], ["area_a", "area_b"]),
])
def test_select_areas_to_run_passes_through_explicit_areas(areas, expected):
    df = pd.DataFrame({"area": ["area_a"], "p_fdr": [0.01], "valence": [1]})
    assert lss.select_areas_to_run(df, areas) == expected


def test_select_areas_to_run_auto_filters_on_min_per_class():
    rows = []
    for area, n_pos, n_neg in [("rich", 25, 25), ("sparse", 5, 5)]:
        rows += [{"area": area, "p_fdr": 0.01, "valence": 1}] * n_pos
        rows += [{"area": area, "p_fdr": 0.01, "valence": 0}] * n_neg
    df = pd.DataFrame(rows)

    areas_to_run = lss.select_areas_to_run(df, "auto", p_fdr_thresh=0.05, min_per_class=20)

    assert areas_to_run == ["rich"]


def test_balance_classes_downsample_and_upsample_equalize_counts():
    df = pd.DataFrame({"valence": [1] * 10 + [0] * 3})

    down = lss.balance_classes(df, strategy="downsample", random_state=0)
    up = lss.balance_classes(df, strategy="upsample", random_state=0)

    assert down["valence"].value_counts().to_dict() == {1: 3, 0: 3}
    assert up["valence"].value_counts().to_dict() == {1: 10, 0: 10}


def test_balance_classes_none_strategy_returns_unchanged_copy():
    df = pd.DataFrame({"valence": [1, 1, 0]})
    out = lss.balance_classes(df, strategy=None)
    assert out.equals(df)
    assert out is not df


def test_balance_classes_rejects_unknown_strategy():
    df = pd.DataFrame({"valence": [1, 0]})
    with pytest.raises(ValueError):
        lss.balance_classes(df, strategy="bogus")


def test_evaluate_lda_separation_detects_well_separated_classes():
    df = _make_separated_neurons()
    results = lss.evaluate_lda_separation(df, cv_folds=5, random_state=0)

    assert results["auc_mean"] > 0.9
    assert results["fisher_ratio"] > 1.0
    # LDA axis should point predominantly along x, the separating coordinate.
    assert abs(results["lda_vector_x"]) > abs(results["lda_vector_y"])
    assert abs(results["lda_vector_x"]) > abs(results["lda_vector_z"])
    assert results["proj"].shape == (len(df),)
    assert results["y"].shape == (len(df),)


def test_evaluate_svm_separation_balances_and_scores():
    df = _make_separated_neurons(n_per_class=30)
    df = pd.concat([df, df[df["valence"] == 1].iloc[:5]], ignore_index=True)  # imbalance

    none_res = lss.evaluate_svm_separation(df, balance=None, random_state=0)
    down_res = lss.evaluate_svm_separation(df, balance="downsample", random_state=0)

    assert none_res["n_pos"] == 35 and none_res["n_neg"] == 30
    assert down_res["n_pos"] == down_res["n_neg"] == 30
    assert down_res["auc_mean"] > 0.9


def test_evaluate_svm_separation_rejects_unknown_balance_strategy():
    df = _make_separated_neurons(n_per_class=10)
    with pytest.raises(ValueError):
        lss.evaluate_svm_separation(df, balance="bogus")


def test_permutation_test_auc_gives_low_p_for_real_separation():
    df = _make_separated_neurons(n_per_class=20)
    result = lss.permutation_test_auc(df, method="lda", n_perm=50, cv_folds=5, random_state=0)

    assert result["observed_auc"] > 0.9
    assert result["p_value"] < 0.1
    assert result["perm_aucs"].shape == (50,)


def test_permutation_test_auc_p_value_uses_plus_one_correction():
    df = _make_separated_neurons(n_per_class=15)
    result = lss.permutation_test_auc(df, method="svm", balance="downsample", n_perm=20, cv_folds=5, random_state=0)
    # The "+1" correction means p can never be exactly 0.
    assert result["p_value"] >= 1 / 21


def test_permutation_test_auc_rejects_unknown_method():
    df = _make_separated_neurons(n_per_class=10)
    with pytest.raises(ValueError):
        lss.permutation_test_auc(df, method="bogus")


def test_angles_to_axes_unit_vectors():
    angles = lss.angles_to_axes([1, 0, 0])
    # arccos near 0 is numerically ill-conditioned; a small absolute tolerance covers it.
    assert np.isclose(angles["angle_to_x"], 0.0, atol=1e-3)
    assert np.isclose(angles["angle_to_y"], 90.0)
    assert np.isclose(angles["angle_to_z"], 90.0)


def test_run_area_analysis_skips_areas_with_too_few_neurons():
    df = pd.DataFrame({
        "area": ["tiny"] * 3,
        "p_fdr": [0.01, 0.01, 0.5],
        "valence": [1, 0, 0],
        "x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0], "z": [0.0, 1.0, 2.0],
    })
    results_df, perm_df = lss.run_area_analysis(df, ["tiny"], n_permutations=0)
    assert results_df.empty
    assert perm_df.empty


def test_run_area_analysis_produces_one_lda_and_three_svm_rows_per_area():
    df = _make_separated_neurons(n_per_class=25, area="area_a")
    results_df, perm_df = lss.run_area_analysis(
        df, ["area_a"], cv_folds=5, random_state=0, n_permutations=5,
    )

    assert sorted(results_df["method"]) == sorted(["lda", "svm_none", "svm_downsample", "svm_upsample"])
    assert sorted(perm_df["method"]) == sorted(["lda", "svm_none", "svm_downsample", "svm_upsample"])

    lda_row = results_df[results_df["method"] == "lda"].iloc[0]
    svm_rows = results_df[results_df["method"] != "lda"]
    assert not lda_row[["lda_vector_x", "lda_vector_y", "lda_vector_z"]].isna().any()
    assert svm_rows["lda_vector_x"].isna().all()


def test_plotting_functions_render_figures():
    df = _make_separated_neurons(n_per_class=20)
    results_df, _ = lss.run_area_analysis(df, ["area_a"], cv_folds=5, random_state=0, n_permutations=0)
    val_neurons, nonval_neurons = lss.filter_area(df, "area_a")
    lda_row = results_df[results_df["method"] == "lda"].iloc[0]
    lda_vec = [lda_row["lda_vector_x"], lda_row["lda_vector_y"], lda_row["lda_vector_z"]]

    lda_eval = lss.evaluate_lda_separation(val_neurons, cv_folds=5, random_state=0)
    figs = [
        lss.plot_lda_kde_projection(lda_eval["proj"], lda_eval["y"], lda_eval, title="area_a"),
        lss.plot_lda_projection_histogram(val_neurons, title="area_a"),
        lss.plot_method_comparison(results_df, "area_a"),
        lss.plot_basic_3d_scatter(val_neurons, nonval_neurons),
        lss.plot_lda_axis_plane_3d(val_neurons, lda_vec),
    ]

    slice_data = lss.compute_svm_decision_boundary_slice(val_neurons, balance="downsample", random_state=0)
    figs.append(lss.plot_svm_decision_boundary_slice(slice_data))

    for fig in figs:
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
