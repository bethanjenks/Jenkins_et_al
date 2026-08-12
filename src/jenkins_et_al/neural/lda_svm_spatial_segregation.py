"""Spatial segregation of valence-correlated neurons within brain areas, tested
via LDA (linear, interpretable axis) and SVM (RBF, with none/downsample/upsample
class balancing), plus permutation tests on cross-validated AUC.

Ported from notebooks/neural/lda_svm_spatial_segregation.ipynb; logic unchanged,
split into loading/processing/plotting functions per the /tdd REFACTOR checklist.
The notebook itself ran cleanly top-to-bottom as saved (sequential execution
counts, no gaps) -- re-verified via a fresh clean-kernel run of its own cell
source for golden capture, per the golden-refactor skill's rule not to trust
saved outputs on appearance alone. No bugs found in this notebook; unlike most
prior ports it was already well-parametrized (paths, CV folds, thresholds) with
no hardcoded absolute paths inside its functions.

**Owns the "+1"-corrected permutation-test formula**: `permutation_test_auc`'s
p-value is `(sum(perm >= obs) + 1) / (n_perm + 1)` (Davison & Hinkley's
standard correction, avoiding a p-value of exactly 0). This differs from the
plain-ratio formula in `lda_svm_classification.ipynb` (#14, not yet ported)
and `template_matching_classification.py`'s (#9) own permutation test --
confirmed different by design, not unified.

**Refactor-shape change, consistent with every other ported plotting function
in this project**: none of the plotting functions here call `plt.show()`
before returning; each returns its `Figure` for the caller to display/save.

**Refactor-shape change, for testability**: `evaluate_lda_separation` and
`compute_svm_decision_boundary_slice` were split so the plotting functions
that reuse their output (`plot_lda_kde_projection`, `plot_svm_decision_boundary_slice`)
take precomputed data rather than recomputing it -- the notebook's own
`evaluate_lda_separation(..., plot=True)` and `plot_svm_decision_boundary_slice_paper`
computed and plotted in one step; here the same numbers feed both the
regression test and the plot.

**Refactor-shape change, DRY**: the notebook's own `permutation_test_auc`
called `balance_classes` twice for `method="svm"` (once directly, once again
inside its own call to `evaluate_svm_separation`) -- harmless since both
calls are deterministic given the same `random_state`, but redundant.
`evaluate_svm_separation` now also returns the balanced `X`/`y` it fit on
(`"X_bal"`/`"y_bal"`, alongside `evaluate_lda_separation`'s existing
`"proj"`/`"y"`), so `permutation_test_auc` reuses them instead of
re-balancing. Re-verified output-identical against the golden capture after
this change.

Golden capture (fresh top-to-bottom run of the notebook's own cell source,
default `AREAS = 'olfactory_bulb'`, `N_PERMUTATIONS = 1000`): 13606 neurons in
olfactory_bulb, 92 positive / 58 negative surviving `p_fdr < 0.05`; LDA CV AUC
0.700 (p=0.001), SVM AUC 0.683/0.649/0.804 for none/downsample/upsample
balancing (p=0.003/0.016/0.001) -- all matched exactly by this port.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import resample

#: Source: cell 2. Axis labels used by the two 3D/2D "paper style" plots.
AXIS_TEXT = {
    "x": "X μm (L–M)",
    "y": "Y μm (A–P)",
    "z": "Z μm (D–V)",
}

#: Source: cell 3 (`_axis_index_map`). Maps a data-axis label to its column index.
_AXIS_TO_COL = {"x": 0, "y": 1, "z": 2}


# ---------------------------------------------------------------------------
# Loading (source: cell 3, cell 4)
# ---------------------------------------------------------------------------

def parse_coords(x) -> tuple[float, float, float]:
    """Parse a coordinate field (list/tuple/ndarray or its string repr) to a 3-tuple."""
    if isinstance(x, (list, tuple, np.ndarray)):
        coords = list(x)
    elif isinstance(x, str):
        coords = list(ast.literal_eval(x))
    else:
        raise ValueError(f"Unsupported coords type: {type(x)}")

    if len(coords) != 3:
        raise ValueError(f"Expected 3D coords, got length {len(coords)}: {coords}")

    return tuple(coords)


def load_valence_table(pos_csv: str | Path, neg_csv: str | Path) -> pd.DataFrame:
    """Load positive/negative valence-correlation CSVs into one labeled, coordinate-parsed table.

    Adds a `valence` column (1 = positive, 0 = negative) and `x`/`y`/`z` columns
    unpacked from `coords` (parsed as `[z, y, x]`, matching the original notebook's
    coordinate convention -- see its own module-level note on this).
    """
    pos = pd.read_csv(pos_csv).copy()
    neg = pd.read_csv(neg_csv).copy()
    pos["valence"] = 1
    neg["valence"] = 0
    df = pd.concat([pos, neg], ignore_index=True)

    if "coords" not in df.columns:
        raise KeyError("Expected a 'coords' column in the input CSVs.")

    df["coords_parsed"] = df["coords"].apply(parse_coords)
    df[["z", "y", "x"]] = pd.DataFrame(df["coords_parsed"].tolist(), index=df.index)
    return df


# ---------------------------------------------------------------------------
# Area/class selection (source: cell 3, cell 5)
# ---------------------------------------------------------------------------

def filter_area(df: pd.DataFrame, area: str, p_fdr_thresh: float = 0.05) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (valence_neurons, nonvalence_neurons) within a given area."""
    area_df = df[df["area"] == area].copy()
    valence_neurons = area_df[area_df["p_fdr"] < p_fdr_thresh].copy()
    nonvalence_neurons = area_df[area_df["p_fdr"] >= p_fdr_thresh].copy()
    return valence_neurons, nonvalence_neurons


def select_areas_to_run(
    df: pd.DataFrame,
    areas: str | Sequence[str],
    p_fdr_thresh: float = 0.05,
    min_per_class: int = 20,
) -> list[str]:
    """Resolve the `AREAS` config option (`'auto'`, a single area, or a list) to a list of areas.

    `'auto'` selects every area with at least `min_per_class` significant
    neurons in both valence directions.
    """
    if isinstance(areas, str) and areas.lower() == "auto":
        areas_to_run = []
        for area in sorted(df["area"].dropna().unique()):
            valence_neurons, _ = filter_area(df, area, p_fdr_thresh=p_fdr_thresh)
            n_pos = (valence_neurons["valence"] == 1).sum()
            n_neg = (valence_neurons["valence"] == 0).sum()
            if n_pos >= min_per_class and n_neg >= min_per_class:
                areas_to_run.append(area)
        return areas_to_run
    elif isinstance(areas, str):
        return [areas]
    return list(areas)


def balance_classes(
    df: pd.DataFrame,
    label_col: str = "valence",
    strategy: str | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Balance classes by downsampling the majority or upsampling the minority class."""
    if strategy is None:
        return df.copy()

    pos = df[df[label_col] == 1]
    neg = df[df[label_col] == 0]

    if len(pos) == 0 or len(neg) == 0:
        return df.copy()

    if strategy == "downsample":
        n = min(len(pos), len(neg))
        pos_bal = resample(pos, replace=False, n_samples=n, random_state=random_state)
        neg_bal = resample(neg, replace=False, n_samples=n, random_state=random_state)
    elif strategy == "upsample":
        n = max(len(pos), len(neg))
        pos_bal = resample(pos, replace=True, n_samples=n, random_state=random_state)
        neg_bal = resample(neg, replace=True, n_samples=n, random_state=random_state)
    else:
        raise ValueError("strategy must be one of {None, 'downsample', 'upsample'}")

    out = pd.concat([pos_bal, neg_bal], ignore_index=True)
    return out.sample(frac=1, random_state=random_state).reset_index(drop=True)


# ---------------------------------------------------------------------------
# LDA / SVM evaluation (source: cell 3)
# ---------------------------------------------------------------------------

def _lda_svm_cv(random_state: int) -> StratifiedKFold:
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)


def evaluate_lda_separation(
    df: pd.DataFrame,
    coord_cols: tuple[str, str, str] = ("x", "y", "z"),
    label_col: str = "valence",
    cv_folds: int = 5,
    random_state: int = 42,
) -> dict:
    """LDA separation: CV accuracy/AUC + Fisher ratio + LDA direction + full-data projection.

    `"proj"`/`"y"` (the full-data LDA projection and its labels, for
    `plot_lda_kde_projection`) are included in the returned dict but are not
    scalar -- drop them before building a results table row.
    """
    X = df[list(coord_cols)].values
    y = df[label_col].values

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    pipe = make_pipeline(
        LinearDiscriminantAnalysis(n_components=1),
        LogisticRegression(max_iter=1000),
    )

    acc_scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    auc_scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")

    lda = LinearDiscriminantAnalysis(n_components=1)
    proj = lda.fit_transform(X, y).ravel()

    lda_vec = lda.coef_[0]
    lda_vec = lda_vec / (np.linalg.norm(lda_vec) + 1e-12)

    mean_pos, mean_neg = proj[y == 1].mean(), proj[y == 0].mean()
    var_pos, var_neg = proj[y == 1].var(), proj[y == 0].var()
    fisher_ratio = (mean_pos - mean_neg) ** 2 / (var_pos + var_neg + 1e-12)

    return {
        "accuracy_mean": float(acc_scores.mean()),
        "accuracy_std": float(acc_scores.std()),
        "auc_mean": float(auc_scores.mean()),
        "auc_std": float(auc_scores.std()),
        "fisher_ratio": float(fisher_ratio),
        "lda_vector_x": float(lda_vec[0]),
        "lda_vector_y": float(lda_vec[1]),
        "lda_vector_z": float(lda_vec[2]),
        "proj": proj,
        "y": y,
    }


def evaluate_svm_separation(
    df: pd.DataFrame,
    balance: str | None = None,
    coord_cols: tuple[str, str, str] = ("x", "y", "z"),
    label_col: str = "valence",
    cv_folds: int = 5,
    random_state: int = 42,
) -> dict:
    """SVM (RBF) separation with optional class balancing.

    `"X_bal"`/`"y_bal"` (the balanced feature/label arrays actually fit on,
    for `permutation_test_auc` to reuse without re-balancing) are included in
    the returned dict but are not scalar -- drop them before building a
    results table row.
    """
    df_bal = balance_classes(df, label_col=label_col, strategy=balance, random_state=random_state)

    X = df_bal[list(coord_cols)].values
    y = df_bal[label_col].values

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=random_state))

    acc_scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    auc_scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")

    return {
        "balance": balance if balance is not None else "none",
        "n_total": int(len(df_bal)),
        "n_pos": int((df_bal[label_col] == 1).sum()),
        "n_neg": int((df_bal[label_col] == 0).sum()),
        "accuracy_mean": float(acc_scores.mean()),
        "accuracy_std": float(acc_scores.std()),
        "auc_mean": float(auc_scores.mean()),
        "auc_std": float(auc_scores.std()),
        "X_bal": X,
        "y_bal": y,
    }


def permutation_test_auc(
    df: pd.DataFrame,
    method: str,
    balance: str | None = None,
    coord_cols: tuple[str, str, str] = ("x", "y", "z"),
    label_col: str = "valence",
    cv_folds: int = 5,
    n_perm: int = 500,
    random_state: int = 42,
) -> dict:
    """Permutation test on cross-validated AUC (one-sided: perm >= observed).

    p-value uses the "+1" correction: `(sum(perm >= obs) + 1) / (n_perm + 1)`.
    """
    rng = np.random.default_rng(random_state)

    if method == "lda":
        obs = evaluate_lda_separation(df, coord_cols, label_col, cv_folds, random_state)["auc_mean"]
        X = df[list(coord_cols)].values
        y = df[label_col].values
        pipe = make_pipeline(LinearDiscriminantAnalysis(n_components=1), LogisticRegression(max_iter=1000))
    elif method == "svm":
        svm_res = evaluate_svm_separation(df, balance, coord_cols, label_col, cv_folds, random_state)
        obs = svm_res["auc_mean"]
        X, y = svm_res["X_bal"], svm_res["y_bal"]
        pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=random_state))
    else:
        raise ValueError("method must be 'lda' or 'svm'")

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    perm_aucs = []
    for _ in range(int(n_perm)):
        y_perm = rng.permutation(y)
        auc_scores = cross_val_score(pipe, X, y_perm, cv=cv, scoring="roc_auc")
        perm_aucs.append(auc_scores.mean())

    perm_aucs = np.array(perm_aucs)
    p = (np.sum(perm_aucs >= obs) + 1) / (len(perm_aucs) + 1)

    return {"observed_auc": float(obs), "p_value": float(p), "perm_aucs": perm_aucs}


def angles_to_axes(vec_xyz: Sequence[float]) -> dict[str, float]:
    """Angles (degrees) between vec_xyz and the x/y/z axes."""
    v = np.array(vec_xyz, dtype=float)
    v = v / (np.linalg.norm(v) + 1e-12)
    axes = {"x": np.array([1, 0, 0]), "y": np.array([0, 1, 0]), "z": np.array([0, 0, 1])}
    return {
        f"angle_to_{k}": float(np.degrees(np.arccos(np.clip(np.dot(v, a), -1, 1))))
        for k, a in axes.items()
    }


# ---------------------------------------------------------------------------
# Per-area pipeline (source: cell 6)
# ---------------------------------------------------------------------------

def run_area_analysis(
    valence_df: pd.DataFrame,
    areas_to_run: Sequence[str],
    p_fdr_thresh: float = 0.05,
    min_per_class: int = 20,
    cv_folds: int = 5,
    random_state: int = 42,
    n_permutations: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run LDA + SVM (none/downsample/upsample) + permutation tests for each area.

    Areas with fewer than 2 neurons in either class are skipped; areas below
    `min_per_class` in either class are still run but flagged noisy (matching
    the notebook's own `[SKIP]`/`[WARN]` console messages -- not reproduced
    here as prints, callers can derive the same condition from `n_pos`/`n_neg`).

    Returns `(results_df, perm_df)`, matching the notebook's own
    `spatial_method_results.csv`/`spatial_permutation_pvalues.csv`.
    """
    all_results = []
    all_perm = []

    for area in areas_to_run:
        val_neurons, _ = filter_area(valence_df, area, p_fdr_thresh=p_fdr_thresh)
        n_pos = int((val_neurons["valence"] == 1).sum())
        n_neg = int((val_neurons["valence"] == 0).sum())

        if n_pos < 2 or n_neg < 2:
            continue

        lda_res = evaluate_lda_separation(val_neurons, cv_folds=cv_folds, random_state=random_state)
        lda_vec = np.array([lda_res["lda_vector_x"], lda_res["lda_vector_y"], lda_res["lda_vector_z"]])
        lda_angles = angles_to_axes(lda_vec)
        lda_row = {k: v for k, v in lda_res.items() if k not in ("proj", "y")}

        all_results.append({
            "area": area,
            "method": "lda",
            "n_pos": n_pos,
            "n_neg": n_neg,
            **lda_row,
            **lda_angles,
        })

        for bal in [None, "downsample", "upsample"]:
            svm_res = evaluate_svm_separation(val_neurons, balance=bal, cv_folds=cv_folds, random_state=random_state)
            all_results.append({
                "area": area,
                "method": f"svm_{svm_res['balance']}",
                "n_pos": n_pos,
                "n_neg": n_neg,
                **{k: v for k, v in svm_res.items() if k not in ("balance", "X_bal", "y_bal")},
            })

        if n_permutations and n_permutations > 0:
            perm_lda = permutation_test_auc(
                val_neurons, method="lda", n_perm=n_permutations, cv_folds=cv_folds, random_state=random_state,
            )
            all_perm.append({
                "area": area, "method": "lda",
                "observed_auc": perm_lda["observed_auc"], "p_value": perm_lda["p_value"],
            })

            for bal in [None, "downsample", "upsample"]:
                perm_svm = permutation_test_auc(
                    val_neurons, method="svm", balance=bal, n_perm=n_permutations,
                    cv_folds=cv_folds, random_state=random_state,
                )
                all_perm.append({
                    "area": area, "method": f"svm_{bal if bal else 'none'}",
                    "observed_auc": perm_svm["observed_auc"], "p_value": perm_svm["p_value"],
                })

    return pd.DataFrame(all_results), pd.DataFrame(all_perm)


# ---------------------------------------------------------------------------
# Plotting (source: cell 3, cell 7)
# ---------------------------------------------------------------------------

def plot_lda_kde_projection(
    proj: np.ndarray,
    y: np.ndarray,
    results: dict,
    title: str = "",
    output_dir: str | Path | None = None,
    save_name: str | None = None,
) -> Figure:
    """KDE plot of the LDA projection, annotated with AUC/accuracy/Fisher ratio.

    `proj`/`y`/`results` are `evaluate_lda_separation`'s own return values.
    """
    proj_df = pd.DataFrame({"lda_proj": proj, "valence": y})

    fig = plt.figure(figsize=(6, 4))
    sns.kdeplot(
        data=proj_df, x="lda_proj", hue="valence",
        palette={0: "magenta", 1: "green"}, fill=True, common_norm=False,
    )

    plt.title(f"LDA Separation - {title}", fontsize=20)
    plt.xlabel("LDA Projection", fontsize=20)
    plt.ylabel("Density", fontsize=20)

    label_text = (
        f"AUC: {results['auc_mean']:.2f} ± {results['auc_std']:.2f}\n"
        f"Acc: {results['accuracy_mean']:.2f} ± {results['accuracy_std']:.2f}\n"
        f"Fisher: {results['fisher_ratio']:.2f}"
    )

    ax = plt.gca()
    if ax.get_legend() is not None:
        ax.get_legend().remove()

    ax.text(
        0.95, 0.95, label_text, transform=ax.transAxes,
        fontsize=12, color="black", ha="right", va="top",
    )

    plt.tight_layout()
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fname = (save_name or f"{title}_LDA_projection_KDE.svg").replace(" ", "_")
        fig.savefig(output_dir / fname, format="svg", dpi=300, bbox_inches="tight")

    return fig


def plot_lda_projection_histogram(
    df: pd.DataFrame,
    title: str = "",
    coord_cols: tuple[str, str, str] = ("x", "y", "z"),
    label_col: str = "valence",
) -> Figure:
    """Histogram of the LDA projection (a second, simpler style used by the notebook)."""
    X = df[list(coord_cols)].values
    y = df[label_col].values

    lda = LinearDiscriminantAnalysis(n_components=1)
    X_lda = lda.fit_transform(X, y).ravel()

    fig = plt.figure(figsize=(6, 4))
    plt.hist(X_lda[y == 1], bins=30, alpha=0.6, color="green", label="Positive")
    plt.hist(X_lda[y == 0], bins=30, alpha=0.6, color="magenta", label="Negative")
    plt.axvline(0, color="black", linestyle="--", label="Decision boundary")
    plt.xlabel("LDA Projection Value")
    plt.ylabel("Neuron Count")
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    return fig


def plot_method_comparison(results_df: pd.DataFrame, area: str) -> Figure:
    """Bar plot of CV AUC across methods for one area."""
    sub = results_df[results_df["area"] == area].copy().sort_values("auc_mean", ascending=False)
    fig = plt.figure(figsize=(7, 4))
    plt.bar(sub["method"], sub["auc_mean"])
    plt.ylabel("CV AUC")
    plt.title(f"Method comparison (AUC) — {area}")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    return fig


def plot_basic_3d_scatter(
    valence_neurons: pd.DataFrame,
    nonvalence_neurons: pd.DataFrame | None = None,
    plot_axes: tuple[str, str, str] = ("x", "y", "z"),
) -> Figure:
    """3D scatter of positive/negative (and optionally non-valence) neurons."""
    idx = [_AXIS_TO_COL[a] for a in plot_axes]

    def sel(df):
        return df[["x", "y", "z"]].values[:, idx]

    X_pos = sel(valence_neurons[valence_neurons["valence"] == 1])
    X_neg = sel(valence_neurons[valence_neurons["valence"] == 0])
    X_other = sel(nonvalence_neurons) if (nonvalence_neurons is not None and len(nonvalence_neurons) > 0) else None

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    if X_other is not None:
        ax.scatter(X_other[:, 0], X_other[:, 1], X_other[:, 2],
                   color="lightgray", alpha=0.1, s=10, label="Non-valence")

    ax.scatter(X_pos[:, 0], X_pos[:, 1], X_pos[:, 2], color="green", alpha=1, s=20, label="Positive")
    ax.scatter(X_neg[:, 0], X_neg[:, 1], X_neg[:, 2], color="magenta", alpha=1, s=20, label="Negative")

    ax.set_xlabel(plot_axes[0].upper())
    ax.set_ylabel(plot_axes[1].upper())
    ax.set_zlabel(plot_axes[2].upper())
    ax.legend()
    plt.tight_layout()
    return fig


class _Arrow3D(FancyArrowPatch):
    """3D arrow patch (source: cell 3, `plot_lda_axis_plane_3d_paper`)."""

    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, _ = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self):
        return 0


def _orthonormal_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors spanning the plane perpendicular to `normal` (Gram-Schmidt)."""
    if abs(normal[0]) < abs(normal[1]):
        v = np.array([1, 0, 0])
    else:
        v = np.array([0, 1, 0])
    v1 = np.cross(normal, v)
    v1 = v1 / (np.linalg.norm(v1) + 1e-12)
    v2 = np.cross(normal, v1)
    v2 = v2 / (np.linalg.norm(v2) + 1e-12)
    return v1, v2


def plot_lda_axis_plane_3d(
    valence_neurons: pd.DataFrame,
    lda_vector_xyz: Sequence[float],
    title: str = "LDA Axis Separating Valence Neurons",
    plot_axes: tuple[str, str, str] = ("x", "y", "z"),
    elev: float = 15,
    azim: float = 45,
    plane_size: float = 80,
    plane_grid_n: int = 10,
    arrow_len: float = 50,
    savepath: str | Path | None = None,
) -> Figure:
    """3D scatter + bidirectional LDA-axis arrow + class-centroid separation plane.

    `plot_axes` permutes which data axis appears on the plot's X/Y/Z; `elev`/
    `azim` set the viewpoint. Axis-angle labels are computed in data space but
    attached to whichever axis displays that data dimension.
    """
    idx = [_AXIS_TO_COL[a] for a in plot_axes]

    def reorder_vec(v):
        return np.asarray(v, dtype=float)[idx]

    def reorder_points(P):
        return np.asarray(P, dtype=float)[:, idx]

    X = reorder_points(valence_neurons[["x", "y", "z"]].values)
    y = valence_neurons["valence"].values

    lda_vector_plot = reorder_vec(lda_vector_xyz)
    lda_vector_plot = lda_vector_plot / (np.linalg.norm(lda_vector_plot) + 1e-12)

    centroid = X.mean(axis=0)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], X[y == 1][:, 2], c="green", alpha=0.5, label="Positive")
    ax.scatter(X[y == 0][:, 0], X[y == 0][:, 1], X[y == 0][:, 2], c="magenta", alpha=0.5, label="Negative")

    start = centroid - lda_vector_plot * arrow_len
    end = centroid + lda_vector_plot * arrow_len

    ax.add_artist(_Arrow3D(
        [start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
        mutation_scale=20, lw=2, arrowstyle="-|>", color="black",
    ))
    ax.add_artist(_Arrow3D(
        [end[0], start[0]], [end[1], start[1]], [end[2], start[2]],
        mutation_scale=20, lw=2, arrowstyle="-|>", color="black",
    ))

    pos_centroid = X[y == 1].mean(axis=0)
    neg_centroid = X[y == 0].mean(axis=0)
    midpoint = (pos_centroid + neg_centroid) / 2
    v1, v2 = _orthonormal_basis(lda_vector_plot)
    grid_range = np.linspace(-plane_size, plane_size, plane_grid_n)
    xx, yy = np.meshgrid(grid_range, grid_range)
    plane_points = midpoint.reshape(3, 1, 1) + v1.reshape(3, 1, 1) * xx + v2.reshape(3, 1, 1) * yy
    X_plane, Y_plane, Z_plane = plane_points
    ax.plot_surface(X_plane, Y_plane, Z_plane, color="gray", alpha=0.3, edgecolor="none")

    lda_vec_data = np.asarray(lda_vector_xyz, dtype=float)
    lda_vec_data = lda_vec_data / (np.linalg.norm(lda_vec_data) + 1e-12)
    axes_data = {"x": np.array([1, 0, 0]), "y": np.array([0, 1, 0]), "z": np.array([0, 0, 1])}
    axis_angles = {k: np.degrees(np.arccos(np.clip(np.dot(lda_vec_data, v), -1, 1))) for k, v in axes_data.items()}

    ax.set_xlabel(f"{AXIS_TEXT[plot_axes[0]]}\n{axis_angles[plot_axes[0]]:.0f}° to LDA", fontsize=15)
    ax.set_ylabel(f"{AXIS_TEXT[plot_axes[1]]}\n{axis_angles[plot_axes[1]]:.0f}° to LDA", fontsize=15)
    ax.set_zlabel(f"{AXIS_TEXT[plot_axes[2]]}\n{axis_angles[plot_axes[2]]:.0f}° to LDA", fontsize=16)

    ax.set_xticks(ax.get_xticks()[::2])
    ax.set_yticks(ax.get_yticks()[::2])
    ax.set_zticks(ax.get_zticks()[::2])

    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20
    ax.zaxis.labelpad = 10
    ax.tick_params(axis="both", which="major", labelsize=14, width=1)

    ax.set_title(title, fontsize=21, ha="center")
    ax.view_init(elev=elev, azim=azim)

    if savepath is not None:
        savepath = Path(savepath)
        fig.savefig(savepath, format=savepath.suffix.lstrip("."), dpi=300, bbox_inches="tight")

    return fig


def compute_svm_decision_boundary_slice(
    valence_neurons: pd.DataFrame,
    balance: str | None = "downsample",
    fixed_axis: str = "x",
    fixed_value: float | None = None,
    margin: float = 10,
    plot_axes: tuple[str, str] = ("y", "z"),
    cv_folds: int = 5,
    random_state: int = 42,
) -> dict:
    """CV-fit SVM + predicted-probability grid on one axis-aligned slice.

    `fixed_value=None` uses the median along `fixed_axis`. Returns the data
    `plot_svm_decision_boundary_slice` needs to draw the figure, kept separate
    so the numeric grid can be regression-tested independently of rendering.
    """
    df_bal = balance_classes(valence_neurons, label_col="valence", strategy=balance, random_state=random_state)
    X_bal = df_bal[["x", "y", "z"]].values
    y_bal = df_bal["valence"].values

    pipeline = make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=random_state))
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    auc_scores = cross_val_score(pipeline, X_bal, y_bal, cv=cv, scoring="roc_auc")
    acc_scores = cross_val_score(pipeline, X_bal, y_bal, cv=cv, scoring="accuracy")

    pipeline.fit(X_bal, y_bal)

    fixed_col = _AXIS_TO_COL[fixed_axis]
    plot_col0 = _AXIS_TO_COL[plot_axes[0]]
    plot_col1 = _AXIS_TO_COL[plot_axes[1]]

    if fixed_value is None:
        fixed_value = float(np.median(X_bal[:, fixed_col]))

    distances = np.abs(X_bal[:, fixed_col] - fixed_value)

    a0_range = np.linspace(X_bal[:, plot_col0].min(), X_bal[:, plot_col0].max(), 100)
    a1_range = np.linspace(X_bal[:, plot_col1].min(), X_bal[:, plot_col1].max(), 100)
    A0, A1 = np.meshgrid(a0_range, a1_range)

    grid = np.zeros((A0.size, 3), dtype=float)
    grid[:, fixed_col] = fixed_value
    grid[:, plot_col0] = A0.ravel()
    grid[:, plot_col1] = A1.ravel()

    probs = pipeline.predict_proba(grid)[:, 1].reshape(A0.shape)

    return {
        "auc_mean": float(auc_scores.mean()), "auc_std": float(auc_scores.std()),
        "acc_mean": float(acc_scores.mean()), "acc_std": float(acc_scores.std()),
        "fixed_value": fixed_value, "distances": distances,
        "A0": A0, "A1": A1, "probs": probs,
        "X_bal": X_bal, "y_bal": y_bal,
    }


def plot_svm_decision_boundary_slice(
    data: dict,
    fixed_axis: str = "x",
    margin: float = 10,
    plot_axes: tuple[str, str] = ("y", "z"),
    title_prefix: str = "SVM Decision Boundary",
    savepath: str | Path | None = None,
) -> Figure:
    """Draw the 2D SVM decision-boundary slice from `compute_svm_decision_boundary_slice`'s output."""
    plot_col0 = _AXIS_TO_COL[plot_axes[0]]
    plot_col1 = _AXIS_TO_COL[plot_axes[1]]
    X_bal, y_bal = data["X_bal"], data["y_bal"]
    A0, A1, probs = data["A0"], data["A1"], data["probs"]
    distances = data["distances"]

    fig = plt.figure(figsize=(8, 6))
    contour = plt.contourf(A0, A1, probs, levels=np.linspace(0, 1, 21), cmap="coolwarm", alpha=0.7)
    cbar = plt.colorbar(contour)
    cbar.set_label("P (Positive)", fontsize=25, labelpad=15)
    cbar.ax.tick_params(labelsize=20)

    normalized = 1 - np.clip(distances / margin, 0, 1)
    alphas = 0.2 + 0.6 * normalized

    for val, color, label in zip([1, 0], ["green", "magenta"], ["Positive", "Negative"]):
        m = y_bal == val
        plt.scatter(X_bal[m][:, plot_col0], X_bal[m][:, plot_col1], color=color, s=30, alpha=alphas[m], label=label)

    text = f"AUC: {data['auc_mean']:.2f} ± {data['auc_std']:.2f}\nAcc: {data['acc_mean']:.2f} ± {data['acc_std']:.2f}"
    plt.text(
        0.05, 0.95, text, transform=plt.gca().transAxes,
        fontsize=12, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.contour(A0, A1, probs, levels=[0.5], colors="black", linewidths=2)

    plt.xlabel(AXIS_TEXT[plot_axes[0]], fontsize=23)
    plt.ylabel(AXIS_TEXT[plot_axes[1]], fontsize=23)

    ax = plt.gca()
    ax.tick_params(axis="both", which="major", labelsize=15, width=1)

    plt.title(
        f"{title_prefix}\n ({plot_axes[0].upper()}{plot_axes[1].upper()} slice @ "
        f"{fixed_axis.upper()}={data['fixed_value']:.1f})",
        fontsize=18, pad=20,
    )
    plt.tight_layout()

    if savepath is not None:
        savepath = Path(savepath)
        fig.savefig(savepath, format=savepath.suffix.lstrip("."), dpi=300, bbox_inches="tight")

    return fig
