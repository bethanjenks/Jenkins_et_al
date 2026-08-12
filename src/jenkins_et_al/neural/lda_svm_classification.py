"""7-odor classification from whole-brain neural population responses, compared
across two classifiers: LDA (with PCA preprocessing) and linear SVM. Per fish
and per brain area: cross-validated classification accuracy, a permutation
test for significance, Fisher's-method + FDR combination across fish, and
"template matching style" confusion-matrix analysis/visualization.

Ported from notebooks/neural/lda_svm_classification.ipynb; logic unchanged,
split into loading/processing/plotting functions per the /tdd REFACTOR
checklist. **Never run to completion as saved**: every cell's `execution_count`
is `None`.

**Independent data source**: reads the whole-brain response-vector HDF5
directly (`7dpf_wb_population_general_response_vector.h5`), not the
positive/negative valence-correlation CSVs #11-#13 share.

**Owns the plain-ratio permutation-test formula**: `permutation_test_classifier`'s
p-value is `sum(null >= obs) / n_perm`, with no "+1" correction -- can return
exactly 0. Confirmed different from #13's `(sum(perm >= obs) + 1) / (n_perm + 1)`
formula and from #9's own permutation test; not unified, per the porting
plan's note that this notebook owns a third, independent formula.

**Owns a third, independent Fisher+FDR combination**: `combine_fish_statistics`
(`scipy.stats.combine_pvalues(method="fisher")` + `statsmodels`'
`multipletests(method="fdr_bh")`) -- confirmed different call sites from #9's
and notebook #8's (not yet ported) own implementations; not unified.

**Confirmed non-deterministic, inherent to the notebook, not this port**:
`permutation_test_classifier`'s shuffle has no seed anywhere
(`sklearn.utils.shuffle(y, random_state=None)`) -- every call draws fresh
global-RNG entropy, so exact reproduction is impossible even in principle.
The full real-data pipeline (10 fish x ~36 qualifying areas x 1000
permutations x 2 classifiers) is also expensive: timed at ~2.5-3 hours for a
complete run. Per the user's decision (2026-08-12, mirroring the same call
already made for #9's near-identical situation): **not golden-captured**.
Verified instead against small synthetic data (deterministic per function)
plus a real-data smoke test (a handful of areas, reduced `n_permutations`)
confirming the real pipeline runs end-to-end without error and lands in a
sane range -- see tests/test_lda_svm_classification.py.

**Refactor-shape change, parametrized for testability**: `permutation_test_classifier`
gained an optional `shuffle_random_state: int | None = None` parameter
(default preserves the notebook's exact unseeded behavior; named distinctly
from any classifier-internal `random_state` forwarded via `**kwargs` so the
two can never collide). When given an int, a single `numpy.random.RandomState`
instance is built once and reused across all `n_permutations` calls to
`sklearn.utils.shuffle` -- the instance's internal state still advances every
call (so permutations still differ from each other), but the whole sequence
becomes reproducible run-to-run. `PCA`'s/`StratifiedKFold`'s/`LinearSVC`'s own
`random_state=42` were already fixed in
the notebook; kept as each function's default, now a parameter.

**Refactor-shape change, DRY**: the notebook's own per-fish/per-area data
loading and pivoting, and its confusion-matrix aggregation, are each
duplicated verbatim between the LDA section (cells ~20-39) and the SVM
section (cells ~42-59) -- identical code, differing only in which
classification function is called. Unified into `load_fish_stimulus_data`,
`pivot_area_response_matrix`, `run_classification_pipeline`,
`aggregate_confusion_matrices`, and `aggregate_confusion_matrices_per_fish`,
each now shared by both classifiers instead of duplicated. This is
within-notebook duplication, not a premature cross-notebook unification --
this project's other independent Fisher+FDR/permutation-formula
implementations (#9, #13) are left untouched.

**Refactor-shape change, DRY (REFACTOR pass)**: `run_lda_classification`/
`run_svm_classification` shared identical CV-scoring + full-data-refit +
confusion-matrix logic after their differing preprocessing/classifier-object
setup, extracted to `_cross_validate_and_refit`. `aggregate_confusion_matrices`/
`aggregate_confusion_matrices_per_fish`/`filter_significant_confusion_matrices`
shared an identical sum-then-average pattern over different iterables,
extracted to `_average_confusion_matrices`. Both are pure code-shape changes;
re-verified against the real-data smoke test after applying.

**Confirmed dead code, not reproduced as such**: `jitter` is defined but never
called anywhere else in the notebook (cell "Visualization Utilities"). Kept as
a standalone utility, matching the precedent set by #6's `is_within_3d_mask`.

**Confirmed asymmetry between the LDA and SVM sections, not a bug**: the LDA
section has a "Significant Areas Only" confusion-matrix figure (cells 34-35)
that the otherwise-parallel SVM section has no equivalent of. Reproduced as-is
via `filter_significant_confusion_matrices` + `plot_aggregate_confusion_matrix`,
callable for either classifier but only demonstrated for LDA, matching the
notebook's own structure.

**Noted, not acted on**: this notebook's `OUTPUT_DIR` points at
`/Volumes/LaCie/larval_HuC/imaging` (the real production data drive) --
no I/O happens inside any ported function here (all return DataFrames/arrays;
saving to CSV is left to the caller, as in every other ported module), so
there's no risk of this port overwriting production data.

**Noted, not acted on**: `ATTRACTIVE`/`AVERSIVE` here use the spelling
`"fex_3"` (underscore), while `config.STIMULUS_LABELS` only knows `"fex3"`
(no underscore) for the same low-concentration food-extract stimulus. This
notebook doesn't consume `config.STIMULUS_LABELS` (it defines its own local
valence groupings, restricted to `KEY_STIMULI`), so it doesn't need the alias
-- flagged here in case a later port needs to add `"fex_3"` to
`config.STIMULUS_LABELS`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from scipy.stats import combine_pvalues
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle
from statsmodels.stats.multitest import multipletests


# ---------------------------------------------------------------------------
# Loading (source: cell 22/44's identical data-loading preamble)
# ---------------------------------------------------------------------------

def load_fish_stimulus_data(
    neural_data_path: str | Path,
    fb_id: str,
    hb_id: str,
    key_stimuli: Sequence[str],
) -> pd.DataFrame:
    """Load one fish's forebrain+hindbrain sessions, filtered to `key_stimuli`.

    Parses `coords_serialised` (a JSON string) into a `coords` array column.
    """
    data_fb = pd.read_hdf(neural_data_path, where=f'fish_id == "{fb_id}"')
    data_hb = pd.read_hdf(neural_data_path, where=f'fish_id == "{hb_id}"')
    data = pd.concat([data_fb, data_hb])

    stimulus_df = data[data["stimulus"].isin(key_stimuli)].copy()
    stimulus_df["coords"] = stimulus_df["coords_serialised"].apply(lambda x: np.asarray(json.loads(x)))
    stimulus_df.drop("coords_serialised", axis=1, inplace=True)
    return stimulus_df


def pivot_area_response_matrix(stimulus_df: pd.DataFrame, area: str) -> tuple[np.ndarray, np.ndarray]:
    """Pivot one brain area's responses to `(X, y)`: trials x neurons, and each trial's stimulus.

    Neurons with any missing trial are dropped (`dropna`), matching the
    notebook's own `resp_clean`.
    """
    area_df = stimulus_df.query("area == @area")
    df_resp = area_df.pivot_table(index=["fish_id", "neuron_id"], columns=["stimulus", "trial_number"], values="resp")
    stimulus_names = [col[0] for col in df_resp.columns]
    resp_clean = df_resp.dropna()

    X = resp_clean.values.T
    y = np.array(stimulus_names)
    return X, y


# ---------------------------------------------------------------------------
# Classification (source: cell 7, cell 8)
# ---------------------------------------------------------------------------

def _cross_validate_and_refit(clf, X: np.ndarray, y: np.ndarray, n_folds: int, random_state: int) -> dict:
    """CV accuracy + a confusion matrix from a full-data refit -- shared by both classifiers below."""
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(clf, X, y, cv=cv)

    clf.fit(X, y)
    y_pred = clf.predict(X)

    unique_labels = sorted(set(y))
    cm = confusion_matrix(y, y_pred, labels=unique_labels)

    return {
        "mean_score": np.mean(cv_scores),
        "cv_scores": cv_scores,
        "confusion_matrix": cm,
        "labels": unique_labels,
    }


def run_lda_classification(X: np.ndarray, y: np.ndarray, n_pca: int = 5, n_folds: int = 3, random_state: int = 42) -> dict:
    """PCA -> LDA classification: CV accuracy + a confusion matrix from a full-data refit."""
    pca = PCA(n_components=n_pca, random_state=random_state)
    X_pca = pca.fit_transform(X)
    return _cross_validate_and_refit(LinearDiscriminantAnalysis(), X_pca, y, n_folds, random_state)


def run_svm_classification(X: np.ndarray, y: np.ndarray, n_folds: int = 3, random_state: int = 42) -> dict:
    """Linear SVM classification (no PCA): CV accuracy + a confusion matrix from a full-data refit."""
    svm = LinearSVC(random_state=random_state, max_iter=10000)
    return _cross_validate_and_refit(svm, X, y, n_folds, random_state)


# ---------------------------------------------------------------------------
# Permutation testing (source: cell 10)
# ---------------------------------------------------------------------------

def permutation_test_classifier(
    X: np.ndarray,
    y: np.ndarray,
    classifier_func: Callable[..., dict],
    n_permutations: int = 1000,
    shuffle_random_state: int | None = None,
    **kwargs,
) -> float:
    """One-sided permutation p-value: `sum(null_accuracy >= real_accuracy) / n_permutations`.

    No "+1" correction, matching the notebook -- can return exactly 0.
    `shuffle_random_state=None` (the default) reproduces the notebook's own
    unseeded behavior exactly (every shuffle draws fresh global-RNG entropy).
    Passing an int builds one `numpy.random.RandomState`, reused across every
    permutation so successive shuffles still differ from each other but the
    whole sequence is reproducible run-to-run. Named distinctly from any
    `random_state` a classifier itself takes (forwarded via `**kwargs`) so the
    two can never collide -- `run_lda_classification`/`run_svm_classification`
    both also accept their own, unrelated `random_state`.
    """
    real_result = classifier_func(X, y, **kwargs)
    real_accuracy = real_result["mean_score"]

    rng = np.random.RandomState(shuffle_random_state) if shuffle_random_state is not None else None

    null_accuracies = []
    for _ in range(n_permutations):
        shuffled_y = shuffle(y, random_state=rng)
        perm_result = classifier_func(X, shuffled_y, **kwargs)
        null_accuracies.append(perm_result["mean_score"])

    return float(np.sum(np.array(null_accuracies) >= real_accuracy) / n_permutations)


# ---------------------------------------------------------------------------
# Per-fish/per-area pipeline (source: cell 22/44, unified)
# ---------------------------------------------------------------------------

def run_classification_pipeline(
    neural_data_path: str | Path,
    fish_ids: Sequence[tuple[str, str]],
    key_stimuli: Sequence[str],
    classifier_func: Callable[..., dict],
    classifier_kwargs: dict | None = None,
    n_permutations: int = 1000,
    min_neurons: int = 5,
    permutation_random_state: int | None = None,
) -> tuple[dict, dict]:
    """Classify each area of each fish, with a permutation test per area.

    Shared by both the LDA and SVM sections (`classifier_func` is
    `run_lda_classification` or `run_svm_classification`) -- the notebook
    duplicated this loop verbatim between the two.

    Returns `(all_results, all_confusion_matrices)`, both keyed
    `{"{fb_id}_{hb_id}": {area: ...}}`, matching the notebook's own
    `all_lda_results`/`all_lda_confusion_matrices` (or `_svm_`) dicts.
    """
    classifier_kwargs = classifier_kwargs or {}
    all_results: dict = {}
    all_confusion_matrices: dict = {}

    for fb_id, hb_id in fish_ids:
        stimulus_df = load_fish_stimulus_data(neural_data_path, fb_id, hb_id, key_stimuli)
        unique_areas = stimulus_df["area"].unique()

        fish_results = {}
        fish_conf_matrices = {}

        for area in unique_areas:
            X, y = pivot_area_response_matrix(stimulus_df, area)
            n_neurons = X.shape[1]  # X is trials x neurons (post-transpose); neurons are the feature columns.
            if n_neurons < min_neurons:
                continue

            result = classifier_func(X, y, **classifier_kwargs)
            p_value = permutation_test_classifier(
                X, y, classifier_func=classifier_func,
                n_permutations=n_permutations, shuffle_random_state=permutation_random_state,
                **classifier_kwargs,
            )

            fish_results[area] = {
                "mean_score": result["mean_score"],
                "p_value": p_value,
                "n_neurons": n_neurons,
            }
            fish_conf_matrices[area] = {
                "confusion_matrix": result["confusion_matrix"],
                "labels": result["labels"],
            }

        fish_key = f"{fb_id}_{hb_id}"
        all_results[fish_key] = fish_results
        all_confusion_matrices[fish_key] = fish_conf_matrices

    return all_results, all_confusion_matrices


# ---------------------------------------------------------------------------
# Statistical combination (source: cell 12, cell 24/46's reorganize step)
# ---------------------------------------------------------------------------

def reorganize_by_area(all_results: dict) -> dict:
    """`{fish_key: {area: {...}}}` -> `{area: [(fish_key, mean_score, p_value), ...]}`."""
    all_areas: dict = {}
    for fish_id, fish_results in all_results.items():
        for area_name, area_results in fish_results.items():
            all_areas.setdefault(area_name, []).append(
                (fish_id, area_results["mean_score"], area_results["p_value"])
            )
    return all_areas


def combine_fish_statistics(all_areas_dict: dict, fdr_alpha: float = 0.05) -> pd.DataFrame:
    """Combine each area's per-fish p-values via Fisher's method, then FDR-correct across areas.

    `combine_pvalues` failures (e.g. all p-values are 1.0) fall back to
    `combined_p = 1.0`, matching the notebook's own bare `except`.
    """
    combined_stats = {}

    for area, fish_results in all_areas_dict.items():
        p_values = [p for _, _, p in fish_results]
        scores = [s for _, s, _ in fish_results]

        try:
            _, combined_p = combine_pvalues(p_values, method="fisher")
        except Exception:
            combined_p = 1.0

        combined_stats[area] = {
            "combined_p": combined_p,
            "mean_accuracy": np.mean(scores),
            "std_accuracy": np.std(scores),
            "n_fish": len(fish_results),
        }

    areas = list(combined_stats.keys())
    p_array = [combined_stats[a]["combined_p"] for a in areas]
    _, p_corrected, _, _ = multipletests(p_array, alpha=fdr_alpha, method="fdr_bh")

    for area, p_fdr in zip(areas, p_corrected):
        combined_stats[area]["fdr_corrected"] = p_fdr
        combined_stats[area]["is_significant"] = p_fdr < fdr_alpha

    records = [{"area": area, **stats} for area, stats in combined_stats.items()]
    return pd.DataFrame(records)[
        ["area", "mean_accuracy", "std_accuracy", "combined_p", "fdr_corrected", "is_significant", "n_fish"]
    ].sort_values("fdr_corrected")


def compare_methods(df_lda_combined: pd.DataFrame, df_svm_combined: pd.DataFrame) -> dict:
    """Mean-accuracy comparison between the two methods (source: cell 61's "Method Comparison")."""
    lda_mean = df_lda_combined["mean_accuracy"].mean()
    svm_mean = df_svm_combined["mean_accuracy"].mean()

    if lda_mean > svm_mean:
        winner = "lda"
    elif svm_mean > lda_mean:
        winner = "svm"
    else:
        winner = "tie"

    return {
        "lda_mean_accuracy": float(lda_mean),
        "svm_mean_accuracy": float(svm_mean),
        "lda_n_significant": int(df_lda_combined["is_significant"].sum()),
        "svm_n_significant": int(df_svm_combined["is_significant"].sum()),
        "winner": winner,
    }


# ---------------------------------------------------------------------------
# Confusion matrix aggregation + analysis (source: cell 29/51, cell 33/55, cell 15)
# ---------------------------------------------------------------------------

def _average_confusion_matrices(matrices: Sequence[np.ndarray], n_stimuli: int) -> tuple[np.ndarray, int]:
    """Sum then average a sequence of same-shaped confusion matrices -- shared by the three functions below."""
    total = np.zeros((n_stimuli, n_stimuli))
    for cm in matrices:
        total += cm
    return total / len(matrices), len(matrices)


def aggregate_confusion_matrices(all_confusion_matrices: dict, key_stimuli: Sequence[str]) -> tuple[np.ndarray, int]:
    """Average every fish's every area's confusion matrix."""
    matrices = [
        conf_data["confusion_matrix"]
        for fish_conf_dict in all_confusion_matrices.values()
        for conf_data in fish_conf_dict.values()
    ]
    return _average_confusion_matrices(matrices, len(key_stimuli))


def aggregate_confusion_matrices_per_fish(fish_conf_dict: dict, key_stimuli: Sequence[str]) -> tuple[np.ndarray, int]:
    """Average one fish's confusion matrices across its areas."""
    matrices = [conf_data["confusion_matrix"] for conf_data in fish_conf_dict.values()]
    return _average_confusion_matrices(matrices, len(key_stimuli))


def filter_significant_confusion_matrices(
    all_confusion_matrices: dict,
    significant_areas: Sequence[str],
    key_stimuli: Sequence[str],
) -> tuple[np.ndarray, int]:
    """Average confusion matrices restricted to `significant_areas` (source: cell 35, LDA-only in the notebook)."""
    matrices = [
        conf_data["confusion_matrix"]
        for fish_conf_dict in all_confusion_matrices.values()
        for area, conf_data in fish_conf_dict.items()
        if area in significant_areas
    ]
    return _average_confusion_matrices(matrices, len(key_stimuli))


def analyze_confusion_pairs(
    cm: np.ndarray,
    labels: Sequence[str],
    attractive: Sequence[str],
    aversive: Sequence[str],
) -> dict:
    """Rank off-diagonal confusion pairs and compare within- vs. cross-valence confusion rates."""
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    confusion_pairs = [
        (labels[i], labels[j], cm_norm[i, j])
        for i in range(len(labels)) for j in range(len(labels)) if i != j
    ]
    confusion_pairs.sort(key=lambda p: p[2], reverse=True)

    within_attr = [p for p in confusion_pairs if p[0] in attractive and p[1] in attractive]
    within_aver = [p for p in confusion_pairs if p[0] in aversive and p[1] in aversive]
    cross = [
        p for p in confusion_pairs
        if (p[0] in attractive and p[1] in aversive) or (p[0] in aversive and p[1] in attractive)
    ]

    return {
        "top_confusions": confusion_pairs[:10],
        "within_attractive": within_attr,
        "within_aversive": within_aver,
        "cross_valence": cross,
        "within_attr_avg": np.mean([p[2] for p in within_attr]) if within_attr else 0,
        "within_aver_avg": np.mean([p[2] for p in within_aver]) if within_aver else 0,
        "cross_avg": np.mean([p[2] for p in cross]) if cross else 0,
    }


# ---------------------------------------------------------------------------
# Visualization utilities (source: cell 17)
# ---------------------------------------------------------------------------

def jitter(values: np.ndarray, amount: float = 0.05) -> np.ndarray:
    """Add uniform random jitter for scatter-plot visualization. Unused in the notebook itself."""
    return values + np.random.uniform(-amount, amount, len(values))


# ---------------------------------------------------------------------------
# Plotting (source: cell 14, cell 31/53, cell 33/55)
# ---------------------------------------------------------------------------

def plot_confusion_matrix_styled(
    cm: np.ndarray,
    labels: Sequence[str],
    title: str = "Confusion Matrix",
    ax: plt.Axes | None = None,
) -> AxesImage:
    """Row-normalized confusion-matrix heatmap with text annotations ("template matching" style)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))

    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    im = ax.imshow(cm_norm, cmap="Blues", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Predicted Stimulus", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Stimulus", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")

    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", color=color, fontsize=9, fontweight="bold")

    ax.set_xticks(np.arange(len(labels)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(labels)) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)

    return im


def plot_aggregate_confusion_matrix(cm: np.ndarray, labels: Sequence[str], title: str) -> Figure:
    """Aggregate confusion-matrix figure with colorbar (source: cell 31/53)."""
    fig, ax = plt.subplots(figsize=(10, 8))

    im = plot_confusion_matrix_styled(cm, labels, title=title, ax=ax)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Prediction Probability", fontsize=12, fontweight="bold")

    plt.tight_layout()
    return fig


def plot_per_fish_confusion_matrices(
    all_confusion_matrices: dict,
    fish_ids: Sequence[tuple[str, str]],
    key_stimuli: Sequence[str],
    suptitle: str,
) -> Figure:
    """2x5 grid of per-fish confusion matrices, averaged across each fish's areas (source: cell 33/55)."""
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    simplified_fish_ids = {f"{fb}_{hb}": f"F{i + 1}" for i, (fb, hb) in enumerate(fish_ids)}

    for ax, (fish_key, fish_conf_dict) in zip(axes, all_confusion_matrices.items()):
        fish_cm_avg, n_areas = aggregate_confusion_matrices_per_fish(fish_conf_dict, key_stimuli)
        plot_confusion_matrix_styled(
            fish_cm_avg, list(key_stimuli),
            title=f"{simplified_fish_ids[fish_key]} (n={n_areas} areas)", ax=ax,
        )

    fig.suptitle(suptitle, fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig
