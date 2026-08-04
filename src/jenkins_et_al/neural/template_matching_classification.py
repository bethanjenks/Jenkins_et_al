"""Template-matching classification of neural population responses by brain area.

Ported from notebooks/neural/template_matching_classification.ipynb; logic is
unchanged from the notebook, only moved into a module, split into
loading/processing/stats/plotting functions, and given a docstring/type-hint
pass per the /tdd REFACTOR checklist.

**This notebook has never been run to completion as saved** -- every cell's
`execution_count` is `None`, and it contains two confirmed blocking bugs:

1. A `NameError`: a Spearman/Pearson correlation cell (and a `df.groupby(...)`
   / scatterplot cell) reference a variable `df` several cells before `df` is
   ever assigned (under a later "### Save Odor Results" heading). Run
   top-to-bottom, execution cannot reach past the first of these. **Not
   ported** -- the user chose to skip this diagnostic aside rather than guess
   its intended input.
2. A `KeyError`: `plot_classification_accuracy_boxplot` (defined twice; the
   second silently overrides the first -- only the second, "live" definition
   is ported here) hardcodes an 18-entry palette covering only hindbrain area
   codes. It's called twice: once with the hindbrain area list (fine), and
   once with the *forebrain* area list -- every forebrain area (`"Pal"`,
   `"SubP"`, `"OE"`, ...) is absent from that palette, so `palette[area]`
   would raise `KeyError` immediately (confirmed: all of them are present in
   the underlying data). **Fixed here as a refactor-shape change, not an
   analysis change**: `palette` is now a parameter (default
   `config.HINDBRAIN_AREA_COLORS`, matching the hardcoded value), so the
   forebrain call becomes reproducible by passing `config.FOREBRAIN_AREA_COLORS`
   -- a figure the original notebook could never actually produce as saved.

**Real analysis-choice deviation, at the user's explicit request**: the
notebook's own forebrain-vs-OB cell used unpaired Mann-Whitney U
(`compute_area_pvalues_unpaired_mannwhitney` existed in an earlier version of
this port). The user asked for Wilcoxon instead, matching the hindbrain
figure's test. `compute_area_pvalues_paired_wilcoxon` already computes a
p-value for every non-reference area in one pass (not just the hindbrain
ones), so both figures now share its output -- no separate Mann-Whitney
function remains. This mirrors the same Mann-Whitney -> Wilcoxon change the
user made directly to `lifetime_sparseness.ipynb` before its port, except
here the original notebook itself is untouched (still literally uses
Mann-Whitney for this cell) -- the deviation lives only in this port, by the
user's direction.

**The odor/valence classification loops use unseeded randomness**
(`sklearn.utils.shuffle(y, random_state=None)` inside `permutation_test`, and
plain `np.random.uniform` inside `jitter`) and are individually expensive
(1000 permutations x cross-validated template matching, per brain area, per
fish, x2 for odor and valence) -- exact reproduction is impossible and a real
run is on the order of hours. Per the user's decision, these are verified
only against small synthetic inputs (sanity checks), not a golden capture
against the real ~200k-neuron dataset.

**The two real, deterministic, paper-relevant figures** (hindbrain-vs-OB and
forebrain-vs-OB, both Wilcoxon-vs-OB) do **not** depend on
that expensive loop at all: an "OPTIONAL" cell immediately discards the
freshly-computed per-fish/area accuracy table and reloads a pre-existing one
from a relative-path CSV (`classification_accuracy_per_fish_early_resp.csv`,
found off-repo at
`/Users/bethanjenkins/Documents/valence_paper_code/Sense/`) instead. These
two figures + their stats are golden-verified against that real file.

Reuses `jenkins_et_al.neural.lifetime_sparseness.is_within_3d_mask` (identical
logic) rather than redefining it, and
`jenkins_et_al.config.AREA_SHORTHAND`/`HINDBRAIN_AREA_COLORS`/
`FOREBRAIN_AREA_COLORS`.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import combine_pvalues, wilcoxon
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import cosine_distances
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from statsmodels.stats.multitest import multipletests

from jenkins_et_al.config import AREA_SHORTHAND, HINDBRAIN_AREA_COLORS
from jenkins_et_al.neural.lifetime_sparseness import is_within_3d_mask
from jenkins_et_al.plotting import plot_area_comparison_boxplot

#: Source: cell 2 ("Configuration Parameters").
FISH_IDS = (
    ("230713_fb", "230713_hb"), ("230714_fb", "230714_hb"),
    ("230720_fb", "230720_hb"), ("230727_fb", "230727_hb"),
    ("230728_fb", "230728_hb"), ("230810_fb", "230810_hb"),
    ("230811_fb", "230811_hb"), ("230817_fb", "230817_hb"),
    ("230818_fb", "230818_hb"), ("230919_fb", "230919_hb"),
)

#: 7 odors: 3 attractive, 4 aversive. Source: cell 2.
ORDERED_STIMULI = ("fex_1", "kw", "pro_2.5mm", "ade", "cad_2.5mm", "ph4.5", "qui_2.5mm")

MIN_NEURONS_ODOR = 10
N_FOLDS_ODOR = 3
N_PERMUTATIONS = 1000

#: Source: cell 2.
SELECTED_STIMULI_VALENCE = ("ade", "cad_2.5mm", "fex_1", "kw", "pro_2.5mm", "qui_2.5mm")
VALENCE_MAP = {"ade": 0, "cad_2.5mm": 0, "qui_2.5mm": 0, "fex_1": 1, "kw": 1, "pro_2.5mm": 1}
MIN_NEURONS_VALENCE = 5
N_FOLDS_VALENCE = 5
FDR_ALPHA = 0.05

#: Source: cell 33 ("Statistics" for the hindbrain/Wilcoxon boxplot).
HINDBRAIN_AREA_ORDER = (
    "OB", "PTec", "Tect", "NI", "Cb", "SGN", "sDMO", "sRaphe", "sVMO",
    "aTriMN", "itDMO", "itVMO", "iRaphe", "fMN", "ifDMO", "ifVMO", "vMN", "VSL",
)

#: Source: cell 34 ("Statistics" for the forebrain boxplot -- the notebook's
#: own cell used Mann-Whitney U here; the port uses Wilcoxon, see module docstring).
FOREBRAIN_AREA_ORDER = ("OE", "OB", "Pal", "SubP", "POA", "PTh", "EmT", "dHb", "vHb", "PT", "HR", "HI")

REFERENCE_AREA = "OB"


# ---------------------------------------------------------------------------
# Template matching classification + permutation test (source: cells 12, 14)
# ---------------------------------------------------------------------------

def template_matching_cross_validation(
    X: np.ndarray, y: np.ndarray, labels: list, n_splits: int = 3,
) -> dict[str, object]:
    """Cross-validated template-matching classification.

    Per fold: build one template per label (mean training-set response for
    that label), classify each test trial by minimum cosine distance to a
    template. `StratifiedKFold(shuffle=True, random_state=42)` -- seeded,
    deterministic given fixed `X`/`y`.

    Returns `cv_scores` (per-fold accuracy), `mean_score`, and
    `confusion_matrix` (fold-summed then averaged, ordered per `labels`).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = []
    conf_matrix_sum = np.zeros((len(labels), len(labels)))
    n_folds = 0

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        templates = {label: np.mean(X_train[y_train == label], axis=0) for label in labels}

        y_pred = []
        for test_sample in X_test:
            distances = {
                label: cosine_distances([test_sample], [templates[label]]).flatten()[0]
                for label in labels
            }
            y_pred.append(min(distances, key=distances.get))

        cv_scores.append(np.mean(np.array(y_pred) == y_test))
        conf_matrix_sum += confusion_matrix(y_test, y_pred, labels=labels)
        n_folds += 1

    return {
        "cv_scores": cv_scores,
        "mean_score": np.mean(cv_scores),
        "confusion_matrix": conf_matrix_sum / n_folds,
    }


def permutation_test(
    X: np.ndarray, y: np.ndarray, labels: list, n_splits: int = 3, n_permutations: int = N_PERMUTATIONS,
) -> float:
    """One-tailed permutation p-value: fraction of label-shuffled accuracies >= the real accuracy.

    Uses `sklearn.utils.shuffle(y, random_state=None)` -- **unseeded**, so
    this function is not exactly reproducible run-to-run; verify on synthetic
    data only (see module docstring).
    """
    real_accuracy = template_matching_cross_validation(X, y, labels, n_splits)["mean_score"]

    null_accuracies = [
        template_matching_cross_validation(X, shuffle(y, random_state=None), labels, n_splits)["mean_score"]
        for _ in range(n_permutations)
    ]

    return np.sum(np.array(null_accuracies) >= real_accuracy) / n_permutations


def jitter(values: np.ndarray, amount: float = 0.05) -> np.ndarray:
    """Add uniform jitter in `[-amount, amount]` to `values`.

    Uses the global `np.random` state -- **unseeded**, not exactly
    reproducible run-to-run.
    """
    return values + np.random.uniform(-amount, amount, len(values))


# ---------------------------------------------------------------------------
# Fisher's method + FDR combination across fish (source: cell 16)
# ---------------------------------------------------------------------------

def combine_fish_statistics(all_areas_dict: dict[str, list[tuple]], fdr_alpha: float = FDR_ALPHA) -> pd.DataFrame:
    """Combine per-fish p-values per area via Fisher's method, then FDR-correct (BH) across areas.

    `all_areas_dict[area]` is a list of `(fish_id, accuracy, p_value)` tuples.
    Returns one row per area, sorted by FDR-corrected p-value.
    """
    combined_stats = {}

    for area, fish_results in all_areas_dict.items():
        p_values = [p for _, _, p in fish_results]
        scores = [s for _, s, _ in fish_results]

        try:
            _, combined_p = combine_pvalues(p_values, method="fisher")
        except ValueError:
            combined_p = 1.0

        combined_stats[area] = {
            "combined_p": combined_p,
            "mean_accuracy": np.mean(scores),
            "std_accuracy": np.std(scores),
            "n_fish": len(fish_results),
        }

    areas = list(combined_stats.keys())
    _, p_corrected, _, _ = multipletests(
        [combined_stats[area]["combined_p"] for area in areas], alpha=fdr_alpha, method="fdr_bh",
    )

    records = []
    for area, p_fdr in zip(areas, p_corrected):
        stats = combined_stats[area]
        records.append({
            "area": area,
            "mean_accuracy": stats["mean_accuracy"],
            "std_accuracy": stats["std_accuracy"],
            "combined_p": stats["combined_p"],
            "fdr_corrected": p_fdr,
            "is_significant": p_fdr < fdr_alpha,
            "n_fish": stats["n_fish"],
        })

    return pd.DataFrame(records).sort_values("fdr_corrected")


# ---------------------------------------------------------------------------
# Odor classification loop (source: cells 21, 23)
# ---------------------------------------------------------------------------

def _load_fish_data_with_areas(
    neural_data_path: Path, fb_id: str, hb_id: str, stimuli: tuple[str, ...], nmlf_mask: np.ndarray,
    prefix_neuron_id: bool = False,
) -> pd.DataFrame:
    """Load + concatenate one fish's forebrain/hindbrain data, filter to `stimuli`, reassign nMLF-mask neurons.

    `prefix_neuron_id` matches the valence loop's `neuron_id = fish_id + neuron_id`
    (needed there to match `load_valence_neurons`'s IDs; the odor loop doesn't
    do this). Shared by `compute_odor_classification_results` and
    `compute_valence_classification_results` (source cells 21 and 53 are
    otherwise identical here).
    """
    data_fb = pd.read_hdf(neural_data_path, where=f'fish_id == "{fb_id}"')
    data_hb = pd.read_hdf(neural_data_path, where=f'fish_id == "{hb_id}"')
    data = pd.concat([data_fb, data_hb])

    filtered_df = data[data["stimulus"].isin(stimuli)].copy()
    filtered_df["coords"] = filtered_df["coords_serialised"].apply(lambda x: np.asarray(json.loads(x)))
    if prefix_neuron_id:
        filtered_df["neuron_id"] = filtered_df["fish_id"] + filtered_df["neuron_id"]
    filtered_df = filtered_df.drop("coords_serialised", axis=1, errors="ignore")
    filtered_df["area"] = filtered_df.apply(
        lambda row: "nMLF" if is_within_3d_mask(row["coords"], nmlf_mask) else row["area"], axis=1,
    )

    return filtered_df


def compute_odor_classification_results(
    neural_data_path: Path,
    nmlf_mask: np.ndarray,
    fish_ids: tuple[tuple[str, str], ...] = FISH_IDS,
    ordered_stimuli: tuple[str, ...] = ORDERED_STIMULI,
    min_neurons: int = MIN_NEURONS_ODOR,
    n_folds: int = N_FOLDS_ODOR,
    n_permutations: int = N_PERMUTATIONS,
) -> tuple[dict, dict]:
    """Per-fish, per-area 7-odor template-matching classification + permutation test.

    For each fish: load forebrain+hindbrain data, reassign nMLF-mask neurons
    (unlike lifetime_sparseness's dead nMLF reassignment, this one runs
    *before* the per-area pivot, so it's a real, effective step here), then
    for each area with >= `min_neurons` clean-response neurons, classify and
    test. Expensive (1000 permutations/area/fish) and non-deterministic
    (unseeded shuffle inside `permutation_test`) -- see module docstring.

    Returns `(all_odor_results, all_odor_conf_matrices)`, each keyed by
    `"{fb_id}_{hb_id}"` then by full area name.
    """
    all_odor_results: dict[str, dict] = {}
    all_odor_conf_matrices: dict[str, dict] = {}

    for fb_id, hb_id in fish_ids:
        stimulus_df = _load_fish_data_with_areas(neural_data_path, fb_id, hb_id, ordered_stimuli, nmlf_mask)

        fish_results: dict[str, dict] = {}
        fish_conf_matrices: dict[str, dict] = {}

        for area in stimulus_df["area"].unique():
            area_df = stimulus_df.query("area == @area")

            df_resp = area_df.pivot_table(index=["fish_id", "neuron_id"], columns=["stimulus", "trial_number"], values="resp")
            df_resp = df_resp.reindex(columns=sorted(df_resp.columns, key=lambda x: ordered_stimuli.index(x[0])))
            resp_clean = df_resp.dropna()

            if len(resp_clean) < min_neurons:
                continue

            X = resp_clean.values.T
            y = np.array([col[0] for col in resp_clean.columns])

            cv_result = template_matching_cross_validation(X, y, list(ordered_stimuli), n_splits=n_folds)
            p_value = permutation_test(X, y, list(ordered_stimuli), n_splits=n_folds, n_permutations=n_permutations)

            fish_results[area] = {
                "accuracy_score": cv_result["mean_score"],
                "p_value": p_value,
                "n_neurons": len(resp_clean),
            }
            fish_conf_matrices[area] = {
                "confusion_matrix": cv_result["confusion_matrix"],
                "stimulus_labels": ordered_stimuli,
            }

        fish_key = f"{fb_id}_{hb_id}"
        all_odor_results[fish_key] = fish_results
        all_odor_conf_matrices[fish_key] = fish_conf_matrices

    return all_odor_results, all_odor_conf_matrices


def average_confusion_matrices_by_area(
    area_order: tuple[str, ...], all_odor_conf_matrices: dict,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Row-normalize each fish's confusion matrix, then average across fish, per area.

    Row-normalizing first means every fish contributes equally regardless of
    its trial count.
    """
    area_cm_avg = {}
    area_cm_n_fish = {}

    for area in area_order:
        fish_cms = []

        for fish_conf_dict in all_odor_conf_matrices.values():
            if area not in fish_conf_dict:
                continue

            cm = fish_conf_dict[area]["confusion_matrix"].astype(float)
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_norm = np.divide(cm, row_sums, out=np.full_like(cm, np.nan, dtype=float), where=row_sums != 0)
            fish_cms.append(cm_norm)

        if fish_cms:
            area_cm_avg[area] = np.nanmean(np.stack(fish_cms, axis=0), axis=0)
            area_cm_n_fish[area] = len(fish_cms)

    return area_cm_avg, area_cm_n_fish


# ---------------------------------------------------------------------------
# Valence classification loop (source: cells 49, 53)
# ---------------------------------------------------------------------------

def load_valence_neurons(pos_path: Path, neg_path: Path, fdr_alpha: float = FDR_ALPHA) -> set[str]:
    """Union of neuron IDs with significant (FDR < `fdr_alpha`) positive or negative valence correlation."""
    pos = pd.read_csv(pos_path)
    pos["neuron_id"] = pos["fish_id"] + pos["neuron_id"]

    neg = pd.read_csv(neg_path)
    neg["neuron_id"] = neg["fish_id"] + neg["neuron_id"]

    return set(pos[pos["p_fdr"] < fdr_alpha]["neuron_id"]).union(neg[neg["p_fdr"] < fdr_alpha]["neuron_id"])


def compute_valence_classification_results(
    neural_data_path: Path,
    nmlf_mask: np.ndarray,
    valence_neurons: set[str],
    fish_ids: tuple[tuple[str, str], ...] = FISH_IDS,
    selected_stimuli: tuple[str, ...] = SELECTED_STIMULI_VALENCE,
    valence_map: dict[str, int] = VALENCE_MAP,
    min_neurons: int = MIN_NEURONS_VALENCE,
    n_folds: int = N_FOLDS_VALENCE,
    n_permutations: int = N_PERMUTATIONS,
) -> dict:
    """Per-fish, per-area binary (attractive vs. aversive) template-matching classification + permutation test.

    Restricted to `valence_neurons` (from `load_valence_neurons`). Same
    expense/non-determinism caveats as `compute_odor_classification_results`.
    """
    all_valence_results: dict[str, dict] = {}

    for fb_id, hb_id in fish_ids:
        filtered_df = _load_fish_data_with_areas(
            neural_data_path, fb_id, hb_id, selected_stimuli, nmlf_mask, prefix_neuron_id=True,
        )
        valence_df = filtered_df[filtered_df["neuron_id"].isin(valence_neurons)]

        fish_results: dict[str, dict] = {}

        for area in valence_df["area"].unique():
            area_df = valence_df.query("area == @area")
            df_resp = area_df.pivot_table(index=["fish_id", "neuron_id"], columns=["stimulus", "trial_number"], values="resp")

            valence_labels = [valence_map[stim] for stim, _ in df_resp.columns]
            resp_clean = df_resp.dropna()

            if len(resp_clean) < min_neurons:
                continue

            X = resp_clean.values.T
            y = np.array(valence_labels)

            cv_result = template_matching_cross_validation(X, y, [0, 1], n_splits=n_folds)
            p_value = permutation_test(X, y, [0, 1], n_splits=n_folds, n_permutations=n_permutations)

            fish_results[area] = {
                "mean_score": cv_result["mean_score"],
                "p_value": p_value,
                "n_neurons": len(resp_clean),
            }

        all_valence_results[f"{fb_id}_{hb_id}"] = fish_results

    return all_valence_results


# ---------------------------------------------------------------------------
# Shared accuracy-scatter visualization (source: cells 27, 57 -- identical up
# to variable names, chance level, and title; genuinely duplicated within
# this one notebook, so merged into a single parametrized function)
# ---------------------------------------------------------------------------

def reorganize_results_by_area(all_results: dict) -> dict[str, list[tuple]]:
    """Invert a `{fish_key: {area: {...}}}` results dict to `{area: [(fish_key, score, p_value), ...]}`."""
    all_areas: dict[str, list[tuple]] = {}

    for fish_id, fish_results in all_results.items():
        for area_name, area_results in fish_results.items():
            # The odor loop's per-area dict key is "accuracy_score"; the
            # valence loop's is "mean_score" -- an inconsistency in the
            # notebook's own two loops (cells 21 vs 53), not unified there.
            score_key = "accuracy_score" if "accuracy_score" in area_results else "mean_score"
            all_areas.setdefault(area_name, []).append((fish_id, area_results[score_key], area_results["p_value"]))

    return all_areas


def plot_area_accuracy_scatter(
    combined_df: pd.DataFrame,
    all_areas: dict[str, list[tuple]],
    fish_ids: tuple[tuple[str, str], ...],
    chance_level: float,
    title: str,
    chance_label: str | None = None,
    figsize: tuple[float, float] = (12, 8),
):
    """Per-area scatter of individual fish accuracies (jittered, sized/alpha'd by significance) + mean +/- std.

    `chance_label` defaults to `f"Chance ({chance_level:.3f})"`, matching the source cells.
    """
    fig, ax = plt.subplots(figsize=figsize)

    sorted_areas = list(combined_df["area"])
    area_indices = {area: i for i, area in enumerate(sorted_areas)}

    fish_colors = plt.cm.tab10(np.linspace(0, 1, len(fish_ids)))
    simplified_fish_ids = {f"{fb}_{hb}": f"F{i + 1}" for i, (fb, hb) in enumerate(fish_ids)}
    fish_order = list(simplified_fish_ids.values())

    labels_handled = set()
    for area in sorted_areas:
        if area not in all_areas:
            continue

        fish_results = all_areas[area]
        jittered_y = jitter(np.full(len(fish_results), area_indices[area], dtype=float), amount=0.3)

        for y, (fish_id, score, p_value) in zip(jittered_y, fish_results):
            simple_id = simplified_fish_ids[fish_id]
            fish_idx = fish_order.index(simple_id)
            alpha = 1.0 if p_value < 0.05 else 0.3
            marker_size = 60 if p_value < 0.05 else 30

            label = simple_id if simple_id not in labels_handled else None
            ax.scatter(score, y, c=[fish_colors[fish_idx]], alpha=alpha, s=marker_size, label=label)
            labels_handled.add(simple_id)

    means = combined_df.set_index("area").loc[sorted_areas, "mean_accuracy"].to_numpy()
    stds = combined_df.set_index("area").loc[sorted_areas, "std_accuracy"].to_numpy()

    ax.errorbar(means, range(len(sorted_areas)), xerr=stds, fmt="none", ecolor="black", elinewidth=2, capsize=4, alpha=0.7)
    ax.scatter(means, range(len(sorted_areas)), c="black", s=100, marker="D", zorder=10, label="Mean")

    ax.axvline(chance_level, color="red", linestyle="--", linewidth=2, label=chance_label or f"Chance ({chance_level:.3f})")

    ax.set_yticks(range(len(sorted_areas)))
    ax.set_yticklabels(sorted_areas, fontsize=9)
    ax.set_xlabel("Classification Accuracy", fontsize=12, fontweight="bold")
    ax.set_ylabel("Brain Area", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Deterministic boxplot-vs-OB stats + plotting (source: cells 30, 33-34;
# these read a pre-existing accuracy CSV, not the live loop -- see module
# docstring)
# ---------------------------------------------------------------------------

def load_classification_accuracy_csv(csv_path: Path, area_shorthand: dict[str, str] = AREA_SHORTHAND) -> pd.DataFrame:
    """Load a per-fish/area classification-accuracy CSV and (re)map `area` from `area_full`.

    The `area` column is always recomputed from `area_full` here, matching
    the notebook's own unconditional remap -- any `area` values already in
    the file are discarded, not trusted.
    """
    df = pd.read_csv(csv_path)
    df["area"] = df["area_full"].map(area_shorthand)
    return df


def clean_and_average_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with no area, then average duplicate (area, fish_id) rows.

    Reduces to exactly `["area", "fish_id", "classification_accuracy"]` --
    all other columns are dropped by the groupby-mean, matching the
    notebook's own reassignment.
    """
    df = df.dropna(subset=["area"])
    return df.groupby(["area", "fish_id"], as_index=False)["classification_accuracy"].mean()


def compute_area_pvalues_paired_wilcoxon(
    df: pd.DataFrame, reference_area: str = REFERENCE_AREA, alternative: str = "greater",
) -> tuple[dict[str, float], pd.DataFrame]:
    """Per-area Wilcoxon signed-rank p-value vs. `reference_area`, paired by `fish_id` (inner join).

    Returns `(pvals, pval_df)`; `pval_df` also carries the Wilcoxon statistic
    and the number of paired fish per area.
    """
    reference_df = df[df["area"] == reference_area][["fish_id", "classification_accuracy"]].rename(
        columns={"classification_accuracy": "reference_accuracy"},
    )

    pvals, stats, n_pairs = {}, {}, {}

    for area in df["area"].unique():
        if area == reference_area:
            continue

        area_df = df[df["area"] == area][["fish_id", "classification_accuracy"]].rename(
            columns={"classification_accuracy": "area_accuracy"},
        )
        paired_df = reference_df.merge(area_df, on="fish_id", how="inner")
        n_pairs[area] = len(paired_df)

        if len(paired_df) > 0:
            try:
                stat, p = wilcoxon(paired_df["reference_accuracy"], paired_df["area_accuracy"], alternative=alternative)
                stats[area], pvals[area] = stat, p
            except ValueError:
                stats[area], pvals[area] = np.nan, np.nan
        else:
            stats[area], pvals[area] = np.nan, np.nan

    pval_df = pd.DataFrame({
        "area": list(pvals.keys()),
        "wilcoxon_stat": [stats[area] for area in pvals],
        "p_value": [pvals[area] for area in pvals],
        "n_pairs": [n_pairs[area] for area in pvals],
    })

    return pvals, pval_df


def plot_classification_accuracy_boxplot(
    data: pd.DataFrame,
    areas_to_plot: tuple[str, ...],
    area_order: tuple[str, ...],
    pvals: dict[str, float],
    reference_area: str | None = None,
    palette: dict[str, str] = HINDBRAIN_AREA_COLORS,
):
    """Classification-accuracy boxplot by brain area vs. `reference_area`.

    Thin wrapper around `jenkins_et_al.plotting.plot_area_comparison_boxplot`
    (unified 2026-08-04 with lifetime_sparseness's near-identical boxplot,
    at the user's request, after this function's fixed `(11, 6)` figsize made
    a 12-area (forebrain) call look disproportionately wide next to
    lifetime_sparseness's own 12-area boxplot -- see that function's
    docstring). `palette` defaults to the hindbrain palette this function
    originally hardcoded; pass `config.FOREBRAIN_AREA_COLORS` for the
    forebrain figure.
    """
    return plot_area_comparison_boxplot(
        data, "classification_accuracy", areas_to_plot, area_order, pvals, palette,
        reference_area=reference_area, ylabel="Classification Accuracy",
        ylim=(-0.03, 1.1), yticks=list(np.arange(0, 1.1, 0.2)),
    )
