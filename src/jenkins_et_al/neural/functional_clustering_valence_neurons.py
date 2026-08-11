"""Functional clustering of valence-correlated neurons: k-means clusters of
neurons with a significant positive or negative correlation to valence,
based on their normalized response profile across 6 high-concentration
stimuli, visualized via PCA and a spatial brain projection.

Ported from notebooks/neural/functional_clustering_valence_neurons.ipynb;
logic unchanged, only moved into a module and split into
loading/processing/plotting functions per the /tdd REFACTOR checklist.

**Three analysis modes** select which valence-correlated neurons to cluster
(the notebook's own `MODE` switch, cell 33): `"combined"` (positive +
negative together -- the notebook's own default), `"positive_only"`,
`"negative_only"`.

**Confirmed dead code, not reproduced**: the notebook's own "Filtered for 6
stimuli" cell computes `conc_df = merged_df[merged_df['stimulus'].isin(STIMULI)]`
and prints its length, but nothing downstream reads it -- `averaged_responses`
groups `merged_df` directly, not `conc_df`. Any stimulus values outside
`STIMULI` present in `merged_df` only ever end up as extra pivot-table
columns that are never selected (`pivot_df[STIMULI]`) or checked for NaN
(`dropna(subset=STIMULI)`), so this has no effect on any output. Not ported.

**Confirmed bug, fixed in the port -- the notebook cannot run to completion
as saved**: `average_and_pivot_responses`'s `pivot_table` call (cell 47) pivots
on an ordered-`Categorical` `stimulus` column with pandas' default
`observed=False`. Because `coords` (also part of the pivot's `index=`) is
effectively unique per neuron, `observed=False` tries to materialize the full
cross-product of every index column's distinct values x every stimulus
category -- tens of thousands of neurons x dozens of areas x 6 stimuli
explodes past available memory and the kernel is killed before the cell
finishes (confirmed: reproduced the OOM with a structurally-identical
synthetic dataset at the same row count, independent of any real data
content). Bug-for-bug preservation isn't possible here (there is no
completed run to preserve). This port passes `observed=True` instead.
**Proven not to change results**: every phantom row `observed=False` would
add beyond what `observed=True` produces has no data for *any* stimulus (an
invented index-column combination that never co-occurred with a real
response value), so it is all-NaN across every `stimuli` column and is
always removed by the very next line, `dropna(subset=stimuli)` -- verified
this equivalence directly (`observed=True` vs. `observed=False`, post-dropna)
on a small synthetic dataset small enough for `observed=False` to actually
complete. This is a pure memory fix with no effect on any output.

**Second confirmed bug, fixed in the port -- also unreachable in the
never-completed notebook**: the "Extract coordinates" cell
(`coords = np.vstack(scaled_df['coords'].values)`) never parses `coords`
out of its JSON-string form (confirmed: `data_resp['coords']` is `str`,
e.g. `"[251.66357421875, 217.15609741210938, 354.30865...]"`, immediately
after `pd.read_hdf`). `np.vstack` over a list of strings does not raise --
`np.atleast_2d` treats each whole string as one scalar element, silently
producing an `(N, 1)` array of strings instead of `(N, 3)` floats, so
`plot_brain_spatial` would silently plot garbage (or error confusingly much
later) rather than fail loudly at the source. `lifetime_sparseness.ipynb`
hits the identical situation for the same kind of field
(`coords_serialised`) and fixes it with `json.loads`; `parse_coordinates`
below does the same here. Every other use of `coords` in this pipeline (the
`merge`/`correlation`-lookup join keys) stays string-keyed, matching the
notebook exactly -- only the final float-array extraction for plotting
needed this fix.

****Third confirmed bug, fixed in the port -- user-reported after visual
inspection, not from a failed run**: `plot_brain_spatial`'s sagittal-panel
scatter plotted `(coords[:, 0], coords[:, 1])` (z, y). `proj_sagittal =
np.nanmean(ref_brain, axis=2)` has shape `(z, y)` in `(row, col)` order, so
the sagittal image's x-axis is `y` and y-axis is `z` -- the scatter needs
`(coords[:, 1], coords[:, 0])` instead. Confirmed with real data: `z` ranges
45-345 (image height 359, so it barely fits when on the y-axis) but the
notebook's own version puts it on the x-axis (image width 974, so most of
the axis is empty); `y` ranges 66-857 (image width 974) but gets put on the
y-axis (height 359), so most points fall outside the visible image entirely.
The already-verified `jenkins_et_al.plotting.plot_neuron_scatter_on_brain`
(pixel-matched against `lifetime_sparseness.ipynb`, see PORTING_PLAN.md #6)
uses the correct `(coords[:, 1], coords[:, 0])` order for its own sagittal
panel -- this notebook's `plot_brain_spatial` just has the two swapped. The
horizontal panel is unaffected (already matches that convention). Unlike the
two bugs above, this one doesn't block the notebook from completing a run --
it only produces a visibly wrong overlay -- so bug-for-bug preservation was
possible here; fixed instead at the user's explicit request after reviewing
the rendered plots.

`CLUSTER_COLORS` here is int-cluster-id -> color name, reassigned per run
by `assign_cluster_colors_by_response`** -- a different meaning from the
same-named module-level constant in
`functional_clustering_concentration_response.py` (there: a fixed
response-type-name -> hex color, hand-derived once for one odorant). Do not
merge; PORTING_PLAN.md flags this explicitly. The module-level
`CLUSTER_COLORS` below is only the notebook's own hard-coded fallback
default (cell 33) -- the real, per-run palette comes back from
`assign_cluster_colors_by_response`.

The horizontal+sagittal brain-projection scatter (`plot_brain_spatial`) is a
categorical variant (one flat color per cluster ID, not a value+colormap) of
the same underlying projection as
`functional_clustering_concentration_response.plot_response_conditions_2view`
and `jenkins_et_al.plotting.plot_neuron_scatter_on_brain` -- kept as its own
function here rather than unified, consistent with those two modules' own
docstrings, which already earmark this notebook as a third site of the same
categorical pattern. Unify all three once
`notebooks/neural/plot_regression_spatial_maps.ipynb` (the last, simplest
consumer of this same pattern) is also independently ported and verified.

`ATTRACTIVE`/`AVERSIVE` are this notebook's own 3-stimulus valence-preference
groups (a subset of the 6-stimulus `STIMULI` list) -- not the same as
`jenkins_et_al.config.POSITIVE_STIMULI`/`NEGATIVE_STIMULI`, which cover all
three concentration tiers including `ph4.5` (absent from `STIMULI` here).
Kept notebook-local rather than merged, since the sets genuinely differ in
scope.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from skimage import io
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

#: Source: cell 33.
STIMULI = ["fex_1", "kw", "pro_2.5mm", "ade", "cad_2.5mm", "qui_2.5mm"]
ATTRACTIVE = ["fex_1", "kw", "pro_2.5mm"]
AVERSIVE = ["ade", "cad_2.5mm", "qui_2.5mm"]

#: Source: cell 33.
K_RANGE = [2, 3, 4, 5]
P_VALUE_THRESHOLD = 0.05
RANDOM_STATE = 42

#: Source: cell 33. Notebook's own hard-coded fallback -- see module
#: docstring for why this differs in meaning from #10's `CLUSTER_COLORS`.
CLUSTER_COLORS = {0: "magenta", 1: "green", 2: "pink", 3: "limegreen", 4: "olive"}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_data(
    pos_valence_path: str | Path,
    neg_valence_path: str | Path,
    neural_data_path: str | Path,
    ref_brain_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Load valence-correlation tables, the full response table, and the reference brain.

    Adds a `valence` label column (1 = positive, 0 = negative) to each
    correlation table, and renames the response table's `coords_serialised`
    column to `coords` to match the correlation tables' column name.

    Source: cells 43, 45, 90 (data loads, gathered here per this port's
    loading/processing/plotting split -- see module docstring).
    """
    pos_regress = pd.read_csv(pos_valence_path)
    neg_regress = pd.read_csv(neg_valence_path)
    pos_regress = pos_regress.copy()
    neg_regress = neg_regress.copy()
    pos_regress["valence"] = 1
    neg_regress["valence"] = 0

    data_resp = pd.read_hdf(neural_data_path)
    data_resp = data_resp.rename(columns={"coords_serialised": "coords"})

    ref_brain = io.imread(ref_brain_path)

    return pos_regress, neg_regress, data_resp, ref_brain


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def select_neurons_by_mode(
    pos_regress: pd.DataFrame,
    neg_regress: pd.DataFrame,
    mode: str = "combined",
    p_value_threshold: float = P_VALUE_THRESHOLD,
) -> tuple[pd.DataFrame, str]:
    """Select valence-correlated neurons by `p_fdr < p_value_threshold`, per `mode`.

    `mode`: `"positive_only"`, `"negative_only"`, or `"combined"` (default --
    concatenates both, positive rows first).

    Source: cell 44.
    """
    if mode == "positive_only":
        valence = pos_regress[pos_regress["p_fdr"] < p_value_threshold].copy()
        mode_description = "Positive valence neurons only"
    elif mode == "negative_only":
        valence = neg_regress[neg_regress["p_fdr"] < p_value_threshold].copy()
        mode_description = "Negative valence neurons only"
    else:
        valence = pd.concat([
            pos_regress[pos_regress["p_fdr"] < p_value_threshold],
            neg_regress[neg_regress["p_fdr"] < p_value_threshold],
        ], ignore_index=True)
        mode_description = "Combined positive + negative valence neurons"

    return valence, mode_description


def merge_valence_with_responses(valence: pd.DataFrame, data_resp: pd.DataFrame) -> pd.DataFrame:
    """Inner-join selected valence neurons with their per-trial response records, on `coords`.

    Source: cell 45.
    """
    stim_columns = valence[["correlation", "coords", "p_fdr", "valence"]]
    return pd.merge(stim_columns, data_resp, on=["coords"], how="inner")


def average_and_pivot_responses(merged_df: pd.DataFrame, stimuli: list[str] = STIMULI) -> pd.DataFrame:
    """Average repeated trials, then pivot to one row per neuron, one column per stimulus.

    Drops neurons with any missing response among `stimuli`. Pivots with
    `observed=True` -- see module docstring for why (the notebook's own
    `observed=False` default cannot complete; proven equivalent post-dropna).

    Source: cell 47 ("Average and Pivot Responses").
    """
    averaged_responses = (
        merged_df
        .groupby(["fish_id", "neuron_id", "stimulus", "coords", "area", "valence"])["resp"]
        .mean()
        .reset_index()
    )

    averaged_responses["stimulus"] = pd.Categorical(
        averaged_responses["stimulus"], categories=stimuli, ordered=True,
    )
    averaged_responses = averaged_responses.sort_values(["fish_id", "neuron_id", "stimulus"])

    pivot_df = averaged_responses.pivot_table(
        index=["fish_id", "neuron_id", "coords", "area", "valence"],
        columns="stimulus", values="resp", observed=True,
    ).reset_index()

    return pivot_df.dropna(subset=stimuli)


def normalize_responses(pivot_df: pd.DataFrame, stimuli: list[str] = STIMULI) -> tuple[pd.DataFrame, np.ndarray]:
    """MinMax-scale each neuron's response vector to [0, 1] (row-wise, per neuron).

    Returns `(scaled_df, scaled_responses)`: `scaled_df` has the same
    metadata columns as `pivot_df` (`fish_id`, `neuron_id`, `coords`, `area`,
    `valence`) plus the scaled `stimuli` columns; `scaled_responses` is the
    raw `(n_neurons, n_stimuli)` array used directly by clustering/PCA.

    Source: cell 48 ("Normalize Responses").
    """
    pivot_df = pivot_df.reset_index(drop=True)
    responses = pivot_df[stimuli].values

    scaled_responses = np.array([
        MinMaxScaler().fit_transform(r.reshape(-1, 1)).flatten() for r in responses
    ])

    metadata_df = pivot_df[["fish_id", "neuron_id", "coords", "area", "valence"]].reset_index(drop=True)
    scaled_df = pd.concat(
        [metadata_df, pd.DataFrame(scaled_responses, columns=stimuli)], axis=1,
    )

    return scaled_df, scaled_responses


def run_kmeans_clustering(
    data: np.ndarray, k_range: list[int] = K_RANGE, random_state: int = RANDOM_STATE,
) -> dict[int, dict[str, object]]:
    """K-means cluster `data` for each `k` in `k_range`; silhouette score per k (0.0 when k=1).

    Source: cell 34 (`run_kmeans_clustering`).
    """
    results = {}
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        labels = kmeans.fit_predict(data)
        sil_score = silhouette_score(data, labels) if k > 1 else 0.0
        results[k] = {"labels": labels, "silhouette": sil_score, "kmeans": kmeans}
    return results


def find_optimal_k(clustering_results: dict[int, dict[str, object]]) -> int:
    """The k with the highest silhouette score.

    Source: cell 50 ("Find optimal k").
    """
    return max(clustering_results.keys(), key=lambda k: clustering_results[k]["silhouette"])


def compute_pca(
    scaled_responses: np.ndarray, random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """2-component PCA of `scaled_responses`; returns `(pca_result, explained_variance_pct)`.

    Source: cell 51 ("PCA for Visualization").
    """
    pca = PCA(n_components=2, random_state=random_state)
    pca_result = pca.fit_transform(scaled_responses)
    explained_var = pca.explained_variance_ratio_ * 100
    return pca_result, explained_var


def add_pca_and_correlation(scaled_df: pd.DataFrame, pca_result: np.ndarray, valence: pd.DataFrame) -> pd.DataFrame:
    """Attach `PC1`/`PC2` and, if available, each neuron's `correlation` (looked up from `valence` by `coords`).

    Source: cell 51.
    """
    scaled_df = scaled_df.copy()
    scaled_df["PC1"] = pca_result[:, 0]
    scaled_df["PC2"] = pca_result[:, 1]

    if "correlation" in valence.columns:
        corr_map = valence.set_index("coords")["correlation"].to_dict()
        scaled_df["correlation"] = scaled_df["coords"].apply(lambda c: corr_map.get(c, np.nan))

    return scaled_df


def add_cluster_labels(scaled_df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Attach the optimal-k cluster assignment as a `cluster` column.

    Source: cell 55 ("Add Cluster Labels to Data").
    """
    scaled_df = scaled_df.copy()
    scaled_df["cluster"] = labels
    return scaled_df


def characterize_clusters(df: pd.DataFrame, cluster_labels: np.ndarray, stimuli: list[str] = STIMULI) -> pd.DataFrame:
    """Per-cluster summary: size, valence composition, mean +/- SEM per-stimulus response, top areas, per-fish counts.

    `sem_{stim}` (sample standard error of the mean, ddof=1) is `NaN` for a
    cluster with a single neuron -- not present in the original notebook (see
    module docstring), added to support `plot_cluster_response_profiles`'s
    error bars.

    Source: cell 35 (`characterize_clusters`).
    """
    df = df.copy()
    df["cluster"] = cluster_labels

    cluster_stats = []
    for cluster_id in sorted(df["cluster"].unique()):
        cluster_neurons = df[df["cluster"] == cluster_id]
        n_neurons = len(cluster_neurons)
        mean_responses = cluster_neurons[stimuli].mean()
        sem_responses = cluster_neurons[stimuli].sem()

        if "valence" in df.columns:
            n_positive = (cluster_neurons["valence"] == 1).sum()
            n_negative = (cluster_neurons["valence"] == 0).sum()
            pct_positive = n_positive / n_neurons * 100 if n_neurons > 0 else 0
        else:
            n_positive = n_negative = pct_positive = np.nan

        top_areas = cluster_neurons["area"].value_counts().head(3).to_dict()
        fish_counts = cluster_neurons["fish_id"].value_counts().to_dict()

        stats = {
            "cluster": cluster_id,
            "n_neurons": n_neurons,
            "pct_positive": pct_positive,
            "n_positive": n_positive,
            "n_negative": n_negative,
            "top_areas": top_areas,
            "fish_distribution": fish_counts,
        }
        for stim in stimuli:
            stats[f"mean_{stim}"] = mean_responses[stim]
            stats[f"sem_{stim}"] = sem_responses[stim]
        cluster_stats.append(stats)

    return pd.DataFrame(cluster_stats)


def assign_cluster_colors_by_response(
    cluster_summary: pd.DataFrame,
    stimuli: list[str] = STIMULI,
    attractive: list[str] = ATTRACTIVE,
    aversive: list[str] = AVERSIVE,
) -> dict[int, str]:
    """Assign each cluster a color from its response profile: green (attractive-preferring),
    pink/magenta (aversive-preferring), or orange/yellow (low-response or balanced).

    Source: cell 39 (`assign_cluster_colors_by_response`).
    """
    color_map = {}
    attractive_colors = ["limegreen", "forestgreen", "green", "darkgreen", "seagreen"]
    aversive_colors = ["magenta", "deeppink", "hotpink", "orchid", "mediumvioletred"]
    mixed_colors = ["orange", "gold", "darkorange", "coral", "goldenrod"]

    for idx, row in cluster_summary.iterrows():
        cluster_id = int(row["cluster"])

        attr_responses = [row[f"mean_{s}"] for s in attractive if f"mean_{s}" in row]
        aver_responses = [row[f"mean_{s}"] for s in aversive if f"mean_{s}" in row]
        attr_mean = np.mean(attr_responses) if attr_responses else 0
        aver_mean = np.mean(aver_responses) if aver_responses else 0

        all_responses = [row[f"mean_{s}"] for s in stimuli if f"mean_{s}" in row]
        max_response = max(all_responses) if all_responses else 0

        if max_response < 0.3:
            color = mixed_colors[idx % len(mixed_colors)]
        elif abs(attr_mean - aver_mean) < 0.15:
            color = mixed_colors[idx % len(mixed_colors)]
        elif attr_mean > aver_mean:
            color = attractive_colors[idx % len(attractive_colors)]
        else:
            color = aversive_colors[idx % len(aversive_colors)]

        color_map[cluster_id] = color

    return color_map


def describe_cluster_colors(
    cluster_summary: pd.DataFrame,
    color_map: dict[int, str],
    stimuli: list[str] = STIMULI,
    attractive: list[str] = ATTRACTIVE,
    aversive: list[str] = AVERSIVE,
) -> str:
    """Build the human-readable rationale for each cluster's assigned color.

    Returns the report as a string (the notebook's own version printed it
    directly -- same text, just returned here as well so callers/tests can
    inspect it without capturing stdout).

    Source: cell 40 (`describe_cluster_colors`).
    """
    lines = ["\n" + "=" * 80, "CLUSTER COLOR ASSIGNMENTS & RATIONALE", "=" * 80]

    for _, row in cluster_summary.iterrows():
        cluster_id = int(row["cluster"])
        color = color_map.get(cluster_id, "gray")

        attr_responses = [row[f"mean_{s}"] for s in attractive if f"mean_{s}" in row]
        aver_responses = [row[f"mean_{s}"] for s in aversive if f"mean_{s}" in row]
        attr_mean = np.mean(attr_responses) if attr_responses else 0
        aver_mean = np.mean(aver_responses) if aver_responses else 0

        mean_responses = {s: row[f"mean_{s}"] for s in stimuli if f"mean_{s}" in row}
        if mean_responses:
            max_stim = max(mean_responses, key=mean_responses.get)
            max_val = mean_responses[max_stim]
        else:
            max_stim, max_val = "N/A", 0

        lines.append(f"\nCluster {cluster_id}: {color.upper()}")
        lines.append(f"  Dominant stimulus: {max_stim} (response: {max_val:.3f})")
        lines.append(f"  Attractive mean: {attr_mean:.3f}")
        lines.append(f"  Aversive mean: {aver_mean:.3f}")

        if abs(attr_mean - aver_mean) < 0.15:
            preference = "Balanced/Mixed"
        elif attr_mean > aver_mean:
            preference = f"Attractive ({attr_mean - aver_mean:.3f} higher)"
        else:
            preference = f"Aversive ({aver_mean - attr_mean:.3f} higher)"
        lines.append(f"  Preference: {preference}")

    lines.append("\n" + "=" * 80)
    report = "\n".join(lines)
    print(report)
    return report


def compute_fish_type(scaled_df: pd.DataFrame) -> pd.DataFrame:
    """Attach a `fish_type` column: `"Forebrain"` if `fish_id` contains `"_fb"`, else `"Hindbrain"`.

    Source: cell 60 ("Cluster Composition (Forebrain vs Hindbrain)").
    """
    scaled_df = scaled_df.copy()
    scaled_df["fish_type"] = scaled_df["fish_id"].apply(lambda x: "Forebrain" if "_fb" in x else "Hindbrain")
    return scaled_df


def compute_cluster_composition(scaled_df: pd.DataFrame) -> pd.DataFrame:
    """Fraction of each fish type's neurons falling in each cluster, as a `fish_type` x `cluster` table.

    `scaled_df` must already have a `fish_type` column (see `compute_fish_type`).

    Source: cell 60.
    """
    comp_counts = scaled_df.groupby(["fish_type", "cluster"]).size().reset_index(name="count")
    comp_totals = comp_counts.groupby("fish_type")["count"].transform("sum")
    comp_counts["fraction"] = comp_counts["count"] / comp_totals
    return comp_counts.pivot(index="fish_type", columns="cluster", values="fraction").fillna(0)


def parse_coordinates(scaled_df: pd.DataFrame) -> np.ndarray:
    """Parse the `coords` column's JSON-string values into an `(n_neurons, 3)` float array.

    See module docstring ("Second confirmed bug") for why this parsing step
    is required -- the notebook's own cell skips it.

    Source: cell 92 ("Extract coordinates"), fixed.
    """
    return np.vstack([json.loads(c) for c in scaled_df["coords"].values])


def save_results(
    scaled_df: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    output_dir: str | Path,
    mode: str,
    optimal_k: int,
) -> tuple[Path, Path]:
    """Write `scaled_df` and `cluster_summary` to `{output_dir}/valence_clustering_{mode}_k{optimal_k}.csv`
    and `{output_dir}/valence_cluster_summary_{mode}_k{optimal_k}.csv`. Returns both paths.

    Source: cell 67 ("Save Results").
    """
    output_dir = Path(output_dir)
    output_file = output_dir / f"valence_clustering_{mode}_k{optimal_k}.csv"
    summary_file = output_dir / f"valence_cluster_summary_{mode}_k{optimal_k}.csv"

    scaled_df.to_csv(output_file, index=False)
    cluster_summary.to_csv(summary_file, index=False)

    return output_file, summary_file


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_pca_clusters(
    pca_result: np.ndarray,
    labels: np.ndarray,
    k: int,
    silhouette: float,
    explained_var: np.ndarray,
    cluster_colors: dict[int, str] = CLUSTER_COLORS,
    title: str = "",
) -> plt.Figure:
    """PCA scatter colored by cluster assignment, for one k.

    Source: cell 36 (`plot_pca_clusters`).
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = [cluster_colors.get(label, "gray") for label in labels]
    ax.scatter(pca_result[:, 0], pca_result[:, 1], c=colors, s=30, alpha=0.7)

    ax.set_xlabel(f"PC1 ({explained_var[0]:.1f}%)", fontsize=14)
    ax.set_ylabel(f"PC2 ({explained_var[1]:.1f}%)", fontsize=14)
    ax.set_title(f"{title}\nk = {k}, Silhouette = {silhouette:.3f}", fontsize=16)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    return fig


def plot_pca_all_k(
    clustering_results: dict[int, dict[str, object]],
    pca_result: np.ndarray,
    explained_var: np.ndarray,
    k_range: list[int] = K_RANGE,
    cluster_colors: dict[int, str] = CLUSTER_COLORS,
    mode: str = "",
) -> plt.Figure:
    """PCA scatter colored by cluster, one panel per k in `k_range`.

    Source: cell 56 ("Visualize Clustering Results").
    """
    fig, axes = plt.subplots(1, len(k_range), figsize=(6 * len(k_range), 5))
    if len(k_range) == 1:
        axes = [axes]

    for i, k in enumerate(k_range):
        ax = axes[i]
        result = clustering_results[k]
        colors = [cluster_colors.get(label, "gray") for label in result["labels"]]
        ax.scatter(pca_result[:, 0], pca_result[:, 1], c=colors, s=30, alpha=0.7)
        ax.set_xlabel(f"PC1 ({explained_var[0]:.1f}%)", fontsize=14)
        ax.set_ylabel(f"PC2 ({explained_var[1]:.1f}%)", fontsize=14)
        ax.set_title(f"k = {k}\nSilhouette = {result['silhouette']:.3f}", fontsize=16)
        ax.tick_params(labelsize=12)

    plt.suptitle(f"Clustering Results - MODE: {mode}", fontsize=18, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_pca_by_valence(pca_result: np.ndarray, df: pd.DataFrame, explained_var: np.ndarray) -> plt.Figure | None:
    """PCA scatter colored by valence sign and correlation strength.

    Returns `None` (and does not plot) if `df` lacks `valence`/`correlation`
    columns -- matching the notebook's own guard.

    Source: cell 37 (`plot_pca_by_valence`).
    """
    if "valence" not in df.columns or "correlation" not in df.columns:
        return None

    fig, ax = plt.subplots(figsize=(8, 6))

    cmap_pos = LinearSegmentedColormap.from_list("pos", ["lightgreen", "darkgreen"])
    cmap_neg = LinearSegmentedColormap.from_list("neg", ["pink", "magenta"])
    norm = Normalize(vmin=0, vmax=0.8)

    pos_mask = df["valence"] == 1
    neg_mask = df["valence"] == 0

    sc1 = ax.scatter(
        pca_result[pos_mask, 0], pca_result[pos_mask, 1],
        c=df.loc[pos_mask, "correlation"], cmap=cmap_pos, norm=norm,
        s=40, alpha=0.9, label="Positive valence",
    )
    sc2 = ax.scatter(
        pca_result[neg_mask, 0], pca_result[neg_mask, 1],
        c=df.loc[neg_mask, "correlation"], cmap=cmap_neg, norm=norm,
        s=40, alpha=0.9, label="Negative valence",
    )

    ax.set_xlabel(f"PC1 ({explained_var[0]:.1f}%)", fontsize=14)
    ax.set_ylabel(f"PC2 ({explained_var[1]:.1f}%)", fontsize=14)
    ax.set_title("PCA: Colored by Valence & Correlation", fontsize=16)
    ax.legend(fontsize=12)

    fig.colorbar(sc1, ax=ax, label="Correlation (positive)")
    fig.colorbar(sc2, ax=ax, label="Correlation (negative)")

    plt.tight_layout()
    return fig


def plot_cluster_heatmap(cluster_summary: pd.DataFrame, stimuli: list[str] = STIMULI) -> plt.Figure:
    """Heatmap of each cluster's mean per-stimulus response.

    Source: cell 38 (`plot_cluster_heatmap`).
    """
    mean_cols = [f"mean_{s}" for s in stimuli]
    heatmap_data = cluster_summary[mean_cols].values

    fig, ax = plt.subplots(figsize=(8, len(cluster_summary) * 0.8))
    im = ax.imshow(heatmap_data, cmap="RdBu_r", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(stimuli)))
    ax.set_yticks(range(len(cluster_summary)))
    ax.set_xticklabels(stimuli, rotation=45, ha="right", fontsize=12)
    ax.set_yticklabels([f"Cluster {c}" for c in cluster_summary["cluster"]], fontsize=12)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Normalized Response", fontsize=12)

    ax.set_title("Mean Response Profile per Cluster", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_cluster_response_profiles(
    cluster_summary: pd.DataFrame,
    color_map: dict[int, str],
    stimuli: list[str] = STIMULI,
    attractive: list[str] = ATTRACTIVE,
) -> plt.Figure:
    """Line plot of each cluster's mean +/- SEM response profile across stimuli.

    SEM error bars are not in the original notebook (see module docstring) --
    added here, sourced from `characterize_clusters`'s `sem_{stim}` columns.

    Source: cell 41 (`plot_cluster_response_profiles`).
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for _, row in cluster_summary.iterrows():
        cluster_id = int(row["cluster"])
        color = color_map.get(cluster_id, "gray")
        responses = [row[f"mean_{s}"] for s in stimuli]
        sems = [row[f"sem_{s}"] for s in stimuli]
        ax.errorbar(
            stimuli, responses, yerr=sems, color=color, marker="o", linewidth=2.5, markersize=5,
            capsize=5, elinewidth=1.8, capthick=1.8, ecolor="black",
            label=f'Cluster {cluster_id} (n={int(row["n_neurons"])})', alpha=0.8,
        )

    ax.set_xlabel("Stimulus", fontsize=14, fontweight="bold")
    ax.set_ylabel("Normalized Response", fontsize=14, fontweight="bold")
    ax.set_title("Cluster Response Profiles (Colored by Preference)", fontsize=16, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(fontsize=11, loc="best")

    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(fontsize=12)

    if len(attractive) > 0:
        ax.axvline(x=len(attractive) - 0.5, color="gray", linestyle="--", alpha=0.5, linewidth=1.5)
        ax.text(len(attractive) / 2, 1.02, "Attractive", ha="center", fontsize=11, style="italic")
        ax.text(
            len(attractive) + (len(stimuli) - len(attractive)) / 2, 1.02, "Aversive",
            ha="center", fontsize=11, style="italic",
        )

    plt.tight_layout()
    return fig


def plot_cluster_composition(
    comp_pivot: pd.DataFrame, cluster_colors: dict[int, str] = CLUSTER_COLORS, mode: str = "",
) -> plt.Figure:
    """Stacked bar chart of cluster fractions per fish type (forebrain/hindbrain).

    `comp_pivot` is `compute_cluster_composition`'s output.

    Source: cell 60.
    """
    cluster_order = sorted(comp_pivot.columns)
    colors = [cluster_colors.get(c, "gray") for c in cluster_order]

    fig, ax = plt.subplots(figsize=(8, 6))
    comp_pivot[cluster_order].plot(kind="bar", stacked=True, color=colors, ax=ax)

    ax.set_ylabel("Fraction of Neurons", fontsize=14)
    ax.set_xlabel("")
    ax.set_xticklabels(["Forebrain", "Hindbrain"], rotation=0, fontsize=14)
    ax.set_title(f"Cluster Composition by Brain Region - MODE: {mode}", fontsize=16, fontweight="bold")
    ax.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    return fig


def plot_brain_spatial(
    ref_brain: np.ndarray,
    coords: np.ndarray,
    labels: np.ndarray,
    cluster_colors: dict[int, str] = CLUSTER_COLORS,
    title: str = "",
) -> plt.Figure:
    """Dorsal + sagittal brain projections, neurons colored by cluster (categorical, not colormapped).

    See module docstring for why this isn't unified with
    `jenkins_et_al.plotting.plot_neuron_scatter_on_brain` yet, and for the
    sagittal-panel axis-swap bug fixed here (third confirmed bug).

    Source: cell 42 (`plot_brain_spatial`), fixed.
    """
    fig = plt.figure(figsize=(10, 12))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
    gs.update(hspace=-0.6)

    ax_horizontal = fig.add_subplot(gs[0])
    ax_sagittal = fig.add_subplot(gs[1])

    unique_labels = np.unique(labels)

    proj_horizontal = np.nanmean(ref_brain, axis=0)
    proj_horizontal = np.rot90(proj_horizontal, k=1)
    ax_horizontal.imshow(proj_horizontal, cmap="gray_r", alpha=0.5, origin="lower")

    for label in unique_labels:
        color = cluster_colors.get(label, "gray")
        label_coords = coords[labels == label]
        ax_horizontal.scatter(
            label_coords[:, 1], proj_horizontal.shape[0] - label_coords[:, 2],
            c=color, s=20, alpha=0.9, label=f"Cluster {label}",
        )

    ax_horizontal.axis("off")
    ax_horizontal.set_aspect("equal")
    ax_horizontal.legend(loc="upper right", fontsize=10)

    proj_sagittal = np.nanmean(ref_brain, axis=2)
    ax_sagittal.imshow(proj_sagittal, cmap="gray_r", alpha=0.5, origin="lower")

    for label in unique_labels:
        color = cluster_colors.get(label, "gray")
        label_coords = coords[labels == label]
        ax_sagittal.scatter(label_coords[:, 1], label_coords[:, 0], c=color, s=20, alpha=0.9)

    ax_sagittal.axis("off")
    ax_sagittal.set_aspect("equal")

    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

    plt.tight_layout()
    return fig
