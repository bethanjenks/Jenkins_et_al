"""Concentration-response clustering: k-means clusters of neurons by how
their response scales across an odorant's three concentrations, hand-labeled
into four response types (Steep-Ramping, High-threshold, Shallow-Ramping,
Low-threshold).

Ported from notebooks/neural/functional_clustering_concentration_response.ipynb;
logic unchanged, only moved into a module and split into
loading/processing/plotting functions per the /tdd REFACTOR checklist.

**This analysis has a required human decision point** the notebook itself
calls out (its own markdown: "MANUAL STEP: Assign cluster IDs to response
types") -- after `perform_clustering`, a person must look at the cluster-ID
curves and decide which cluster ID maps to which of the four named response
types before `assign_cluster_names` can run. This port preserves that gate
rather than guessing a mapping: `show_cluster_id_curves` opens that plot in
the default web browser (a stand-in for Jupyter's inline display when
running this as a script, via `jenkins_et_al.plotting.show_in_browser`);
`show_silhouette_analysis` does the same for the notebook's optional
silhouette-score diagnostic (its own source cell is commented out with
"Uncomment to run" -- opt-in there, opt-in here too).

**Three bugs found in the notebook while porting, all now fixed by the user
directly in the notebook (not by this port -- originals are read-only here)**:

1. A missing comma between the `'proline'` and `'food'` entries of
   `ODORANT_CONCENTRATIONS` (cell 3) made the dict literal a syntax error --
   fixed in commit `1f20b4b`.
2. `CURRENT_ODORANT` was set to `'cadaverine'`, whose data produces 5
   k-means clusters, but the hand-written `cluster_name_map` (cell 27) only
   maps 4 -- `assign_cluster_names`'s own validation check raises `ValueError`
   for the unmapped 5th cluster. Cell 27's own comment ("EXAMPLE from
   original quinine analysis") plus the cluster_name_map's content (unchanged
   since) indicates quinine is what this mapping was actually derived from;
   commit `1f20b4b` switched `CURRENT_ODORANT` back to `'quinine'`, whose
   data does produce exactly 4 clusters, resolving the mismatch.
3. **Found during this port, not previously flagged**: the notebook defined
   `plot_response_conditions_2view` twice (cell 22 correct; cell 23 shadowing
   it, since a later definition wins). Cell 23's *entire* source had no
   newline characters at all -- confirmed present already in the frozen
   baseline `2d19f3a`, not introduced by the recent edits. Since Python
   comments run to the next newline, and this cell had none after its first
   `#` inside the function body, everything following that `#` -- the real
   downsampling logic, all plotting calls, `return fig` -- was silently
   swallowed into one comment. The function still defined and ran without
   error; it just did nothing and returned `None`. Confirmed by executing
   that exact cell source directly (both current and at the frozen
   baseline) and calling the resulting function: both return `None`. Net
   effect: a fresh top-to-bottom run of the unfixed notebook produced no
   brain-projection figure at all, silently. The user deleted the broken
   duplicate cell directly in the notebook, leaving cell 22's working
   definition live; this port's `plot_response_conditions_2view` is that
   surviving definition, confirmed to actually render the figure (golden
   capture: 22526 neurons plotted, cluster sizes `{0: 16407, 1: 403,
   2: 2932, 3: 2784}`).

A second squished cell (a duplicate of cell 4's "ANALYSIS PARAMETERS" block,
same missing-newline symptom) is inert, not a bug: since it starts with `#`
and has no newline anywhere, the entire cell is one comment, so it never
overwrites cell 4's real constant values. Left un-ported (dead weight, not
executable logic).

The horizontal+sagittal brain-projection pattern here is a *categorical*
variant (discrete `response_condition` labels + per-condition legend, no
colorbar) of the *continuous* value+colormap pattern in
`jenkins_et_al.plotting.plot_neuron_scatter_on_brain` -- kept as its own
function here rather than unified, per that function's own docstring, which
also earmarked `functional_clustering_valence_neurons.ipynb` (#11, confirmed
a matching categorical variant, still not unified) as a second site.
**Correction**: this note previously also named `plot_regression_spatial_maps.ipynb`
(#12) here -- wrong, checked while porting it: that notebook's brain-scatter
is a fourth, different pattern (continuous colormap, but a different one per
valence group), not this categorical one. Not a unification candidate for
this function.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import io
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from jenkins_et_al.plotting import show_in_browser

#: Source: cell 3.
ODORANT_CONCENTRATIONS = {
    "quinine": {
        "stimuli": ["qui_25um", "qui_250um", "qui_2.5mm"],
        "labels": ["Qui 25µM", "Qui 250µM", "Qui 2.5mM"],
        "short_name": "qui",
    },
    "cadaverine": {
        "stimuli": ["cad_25um", "cad_250um", "cad_2.5mm"],
        "labels": ["Cad 25µM", "Cad 250µM", "Cad 2.5mM"],
        "short_name": "cad",
    },
    "proline": {
        "stimuli": ["pro_25um", "pro_250um", "pro_2.5mm"],
        "labels": ["Pro 25µM", "Pro 250µM", "Pro 2.5mM"],
        "short_name": "pro",
    },
    "food": {
        "stimuli": ["fex_3", "fex_2", "fex_1"],
        "labels": ["Fex 1:1000", "Fex 1:100", "Fex 1:10"],
        "short_name": "fex",
    },
}

#: Source: cell 4.
N_CLUSTERS = 4
RANDOM_STATE = 42
CLUSTER_RANGE = range(2, 6)
DOWNSAMPLING_FACTOR = 10
SCATTER_SIZE = 10
SCATTER_ALPHA = 0.6
SCALE_BAR_Y = 20
SCALE_BAR_XMIN = 20
SCALE_BAR_XMAX = 120

#: Source: cell 4.
CLUSTER_COLORS = {
    "Steep-Ramping": "#FFDF00",
    "High-threshold": "#3b528b",
    "Shallow-Ramping": "#440154",
    "Low-threshold": "#21908d",
}

#: Source: cell 27 (the hand-derived quinine cluster-ID -> response-type
#: mapping -- see module docstring point 2). Not reusable for another
#: odorant's `CURRENT_ODORANT` without re-doing the manual inspection step,
#: since k-means cluster IDs aren't stable labels across different input data.
QUININE_CLUSTER_NAME_MAP = {
    1: "Steep-Ramping",
    0: "High-threshold",
    2: "Shallow-Ramping",
    3: "Low-threshold",
}

#: Source: cell 30's `cluster_order`.
RESPONSE_CONDITION_ORDER = ["Steep-Ramping", "High-threshold", "Shallow-Ramping", "Low-threshold"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_data(
    ref_brain_path: str | Path,
    regression_data_path: str | Path,
    response_vector_path: str | Path,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Load the reference brain volume, regression results, and response vectors.

    Source: cells 9-11.
    """
    ref_brain = io.imread(ref_brain_path)
    regress = pd.read_csv(regression_data_path)
    data_resp = pd.read_hdf(response_vector_path)
    data_resp["coords"] = data_resp["coords_serialised"]
    return ref_brain, regress, data_resp


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def prepare_concentration_data(
    data: pd.DataFrame,
    odorant_config: dict,
    regression_data: pd.DataFrame | None = None,
    correlation_threshold: float = 0.3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter and pivot neuron responses for one odorant's concentration series.

    Filters neurons by correlation with the highest concentration (r >=
    `correlation_threshold`) when `regression_data` is given, restricts to
    `odorant_config`'s stimuli, averages repeated trials per neuron, and
    pivots to one row per neuron with one column per concentration. Returns
    `(pivot_df, averaged_responses)`: `pivot_df` is the wide-format table
    used for clustering, `averaged_responses` the long-format table
    `merge_coordinates` later re-derives neuron coordinates from.

    Source: cell 13.
    """
    stimuli = odorant_config["stimuli"]

    if regression_data is not None:
        highest_conc = stimuli[-1]
        corr_column = f"{highest_conc}_correlation"

        if corr_column in regression_data.columns:
            stim_columns = regression_data[[corr_column, "coords", "area"]]
            filtered_reg = stim_columns[stim_columns[corr_column] >= correlation_threshold]
            data_filtered = pd.merge(filtered_reg, data, on=["area", "coords"], how="inner")
        else:
            data_filtered = data.copy()
    else:
        data_filtered = data.copy()

    data_filtered = data_filtered[data_filtered["stimulus"].isin(stimuli)].copy()

    averaged_responses = (
        data_filtered
        .groupby(["fish_id", "neuron_id", "stimulus", "coords", "area"])["resp"]
        .mean()
        .reset_index()
    )

    averaged_responses["stimulus"] = pd.Categorical(
        averaged_responses["stimulus"], categories=stimuli, ordered=True,
    )
    averaged_responses = averaged_responses.sort_values(["fish_id", "neuron_id", "stimulus"])

    pivot_df = averaged_responses.pivot_table(
        index=["fish_id", "neuron_id", "area"], columns="stimulus", values="resp",
    ).reset_index()
    pivot_df = pivot_df.dropna(subset=stimuli)

    return pivot_df, averaged_responses


def perform_clustering(
    pivot_df: pd.DataFrame,
    stimuli: list[str],
    n_clusters: int = N_CLUSTERS,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, KMeans, np.ndarray]:
    """K-means cluster neurons by their (standardized) per-concentration responses.

    Source: cell 14.
    """
    response_data = pivot_df[stimuli].values
    scaled_data = StandardScaler().fit_transform(response_data)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    pivot_df = pivot_df.copy()
    pivot_df["cluster"] = kmeans.fit_predict(scaled_data)

    return pivot_df, kmeans, scaled_data


def assign_cluster_names(pivot_df: pd.DataFrame, cluster_name_map: dict[int, str]) -> pd.DataFrame:
    """Map each cluster ID to its hand-assigned response-type name.

    Raises `ValueError` if `cluster_name_map` doesn't cover every cluster ID
    present in `pivot_df` -- this is the notebook's own manual-step
    validation check (cell 15), not new to this port.

    Source: cell 15.
    """
    unique_clusters = set(pivot_df["cluster"].unique())
    mapped_clusters = set(cluster_name_map.keys())

    if unique_clusters != mapped_clusters:
        missing = unique_clusters - mapped_clusters
        raise ValueError(
            f"Not all clusters are mapped! Missing clusters: {missing}\n"
            f"Please update cluster_name_map to include all cluster IDs."
        )

    pivot_df = pivot_df.copy()
    pivot_df["response_condition"] = pivot_df["cluster"].map(cluster_name_map)
    return pivot_df


def calculate_response_fractions(
    pivot_df: pd.DataFrame, cluster_order: list[str] | None = None,
) -> pd.DataFrame:
    """Neuron count and fraction per response type.

    Source: cell 16.
    """
    counts = pivot_df["response_condition"].value_counts()
    total = len(pivot_df)

    fractions_df = pd.DataFrame({
        "response_condition": counts.index,
        "count": counts.values,
        "fraction": (counts / total).values,
    })

    if cluster_order:
        fractions_df["response_condition"] = pd.Categorical(
            fractions_df["response_condition"], categories=cluster_order, ordered=True,
        )
        fractions_df = fractions_df.sort_values("response_condition")

    return fractions_df


def merge_coordinates(pivot_df: pd.DataFrame, averaged_responses: pd.DataFrame) -> pd.DataFrame:
    """Re-attach each neuron's coordinates from `averaged_responses`.

    Source: cell 17.
    """
    coords_df = averaged_responses[["fish_id", "neuron_id", "coords"]].drop_duplicates()
    return pivot_df.merge(coords_df, on=["fish_id", "neuron_id"], how="left")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_silhouette_analysis(
    scaled_data: np.ndarray,
    cluster_range: range = CLUSTER_RANGE,
    random_state: int = RANDOM_STATE,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, dict[int, float]]:
    """Silhouette score vs. number of clusters, for choosing `n_clusters`.

    Optional in the notebook (its own cell is commented out with "Uncomment
    to run") -- a diagnostic for picking `N_CLUSTERS`, not required by the
    rest of the pipeline. Source: cell 19.
    """
    silhouette_scores = {}
    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        cluster_labels = kmeans.fit_predict(scaled_data)
        silhouette_scores[n_clusters] = silhouette_score(scaled_data, cluster_labels)

    fig = plt.figure(figsize=(8, 5))
    plt.plot(list(silhouette_scores.keys()), list(silhouette_scores.values()), marker="o", linewidth=2)
    plt.xlabel("Number of Clusters", fontsize=14)
    plt.ylabel("Silhouette Score", fontsize=14)
    plt.title("Silhouette Analysis for Optimal Clusters", fontsize=16)
    plt.grid(alpha=0.3)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="svg", dpi=300, bbox_inches="tight")

    return fig, silhouette_scores


def plot_concentration_response_curves(
    pivot_df: pd.DataFrame,
    stimuli: list[str],
    labels: list[str],
    cluster_color_map: dict[str, str] | None = None,
    use_cluster_ids: bool = False,
    title: str | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Mean +/- STD response curve per cluster, across concentrations.

    `use_cluster_ids=True` (before `assign_cluster_names`) labels curves by
    raw cluster ID -- this is the plot the notebook's manual naming step
    requires a human to read. `use_cluster_ids=False` (after) labels by
    `response_condition` and colors via `cluster_color_map`.

    Source: cell 20.
    """
    fig = plt.figure(figsize=(8, 5))

    if use_cluster_ids:
        for cluster_id, group in pivot_df.groupby("cluster"):
            mean_resp = group[stimuli].mean()
            std_resp = group[stimuli].std()
            plt.plot(labels, mean_resp, marker="o", linewidth=2, label=f"Cluster {cluster_id}")
            plt.fill_between(range(len(labels)), mean_resp - std_resp, mean_resp + std_resp, alpha=0.2)
    else:
        for condition, group in pivot_df.groupby("response_condition"):
            mean_resp = group[stimuli].mean()
            std_resp = group[stimuli].std()
            color = cluster_color_map.get(condition, "gray") if cluster_color_map else None
            plt.plot(labels, mean_resp, marker="o", linewidth=2, label=condition, color=color)
            plt.fill_between(range(len(labels)), mean_resp - std_resp, mean_resp + std_resp, alpha=0.2, color=color)

    plt.xlabel("Stimulus Concentration", fontsize=14)
    plt.ylabel("Average Response", fontsize=14)
    plt.title(title or "Concentration-Response Curves", fontsize=16)
    plt.legend(loc="best", fontsize=11)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="svg", dpi=300, bbox_inches="tight")

    return fig


def plot_classified_neuron_fractions(
    pivot_df: pd.DataFrame,
    cluster_color_map: dict[str, str],
    odorant_name: str = "",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Stacked bar chart of response-type fractions, forebrain vs. hindbrain fish.

    Source: cell 21.
    """
    classified_responses = pivot_df.copy()
    classified_responses["fish_type"] = classified_responses["fish_id"].apply(
        lambda x: "_fb" if "_fb" in x else "_hb"
    )

    neuron_counts = (
        classified_responses
        .groupby(["fish_type", "response_condition"])
        .size()
        .reset_index(name="count")
    )
    neuron_totals = neuron_counts.groupby("fish_type")["count"].transform("sum")
    neuron_counts["fraction"] = neuron_counts["count"] / neuron_totals

    plot_df = neuron_counts.pivot(
        index="fish_type", columns="response_condition", values="fraction",
    ).fillna(0)

    cluster_order = [col for col in plot_df.columns if col in cluster_color_map]
    colors = [cluster_color_map[cluster] for cluster in cluster_order]

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_df[cluster_order].plot(kind="bar", stacked=True, color=colors, ax=ax)

    plt.ylabel("Fraction of Classified Neurons", fontsize=20)
    plt.xlabel("")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Forebrain", "Hindbrain"], fontsize=20, rotation=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.tick_params(width=2)
    ax.tick_params(axis="both", which="major", labelsize=22)

    if odorant_name:
        plt.title(odorant_name, ha="right", fontsize=20, weight="bold")

    ax.legend(
        title="Response Type", title_fontsize=16, fontsize=14,
        loc="upper left", bbox_to_anchor=(1, 1), labels=cluster_order, frameon=False,
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="svg", dpi=300, bbox_inches="tight")

    return fig


def plot_per_fish_fractions(
    pivot_df: pd.DataFrame,
    cluster_color_map: dict[str, str],
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Stacked bar chart of response-type fractions, one bar per fish.

    Source: cell 22.
    """
    pivot_df_copy = pivot_df.copy()
    pivot_df_copy["fish_base"] = pivot_df_copy["fish_id"].str.split("_").str[0]

    cluster_counts = (
        pivot_df_copy.groupby(["fish_base", "response_condition"])
        .size()
        .reset_index(name="count")
    )
    cluster_pivot = cluster_counts.pivot(
        index="fish_base", columns="response_condition", values="count",
    ).fillna(0)
    cluster_prop = cluster_pivot.div(cluster_pivot.sum(axis=1), axis=0)

    valid_clusters = [c for c in cluster_color_map if c in cluster_prop.columns]
    cluster_prop = cluster_prop[valid_clusters]
    colors = [cluster_color_map[c] for c in cluster_prop.columns]

    fig, ax = plt.subplots(figsize=(7, 6))
    cluster_prop.plot(kind="bar", stacked=True, color=colors, edgecolor="black", ax=ax)

    plt.ylabel("Fraction of Neurons", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel("Fish ID", fontsize=20, labelpad=10)
    plt.legend("")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="svg", dpi=300, bbox_inches="tight")

    return fig


def plot_response_conditions_2view(
    coords_array: np.ndarray,
    response_conditions: np.ndarray,
    ref_brain: np.ndarray,
    cluster_color_map: dict[str, str],
    downsampling: int = DOWNSAMPLING_FACTOR,
    scatter_size: float = SCATTER_SIZE,
    scatter_alpha: float = SCATTER_ALPHA,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Side-by-side dorsal + sagittal brain projections, neurons colored by response type.

    The categorical counterpart of
    `jenkins_et_al.plotting.plot_neuron_scatter_on_brain` -- see this
    module's docstring for why it isn't unified with that function yet.

    Source: cell 23 (the surviving, correctly-formatted definition -- see
    module docstring point 3 for the broken duplicate this replaced).
    """
    coords = coords_array[::downsampling]
    response_conditions = response_conditions[::downsampling]
    unique_conditions = np.unique(response_conditions)

    fig = plt.figure(figsize=(12, 5))
    gs = gridspec.GridSpec(1, 2, figure=fig)
    ax_horizontal = fig.add_subplot(gs[0, 0])
    ax_sagittal = fig.add_subplot(gs[0, 1])

    proj_horizontal = np.nanmean(ref_brain, axis=0)
    proj_horizontal = np.rot90(proj_horizontal, k=1)
    ax_horizontal.imshow(proj_horizontal, cmap="gray_r", alpha=0.5, origin="lower")

    for condition in unique_conditions:
        color = cluster_color_map.get(condition, "gray")
        cond_coords = coords[response_conditions == condition]
        ax_horizontal.scatter(
            cond_coords[:, 1], proj_horizontal.shape[0] - cond_coords[:, 2],
            c=color, s=scatter_size, edgecolor="none", alpha=scatter_alpha, label=condition,
        )

    ax_horizontal.axis("off")
    ax_horizontal.set_aspect("equal")
    ax_horizontal.set_xlim(0, proj_horizontal.shape[1])
    ax_horizontal.set_ylim(0, proj_horizontal.shape[0])
    ax_horizontal.hlines(
        y=proj_horizontal.shape[0] - SCALE_BAR_Y,
        xmin=SCALE_BAR_XMIN, xmax=SCALE_BAR_XMAX, color="black", linewidth=3,
    )

    proj_sagittal = np.nanmean(ref_brain, axis=2)
    ax_sagittal.imshow(proj_sagittal, cmap="gray_r", alpha=0.5, origin="lower")

    for condition in unique_conditions:
        color = cluster_color_map.get(condition, "gray")
        cond_coords = coords[response_conditions == condition]
        ax_sagittal.scatter(
            cond_coords[:, 1], cond_coords[:, 0],
            c=color, s=scatter_size, edgecolor="none", alpha=scatter_alpha,
        )

    ax_sagittal.axis("off")
    ax_sagittal.set_aspect("equal")
    ax_sagittal.set_xlim(0, proj_sagittal.shape[1])
    ax_sagittal.set_ylim(0, proj_sagittal.shape[0])

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=cond,
                   markerfacecolor=cluster_color_map.get(cond, "gray"), markersize=6)
        for cond in unique_conditions
    ]
    fig.legend(handles=handles, loc="upper left", title="Response Type",
               bbox_to_anchor=(0.85, 0.82), frameon=False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="svg", dpi=300, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Manual-inspection steps
# ---------------------------------------------------------------------------
# The notebook's own two "look at this plot, then decide" steps (see module
# docstring): reading the cluster-ID curves to write `cluster_name_map`, and
# the optional silhouette-score check before picking `n_clusters`. Both open
# their figure in the default web browser via `show_in_browser` -- a
# stand-in for Jupyter's inline display when running this as a script.

def show_cluster_id_curves(
    pivot_df: pd.DataFrame, stimuli: list[str], labels: list[str], title: str | None = None,
) -> Path:
    """Open the cluster-ID concentration-response curves in a browser tab.

    Read this plot, then build a `cluster_name_map` (cluster ID -> response
    type) and pass it to `assign_cluster_names`.
    """
    fig = plot_concentration_response_curves(pivot_df, stimuli, labels, use_cluster_ids=True, title=title)
    path = show_in_browser(fig)
    plt.close(fig)
    return path


def show_silhouette_analysis(
    scaled_data: np.ndarray, cluster_range: range = CLUSTER_RANGE, random_state: int = RANDOM_STATE,
) -> tuple[dict[int, float], Path]:
    """Open the silhouette-score-vs-n_clusters plot in a browser tab.

    Optional diagnostic (source cell is commented out in the notebook) for
    choosing `N_CLUSTERS` before calling `perform_clustering`.
    """
    fig, silhouette_scores = plot_silhouette_analysis(scaled_data, cluster_range, random_state)
    path = show_in_browser(fig)
    plt.close(fig)
    return silhouette_scores, path
