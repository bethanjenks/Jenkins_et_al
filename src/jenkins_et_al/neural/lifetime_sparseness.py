"""Lifetime sparseness of neuronal responses across stimuli, per brain area.

Ported from notebooks/neural/lifetime_sparseness.ipynb; logic is unchanged
from the notebook, only moved into a module, split into
loading/processing/stats/plotting functions, and given a docstring/type-hint
pass per the /tdd REFACTOR checklist. The whole-brain neuron scatter plot
(originally `plot_sparseness_brain`, cell 27) is now a thin wrapper around
the general `jenkins_et_al.plotting.plot_neuron_scatter_on_brain`, since that
projection pattern recurs across several other (not yet ported) neural
notebooks with different colormaps.

**Per-area significance test is Wilcoxon signed-rank, not the notebook's
original Mann-Whitney U** -- the user changed this directly in the notebook
(commit 95133c5, ahead of this port) as a real analysis decision, not a
refactor-shape change. This port reflects that current file content, per the
skill's "re-read fresh at port time" rule. `notebooks/neural/response_latency.ipynb`
and `notebooks/neural/template_matching_classification.ipynb` share this
same "boxplot vs. reference area" pattern and, as of this port, still use
Mann-Whitney U -- flagging this as a new cross-notebook inconsistency
introduced by the edit, not silently unifying it. Resolve when those
notebooks are ported.

**Confirmed bug, not reproduced**: the notebook's own nMLF-mask reassignment
section (cell 9, `is_within_3d_mask` + the `data['area'] = data.apply(...)`
call) has no effect on any downstream output. `resp_pop` (source of `df`,
and everything derived from it) is pivoted from `data` in cell 7, *before*
the nMLF reassignment mutates `data['area']` in cell 9 -- so the reassigned
area labels are never read again. Confirmed by comparing `resp_pop`'s area
index against the reassignment: 'nMLF' never appears in it. `is_within_3d_mask`
is kept here as a standalone utility (it's independently correct/useful) but
is not wired into `run_lifetime_sparseness_analysis`, since doing so would
change the paper's actual (frozen) output rather than reproduce it.

Reuses `jenkins_et_al.config.FOREBRAIN_AREA_COLORS` and
`jenkins_et_al.config.AREA_SHORTHAND` (transcribed from the same palette and
the same external `area_shorthand_dict.json` this notebook loads at
runtime -- confirmed identical) instead of hard-coding a second copy.
`config.REFERENCE_AREA` ("olfactory_bulb", sourced from
response_latency.ipynb) is *not* reused here: this notebook's own
`REFERENCE_AREA` is the post-shorthand-mapping code `"OB"`, a different
representation for a different pipeline stage -- flagged, not merged.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from jenkins_et_al.config import AREA_SHORTHAND, FOREBRAIN_AREA_COLORS
from jenkins_et_al.plotting import plot_neuron_scatter_on_brain

#: Source: cell 3.
KEY_STIMULI = ("ade", "pro_2.5mm", "qui_2.5mm", "cad_2.5mm", "fex_1", "kw", "ph4.5")

#: Reference area for the vs.-OB significance test. Source: cell 3.
REFERENCE_AREA = "OB"

#: Source: cell 3.
AREAS_TO_PLOT = ("OE", "OB", "Pal", "SubP", "PT", "dHb", "vHb", "POA", "PTh", "EmT", "HI", "HR")
AREA_ORDER = ("OE", "OB", "Pal", "SubP", "POA", "PTh", "EmT", "dHb", "vHb", "PT", "HR", "HI")

#: Fraction of neurons taken from each tail of the sparseness distribution
#: for the brain-projection plot. Source: cell 3.
BRAIN_PLOT_PERCENTAGE = 0.02


# ---------------------------------------------------------------------------
# nMLF area mask (source: cell 9) -- standalone utility, not wired into the
# pipeline below; see module docstring for why.
# ---------------------------------------------------------------------------

def is_within_3d_mask(coordinates: np.ndarray, mask: np.ndarray) -> bool:
    """True if `coordinates` (z, y, x) index a nonzero voxel of `mask`."""
    coordinates = np.array(coordinates, dtype=int)
    if coordinates.shape != (3,):
        return False

    z, y, x = coordinates
    depth, height, width = mask.shape

    if (0 <= z < depth) and (0 <= y < height) and (0 <= x < width):
        return mask[z, y, x] > 0
    return False


# ---------------------------------------------------------------------------
# Load + compute lifetime sparseness (source: cells 7, 11-14)
# ---------------------------------------------------------------------------

def load_response_pivot(data_path: Path, key_stimuli: tuple[str, ...] = KEY_STIMULI) -> pd.DataFrame:
    """Load the per-trial response table and pivot to neurons x (stimulus, trial)."""
    fish_df = pd.read_hdf(data_path)
    data = fish_df[fish_df["stimulus"].isin(key_stimuli)]

    return data.pivot_table(
        index=("fish_id", "neuron_id", "coords_serialised", "area"),
        columns=["stimulus", "trial_number"],
        values="resp",
    )


def lifetime_sparseness(vector: np.ndarray) -> float:
    """Lifetime sparseness of one neuron's per-stimulus response vector.

    0 = broadly tuned (responds similarly to all stimuli), 1 = narrowly
    tuned. Negative values are handled by shifting the vector by its
    minimum before computing sparseness.
    """
    n = len(vector)
    if n == 0:
        return 0

    min_value = np.min(vector)
    resp_vector = vector - min_value

    sum_r = np.sum(resp_vector)
    sum_r_squared = np.sum(resp_vector**2)

    if sum_r_squared == 0 or sum_r == 0:
        return 0

    return 1 - (sum_r**2 / (n * sum_r_squared))


def compute_population_vector(resp_pop: pd.DataFrame, key_stimuli: tuple[str, ...] = KEY_STIMULI) -> np.ndarray:
    """(n_stimuli, n_neurons) array of each neuron's trial-averaged response per stimulus."""
    selected_pop_vec = [np.mean(resp_pop[stimulus].values, axis=1) for stimulus in key_stimuli]
    return np.asarray(selected_pop_vec)


def build_neuron_sparseness_df(resp_pop: pd.DataFrame, sparseness_values: np.ndarray) -> pd.DataFrame:
    """Attach per-neuron sparseness to `resp_pop`, parse coordinates, and drop unscored neurons.

    `resp_pop` has MultiIndex (stimulus, trial_number) columns from the pivot
    in `load_response_pivot`; a scalar column assigned onto it becomes
    `("sparseness_values", "")`, hence the tuple key below.
    """
    df = resp_pop.copy()
    df["sparseness_values"] = sparseness_values
    df = df.reset_index()

    df["coords_serialised"] = df["coords_serialised"].astype(str)
    df["coords"] = df["coords_serialised"].apply(lambda x: np.asarray(json.loads(x)))
    df = df.drop("coords_serialised", axis=1)

    return df.dropna(subset=[("sparseness_values", "")])


def to_brain_map_format(df: pd.DataFrame) -> pd.DataFrame:
    """Rename `area`/`sparseness_values` to `area_names`/`values` for the neuron-level export."""
    return df.rename(columns={"area": "area_names", "sparseness_values": "values"})


# ---------------------------------------------------------------------------
# Aggregate by area/fish + significance vs. reference area (source: cells
# 16-17, 19-20)
# ---------------------------------------------------------------------------

def map_area_to_shorthand(df: pd.DataFrame, area_shorthand: dict[str, str] = AREA_SHORTHAND) -> pd.DataFrame:
    """Map raw area names to shorthand codes; any area absent from `area_shorthand` becomes NaN.

    In the notebook this mapping (cell 16) is an in-place assignment on the
    same `df` object that both the area/fish aggregation (cell 17) and the
    brain-projection neuron selection (cell 26, `select_extreme_neurons`)
    are run on afterward -- so a raw area name with no shorthand entry
    (confirmed: only `"not_assigned"`, 11228 of 200116 neurons) is silently
    dropped from *both* downstream steps, including the brain-projection
    plot's neuron pool, not just the per-area boxplot. Reproduced here as an
    explicit step feeding both callers, rather than relying on the same
    shared-mutable-state side effect.
    """
    df = df.copy()
    df["area"] = df["area"].map(area_shorthand)
    return df


def aggregate_area_fish_sparseness(
    mapped_df: pd.DataFrame,
    areas_to_plot: tuple[str, ...] = AREAS_TO_PLOT,
) -> pd.DataFrame:
    """Per-fish-per-area mean sparseness, restricted to `areas_to_plot`.

    `mapped_df` must already have shorthand-coded `area` values (see
    `map_area_to_shorthand`).
    """
    mapped_df = mapped_df.copy()
    mapped_df["fish_id"] = mapped_df["fish_id"].str.replace("_fb", "", regex=True).str.replace("_hb", "", regex=True)

    area_fish_sparseness = mapped_df.groupby(["area", "fish_id"])["sparseness_values"].mean().reset_index()
    return area_fish_sparseness[area_fish_sparseness["area"].isin(areas_to_plot)]


def compute_area_pvalues(
    area_fish_sparseness: pd.DataFrame,
    areas: np.ndarray,
    reference_area: str = REFERENCE_AREA,
    alternative: str = "two-sided",
) -> dict[str, float]:
    """Wilcoxon signed-rank p-value for each area's per-fish sparseness vs. `reference_area`'s."""
    reference_data = area_fish_sparseness[area_fish_sparseness["area"] == reference_area]["sparseness_values"]

    pvals = {}
    for area in areas:
        if area == reference_area:
            continue

        area_data = area_fish_sparseness[area_fish_sparseness["area"] == area]["sparseness_values"]
        if len(area_data) > 0:
            _, p = wilcoxon(reference_data, area_data, alternative=alternative)
            pvals[area] = p

    return pvals


def add_pvalues(area_fish_sparseness: pd.DataFrame, pvals: dict[str, float]) -> pd.DataFrame:
    """Merge a per-area `p_value` column onto `area_fish_sparseness`."""
    pval_df = pd.DataFrame.from_dict(pvals, orient="index", columns=["p_value"]).reset_index()
    pval_df = pval_df.rename(columns={"index": "area"})
    return area_fish_sparseness.merge(pval_df, on="area", how="left")


def summarize_by_area(area_fish_sparseness: pd.DataFrame) -> pd.DataFrame:
    """Per-area n_fish/mean/std/p_value summary table (p_value assumed identical within an area)."""
    summary = (
        area_fish_sparseness.groupby("area")
        .agg(
            n_fish=("fish_id", "nunique"),
            mean=("sparseness_values", "mean"),
            std=("sparseness_values", "std"),
            p_value=("p_value", "first"),
        )
        .reset_index()
    )
    summary["test"] = "Wilcoxon"
    return summary[["area", "n_fish", "mean", "std", "test", "p_value"]]


# ---------------------------------------------------------------------------
# Boxplot vs. reference area (source: cell 22)
# ---------------------------------------------------------------------------

def plot_sparseness_boxplot(
    data: pd.DataFrame,
    areas_to_plot: tuple[str, ...],
    area_order: tuple[str, ...],
    pvals: dict[str, float],
    reference_area: str | None = None,
    palette: dict[str, str] = FOREBRAIN_AREA_COLORS,
):
    """Sparseness boxplot by brain area, colored per-area, with jittered points and vs.-reference significance stars."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    plot_data = data[data["area"].isin(areas_to_plot)].copy()

    fig = plt.figure(figsize=(8, 6), facecolor="white")
    ax = plt.gca()
    ax.set_facecolor("white")

    ax = sns.boxplot(
        data=plot_data, x="area", y="sparseness_values", order=area_order,
        color="white", width=0.7, showcaps=False, showfliers=False,
        boxprops={"edgecolor": "black", "linewidth": 3},
        whiskerprops={"color": "black", "linewidth": 3},
        capprops={"color": "black", "linewidth": 3},
        medianprops={"color": "black", "linewidth": 3},
    )

    plotted_areas = [area for area in area_order if area in plot_data["area"].unique()]

    # seaborn draws each box's whiskers, caps, and median as a flat list of
    # Line2D objects, per box, in a fixed order (2 whiskers, 2 caps, then the
    # median) -- not part of its public API, but stable enough to index into.
    lines = ax.get_lines()
    num_boxes = len(plotted_areas)
    num_lines_per_box = len(lines) // num_boxes if num_boxes > 0 else 0

    for i, area in enumerate(plotted_areas):
        color = palette[area]

        box = ax.patches[i]
        box.set_edgecolor(color)
        box.set_linewidth(2.7)
        box.set_facecolor("none")

        for line in lines[i * num_lines_per_box: i * num_lines_per_box + 4]:
            line.set_color(color)
            line.set_linewidth(2.5)

        if i * num_lines_per_box + 4 < len(lines):
            median = lines[i * num_lines_per_box + 4]
            median.set_color(color)
            median.set_linewidth(3)

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=24, rotation=45, ha="center")
    for label, area in zip(ax.get_xticklabels(), plotted_areas):
        label.set_color(palette[area])

    rng = np.random.default_rng(42)

    for i, area in enumerate(plotted_areas):
        area_data = plot_data[plot_data["area"] == area]["sparseness_values"].values
        jittered_x = rng.normal(loc=i, scale=0.1, size=len(area_data))
        color = palette[area]
        ax.scatter(
            jittered_x, area_data, facecolors="white", edgecolors=color,
            s=55, alpha=0.6, linewidth=1.5, zorder=3,
        )

    y_pos = 0.61
    for i, area in enumerate(area_order):
        if reference_area is not None and area == reference_area:
            continue

        p_val = pvals.get(area, 1.0)
        if p_val < 0.001:
            significance = "***"
        elif p_val < 0.01:
            significance = "**"
        elif p_val < 0.05:
            significance = "*"
        else:
            significance = "ns"

        if area in plotted_areas:
            ax.text(i, y_pos, significance, ha="center", va="bottom", fontsize=18, color="black", weight="bold")

    ax.set_xlabel("")
    ax.set_ylabel("Sparseness", fontsize=28)
    ax.set_ylim(0.3, 0.62)
    ax.set_yticks([0.3, 0.4, 0.5, 0.6])
    ax.tick_params(axis="both", labelsize=28)
    ax.tick_params(axis="x", width=2)
    ax.tick_params(axis="y", width=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Brain projection (source: cells 26-28)
# ---------------------------------------------------------------------------

def select_extreme_neurons(data: pd.DataFrame, percentage: float = BRAIN_PLOT_PERCENTAGE) -> pd.DataFrame:
    """Top and bottom `percentage` of neurons by sparseness (equal counts from each tail).

    `data` must be `map_area_to_shorthand`'s output: the `dropna` below only
    has an effect (excludes neurons with no shorthand entry) when `area` has
    already gone through that mapping -- see its docstring. Column keys are
    `(name, "")` tuples per `build_neuron_sparseness_df`'s MultiIndex-column note.
    """
    df_sorted = data.sort_values(by=("sparseness_values", ""), ascending=True)
    df_sorted = df_sorted.dropna(subset=[("area", "")])

    count = int(len(df_sorted) * percentage)
    least_sparse = df_sorted.head(count)
    most_sparse = df_sorted.tail(count)

    return pd.concat([least_sparse, most_sparse], ignore_index=True)


def plot_sparseness_brain(ref_brain: np.ndarray, data: pd.DataFrame, cmap):
    """Sparseness brain projection: thin wrapper around `plotting.plot_neuron_scatter_on_brain`."""
    coords = np.vstack(data["coords"])
    sparseness = data["sparseness_values"].to_numpy()

    return plot_neuron_scatter_on_brain(
        ref_brain, coords, sparseness, cmap,
        colorbar_label="Sparseness", point_size=10, point_alpha=0.5,
    )


# ---------------------------------------------------------------------------
# Full pipeline (source: whole notebook)
# ---------------------------------------------------------------------------

def run_lifetime_sparseness_analysis(
    data_path: Path,
    key_stimuli: tuple[str, ...] = KEY_STIMULI,
    area_shorthand: dict[str, str] = AREA_SHORTHAND,
    areas_to_plot: tuple[str, ...] = AREAS_TO_PLOT,
    reference_area: str = REFERENCE_AREA,
    alternative: str = "two-sided",
) -> dict[str, object]:
    """Run the notebook's full lifetime-sparseness pipeline.

    Returns `neuron_sparseness_df` (per-neuron, raw area names,
    `to_brain_map_format`-ready), `mapped_neuron_df` (per-neuron, shorthand
    area codes -- feed this to `select_extreme_neurons` for the brain plot,
    matching the notebook's own mutated-`df` reuse; see
    `map_area_to_shorthand`), `area_fish_sparseness_df`
    (per-fish-per-area, with p-values), `summary_df`, and `pvals`.
    """
    resp_pop = load_response_pivot(data_path, key_stimuli)
    population_vector = compute_population_vector(resp_pop, key_stimuli)
    sparseness_values = np.apply_along_axis(lifetime_sparseness, 0, population_vector)

    neuron_sparseness_df = build_neuron_sparseness_df(resp_pop, sparseness_values)
    mapped_neuron_df = map_area_to_shorthand(neuron_sparseness_df, area_shorthand)

    area_fish_sparseness_df = aggregate_area_fish_sparseness(mapped_neuron_df, areas_to_plot)
    areas_shorthand = area_fish_sparseness_df["area"].unique()

    pvals = compute_area_pvalues(area_fish_sparseness_df, areas_shorthand, reference_area, alternative)
    area_fish_sparseness_df = add_pvalues(area_fish_sparseness_df, pvals)
    summary_df = summarize_by_area(area_fish_sparseness_df)

    return {
        "neuron_sparseness_df": neuron_sparseness_df,
        "mapped_neuron_df": mapped_neuron_df,
        "area_fish_sparseness_df": area_fish_sparseness_df,
        "summary_df": summary_df,
        "pvals": pvals,
    }
