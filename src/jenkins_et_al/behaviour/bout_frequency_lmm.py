"""All-bout frequency mixed-effects analysis for head-fixed behaviour data.

Ported from notebooks/behaviour_head_fixed/bout_frequency_LMM.ipynb; logic is
unchanged from the notebook, only moved into a module, split into
loading / processing / stats / plotting functions, and given a
docstring/type-hint pass per the /tdd REFACTOR checklist.

This notebook "owns" the fb/hb behaviour-traces-posture loading pattern and
the mixed-model-fitting / paired-dot-plot functions shared with
`bout_type_frequency_LMM.ipynb` and `reorientation_posture_LMM.ipynb` (see
PORTING_PLAN.md). Those two notebooks' ports should reuse this module's
`load_behavior_data`/`prepare_stimulus_df`/`run_mixed_models`/
`plot_paired_fish_by_group` rather than redefining them, once each is
independently verified against its own golden capture.

`valence_colors` here defaults to `config.VALENCE_COLORS` (green/magenta),
the canonical pair chosen across the whole port -- not this notebook's own
default (`{"POS": "limegreen", "NEG": "deeppink"}`, cell 6c90fa43). This is a
deliberate presentation choice already resolved in PORTING_PLAN.md, not a
data/analysis difference, so it doesn't affect golden verification (which
checks the data feeding the plot, not exact pixel colors).
"""
from __future__ import annotations

from ast import literal_eval
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from jenkins_et_al.config import (
    NEGATIVE_STIMULI,
    POSITIVE_STIMULI,
    STIMULUS_COLORS,
    STIMULUS_LABELS,
    VALENCE_COLORS,
    VALENCE_LABELS,
)

#: Source: cell b0bf576a.
SAMPLING_FREQ = 60.85
STIMULUS_ONSET_FRAME = 3225
WINDOW_SEC = 20


# ---------------------------------------------------------------------------
# Loading (source: cells ded5388e, 49ca15a8, 400812f7)
# ---------------------------------------------------------------------------

def load_behavior_data(path: Path, suffix: str) -> pd.DataFrame:
    """Load one fb/hb 7dpf behaviour-traces-posture CSV.

    Appends `suffix` (e.g. "_fb"/"_hb") to `fish_id` and parses the
    stringified `angles`/`bouts` list columns into a `bout_indx` column.
    """
    df = pd.read_csv(path)
    df["fish_id"] = df["fish_id"].astype(str) + suffix
    df["angles"] = df["angles"].apply(literal_eval)
    df["bout_indx"] = df["bouts"].apply(literal_eval)
    return df


def load_fb_hb_behavior_data(fb_path: Path, hb_path: Path) -> pd.DataFrame:
    """Load and concatenate the forebrain ("_fb") and hindbrain ("_hb") CSVs."""
    bhv_fb = load_behavior_data(fb_path, "_fb")
    bhv_hb = load_behavior_data(hb_path, "_hb")
    return pd.concat([bhv_fb, bhv_hb], ignore_index=True)


def prepare_stimulus_df(bhv: pd.DataFrame, key_stimuli: tuple[str, ...]) -> pd.DataFrame:
    """Filter to `key_stimuli` and add `fish_id_clean`/`session` if not already present."""
    stimulus_df = bhv[bhv["stimulus"].isin(key_stimuli)].copy()

    if "fish_id_clean" not in stimulus_df.columns:
        stimulus_df["fish_id_clean"] = (
            stimulus_df["fish_id"].astype(str).str.replace(r"(_fb|_hb)$", "", regex=True)
        )

    if "session" not in stimulus_df.columns:
        stimulus_df["session"] = stimulus_df["fish_id"].astype(str).str.extract(r"(fb|hb)", expand=False)

    return stimulus_df


# ---------------------------------------------------------------------------
# All-bout frequency (source: cell c37dbe1f)
# ---------------------------------------------------------------------------

def calculate_all_bout_frequency(
    bouts: list[list[int]],
    stimulus_onset_frame: int,
    window_frames: int,
    sampling_freq: float,
) -> tuple[float, float]:
    """(pre-stimulus, post-stimulus) all-bout frequency (bouts/s) for one trial."""
    pre_stim_bouts = [
        b for b in bouts
        if stimulus_onset_frame - window_frames <= b[0] < stimulus_onset_frame
    ]
    post_stim_bouts = [
        b for b in bouts
        if stimulus_onset_frame <= b[0] < stimulus_onset_frame + window_frames
    ]

    duration_s = window_frames / sampling_freq
    pre_freq = len(pre_stim_bouts) / duration_s if duration_s > 0 else np.nan
    post_freq = len(post_stim_bouts) / duration_s if duration_s > 0 else np.nan

    return pre_freq, post_freq


def add_all_bout_frequency_columns(
    stimulus_df: pd.DataFrame,
    stimulus_onset_frame: int = STIMULUS_ONSET_FRAME,
    window_sec: int = WINDOW_SEC,
    sampling_freq: float = SAMPLING_FREQ,
) -> pd.DataFrame:
    """Add `all_bout_freq_prestim`/`all_bout_freq_poststim` if not already present.

    Returns a copy; does not mutate `stimulus_df`.
    """
    stimulus_df = stimulus_df.copy()
    window_frames = int(window_sec * sampling_freq)

    if {"all_bout_freq_prestim", "all_bout_freq_poststim"}.issubset(stimulus_df.columns):
        return stimulus_df

    pre_vals = []
    post_vals = []

    for _, row in stimulus_df.iterrows():
        pre_freq, post_freq = calculate_all_bout_frequency(
            bouts=row["bout_indx"],
            stimulus_onset_frame=stimulus_onset_frame,
            window_frames=window_frames,
            sampling_freq=sampling_freq,
        )
        pre_vals.append(pre_freq)
        post_vals.append(post_freq)

    stimulus_df["all_bout_freq_prestim"] = pre_vals
    stimulus_df["all_bout_freq_poststim"] = post_vals

    return stimulus_df


# ---------------------------------------------------------------------------
# Long-format reshaping (source: cell 2e87c0d5)
# ---------------------------------------------------------------------------

def build_bout_frequency_long_df(stimulus_df: pd.DataFrame, key_stimuli: tuple[str, ...]) -> pd.DataFrame:
    """Melt pre/post all-bout-frequency columns into long (fish x stimulus x condition) form."""
    df_long = stimulus_df.melt(
        id_vars=["fish_id_clean", "session", "stimulus", "trial_number"],
        value_vars=["all_bout_freq_prestim", "all_bout_freq_poststim"],
        var_name="condition",
        value_name="bout_frequency",
    )

    df_long["condition"] = df_long["condition"].replace({
        "all_bout_freq_prestim": "Prestim",
        "all_bout_freq_poststim": "Poststim",
    })
    df_long["condition"] = pd.Categorical(
        df_long["condition"], categories=["Prestim", "Poststim"], ordered=True
    )

    return df_long[df_long["stimulus"].isin(key_stimuli)].copy()


# ---------------------------------------------------------------------------
# Valence grouping (source: cell 16984e5d)
# ---------------------------------------------------------------------------

def stimulus_to_valence(
    stim: str,
    positive_stims: tuple[str, ...] = POSITIVE_STIMULI,
    negative_stims: tuple[str, ...] = NEGATIVE_STIMULI,
) -> str | float:
    """"POS"/"NEG" for a known stimulus code, else NaN."""
    if stim in positive_stims:
        return "POS"
    elif stim in negative_stims:
        return "NEG"
    return np.nan


def add_valence_column(
    df_long: pd.DataFrame,
    positive_stims: tuple[str, ...] = POSITIVE_STIMULI,
    negative_stims: tuple[str, ...] = NEGATIVE_STIMULI,
) -> pd.DataFrame:
    """Add a `valence` column and drop rows whose stimulus maps to neither group."""
    df_long = df_long.copy()
    df_long["valence"] = df_long["stimulus"].apply(
        lambda s: stimulus_to_valence(s, positive_stims, negative_stims)
    )
    return df_long.dropna(subset=["valence"]).copy()


# ---------------------------------------------------------------------------
# Mixed-effects model (source: cell 6c90fa43)
# ---------------------------------------------------------------------------

def run_mixed_models(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Fit one `bout_frequency ~ condition` mixed model per `group_col` value.

    Random effect on `fish_id_clean`; adds a `session` variance-component
    term when `session` is present and has any non-null value.
    """
    results = []

    for group_name in sorted(df[group_col].dropna().unique()):
        g = df[df[group_col] == group_name].copy()

        if g.empty:
            continue

        use_session = "session" in g.columns and g["session"].notna().any()

        if use_session:
            model = smf.mixedlm(
                "bout_frequency ~ condition",
                g,
                groups=g["fish_id_clean"],
                vc_formula={"session": "0 + C(session)"},
            )
        else:
            model = smf.mixedlm("bout_frequency ~ condition", g, groups=g["fish_id_clean"])

        fit = model.fit()

        pvals = fit.pvalues
        if "condition[T.Poststim]" in pvals:
            p_value = pvals["condition[T.Poststim]"]
        elif "condition[T.Prestim]" in pvals:
            p_value = pvals["condition[T.Prestim]"]
        else:
            p_value = np.nan

        results.append({
            group_col: group_name,
            "n_fish": g["fish_id_clean"].nunique(),
            "p_value": p_value,
        })

    return pd.DataFrame(results)


def summarize_fish_plot_df(df_long: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Per-fish-per-group-per-condition mean/std/count of `bout_frequency`."""
    return (
        df_long.groupby(["fish_id_clean", group_col, "condition"], observed=False)["bout_frequency"]
        .agg(fish_mean="mean", fish_std="std", n_observations="count")
        .reset_index()
    )


# ---------------------------------------------------------------------------
# Plotting (source: cell 6c90fa43)
# ---------------------------------------------------------------------------

def plot_paired_fish_by_group(
    fish_plot_df: pd.DataFrame,
    results_df: pd.DataFrame,
    group_col: str,
    group_order: list[str],
    group_title_map: dict[str, str],
    group_color_map: dict[str, str],
    ylabel: str = "Bout Frequency (bouts/s)",
    figsize: tuple[float, float] = (18, 5),
    save_path: Path | None = None,
):
    """Paired pre/post dot plot (one panel per group), with the mixed-model p-value annotated."""
    fig, axes = plt.subplots(1, len(group_order), figsize=figsize, sharey=True)

    if len(group_order) == 1:
        axes = [axes]

    for i, group_name in enumerate(group_order):
        ax = axes[i]

        g = fish_plot_df[fish_plot_df[group_col] == group_name].copy()

        paired = g.pivot_table(
            index="fish_id_clean", columns="condition", values="bout_frequency", aggfunc="mean"
        ).dropna()

        if paired.empty:
            ax.set_title(group_title_map.get(group_name, group_name))
            continue

        for _, row in paired.iterrows():
            ax.plot(
                [0, 1], [row["Prestim"], row["Poststim"]],
                color="0.7", linewidth=1.2, alpha=0.8, zorder=1,
            )

        ax.scatter(np.zeros(len(paired)), paired["Prestim"], color="black", s=35, zorder=2)
        ax.scatter(
            np.ones(len(paired)), paired["Poststim"],
            color=group_color_map.get(group_name, "gray"), s=35, zorder=2,
        )

        pre_mean = paired["Prestim"].mean()
        post_mean = paired["Poststim"].mean()

        ax.hlines(pre_mean, -0.12, 0.12, color="black", linewidth=2.5, zorder=3)
        ax.hlines(post_mean, 0.88, 1.12, color="black", linewidth=2.5, zorder=3)

        p_row = results_df[results_df[group_col] == group_name]
        if not p_row.empty:
            p = p_row.iloc[0]["p_value"]
            ax.text(0.5, 0.8, f"p = {p:.3f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pre", "Post"], fontsize=12)
        ax.set_title(group_title_map.get(group_name, group_name), fontsize=18)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", width=1.5, length=4, direction="out", labelsize=20)

    axes[0].set_ylabel(ylabel, fontsize=20)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, format="svg", dpi=300, bbox_inches="tight")

    return fig, axes


# ---------------------------------------------------------------------------
# Full pipeline (source: whole notebook)
# ---------------------------------------------------------------------------

def run_bout_frequency_analysis(
    fb_path: Path,
    hb_path: Path,
    key_stimuli: tuple[str, ...],
    stimulus_labels: dict[str, str] = STIMULUS_LABELS,
    stimulus_colors: dict[str, str] = STIMULUS_COLORS,
    valence_labels: dict[str, str] = VALENCE_LABELS,
    valence_colors: dict[str, str] = VALENCE_COLORS,
    positive_stims: tuple[str, ...] = POSITIVE_STIMULI,
    negative_stims: tuple[str, ...] = NEGATIVE_STIMULI,
) -> dict[str, pd.DataFrame]:
    """Run the notebook's full per-stimulus + valence mixed-model analysis.

    Returns the four tables the notebook saves to CSV:
    `mixed_results_stimulus`, `mixed_results_valence`,
    `fish_plot_df_stimulus`, `fish_plot_df_valence`.
    """
    bhv = load_fb_hb_behavior_data(fb_path, hb_path)
    stimulus_df = prepare_stimulus_df(bhv, key_stimuli)
    stimulus_df = add_all_bout_frequency_columns(stimulus_df)

    df_long = build_bout_frequency_long_df(stimulus_df, key_stimuli)

    mixed_results_stimulus = run_mixed_models(df_long, group_col="stimulus")
    mixed_results_stimulus["stimulus_label"] = mixed_results_stimulus["stimulus"].replace(stimulus_labels)
    fish_plot_df_stimulus = summarize_fish_plot_df(df_long, group_col="stimulus")

    df_long_valence = add_valence_column(df_long, positive_stims, negative_stims)
    mixed_results_valence = run_mixed_models(df_long_valence, group_col="valence")
    fish_plot_df_valence = summarize_fish_plot_df(df_long_valence, group_col="valence")

    return {
        "mixed_results_stimulus": mixed_results_stimulus,
        "mixed_results_valence": mixed_results_valence,
        "fish_plot_df_stimulus": fish_plot_df_stimulus,
        "fish_plot_df_valence": fish_plot_df_valence,
    }
