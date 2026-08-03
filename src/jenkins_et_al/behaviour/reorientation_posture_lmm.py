"""Posture (tail-angle) trace and pre/post mixed-effects analysis for head-fixed data.

Ported from notebooks/behaviour_head_fixed/reorientation_posture_LMM.ipynb;
logic is unchanged from the notebook, only moved into a module, split into
loading/processing/stats/plotting functions, and given a docstring/type-hint
pass per the /tdd REFACTOR checklist.

**Golden reference is a fresh top-to-bottom run, not the notebook's saved
output.** The saved notebook's `trial_summary_df` shows `n_trials=5` for
"Ade" while every other stimulus shows 6 -- inconsistent with the
uniform-6-trials-per-fish pattern confirmed in every other notebook in this
folder (#18-#20). Running the notebook's own exported cell source fresh,
top-to-bottom, gives a self-consistent `n_trials=6` for all seven stimuli
and correspondingly different (not just rounding-different) LMM estimates
from what the notebook displays. Per the skill's "run from a clean kernel"
rule, the fresh run is the golden reference; this port and its regression
test are verified against that, not against the notebook's stale display.

**Confirmed bug, not reproduced**: the notebook's own "n fish per stimulus"
diagnostic print (cell a73e874f) casts `fish_plot_df_stimulus["stimulus"]` to
a `Categorical` using raw stimulus codes ("ade", "cad_2.5mm", ...) as
categories -- but by that point `stimulus_df["stimulus"]` (and everything
derived from it) already holds display labels ("Ade", "Cad", ...) from the
early `stimulus_df['stimulus'] = stimulus_df['stimulus'].replace(stimulus_labels)`
relabeling two cells earlier. Every value falls outside the mismatched
categories, so the printed counts are all 0. This is a diagnostic-print-only
bug: the paired plot, the LMM results, and the effects table all key off the
separately-populated `stimulus_label` column, so it has no effect on any
saved/plotted analysis output. Not reproduced here since it would just be
dead, misleading code; noted in PORTING_PLAN.md.

Reuses `jenkins_et_al.behaviour.bout_frequency_lmm.load_fb_hb_behavior_data`
(identical loading logic to this notebook's own copy).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from matplotlib.ticker import MultipleLocator
from scipy.signal import butter, filtfilt
from scipy.stats import circmean

from jenkins_et_al.behaviour.bout_frequency_lmm import load_fb_hb_behavior_data

SAMPLING_FREQ = 60.85
STIMULUS_ONSET_TIME_S = 53

#: Source: cell 8e0cd312.
STIMULUS_ORDER = ("Ade", "Cad", "HCl", "Qun", "Fex", "Kin", "Pro")

#: Raw stimulus codes for the low/high concentration comparison, keyed by
#: display-label "family". Source: cell ac828977.
CONCENTRATION_PAIRS = {
    "Cad": {"low": "cad_25um", "high": "cad_2.5mm"},
    "Qun": {"low": "qui_25um", "high": "qui_2.5mm"},
    "Pro": {"low": "pro_25um", "high": "pro_2.5mm"},
    "Fex": {"low": "fex3", "high": "fex1"},
}
STIMULUS_FAMILY_ORDER = ("Cad", "Qun", "Pro", "Fex")


# ---------------------------------------------------------------------------
# Posture trace computation (source: cells b032b3f1, e347c8a8)
# ---------------------------------------------------------------------------

def get_posture_trace(fish_angles: np.ndarray, bout_indices_list: list[list[int]], extension_frames: int = 10) -> np.ndarray:
    """Linearly interpolate over each bout (extended by `extension_frames`), then low-pass filter.

    A bout is only interpolated over if there are valid frames on both sides
    of its extended range; bouts touching the trace start/end are left as-is.
    """
    modified_angles = np.array(fish_angles)

    for bout_indices in bout_indices_list:
        start = bout_indices[0]
        end = bout_indices[-1]
        extended_end = min(end + extension_frames, len(fish_angles) - 1)

        if start > 0 and extended_end < len(fish_angles) - 1:
            pre_bout_value = fish_angles[start - 1]
            post_bout_value = fish_angles[extended_end + 1]

            bout_range = np.linspace(0, 1, extended_end - start + 1)
            interpolated_values = pre_bout_value + bout_range * (post_bout_value - pre_bout_value)

            modified_angles[start:extended_end + 1] = interpolated_values

    return filtfilt(*butter(2, 0.5 / (SAMPLING_FREQ / 2), btype="low"), modified_angles)


def subtract_baseline(posture_trace: np.ndarray, baseline_range: slice = slice(1560, 3000)) -> np.ndarray:
    """Subtract the mean of `posture_trace[baseline_range]` from the whole trace."""
    baseline_mean = np.mean(posture_trace[baseline_range])
    return posture_trace - baseline_mean


def add_posture_column(
    stimulus_df: pd.DataFrame,
    extension_frames: int = 20,
    baseline_range: slice = slice(1560, 3000),
) -> pd.DataFrame:
    """Add a baseline-subtracted `posture` column. Returns a copy."""
    stimulus_df = stimulus_df.copy()
    stimulus_df["posture"] = stimulus_df.apply(
        lambda row: get_posture_trace(row["angles"], row["bout_indx"], extension_frames=extension_frames), axis=1
    )
    stimulus_df["posture"] = stimulus_df["posture"].apply(lambda trace: subtract_baseline(trace, baseline_range))
    return stimulus_df


def prepare_stimulus_df(
    bhv: pd.DataFrame,
    key_stimuli: tuple[str, ...],
    stimulus_labels: dict[str, str],
    extension_frames: int = 20,
) -> pd.DataFrame:
    """Filter to `key_stimuli`, add posture + fish_id_clean, and relabel `stimulus` to display labels.

    Matches the notebook's own ordering: posture is computed from the raw
    stimulus codes, and `stimulus` is only relabeled to display form
    afterwards.
    """
    stimulus_df = bhv[bhv["stimulus"].isin(key_stimuli)].copy()
    stimulus_df = add_posture_column(stimulus_df, extension_frames=extension_frames)
    stimulus_df["fish_id_clean"] = stimulus_df["fish_id"].str.replace(r"(_fb|_hb)$", "", regex=True)
    stimulus_df["stimulus"] = stimulus_df["stimulus"].replace(stimulus_labels)
    return stimulus_df


# ---------------------------------------------------------------------------
# Posture trace group/stimulus means (source: cells 60192756, 3c8f76fa)
# ---------------------------------------------------------------------------

def moving_average(data: np.ndarray, window_size: int = 100) -> np.ndarray:
    return np.convolve(data, np.ones(window_size) / window_size, mode="same")


def assign_valence(df: pd.DataFrame, positive_stims: list[str]) -> pd.DataFrame:
    """Add a "valence" column ("POS"/"NEG") from display-label `stimulus` values."""
    df = df.copy()
    df["valence"] = df["stimulus"].apply(lambda x: "POS" if x in positive_stims else "NEG")
    return df


def compute_group_posture_traces(
    df: pd.DataFrame, group_col: str = "valence", posture_col: str = "posture", use_abs: bool = True
) -> dict[str, dict]:
    """Mean/SEM posture trace per `group_col` value."""
    grouped_traces = {}

    for group_name, group in df.groupby(group_col):
        traces = np.stack(group[posture_col].values)
        if use_abs:
            traces = np.abs(traces)

        mean_trace = np.nanmean(traces, axis=0)
        sem_trace = np.nanstd(traces, axis=0) / np.sqrt(np.sum(~np.isnan(traces), axis=0))

        grouped_traces[group_name] = {"mean": mean_trace, "sem": sem_trace, "n": len(group)}

    return grouped_traces


def compute_stimulus_posture_traces(
    df: pd.DataFrame, posture_col: str = "posture", use_abs: bool = True
) -> dict[str, dict]:
    """Mean/SEM posture trace per stimulus."""
    traces_by_stimulus = {}

    for stimulus, group in df.groupby("stimulus"):
        traces = np.stack(group[posture_col].values)
        if use_abs:
            traces = np.abs(traces)

        mean_trace = np.nanmean(traces, axis=0)
        sem_trace = np.nanstd(traces, axis=0) / np.sqrt(np.sum(~np.isnan(traces), axis=0))

        traces_by_stimulus[stimulus] = {"mean": mean_trace, "sem": sem_trace, "n": len(group)}

    return traces_by_stimulus


def smooth_group_traces(grouped_traces: dict[str, dict], window_size: int = 100) -> dict[str, dict]:
    """Moving-average smooth each group's mean/SEM trace."""
    return {
        group_name: {
            "mean": moving_average(data["mean"], window_size),
            "sem": moving_average(data["sem"], window_size),
            "n": data["n"],
        }
        for group_name, data in grouped_traces.items()
    }


def add_stimulus_shading(ax, start: float = 0, solid_end: float = 10, fade_end: float = 20,
                          alpha_max: float = 0.3, y_top: float | None = None) -> None:
    """Grey stimulus-window shading with a fade-out gradient, plus a solid onset bar."""
    if y_top is None:
        y_top = ax.get_ylim()[1]

    gradient = np.ones((2, 256, 4))
    gradient[:, :, :3] *= 0.5
    gradient[:, :, 3] = np.concatenate([np.full(128, alpha_max), np.linspace(alpha_max, 0, 128)])

    extent = [start, fade_end, ax.get_ylim()[0], y_top]
    ax.imshow(gradient, extent=extent, origin="lower", aspect="auto", zorder=-1)
    ax.hlines(y=y_top, xmin=start, xmax=solid_end, colors="black", linewidth=4)


def plot_group_posture_traces(
    grouped_traces: dict[str, dict],
    sampling_freq: float,
    stimulus_onset_time: float = STIMULUS_ONSET_TIME_S,
    colors: dict[str, str] | None = None,
    labels: dict[str, str] | None = None,
    window_size: int = 100,
    xlim: tuple[float, float] = (-30, 30),
    ylim: tuple[float, float] = (0, 9),
):
    """Grouped (e.g. appetitive/aversive) mean posture traces with SEM bands."""
    if colors is None:
        colors = {"POS": "green", "NEG": "magenta"}
    if labels is None:
        labels = {"POS": "Appetitive", "NEG": "Aversive"}

    smoothed_traces = smooth_group_traces(grouped_traces, window_size=window_size)

    fig, ax = plt.subplots(figsize=(10, 5))

    for group_name, data in smoothed_traces.items():
        mean_deg = np.degrees(data["mean"])
        sem_deg = np.degrees(data["sem"])
        time = np.arange(len(mean_deg)) / sampling_freq - stimulus_onset_time

        ax.plot(time, mean_deg, label=labels.get(group_name, group_name), color=colors.get(group_name, "black"), linewidth=2.5)
        ax.fill_between(time, mean_deg - sem_deg, mean_deg + sem_deg, color=colors.get(group_name, "black"), alpha=0.15)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    add_stimulus_shading(ax, start=0, solid_end=10, fade_end=20, y_top=ylim[1])
    ax.axvline(0, linestyle="--", color="black", linewidth=2.5)

    ax.set_xlabel("Time Relative to Stimulus Onset (s)", fontsize=28)
    ax.set_ylabel("Posture Trace (°)", fontsize=28)
    ax.legend(fontsize=22, frameon=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=28)
    ax.set_xticks(np.arange(-10, 31, 5))
    ax.xaxis.set_tick_params(which="both", direction="out", length=5, bottom=True, top=False)
    ax.yaxis.set_tick_params(which="both", direction="out", length=5, left=True, right=False)

    plt.tight_layout()
    return fig, ax


def plot_stimulus_posture_traces(
    traces_by_stimulus: dict[str, dict],
    sampling_freq: float,
    stimulus_onset_time: float = STIMULUS_ONSET_TIME_S,
    stimulus_order: tuple[str, ...] | None = None,
    stimulus_label_map: dict[str, str] | None = None,
    stimulus_color_map: dict[str, str] | None = None,
    window_size: int = 100,
    xlim: tuple[float, float] = (-30, 30),
    ylim: tuple[float, float] = (0, 9),
):
    """One overlaid mean posture trace per stimulus."""
    fig, ax = plt.subplots(figsize=(8, 6))

    if stimulus_order is None:
        stimulus_order = list(traces_by_stimulus.keys())

    for stimulus in stimulus_order:
        if stimulus not in traces_by_stimulus:
            continue

        data = traces_by_stimulus[stimulus]
        smoothed_trace_degrees = np.degrees(moving_average(data["mean"], window_size))
        smoothed_sem_degrees = np.degrees(moving_average(data["sem"], window_size))
        time = np.arange(len(smoothed_trace_degrees)) / sampling_freq - stimulus_onset_time

        label = stimulus_label_map.get(stimulus, stimulus) if stimulus_label_map else stimulus
        color = stimulus_color_map.get(stimulus, None) if stimulus_color_map else None

        ax.plot(time, smoothed_trace_degrees, label=label, color=color, linewidth=2.5)
        ax.fill_between(time, smoothed_trace_degrees - smoothed_sem_degrees, smoothed_trace_degrees + smoothed_sem_degrees,
                         color=color, alpha=0.1)

    gradient = np.ones((2, 256, 4))
    gradient[:, :, :3] *= 0.5
    gradient[:, :, 3] = np.concatenate([np.full(128, 0.3), np.linspace(0.3, 0, 128)])
    ax.imshow(gradient, extent=[0, 20, ylim[0], 10], origin="lower", aspect="auto", zorder=-1)
    ax.hlines(y=ylim[1], xmin=0, xmax=10, colors="black", linewidth=4)

    ax.set_xlabel("Time Relative to Stimulus Onset (s)", fontsize=28)
    ax.set_ylabel("Posture Trace (°)", fontsize=28)
    ax.legend(fontsize=22, frameon=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=28)
    ax.xaxis.set_tick_params(which="both", direction="out", length=5, bottom=True, top=False)
    ax.yaxis.set_tick_params(which="both", direction="out", length=5, left=True, right=False)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.xaxis.set_major_locator(MultipleLocator(10))

    plt.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Posture pre/post mixed-effects model (source: cell ea3de6c9)
# ---------------------------------------------------------------------------

def split_fish_id_and_session(full_fish_id: str) -> tuple[str, str]:
    """Split an id like "230910_fb" into `("230910", "fb")`."""
    fish_base, session = str(full_fish_id).rsplit("_", 1)
    return fish_base, session


def p_to_symbol(p_value: float) -> str:
    """4-tier significance stars (plus "ns"/"NA"), matching this notebook's own thresholds."""
    if pd.isna(p_value):
        return "NA"
    if p_value < 0.0001:
        return "****"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def add_stimulus_display_columns(df: pd.DataFrame, stimulus_labels: dict[str, str]) -> pd.DataFrame:
    """Add a `stimulus_label` column mapping `stimulus` through `stimulus_labels` (identity if not found)."""
    df = df.copy()
    df["stimulus_label"] = df["stimulus"].map(lambda x: stimulus_labels.get(x, x))
    return df


def compute_trial_and_lmm_posture_tables(
    df: pd.DataFrame,
    posture_col: str = "posture",
    stimulus_col: str = "stimulus",
    fish_id_col: str = "fish_id",
    frame_rate: float = SAMPLING_FREQ,
    stimulus_onset_s: float = STIMULUS_ONSET_TIME_S,
    pre_window_s: float = 20,
    post_window_s: float = 20,
    use_abs: bool = True,
    low: float = 0,
    high: float = np.pi,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-trial pre/post circular-mean posture amplitude, and its long-format melt for LMM fitting.

    A trial is dropped if its pre- or post-stimulus window is empty, all-NaN,
    or has an undefined circular mean.
    """
    df = df.copy()

    stimulus_frame = int(stimulus_onset_s * frame_rate)
    pre_frames = int(pre_window_s * frame_rate)
    post_frames = int(post_window_s * frame_rate)

    rows = []

    for _, row in df.iterrows():
        trace = np.asarray(row[posture_col], dtype=float)
        if use_abs:
            trace = np.abs(trace)

        pre_start = max(0, stimulus_frame - pre_frames)
        pre_end = stimulus_frame
        post_start = stimulus_frame
        post_end = min(len(trace), stimulus_frame + post_frames)

        pre_trace = trace[pre_start:pre_end]
        post_trace = trace[post_start:post_end]

        if len(pre_trace) == 0 or len(post_trace) == 0:
            continue
        if np.all(np.isnan(pre_trace)) or np.all(np.isnan(post_trace)):
            continue

        pre_mean = np.degrees(circmean(pre_trace, high=high, low=low, nan_policy="omit"))
        post_mean = np.degrees(circmean(post_trace, high=high, low=low, nan_policy="omit"))

        if np.isnan(pre_mean) or np.isnan(post_mean):
            continue

        fish_base, session = split_fish_id_and_session(row[fish_id_col])

        rows.append({
            "fish_id_full": row[fish_id_col],
            "fish_base": str(fish_base),
            "fish_id_clean": str(fish_base),
            "session": str(session),
            "stimulus": row[stimulus_col],
            "pre_amplitude_deg": pre_mean,
            "post_amplitude_deg": post_mean,
        })

    trial_summary_df = pd.DataFrame(rows)
    trial_summary_df["trial_id"] = trial_summary_df.groupby(["fish_base", "stimulus"]).cumcount() + 1

    lmm_df = trial_summary_df.melt(
        id_vars=["fish_id_full", "fish_base", "fish_id_clean", "session", "stimulus", "trial_id"],
        value_vars=["pre_amplitude_deg", "post_amplitude_deg"],
        var_name="period",
        value_name="amplitude_deg",
    )
    lmm_df["period"] = lmm_df["period"].map({"pre_amplitude_deg": "Prestim", "post_amplitude_deg": "Poststim"})
    lmm_df["condition"] = lmm_df["period"]

    return trial_summary_df, lmm_df


def run_posture_mixed_models(df: pd.DataFrame, group_col: str) -> tuple[pd.DataFrame, dict]:
    """Fit one `amplitude_deg ~ condition` mixed model per `group_col` value.

    `session` variance-component term is included only when `session` has
    more than one distinct non-null value in that group. Any fitting
    exception is caught per-group and recorded in a `model_error` column
    rather than propagating.
    """
    results = []
    fitted_models = {}

    for group_name in sorted(df[group_col].dropna().unique()):
        g = df[df[group_col] == group_name].copy()

        if g.empty:
            continue

        use_session = "session" in g.columns and g["session"].notna().any() and g["session"].nunique() > 1

        try:
            if use_session:
                model = smf.mixedlm(
                    "amplitude_deg ~ condition", g, groups=g["fish_id_clean"],
                    vc_formula={"session": "0 + C(session)"},
                )
            else:
                model = smf.mixedlm("amplitude_deg ~ condition", g, groups=g["fish_id_clean"])

            fit = model.fit()
            fitted_models[group_name] = fit

            pvals = fit.pvalues
            if "condition[T.Poststim]" in pvals:
                p_value = pvals["condition[T.Poststim]"]
                effect_name = "condition[T.Poststim]"
            elif "condition[T.Prestim]" in pvals:
                p_value = pvals["condition[T.Prestim]"]
                effect_name = "condition[T.Prestim]"
            else:
                p_value = np.nan
                effect_name = None

            ci = fit.conf_int()
            ci_low = ci.loc[effect_name, 0] if effect_name in ci.index else np.nan
            ci_high = ci.loc[effect_name, 1] if effect_name in ci.index else np.nan
            estimate = fit.params.get(effect_name, np.nan)

            results.append({
                group_col: group_name,
                "n_fish": g["fish_id_clean"].nunique(),
                "n_sessions": g["session"].nunique(),
                "n_trials": g["trial_id"].nunique(),
                "estimate_post_vs_pre": estimate,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_value": p_value,
                "significance": p_to_symbol(p_value),
            })
        except Exception as exc:
            results.append({
                group_col: group_name,
                "n_fish": g["fish_id_clean"].nunique(),
                "n_sessions": g["session"].nunique(),
                "n_trials": g["trial_id"].nunique(),
                "estimate_post_vs_pre": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "p_value": np.nan,
                "significance": "NA",
                "model_error": str(exc),
            })

    return pd.DataFrame(results), fitted_models


def make_fish_plot_summary(lmm_df: pd.DataFrame) -> pd.DataFrame:
    """Per-fish-per-stimulus-per-condition mean amplitude, for the paired plot."""
    return (
        lmm_df.groupby(["fish_id_clean", "stimulus", "stimulus_label", "condition"], observed=False)["amplitude_deg"]
        .mean()
        .reset_index()
    )


def make_per_fish_summary_table(lmm_df: pd.DataFrame) -> pd.DataFrame:
    """Per-fish-per-stimulus-per-condition mean/std/n_observations, keyed by display label."""
    return (
        lmm_df.groupby(["fish_id_clean", "stimulus_label", "condition"], observed=False)
        .agg(fish_mean=("amplitude_deg", "mean"), fish_std=("amplitude_deg", "std"), n_observations=("trial_id", "nunique"))
        .reset_index()
        .rename(columns={"stimulus_label": "stimulus"})
    )


def plot_paired_fish_by_stimulus(
    fish_plot_df: pd.DataFrame,
    results_df: pd.DataFrame,
    group_col: str,
    group_order: list[str],
    group_title_map: dict[str, str],
    group_color_map: dict[str, str],
    ylabel: str = "Posture Angle (°)",
    figsize: tuple[float, float] = (12, 5),
):
    """Paired pre/post dot plot (one panel per stimulus), with a significance star (not a numeric p)."""
    fig, axes = plt.subplots(1, len(group_order), figsize=figsize, sharey=True)

    if len(group_order) == 1:
        axes = [axes]

    for i, group_name in enumerate(group_order):
        ax = axes[i]

        g = fish_plot_df[fish_plot_df[group_col] == group_name].copy()

        paired = g.pivot_table(
            index="fish_id_clean", columns="condition", values="amplitude_deg", aggfunc="mean"
        ).dropna()

        if paired.empty:
            ax.set_title(group_title_map.get(group_name, group_name), fontsize=18)
            continue

        for _, row in paired.iterrows():
            ax.plot([0, 1], [row["Prestim"], row["Poststim"]], color="0.7", linewidth=1.2, alpha=0.8, zorder=1)

        ax.scatter(np.zeros(len(paired)), paired["Prestim"], color="black", s=35, zorder=2)
        ax.scatter(np.ones(len(paired)), paired["Poststim"], color=group_color_map.get(group_name, "gray"), s=35, zorder=2)

        pre_mean = paired["Prestim"].mean()
        post_mean = paired["Poststim"].mean()
        ax.hlines(pre_mean, -0.12, 0.12, color="black", linewidth=2.5, zorder=3)
        ax.hlines(post_mean, 0.88, 1.12, color="black", linewidth=2.5, zorder=3)

        p_row = results_df[results_df[group_col] == group_name]
        if not p_row.empty:
            p = p_row.iloc[0]["p_value"]
            ax.text(0.5, 0.98, p_to_symbol(p), transform=ax.transAxes, ha="center", va="top",
                    fontsize=12, fontweight="bold")

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pre", "Post"], fontsize=12)
        ax.set_title(group_title_map.get(group_name, group_name), fontsize=18)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", width=1.5, length=4, direction="out", labelsize=14)

    axes[0].set_ylabel(ylabel, fontsize=16)
    plt.tight_layout()
    return fig, axes


def extract_lmm_effects_with_ci(fitted_models: dict, group_title_map: dict[str, str]) -> pd.DataFrame:
    """One row per fitted model's post-vs-pre effect estimate, CI, and p-value."""
    rows = []

    for stim, fit in fitted_models.items():
        pvals = fit.pvalues
        if "condition[T.Poststim]" in pvals:
            effect_name = "condition[T.Poststim]"
        elif "condition[T.Prestim]" in pvals:
            effect_name = "condition[T.Prestim]"
        else:
            effect_name = None

        ci = fit.conf_int()
        rows.append({
            "stimulus": stim,
            "stimulus_label": group_title_map.get(stim, stim),
            "estimate": fit.params.get(effect_name, np.nan) if effect_name else np.nan,
            "ci_low": ci.loc[effect_name, 0] if effect_name in ci.index else np.nan,
            "ci_high": ci.loc[effect_name, 1] if effect_name in ci.index else np.nan,
            "p_value": fit.pvalues.get(effect_name, np.nan) if effect_name else np.nan,
        })

    return pd.DataFrame(rows)


def plot_lmm_mean_shift_ci(
    effects_df: pd.DataFrame,
    stimulus_order: tuple[str, ...],
    stimulus_colors: dict[str, str],
    ylabel: str = "LMM mean shift (Post - Pre, °)",
    figsize: tuple[float, float] = (9, 5),
):
    """Per-stimulus mean-shift-with-CI plot, with a significance star per stimulus."""
    plot_df = effects_df.copy()
    plot_df["stimulus_label"] = pd.Categorical(plot_df["stimulus_label"], categories=stimulus_order, ordered=True)
    plot_df = plot_df.sort_values("stimulus_label")

    fig, ax = plt.subplots(figsize=figsize)

    max_abs = np.nanmax(np.abs(plot_df[["ci_low", "ci_high"]].to_numpy())) if not plot_df.empty else 1
    if not np.isfinite(max_abs) or max_abs == 0:
        max_abs = 1

    for i, stim_label in enumerate(stimulus_order):
        row = plot_df[plot_df["stimulus_label"] == stim_label]
        if row.empty:
            continue

        est = row["estimate"].iloc[0]
        ci_low = row["ci_low"].iloc[0]
        ci_high = row["ci_high"].iloc[0]
        p = row["p_value"].iloc[0]
        color = stimulus_colors.get(stim_label, "gray")

        if not (pd.isna(est) or pd.isna(ci_low) or pd.isna(ci_high)):
            ax.vlines(i, ci_low, ci_high, linewidth=2)
            ax.scatter(i, est, s=80, color=color, edgecolors="black", linewidths=0.4, zorder=3)
            ax.text(i, ci_high + 0.05 * max_abs, p_to_symbol(p), ha="center", va="bottom", fontsize=14, fontweight="bold")

    ax.axhline(0, color="0.5", linestyle="--", linewidth=1)
    ax.set_xticks(np.arange(len(stimulus_order)))
    ax.set_xticklabels(stimulus_order, rotation=45, ha="right", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=15)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.tick_params(axis="y", labelsize=12)
    ax.tick_params(axis="x", length=0)
    plt.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Low vs. high concentration, post-stimulus only (source: cells ac828977-3823140a)
# ---------------------------------------------------------------------------

def assign_concentration_group(
    stim: str, concentration_pairs: dict[str, dict[str, str]] = CONCENTRATION_PAIRS
) -> tuple[str, str] | tuple[float, float]:
    """(stimulus_family, "Low"/"High") for a raw stimulus code in `concentration_pairs`, else (NaN, NaN)."""
    for stim_family, pair in concentration_pairs.items():
        if stim == pair["low"]:
            return stim_family, "Low"
        if stim == pair["high"]:
            return stim_family, "High"
    return np.nan, np.nan


def build_concentration_post_df(
    conc_trial_summary_df: pd.DataFrame,
    concentration_pairs: dict[str, dict[str, str]] = CONCENTRATION_PAIRS,
) -> pd.DataFrame:
    """Post-stimulus-only amplitude table with stimulus_family/concentration columns, for the Low-vs-High LMMs."""
    conc_post_df = conc_trial_summary_df[
        ["fish_id_full", "fish_base", "fish_id_clean", "session", "stimulus", "trial_id", "post_amplitude_deg"]
    ].copy().rename(columns={"post_amplitude_deg": "amplitude_deg"})

    conc_post_df[["stimulus_family", "concentration"]] = conc_post_df["stimulus"].apply(
        lambda x: pd.Series(assign_concentration_group(x, concentration_pairs))
    )
    conc_post_df = conc_post_df.dropna(subset=["stimulus_family", "concentration", "amplitude_deg"]).copy()
    conc_post_df["concentration"] = pd.Categorical(conc_post_df["concentration"], categories=["Low", "High"], ordered=True)

    return conc_post_df


def run_low_high_poststim_lmms(
    conc_post_df: pd.DataFrame, stimulus_family_order: tuple[str, ...] = STIMULUS_FAMILY_ORDER
) -> tuple[pd.DataFrame, dict]:
    """Fit one `amplitude_deg ~ concentration` mixed model per stimulus family (post-stimulus values only)."""
    results = []
    fitted_models = {}

    for stim_family in stimulus_family_order:
        g = conc_post_df[conc_post_df["stimulus_family"] == stim_family].copy()

        if g.empty:
            continue

        use_session = "session" in g.columns and g["session"].notna().any() and g["session"].nunique() > 1

        try:
            if use_session:
                model = smf.mixedlm(
                    "amplitude_deg ~ concentration", g, groups=g["fish_id_clean"],
                    vc_formula={"session": "0 + C(session)"},
                )
            else:
                model = smf.mixedlm("amplitude_deg ~ concentration", g, groups=g["fish_id_clean"])

            fit = model.fit()
            fitted_models[stim_family] = fit

            effect_name = "concentration[T.High]"
            ci = fit.conf_int()
            p_value = fit.pvalues.get(effect_name, np.nan)

            results.append({
                "stimulus_family": stim_family,
                "estimate_high_vs_low": fit.params.get(effect_name, np.nan),
                "ci_low": ci.loc[effect_name, 0] if effect_name in ci.index else np.nan,
                "ci_high": ci.loc[effect_name, 1] if effect_name in ci.index else np.nan,
                "p_value": p_value,
                "significance": p_to_symbol(p_value),
                "n_fish": g["fish_id_clean"].nunique(),
                "n_sessions": g["session"].nunique(),
                "n_trials": len(g),
            })
        except Exception as exc:
            results.append({
                "stimulus_family": stim_family,
                "estimate_high_vs_low": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "p_value": np.nan,
                "significance": "NA",
                "n_fish": g["fish_id_clean"].nunique(),
                "n_sessions": g["session"].nunique(),
                "n_trials": len(g),
                "model_error": str(exc),
            })

    return pd.DataFrame(results), fitted_models


def build_concentration_fish_plot_df(conc_post_df: pd.DataFrame) -> pd.DataFrame:
    """Per-fish-per-family mean post-stimulus amplitude, by Low/High concentration."""
    return (
        conc_post_df.groupby(["fish_id_clean", "stimulus_family", "concentration"], observed=False)["amplitude_deg"]
        .mean()
        .reset_index()
    )


def plot_single_stimulus_low_high_poststim(
    fish_plot_df: pd.DataFrame,
    results_df: pd.DataFrame,
    stimulus_family: str,
    stimulus_colors: dict[str, str],
    ylabel: str = "Post-stimulus posture angle (°)",
    figsize: tuple[float, float] = (3, 4),
    save_path: Path | None = None,
):
    """Paired Low-vs-High post-stimulus dot plot for one stimulus family."""
    g = fish_plot_df[fish_plot_df["stimulus_family"] == stimulus_family].copy()

    paired = g.pivot_table(
        index="fish_id_clean", columns="concentration", values="amplitude_deg", aggfunc="mean"
    ).dropna(subset=["Low", "High"])

    fig, ax = plt.subplots(figsize=figsize)

    if paired.empty:
        ax.set_title(stimulus_family, fontsize=18)
        ax.text(0.5, 0.5, "No paired Low/High data", ha="center", va="center", transform=ax.transAxes)
        return fig, ax

    low_color = stimulus_colors.get(f"{stimulus_family} Low", "lightgray")
    high_color = stimulus_colors.get(stimulus_family, "gray")

    for _, row in paired.iterrows():
        ax.plot([0, 1], [row["Low"], row["High"]], color="0.7", linewidth=1.2, alpha=0.8, zorder=1)

    ax.scatter(np.zeros(len(paired)), paired["Low"], color=low_color, edgecolors="black", linewidths=0.4, s=40, zorder=2)
    ax.scatter(np.ones(len(paired)), paired["High"], color=high_color, edgecolors="black", linewidths=0.4, s=40, zorder=2)

    ax.hlines(paired["Low"].mean(), -0.12, 0.12, color=low_color, linewidth=3, zorder=3)
    ax.hlines(paired["High"].mean(), 0.88, 1.12, color=high_color, linewidth=3, zorder=3)

    p_row = results_df[results_df["stimulus_family"] == stimulus_family]
    if not p_row.empty:
        p = p_row.iloc[0]["p_value"]
        sig = p_to_symbol(p)

        y_min = np.nanmin(paired[["Low", "High"]].to_numpy())
        y_max = np.nanmax(paired[["Low", "High"]].to_numpy())
        y_range = y_max - y_min
        if y_range == 0 or not np.isfinite(y_range):
            y_range = 1

        y_sig = y_max + 0.12 * y_range

        ax.plot(
            [0, 0, 1, 1],
            [y_sig - 0.03 * y_range, y_sig, y_sig, y_sig - 0.03 * y_range],
            color="black", linewidth=1.2,
        )
        ax.text(0.5, y_sig + 0.03 * y_range, sig, ha="center", va="bottom", fontsize=14, fontweight="bold")

        ax.set_ylim(y_min - 0.1 * y_range, y_sig + 0.2 * y_range)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Low", "High"], fontsize=22, rotation=45)
    ax.set_ylabel(ylabel, fontsize=22)
    ax.set_title(stimulus_family, fontsize=18)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", width=1.5, length=4, direction="out", labelsize=22)

    fig.subplots_adjust(left=0.30, right=0.95, bottom=0.20, top=0.90)

    if save_path is not None:
        fig.savefig(save_path, format="svg", dpi=300, bbox_inches="tight")

    return fig, ax


# ---------------------------------------------------------------------------
# Full pipeline (source: whole notebook)
# ---------------------------------------------------------------------------

def run_reorientation_posture_analysis(
    fb_path: Path,
    hb_path: Path,
    key_stimuli: tuple[str, ...],
    stimulus_labels: dict[str, str],
    stimulus_order: tuple[str, ...] = STIMULUS_ORDER,
) -> dict[str, pd.DataFrame]:
    """Run the notebook's full per-stimulus posture LMM + Low-vs-High-concentration LMM analysis.

    Returns `lmm_results_df`, `per_fish_summary_df`, `effects_df`,
    `conc_results_df`, `conc_fish_plot_df`.
    """
    bhv = load_fb_hb_behavior_data(fb_path, hb_path)
    stimulus_df = prepare_stimulus_df(bhv, key_stimuli, stimulus_labels)

    trial_summary_df, lmm_df = compute_trial_and_lmm_posture_tables(
        stimulus_df, stimulus_onset_s=STIMULUS_ONSET_TIME_S,
    )
    trial_summary_df = add_stimulus_display_columns(trial_summary_df, stimulus_labels)
    lmm_df = add_stimulus_display_columns(lmm_df, stimulus_labels)

    lmm_results_df, fitted_lmm_models = run_posture_mixed_models(lmm_df, group_col="stimulus")
    lmm_results_df["stimulus_label"] = lmm_results_df["stimulus"].replace(stimulus_labels)
    lmm_results_df["stimulus_label"] = pd.Categorical(lmm_results_df["stimulus_label"], categories=stimulus_order, ordered=True)
    lmm_results_df = lmm_results_df.sort_values("stimulus_label").reset_index(drop=True)

    per_fish_summary_df = make_per_fish_summary_table(lmm_df)
    per_fish_summary_df["stimulus"] = pd.Categorical(per_fish_summary_df["stimulus"], categories=stimulus_order, ordered=True)
    per_fish_summary_df["condition"] = pd.Categorical(per_fish_summary_df["condition"], categories=["Poststim", "Prestim"], ordered=True)
    per_fish_summary_df = per_fish_summary_df.sort_values(["fish_id_clean", "stimulus", "condition"]).reset_index(drop=True)

    effects_df = extract_lmm_effects_with_ci(fitted_lmm_models, stimulus_labels)
    effects_df["stimulus_label"] = pd.Categorical(effects_df["stimulus_label"], categories=stimulus_order, ordered=True)
    effects_df = effects_df.sort_values("stimulus_label").reset_index(drop=True)

    conc_stims = [stim for pair in CONCENTRATION_PAIRS.values() for stim in [pair["low"], pair["high"]]]
    conc_df = bhv[bhv["stimulus"].isin(conc_stims)].copy()
    conc_df = add_posture_column(conc_df)

    conc_trial_summary_df, _ = compute_trial_and_lmm_posture_tables(
        conc_df, stimulus_onset_s=STIMULUS_ONSET_TIME_S,
    )
    conc_post_df = build_concentration_post_df(conc_trial_summary_df)
    conc_results_df, conc_fitted_models = run_low_high_poststim_lmms(conc_post_df)
    conc_fish_plot_df = build_concentration_fish_plot_df(conc_post_df)

    return {
        "lmm_results_df": lmm_results_df,
        "per_fish_summary_df": per_fish_summary_df,
        "effects_df": effects_df,
        "conc_results_df": conc_results_df,
        "conc_fish_plot_df": conc_fish_plot_df,
    }
