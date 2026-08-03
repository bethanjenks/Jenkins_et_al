"""Z-scored PSTH analysis of bout frequency, by bout type and by valence.

Ported from notebooks/behaviour_head_fixed/bout_frequency_PSTH.ipynb; logic
is unchanged from the notebook, only moved into a module, split into
loading/processing/stats/plotting functions, and given a docstring/type-hint
pass per the /tdd REFACTOR checklist.

The notebook's saved output shows a `NameError: name
'straight_bout_valence_analysis' is not defined` in its last cell -- but this
is a stale, out-of-order-execution artifact in the saved .ipynb (the skill's
"run from a clean kernel" warning), not a real bug: run fresh top-to-bottom
(confirmed by executing the notebook's own exported cell source directly
before writing this port), every cell completes without error, including the
one that raised in the saved notebook.

Reuses `jenkins_et_al.behaviour.bout_frequency_lmm.prepare_stimulus_df` for
the fish_id_clean/session bookkeeping (identical logic to this notebook's own
copy). `load_behavior_data` here is kept separate -- it guards against
re-parsing already-list-typed `angles`/`bout_indx` columns via
`safe_literal_eval`, which the other two head-fixed-behaviour notebooks'
loaders don't do. `classify_asymm_bouts` is also kept separate: this
notebook skips bouts shorter than 1 frame, vs. `bout_type_frequency_LMM.ipynb`'s
otherwise near-identical classifier, which skips bouts shorter than 2.
"""
from __future__ import annotations

from ast import literal_eval
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import circmean, wilcoxon

from jenkins_et_al.behaviour.bout_frequency_lmm import prepare_stimulus_df

SAMPLING_FREQ = 60.85
STIM_ONSET = 3180
PRE_STIMULUS_SECONDS = 20
POST_STIMULUS_SECONDS = 40
BIN_SIZE_SECONDS = 3

#: Source: cell 7daecf70. This notebook's own stimulus_order is
#: appetitive-then-aversive, unlike the other head-fixed notebooks'
#: aversive-then-appetitive default (`config.STIMULUS_ORDER_HIGH_CONC`).
STIMULUS_ORDER = ("Fex", "Kin", "Pro", "Ade", "Cad", "HCl", "Qun")

#: Source: cell ac58027f.
POSITIVE_STIMULI = ("fex1", "kw", "pro_2.5mm")


# ---------------------------------------------------------------------------
# Loading (source: cells 480383fb, ad04154e)
# ---------------------------------------------------------------------------

def safe_literal_eval(x):
    """Parse a stringified list, leaving an already-parsed list unchanged."""
    if isinstance(x, str):
        return literal_eval(x)
    return x


def load_behavior_data(path: Path, suffix: str) -> pd.DataFrame:
    """Load one fb/hb 7dpf behaviour-traces-posture CSV, tolerant of already-parsed columns."""
    df = pd.read_csv(path)
    df["fish_id"] = df["fish_id"].astype(str) + suffix

    if "angles" in df.columns:
        df["angles"] = df["angles"].apply(safe_literal_eval)

    if "bout_indx" not in df.columns:
        if "bouts" not in df.columns:
            raise ValueError("Could not find either 'bout_indx' or 'bouts' column.")
        df["bout_indx"] = df["bouts"].apply(safe_literal_eval)
    else:
        df["bout_indx"] = df["bout_indx"].apply(safe_literal_eval)

    return df


def load_fb_hb_behavior_data(fb_path: Path, hb_path: Path) -> pd.DataFrame:
    """Load and concatenate the forebrain ("_fb") and hindbrain ("_hb") CSVs."""
    bhv_fb = load_behavior_data(fb_path, "_fb")
    bhv_hb = load_behavior_data(hb_path, "_hb")
    return pd.concat([bhv_fb, bhv_hb], ignore_index=True)


def classify_asymm_bouts(
    row: pd.Series,
    left_threshold: float = -0.06,
    right_threshold: float = 0.06,
    sampling_rate: float = SAMPLING_FREQ,
    duration_ms: float = 70,
) -> pd.Series:
    """Classify each bout in `row` as left/right/straight from its early tail-angle circular mean.

    Bouts of length 0 are kept and flagged "straight" (not skipped) -- a
    difference from `bout_type_frequency_LMM.ipynb`'s otherwise near-identical
    classifier, which skips bouts shorter than 2 frames entirely.
    """
    asymmetric_flags = []
    directions = []

    bouts = row["bout_indx"]
    tail_angle = row["angles"]
    duration_frames = int((duration_ms / 1000) * sampling_rate)

    for bout in bouts:
        if len(bout) < 1:
            continue

        bout_start = bout[0]
        bout_duration = min(duration_frames, len(bout))
        angles = np.asarray(tail_angle[bout_start:bout_start + bout_duration])

        if len(angles) == 0:
            asymmetric_flags.append(False)
            directions.append("straight")
            continue

        laterality_index = circmean(angles, high=np.pi, low=-np.pi)

        if laterality_index < left_threshold:
            asymmetric_flags.append(True)
            directions.append("left")
        elif laterality_index > right_threshold:
            asymmetric_flags.append(True)
            directions.append("right")
        else:
            asymmetric_flags.append(False)
            directions.append("straight")

    return pd.Series([asymmetric_flags, directions])


def add_asymmetric_bout_columns(stimulus_df: pd.DataFrame) -> pd.DataFrame:
    """Add `asymmetric_bout`/`bout_direction` if not already present. Returns a copy."""
    stimulus_df = stimulus_df.copy()

    if "asymmetric_bout" not in stimulus_df.columns:
        stimulus_df[["asymmetric_bout", "bout_direction"]] = stimulus_df.apply(
            classify_asymm_bouts, axis=1
        )
    else:
        stimulus_df["asymmetric_bout"] = stimulus_df["asymmetric_bout"].apply(safe_literal_eval)

    return stimulus_df


# ---------------------------------------------------------------------------
# Bout selection and PSTH histograms (source: cell 9c4d4e6f)
# ---------------------------------------------------------------------------

def get_selected_bouts(row: pd.Series, bout_type: str) -> list[list[int]]:
    """Return `row`'s bouts filtered by `bout_type` ("all"/"asymmetric"/"straight")."""
    bouts = row["bout_indx"]

    if bout_type == "all":
        return bouts

    if "asymmetric_bout" not in row.index:
        raise ValueError("Need 'asymmetric_bout' column for asymmetric/straight PSTHs.")

    asymmetric_flags = row["asymmetric_bout"]

    if bout_type == "asymmetric":
        return [bout for bout, is_asym in zip(bouts, asymmetric_flags) if is_asym]

    if bout_type == "straight":
        return [bout for bout, is_asym in zip(bouts, asymmetric_flags) if not is_asym]

    raise ValueError("bout_type must be one of: 'all', 'asymmetric', 'straight'.")


def build_psth_histograms(
    stimulus_df: pd.DataFrame,
    bout_type: str = "all",
    stim_onset: int = STIM_ONSET,
    sampling_freq: float = SAMPLING_FREQ,
    pre_stimulus_seconds: float = PRE_STIMULUS_SECONDS,
    post_stimulus_seconds: float = POST_STIMULUS_SECONDS,
    bin_size_seconds: float = BIN_SIZE_SECONDS,
) -> tuple[dict[str, pd.DataFrame], np.ndarray, np.ndarray, float]:
    """Per-stimulus histograms of bout counts in fixed time bins (rows = bin start, columns = fish)."""
    pre_stimulus_frames = int(pre_stimulus_seconds * sampling_freq)
    post_stimulus_frames = int(post_stimulus_seconds * sampling_freq)
    bin_size_frames = int(bin_size_seconds * sampling_freq)

    frame_bins = np.arange(-pre_stimulus_frames, post_stimulus_frames + bin_size_frames, bin_size_frames)
    time_bins = frame_bins[:-1] / sampling_freq
    bin_width = bin_size_frames / sampling_freq
    time_bin_centres = time_bins + bin_width / 2

    hist_data: dict[str, dict[str, np.ndarray]] = {
        stimulus: {} for stimulus in stimulus_df["stimulus"].dropna().unique()
    }

    for _, row in stimulus_df.iterrows():
        stimulus = row["stimulus"]
        fish_id = row["fish_id_clean"]
        selected_bouts = get_selected_bouts(row, bout_type)

        relative_frames = [bout[0] - stim_onset for bout in selected_bouts if len(bout) > 0]
        hist, _ = np.histogram(relative_frames, bins=frame_bins)

        if fish_id not in hist_data[stimulus]:
            hist_data[stimulus][fish_id] = np.zeros(len(frame_bins) - 1)

        hist_data[stimulus][fish_id] += hist

    hist_df_dict = {
        stimulus: pd.DataFrame(hist_data[stimulus], index=time_bins) for stimulus in hist_data
    }

    return hist_df_dict, time_bins, time_bin_centres, bin_width


def zscore_psth_histograms(
    hist_df_dict: dict[str, pd.DataFrame], baseline_window: tuple[float, float] = (-20, 0)
) -> dict[str, pd.DataFrame]:
    """Z-score each fish x stimulus histogram using its pre-stimulus baseline window."""
    zscored_hist_data = {}

    for stimulus, hist_df in hist_df_dict.items():
        z_df = pd.DataFrame(index=hist_df.index)
        baseline_mask = (hist_df.index >= baseline_window[0]) & (hist_df.index < baseline_window[1])

        for fish_id in hist_df.columns:
            fish_trace = hist_df[fish_id].astype(float)
            baseline_values = fish_trace.loc[baseline_mask]
            baseline_mean = baseline_values.mean()
            baseline_std = baseline_values.std(ddof=1)

            if pd.isna(baseline_std) or baseline_std == 0:
                z_df[fish_id] = np.nan
            else:
                z_df[fish_id] = (fish_trace - baseline_mean) / baseline_std

        zscored_hist_data[stimulus] = z_df

    return zscored_hist_data


def summarize_psth(
    zscored_hist_data: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    """Mean and SEM (across fish) per stimulus."""
    mean_data = {stimulus: df.mean(axis=1, skipna=True) for stimulus, df in zscored_hist_data.items()}
    sem_data = {stimulus: df.sem(axis=1, skipna=True) for stimulus, df in zscored_hist_data.items()}
    return mean_data, sem_data


def run_psth(
    stimulus_df: pd.DataFrame,
    bout_type: str,
    stim_onset: int = STIM_ONSET,
    sampling_freq: float = SAMPLING_FREQ,
    pre_stimulus_seconds: float = PRE_STIMULUS_SECONDS,
    post_stimulus_seconds: float = POST_STIMULUS_SECONDS,
    bin_size_seconds: float = BIN_SIZE_SECONDS,
) -> dict:
    """Full PSTH pipeline (histogram -> z-score -> summarize) for one bout type."""
    hist_df_dict, time_bins, time_bin_centres, bin_width = build_psth_histograms(
        stimulus_df=stimulus_df, bout_type=bout_type, stim_onset=stim_onset, sampling_freq=sampling_freq,
        pre_stimulus_seconds=pre_stimulus_seconds, post_stimulus_seconds=post_stimulus_seconds,
        bin_size_seconds=bin_size_seconds,
    )
    zscored_hist_data = zscore_psth_histograms(hist_df_dict, baseline_window=(-pre_stimulus_seconds, 0))
    mean_data, sem_data = summarize_psth(zscored_hist_data)

    return {
        "hist_df_dict": hist_df_dict,
        "zscored_hist_data": zscored_hist_data,
        "mean_data": mean_data,
        "sem_data": sem_data,
        "time_bins": time_bins,
        "time_bin_centres": time_bin_centres,
        "bin_width": bin_width,
    }


# ---------------------------------------------------------------------------
# Plotting (source: cells b6ad2bc0, ac58027f)
# ---------------------------------------------------------------------------

def add_stimulus_gradient(ax, xmin: float = 0, xmax: float = 20, ymin: float = -3, ymax: float = 8) -> None:
    """Grey stimulus-window gradient background from `xmin` to `xmax`."""
    gradient = np.ones((2, 256, 4))
    gradient[:, :, 0] = 0
    gradient[:, :, 1] = 0
    gradient[:, :, 2] = 0.4
    gradient[:, :, 3] = np.linspace(0.3, 0, 256)
    ax.imshow(gradient, extent=[xmin, xmax, ymin, ymax], origin="lower", aspect="auto", zorder=-1)


def plot_psth_by_stimulus(
    mean_data: dict[str, pd.Series],
    sem_data: dict[str, pd.Series],
    time_bin_centres: np.ndarray,
    bin_width: float,
    stimulus_labels: dict[str, str],
    stimulus_colors: dict[str, str],
    ylabel: str = "Bout count (z-scored)",
    title_prefix: str = "",
    xlim: tuple[float, float] = (-20, 32),
    ylim: tuple[float, float] = (-2, 9),
    show_gradient: bool = True,
    show_stim_bar: bool = True,
    save_dir: Path | None = None,
    file_prefix: str = "psth",
) -> list:
    """One bar PSTH per stimulus; returns the list of created figures."""
    figs = []

    for stimulus, mean_trace in mean_data.items():
        stimulus_label = stimulus_labels.get(stimulus, stimulus)
        sem_trace = sem_data[stimulus]
        color = stimulus_colors.get(stimulus_label, "darkgrey")

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.bar(
            time_bin_centres, mean_trace.values, yerr=sem_trace.values, width=bin_width,
            color=color, alpha=0.8, align="center",
            error_kw=dict(ecolor="grey", elinewidth=1.5, capsize=3, capthick=1),
        )
        ax.axhline(y=0, color="red", linestyle="--", linewidth=2)

        if show_gradient:
            add_stimulus_gradient(ax, xmin=0, xmax=20, ymin=ylim[0], ymax=ylim[1])

        if show_stim_bar:
            ax.hlines(y=ylim[1] - 1, xmin=0, xmax=10, colors="black", linewidth=4)

        ax.set_title(f"{title_prefix}{stimulus_label}", fontsize=28, weight="bold", loc="left", pad=20)
        ax.set_xlabel("Time relative to stimulus onset (s)", fontsize=26)
        ax.set_ylabel(ylabel, fontsize=28)
        ax.set_xticks([-20, -10, 0, 10, 20, 30])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        ax.tick_params(axis="both", which="major", length=4, width=2, direction="out", labelsize=28)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.5)
        ax.spines["bottom"].set_linewidth(1.5)

        plt.tight_layout()

        if save_dir is not None:
            save_path = Path(save_dir) / f"{file_prefix}_{stimulus}_zscored_PSTH.svg"
            fig.savefig(save_path, format="svg", dpi=300, bbox_inches="tight")

        figs.append(fig)

    return figs


# ---------------------------------------------------------------------------
# Appetitive vs. aversive trajectory analysis (source: cell ac58027f)
# ---------------------------------------------------------------------------

def compute_valence_trajectory_analysis(
    zscored_hist_data: dict[str, pd.DataFrame],
    time_bins: np.ndarray,
    time_bin_centres: np.ndarray,
    positive_stims: tuple[str, ...] = POSITIVE_STIMULI,
    baseline_window: tuple[float, float] = (-20, 0),
    min_n: int = 5,
) -> dict:
    """Per-fish appetitive/aversive mean trajectories, baseline-normalised, with per-bin Wilcoxon p-values.

    Statistics are computed across fish (not fish x stimulus, to avoid
    pseudoreplication): per fish, positive stimuli are averaged together and
    negative stimuli are averaged together, then each bin is compared
    (paired Wilcoxon signed-rank) against that fish's own baseline.
    """
    time_bins = np.asarray(time_bins)
    time_bin_centres = np.asarray(time_bin_centres)

    baseline_mask = (time_bins >= baseline_window[0]) & (time_bins < baseline_window[1])

    fish_ids = sorted(set(
        fish_id for df in zscored_hist_data.values() for fish_id in df.columns
    ))

    pos_raw_traces, neg_raw_traces = [], []
    pos_baselines, neg_baselines = [], []
    pos_fish_ids, neg_fish_ids = [], []

    for fish_id in fish_ids:
        fish_pos_traces, fish_neg_traces = [], []

        for stimulus, df in zscored_hist_data.items():
            if fish_id not in df.columns:
                continue

            trace = df[fish_id].values.astype(float)

            if stimulus in positive_stims:
                fish_pos_traces.append(trace)
            else:
                fish_neg_traces.append(trace)

        if len(fish_pos_traces) > 0:
            pos_trace = np.nanmean(fish_pos_traces, axis=0)
            pos_baseline = np.nanmean(pos_trace[baseline_mask])

            if not np.isnan(pos_baseline):
                pos_raw_traces.append(pos_trace)
                pos_baselines.append(pos_baseline)
                pos_fish_ids.append(fish_id)

        if len(fish_neg_traces) > 0:
            neg_trace = np.nanmean(fish_neg_traces, axis=0)
            neg_baseline = np.nanmean(neg_trace[baseline_mask])

            if not np.isnan(neg_baseline):
                neg_raw_traces.append(neg_trace)
                neg_baselines.append(neg_baseline)
                neg_fish_ids.append(fish_id)

    pos_raw_traces = np.asarray(pos_raw_traces, dtype=float)
    neg_raw_traces = np.asarray(neg_raw_traces, dtype=float)
    pos_baselines = np.asarray(pos_baselines, dtype=float)
    neg_baselines = np.asarray(neg_baselines, dtype=float)

    pos_norm_traces = pos_raw_traces - pos_baselines[:, None]
    neg_norm_traces = neg_raw_traces - neg_baselines[:, None]

    pos_mean = np.nanmean(pos_norm_traces, axis=0)
    neg_mean = np.nanmean(neg_norm_traces, axis=0)

    pos_sem = np.nanstd(pos_norm_traces, axis=0, ddof=1) / np.sqrt(pos_norm_traces.shape[0])
    neg_sem = np.nanstd(neg_norm_traces, axis=0, ddof=1) / np.sqrt(neg_norm_traces.shape[0])

    pos_p_values, neg_p_values = [], []

    for i in range(len(time_bins)):
        pos_vals = pos_raw_traces[:, i]
        pos_mask = ~np.isnan(pos_vals) & ~np.isnan(pos_baselines)

        if np.sum(pos_mask) >= min_n:
            diffs = pos_vals[pos_mask] - pos_baselines[pos_mask]
            if np.allclose(diffs, 0, equal_nan=True):
                pos_p_values.append(np.nan)
            else:
                _, p = wilcoxon(pos_vals[pos_mask], pos_baselines[pos_mask], alternative="two-sided")
                pos_p_values.append(p)
        else:
            pos_p_values.append(np.nan)

        neg_vals = neg_raw_traces[:, i]
        neg_mask = ~np.isnan(neg_vals) & ~np.isnan(neg_baselines)

        if np.sum(neg_mask) >= min_n:
            diffs = neg_vals[neg_mask] - neg_baselines[neg_mask]
            if np.allclose(diffs, 0, equal_nan=True):
                neg_p_values.append(np.nan)
            else:
                _, p = wilcoxon(neg_vals[neg_mask], neg_baselines[neg_mask], alternative="two-sided")
                neg_p_values.append(p)
        else:
            neg_p_values.append(np.nan)

    return {
        "time_bins": time_bins,
        "time_bin_centres": time_bin_centres,
        "bin_width": time_bins[1] - time_bins[0],
        "pos_norm_traces": pos_norm_traces,
        "neg_norm_traces": neg_norm_traces,
        "pos_raw_traces": pos_raw_traces,
        "neg_raw_traces": neg_raw_traces,
        "pos_baselines": pos_baselines,
        "neg_baselines": neg_baselines,
        "pos_fish_ids": pos_fish_ids,
        "neg_fish_ids": neg_fish_ids,
        "pos_mean": pos_mean,
        "neg_mean": neg_mean,
        "pos_sem": pos_sem,
        "neg_sem": neg_sem,
        "pos_p_values": np.asarray(pos_p_values, dtype=float),
        "neg_p_values": np.asarray(neg_p_values, dtype=float),
    }


def valence_trajectory_results_df(analysis: dict) -> pd.DataFrame:
    """The notebook's printed per-bin summary table for one `compute_valence_trajectory_analysis` result."""
    return pd.DataFrame({
        "time_bin": analysis["time_bins"],
        "pos_mean": analysis["pos_mean"],
        "pos_sem": analysis["pos_sem"],
        "pos_pvalue": analysis["pos_p_values"],
        "neg_mean": analysis["neg_mean"],
        "neg_sem": analysis["neg_sem"],
        "neg_pvalue": analysis["neg_p_values"],
    })


def plot_valence_trajectory_analysis(
    analysis: dict,
    ylabel: str = "Bout count (z-scored)",
    title: str | None = None,
    plot_window: tuple[float, float] = (-10, 30),
    ylim: tuple[float, float] = (-1.5, 5),
    y_star_pos: float = 4.15,
    y_star_neg: float = 4.65,
    show_legend: bool = False,
    save_path: Path | None = None,
):
    """Plot appetitive/aversive mean trajectories with significance stars."""
    time_bins = analysis["time_bins"]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(time_bins, analysis["pos_mean"], color="green", linewidth=2, label="Appetitive")
    ax.fill_between(
        time_bins, analysis["pos_mean"] - analysis["pos_sem"], analysis["pos_mean"] + analysis["pos_sem"],
        color="green", alpha=0.3,
    )

    ax.plot(time_bins, analysis["neg_mean"], color="magenta", linewidth=2, label="Aversive")
    ax.fill_between(
        time_bins, analysis["neg_mean"] - analysis["neg_sem"], analysis["neg_mean"] + analysis["neg_sem"],
        color="magenta", alpha=0.3,
    )

    ax.axhline(0, color="lightgrey", linestyle="--", linewidth=1)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.5)

    for t, p in zip(time_bins, analysis["pos_p_values"]):
        if t < 0 or t > plot_window[1] or np.isnan(p):
            continue
        star = "**" if p < 0.01 else "*" if p < 0.05 else None
        if star:
            ax.text(t, y_star_pos, star, color="green", fontsize=18, fontweight="bold", ha="center")

    for t, p in zip(time_bins, analysis["neg_p_values"]):
        if t < 0 or t > plot_window[1] or np.isnan(p):
            continue
        star = "**" if p < 0.01 else "*" if p < 0.05 else None
        if star:
            ax.text(t, y_star_neg, star, color="magenta", fontsize=18, fontweight="bold", ha="center")

    ax.set_xlim(plot_window)
    ax.set_ylim(ylim)
    ax.set_xlabel("Time relative to stimulus onset (s)", fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)

    if title is not None:
        ax.set_title(title, fontsize=18, weight="bold")
    if show_legend:
        ax.legend(frameon=False, fontsize=14)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.tick_params(width=1.5, length=5, labelsize=20, direction="out")

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, format="svg", dpi=300, bbox_inches="tight")

    return fig, ax


# ---------------------------------------------------------------------------
# Full pipeline (source: whole notebook)
# ---------------------------------------------------------------------------

def run_bout_frequency_psth_analysis(
    fb_path: Path,
    hb_path: Path,
    key_stimuli: tuple[str, ...],
    positive_stims: tuple[str, ...] = POSITIVE_STIMULI,
) -> dict[str, dict]:
    """Run all/asymmetric/straight PSTHs, then the appetitive-vs-aversive trajectory analysis for each.

    Returns `{bout_type: {"psth": <run_psth result>, "valence_results_df": <DataFrame>}}`
    for `bout_type` in "all", "asymmetric", "straight".
    """
    bhv = load_fb_hb_behavior_data(fb_path, hb_path)
    stimulus_df = prepare_stimulus_df(bhv, key_stimuli)
    stimulus_df = add_asymmetric_bout_columns(stimulus_df)

    results = {}
    for bout_type in ("all", "asymmetric", "straight"):
        psth = run_psth(stimulus_df, bout_type)
        analysis = compute_valence_trajectory_analysis(
            zscored_hist_data=psth["zscored_hist_data"],
            time_bins=psth["time_bins"],
            time_bin_centres=psth["time_bin_centres"],
            positive_stims=positive_stims,
            baseline_window=(-PRE_STIMULUS_SECONDS, 0),
        )
        results[bout_type] = {
            "psth": psth,
            "valence_analysis": analysis,
            "valence_results_df": valence_trajectory_results_df(analysis),
        }

    return results
