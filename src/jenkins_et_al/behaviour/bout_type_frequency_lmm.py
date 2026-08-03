"""Asymmetric/straight/all-bout frequency analysis, split by bout type.

Ported from notebooks/behaviour_head_fixed/bout_type_frequency_LMM.ipynb;
logic is unchanged from the notebook (with one bug fix, see below), only
moved into a module, split into loading/processing/stats/plotting functions,
and given a docstring/type-hint pass per the /tdd REFACTOR checklist.

Reuses `jenkins_et_al.behaviour.bout_frequency_lmm`'s already-verified
`load_fb_hb_behavior_data`, `prepare_stimulus_df`, `calculate_all_bout_frequency`,
`stimulus_to_valence`, and `add_valence_column` rather than redefining them --
the notebook's own copies are identical in behaviour. This notebook's mixed
models unconditionally add a `session` variance-component term (unlike
`bout_frequency_lmm.run_mixed_models`, which only adds it when `session` has
a non-null value), and its `classify_asymm_bouts` skips bouts with fewer than
2 frames rather than fewer than 1 (unlike `bout_frequency_PSTH.ipynb`'s
version) -- both kept as separate, notebook-specific functions rather than
unified with the other ports, per the survey's "don't merge until verified"
guidance.

**Bug fixed, not preserved**: the notebook's straight-bout valence Wilcoxon
cell (cell "In[99]" in the nbconvert export) melts columns named
`pre_stim_straight_freq`/`post_stim_straight_freq`, which are never created
anywhere in the notebook -- only `straight_bout_freq_prestim`/
`straight_bout_freq_poststim` exist (added earlier, cell "In[94]"). Run fresh
top-to-bottom, this raises `KeyError: "The following id_vars or value_vars
are not present in the DataFrame: ['pre_stim_straight_freq',
'post_stim_straight_freq']"` -- the notebook cannot complete a clean run as
written. `run_straight_bout_valence_wilcoxon` below uses the columns that
actually exist (`straight_bout_freq_prestim`/`straight_bout_freq_poststim`),
which is the only data those undefined names could plausibly have meant.
Flagged here and in PORTING_PLAN.md rather than silently ported bug-for-bug,
since bug-for-bug would mean this analysis cannot run at all.

**Not fixed, only noted**: the notebook's active `key_stimuli` selection
(first uncommented line) is the 7 HIGH-concentration stimuli, but the CSV
filenames it saves for the asymmetric/straight per-stimulus sections say
"Intermediate_concs"/"Low_conc" -- a stale filename from an earlier run with
a different `key_stimuli`, not a data bug. This module's callers choose their
own output filenames, so the mismatch doesn't carry over; noted here only so
a golden re-run isn't mistaken for the wrong concentration tier.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import circmean, wilcoxon

from jenkins_et_al.behaviour.bout_frequency_lmm import (
    add_valence_column,
    calculate_all_bout_frequency,
    load_fb_hb_behavior_data,
    prepare_stimulus_df,
    stimulus_to_valence,  # noqa: F401  (re-exported for callers)
)

#: Source: cell (nbconvert `In[31]`).
SAMPLING_FREQ = 60.85
STIMULUS_ONSET_FRAME = 3225
WINDOW_SEC = 20

#: This notebook's own valence grouping -- only the 7 high-concentration
#: stimuli, unlike `bout_frequency_lmm.POSITIVE_STIMULI`/`NEGATIVE_STIMULI`
#: which also cover the intermediate/low tiers. Source: cells `In[99]`/`In[100]`.
POSITIVE_STIMULI_HIGH = ("fex1", "kw", "pro_2.5mm")
NEGATIVE_STIMULI_HIGH = ("ade", "cad_2.5mm", "ph4.5", "qui_2.5mm")


# ---------------------------------------------------------------------------
# Asymmetric-bout classification (source: cell `In[32]`)
# ---------------------------------------------------------------------------

def classify_asymm_bouts(
    row: pd.Series,
    left_threshold: float = -0.066,
    right_threshold: float = 0.066,
    sampling_rate: float = SAMPLING_FREQ,
    duration_ms: float = 70,
) -> pd.Series:
    """Classify each bout in `row` as left/right/straight from its early tail-angle circular mean.

    Bouts shorter than 2 frames are skipped entirely (not counted as
    straight) -- a difference from `bout_frequency_PSTH.ipynb`'s otherwise
    near-identical classifier, which only skips bouts of length 0.
    """
    asymmetric_flags = []
    directions = []

    bouts = row["bout_indx"]
    tail_angle = row["angles"]
    duration_frames = int((duration_ms / 1000) * sampling_rate)

    for bout in bouts:
        if len(bout) < 2:
            continue

        bout_start = bout[0]
        bout_duration = min(duration_frames, len(bout))
        angles = np.array(tail_angle[bout_start:bout_start + bout_duration])

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


def add_asymmetric_bout_columns(
    stimulus_df: pd.DataFrame,
    left_threshold: float = -0.06,
    right_threshold: float = 0.06,
) -> pd.DataFrame:
    """Add `asymmetric_bout`/`bout_direction` columns via `classify_asymm_bouts`. Returns a copy."""
    stimulus_df = stimulus_df.copy()
    stimulus_df[["asymmetric_bout", "bout_direction"]] = stimulus_df.apply(
        lambda row: classify_asymm_bouts(row, left_threshold=left_threshold, right_threshold=right_threshold),
        axis=1,
    )
    return stimulus_df


# ---------------------------------------------------------------------------
# Asymmetric-bout frequency (source: cell `In[34]`)
# ---------------------------------------------------------------------------

def calculate_asymmetric_bout_frequency(
    bouts: list[list[int]],
    asymmetric_flags: list[bool],
    stimulus_onset_frame: int,
    window_frames: int,
    sampling_freq: float,
) -> tuple[float, float]:
    """(pre-stimulus, post-stimulus) asymmetric-bout frequency (bouts/s) for one trial."""
    asymmetric_bouts = [b for b, is_asym in zip(bouts, asymmetric_flags) if is_asym]

    pre_stim_bouts = [b for b in asymmetric_bouts if stimulus_onset_frame - window_frames <= b[0] < stimulus_onset_frame]
    post_stim_bouts = [b for b in asymmetric_bouts if stimulus_onset_frame <= b[0] < stimulus_onset_frame + window_frames]

    pre_stim_duration = window_frames / sampling_freq
    post_stim_duration = window_frames / sampling_freq

    pre_stim_asym_freq = len(pre_stim_bouts) / pre_stim_duration if pre_stim_duration > 0 else 0
    post_stim_asym_freq = len(post_stim_bouts) / post_stim_duration if post_stim_duration > 0 else 0

    return pre_stim_asym_freq, post_stim_asym_freq


def add_asymmetric_bout_frequency_columns(
    stimulus_df: pd.DataFrame,
    stimulus_onset_frame: int = STIMULUS_ONSET_FRAME,
    window_sec: int = WINDOW_SEC,
    sampling_freq: float = SAMPLING_FREQ,
) -> pd.DataFrame:
    """Add `asym_bout_freq_prestim`/`asym_bout_freq_poststim`. Returns a copy; unconditional (no presence check)."""
    stimulus_df = stimulus_df.copy()
    window_frames = int(window_sec * sampling_freq)

    pre_vals, post_vals = [], []
    for _, row in stimulus_df.iterrows():
        pre_freq, post_freq = calculate_asymmetric_bout_frequency(
            bouts=row["bout_indx"],
            asymmetric_flags=row["asymmetric_bout"],
            stimulus_onset_frame=stimulus_onset_frame,
            window_frames=window_frames,
            sampling_freq=sampling_freq,
        )
        pre_vals.append(pre_freq)
        post_vals.append(post_freq)

    stimulus_df["asym_bout_freq_prestim"] = pre_vals
    stimulus_df["asym_bout_freq_poststim"] = post_vals

    return stimulus_df


# ---------------------------------------------------------------------------
# Straight-bout frequency (source: cell `In[94]`)
# ---------------------------------------------------------------------------

def calculate_straight_bout_frequency(
    bouts: list[list[int]],
    bout_directions: list[str],
    stimulus_onset_frame: int,
    window_frames: int,
    sampling_freq: float,
) -> tuple[float, float]:
    """(pre-stimulus, post-stimulus) straight-bout frequency (bouts/s) for one trial."""
    straight_bouts = [b for b, direction in zip(bouts, bout_directions) if direction == "straight"]

    pre_stim_bouts = [b for b in straight_bouts if stimulus_onset_frame - window_frames <= b[0] < stimulus_onset_frame]
    post_stim_bouts = [b for b in straight_bouts if stimulus_onset_frame <= b[0] < stimulus_onset_frame + window_frames]

    pre_stim_duration = window_frames / sampling_freq
    post_stim_duration = window_frames / sampling_freq

    pre_stim_straight_freq = len(pre_stim_bouts) / pre_stim_duration if pre_stim_duration > 0 else 0
    post_stim_straight_freq = len(post_stim_bouts) / post_stim_duration if post_stim_duration > 0 else 0

    return pre_stim_straight_freq, post_stim_straight_freq


def add_straight_bout_frequency_columns(
    stimulus_df: pd.DataFrame,
    stimulus_onset_frame: int = STIMULUS_ONSET_FRAME,
    window_sec: int = WINDOW_SEC,
    sampling_freq: float = SAMPLING_FREQ,
) -> pd.DataFrame:
    """Add `straight_bout_freq_prestim`/`straight_bout_freq_poststim`. Returns a copy."""
    stimulus_df = stimulus_df.copy()
    window_frames = int(window_sec * sampling_freq)

    pre_vals, post_vals = [], []
    for _, row in stimulus_df.iterrows():
        pre_rate, post_rate = calculate_straight_bout_frequency(
            bouts=row["bout_indx"],
            bout_directions=row["bout_direction"],
            stimulus_onset_frame=stimulus_onset_frame,
            window_frames=window_frames,
            sampling_freq=sampling_freq,
        )
        pre_vals.append(pre_rate)
        post_vals.append(post_rate)

    stimulus_df["straight_bout_freq_prestim"] = pre_vals
    stimulus_df["straight_bout_freq_poststim"] = post_vals

    return stimulus_df


# ---------------------------------------------------------------------------
# Long-format reshaping (source: cells `In[36]`, `In[95]`, `In[145]`)
# ---------------------------------------------------------------------------

def build_bout_type_long_df(
    stimulus_df: pd.DataFrame,
    prestim_col: str,
    poststim_col: str,
    value_name: str,
    include_session: bool = True,
) -> pd.DataFrame:
    """Melt a pre/post frequency column pair into long (fish x stimulus x condition) form."""
    id_vars = ["fish_id_clean", "session", "stimulus", "trial_number"] if include_session else [
        "fish_id_clean", "stimulus", "trial_number"
    ]

    df_long = stimulus_df.melt(
        id_vars=id_vars,
        value_vars=[prestim_col, poststim_col],
        var_name="condition",
        value_name=value_name,
    )
    df_long["condition"] = df_long["condition"].replace({prestim_col: "Prestim", poststim_col: "Poststim"})
    df_long["condition"] = pd.Categorical(df_long["condition"], categories=["Prestim", "Poststim"])

    return df_long


# ---------------------------------------------------------------------------
# Mixed-effects model with an unconditional session term (source: cells
# `In[36]`, `In[95]`, `In[146]`)
# ---------------------------------------------------------------------------

def run_mixed_models_with_session(df: pd.DataFrame, value_col: str, group_col: str = "stimulus") -> pd.DataFrame:
    """Fit one `<value_col> ~ condition` mixed model per `group_col` value.

    Unlike `bout_frequency_lmm.run_mixed_models`, the `session`
    variance-component term is always included, regardless of whether
    `session` actually has more than one distinct value in a given group.
    """
    results = []

    for group_name in df[group_col].unique():
        stim_df = df[df[group_col] == group_name]

        model = smf.mixedlm(
            f"{value_col} ~ condition",
            stim_df,
            groups=stim_df["fish_id_clean"],
            vc_formula={"session": "0 + C(session)"},
        )
        fit = model.fit()

        p_value = fit.pvalues.get("condition[T.Poststim]", None)

        results.append({
            group_col: group_name,
            "n_fish": stim_df["fish_id_clean"].nunique(),
            "p_value": p_value,
        })

    return pd.DataFrame(results)


def summarize_fish_by_stimulus(df_long: pd.DataFrame, value_col: str, stimulus_labels: dict[str, str]) -> pd.DataFrame:
    """Per-fish-per-stimulus-per-condition mean/std/count, with display-label stimulus names."""
    fish_summary_df = (
        df_long.groupby(["fish_id_clean", "stimulus", "condition"], observed=False)[value_col]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "fish_mean", "std": "fish_std", "count": "n_observations"})
    )
    fish_summary_df["stimulus"] = fish_summary_df["stimulus"].replace(stimulus_labels)
    return fish_summary_df


def relabel_mixed_results(mixed_results: pd.DataFrame, stimulus_labels: dict[str, str]) -> pd.DataFrame:
    """Return a copy of `mixed_results` with `stimulus` mapped to its display label."""
    mixed_results_df = mixed_results.copy()
    mixed_results_df["stimulus"] = mixed_results_df["stimulus"].replace(stimulus_labels)
    return mixed_results_df


# ---------------------------------------------------------------------------
# Valence Wilcoxon tests (source: cells `In[99]` (bug-fixed), `In[100]`)
# ---------------------------------------------------------------------------

def run_straight_bout_valence_wilcoxon(
    stimulus_df: pd.DataFrame,
    positive_stims: tuple[str, ...] = POSITIVE_STIMULI_HIGH,
    negative_stims: tuple[str, ...] = NEGATIVE_STIMULI_HIGH,
) -> pd.DataFrame:
    """Paired Wilcoxon signed-rank test (pre vs. post) of straight-bout frequency, per valence.

    Uses `straight_bout_freq_prestim`/`straight_bout_freq_poststim` -- see
    module docstring for why (the notebook's own cell references undefined
    columns and cannot run as written).
    """
    df_long = build_bout_type_long_df(
        stimulus_df, "straight_bout_freq_prestim", "straight_bout_freq_poststim",
        value_name="straight_bout_frequency", include_session=False,
    )
    df_long = add_valence_column(df_long, positive_stims, negative_stims)

    fish_plot_df = (
        df_long.groupby(["fish_id_clean", "valence", "condition"], as_index=False)["straight_bout_frequency"].mean()
    )

    results = []
    for valence, g in fish_plot_df.groupby("valence"):
        paired = g.pivot_table(
            index="fish_id_clean", columns="condition", values="straight_bout_frequency", aggfunc="mean"
        ).dropna()

        if paired.empty:
            continue

        _, p = wilcoxon(paired["Prestim"], paired["Poststim"])

        results.append({
            "valence": valence,
            "n_fish": paired.index.nunique(),
            "wilcoxon_p": p,
            "prestim_mean": paired["Prestim"].mean(),
            "prestim_std": paired["Prestim"].std(ddof=1),
            "poststim_mean": paired["Poststim"].mean(),
            "poststim_std": paired["Poststim"].std(ddof=1),
        })

    return pd.DataFrame(results)


def run_asymmetric_bout_valence_wilcoxon(
    stimulus_df: pd.DataFrame,
    positive_stims: tuple[str, ...] = POSITIVE_STIMULI_HIGH,
    negative_stims: tuple[str, ...] = NEGATIVE_STIMULI_HIGH,
) -> pd.DataFrame:
    """Paired Wilcoxon signed-rank test (pre vs. post) of asymmetric-bout frequency, per valence."""
    df_long = build_bout_type_long_df(
        stimulus_df, "asym_bout_freq_prestim", "asym_bout_freq_poststim",
        value_name="asym_bout_frequency", include_session=False,
    )
    df_long = add_valence_column(df_long, positive_stims, negative_stims)

    fish_df = (
        df_long.groupby(["fish_id_clean", "valence", "condition"], as_index=False)["asym_bout_frequency"].mean()
    )

    results = []
    for valence, g in fish_df.groupby("valence"):
        paired = g.pivot_table(
            index="fish_id_clean", columns="condition", values="asym_bout_frequency"
        ).dropna()

        _, p = wilcoxon(paired["Prestim"], paired["Poststim"])

        results.append({"valence": valence, "n_fish": paired.index.nunique(), "wilcoxon_p": p})

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Full pipeline (source: whole notebook)
# ---------------------------------------------------------------------------

def run_bout_type_frequency_analysis(
    fb_path: Path,
    hb_path: Path,
    key_stimuli: tuple[str, ...],
    stimulus_labels: dict[str, str],
) -> dict[str, pd.DataFrame]:
    """Run the notebook's asymmetric/straight/all-bout-frequency LMM + valence-Wilcoxon analyses.

    Returns the seven result tables (the four the notebook itself saves to
    CSV, plus the three it only prints/plots): `asym_mixed_results`,
    `asym_fish_summary`, `straight_mixed_results`, `straight_fish_summary`,
    `straight_valence_wilcoxon`, `asym_valence_wilcoxon`,
    `all_bout_mixed_results`.
    """
    bhv = load_fb_hb_behavior_data(fb_path, hb_path)
    stimulus_df = prepare_stimulus_df(bhv, key_stimuli)
    stimulus_df = add_asymmetric_bout_columns(stimulus_df)

    # Asymmetric-bout frequency
    stimulus_df = add_asymmetric_bout_frequency_columns(stimulus_df)
    asym_long = build_bout_type_long_df(
        stimulus_df, "asym_bout_freq_prestim", "asym_bout_freq_poststim", value_name="asym_bout_frequency"
    )
    asym_mixed_results = run_mixed_models_with_session(asym_long, value_col="asym_bout_frequency")
    asym_fish_summary = summarize_fish_by_stimulus(asym_long, "asym_bout_frequency", stimulus_labels)
    asym_mixed_results_labeled = relabel_mixed_results(asym_mixed_results, stimulus_labels)

    # Straight-bout frequency
    stimulus_df = add_straight_bout_frequency_columns(stimulus_df)
    straight_long = build_bout_type_long_df(
        stimulus_df, "straight_bout_freq_prestim", "straight_bout_freq_poststim", value_name="straight_bout_frequency"
    )
    straight_mixed_results = run_mixed_models_with_session(straight_long, value_col="straight_bout_frequency")
    straight_fish_summary = summarize_fish_by_stimulus(straight_long, "straight_bout_frequency", stimulus_labels)
    straight_mixed_results_labeled = relabel_mixed_results(straight_mixed_results, stimulus_labels)

    # Valence Wilcoxon tests
    straight_valence_wilcoxon = run_straight_bout_valence_wilcoxon(stimulus_df)
    asym_valence_wilcoxon = run_asymmetric_bout_valence_wilcoxon(stimulus_df)

    # All-bout frequency
    stimulus_df = add_all_bout_frequency_columns_unconditional(stimulus_df)
    all_bout_long = build_bout_type_long_df(
        stimulus_df, "all_bout_freq_prestim", "all_bout_freq_poststim", value_name="bout_frequency"
    )
    all_bout_mixed_results = run_mixed_models_with_session(all_bout_long, value_col="bout_frequency")

    return {
        "asym_mixed_results": asym_mixed_results_labeled,
        "asym_fish_summary": asym_fish_summary,
        "straight_mixed_results": straight_mixed_results_labeled,
        "straight_fish_summary": straight_fish_summary,
        "straight_valence_wilcoxon": straight_valence_wilcoxon,
        "asym_valence_wilcoxon": asym_valence_wilcoxon,
        "all_bout_mixed_results": all_bout_mixed_results,
    }


def add_all_bout_frequency_columns_unconditional(
    stimulus_df: pd.DataFrame,
    stimulus_onset_frame: int = STIMULUS_ONSET_FRAME,
    window_sec: int = WINDOW_SEC,
    sampling_freq: float = SAMPLING_FREQ,
) -> pd.DataFrame:
    """Add `all_bout_freq_prestim`/`all_bout_freq_poststim` unconditionally (no presence check).

    Uses `bout_frequency_lmm.calculate_all_bout_frequency` (identical logic
    to this notebook's own copy). Source: cells `In[143]`-`In[144]`.
    """
    stimulus_df = stimulus_df.copy()
    window_frames = int(window_sec * sampling_freq)

    pre_vals, post_vals = [], []
    for _, row in stimulus_df.iterrows():
        pre, post = calculate_all_bout_frequency(
            bouts=row["bout_indx"],
            stimulus_onset_frame=stimulus_onset_frame,
            window_frames=window_frames,
            sampling_freq=sampling_freq,
        )
        pre_vals.append(pre)
        post_vals.append(post)

    stimulus_df["all_bout_freq_prestim"] = pre_vals
    stimulus_df["all_bout_freq_poststim"] = post_vals

    return stimulus_df
