"""Response latency: derivative-based stimulus-onset detection per fish x
brain area, plus whole-forebrain-vs-whole-hindbrain and per-area-vs-OB
latency comparisons.

Ported from notebooks/neural/response_latency.ipynb; logic unchanged, split
into loading/processing/plotting functions per the /tdd REFACTOR checklist.
Debug-plot I/O (writing a PNG per failed detection) is separated from the
pure detection functions (`process_fish_by_area`/`process_whole_region`),
which now only return data -- the notebook's own version wrote failure PNGs
as a side effect of the same loop.

**Reuses `jenkins_et_al.config.FOREBRAIN_AREA_COLORS`, `config.REFERENCE_AREA`,
and `config.AREA_SHORTHAND`** instead of a second hardcoded copy -- confirmed
identical to this notebook's own `PALETTE`/`REFERENCE_AREA`/
`AREA_SHORTHAND_DICT` (cell 4) for all 12 areas this notebook uses. `config.py`
already documents `REFERENCE_AREA` and `FOREBRAIN_AREA_COLORS` as sourced
(in part) from this notebook. This resolves PORTING_PLAN.md's flagged
question of whether this notebook's hardcoded shorthand dict was deliberate
or just not-yet-unified -- it's the latter; both were transcriptions of the
same external `area_shorthand_dict.json` / shared palette.

**`AREA_ORDER` stays notebook-local** (raw area names, not shorthand codes --
the detection loop groups by raw `area` values from the source dataframe).
Its shorthand order coincidentally matches `lifetime_sparseness.AREA_ORDER`
exactly (`OE, OB, Pal, SubP, POA, PTh, EmT, dHb, vHb, PT, HR, HI`) -- flagged,
not merged, since the two modules need different representations (raw names
here vs. already-shorthand there) and per-notebook independent verification.

**`plot_relative_latency` (cell 35) reuses `plotting.plot_area_comparison_boxplot`**
(the same boxplot function `lifetime_sparseness.py`/
`template_matching_classification.py` share), plus one addition the shared
function doesn't have: a dashed grey `y=0` reference line, meaningful only
for this "latency relative to OB" plot. Two cosmetic differences from the
notebook's own styling are accepted as part of this reuse, per this project's
"figures aren't pixel-exact, only their underlying data is verified exactly"
rule: the shared function's x-tick-label font is hardcoded at 24pt vs. this
notebook's 26pt, and its box/line recoloring walks matplotlib `Line2D`
objects by x-position rather than by a fixed per-box line count -- both
purely cosmetic, confirmed to not affect any plotted value, position, or
significance symbol.

**`plot_forebrain_hindbrain_latency` (cell 26) is kept as its own function**,
not unified with `behaviour.bout_frequency_lmm.plot_paired_fish_by_group` --
both are "paired per-subject dot plot" style, but differ in real ways (single
two-condition panel with a significance bracket here vs. that function's
one-panel-per-group layout with per-group p-value text and pre/post mean
tick marks) -- flagged as a possible future unification candidate, not
forced, consistent with this project's "verify each site independently
first" rule.

**Confirmed dead code, not reproduced**: cell 20's `manual_latency_s = 40 -
STIM_TIME_S` (= 17.0) is computed and displayed (cell 21) but never used --
the actual manual override applied two cells later (cell 22, fish 230811's
whole-hindbrain latency) hardcodes a different, unrelated literal (`8`) with
no traceable arithmetic connection to `manual_latency_s`. Confirmed via
golden capture: fish 230811's hindbrain trace was automatically detected via
the peak-fraction fallback method at `latency_s=0.667` (`status=
detected_fallback`) -- cell 22 overrides this specific case to `latency_s=8,
status="manual"`, presumably after visually inspecting that trace's debug
plot and judging the automated onset wrong. This project's practice on such
discrepancies is to stop and report both values rather than decide
unilaterally which is "right" -- reported to the user during this port;
`FISH_230811_HINDBRAIN_MANUAL_LATENCY_S` below preserves the notebook's own
applied value (`8`, not `17`) bug-for-bug, since that's what the frozen
analysis actually used. `manual_latency_s` itself is not reproduced (dead).

**Made easy to redo, at the user's request**: cells 37-40 are the notebook's
own "manual checking" step -- list failed/fallback detections, then
interactively regenerate one fish x area's debug plot to eyeball whether the
automated onset looks right, and optionally hand-correct it the same way
cell 22 did for fish 230811. `get_failed_detections` (cell 38),
`show_failure_debug_plot` (cell 39-40, using
`jenkins_et_al.plotting.show_in_browser` -- the same pattern
`functional_clustering_concentration_response.py`'s `show_cluster_id_curves`/
`show_silhouette_analysis` use for their own "look at this, then decide"
steps) and `apply_manual_override` (generalizing cell 22's one-off
`df.loc[...] = [...]` into a reusable function, usable on either
`area_latency_df` or `region_latency_df`) turn that workflow into three
callable functions instead of notebook cells with hand-edited variables.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import wilcoxon

from jenkins_et_al.config import AREA_SHORTHAND, FOREBRAIN_AREA_COLORS, REFERENCE_AREA
from jenkins_et_al.plotting import plot_area_comparison_boxplot, show_in_browser

#: Source: cell 4.
KEY_STIMULI = ("fex_1", "kw", "pro_2.5mm", "ade", "cad_2.5mm", "ph4.5", "qui_2.5mm")

#: Raw area names, in display order. Source: cell 4. See module docstring
#: for why this stays notebook-local (raw names) rather than reusing
#: `lifetime_sparseness.AREA_ORDER` (shorthand codes, same order).
AREA_ORDER = (
    "olfactory_epithelium", "olfactory_bulb", "pallium", "subpallium", "preoptic",
    "prethalamus", "eminentia_thalami", "dorsal_habenula", "ventral_habenula",
    "posterior_tuberculum", "rostral_hypothalamus", "intermediate_hypothalamus",
)

FOREBRAIN_FISH = (230713, 230714, 230720, 230727, 230728, 230810, 230811, 230817, 230818, 230919)
HINDBRAIN_FISH = FOREBRAIN_FISH  #: Source: cell 4 -- same 10 animals, two imaging sessions each.

#: Source: cell 4. Frame rate and onset-detection window, all in the
#: already-30s-trimmed trace's own time base (frame 0 = TRIM_START_S).
FPS = 3.0
TRIM_START_S = 30.0
STIM_TIME_S = 53.0 - TRIM_START_S
SEARCH_START_S = 53.0 - TRIM_START_S
SEARCH_END_S = 80.0 - TRIM_START_S
SMOOTHING_SIGMA_FRAMES = 3
DERIVATIVE_THRESHOLD = 0.01
MIN_CONSECUTIVE_DERIVATIVE_FRAMES = 1

#: Extra lead-in trim applied to every mean trace before detection. Source:
#: cells 12/13 (`mean_trace = mean_trace[90:]`), applied identically to both
#: per-area and whole-region traces.
TRIM_LEADIN_FRAMES = 90

#: Source: cell 22. See module docstring's "Confirmed dead code" note --
#: this is the value actually applied, not the unused `manual_latency_s`.
FISH_230811_HINDBRAIN_MANUAL_LATENCY_S = 8.0


# ---------------------------------------------------------------------------
# Loading (source: cell 6)
# ---------------------------------------------------------------------------

def parse_trace(x: object) -> np.ndarray:
    """Parse one trace cell (array, list, or JSON/literal string) into a float array."""
    if isinstance(x, np.ndarray):
        return x.astype(float)
    if isinstance(x, list):
        return np.asarray(x, dtype=float)
    if isinstance(x, str):
        try:
            return np.asarray(json.loads(x), dtype=float)
        except (json.JSONDecodeError, ValueError):
            return np.asarray(ast.literal_eval(x), dtype=float)
    raise ValueError(f"Cannot parse trace of type {type(x)}")


def load_fish_df(file_path: str | Path, fish: int | str, suffix: str | None = None) -> pd.DataFrame:
    """Load one fish's rows from an HDF5 file, trying `fish_id` with and without `suffix`.

    Falls back to a full-file load + filter if the indexed `where` query
    finds nothing (e.g. the file has no `fish_id` index).
    """
    candidates = [str(fish)]
    if suffix is not None:
        candidates.append(f"{fish}{suffix}")

    for candidate in candidates:
        try:
            df = pd.read_hdf(file_path, where=f'fish_id == "{candidate}"')
            if not df.empty:
                return df
        except (KeyError, ValueError, TypeError):
            pass

    df = pd.read_hdf(file_path)
    for candidate in candidates:
        out = df[df["fish_id"].astype(str) == str(candidate)].copy()
        if not out.empty:
            return out

    raise ValueError(f"No data found for fish {fish} in {file_path}")


# ---------------------------------------------------------------------------
# Trace averaging + onset detection (source: cells 6, 8)
# ---------------------------------------------------------------------------

def make_mean_trace(
    fish_df: pd.DataFrame, area: str | None = None, key_stimuli: tuple[str, ...] = KEY_STIMULI,
) -> tuple[np.ndarray | None, str, int]:
    """Mean trace across `key_stimuli` trials, optionally restricted to one `area`.

    Returns `(mean_trace, status, n_traces)`; `mean_trace` is `None` when no
    valid trace data is found (status explains why).
    """
    df = fish_df.copy()

    if area is not None:
        df = df[df["area"] == area].copy()

    df = df[df["stimulus"].isin(key_stimuli)].copy()

    if df.empty:
        return None, "failed_no_key_stimulus_data", 0

    df["trace"] = df["trace_serialised"].apply(parse_trace)
    traces = [t for t in df["trace"].values if len(t) > 0 and np.any(np.isfinite(t))]

    if len(traces) == 0:
        return None, "failed_no_valid_traces", 0

    min_len = min(len(t) for t in traces)
    traces = [t[:min_len] for t in traces]
    mean_trace = np.nanmean(np.stack(traces), axis=0)

    return mean_trace, "ok", len(traces)


def first_sustained_crossing_bool(mask: np.ndarray, min_consecutive: int = 1) -> float:
    """First index where `mask` is `True` for `min_consecutive` consecutive frames, or `nan`."""
    mask = np.asarray(mask, dtype=bool)

    if min_consecutive <= 1:
        idx = np.where(mask)[0]
        return idx[0] if len(idx) > 0 else np.nan

    for i in range(0, len(mask) - min_consecutive + 1):
        if np.all(mask[i:i + min_consecutive]):
            return i

    return np.nan


def fallback_peak_fraction_onset(
    smoothed: np.ndarray, stim_idx: int, search_start_idx: int, search_end_idx: int, fraction: float = 0.30,
) -> tuple[float, float]:
    """Onset as the first frame reaching `fraction` of the way from pre-stim baseline to peak.

    Used when no derivative crossing is found. Returns `(onset_idx, threshold)`;
    `onset_idx` is `nan` if the search window never reaches `threshold`.
    """
    search = smoothed[search_start_idx:search_end_idx]
    baseline_mean = np.nanmean(smoothed[:stim_idx])
    peak = np.nanmax(search)
    threshold = baseline_mean + fraction * (peak - baseline_mean)

    candidates = np.where(search >= threshold)[0]
    if len(candidates) == 0:
        return np.nan, threshold

    return search_start_idx + candidates[0], threshold


def detect_onset_derivative_restricted(
    mean_trace: np.ndarray,
    fps: float = FPS,
    stim_time_s: float = STIM_TIME_S,
    search_start_s: float = SEARCH_START_S,
    search_end_s: float | None = SEARCH_END_S,
    smoothing_sigma_frames: float = SMOOTHING_SIGMA_FRAMES,
    derivative_threshold: float = DERIVATIVE_THRESHOLD,
    min_consecutive_frames: int = MIN_CONSECUTIVE_DERIVATIVE_FRAMES,
) -> dict[str, object]:
    """Onset = first derivative crossing above `derivative_threshold` within the search window.

    Falls back to `fallback_peak_fraction_onset` (10% threshold) when no
    crossing is found. Returns a dict with `smoothed_trace`, `derivative`,
    `onset_idx`, `latency_s`, `stim_idx`, `search_start_idx`,
    `search_end_idx`, `status` (`"detected"`, `"detected_fallback"`, or a
    `"failed_*"` reason).

    Source: cell 8.
    """
    mean_trace = np.asarray(mean_trace, dtype=float)

    if len(mean_trace) == 0 or np.all(~np.isfinite(mean_trace)):
        return {
            "smoothed_trace": mean_trace, "derivative": np.full_like(mean_trace, np.nan, dtype=float),
            "onset_idx": np.nan, "latency_s": np.nan, "stim_idx": np.nan,
            "search_start_idx": np.nan, "search_end_idx": np.nan, "status": "failed_empty_trace",
        }

    stim_idx = int(round(stim_time_s * fps))
    search_start_idx = int(round(search_start_s * fps))
    search_end_idx = len(mean_trace) if search_end_s is None else int(round(search_end_s * fps))

    search_start_idx = max(0, min(search_start_idx, len(mean_trace)))
    search_end_idx = max(search_start_idx, min(search_end_idx, len(mean_trace)))

    smoothed = gaussian_filter1d(mean_trace, sigma=smoothing_sigma_frames, mode="nearest")
    derivative = np.gradient(smoothed)

    if search_end_idx <= search_start_idx:
        return {
            "smoothed_trace": smoothed, "derivative": derivative, "onset_idx": np.nan, "latency_s": np.nan,
            "stim_idx": stim_idx, "search_start_idx": search_start_idx, "search_end_idx": search_end_idx,
            "status": "failed_bad_search_window",
        }

    crossing_mask = derivative[search_start_idx:search_end_idx] > derivative_threshold
    rel_onset = first_sustained_crossing_bool(crossing_mask, min_consecutive=min_consecutive_frames)

    if not np.isfinite(rel_onset):
        fallback_onset, _ = fallback_peak_fraction_onset(
            smoothed, stim_idx, search_start_idx, search_end_idx, fraction=0.10,
        )
        if np.isfinite(fallback_onset):
            onset_idx = int(fallback_onset)
            status = "detected_fallback"
        else:
            onset_idx = np.nan
            status = "failed_no_derivative_crossing"
    else:
        onset_idx = int(search_start_idx + rel_onset)
        status = "detected"

    latency_s = (onset_idx - stim_idx) / fps if np.isfinite(onset_idx) else np.nan

    return {
        "smoothed_trace": smoothed, "derivative": derivative, "onset_idx": onset_idx, "latency_s": latency_s,
        "stim_idx": stim_idx, "search_start_idx": search_start_idx, "search_end_idx": search_end_idx,
        "status": status,
    }


# ---------------------------------------------------------------------------
# Per-fish x area / whole-region processing (source: cell 12)
# ---------------------------------------------------------------------------

def process_fish_by_area(
    fish: int, file_path: str | Path, dataset_name: str, suffix: str | None = None,
    areas: tuple[str, ...] = AREA_ORDER, area_shorthand: dict[str, str] = AREA_SHORTHAND,
    trim_leadin_frames: int = TRIM_LEADIN_FRAMES,
) -> list[dict[str, object]]:
    """Detect onset latency for one fish, independently for each area in `areas`.

    Pure data function -- unlike the notebook's version, does not write debug
    plots as a side effect (see `save_failure_debug_plots` for that, split
    out per the module docstring).
    """
    fish_df = load_fish_df(file_path, fish, suffix=suffix)
    results = []

    for area in areas:
        area_label = area_shorthand.get(area, area)
        mean_trace, trace_status, n_traces = make_mean_trace(fish_df, area=area)

        if mean_trace is None:
            results.append({
                "fish_id": str(fish), "dataset": dataset_name, "area": area, "area_label": area_label,
                "n_traces": n_traces, "onset_idx": np.nan, "latency_s": np.nan, "status": trace_status,
            })
            continue

        detection = detect_onset_derivative_restricted(mean_trace[trim_leadin_frames:])
        results.append({
            "fish_id": str(fish), "dataset": dataset_name, "area": area, "area_label": area_label,
            "n_traces": n_traces, "onset_idx": detection["onset_idx"], "latency_s": detection["latency_s"],
            "status": detection["status"],
        })

    return results


def process_whole_region(
    fish: int, file_path: str | Path, dataset_name: str, suffix: str | None = None,
    trim_leadin_frames: int = TRIM_LEADIN_FRAMES,
) -> dict[str, object]:
    """Detect onset latency for one fish's whole-region (all-area) grand-average trace.

    Source: cell 12 (`process_whole_region`).
    """
    fish_df = load_fish_df(file_path, fish, suffix=suffix)
    mean_trace, trace_status, n_traces = make_mean_trace(fish_df, area=None)

    if mean_trace is None:
        return {
            "fish_id": str(fish), "region": dataset_name, "n_traces": n_traces,
            "onset_idx": np.nan, "latency_s": np.nan, "status": trace_status,
        }

    detection = detect_onset_derivative_restricted(mean_trace[trim_leadin_frames:])
    return {
        "fish_id": str(fish), "region": dataset_name, "n_traces": n_traces,
        "onset_idx": detection["onset_idx"], "latency_s": detection["latency_s"], "status": detection["status"],
    }


def apply_manual_override(
    df: pd.DataFrame, fish_id: str, group_col: str, group_value: str, latency_s: float,
) -> pd.DataFrame:
    """Hand-correct one fish x group row's `latency_s` after visual inspection, marking `status="manual"`.

    `group_col` is `"area"` for `area_latency_df` or `"region"` for
    `region_latency_df`. Generalizes cell 22's one-off
    `region_latency_df.loc[...] = [8, "manual"]` (fish 230811's hindbrain,
    applied via `FISH_230811_HINDBRAIN_MANUAL_LATENCY_S` -- see module
    docstring) into a reusable function for the same workflow on new data.
    """
    df = df.copy()
    mask = (df["fish_id"].astype(str) == str(fish_id)) & (df[group_col] == group_value)
    df.loc[mask, ["latency_s", "status"]] = [latency_s, "manual"]
    return df


# ---------------------------------------------------------------------------
# Forebrain vs hindbrain paired comparison (source: cells 25, 28)
# ---------------------------------------------------------------------------

def compare_forebrain_hindbrain(region_latency_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paired Wilcoxon signed-rank test of whole-forebrain vs. whole-hindbrain latency.

    Only `"detected"`/`"manual"` rows are used -- `"detected_fallback"` rows
    are excluded here (though not from the per-area analysis below), so a
    fish with only a fallback-detected region latency and no manual
    correction is dropped from the pair. Returns `(paired, stats)`.

    Source: cell 25.
    """
    valid_statuses = ["detected", "manual"]
    detected = region_latency_df[region_latency_df["status"].isin(valid_statuses)].copy()
    paired = detected.pivot_table(index="fish_id", columns="region", values="latency_s", aggfunc="first").dropna()

    if {"forebrain", "hindbrain"}.issubset(paired.columns) and len(paired) >= 2:
        stat, p_val = wilcoxon(paired["forebrain"], paired["hindbrain"])
    else:
        stat, p_val = np.nan, np.nan

    stats = pd.DataFrame([{"comparison": "forebrain_vs_hindbrain", "n": len(paired), "statistic": stat, "p_value": p_val}])
    return paired, stats


def compute_fb_hb_difference_stats(paired_region: pd.DataFrame) -> pd.Series:
    """Mean/std/median of hindbrain-minus-forebrain latency across paired fish. Source: cell 28."""
    return (paired_region["hindbrain"] - paired_region["forebrain"]).agg(["mean", "std", "median"])


# ---------------------------------------------------------------------------
# Latency relative to OB (source: cells 30, 33, 36)
# ---------------------------------------------------------------------------

def compute_relative_to_reference(
    area_latency_df: pd.DataFrame, reference_area: str = REFERENCE_AREA, area_shorthand: dict[str, str] = AREA_SHORTHAND,
) -> pd.DataFrame:
    """Per-fish `area latency - reference_area latency`, long-format (`fish_id`, `area`, `relative_latency_s`, `area_label`).

    Only `"detected"`/`"detected_fallback"` rows are used here (unlike
    `compare_forebrain_hindbrain`, fallback detections count). `reference_area`
    itself ends up with `relative_latency_s = 0` for every fish where it was
    detected. Source: cell 30.
    """
    valid_statuses = ["detected", "detected_fallback"]
    detected = area_latency_df[area_latency_df["status"].isin(valid_statuses)].copy()
    wide = detected.pivot_table(index="fish_id", columns="area", values="latency_s", aggfunc="first")

    if reference_area not in wide.columns:
        raise ValueError(f"Reference area {reference_area} not found in detected data.")

    relative = wide.subtract(wide[reference_area], axis=0)
    rel_df = relative.reset_index().melt(id_vars="fish_id", var_name="area", value_name="relative_latency_s")
    rel_df["area_label"] = rel_df["area"].map(lambda x: area_shorthand.get(x, x))
    return rel_df


def wilcoxon_relative_latency(
    rel_df: pd.DataFrame, reference_area: str = REFERENCE_AREA, area_order: tuple[str, ...] = AREA_ORDER,
    area_shorthand: dict[str, str] = AREA_SHORTHAND,
) -> pd.DataFrame:
    """One-sample Wilcoxon signed-rank test of each area's OB-relative latency against zero.

    Source: cell 33.
    """
    rows = []
    for area in area_order:
        if area == reference_area:
            continue

        vals = rel_df.loc[rel_df["area"] == area, "relative_latency_s"].dropna().values
        if len(vals) >= 2 and np.any(vals != 0):
            try:
                stat, p_val = wilcoxon(vals, alternative="two-sided")
            except ValueError:
                stat, p_val = np.nan, np.nan
        else:
            stat, p_val = np.nan, np.nan

        rows.append({
            "area": area, "area_label": area_shorthand.get(area, area), "n": len(vals),
            "statistic": stat, "p_value": p_val,
        })

    return pd.DataFrame(rows)


def p_to_symbol(p: float) -> str:
    """Significance stars for a p-value: `"na"`, `"ns"`, `"*"`, `"**"`, or `"***"`."""
    if not np.isfinite(p):
        return "na"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def summarize_relative_latency(
    rel_df: pd.DataFrame, stats_df: pd.DataFrame, area_order: tuple[str, ...] = AREA_ORDER,
) -> pd.DataFrame:
    """Per-area mean/std OB-relative latency with its Wilcoxon p-value and symbol, in `area_order`.

    **`symbol` here uses its own NaN-unsafe threshold logic, not `p_to_symbol`
    -- confirmed real, not unified.** The notebook's cell 36 recomputes stars
    with an inline `lambda p: "***" if p<0.001 else ... else "ns"` that, unlike
    cell 33's `p_to_symbol` (used for `wilcoxon_relative_latency`'s own
    `stats_df`), has no NaN case: `p < threshold` is `False` for `NaN` at
    every branch, so it silently falls through to `"ns"`. This only differs
    from `p_to_symbol` (which returns `"na"`) for the reference area itself
    (`p_value` is `NaN` there, since it's excluded from the Wilcoxon test) --
    reproduced here bug-for-bug rather than reusing `p_to_symbol`, since the
    frozen notebook genuinely computes this table's stars a second,
    inconsistent way.

    Source: cell 36.
    """
    summary_df = rel_df.groupby(["area", "area_label"])["relative_latency_s"].agg(["mean", "std"]).reset_index()
    summary_df["p_value"] = summary_df["area"].map(dict(zip(stats_df["area"], stats_df["p_value"])))
    summary_df["symbol"] = summary_df["p_value"].apply(
        lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    )
    summary_df["area"] = pd.Categorical(summary_df["area"], categories=area_order, ordered=True)
    return summary_df.sort_values("area")


def get_failed_detections(area_latency_df: pd.DataFrame) -> pd.DataFrame:
    """Rows whose onset detection did not fully succeed (`status != "detected"`, includes fallbacks).

    Source: cell 38.
    """
    return area_latency_df[area_latency_df["status"] != "detected"].copy()


# ---------------------------------------------------------------------------
# Plotting (source: cells 10, 26, 35)
# ---------------------------------------------------------------------------

def plot_debug_trace(
    mean_trace: np.ndarray, detection: dict[str, object], fish_id: int | str, area_label: str,
    dataset_name: str, title_status: str | None = None, fps: float = FPS,
) -> plt.Figure:
    """Trace + derivative plot showing the detection decision (mean, smoothed, stim/search/onset markers).

    Source: cell 10 (`save_debug_plot`), split from its own file-writing --
    see module docstring.
    """
    mean_trace = np.asarray(mean_trace, dtype=float)
    smoothed = detection["smoothed_trace"]
    derivative = detection["derivative"]
    stim_idx = detection["stim_idx"]
    search_start_idx = detection["search_start_idx"]
    search_end_idx = detection["search_end_idx"]
    onset_idx = detection["onset_idx"]
    status = detection["status"] if title_status is None else title_status

    time_s = np.arange(len(mean_trace)) / fps

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    axes[0].plot(time_s, mean_trace, color="lightgrey", linewidth=2, label="Mean trace")
    axes[0].plot(time_s, smoothed, color="black", linewidth=2, label="Smoothed")
    if np.isfinite(stim_idx):
        axes[0].axvline(stim_idx / fps, color="blue", linestyle="--", linewidth=2, label="Expected stim")
    if np.isfinite(search_start_idx):
        axes[0].axvline(search_start_idx / fps, color="purple", linestyle=":", linewidth=2, label="Search start")
    if np.isfinite(search_end_idx) and search_end_idx < len(mean_trace):
        axes[0].axvline(search_end_idx / fps, color="purple", linestyle=":", linewidth=2, label="Search end")
    if np.isfinite(onset_idx):
        axes[0].axvline(onset_idx / fps, color="red", linestyle="--", linewidth=2, label="Detected onset")

    axes[0].set_ylabel("Activity")
    axes[0].set_title(f"{dataset_name} | Fish {fish_id} | {area_label} | {status}")
    axes[0].legend(fontsize=8, frameon=False)

    axes[1].plot(time_s, derivative, color="black", linewidth=1.5)
    axes[1].axhline(DERIVATIVE_THRESHOLD, color="red", linestyle="--", linewidth=1.5)
    if np.isfinite(stim_idx):
        axes[1].axvline(stim_idx / fps, color="blue", linestyle="--", linewidth=2)
    if np.isfinite(search_start_idx):
        axes[1].axvline(search_start_idx / fps, color="purple", linestyle=":", linewidth=2)
    if np.isfinite(onset_idx):
        axes[1].axvline(onset_idx / fps, color="red", linestyle="--", linewidth=2)

    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Derivative")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    return fig


def save_failure_debug_plots(
    area_latency_df: pd.DataFrame, forebrain_path: str | Path, hindbrain_path: str | Path, output_dir: str | Path,
    trim_leadin_frames: int = TRIM_LEADIN_FRAMES,
) -> list[Path]:
    """Regenerate and save a debug plot for every non-`"detected"` row in `area_latency_df`.

    Source: cell 12's `if detection["status"] != "detected": save_debug_plot(...)`
    branch, split out into its own I/O step -- see module docstring.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    for _, row in get_failed_detections(area_latency_df).iterrows():
        file_path = forebrain_path if row["dataset"] == "forebrain" else hindbrain_path
        suffix = "_fb" if row["dataset"] == "forebrain" else "_hb"

        fish_df = load_fish_df(file_path, row["fish_id"], suffix=suffix)
        mean_trace, _, _ = make_mean_trace(fish_df, area=row["area"])
        detection = detect_onset_derivative_restricted(mean_trace[trim_leadin_frames:])

        fig = plot_debug_trace(
            mean_trace[trim_leadin_frames:], detection, row["fish_id"], row["area_label"], row["dataset"],
            title_status=f"FAILED: {detection['status']}",
        )
        safe_area = row["area_label"].replace("/", "_").replace(" ", "_")
        path = output_dir / f"FAILED_{row['dataset']}_{row['fish_id']}_{safe_area}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(path)

    return saved_paths


def show_failure_debug_plot(
    fish_id: int | str, area: str, area_latency_df: pd.DataFrame, forebrain_path: str | Path,
    hindbrain_path: str | Path, trim_leadin_frames: int = TRIM_LEADIN_FRAMES,
) -> Path:
    """Regenerate one fish x area's debug plot and open it in the default web browser.

    The manual-checking step from cells 39-40, generalized: pick a
    `fish_id`/`area` from `get_failed_detections(area_latency_df)`, inspect
    the trace, and decide whether to `apply_manual_override` it.
    """
    row = area_latency_df[
        (area_latency_df["fish_id"].astype(str) == str(fish_id)) & (area_latency_df["area"] == area)
    ].iloc[0]

    file_path = forebrain_path if row["dataset"] == "forebrain" else hindbrain_path
    suffix = "_fb" if row["dataset"] == "forebrain" else "_hb"

    fish_df = load_fish_df(file_path, fish_id, suffix=suffix)
    mean_trace, _, _ = make_mean_trace(fish_df, area=area)
    mean_trace = mean_trace[trim_leadin_frames:]
    detection = detect_onset_derivative_restricted(mean_trace)

    fig = plot_debug_trace(mean_trace, detection, fish_id, row["area_label"], row["dataset"])
    path = show_in_browser(fig)
    plt.close(fig)
    return path


def plot_forebrain_hindbrain_latency(paired_region: pd.DataFrame, stats_df: pd.DataFrame) -> plt.Figure:
    """Paired per-fish dot plot of whole-forebrain vs. whole-hindbrain latency, with a significance bracket.

    See module docstring for why this isn't unified with
    `behaviour.bout_frequency_lmm.plot_paired_fish_by_group`.

    Source: cell 26.
    """
    fb_color, hb_color = "#FFD400", "#EB7F25"
    p_val = stats_df.loc[0, "p_value"]
    sig = p_to_symbol(p_val).replace("na", "ns")

    fig, ax = plt.subplots(figsize=(3, 4.5), facecolor="white")
    x_fb, x_hb = 0, 1

    for _, row in paired_region.iterrows():
        ax.plot([x_fb, x_hb], [row["forebrain"], row["hindbrain"]], color="black", linewidth=0.8, alpha=0.7, zorder=1)
        ax.scatter(x_fb, row["forebrain"], color=fb_color, s=50, alpha=0.7, zorder=2)
        ax.scatter(x_hb, row["hindbrain"], color=hb_color, s=50, alpha=0.7, zorder=2)

    y_max = np.nanmax(paired_region[["forebrain", "hindbrain"]].values)
    y_bar, h = y_max + 0.5, 0.25

    ax.plot([x_fb, x_fb, x_hb, x_hb], [y_bar, y_bar + h, y_bar + h, y_bar], color="black", linewidth=1.5)
    ax.text(0.5, y_bar + h + 0.05, sig, ha="center", va="bottom", fontsize=18, weight="bold")

    ax.set_xticks([x_fb, x_hb])
    ax.set_xticklabels(["fb", "hb"], fontsize=22)
    ax.get_xticklabels()[0].set_color(fb_color)
    ax.get_xticklabels()[1].set_color(hb_color)

    ax.set_ylabel("Response latency (s)", fontsize=25)
    ax.tick_params(axis="y", labelsize=25, width=2)
    ax.tick_params(axis="x", width=2, labelsize=25)
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylim(0, y_bar + h + 0.7)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)

    plt.tight_layout()
    return fig


def plot_relative_latency(
    rel_df: pd.DataFrame, stats_df: pd.DataFrame, area_order: tuple[str, ...] = AREA_ORDER,
    reference_area: str = REFERENCE_AREA, area_shorthand: dict[str, str] = AREA_SHORTHAND,
    palette: dict[str, str] = FOREBRAIN_AREA_COLORS,
) -> plt.Figure:
    """OB-relative latency boxplot per area, with Wilcoxon significance symbols.

    Thin wrapper around `plotting.plot_area_comparison_boxplot` plus a
    `y=0` reference line the shared function doesn't draw -- see module
    docstring for this reuse and its two cosmetic deviations.

    Source: cell 35.
    """
    label_order = [area_shorthand.get(a, a) for a in area_order if a in rel_df["area"].unique()]
    reference_label = area_shorthand.get(reference_area, reference_area)

    plot_data = rel_df.copy()
    plot_data["area"] = plot_data["area_label"]

    pvals = dict(zip(stats_df["area_label"], stats_df["p_value"]))

    fig = plot_area_comparison_boxplot(
        plot_data, "relative_latency_s", tuple(label_order), tuple(label_order), pvals, palette,
        reference_area=reference_label, ylabel="Latency relative to OB (s)",
        ylim=(-1.9, 12), significance_y=11.5,
    )
    fig.axes[0].axhline(0, color="grey", linestyle="--", linewidth=1.5)
    plt.tight_layout()
    return fig
