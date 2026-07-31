"""Free-swim behavioural metrics: preference, speed, dispersion, and bout-angle.

Ported from notebooks/behaviour_free_swim/free_swim_bhv_metrics.ipynb; logic
is unchanged from the notebook, only moved into a module, grouped by
loading / per-fish metric / per-experiment aggregation, and given a
docstring/type-hint pass per the /tdd REFACTOR checklist. Each function's
docstring traces back to its originating notebook cell id.

The notebook computes two output tables from the same underlying per-fish
tracking data (loaded from `jenkins_et_al.preprocessing.trex` CSVs):

- an experiment-level table (one row per experiment x condition, metrics
  averaged over fish), built by `build_experiment_results_table` +
  `build_experiment_angle_results_table` (merged by the caller, as the
  notebook does)
- a fish-level table (one row per fish), built by `build_fish_results_table`

Reading the raw tracking data is split out into `load_stimulus_data`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

# ---------------------------------------------------------------------------
# Constants (source: notebooks/behaviour_free_swim/free_swim_bhv_metrics.ipynb,
# cell 19aaa8eb)
# ---------------------------------------------------------------------------

#: The 7 stimuli this notebook summarizes, as subfolder names under the
#: TRex output tree.
DEFAULT_STIMULI = (
    "fex_1_10", "kin_water", "proline_100mM", "adenosine_2.5mM",
    "cadaverine_25mM", "HCl_56mM", "quinine_100mM",
)

#: A fish must travel at least this far (cm) over the trial to be included.
MOVEMENT_THRESHOLD_CM = 60.0

#: A fish must be tracked for at least this fraction (%) of the trial to be
#: included.
TRACKING_THRESHOLD_PCT = 90.0

#: Video frame rate (Hz), used to convert frame counts to seconds.
#: Source: cells e7533e4d, de08a4b2, 0f5f2ddf.
DEFAULT_FRAME_RATE = 30

#: Spatial zone boundaries (cm from the left wall) used throughout: "half"
#: is the half of the arena used for the preference index; "quarter" is
#: half of that. Source: cells f810ecc4, de08a4b2, b21aa197.
ZONE_HALF_X = 7.0
ZONE_QUARTER_X = 3.5

#: Speed above which a frame is excluded from speed/dispersion/center-shift
#: zone metrics (cm/s). Source: cells 961a9d46, 2dbe1043, f810ecc4.
DEFAULT_SPEED_THRESHOLD = 3.0

#: Spline smoothing factor for `smooth_positions`. Source: cell 2dbe1043.
DEFAULT_SMOOTHING_FACTOR = 0.5

#: Default bout-detection parameters. Source: cell b755f281.
DEFAULT_BOUT_THRESHOLD = 0.5
DEFAULT_BOUT_MIN_DURATION = 1
DEFAULT_BOUT_EXPAND = 1
DEFAULT_BOUT_GAP_TOLERANCE = 2
DEFAULT_BOUT_MAX_SPEED = 3.0


# ---------------------------------------------------------------------------
# Loading (source: cells 19aaa8eb, b2940f9f)
# ---------------------------------------------------------------------------

def load_dataframes_from_folder(folder: Path) -> list[pd.DataFrame]:
    """Load every per-fish tracking CSV in `folder` into a DataFrame."""
    dataframes = []

    for file in folder.iterdir():
        if file.suffix == ".csv":
            dataframes.append(pd.read_csv(file))

    return dataframes


def total_distance_travelled(df: pd.DataFrame) -> float:
    """Total path length (cm) over `df`'s tracked (mask == True) frames."""
    dx = df["x"][df["mask"]].diff()
    dy = df["y"][df["mask"]].diff()
    return np.sqrt(dx**2 + dy**2).sum()


def is_fish_significantly_moving(
    df: pd.DataFrame,
    movement_threshold: float = MOVEMENT_THRESHOLD_CM,
    tracking_threshold: float = TRACKING_THRESHOLD_PCT,
) -> bool:
    """Whether a fish meets the tracking-coverage and movement thresholds."""
    total_frames = len(df)
    tracked_frames = len(df[df["mask"]])

    tracking_percentage = (tracked_frames / total_frames) * 100
    if tracking_percentage < tracking_threshold:
        return False

    valid_data = df[df["mask"]]
    return total_distance_travelled(valid_data) > movement_threshold


def load_stimulus_data(
    root_dir: Path,
    stimuli: Iterable[str] = DEFAULT_STIMULI,
    movement_threshold: float = MOVEMENT_THRESHOLD_CM,
    tracking_threshold: float = TRACKING_THRESHOLD_PCT,
) -> dict[str, dict[str, dict[str, list[pd.DataFrame]]]]:
    """Load and filter per-fish tracking data for every stimulus/condition/experiment.

    `root_dir` is the TRex output tree root (one subfolder per stimulus, each
    with `stim`/`control` subfolders of per-experiment folders of per-fish
    CSVs -- see `jenkins_et_al.preprocessing.trex`).

    Returns `{stimulus: {"stim"|"control": {experiment_name: [fish_df, ...]}}}`,
    keeping only fish that pass `is_fish_significantly_moving`.
    """
    stimulus_data: dict[str, dict[str, dict[str, list[pd.DataFrame]]]] = {}

    for stimulus in stimuli:
        parent_dir = root_dir / stimulus
        stimulus_data[stimulus] = {"stim": {}, "control": {}}

        for condition in ["stim", "control"]:
            condition_dir = parent_dir / condition

            for exp_folder in condition_dir.iterdir():
                if not exp_folder.is_dir():
                    continue

                exp_name = exp_folder.name
                fish_dfs = load_dataframes_from_folder(exp_folder)

                filtered_fish = [
                    fish_df for fish_df in fish_dfs
                    if is_fish_significantly_moving(
                        fish_df, movement_threshold, tracking_threshold
                    )
                ]

                stimulus_data[stimulus][condition][exp_name] = filtered_fish

    return stimulus_data


# ---------------------------------------------------------------------------
# Preference index (source: cell b22444b1)
# ---------------------------------------------------------------------------

def time_spent_in_left_half_last_half(
    df: pd.DataFrame, frame_rate: int, x_limit: float = ZONE_HALF_X
) -> tuple[float, float]:
    """(time in left half, total valid time), in seconds, over the trial's last half."""
    start_index = len(df) // 2
    last_half_df = df.iloc[start_index:]

    valid_data = last_half_df[last_half_df["mask"]]

    time_in_left_half = len(valid_data[valid_data["x"] < x_limit]) / frame_rate
    total_valid_time = len(valid_data) / frame_rate

    return time_in_left_half, total_valid_time


def calculate_fish_preference_indices(
    dataframes: list[pd.DataFrame], frame_rate: int
) -> list[float]:
    """Per-fish % of the trial's last half spent in the left half of the arena."""
    fish_preferences = []

    for df in dataframes:
        left_time, total_time = time_spent_in_left_half_last_half(df, frame_rate)
        pref = (left_time / total_time) * 100 if total_time > 0 else np.nan
        fish_preferences.append(pref)

    return fish_preferences


def get_pair_key(exp_name: str) -> str:
    """The experiment-pair identifier (e.g. "231221_G1") an experiment name starts with."""
    parts = exp_name.split("_")
    return "_".join(parts[:2])


# ---------------------------------------------------------------------------
# Speed (source: cell 961a9d46)
# ---------------------------------------------------------------------------

def average_speed_in_zone_last_half(
    df: pd.DataFrame, x_limit: float, speed_threshold: float = DEFAULT_SPEED_THRESHOLD
) -> float:
    """Mean speed (mm/s) in the trial's last half, within the zone x < `x_limit`."""
    start_index = len(df) // 2
    last_half_df = df.iloc[start_index:]

    valid_data = last_half_df[last_half_df["mask"]]

    filtered = valid_data[
        (valid_data["x"] < x_limit) &
        (valid_data["speed"] < speed_threshold)
    ]

    if filtered.empty:
        return np.nan

    return filtered["speed"].mean() * 10  # convert cm/s -> mm/s


def compute_avg_speeds_zone(
    dataframes: list[pd.DataFrame],
    x_limit: float,
    speed_threshold: float = DEFAULT_SPEED_THRESHOLD,
) -> list[float]:
    """Per-fish `average_speed_in_zone_last_half`."""
    return [
        average_speed_in_zone_last_half(df, x_limit, speed_threshold)
        for df in dataframes
    ]


# ---------------------------------------------------------------------------
# Dispersion and center shift (source: cell 2dbe1043)
# ---------------------------------------------------------------------------

def compute_minimum_enclosing_circle(
    points: np.ndarray,
) -> tuple[float, tuple[float, float]]:
    """Minimum enclosing circle (radius, (center_x, center_y)) for 2D `points`."""
    if len(points) < 2:
        return np.nan, (np.nan, np.nan)

    points = np.asarray(points, dtype=np.float32)
    (cx, cy), radius = cv2.minEnclosingCircle(points)
    return radius, (cx, cy)


def smooth_positions(
    df: pd.DataFrame,
    time_column: str = "time",
    x_column: str = "x",
    y_column: str = "y",
    smoothing_factor: float = DEFAULT_SMOOTHING_FACTOR,
) -> pd.DataFrame:
    """Smooth `x_column`/`y_column` with a cubic spline; returns a copy of `df`."""
    df = df.copy()

    times = df[time_column].values
    x_vals = df[x_column].values
    y_vals = df[y_column].values

    if len(df) < 4:
        return df

    spline_x = UnivariateSpline(times, x_vals, s=smoothing_factor * len(times))
    spline_y = UnivariateSpline(times, y_vals, s=smoothing_factor * len(times))

    df[x_column] = spline_x(times)
    df[y_column] = spline_y(times)

    return df


def prepare_zone_data(
    df: pd.DataFrame,
    x_limit: float,
    time_column: str = "time",
    x_column: str = "x",
    y_column: str = "y",
    mask_column: str = "mask",
    speed_column: str = "speed",
    speed_threshold: float = DEFAULT_SPEED_THRESHOLD,
    smoothing_factor: float = DEFAULT_SMOOTHING_FACTOR,
) -> pd.DataFrame:
    """Filter to the trial's last half, valid+slow frames, and zone x < `x_limit`, then smooth.

    Returns an empty DataFrame if fewer than 5 rows remain after filtering
    (before or after smoothing).
    """
    df = df.copy()

    midpoint = len(df) // 2
    df = df.iloc[midpoint:].copy()

    df = df[
        (df[mask_column]) &
        (df[speed_column] < speed_threshold)
    ].copy()

    df = df.dropna(subset=[time_column, x_column, y_column])

    if len(df) < 5:
        return pd.DataFrame()

    df = df[df[x_column] < x_limit].copy()

    df = smooth_positions(
        df,
        time_column=time_column,
        x_column=x_column,
        y_column=y_column,
        smoothing_factor=smoothing_factor,
    )

    if len(df) < 5:
        return pd.DataFrame()

    return df


def compute_dispersion(
    df: pd.DataFrame,
    x_limit: float,
    time_column: str = "time",
    x_column: str = "x",
    y_column: str = "y",
    mask_column: str = "mask",
    speed_column: str = "speed",
    bin_size: float = 3,
    smoothing_factor: float = DEFAULT_SMOOTHING_FACTOR,
    speed_threshold: float = DEFAULT_SPEED_THRESHOLD,
) -> float:
    """Mean dispersion for one fish: mean minimum-enclosing-circle radius across time bins."""
    df = prepare_zone_data(
        df=df,
        x_limit=x_limit,
        time_column=time_column,
        x_column=x_column,
        y_column=y_column,
        mask_column=mask_column,
        speed_column=speed_column,
        speed_threshold=speed_threshold,
        smoothing_factor=smoothing_factor,
    )

    if df.empty:
        return np.nan

    df["time_bin"] = (df[time_column] // bin_size).astype(int) * bin_size

    radii = []

    for t in sorted(df["time_bin"].unique()):
        subset = df[df["time_bin"] == t][[x_column, y_column]].dropna()

        if len(subset) >= 2:
            radius, _ = compute_minimum_enclosing_circle(subset.values)
            radii.append(radius)

    return np.nanmean(radii) if len(radii) > 0 else np.nan


def compute_fish_center_shift(
    df: pd.DataFrame,
    x_limit: float,
    time_column: str = "time",
    x_column: str = "x",
    y_column: str = "y",
    mask_column: str = "mask",
    speed_column: str = "speed",
    bin_size: float = 3,
    smoothing_factor: float = DEFAULT_SMOOTHING_FACTOR,
    speed_threshold: float = DEFAULT_SPEED_THRESHOLD,
) -> float:
    """Mean center shift for one fish: mean distance between consecutive time bins' enclosing-circle centers."""
    df = prepare_zone_data(
        df=df,
        x_limit=x_limit,
        time_column=time_column,
        x_column=x_column,
        y_column=y_column,
        mask_column=mask_column,
        speed_column=speed_column,
        speed_threshold=speed_threshold,
        smoothing_factor=smoothing_factor,
    )

    if df.empty:
        return np.nan

    df["time_bin"] = (df[time_column] // bin_size).astype(int) * bin_size

    centers = []

    for t in sorted(df["time_bin"].unique()):
        subset = df[df["time_bin"] == t][[x_column, y_column]].dropna()

        if len(subset) >= 2:
            _, (cx, cy) = compute_minimum_enclosing_circle(subset.values)
            centers.append((t, cx, cy))

    shifts = []

    for (_, x1, y1), (_, x2, y2) in zip(centers[:-1], centers[1:]):
        shifts.append(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))

    return np.nanmean(shifts) if len(shifts) > 0 else np.nan


def compute_fish_dispersions(
    fish_list: list[pd.DataFrame],
    x_limit: float,
    time_column: str = "time",
    x_column: str = "x",
    y_column: str = "y",
    mask_column: str = "mask",
    speed_column: str = "speed",
    bin_size: float = 10,
    smoothing_factor: float = DEFAULT_SMOOTHING_FACTOR,
    speed_threshold: float = DEFAULT_SPEED_THRESHOLD,
) -> list[float]:
    """Per-fish `compute_dispersion`."""
    return [
        compute_dispersion(
            df=fish_df,
            x_limit=x_limit,
            time_column=time_column,
            x_column=x_column,
            y_column=y_column,
            mask_column=mask_column,
            speed_column=speed_column,
            bin_size=bin_size,
            smoothing_factor=smoothing_factor,
            speed_threshold=speed_threshold,
        )
        for fish_df in fish_list
    ]


def compute_fish_center_shifts(
    fish_list: list[pd.DataFrame],
    x_limit: float,
    time_column: str = "time",
    x_column: str = "x",
    y_column: str = "y",
    mask_column: str = "mask",
    speed_column: str = "speed",
    bin_size: float = 10,
    smoothing_factor: float = DEFAULT_SMOOTHING_FACTOR,
    speed_threshold: float = DEFAULT_SPEED_THRESHOLD,
) -> list[float]:
    """Per-fish `compute_fish_center_shift`."""
    return [
        compute_fish_center_shift(
            df=fish_df,
            x_limit=x_limit,
            time_column=time_column,
            x_column=x_column,
            y_column=y_column,
            mask_column=mask_column,
            speed_column=speed_column,
            bin_size=bin_size,
            smoothing_factor=smoothing_factor,
            speed_threshold=speed_threshold,
        )
        for fish_df in fish_list
    ]


# ---------------------------------------------------------------------------
# Bout detection and bout-angle change (source: cell b755f281)
# ---------------------------------------------------------------------------

def find_bouts(
    speed: np.ndarray,
    threshold: float = DEFAULT_BOUT_THRESHOLD,
    min_duration: int = DEFAULT_BOUT_MIN_DURATION,
    expand: int = DEFAULT_BOUT_EXPAND,
    gap_tolerance: int = DEFAULT_BOUT_GAP_TOLERANCE,
    max_speed: float = DEFAULT_BOUT_MAX_SPEED,
) -> list[tuple[int, int]]:
    """Find (start_idx, end_idx) bout boundaries from a speed trace.

    A bout is a run of `threshold < speed <= max_speed` frames at least
    `min_duration` long, expanded by `expand` frames on each side, then
    merged with the previous bout if separated by <= `gap_tolerance` frames.
    """
    speed = np.asarray(speed, dtype=float)

    valid_speed = (speed > threshold) & (speed <= max_speed)
    valid_speed = np.nan_to_num(valid_speed, nan=False)

    bouts = []
    start_idx = None

    for i, is_valid in enumerate(valid_speed):
        if is_valid and start_idx is None:
            start_idx = i

        elif not is_valid and start_idx is not None:
            if (i - start_idx) >= min_duration:
                bout_start = max(0, start_idx - expand)
                bout_end = min(len(speed) - 1, i - 1 + expand)

                if bouts and (bout_start - bouts[-1][1] <= gap_tolerance):
                    bouts[-1] = (bouts[-1][0], bout_end)
                else:
                    bouts.append((bout_start, bout_end))

            start_idx = None

    if start_idx is not None and (len(speed) - start_idx) >= min_duration:
        bout_start = max(0, start_idx - expand)
        bout_end = len(speed) - 1

        if bouts and (bout_start - bouts[-1][1] <= gap_tolerance):
            bouts[-1] = (bouts[-1][0], bout_end)
        else:
            bouts.append((bout_start, bout_end))

    return bouts


def get_valid_bout_endpoint(
    df: pd.DataFrame,
    start: int,
    end: int,
    which: str = "start",
    mask_col: str = "mask",
    x_col: str = "x",
    y_col: str = "y",
) -> int | None:
    """Nearest tracked, non-NaN frame index within [start, end], searching from `which` end."""
    if which == "start":
        indices = range(start, end + 1)
    elif which == "end":
        indices = range(end, start - 1, -1)
    else:
        raise ValueError("which must be 'start' or 'end'")

    for idx in indices:
        if (
            bool(df.loc[idx, mask_col]) and
            pd.notna(df.loc[idx, x_col]) and
            pd.notna(df.loc[idx, y_col])
        ):
            return idx

    return None


def get_heading_from_bout(
    df: pd.DataFrame,
    start: int,
    end: int,
    mask_col: str = "mask",
    x_col: str = "x",
    y_col: str = "y",
) -> tuple[float | None, float]:
    """Heading (radians) and mean tracked x-position for one bout.

    Heading is computed from the nearest valid tracked frames at the bout's
    start and end. Returns (None, nan) if either endpoint or the bout itself
    (zero displacement) is invalid.
    """
    start_valid = get_valid_bout_endpoint(
        df, start, end, which="start", mask_col=mask_col, x_col=x_col, y_col=y_col
    )
    end_valid = get_valid_bout_endpoint(
        df, start, end, which="end", mask_col=mask_col, x_col=x_col, y_col=y_col
    )

    if start_valid is None or end_valid is None:
        return None, np.nan

    x0 = df.loc[start_valid, x_col]
    y0 = df.loc[start_valid, y_col]
    x1 = df.loc[end_valid, x_col]
    y1 = df.loc[end_valid, y_col]

    dx = x1 - x0
    dy = y1 - y0

    if dx == 0 and dy == 0:
        return None, np.nan

    heading = np.arctan2(dy, dx)

    valid_bout_frames = df.loc[start:end]
    valid_bout_frames = valid_bout_frames[
        valid_bout_frames[mask_col] &
        valid_bout_frames[x_col].notna()
    ]
    x_mean = valid_bout_frames[x_col].mean() if not valid_bout_frames.empty else np.nan

    return heading, x_mean


def compute_mean_bout_angle_change(
    fish_df: pd.DataFrame,
    x_limit: float = ZONE_HALF_X,
    last_half_only: bool = True,
    mask_col: str = "mask",
    speed_col: str = "speed",
    x_col: str = "x",
    y_col: str = "y",
    bout_threshold: float = DEFAULT_BOUT_THRESHOLD,
    bout_min_duration: int = DEFAULT_BOUT_MIN_DURATION,
    bout_expand: int = DEFAULT_BOUT_EXPAND,
    bout_gap_tolerance: int = DEFAULT_BOUT_GAP_TOLERANCE,
    bout_max_speed: float = DEFAULT_BOUT_MAX_SPEED,
) -> float:
    """Mean angular change (degrees) between consecutive bouts for one fish.

    Bouts are detected from speed (invalid-tracking frames masked to NaN
    first). Only bout pairs whose current bout's mean x lies in the zone
    x < `x_limit` are included; optionally restricted to bouts starting in
    the trial's last half.
    """
    df = fish_df.copy()

    speed = df[speed_col].copy()
    speed.loc[~df[mask_col]] = np.nan

    bouts = find_bouts(
        speed=speed.values,
        threshold=bout_threshold,
        min_duration=bout_min_duration,
        expand=bout_expand,
        gap_tolerance=bout_gap_tolerance,
        max_speed=bout_max_speed,
    )

    if last_half_only:
        midpoint = len(df) // 2
        bouts = [b for b in bouts if b[0] >= midpoint]

    angle_changes = []

    for i in range(1, len(bouts)):
        prev_start, prev_end = bouts[i - 1]
        curr_start, curr_end = bouts[i]

        prev_heading, _ = get_heading_from_bout(
            df, prev_start, prev_end, mask_col=mask_col, x_col=x_col, y_col=y_col
        )
        curr_heading, curr_x_mean = get_heading_from_bout(
            df, curr_start, curr_end, mask_col=mask_col, x_col=x_col, y_col=y_col
        )

        if prev_heading is None or curr_heading is None:
            continue

        if pd.isna(curr_x_mean) or curr_x_mean >= x_limit:
            continue

        angle_diff = np.abs(curr_heading - prev_heading)
        if angle_diff > np.pi:
            angle_diff = 2 * np.pi - angle_diff

        angle_changes.append(np.degrees(angle_diff))

    return np.nanmean(angle_changes) if len(angle_changes) > 0 else np.nan


def compute_fish_bout_angle_changes(
    fish_list: list[pd.DataFrame],
    x_limit: float = ZONE_HALF_X,
    last_half_only: bool = True,
    mask_col: str = "mask",
    speed_col: str = "speed",
    x_col: str = "x",
    y_col: str = "y",
    bout_threshold: float = DEFAULT_BOUT_THRESHOLD,
    bout_min_duration: int = DEFAULT_BOUT_MIN_DURATION,
    bout_expand: int = DEFAULT_BOUT_EXPAND,
    bout_gap_tolerance: int = DEFAULT_BOUT_GAP_TOLERANCE,
    bout_max_speed: float = DEFAULT_BOUT_MAX_SPEED,
) -> list[float]:
    """Per-fish `compute_mean_bout_angle_change`."""
    return [
        compute_mean_bout_angle_change(
            fish_df=fish_df,
            x_limit=x_limit,
            last_half_only=last_half_only,
            mask_col=mask_col,
            speed_col=speed_col,
            x_col=x_col,
            y_col=y_col,
            bout_threshold=bout_threshold,
            bout_min_duration=bout_min_duration,
            bout_expand=bout_expand,
            bout_gap_tolerance=bout_gap_tolerance,
            bout_max_speed=bout_max_speed,
        )
        for fish_df in fish_list
    ]


# ---------------------------------------------------------------------------
# Per-experiment aggregation (source: cells f810ecc4, e7533e4d, de08a4b2)
# ---------------------------------------------------------------------------

def summarize_experiment(
    fish_list: list[pd.DataFrame],
    frame_rate: int,
    speed_threshold: float = DEFAULT_SPEED_THRESHOLD,
    dispersion_bin_size: float = 3,
    dispersion_smoothing_factor: float = DEFAULT_SMOOTHING_FACTOR,
) -> dict:
    """Preference, speed, dispersion, and center-shift metrics for one experiment, averaged over fish."""
    fish_prefs = calculate_fish_preference_indices(fish_list, frame_rate)
    preference_index = np.nanmean(fish_prefs) if len(fish_prefs) > 0 else np.nan

    fish_speed_half = compute_avg_speeds_zone(
        fish_list, x_limit=ZONE_HALF_X, speed_threshold=speed_threshold
    )
    speed_half = np.nanmean(fish_speed_half) if len(fish_speed_half) > 0 else np.nan

    fish_speed_quarter = compute_avg_speeds_zone(
        fish_list, x_limit=ZONE_QUARTER_X, speed_threshold=speed_threshold
    )
    speed_quarter = np.nanmean(fish_speed_quarter) if len(fish_speed_quarter) > 0 else np.nan

    fish_dispersion_half = compute_fish_dispersions(
        fish_list,
        x_limit=ZONE_HALF_X,
        bin_size=dispersion_bin_size,
        smoothing_factor=dispersion_smoothing_factor,
        speed_threshold=speed_threshold,
    )
    dispersion_half = np.nanmean(fish_dispersion_half) if len(fish_dispersion_half) > 0 else np.nan

    fish_dispersion_quarter = compute_fish_dispersions(
        fish_list,
        x_limit=ZONE_QUARTER_X,
        bin_size=dispersion_bin_size,
        smoothing_factor=dispersion_smoothing_factor,
        speed_threshold=speed_threshold,
    )
    dispersion_quarter = np.nanmean(fish_dispersion_quarter) if len(fish_dispersion_quarter) > 0 else np.nan

    fish_center_shift_half = compute_fish_center_shifts(
        fish_list,
        x_limit=ZONE_HALF_X,
        bin_size=dispersion_bin_size,
        smoothing_factor=dispersion_smoothing_factor,
        speed_threshold=speed_threshold,
    )
    center_shift_half = np.nanmean(fish_center_shift_half) if len(fish_center_shift_half) > 0 else np.nan

    fish_center_shift_quarter = compute_fish_center_shifts(
        fish_list,
        x_limit=ZONE_QUARTER_X,
        bin_size=dispersion_bin_size,
        smoothing_factor=dispersion_smoothing_factor,
        speed_threshold=speed_threshold,
    )
    center_shift_quarter = np.nanmean(fish_center_shift_quarter) if len(fish_center_shift_quarter) > 0 else np.nan

    return {
        "preference_index": preference_index,
        "speed_half": speed_half,
        "speed_quarter": speed_quarter,
        "dispersion_half": dispersion_half,
        "dispersion_quarter": dispersion_quarter,
        "center_shift_half": center_shift_half,
        "center_shift_quarter": center_shift_quarter,
        "n_fish": len(fish_list),
    }


def summarize_experiment_angles(
    fish_list: list[pd.DataFrame],
    frame_rate: int = DEFAULT_FRAME_RATE,
    bout_threshold: float = DEFAULT_BOUT_THRESHOLD,
    bout_min_duration: int = DEFAULT_BOUT_MIN_DURATION,
    bout_expand: int = DEFAULT_BOUT_EXPAND,
    bout_gap_tolerance: int = DEFAULT_BOUT_GAP_TOLERANCE,
    bout_max_speed: float = DEFAULT_BOUT_MAX_SPEED,
) -> dict:
    """Bout-angle-change metrics for one experiment, averaged over fish.

    `frame_rate` is accepted for parity with the notebook's call signature
    but unused, matching the original.
    """
    fish_angle_change_half = compute_fish_bout_angle_changes(
        fish_list,
        x_limit=ZONE_HALF_X,
        last_half_only=True,
        bout_threshold=bout_threshold,
        bout_min_duration=bout_min_duration,
        bout_expand=bout_expand,
        bout_gap_tolerance=bout_gap_tolerance,
        bout_max_speed=bout_max_speed,
    )
    angle_change_half = np.nanmean(fish_angle_change_half) if len(fish_angle_change_half) > 0 else np.nan

    fish_angle_change_quarter = compute_fish_bout_angle_changes(
        fish_list,
        x_limit=ZONE_QUARTER_X,
        last_half_only=True,
        bout_threshold=bout_threshold,
        bout_min_duration=bout_min_duration,
        bout_expand=bout_expand,
        bout_gap_tolerance=bout_gap_tolerance,
        bout_max_speed=bout_max_speed,
    )
    angle_change_quarter = np.nanmean(fish_angle_change_quarter) if len(fish_angle_change_quarter) > 0 else np.nan

    return {
        "angle_change_half": angle_change_half,
        "angle_change_quarter": angle_change_quarter,
    }


def _iter_experiments(
    stimulus_data: dict[str, dict[str, dict[str, list[pd.DataFrame]]]],
) -> Iterable[tuple[str, str, str, list[pd.DataFrame]]]:
    """Yield (stimulus, condition, experiment_name, fish_list) in the notebook's iteration order."""
    for stimulus, conds in stimulus_data.items():
        for condition in ["stim", "control"]:
            for exp_name, fish_list in conds[condition].items():
                yield stimulus, condition, exp_name, fish_list


def build_experiment_results_table(
    stimulus_data: dict[str, dict[str, dict[str, list[pd.DataFrame]]]],
    frame_rate: int = DEFAULT_FRAME_RATE,
    speed_threshold: float = DEFAULT_SPEED_THRESHOLD,
    dispersion_bin_size: float = 5,
    dispersion_smoothing_factor: float = DEFAULT_SMOOTHING_FACTOR,
) -> pd.DataFrame:
    """One row per experiment x condition of `summarize_experiment` metrics. Source: cell e7533e4d."""
    results = []

    for stimulus, condition, exp_name, fish_list in _iter_experiments(stimulus_data):
        summary = summarize_experiment(
            fish_list=fish_list,
            frame_rate=frame_rate,
            speed_threshold=speed_threshold,
            dispersion_bin_size=dispersion_bin_size,
            dispersion_smoothing_factor=dispersion_smoothing_factor,
        )

        results.append({
            "stimulus": stimulus,
            "pair_key": get_pair_key(exp_name),
            "experiment": exp_name,
            "condition": condition,
            **summary,
        })

    return pd.DataFrame(results)


def build_experiment_angle_results_table(
    stimulus_data: dict[str, dict[str, dict[str, list[pd.DataFrame]]]],
    frame_rate: int = DEFAULT_FRAME_RATE,
    bout_threshold: float = DEFAULT_BOUT_THRESHOLD,
    bout_min_duration: int = DEFAULT_BOUT_MIN_DURATION,
    bout_expand: int = DEFAULT_BOUT_EXPAND,
    bout_gap_tolerance: int = DEFAULT_BOUT_GAP_TOLERANCE,
    bout_max_speed: float = DEFAULT_BOUT_MAX_SPEED,
) -> pd.DataFrame:
    """One row per experiment x condition of `summarize_experiment_angles` metrics. Source: cell de08a4b2."""
    angle_results = []

    for stimulus, condition, exp_name, fish_list in _iter_experiments(stimulus_data):
        angle_summary = summarize_experiment_angles(
            fish_list=fish_list,
            frame_rate=frame_rate,
            bout_threshold=bout_threshold,
            bout_min_duration=bout_min_duration,
            bout_expand=bout_expand,
            bout_gap_tolerance=bout_gap_tolerance,
            bout_max_speed=bout_max_speed,
        )

        angle_results.append({
            "stimulus": stimulus,
            "pair_key": get_pair_key(exp_name),
            "experiment": exp_name,
            "condition": condition,
            **angle_summary,
        })

    return pd.DataFrame(angle_results)


# ---------------------------------------------------------------------------
# Per-fish aggregation (source: cells b21aa197, 0f5f2ddf)
# ---------------------------------------------------------------------------

def summarize_fish_list(
    fish_list: list[pd.DataFrame],
    frame_rate: int,
    speed_threshold: float = DEFAULT_SPEED_THRESHOLD,
    dispersion_bin_size: float = 5,
    dispersion_smoothing_factor: float = DEFAULT_SMOOTHING_FACTOR,
    include_angles: bool = False,
) -> list[dict]:
    """One row per fish with preference/speed/dispersion/center-shift (and optionally bout-angle) metrics."""
    fish_prefs = calculate_fish_preference_indices(fish_list, frame_rate)

    fish_speed_half = compute_avg_speeds_zone(
        fish_list, x_limit=ZONE_HALF_X, speed_threshold=speed_threshold
    )
    fish_speed_quarter = compute_avg_speeds_zone(
        fish_list, x_limit=ZONE_QUARTER_X, speed_threshold=speed_threshold
    )

    fish_dispersion_half = compute_fish_dispersions(
        fish_list,
        x_limit=ZONE_HALF_X,
        bin_size=dispersion_bin_size,
        smoothing_factor=dispersion_smoothing_factor,
        speed_threshold=speed_threshold,
    )
    fish_dispersion_quarter = compute_fish_dispersions(
        fish_list,
        x_limit=ZONE_QUARTER_X,
        bin_size=dispersion_bin_size,
        smoothing_factor=dispersion_smoothing_factor,
        speed_threshold=speed_threshold,
    )

    fish_center_shift_half = compute_fish_center_shifts(
        fish_list,
        x_limit=ZONE_HALF_X,
        bin_size=dispersion_bin_size,
        smoothing_factor=dispersion_smoothing_factor,
        speed_threshold=speed_threshold,
    )
    fish_center_shift_quarter = compute_fish_center_shifts(
        fish_list,
        x_limit=ZONE_QUARTER_X,
        bin_size=dispersion_bin_size,
        smoothing_factor=dispersion_smoothing_factor,
        speed_threshold=speed_threshold,
    )

    fish_rows = []

    for i in range(len(fish_list)):
        row = {
            "fish_number": i,
            "preference_index": fish_prefs[i] if i < len(fish_prefs) else np.nan,
            "speed_half": fish_speed_half[i] if i < len(fish_speed_half) else np.nan,
            "speed_quarter": fish_speed_quarter[i] if i < len(fish_speed_quarter) else np.nan,
            "dispersion_half": fish_dispersion_half[i] if i < len(fish_dispersion_half) else np.nan,
            "dispersion_quarter": fish_dispersion_quarter[i] if i < len(fish_dispersion_quarter) else np.nan,
            "center_shift_half": fish_center_shift_half[i] if i < len(fish_center_shift_half) else np.nan,
            "center_shift_quarter": fish_center_shift_quarter[i] if i < len(fish_center_shift_quarter) else np.nan,
        }
        fish_rows.append(row)

    if include_angles:
        fish_angle_change_half = compute_fish_bout_angle_changes(
            fish_list, x_limit=ZONE_HALF_X, last_half_only=True
        )
        fish_angle_change_quarter = compute_fish_bout_angle_changes(
            fish_list, x_limit=ZONE_QUARTER_X, last_half_only=True
        )

        for i in range(len(fish_rows)):
            fish_rows[i]["angle_change_half"] = (
                fish_angle_change_half[i] if i < len(fish_angle_change_half) else np.nan
            )
            fish_rows[i]["angle_change_quarter"] = (
                fish_angle_change_quarter[i] if i < len(fish_angle_change_quarter) else np.nan
            )

    return fish_rows


def build_fish_results_table(
    stimulus_data: dict[str, dict[str, dict[str, list[pd.DataFrame]]]],
    frame_rate: int = DEFAULT_FRAME_RATE,
    speed_threshold: float = DEFAULT_SPEED_THRESHOLD,
    dispersion_bin_size: float = 3,
    dispersion_smoothing_factor: float = DEFAULT_SMOOTHING_FACTOR,
    include_angles: bool = True,
) -> pd.DataFrame:
    """One row per fish, across every experiment x condition. Source: cell 0f5f2ddf."""
    fish_results = []

    for stimulus, condition, exp_name, fish_list in _iter_experiments(stimulus_data):
        fish_rows = summarize_fish_list(
            fish_list=fish_list,
            frame_rate=frame_rate,
            speed_threshold=speed_threshold,
            dispersion_bin_size=dispersion_bin_size,
            dispersion_smoothing_factor=dispersion_smoothing_factor,
            include_angles=include_angles,
        )

        for row in fish_rows:
            fish_results.append({
                "stimulus": stimulus,
                "pair_key": get_pair_key(exp_name),
                "experiment": exp_name,
                "condition": condition,
                **row,
            })

    return pd.DataFrame(fish_results)
