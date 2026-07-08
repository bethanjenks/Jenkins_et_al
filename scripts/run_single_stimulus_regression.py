"""
Vectorised Single Stimulus-Response Correlation Analysis

Calculates direct Pearson correlations between neural activity and individual
stimulus regressors WITHOUT permutation/significance testing.

Speed-up changes compared with the previous refactored script:
    1. Loads each fish/brain-region HDF5 data once, then reuses it for all stimuli.
    2. Parses serialised traces once per loaded fish/region.
    3. Avoids pd.concat inside inner loops; accumulates lists and concatenates once.
    4. Computes all neuron-regressor correlations for a fish/stimulus in one NumPy call.
    5. Combines stimulus outputs by neuron identity using merges, rather than relying on
       row order from pd.concat(axis=1).

Outputs:
    - One CSV per stimulus:
        4dpf_<stimulus>_sustained_25s_response_correlations.csv
    - One combined CSV:
        4dpf_combined_stimulus_response_sustained_25s_correlations.csv
"""

from __future__ import annotations

import argparse
import json
import pickle
from functools import reduce
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

FISH_LIST = [
    230710, 230711, 230717, 230718, 230724,
    230725, 230809, 230821, 230822, 230823,
]

STIMULI = [
    "ade", "cad_2.5mm", "cad_25um", "cad_250um",
    "fex_1", "fex_2", "fex_3", "kw", "ph4.5",
    "pro_2.5mm", "pro_25um", "pro_250um",
    "qui_2.5mm", "qui_25um", "qui_250um",
]

FB_NEURAL_PATH = Path("/Volumes/LaCie/larval_HuC/imaging/4dpf_fb_fish_dfs_final.h5")
HB_NEURAL_PATH = Path("/Volumes/LaCie/larval_HuC/imaging/4dpf_hb_fish_dfs_final.h5")
REGRESSOR_PATH = Path("/Volumes/LaCie/larval_HuC/regressors/single_regressor_sustained_25s.pkl")
OUTPUT_DIR = Path("/Volumes/LaCie/larval_HuC/imaging")

IDENTITY_COLUMNS = ["fish_id", "neuron_id", "coords", "area"]
PIVOT_INDEX_COLUMNS = ["fish_id", "neuron_id", "coords_serialised", "area"]


# =============================================================================
# DATA LOADING / PARSING
# =============================================================================

def parse_trace(value: Any) -> np.ndarray:
    """Convert a stored trace value into a 1D NumPy array."""
    if isinstance(value, np.ndarray):
        return value.astype(float, copy=False)
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=float)
    return np.asarray(json.loads(value), dtype=float)


def parse_coords(value: Any) -> Any:
    """Parse coords safely from JSON-like strings, while tolerating existing objects."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            # Some older files may use Python literal formatting rather than JSON.
            from ast import literal_eval
            return literal_eval(value)
    return value


def load_neural_data(file_path: Path, fish: int | str, suffix: str) -> pd.DataFrame:
    """Load one fish/region from HDF5 and parse traces once."""
    fish_key = f"{fish}{suffix}"
    neural_data = pd.read_hdf(file_path, where="fish_id == fish_key")

    if neural_data.empty:
        return neural_data

    # map is faster and clearer than DataFrame.apply(axis=1) here.
    neural_data = neural_data.copy()
    neural_data["img_trace"] = neural_data["trace_serialised"].map(parse_trace)
    return neural_data


# =============================================================================
# TRACE MATRIX BUILDING
# =============================================================================

def concatenate_trace_cells(cells: Sequence[Any]) -> Optional[np.ndarray]:
    """Concatenate trial traces from a pivot row, ignoring missing cells."""
    arrays = [cell for cell in cells if isinstance(cell, np.ndarray)]
    if not arrays:
        return None
    return np.concatenate(arrays).astype(float, copy=False)


def build_trace_table(neural_data: pd.DataFrame, stimulus: str) -> pd.DataFrame:
    """Return one row per neuron with concatenated trial traces for one stimulus."""
    if neural_data.empty:
        return pd.DataFrame(columns=PIVOT_INDEX_COLUMNS + ["concat_trace"])

    filtered = neural_data.loc[neural_data["stimulus"] == stimulus]
    if filtered.empty:
        return pd.DataFrame(columns=PIVOT_INDEX_COLUMNS + ["concat_trace"])

    pivoted = filtered.pivot_table(
        index=PIVOT_INDEX_COLUMNS,
        columns="trial_number",
        values="img_trace",
        aggfunc="first",
        sort=True,
    )

    trace_cells = pivoted.to_numpy(dtype=object)
    concat_traces = [concatenate_trace_cells(row) for row in trace_cells]

    out = pivoted.reset_index()
    out = out.loc[:, PIVOT_INDEX_COLUMNS]
    out["concat_trace"] = concat_traces
    out = out.loc[out["concat_trace"].notna()].copy()
    return out


def get_regressor(stim_regressors: Dict[str, Any], stimulus: str) -> np.ndarray:
    """Fetch and flatten the matching regressor for a stimulus."""
    if stimulus not in stim_regressors:
        raise KeyError(
            f"No regressor found for stimulus {stimulus!r}. "
            f"Available keys: {sorted(stim_regressors.keys())}"
        )
    return np.asarray(stim_regressors[stimulus], dtype=float).ravel()


# =============================================================================
# VECTORISED CORRELATION
# =============================================================================

def vectorised_pearson_correlation(trace_matrix: np.ndarray, regressor: np.ndarray) -> np.ndarray:
    """Compute Pearson correlation for every row in trace_matrix against regressor."""
    if trace_matrix.ndim != 2:
        raise ValueError(f"trace_matrix must be 2D, got shape {trace_matrix.shape}")

    regressor = np.asarray(regressor, dtype=float).ravel()
    if trace_matrix.shape[1] != regressor.size:
        raise ValueError(
            "Trace/regressor length mismatch: "
            f"traces have length {trace_matrix.shape[1]}, regressor has length {regressor.size}"
        )

    x = trace_matrix.astype(float, copy=False)
    y = regressor.astype(float, copy=False)

    x_centered = x - np.nanmean(x, axis=1, keepdims=True)
    y_centered = y - np.nanmean(y)

    numerator = np.nansum(x_centered * y_centered, axis=1)
    denominator = np.sqrt(np.nansum(x_centered ** 2, axis=1) * np.nansum(y_centered ** 2))

    correlations = np.full(x.shape[0], np.nan, dtype=float)
    valid = denominator > 0
    correlations[valid] = numerator[valid] / denominator[valid]
    return correlations


def add_correlations(trace_table: pd.DataFrame, regressor: np.ndarray) -> pd.DataFrame:
    """Add vectorised regressor correlations to a trace table."""
    if trace_table.empty:
        return trace_table.assign(regressor_corr=pd.Series(dtype=float))

    trace_lengths = trace_table["concat_trace"].map(len)
    expected_length = len(regressor)
    valid = trace_lengths == expected_length

    if not valid.all():
        bad_count = int((~valid).sum())
        print(
            f"    Warning: dropping {bad_count} neuron(s) with trace length != "
            f"regressor length ({expected_length})."
        )
        trace_table = trace_table.loc[valid].copy()

    if trace_table.empty:
        return trace_table.assign(regressor_corr=pd.Series(dtype=float))

    trace_matrix = np.vstack(trace_table["concat_trace"].to_numpy())
    trace_table = trace_table.copy()
    trace_table["regressor_corr"] = vectorised_pearson_correlation(trace_matrix, regressor)
    return trace_table


# =============================================================================
# RESULT FORMATTING
# =============================================================================

def create_results_df(correlated_data: pd.DataFrame, stimulus: str) -> pd.DataFrame:
    """Create the clean per-stimulus output table."""
    if correlated_data.empty:
        return pd.DataFrame(columns=[f"{stimulus}_correlation", *IDENTITY_COLUMNS])

    out = correlated_data.loc[:, [
        "fish_id", "neuron_id", "coords_serialised", "area", "regressor_corr"
    ]].copy()

    out["coords"] = out["coords_serialised"].map(parse_coords)
    out = out.rename(columns={"regressor_corr": f"{stimulus}_correlation"})
    return out.loc[:, [f"{stimulus}_correlation", *IDENTITY_COLUMNS]]


def merge_stimulus_results(stimulus_dfs: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Merge all stimulus result tables by neuron identity."""
    non_empty = [df for df in stimulus_dfs if not df.empty]
    if not non_empty:
        return pd.DataFrame(columns=IDENTITY_COLUMNS)

    # Convert coords to strings for a stable merge key, then restore a user-readable coords column.
    keyed = []
    for df in non_empty:
        temp = df.copy()
        temp["coords_key"] = temp["coords"].map(lambda x: json.dumps(x, sort_keys=True))
        temp = temp.drop(columns=["coords"])
        keyed.append(temp)

    merged = reduce(
        lambda left, right: pd.merge(
            left,
            right,
            on=["fish_id", "neuron_id", "coords_key", "area"],
            how="outer",
        ),
        keyed,
    )

    merged["coords"] = merged["coords_key"].map(parse_coords)
    merged = merged.drop(columns=["coords_key"])

    correlation_cols = [col for col in merged.columns if col.endswith("_correlation")]
    return merged.loc[:, [*correlation_cols, *IDENTITY_COLUMNS]]


# =============================================================================
# MAIN ANALYSIS PIPELINE
# =============================================================================

def process_region_for_stimulus(
    neural_data: pd.DataFrame,
    stimulus: str,
    regressor: np.ndarray,
) -> pd.DataFrame:
    """Build traces and compute correlations for one brain region and stimulus."""
    trace_table = build_trace_table(neural_data, stimulus)
    correlated = add_correlations(trace_table, regressor)
    return create_results_df(correlated, stimulus)


def run_analysis(
    fish_list: Iterable[int | str],
    stim_regressors: Dict[str, Any],
    stimuli: Sequence[str],
    fb_neural_path: Path,
    hb_neural_path: Path,
    output_dir: Path,
) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
    """Run vectorised correlation analysis for all fish and stimuli."""
    output_dir.mkdir(parents=True, exist_ok=True)

    per_stimulus_parts: Dict[str, List[pd.DataFrame]] = {stimulus: [] for stimulus in stimuli}

    for fish in fish_list:
        print(f"Loading fish {fish}...")
        neural_fb = load_neural_data(fb_neural_path, fish, suffix="_fb")
        neural_hb = load_neural_data(hb_neural_path, fish, suffix="_hb")

        print(f"  Loaded {len(neural_fb)} FB rows and {len(neural_hb)} HB rows")

        for stimulus in stimuli:
            regressor = get_regressor(stim_regressors, stimulus)

            fb_results = process_region_for_stimulus(neural_fb, stimulus, regressor)
            hb_results = process_region_for_stimulus(neural_hb, stimulus, regressor)

            combined_region_results = pd.concat(
                [fb_results, hb_results],
                axis=0,
                ignore_index=True,
            )
            per_stimulus_parts[stimulus].append(combined_region_results)

            print(
                f"  {stimulus}: {len(fb_results)} FB neurons, "
                f"{len(hb_results)} HB neurons"
            )

    stimulus_dfs: List[pd.DataFrame] = []
    for stimulus in stimuli:
        stimulus_df = pd.concat(per_stimulus_parts[stimulus], axis=0, ignore_index=True)
        stimulus_dfs.append(stimulus_df)

        output_path = output_dir / f"4dpf_{stimulus}_sustained_25s_response_correlations.csv"
        stimulus_df.to_csv(output_path, index=False)
        print(f"Saved {output_path}")

    print("Combining all stimuli into one table...")
    combined_df = merge_stimulus_results(stimulus_dfs)
    combined_path = output_dir / "4dpf_combined_stimulus_response_sustained_25s_correlations.csv"
    combined_df.to_csv(combined_path, index=False)

    print(f"Saved {combined_path}")
    print(f"Total columns: {len(combined_df.columns)}")
    print(f"Total neurons: {len(combined_df)}")

    return stimulus_dfs, combined_df


# =============================================================================
# CLI / ENTRY POINT
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vectorised single-regressor stimulus-response correlation analysis."
    )
    parser.add_argument("--fb-path", type=Path, default=FB_NEURAL_PATH)
    parser.add_argument("--hb-path", type=Path, default=HB_NEURAL_PATH)
    parser.add_argument("--regressor-path", type=Path, default=REGRESSOR_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--stimuli",
        nargs="+",
        default=STIMULI,
        help="Stimulus names to analyse. Defaults to the configured full list.",
    )
    parser.add_argument(
        "--fish",
        nargs="+",
        default=FISH_LIST,
        help="Fish IDs to analyse. Defaults to the configured full list.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Loading stimulus regressors...")
    with open(args.regressor_path, "rb") as f:
        stim_regressors = pickle.load(f)

    print(f"Loaded regressors for: {list(stim_regressors.keys())}")
    print(f"Analysing {len(args.fish)} fish across {len(args.stimuli)} stimuli")
    print("=" * 80)

    run_analysis(
        fish_list=args.fish,
        stim_regressors=stim_regressors,
        stimuli=args.stimuli,
        fb_neural_path=args.fb_path,
        hb_neural_path=args.hb_path,
        output_dir=args.output_dir,
    )

    print("=" * 80)
    print("Analysis complete!")


if __name__ == "__main__":
    main()
