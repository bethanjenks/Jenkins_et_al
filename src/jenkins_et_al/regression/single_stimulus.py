"""Vectorised single-regressor stimulus-response correlation analysis.

Ported from `scripts/run_single_stimulus_regression.py`. Direct Pearson
correlations between neural activity and a stimulus regressor, deliberately
WITHOUT permutation/significance testing (unlike `core.py`'s shared
z-score/shuffle core used by the other four regression scripts) -- this
script's correlation formula, trace-table building (one table per stimulus,
rather than one wide concatenated-across-stimuli row per neuron), and lack of
a shuffle test are all real structural differences, not oversights, so this
module stays self-contained rather than reusing `regression.core`.

**Confirmed bug, fixed at the user's request**: as saved, this script cannot
run at all. `get_regressor` does a strict per-stimulus-name lookup
(`stim_regressors[stimulus]`) against `STIMULI`'s 15 names, but the script's
own default regressor file (`single_regressor_sustained_25s.pkl`) has exactly
one key, `"single_regressor"`, which matches none of them -- `KeyError` on
the very first stimulus of the very first fish. Inspecting that regressor
resolved it: it's 900 frames (3 trials x 300 frames), identical across all
three trials, a smooth 0-to-1-to-0 ramp over ~103 of each trial's 300 frames
-- a single canonical "sustained response" template, sized to match one
stimulus's own 3-trial concatenated window (`build_trace_table` concatenates
one stimulus's trials at a time, not the full battery). It's meant to be
reused as-is for every stimulus, not looked up by name. `get_single_regressor`
below returns the sole value in the dict regardless of its key, instead of
indexing by stimulus.

**Path drift, not a logic bug**: the script's own default neural-data paths
(`imaging/4dpf_fb_fish_dfs_final.h5`) don't resolve on the current filesystem
-- the real files live one directory deeper, at `imaging/4dpf/...`. Defaults
below point at the real location; still fully overridable via function
arguments, matching how every other port here handles path drift.

**Life-stage parametrized, at the user's request** (`PORTING_PLAN.md`'s own
flag for this script): the original hardcodes `"4dpf"` directly into its
output filenames inside `run_analysis`. `run_analysis` here takes an explicit
`life_stage` parameter (default `"4dpf"`, preserving identical default
filenames) instead.

The `where="fish_id == fish_key"` HDF5 query (referencing a local Python
variable by bare name rather than embedding it in an f-string, unlike the
other four regression scripts) is unusual but confirmed working against real
data -- PyTables' query evaluator picks up local variables from the calling
frame; not a bug.
"""
from __future__ import annotations

import json
from ast import literal_eval
from functools import reduce
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

#: Source: module-level constants in the original script.
FISH_LIST: tuple[int, ...] = (
    230710, 230711, 230717, 230718, 230724,
    230725, 230809, 230821, 230822, 230823,
)
STIMULI: tuple[str, ...] = (
    "ade", "cad_2.5mm", "cad_25um", "cad_250um",
    "fex_1", "fex_2", "fex_3", "kw", "ph4.5",
    "pro_2.5mm", "pro_25um", "pro_250um",
    "qui_2.5mm", "qui_25um", "qui_250um",
)

IDENTITY_COLUMNS: tuple[str, ...] = ("fish_id", "neuron_id", "coords", "area")
PIVOT_INDEX_COLUMNS: tuple[str, ...] = ("fish_id", "neuron_id", "coords_serialised", "area")


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
            return literal_eval(value)
    return value


def load_neural_data(file_path: Path, fish: int | str, suffix: str) -> pd.DataFrame:
    """Load one fish/region from HDF5 and parse traces once."""
    fish_key = f"{fish}{suffix}"
    neural_data = pd.read_hdf(file_path, where="fish_id == fish_key")

    if neural_data.empty:
        return neural_data

    neural_data = neural_data.copy()
    neural_data["img_trace"] = neural_data["trace_serialised"].map(parse_trace)
    return neural_data


def concatenate_trace_cells(cells: Sequence[Any]) -> np.ndarray | None:
    """Concatenate trial traces from a pivot row, ignoring missing cells."""
    arrays = [cell for cell in cells if isinstance(cell, np.ndarray)]
    if not arrays:
        return None
    return np.concatenate(arrays).astype(float, copy=False)


def build_trace_table(neural_data: pd.DataFrame, stimulus: str) -> pd.DataFrame:
    """Return one row per neuron with concatenated trial traces for one stimulus."""
    if neural_data.empty:
        return pd.DataFrame(columns=[*PIVOT_INDEX_COLUMNS, "concat_trace"])

    filtered = neural_data.loc[neural_data["stimulus"] == stimulus]
    if filtered.empty:
        return pd.DataFrame(columns=[*PIVOT_INDEX_COLUMNS, "concat_trace"])

    pivoted = filtered.pivot_table(
        index=list(PIVOT_INDEX_COLUMNS),
        columns="trial_number",
        values="img_trace",
        aggfunc="first",
        sort=True,
    )

    trace_cells = pivoted.to_numpy(dtype=object)
    concat_traces = [concatenate_trace_cells(row) for row in trace_cells]

    out = pivoted.reset_index()
    out = out.loc[:, list(PIVOT_INDEX_COLUMNS)]
    out["concat_trace"] = concat_traces
    out = out.loc[out["concat_trace"].notna()].copy()
    return out


def get_single_regressor(stim_regressors: dict[str, Any]) -> np.ndarray:
    """Return the sole regressor in the dict, reused for every stimulus.

    See module docstring: the original's per-stimulus-name lookup is the
    confirmed bug this replaces.
    """
    if len(stim_regressors) != 1:
        raise ValueError(
            f"Expected exactly one regressor, found {len(stim_regressors)}: "
            f"{sorted(stim_regressors.keys())}"
        )
    (regressor,) = stim_regressors.values()
    return np.asarray(regressor, dtype=float).ravel()


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
    denominator = np.sqrt(np.nansum(x_centered**2, axis=1) * np.nansum(y_centered**2))

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
        trace_table = trace_table.loc[valid].copy()

    if trace_table.empty:
        return trace_table.assign(regressor_corr=pd.Series(dtype=float))

    trace_matrix = np.vstack(trace_table["concat_trace"].to_numpy())
    trace_table = trace_table.copy()
    trace_table["regressor_corr"] = vectorised_pearson_correlation(trace_matrix, regressor)
    return trace_table


def create_results_df(correlated_data: pd.DataFrame, stimulus: str) -> pd.DataFrame:
    """Create the clean per-stimulus output table."""
    if correlated_data.empty:
        return pd.DataFrame(columns=[f"{stimulus}_correlation", *IDENTITY_COLUMNS])

    out = correlated_data.loc[
        :, ["fish_id", "neuron_id", "coords_serialised", "area", "regressor_corr"]
    ].copy()

    out["coords"] = out["coords_serialised"].map(parse_coords)
    out = out.rename(columns={"regressor_corr": f"{stimulus}_correlation"})
    return out.loc[:, [f"{stimulus}_correlation", *IDENTITY_COLUMNS]]


def merge_stimulus_results(stimulus_dfs: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Merge all stimulus result tables by neuron identity."""
    non_empty = [df for df in stimulus_dfs if not df.empty]
    if not non_empty:
        return pd.DataFrame(columns=IDENTITY_COLUMNS)

    keyed = []
    for df in non_empty:
        temp = df.copy()
        temp["coords_key"] = temp["coords"].map(lambda x: json.dumps(x, sort_keys=True))
        temp = temp.drop(columns=["coords"])
        keyed.append(temp)

    merged = reduce(
        lambda left, right: pd.merge(
            left, right, on=["fish_id", "neuron_id", "coords_key", "area"], how="outer",
        ),
        keyed,
    )

    merged["coords"] = merged["coords_key"].map(parse_coords)
    merged = merged.drop(columns=["coords_key"])

    correlation_cols = [col for col in merged.columns if col.endswith("_correlation")]
    return merged.loc[:, [*correlation_cols, *IDENTITY_COLUMNS]]


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
    stim_regressors: dict[str, Any],
    stimuli: Sequence[str],
    fb_neural_path: Path,
    hb_neural_path: Path,
    life_stage: str = "4dpf",
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Run vectorised correlation analysis for all fish and stimuli.

    Returns:
        (per-stimulus results keyed by stimulus name, combined results table)
        -- caller decides whether/where to write CSVs, unlike the original
        script's `run_analysis`, which wrote files as a side effect.
    """
    regressor = get_single_regressor(stim_regressors)

    per_stimulus_parts: dict[str, list[pd.DataFrame]] = {stimulus: [] for stimulus in stimuli}

    for fish in fish_list:
        neural_fb = load_neural_data(fb_neural_path, fish, suffix="_fb")
        neural_hb = load_neural_data(hb_neural_path, fish, suffix="_hb")

        for stimulus in stimuli:
            fb_results = process_region_for_stimulus(neural_fb, stimulus, regressor)
            hb_results = process_region_for_stimulus(neural_hb, stimulus, regressor)

            combined_region_results = pd.concat(
                [fb_results, hb_results], axis=0, ignore_index=True,
            )
            per_stimulus_parts[stimulus].append(combined_region_results)

    stimulus_dfs: dict[str, pd.DataFrame] = {
        stimulus: pd.concat(parts, axis=0, ignore_index=True)
        for stimulus, parts in per_stimulus_parts.items()
    }

    combined_df = merge_stimulus_results(list(stimulus_dfs.values()))
    return stimulus_dfs, combined_df


def save_results(
    stimulus_dfs: dict[str, pd.DataFrame],
    combined_df: pd.DataFrame,
    output_dir: Path,
    life_stage: str = "4dpf",
) -> None:
    """Write one CSV per stimulus plus the combined table, matching the
    original script's output filenames (with `life_stage` substituted for
    its hardcoded `"4dpf"`)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for stimulus, df in stimulus_dfs.items():
        df.to_csv(
            output_dir / f"{life_stage}_{stimulus}_sustained_25s_response_correlations.csv",
            index=False,
        )
    combined_df.to_csv(
        output_dir / f"{life_stage}_combined_stimulus_response_sustained_25s_correlations.csv",
        index=False,
    )
