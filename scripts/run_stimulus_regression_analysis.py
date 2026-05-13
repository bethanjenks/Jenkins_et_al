"""
Stimulus-Regressor Correlation Analysis

Calculates correlations between neural traces and stimulus regressors,
with significance testing via trial-shuffle permutation test.

This vectorized version:
1. Computes correlations for all neurons at once
2. Computes shuffled correlations for all neurons at once
3. Avoids neuron-by-neuron / shuffle-by-shuffle Python loops
"""

import numpy as np
import pandas as pd
import pickle
import json
from ast import literal_eval
import warnings

warnings.filterwarnings("ignore")


# ============================================================================
# CONFIGURATION
# ============================================================================

# Fish identifiers
FISH_LIST = [230713, 230714, 230720, 230727, 230728,
             230810, 230811, 230817, 230818, 230919]

# Stimuli to analyze
NEURO_STIMULI = ['ade', 'cad_2.5mm', 'fex_1', 'kw', 'pro_2.5mm', 'qui_2.5mm']

# File paths
FB_DATA_PATH = '/Volumes/LaCie/larval_HuC/imaging/7dpf_fb_fish_dfs_final.h5'
HB_DATA_PATH = '/Volumes/LaCie/larval_HuC/imaging/7dpf_hb_fish_dfs_final.h5'
REGRESSOR_PATH = '/Volumes/LaCie/larval_HuC/combination_regressors_no_HCl_25s.pkl'
OUTPUT_DIR = '/Volumes/LaCie/larval_HuC/imaging'

# Analysis parameters
NUM_SHUFFLES = 1000
TRIAL_LENGTH = 300

# Reproducibility
RANDOM_SEED = 42


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def concatenate_row(row):
    """Concatenate array-like entries in a row, ignoring missing values."""
    arrays = [x for x in row.values if isinstance(x, np.ndarray)]
    if len(arrays) == 0:
        return np.array([])
    return np.concatenate(arrays)


def load_neural_data(file_path, fish, end):
    """
    Load neural data for specific fish from HDF5 file.
    """
    fish_id = f"{fish}{end}"

    try:
        neural_data = pd.read_hdf(file_path, where=f'fish_id == "{fish_id}"')
    except Exception:
        neural_data = pd.read_hdf(file_path)
        neural_data = neural_data[neural_data["fish_id"] == fish_id].copy()

    if neural_data.empty:
        raise ValueError(f"No neural data found for fish_id {fish_id}")

    neural_data = neural_data.copy()
    neural_data["img_trace"] = neural_data["trace_serialised"].apply(
        lambda x: np.asarray(json.loads(x))
    )

    return neural_data


def filter_and_concatenate(neural_data, neuro_stimuli):
    """
    Filter for specific stimuli and concatenate trial traces.
    """
    filtered_data = neural_data[neural_data["stimulus"].isin(neuro_stimuli)].copy()

    if filtered_data.empty:
        return pd.DataFrame(columns=[
            "fish_id", "neuron_id", "coords_serialised", "area", "concat_trace"
        ])

    filtered_pivoted_df = filtered_data.pivot_table(
        index=["fish_id", "neuron_id", "coords_serialised", "area"],
        columns=["stimulus", "trial_number"],
        values="img_trace",
        aggfunc="first"
    )

    filtered_pivoted_df["concat_trace"] = filtered_pivoted_df.apply(
        concatenate_row,
        axis=1
    )

    filtered_pivoted_df = filtered_pivoted_df.reset_index()
    filtered_pivoted_df = filtered_pivoted_df[
        filtered_pivoted_df["concat_trace"].apply(lambda x: len(x) > 0)
    ].copy()

    return filtered_pivoted_df


# ============================================================================
# FAST CORRELATION + SHUFFLE TEST
# ============================================================================

def zscore_rows(X):
    """Z-score each row of a 2D array."""
    X = np.asarray(X, dtype=float)
    mean = np.nanmean(X, axis=1, keepdims=True)
    std = np.nanstd(X, axis=1, keepdims=True)
    std[std == 0] = np.nan
    return (X - mean) / std


def zscore_1d(x):
    """Z-score a 1D array."""
    x = np.asarray(x, dtype=float)
    mean = np.nanmean(x)
    std = np.nanstd(x)
    if std == 0:
        return np.full_like(x, np.nan, dtype=float)
    return (x - mean) / std


def calculate_correlation_vectorized(key_data, regressor):
    """
    Calculate Pearson correlation between all neural traces and regressor at once.
    """
    key_data = key_data.copy().reset_index(drop=True)

    if key_data.empty:
        key_data["regressor_corr"] = []
        return key_data

    X = np.vstack(key_data["concat_trace"].values).astype(float)
    r = np.asarray(regressor, dtype=float)

    if X.shape[1] != len(r):
        raise ValueError(
            f"Neural trace length ({X.shape[1]}) does not match regressor length ({len(r)})."
        )

    Xz = zscore_rows(X)
    rz = zscore_1d(r)

    correlations = np.nanmean(Xz * rz[None, :], axis=1)
    key_data["regressor_corr"] = correlations

    return key_data


def shuffle_trials(regressor,
                   trial_length=TRIAL_LENGTH,
                   num_shuffles=NUM_SHUFFLES,
                   random_seed=RANDOM_SEED):
    """
    Generate shuffled regressors by permuting trial order.

    Returns:
        Array of shape (num_shuffles, len(regressor))
    """
    regressor = np.asarray(regressor, dtype=float)

    if len(regressor) % trial_length != 0:
        raise ValueError(
            f"Regressor length ({len(regressor)}) must be divisible by "
            f"trial_length ({trial_length})"
        )

    rng = np.random.default_rng(random_seed)

    num_trials = len(regressor) // trial_length
    trials = regressor.reshape(num_trials, trial_length)

    shuffled_regressors = np.empty((num_shuffles, len(regressor)), dtype=float)

    for i in range(num_shuffles):
        perm = rng.permutation(num_trials)
        shuffled_regressors[i] = trials[perm].reshape(-1)

    return shuffled_regressors


def compute_correlations_and_pvalues_fast(key_data,
                                          regressor,
                                          num_shuffles=NUM_SHUFFLES,
                                          trial_length=TRIAL_LENGTH,
                                          random_seed=RANDOM_SEED,
                                          two_tailed=False):
    """
    Compute real correlations and shuffle-test p-values for all neurons at once.
    """
    key_data = key_data.copy().reset_index(drop=True)

    if key_data.empty:
        key_data["regressor_corr"] = []
        key_data["p_value"] = []
        return key_data

    X = np.vstack(key_data["concat_trace"].values).astype(float)
    regressor = np.asarray(regressor, dtype=float)

    if X.shape[1] != len(regressor):
        raise ValueError(
            f"Neural trace length ({X.shape[1]}) does not match regressor length ({len(regressor)})."
        )

    Xz = zscore_rows(X)
    rz = zscore_1d(regressor)

    real_corr = np.nanmean(Xz * rz[None, :], axis=1)

    shuffled_regressors = shuffle_trials(
        regressor=regressor,
        trial_length=trial_length,
        num_shuffles=num_shuffles,
        random_seed=random_seed
    )

    Rz = zscore_rows(shuffled_regressors)
    shuffle_corrs = (Xz @ Rz.T) / Xz.shape[1]

    if two_tailed:
        p_values = np.mean(np.abs(shuffle_corrs) >= np.abs(real_corr[:, None]), axis=1)
    else:
        p_values = np.mean(shuffle_corrs >= real_corr[:, None], axis=1)

    key_data["regressor_corr"] = real_corr
    key_data["p_value"] = p_values

    return key_data


def create_significance_results_df(neuro_data):
    """
    Create clean results DataFrame with correlation and significance.
    """
    neuro_data = neuro_data.copy()

    if neuro_data.empty:
        return pd.DataFrame(columns=[
            "correlation", "coords", "area", "fish_id", "neuron_id", "p_value"
        ])

    neuro_data["coords"] = neuro_data["coords_serialised"].apply(literal_eval)

    return pd.DataFrame({
        "correlation": neuro_data["regressor_corr"],
        "coords": neuro_data["coords"],
        "area": neuro_data["area"],
        "fish_id": neuro_data["fish_id"],
        "neuron_id": neuro_data["neuron_id"],
        "p_value": neuro_data["p_value"]
    })


# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================

def process_one_dataset(neural_data, regressor, neuro_stimuli, dataset_name):
    """
    Process one dataset (fb or hb) for one fish.
    """
    print(f"    Processing {dataset_name}...")
    key_data = filter_and_concatenate(neural_data, neuro_stimuli)

    if key_data.empty:
        return pd.DataFrame()

    key_data = compute_correlations_and_pvalues_fast(
        key_data=key_data,
        regressor=regressor,
        num_shuffles=NUM_SHUFFLES,
        trial_length=TRIAL_LENGTH,
        random_seed=RANDOM_SEED,
        two_tailed=False
    )

    return create_significance_results_df(key_data)


def main_stim(fish_list, stim_regressors, neuro_stimuli,
              fb_data_path, hb_data_path, output_dir):
    """
    Run correlation analysis for all stimuli across all fish.
    """
    for stim, regressor in stim_regressors.items():
        print(f"Processing stimulus: {stim}")

        all_corr = []

        for fish in fish_list:
            fish = str(fish)
            print(f"  Fish: {fish}")

            try:
                neural_fb = load_neural_data(fb_data_path, fish, end="_fb")
                fb_results = process_one_dataset(
                    neural_data=neural_fb,
                    regressor=regressor,
                    neuro_stimuli=neuro_stimuli,
                    dataset_name="forebrain"
                )
            except Exception as e:
                print(f"    Forebrain failed: {e}")
                fb_results = pd.DataFrame()

            try:
                neural_hb = load_neural_data(hb_data_path, fish, end="_hb")
                hb_results = process_one_dataset(
                    neural_data=neural_hb,
                    regressor=regressor,
                    neuro_stimuli=neuro_stimuli,
                    dataset_name="hindbrain"
                )
            except Exception as e:
                print(f"    Hindbrain failed: {e}")
                hb_results = pd.DataFrame()

            if not fb_results.empty:
                all_corr.append(fb_results)
            if not hb_results.empty:
                all_corr.append(hb_results)

        if len(all_corr) == 0:
            print(f"  No results for stimulus {stim}\n")
            continue

        all_corr_df = pd.concat(all_corr, ignore_index=True)

        output_path = f"{output_dir}/7dpf_{stim}_correlations_25s_1000.csv"
        all_corr_df.to_csv(output_path, index=False)
        print(f"  Saved: {output_path}\n")


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    with open(REGRESSOR_PATH, "rb") as f:
        stim_regressors = pickle.load(f)

    main_stim(
        fish_list=FISH_LIST,
        stim_regressors=stim_regressors,
        neuro_stimuli=NEURO_STIMULI,
        fb_data_path=FB_DATA_PATH,
        hb_data_path=HB_DATA_PATH,
        output_dir=OUTPUT_DIR
    )

    print("Analysis complete!")