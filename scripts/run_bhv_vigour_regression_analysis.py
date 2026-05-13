"""
Behavioral-Regressor Correlation Analysis

Calculates correlations between neural activity and behavioral vigor,
with significance testing via trial-shuffle permutation testing.

Behavioral vigor is computed from tail angle traces using running standard
deviation (includes all movement: swim bouts + postural adjustments),
then convolved with a calcium indicator kernel to match neural dynamics.

This version is optimized by:
1. Computing correlations for all neurons at once using matrix operations
2. Computing shuffled correlations for all neurons at once
3. Avoiding neuron-by-neuron / shuffle-by-shuffle Python loops
"""

import numpy as np
import pandas as pd
import json
from ast import literal_eval
from scipy.signal import butter, filtfilt, convolve
import warnings

warnings.filterwarnings("ignore")


# ============================================================================
# CONFIGURATION
# ============================================================================

# Fish identifiers
FISH_LIST = [230713, 230714, 230720, 230727, 230728,
             230810, 230811, 230817, 230818, 230919]

# Stimuli to analyze
NEURO_STIMULI = ['ade', 'cad_2.5mm', 'fex_1', 'kw', 'ph4.5', 'pro_2.5mm', 'qui_2.5mm']

# File paths
FB_BHV_PATH = "/Volumes/LaCie/larval_HuC/behavior/7dpf_fb_behavior_traces.csv"
HB_BHV_PATH = "/Volumes/LaCie/larval_HuC/behavior/7dpf_hb_behavior_traces.csv"
FB_NEURAL_PATH = "/Volumes/LaCie/larval_HuC/imaging/7dpf_fb_fish_dfs_final.h5"
HB_NEURAL_PATH = "/Volumes/LaCie/larval_HuC/imaging/7dpf_hb_fish_dfs_final.h5"
OUTPUT_PATH = "/Volumes/LaCie/larval_HuC/imaging/7dpf_bhv_vigor_all_correlations_1000.csv"

# Behavioral processing parameters
ORIGINAL_FREQ = 60.85  # Hz
TARGET_FREQ = 3        # Hz
WINDOW_SIZE_MS = 100   # ms for running std

# Calcium indicator kernel parameters
SAMPLING_RATE = 3
TAU_RISE = 1.44 / SAMPLING_RATE
TAU_DECAY = 5.05 / SAMPLING_RATE

# Correlation analysis parameters
NUM_SHUFFLES = 1000
TRIAL_LENGTH = 300

# Reproducibility
RANDOM_SEED = 42


# ============================================================================
# CALCIUM INDICATOR KERNEL
# ============================================================================

def calcium_indicator_kernel(t, tau_rise, tau_decay):
    """Generate normalized calcium indicator kernel."""
    kernel = np.exp(-t / tau_decay) - np.exp(-t / tau_rise)
    kernel_sum = np.sum(kernel)
    if kernel_sum == 0:
        raise ValueError("Calcium kernel sum is zero.")
    return kernel / kernel_sum


def convolve_trace(trace, kernel):
    """Convolve trace with calcium kernel and preserve original length."""
    convolved = convolve(trace, kernel, mode="full")
    return convolved[:len(trace)]


# ============================================================================
# BEHAVIORAL PREPROCESSING
# ============================================================================

def concatenate_row(row):
    """Concatenate array-like entries in a row, ignoring missing values."""
    arrays = [x for x in row.values if isinstance(x, np.ndarray)]
    if len(arrays) == 0:
        return np.array([])
    return np.concatenate(arrays)


def downsample_trace(trace, original_freq, target_freq):
    """
    Downsample trace with anti-aliasing filter.
    """
    decimation_factor = int(original_freq / target_freq)

    if decimation_factor <= 1:
        raise ValueError("Target frequency must be lower than original frequency")

    nyquist_rate = original_freq / 2.0
    cutoff = target_freq / 2.0
    b, a = butter(4, cutoff / nyquist_rate)

    filtered_trace = filtfilt(b, a, trace)
    downsampled_trace = filtered_trace[::decimation_factor]

    return downsampled_trace


def running_std(trace, window_size_samples):
    """
    Calculate running standard deviation over trace.
    """
    trace = np.asarray(trace, dtype=float)
    running_std_series = pd.Series(trace).rolling(window=window_size_samples).std()
    running_std_series = running_std_series.fillna(0)
    return running_std_series.to_numpy()


def load_and_process_data(data, original_freq, target_freq, window_size_samples, kernel):
    """
    Process behavioral data into one concatenated convolved vigor regressor per fish.
    """
    data = data.copy()

    # Parse JSON traces
    data["bhv_trace"] = data["angles"].apply(lambda x: np.asarray(json.loads(x)))

    # Calculate vigor
    data["bhv_vigour"] = data["bhv_trace"].apply(
        lambda trace: running_std(trace, window_size_samples)
    )

    # Downsample to neural imaging rate
    data["downsampled_vigour"] = data["bhv_vigour"].apply(
        lambda trace: downsample_trace(trace, original_freq, target_freq)
    )

    # Convolve with calcium kernel
    data["convolved_vigour"] = data["downsampled_vigour"].apply(
        lambda trace: convolve_trace(np.abs(trace), kernel)
    )

    # Optional filter if needed
    if "stimulus" in data.columns:
        data = data[data["stimulus"].isin(NEURO_STIMULI)].copy()

    # Pivot by stimulus/trial and concatenate
    vigor_df = data.pivot_table(
        index="fish_id",
        columns=["stimulus", "trial_number"],
        values="convolved_vigour",
        aggfunc="first"
    )

    vigor_df["concat_trace"] = vigor_df.apply(concatenate_row, axis=1)

    if vigor_df.shape[0] == 0:
        raise ValueError("No behavioral vigor regressor created for this fish.")

    regressor = vigor_df["concat_trace"].iloc[0]

    if len(regressor) == 0:
        raise ValueError("Concatenated behavioral vigor regressor is empty.")

    return regressor


# ============================================================================
# NEURAL DATA FUNCTIONS
# ============================================================================

def load_neural_data(file_path, fish, end):
    """
    Load neural data for one fish from HDF5.

    Requires fish_id to be queryable in the HDF store.
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

    if "stimulus" in neural_data.columns:
        neural_data = neural_data[neural_data["stimulus"].isin(NEURO_STIMULI)].copy()

    return neural_data


def concatenate_neural_data(neural_data):
    """
    Concatenate neural traces across all stimuli and trials for each neuron.
    """
    pivoted_df = neural_data.pivot_table(
        index=["fish_id", "neuron_id", "coords_serialised", "area"],
        columns=["stimulus", "trial_number"],
        values="img_trace",
        aggfunc="first"
    )

    pivoted_df["concat_trace"] = pivoted_df.apply(concatenate_row, axis=1)
    pivoted_df = pivoted_df.reset_index()

    pivoted_df = pivoted_df[pivoted_df["concat_trace"].apply(lambda x: len(x) > 0)].copy()

    if pivoted_df.empty:
        raise ValueError("No valid concatenated neural traces found.")

    return pivoted_df


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
            f"trial_length ({trial_length})."
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

    Rz = zscore_rows(shuffled_regressors)  # shuffles x time

    shuffle_corrs = (Xz @ Rz.T) / Xz.shape[1]  # neurons x shuffles

    if two_tailed:
        p_values = np.mean(np.abs(shuffle_corrs) >= np.abs(real_corr[:, None]), axis=1)
    else:
        p_values = np.mean(shuffle_corrs >= real_corr[:, None], axis=1)

    key_data["regressor_corr"] = real_corr
    key_data["p_value"] = p_values

    return key_data


def create_significance_results_df(neuro_data):
    """Create clean results DataFrame."""
    neuro_data = neuro_data.copy()
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

def process_one_dataset(neural_path, fish, end, regressor, dataset_name):
    """Process one dataset (fb or hb) for one fish."""
    print(f"  Loading {dataset_name} neural data...")
    neural_data = load_neural_data(neural_path, fish, end=end)

    print(f"  Concatenating {dataset_name} neural traces...")
    key_data = concatenate_neural_data(neural_data)

    print(f"  Computing {dataset_name} correlations and p-values...")
    key_data = compute_correlations_and_pvalues_fast(
        key_data=key_data,
        regressor=regressor,
        num_shuffles=NUM_SHUFFLES,
        trial_length=TRIAL_LENGTH,
        random_seed=RANDOM_SEED,
        two_tailed=False
    )

    results = create_significance_results_df(key_data)
    return results


def main_bhv(fish_list, fb_data, hb_data, kernel,
             fb_neural_path, hb_neural_path, output_path):
    """
    Run behavioral vigor-neural correlation analysis across all fish.
    """
    all_corr = []

    window_size_samples = int(ORIGINAL_FREQ * (WINDOW_SIZE_MS / 1000.0))

    for fish in fish_list:
        print(f"\nProcessing fish: {fish}")

        fish_fb = fb_data[fb_data["fish_id"] == fish].copy()
        fish_hb = hb_data[hb_data["fish_id"] == fish].copy()

        if fish_fb.empty:
            print(f"  Skipping {fish}: no forebrain behavior data.")
            continue
        if fish_hb.empty:
            print(f"  Skipping {fish}: no hindbrain behavior data.")
            continue

        print("  Processing forebrain behavior...")
        fb_bhv_vigour = load_and_process_data(
            fish_fb,
            original_freq=ORIGINAL_FREQ,
            target_freq=TARGET_FREQ,
            window_size_samples=window_size_samples,
            kernel=kernel
        )

        print("  Processing hindbrain behavior...")
        hb_bhv_vigour = load_and_process_data(
            fish_hb,
            original_freq=ORIGINAL_FREQ,
            target_freq=TARGET_FREQ,
            window_size_samples=window_size_samples,
            kernel=kernel
        )

        try:
            fb_results = process_one_dataset(
                neural_path=FB_NEURAL_PATH,
                fish=str(fish),
                end="_fb",
                regressor=fb_bhv_vigour,
                dataset_name="forebrain"
            )
        except Exception as e:
            print(f"  Forebrain failed for fish {fish}: {e}")
            fb_results = pd.DataFrame()

        try:
            hb_results = process_one_dataset(
                neural_path=HB_NEURAL_PATH,
                fish=str(fish),
                end="_hb",
                regressor=hb_bhv_vigour,
                dataset_name="hindbrain"
            )
        except Exception as e:
            print(f"  Hindbrain failed for fish {fish}: {e}")
            hb_results = pd.DataFrame()

        if not fb_results.empty:
            all_corr.append(fb_results)
        if not hb_results.empty:
            all_corr.append(hb_results)

        if len(all_corr) > 0:
            temp_df = pd.concat(all_corr, ignore_index=True)
            temp_df.to_csv(output_path, index=False)
            print(f"  Saved progress to: {output_path}")

    if len(all_corr) == 0:
        raise ValueError("No results generated.")

    final_df = pd.concat(all_corr, ignore_index=True)
    final_df.to_csv(output_path, index=False)
    print(f"\nFinal results saved to: {output_path}")


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("Loading behavioral data...")
    fb_data = pd.read_csv(FB_BHV_PATH)
    hb_data = pd.read_csv(HB_BHV_PATH)

    print("Generating calcium indicator kernel...")
    t = np.arange(0, 10, 1 / SAMPLING_RATE)
    kernel = calcium_indicator_kernel(t, TAU_RISE, TAU_DECAY)

    print(f"\nAnalyzing {len(FISH_LIST)} fish...")
    main_bhv(
        fish_list=FISH_LIST,
        fb_data=fb_data,
        hb_data=hb_data,
        kernel=kernel,
        fb_neural_path=FB_NEURAL_PATH,
        hb_neural_path=HB_NEURAL_PATH,
        output_path=OUTPUT_PATH
    )

    print("\nAnalysis complete!")