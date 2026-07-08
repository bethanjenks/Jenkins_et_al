"""
Straight Bout-Regressor Correlation Analysis

Calculates correlations between neural activity and straight swim bouts,
with significance testing via trial-shuffle permutation testing.

Straight bouts are classified using the circular mean of tail angles during the
first 70 ms of each bout. A binary regressor marks frames during straight bouts
(1 = straight bout, 0 = otherwise), which is then downsampled to the neural
sampling rate and convolved with a calcium indicator kernel.

This version is optimized by:
1. Computing correlations for all neurons at once using matrix operations
2. Computing shuffled correlations for all neurons at once
3. Avoiding neuron-by-neuron / shuffle-by-shuffle Python loops
"""

import numpy as np
import pandas as pd
import json
from scipy.signal import convolve
from scipy.stats import circmean
import warnings

warnings.filterwarnings("ignore")


# ============================================================================
# CONFIGURATION
# ============================================================================

# Fish identifiers
FISH_LIST = [230713, 230714, 230720, 230727, 230728,
             230810, 230811, 230817, 230818, 230919]

# File paths
FB_BHV_PATH = "/Volumes/LaCie/larval_HuC/behavior/7dpf_fb_behavior_traces.csv"
HB_BHV_PATH = "/Volumes/LaCie/larval_HuC/behavior/7dpf_hb_behavior_traces.csv"
FB_NEURAL_PATH = "/Volumes/LaCie/larval_HuC/imaging/7dpf_fb_fish_dfs_final.h5"
HB_NEURAL_PATH = "/Volumes/LaCie/larval_HuC/imaging/7dpf_hb_fish_dfs_final.h5"
OUTPUT_PATH = "/Volumes/LaCie/larval_HuC/imaging/7dpf_new_straight_bout_all_correlations_1000.csv"

# Behavioral processing parameters
ORIGINAL_FREQ = 60.85   # Hz
TARGET_FREQ = 3         # Hz
BOUT_DURATION_MS = 70   # ms used for straight/turn classification
LEFT_THRESHOLD = -0.08  # radians
RIGHT_THRESHOLD = 0.08  # radians
EXPECTED_FRAMES_PER_TRIAL = 300

# Calcium indicator kernel parameters
SAMPLING_RATE = 3       # Hz
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
# BOUT CLASSIFICATION
# ============================================================================

def concatenate_row(row):
    """Concatenate array-like entries in a row, ignoring missing values."""
    arrays = [x for x in row.values if isinstance(x, np.ndarray)]
    if len(arrays) == 0:
        return np.array([])
    return np.concatenate(arrays)


def classify_straight_bouts(row,
                            left_threshold=LEFT_THRESHOLD,
                            right_threshold=RIGHT_THRESHOLD,
                            sampling_rate=ORIGINAL_FREQ,
                            duration_ms=BOUT_DURATION_MS):
    """
    Classify each bout as straight (True) or turning/asymmetric (False).

    Assumes row['bout_indx'] contains a list of lists, where each inner list
    contains all indices belonging to a bout.
    """
    straight_flags = []
    bouts = row["bout_indx"]
    tail_angle = row["bhv_trace"]
    duration_frames = max(1, int((duration_ms / 1000) * sampling_rate))

    for bout in bouts:
        if len(bout) < 1:
            straight_flags.append(False)
            continue

        bout_start = bout[0]
        bout_end = min(bout_start + duration_frames, len(tail_angle))
        bout_angles = np.asarray(tail_angle[bout_start:bout_end])

        if len(bout_angles) == 0:
            straight_flags.append(False)
            continue

        laterality_index = circmean(bout_angles, high=np.pi, low=-np.pi)
        is_straight = left_threshold <= laterality_index <= right_threshold
        straight_flags.append(is_straight)

    return straight_flags


def create_straight_bout_regressor(total_frames, bouts, straight_flags):
    """
    Create binary regressor where straight bout frames are marked as 1.
    """
    regressor = np.zeros(total_frames, dtype=float)

    for bout, is_straight in zip(bouts, straight_flags):
        if is_straight and len(bout) > 0:
            bout_start = bout[0]
            bout_end = bout[-1]
            bout_start = max(0, int(bout_start))
            bout_end = min(total_frames - 1, int(bout_end))
            regressor[bout_start:bout_end + 1] = 1

    return regressor


def downsample_binary_regressor(regressor,
                                original_freq=ORIGINAL_FREQ,
                                target_freq=TARGET_FREQ,
                                expected_frames=EXPECTED_FRAMES_PER_TRIAL):
    """
    Downsample binary regressor using max pooling within each window.

    Note:
        This uses an integer approximation to the decimation factor.
        Since 60.85 / 3 is not an integer, this is approximate, but preserves
        your original workflow and ensures exactly expected_frames output.
    """
    decimation_factor = int(original_freq / target_freq)

    if decimation_factor <= 1:
        raise ValueError("Target frequency must be lower than original frequency.")

    downsampled = np.maximum.reduceat(
        regressor,
        np.arange(0, len(regressor), decimation_factor)
    )

    if len(downsampled) > expected_frames:
        downsampled = downsampled[:expected_frames]
    elif len(downsampled) < expected_frames:
        downsampled = np.pad(
            downsampled,
            (0, expected_frames - len(downsampled)),
            mode="constant"
        )

    return downsampled


# ============================================================================
# DATA PROCESSING
# ============================================================================

def load_and_process_data(data, kernel):
    """
    Process behavioral data into one concatenated convolved straight-bout regressor
    per fish.
    """
    data = data.copy()

    # Parse JSON strings
    data["bhv_trace"] = data["angles"].apply(lambda x: np.asarray(json.loads(x)))
    data["bout_indx"] = data["bouts"].apply(lambda x: json.loads(x))

    # Classify bouts
    data["straight_bout"] = data.apply(classify_straight_bouts, axis=1)

    # Create binary regressor
    data["straight_regressor"] = data.apply(
        lambda row: create_straight_bout_regressor(
            total_frames=len(row["bhv_trace"]),
            bouts=row["bout_indx"],
            straight_flags=row["straight_bout"]
        ),
        axis=1
    )

    # Downsample to neural imaging rate
    data["downsampled_straight_regressor"] = data["straight_regressor"].apply(
        downsample_binary_regressor
    )

    # Convolve with calcium kernel
    data["convolved_regressor"] = data["downsampled_straight_regressor"].apply(
        lambda trace: convolve_trace(trace, kernel)
    )

    # Pivot by stimulus/trial and concatenate
    regressor_df = data.pivot_table(
        index="fish_id",
        columns=["stimulus", "trial_number"],
        values="convolved_regressor",
        aggfunc="first"
    )

    regressor_df["concat_trace"] = regressor_df.apply(concatenate_row, axis=1)

    if regressor_df.shape[0] == 0:
        raise ValueError("No behavioral regressor created for this fish.")

    regressor = regressor_df["concat_trace"].iloc[0]

    if len(regressor) == 0:
        raise ValueError("Concatenated behavioral regressor is empty.")

    return regressor


def load_neural_data(file_path, fish, end):
    """
    Load neural data for one fish from HDF5.

    Requires fish_id to be queryable in the HDF store.
    """
    fish_id = f"{fish}{end}"

    try:
        neural_data = pd.read_hdf(file_path, where=f'fish_id == "{fish_id}"')
    except Exception:
        # Fallback: load full file then filter
        neural_data = pd.read_hdf(file_path)
        neural_data = neural_data[neural_data["fish_id"] == fish_id].copy()

    if neural_data.empty:
        raise ValueError(f"No neural data found for fish_id {fish_id}")

    neural_data = neural_data.copy()
    neural_data["img_trace"] = neural_data["trace_serialised"].apply(
        lambda x: np.asarray(json.loads(x))
    )

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

    # Drop neurons with empty traces
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

    Args:
        key_data: DataFrame containing 'concat_trace'
        regressor: 1D behavioral regressor
        num_shuffles: number of shuffled regressors
        trial_length: trial length in frames
        random_seed: random seed for reproducibility
        two_tailed: if True, uses abs(corr) for p-values

    Returns:
        DataFrame with added 'regressor_corr' and 'p_value'
    """
    key_data = key_data.copy().reset_index(drop=True)

    # Stack neural traces into matrix: neurons x time
    X = np.vstack(key_data["concat_trace"].values).astype(float)
    regressor = np.asarray(regressor, dtype=float)

    # Check length match
    if X.shape[1] != len(regressor):
        raise ValueError(
            f"Neural trace length ({X.shape[1]}) does not match regressor length ({len(regressor)})."
        )

    # Z-score neural traces and real regressor
    Xz = zscore_rows(X)
    rz = zscore_1d(regressor)

    # Real correlations for all neurons
    real_corr = np.nanmean(Xz * rz[None, :], axis=1)

    # Create shuffled regressors
    shuffled_regressors = shuffle_trials(
        regressor=regressor,
        trial_length=trial_length,
        num_shuffles=num_shuffles,
        random_seed=random_seed
    )

    # Z-score shuffled regressors row-wise
    Rz = zscore_rows(shuffled_regressors)  # shuffles x time

    # Correlation matrix: neurons x shuffles
    # Since rows are z-scored, Pearson r = mean(zx * zy)
    shuffle_corrs = (Xz @ Rz.T) / Xz.shape[1]

    if two_tailed:
        p_values = np.mean(np.abs(shuffle_corrs) >= np.abs(real_corr[:, None]), axis=1)
    else:
        p_values = np.mean(shuffle_corrs >= real_corr[:, None], axis=1)

    key_data["regressor_corr"] = real_corr
    key_data["p_value"] = p_values

    return key_data


def create_significance_results_df(neuro_data):
    """Create clean results DataFrame."""
    return pd.DataFrame({
        "correlation": neuro_data["regressor_corr"],
        "coords": neuro_data["coords_serialised"],
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
        two_tailed=False  # set True if you want two-sided testing
    )

    results = create_significance_results_df(key_data)
    return results


def main_bhv(fish_list, fb_data, hb_data, kernel,
             fb_neural_path, hb_neural_path, output_path):
    """
    Run straight bout-neural correlation analysis across all fish.
    """
    all_corr = []

    for fish in fish_list:
        print(f"\nProcessing fish: {fish}")

        # Behavioral subsets
        fish_fb = fb_data[fb_data["fish_id"] == fish].copy()
        fish_hb = hb_data[hb_data["fish_id"] == fish].copy()

        if fish_fb.empty:
            print(f"  Skipping {fish}: no forebrain behavior data.")
            continue
        if fish_hb.empty:
            print(f"  Skipping {fish}: no hindbrain behavior data.")
            continue

        # Build regressors
        print("  Processing forebrain behavior...")
        fb_straight_regressor = load_and_process_data(fish_fb, kernel)

        print("  Processing hindbrain behavior...")
        hb_straight_regressor = load_and_process_data(fish_hb, kernel)

        # Process neural data
        try:
            fb_results = process_one_dataset(
                neural_path=fb_neural_path,
                fish=str(fish),
                end="_fb",
                regressor=fb_straight_regressor,
                dataset_name="forebrain"
            )
        except Exception as e:
            print(f"  Forebrain failed for fish {fish}: {e}")
            fb_results = pd.DataFrame()

        try:
            hb_results = process_one_dataset(
                neural_path=hb_neural_path,
                fish=str(fish),
                end="_hb",
                regressor=hb_straight_regressor,
                dataset_name="hindbrain"
            )
        except Exception as e:
            print(f"  Hindbrain failed for fish {fish}: {e}")
            hb_results = pd.DataFrame()

        if not fb_results.empty:
            all_corr.append(fb_results)
        if not hb_results.empty:
            all_corr.append(hb_results)

        # Optional: save incrementally after each fish
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