import pandas as pd
import numpy as np
from scipy.fft import fft
from scipy.signal import find_peaks
from tqdm.auto import tqdm

from .load import get_reference_data
from .constants import f_s, IDNET_PATH

def generate_walk_chunks(df, chunksize=512, window_step=256, is_valid=True):
    """Split an input DataFrame into multiple chunks of data. 
    Arguments:
        df: input DataFrame to split
        chunksize: number of rows for each output chunk.
            Recommended to be power-of-2 if doing downstream FFT.
        window_step: sliding window size (set less than chunksize for overlaps)
        is_valid: if True, this yields only non-NAN data (any chunks with skips are ignored)
    Yields:
        subdf: chunks of the original DataFrame.
    """
    assert window_step <= chunksize
    count = 0
    while count < (len(df) - chunksize):  # While there are still chunksize rows remaining
        subdf = df.iloc[count:count + chunksize, :]
        if len(subdf) == chunksize and not subdf.isna().any(axis=None):  # Return only non-NA
            yield subdf.reset_index().copy()
        count += window_step


def normalize_sensor_data(df, logtype):
    norm = np.linalg.norm(df[[f"{logtype}_x_data", f"{logtype}_y_data", f"{logtype}_z_data"]].values, axis=1)
    norm = (norm - norm.mean()) / (np.percentile(norm, 99) - np.percentile(norm, 1))
    return norm


def get_fft(signal, f_s):
    """f_s = sampling rate (measurements/second)"""
    T = 1/f_s
    N = len(signal)
    f_values = np.linspace(0.0, 1.0/(2.0*T), N//2)
    fft_values_ = fft(signal)
    fft_values = 2.0/N * np.abs(fft_values_[0:N//2])  # take abs (remove phase component)
    return f_values, fft_values


def get_top_signal_peaks(x, y, n):
    peak_idx, peak_props = find_peaks(y, height=0)  # Specify height to force peak height computation
    peak_heights, peak_idx = zip(*sorted(zip(peak_props["peak_heights"], peak_idx), reverse=True)[:n])
    return x[list(peak_idx)], np.array(peak_heights)


def create_reference_data_features_from_fft_peaks(n_peaks=10):
    """Create DataFrame of feature vectors using Fourier peaks."""
    
    counts = {}
    for folder in IDNET_PATH.glob("*"):
        user_id = int(folder.stem[1:4])
        if user_id not in counts:
            counts[user_id] = 1
        else:
            counts[user_id] += 1
    users_with_multiple_walks = [user for user, count in counts.items() if count > 1]
    
    features = []
    for user in tqdm(users_with_multiple_walks, desc="User"):
        for walk in range(1, counts[user]+1):
            df = get_reference_data(user, walk)
            for chunk in generate_walk_chunks(df):
                # TODO: Refactor out [user, walk] - share code with chunk
                norm_acc = normalize_sensor_data(chunk, "linearaccelerometer")
                norm_gyro = normalize_sensor_data(chunk, "gyroscope")
                f_acc, fft_acc = get_fft(norm_acc, f_s)
                peak_f_acc, peak_fft_acc = get_top_signal_peaks(f_acc, fft_acc, n_peaks)
                f_gyro, fft_gyro = get_fft(norm_gyro, f_s)
                peak_f_gyro, peak_fft_gyro = get_top_signal_peaks(f_gyro, fft_gyro, n_peaks)
                # concatenate the features
                feature_vector = np.concatenate([[user, walk], peak_f_acc, peak_fft_acc, peak_f_gyro, peak_fft_gyro])
                features.append(feature_vector)

    df_features = pd.DataFrame(features, 
        columns=["user_id", "walk_id", 
               *[f"acc_f{i}" for i in range(n_peaks)], *[f"acc_fft{i}" for i in range(n_peaks)],
               *[f"gyro_f{i}" for i in range(n_peaks)], *[f"gyro_fft{i}" for i in range(n_peaks)]
    ])
    df_features["user_id"] = df_features["user_id"].astype(int)
    df_features["walk_id"] = df_features["walk_id"].astype(int)
    return df_features


def create_fft_peak_features_from_chunk(chunk, n_peaks, f_s=60):
    norm_acc = normalize_sensor_data(chunk, "linearaccelerometer")
    norm_gyro = normalize_sensor_data(chunk, "gyroscope")
    f_acc, fft_acc = get_fft(norm_acc, f_s)
    peak_f_acc, peak_fft_acc = get_top_signal_peaks(f_acc, fft_acc, n_peaks)
    f_gyro, fft_gyro = get_fft(norm_gyro, f_s)
    peak_f_gyro, peak_fft_gyro = get_top_signal_peaks(f_gyro, fft_gyro, n_peaks)
    # concatenate the features
    feature_vector = np.concatenate([peak_f_acc, peak_fft_acc, peak_f_gyro, peak_fft_gyro])
    return feature_vector