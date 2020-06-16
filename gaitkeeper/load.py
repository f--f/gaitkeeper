from pathlib import Path
import pandas as pd
import numpy as np
import json
from tqdm.auto import tqdm

IDNET_PATH = Path(r"C:\Users\Fred\Documents\Insight\Gait Datasets\IDNet Database\IDNet_dataset")

def _resample_merged_data(merged_df, hz=60, pad=3, truncate_seconds=10):
    # Truncate each dataframe based on latest start / earliest end
    # Alternatively, truncate up to 10s from beginning and end to exclude for start/stop motions
    latest_start_time = max(merged_df.notna().idxmax().max(), merged_df.index[0] + pd.Timedelta(f"{truncate_seconds}s"))
    earliest_end_time = min(merged_df.notna()[::-1].idxmax().min(), merged_df.index[-1] - pd.Timedelta(f"{truncate_seconds}s"))
    # Sensors have different sampling freqs so resample
    merged_df = merged_df.resample(f"{round(1/hz * 1e9)}N").mean()
    merged_df = merged_df.loc[latest_start_time: earliest_end_time].copy()
    # Impute NaN values which are just resulting from a single skip
    merged_df = merged_df.fillna(method='pad', limit=pad)
    # Convert timestamp to relative times
    merged_df = merged_df.set_index(merged_df.index - merged_df.index[0])
    return merged_df


def get_reference_data(user_id, walk_num, hz=60, pad=3):
    """Return DataFrame of walk sensor data from IDNet individual/walk at 10ms intervals (resampled).
    Note the data may include NaN values since sensors are sampled at different frequencies or may skip measurements."""
    # Retrieve folder
    walkid = f"u{str(user_id).zfill(3)}_w{str(walk_num).zfill(3)}"
    folder = IDNET_PATH / walkid
    # For each sensor log, construct df from file
    dfs = []
    for log in folder.glob("*.log"):
        logtype = log.stem.split("_")[-1]  # accelerometer, gyroscope, etc.
        if logtype == "magnetometer":
            continue  # Skip magnetometer since not probably not useful
        df = pd.read_csv(log, sep="\t")
        # Read timestamp column as pandas Timedelta, starting from 0 ns
        df[f"{logtype}_timestamp"] = pd.to_timedelta(df[f"{logtype}_timestamp"], unit="ns")
        df = df.rename(columns={f"{logtype}_timestamp": "timestamp"})
        df = df.set_index("timestamp")
        dfs.append(df)
    merged_df = pd.concat(dfs, axis="columns")
    return _resample_merged_data(merged_df, hz=hz, pad=pad)


def read_form_data(formdata, hz=60):
    """Return DataFrame of walk sensor data from demo app form."""
    dfs = []
    for log, data in formdata.items():
        data = json.loads(data)
        time = data["time"]
        values = np.array(data["value"])
        df = pd.DataFrame({"timestamp": data["time"],
            f"{log}_x_data": values[:,0],
            f"{log}_y_data": values[:,1],
            f"{log}_z_data": values[:,2],
        })
        # Note timestamp of DOMHighResTimeStamp is milliseconds
        df["timestamp"] = pd.to_timedelta(df["timestamp"], unit="ms")
        df = df.set_index("timestamp")
        dfs.append(df)
        # import pdb; pdb.set_trace()
    merged_df = pd.concat(dfs, axis="columns")
    return _resample_merged_data(merged_df, hz=hz)