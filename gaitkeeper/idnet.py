from tqdm.auto import tqdm
import pandas as pd
import numpy as np

from .constants import IDNET_PATH
from .load import get_reference_data
from .preprocess import generate_walk_chunks, normalize_sensor_data


def load_idnet_dataset(chunksize, window_step, normalize="overall", hz=50, pad=3):
    """Return data from IDNet dataset as a list of DataFrames.
    This takes the original time-series data, cleans it, and splits it up into individual samples.
    Arguments:
        chunksize: number of timepoints to take for each chunk
        window_step: size of sliding window (set equal to chunksize for no overlap)
        normalize: "overall" or "within" - this 
        hz: frequency to resample the data at
        pad: number of timepoints to backfill values if there is missing data
    Returns:
        chunks: list of 
        uid_to_labels: dictionary of user IDs from IDNet to consecutively-ordered class labels.
    """
    print("Loading IDNet dataset...")
    # Get number of walks per user
    counts = {}
    for folder in IDNET_PATH.glob("*"):
        user_id = int(folder.stem[1:4])
        if user_id not in counts:
            counts[user_id] = 1
        else:
            counts[user_id] += 1
    users_with_multiple_walks = [user for user, count in counts.items() if count > 1]

    dfs = []
    for folder in tqdm(list(IDNET_PATH.glob("*"))):
        user_id = int(folder.stem[1:4])
        walk_id = int(folder.stem[6:9])
        if (user_id not in users_with_multiple_walks):
            continue  # Limit dataset to 2 walks for individuals with more than 1 walk
        df = get_reference_data(user_id, walk_id, hz=hz, pad=pad).reset_index()
        df.insert(0, "walk_id", walk_id)
        df.insert(0, "user_id", user_id)

        # Normalize relevant features (magnitudes) so that they're orientation-invariant
        # Do this within each walk vs. over entire dataset?
        if normalize == "within":
            for logtype in ["linearaccelerometer", "gyroscope"]:
                df[f"{logtype}_mag"] = np.linalg.norm(df[[f"{logtype}_x_data", f"{logtype}_y_data", f"{logtype}_z_data"]].values, axis=1)
                df[f"{logtype}_mag"] = (df[f"{logtype}_mag"] - df[f"{logtype}_mag"].mean()) / (df["linearaccelerometer_mag"].quantile(q=0.99) - df["linearaccelerometer_mag"].quantile(q=0.01))

        dfs.append(df)
    df = pd.concat(dfs)
    
    if normalize == "overall":
        for logtype in ["linearaccelerometer", "gyroscope"]:
            df[f"{logtype}_mag"] = np.linalg.norm(df[[f"{logtype}_x_data", f"{logtype}_y_data", f"{logtype}_z_data"]].values, axis=1)
            df[f"{logtype}_mag"] = (df[f"{logtype}_mag"] - df[f"{logtype}_mag"].mean()) / (df["linearaccelerometer_mag"].quantile(q=0.99) - df["linearaccelerometer_mag"].quantile(q=0.01))

    # Split into chunks
    chunks = []
    for (user_id, walk_id), subdf in df.groupby(["user_id", "walk_id"]):
        user_chunks = list(generate_walk_chunks(subdf, chunksize=chunksize, window_step=window_step))
        chunks.extend(user_chunks)
    
    # Replace uid with 0-indexed version
    curr_uid = None
    curr_label = -1
    uid_to_label = {}
    for i,chunk in enumerate(chunks):
        # Increment label on new user
        if curr_uid != chunk["user_id"].iloc[0]:
            curr_uid = chunk["user_id"].iloc[0]
            curr_label += 1
            uid_to_label[curr_uid] = curr_label
        chunks[i]["user_id"] = curr_label
    return chunks, uid_to_label