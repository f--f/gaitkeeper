"""
Flask web server for Gaitkeeper web app.
Ensure environment variables GAITKEEPER_DB_HOST and GAITKEEPER_DB_PASSWORD are set if deploying to AWS.
Start server on AWS with: gunicorn server:app -D
"""

from flask import Flask, render_template, request
from psycopg2 import sql
from psycopg2.extras import execute_values
from create_db import connect_to_db
import numpy as np
import pandas as pd
import json
from datetime import datetime

# Use hacky relative module import until I create a proper package
import os
import sys
sys.path.append("../gaitkeeper")
from load import get_walk_data_from_database
from torch.utils.data import DataLoader
from preprocess import generate_walk_chunks
import models
from sklearn.metrics.pairwise import cosine_similarity


# Pre-app initialization (load relevant data into memory):
host = os.getenv("GAITKEEPER_DB_HOST", "localhost")
model = models.load_embedding_model("../models/Classification-trained_EmbeddingNet.pt")
conn = connect_to_db(host, "gaitkeeper", "postgres", os.getenv("GAITKEEPER_DB_PASSWORD", None))

# Create the application object
app = Flask(__name__)


def commit_recording_to_db(formdata):
    """Commit uploaded formdata of recording to database."""
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO walks (name, date)
        VALUES (%s, %s)
        RETURNING walk_id
    """, (formdata["name"], datetime.now().date().isoformat()))
    walk_id = cur.fetchone()[0]
    for sensor, data in formdata.items():
        # Ignore metadata fields, etc.
        if sensor in ("linearaccelerometer", "gyroscope"):
            data = json.loads(data)
            time = data["time"]
            values = data["value"]
            # execute_values is optimized vs. executemany
            # use psycopg2.sql to prevent injection
            insert_query = sql.SQL("""
                INSERT INTO {table} (walk_id, timestamp, {x}, {y}, {z})
                VALUES %s
            """).format(
                table=sql.Identifier(sensor),
                x=sql.Identifier(f"{sensor}_x"),
                y=sql.Identifier(f"{sensor}_y"),
                z=sql.Identifier(f"{sensor}_z")
            )
            execute_values(cur, insert_query, 
                [(walk_id, time[i], v[0], v[1], v[2]) for i, v in enumerate(values)])
    conn.commit()
    cur.close()
    app.logger.info("Walk %d (%s) successfully committed to database!", walk_id, formdata["name"])
    return walk_id


def get_walk_embedding(walk_id):
    # Convert walk data to a dataset
    df = get_walk_data_from_database(conn, walk_id, hz=50, pad=3, truncate_seconds=3)
    # Normalize sensor data
    for logtype in ("linearaccelerometer", "gyroscope"):
        df[f"{logtype}_mag"] = np.linalg.norm(df[[f"{logtype}_x", f"{logtype}_y", f"{logtype}_z"]].values, axis=1)
        df[f"{logtype}_mag"] = (df[f"{logtype}_mag"] - df[f"{logtype}_mag"].mean()) / (df["linearaccelerometer_mag"].quantile(q=0.99) - df["linearaccelerometer_mag"].quantile(q=0.01))
    df["user_id"] = -1  # Use filler user ID value
    chunks = list(generate_walk_chunks(df, chunksize=128, window_step=128))
    dataset = models.GaitDataset(chunks)
    dataloader = DataLoader(dataset, batch_size=64)
    return models.extract_embeddings(dataloader, model)


@app.route("/", methods=["GET"])
def home_page():
    return render_template("index.html")


@app.route("/record", methods=["GET"])
def record_page():
    return render_template("record.html")


@app.route("/compare", methods=["GET"])
def compare_page(similarity=None, selected_walk_ids=None):
    # Get list of recorded walks
    df_walks = pd.read_sql_query(f"SELECT * FROM walks ORDER BY walk_id DESC", conn).set_index("walk_id")
    df_walks = df_walks[df_walks["name"] != ""]  # Ignore walks with no name
    df_walks["frontend_name"] = df_walks["date"].astype(str) + " - " + df_walks["name"]
    if selected_walk_ids is None:  # If walks were not selected, choose first walk_id in list
        selected_walk_ids = (df_walks.index[0], df_walks.index[0])
    return render_template("compare.html", 
        walk_dict=df_walks["frontend_name"].to_dict(),
        selected_walk_ids=selected_walk_ids,
        similarity=similarity
    )


@app.route("/score", methods=["POST"])
def calculate_similarity():
    walk1_id = int(request.form["walk1"])
    walk2_id = int(request.form["walk2"])
    walk1_embedding = get_walk_embedding(walk1_id)
    walk2_embedding = get_walk_embedding(walk2_id)
    similarity = cosine_similarity(walk1_embedding, walk2_embedding).mean()
    # TODO: Return stddev
    return compare_page(similarity=similarity, selected_walk_ids=(walk1_id, walk2_id))


@app.route("/upload", methods=["POST"])
def upload_recording():
    # `request.form[sensor]` variable contains form data for each sensor type
    # TODO: Check validity of data in backend
    # TODO: Output success message once uploaded
    walk_id = commit_recording_to_db(request.form)
    return compare_page()


if __name__ == "__main__":
    # Using a self-signed cert for SSL; need for sensor APIs to work
    # app.run(host="0.0.0.0", debug=False, ssl_context=("cert.pem", "key.pem"))
    # With nginx+gunicorn:
    app.run()