from flask import Flask, render_template, request
from psycopg2 import sql
from psycopg2.extras import execute_values
from create_db import connect_to_db
import numpy as np
import pandas as pd
import json
from datetime import datetime

# Use hacky relative module import until I create a proper package
# import sys
# sys.path.append("../gaitkeeper")
# import load
# import preprocess
# import model


# Pre-app initialization (load relevant data into memory):
conn = connect_to_db("gaitkeeper", "postgres")
print("Loading reference dataset...")
# df_ref = preprocess.create_reference_data_features_from_fft_peaks(10)

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


@app.route("/", methods=["GET"])
def home_page():
    return render_template("index.html")


@app.route("/record", methods=["GET"])
def record_page():
    return render_template("record.html")


@app.route("/compare", methods=["GET"])
def compare_page(similarity=None):
    # Get list of recorded walks
    df_walks = pd.read_sql_query(f"SELECT name, date FROM walks ORDER BY date DESC", conn)
    df_walks = df_walks[df_walks["name"] != ""]  # Ignore walks with no name
    walk_names = df_walks["date"].astype(str) + " - " + df_walks["name"]
    return render_template("compare.html", 
        walk_names=walk_names,
        similarity=similarity
    )


@app.route("/score", methods=["POST"])
def calculate_similarity():
    # Compute similarity score
    return compare_page(similarity=0.9)


@app.route("/upload", methods=["POST"])
def upload_recording():
    # `request.form[sensor]` variable contains form data for each sensor type
    # TODO: Check validity of data in backend
    # TODO: Output success message once uploaded
    user_id, walk_id = commit_recording_to_db(request.form)
    return compare_page()


if __name__ == "__main__":
    # Using a self-signed cert for SSL; need for sensor APIs to work
    app.run(host="0.0.0.0", debug=True, ssl_context=("cert.pem", "key.pem"))
