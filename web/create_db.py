"""
Script to bootstrap PostgreSQL database with necessary tables.

Before running this, create a new database in the CLI:
    createdb -U postgres gaitkeeper
"""

import psycopg2 as pg
from getpass import getpass

# Connect into a new database
try:
    conn = pg.connect(database="gaitkeeper",
                      user="postgres",
                      password=getpass("Enter password for user [postgres]: "))
except pg.OperationalError:
    print("** Failed to connect to database! ** Error: \n")
    raise

cur = conn.cursor()
# Note 'serial' is auto-generated integer (autoincrement)
# Table for storing user data (a unique person)
try:
    cur.execute("""
        CREATE TABLE users (
            user_id    serial   PRIMARY KEY,
            name       varchar
        )
    """)
    # Table for storing walk data (one user can have multiple walks)
    cur.execute("""
        CREATE TABLE walks (
            walk_id    serial   PRIMARY KEY,
            user_id    int      references users(user_id)
        )
    """)
    # Tables for storing data from sensors
    # Need multiple tables (one per sensor object) because timestamps/frequencies can be different
    # double precision reflects precision of objects from browser
    for sensor in ("linearaccelerometer", "gyroscope"):
        cur.execute(f"""
            CREATE TABLE {sensor} (
                {sensor}_id  serial   PRIMARY KEY,
                walk_id      int      references walks(walk_id),
                timestamp    double precision,
                {sensor}_x   double precision,
                {sensor}_y   double precision,
                {sensor}_z   double precision
            )
        """)
except pg.errors.DuplicateTable:
    print("** One or more tables already exists! ** Error: \n")
    raise

# Commit changes and close database connection
conn.commit()
cur.close()
conn.close()
print("Successfully wrote tables!")