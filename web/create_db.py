"""
Script to bootstrap PostgreSQL database with necessary tables.

Before running this, create a new database in the CLI:
    createdb -U postgres gaitkeeper
"""

import psycopg2
from getpass import getpass


def connect_to_db(database, user):
    """Return a PostgreSQL database connection."""
    try:
        conn = psycopg2.connect(database=database,
                        user=user,
                        password=getpass(f"Enter password for user [{user}]: "))
        return conn
    except psycopg2.OperationalError:
        print("** Failed to connect to database! ** Error: \n")
        raise


def initialize_tables(conn):
    """Create initial tables for gaitkeeper."""
    cur = conn.cursor()
    # Note 'serial' is auto-generated integer (autoincrement)
    try:
        # Table for storing walk data
        cur.execute("""
            CREATE TABLE walks (
                walk_id    serial   PRIMARY KEY,
                name       varchar,
                date       date
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
    except psycopg2.errors.DuplicateTable:
        print("** One or more tables already exists! ** Error: \n")
        raise

    # Commit changes and close database connection
    conn.commit()
    cur.close()
    conn.close()
    print("Successfully wrote tables!")


if __name__ == "__main__":
    conn = connect_to_db("gaitkeeper", "postgres")
    initialize_tables(conn)