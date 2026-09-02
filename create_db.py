import os
import sqlite3

# Resolve path relative to this file's location (matches src/orchestration/qlite_writer.py)
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DB_DIR = os.path.join(PROJECT_ROOT, "data")
DB_PATH = os.path.join(DB_DIR, "engine_telemetry.db")

def init_sqlite_database():
    print("======================================================================")
    print("UAV Engine Digital Twin — SQLite Persistence Database Setup")
    print("======================================================================")

    # 1. Ensure target directory exists
    os.makedirs(DB_DIR, exist_ok=True)
    print(f"[1/3] Target directory verified: '{DB_DIR}'")

    # 2. Table creation schema
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS uav_aero_engine_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp REAL NOT NULL,
        can_id TEXT NOT NULL,
        actual_rpm REAL,
        actual_map REAL,
        actual_cht REAL,
        actual_egt REAL,
        physics_cht REAL,
        physics_egt REAL,
        residual_cht REAL,
        residual_egt REAL,
        health_index_pct REAL,
        rul_hours REAL,
        anomaly_flag INTEGER,
        maintenance_urgency TEXT
    );
    """

    # 3. Create database file & table
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(create_table_sql)
        conn.commit()
        conn.close()
        print(f"[2/3] SQLite database & schema successfully initialized!")
        print(f"[3/3] Full Path: '{DB_PATH}'")
        print("----------------------------------------------------------------------")
        print("[+] Setup Complete. SQLite database is ready for ingestion.")
    except Exception as e:
        print(f"[!] Error creating SQLite database: {e}")

if __name__ == "__main__":
    init_sqlite_database()
