import os
import sqlite3
import time
import random

# Absolute target database path
DB_DIR = "/Users/nikhil/Documents/pro/uav-engine-digital-twin/data"
DB_PATH = os.path.join(DB_DIR, "engine_telemetry.db")

def populate_sample_rows(num_records=50):
    print("======================================================================")
    print("UAV Engine Digital Twin — SQLite Database Population Script")
    print("======================================================================")

    # 1. Ensure directory exists and connect
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 2. Ensure table schema exists
    cursor.execute("""
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
    """)

    insert_sql = """
    INSERT INTO uav_aero_engine_metrics (
        timestamp, can_id, actual_rpm, actual_map, actual_cht, actual_egt,
        physics_cht, physics_egt, residual_cht, residual_egt,
        health_index_pct, rul_hours, anomaly_flag, maintenance_urgency
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """

    base_time = time.time() - (num_records * 10)  # Simulate past records
    inserted_count = 0

    # 3. Generate sequential telemetry rows (simulating nominal -> anomaly degradation)
    for i in range(num_records):
        ts = base_time + (i * 10)
        can_id = "0x100" if i % 2 == 0 else "0x200"

        # Baseline physical reference values
        physics_cht = 115.0 + random.uniform(-0.5, 0.5)
        physics_egt = 820.0 + random.uniform(-1.0, 1.0)
        actual_rpm = 5800.0 + random.uniform(-20.0, 20.0)
        actual_map = 101.3 + random.uniform(-0.2, 0.2)

        # Inject thermal degradation/anomaly past step 30
        if i < 30:
            actual_cht = physics_cht + random.uniform(0.1, 2.0)
            actual_egt = physics_egt + random.uniform(0.5, 3.0)
            anomaly = 0
            urgency = "NOMINAL"
            health = round(100.0 - (i * 0.1), 2)
            rul = round(1200.0 - (i * 0.5), 1)
        else:
            # Overheating fault scenario
            degradation_offset = (i - 30) * 2.5
            actual_cht = physics_cht + 15.0 + degradation_offset
            actual_egt = physics_egt + 35.0 + degradation_offset
            anomaly = 1
            urgency = "WARNING" if i < 42 else "CRITICAL"
            health = round(max(0.0, 85.0 - (i - 30) * 3.0), 2)
            rul = round(max(0.0, 950.0 - (i - 30) * 25.0), 1)

        residual_cht = round(abs(actual_cht - physics_cht), 2)
        residual_egt = round(abs(actual_egt - physics_egt), 2)

        data_tuple = (
            ts, can_id, round(actual_rpm, 1), round(actual_map, 1),
            round(actual_cht, 2), round(actual_egt, 2),
            round(physics_cht, 2), round(physics_egt, 2),
            residual_cht, residual_egt,
            health, rul, anomaly, urgency
        )

        cursor.execute(insert_sql, data_tuple)
        inserted_count += 1

    conn.commit()
    conn.close()

    print(f"[+] Successfully inserted {inserted_count} sample rows into '{DB_PATH}'")
    print("----------------------------------------------------------------------")

def verify_rows():
    """Prints the latest 5 rows from SQLite."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, datetime(timestamp, 'unixepoch'), can_id, actual_cht, physics_cht, residual_cht, health_index_pct, maintenance_urgency FROM uav_aero_engine_metrics ORDER BY id DESC LIMIT 5")
    rows = cursor.fetchall()
    conn.close()

    print("\n--- Latest 5 Database Entries ---")
    print(f"{'ID':<4} | {'Timestamp (UTC)':<19} | {'CAN ID':<6} | {'Act CHT':<7} | {'Phys CHT':<8} | {'Res CHT':<7} | {'Health %':<8} | {'Status'}")
    print("-" * 90)
    for row in rows:
        print(f"{row[0]:<4} | {row[1]:<19} | {row[2]:<6} | {row[3]:<7} | {row[4]:<8} | {row[5]:<7} | {row[6]:<8} | {row[7]}")

if __name__ == "__main__":
    populate_sample_rows(num_records=50)
    verify_rows()
