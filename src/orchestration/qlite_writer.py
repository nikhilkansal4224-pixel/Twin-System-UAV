import os
import sqlite3
import time
import logging

class SQLiteWriter:
    def __init__(self, db_path: str = None):
        """
        Initializes SQLite database connection and sets up tables if they do not exist.
        """
        if db_path is None:
            # Place database file in 'data/' directory at project root
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
            data_dir = os.path.join(project_root, "data")
            os.makedirs(data_dir, exist_ok=True)
            self.db_path = os.path.join(data_dir, "engine_telemetry.db")
        else:
            self.db_path = db_path

        self._init_db()

    def _get_connection(self):
        """Returns a connection to the SQLite database."""
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """Creates the engine telemetry table schema."""
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
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(create_table_sql)
                conn.commit()
            logging.info(f"[+] SQLite Database initialized at: '{self.db_path}'")
        except Exception as e:
            logging.error(f"[!] SQLite initialization error: {e}")

    def write_twin_state(self, twin_state: dict) -> bool:
        """
        Inserts a Digital Twin state packet into SQLite.
        """
        timestamp = twin_state.get("timestamp", time.time())
        can_id = str(twin_state.get("can_id", "0x100"))

        telemetry = twin_state.get("telemetry_actual", {})
        baseline = twin_state.get("physics_baseline", {})
        residuals = twin_state.get("residual_deltas", {})

        insert_sql = """
        INSERT INTO uav_aero_engine_metrics (
            timestamp, can_id, actual_rpm, actual_map, actual_cht, actual_egt,
            physics_cht, physics_egt, residual_cht, residual_egt,
            health_index_pct, rul_hours, anomaly_flag, maintenance_urgency
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """

        data_tuple = (
            timestamp,
            can_id,
            float(telemetry.get("RPM", 0.0)),
            float(telemetry.get("MAP", 0.0)),
            float(telemetry.get("CHT", 0.0)),
            float(telemetry.get("EGT", 0.0)),
            float(baseline.get("Physics_CHT", 0.0)),
            float(baseline.get("Physics_EGT", 0.0)),
            float(residuals.get("Delta_CHT", 0.0)),
            float(residuals.get("Delta_EGT", 0.0)),
            float(twin_state.get("health_index_pct", 100.0)),
            float(twin_state.get("rul_hours", 1200.0)),
            1 if twin_state.get("anomaly_flagged", False) else 0,
            str(twin_state.get("maintenance_urgency", "NOMINAL"))
        )

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(insert_sql, data_tuple)
                conn.commit()
            return True
        except Exception as e:
            logging.error(f"[!] Failed writing to SQLite: {e}")
            return False

    def fetch_recent_records(self, limit: int = 5):
        """Helper function to verify written records."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM uav_aero_engine_metrics ORDER BY id DESC LIMIT ?", (limit,))
            return cursor.fetchall()

    def close(self):
        """No-op for SQLite since connections are managed per context."""
        logging.info("[-] SQLite Persistence Manager closed.")