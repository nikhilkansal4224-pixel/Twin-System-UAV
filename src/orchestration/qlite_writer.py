import os
import psycopg
import time
import logging

class PostgresWriter:
    def __init__(self, connection_string: str = None):
        """
        Initializes PostgreSQL database connection parameters and sets up tables if they do not exist.
        """
        if connection_string is None:
            # Reads PostgreSQL connection string from environment variables or uses default local fallback
            self.conn_info = os.getenv(
                "DATABASE_URL", 
                "postgresql://postgres:postgres@localhost:5432/uav_telemetry"
            )
        else:
            self.conn_info = connection_string

        self._init_db()

    def _get_connection(self):
        """Returns a connection to the PostgreSQL database using psycopg3."""
        return psycopg.connect(self.conn_info)

    def _init_db(self):
        """Creates the engine telemetry table schema."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS uav_aero_engine_metrics (
            id SERIAL PRIMARY KEY,
            timestamp DOUBLE PRECISION NOT NULL,
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
                with conn.cursor() as cursor:
                    cursor.execute(create_table_sql)
                conn.commit()
            logging.info("[+] PostgreSQL Database initialized successfully.")
        except Exception as e:
            logging.error(f"[!] PostgreSQL initialization error: {e}")

    def write_twin_state(self, twin_state: dict) -> bool:
        """
        Inserts a Digital Twin state packet into PostgreSQL.
        """
        timestamp = twin_state.get("timestamp", time.time())
        can_id = str(twin_state.get("can_id", "0x100"))

        telemetry = twin_state.get("telemetry_actual", {})
        baseline = twin_state.get("physics_baseline", {})
        residuals = twin_state.get("residual_deltas", {})

        # Use %s syntax for psycopg param substitution
        insert_sql = """
        INSERT INTO uav_aero_engine_metrics (
            timestamp, can_id, actual_rpm, actual_map, actual_cht, actual_egt,
            physics_cht, physics_egt, residual_cht, residual_egt,
            health_index_pct, rul_hours, anomaly_flag, maintenance_urgency
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
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
                with conn.cursor() as cursor:
                    cursor.execute(insert_sql, data_tuple)
                conn.commit()
            return True
        except Exception as e:
            logging.error(f"[!] Failed writing to PostgreSQL: {e}")
            return False

    def fetch_recent_records(self, limit: int = 5):
        """Helper function to verify written records."""
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT * FROM uav_aero_engine_metrics ORDER BY id DESC LIMIT %s;", (limit,))
                return cursor.fetchall()

    def close(self):
        """No-op when connections are opened per operation."""
        logging.info("[-] PostgreSQL Persistence Manager closed.")