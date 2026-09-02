import psycopg
import logging
import os

class PostgresWriter:
    def __init__(self, db_url: str = None):
        self.db_url = db_url or os.getenv(
            "DATABASE_URL", 
            "postgresql://grafana:Grafana%40123@localhost:5432/grafana"
        )

    def write_metrics(self, twin_state: dict):
        try:
            telemetry = twin_state.get("telemetry_actual", {})
            physics = twin_state.get("physics_baseline", {})
            residuals = twin_state.get("residual_deltas", {})

            # Extract raw values with fallbacks
            rpm = float(telemetry.get("RPM", telemetry.get("rpm", 0.0)))
            map_kpa = float(telemetry.get("MAP", telemetry.get("map_kpa", 0.0)))
            actual_cht = float(telemetry.get("CHT", telemetry.get("actual_cht", 0.0)))
            actual_egt = float(telemetry.get("EGT", telemetry.get("actual_egt", 0.0)))
            
            # Safe extraction for Oil Pressure
            oil_p = float(telemetry.get("Oil_Pressure", telemetry.get("oil_pressure", telemetry.get("actual_oil_pressure", 0.0))))

            # Extract physics baseline predictions
            phys_cht = float(physics.get("physics_cht", 0.0))
            phys_egt = float(physics.get("physics_egt", 0.0))
            phys_oil = float(physics.get("physics_oil", 0.0))

            # Residual deltas
            res_cht = float(residuals.get("Delta_CHT", residuals.get("residual_cht", abs(actual_cht - phys_cht))))
            res_egt = float(residuals.get("Delta_EGT", residuals.get("residual_egt", abs(actual_egt - phys_egt))))

            # System metadata
            health_pct = float(twin_state.get("health_index_pct", 100.0))
            rul_hrs = float(twin_state.get("rul_hours", 1200.0))
            urgency = str(twin_state.get("maintenance_urgency", "NOMINAL"))
            anomaly = 1 if twin_state.get("anomaly_flagged", False) else 0
            altitude_m = float(telemetry.get("Altitude", telemetry.get("altitude", telemetry.get("altitude_m", 0.0))))
            ambient_temp = float(telemetry.get("Ambient_Temp", telemetry.get("ambient_temp", telemetry.get("ambient_temp_c", 15.0))))

            with psycopg.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO uav_aero_engine_metrics (
                            rpm, map_kpa, actual_cht, physics_cht, residual_cht,
                            actual_egt, physics_egt, residual_egt, 
                            actual_oil_pressure, physics_oil_pressure,
                            altitude_m, ambient_temp_c,
                            health_index_pct, rul_hours, maintenance_urgency, anomaly_flag
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                        """,
                        (
                            rpm, map_kpa, actual_cht, phys_cht, res_cht,
                            actual_egt, phys_egt, res_egt, 
                            oil_p, phys_oil,
                            altitude_m, ambient_temp,
                            health_pct, rul_hrs, urgency, anomaly
                        )
                    )
                    conn.commit()

        except Exception as e:
            logging.error(f"[!] Failed to write metrics to PostgreSQL: {e}")