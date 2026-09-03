import math
import logging
import os
import psycopg

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class ZeroEngineModel:
    """
    Self-Adjusting 0D Thermodynamic Baseline Engine Model for Rotax 914.
    Uses continuous online learning with statistical error bounding to self-calibrate
    without manual intervention or startup deadlocks.
    """

    T0_K = 288.15        # Sea level standard temperature (15°C)
    P0_KPA = 101.325     # Sea level standard pressure (kPa)
    LAPSE_RATE = 0.0065  # Temperature lapse rate (K/m)
    G0 = 9.80665
    R_AIR = 287.05
    EXPONENT = G0 / (R_AIR * LAPSE_RATE)

    def __init__(self, learning_rate: float = 0.05):
        # Auto-adjusting bias offsets (start at 0.0 and self-tune)
        self.cht_bias = 0.0
        self.egt_bias = 0.0
        self.lr = learning_rate  # Adaptation tracking rate

    def calculate_isa_atmosphere(self, altitude_m: float) -> dict:
        alt = max(0.0, min(altitude_m, 11000.0))
        t_amb_k = self.T0_K - (self.LAPSE_RATE * alt)
        t_amb_c = t_amb_k - 273.15
        p_amb_kpa = self.P0_KPA * math.pow((t_amb_k / self.T0_K), self.EXPONENT)
        density_kg_m3 = (p_amb_kpa * 1000.0) / (self.R_AIR * t_amb_k)

        return {
            "ambient_temp_c": t_amb_c,
            "ambient_pressure_kpa": p_amb_kpa,
            "air_density_kg_m3": density_kg_m3,
            "density_ratio": density_kg_m3 / 1.225
        }

    def compute_physics_baseline(self, rpm: float, map_kpa: float, altitude_m: float = 0.0, **kwargs) -> dict:
        isa = self.calculate_isa_atmosphere(altitude_m)
        t_amb = kwargs.get("ambient_temp_c", isa["ambient_temp_c"])
        rho_ratio = isa["density_ratio"]

        norm_rpm = rpm / 5800.0
        norm_map = map_kpa / 101.325

        # Raw 0D Physics Base Equations
        cooling_efficiency = max(0.4, rho_ratio)
        raw_cht = 110.0 + (35.0 * norm_rpm) + (20.0 * norm_map) + (t_amb - 15.0) * 0.4 + (1.0 - cooling_efficiency) * 15.0
        raw_egt = 720.0 + (110.0 * norm_map) + (30.0 * norm_rpm) - (t_amb - 15.0) * 0.2 + (1.0 - rho_ratio) * 25.0

        # Self-Adjusted Predictions
        adapted_cht = raw_cht + self.cht_bias
        adapted_egt = raw_egt + self.egt_bias

        physics_oil = 2.0 + (2.5 * norm_rpm) - (max(0.0, adapted_cht - 130.0) * 0.01)

        return {
            "physics_cht": round(adapted_cht, 2),
            "physics_egt": round(adapted_egt, 2),
            "physics_oil": round(physics_oil, 2),
            "raw_cht": round(raw_cht, 2),
            "raw_egt": round(raw_egt, 2),
            "isa_ambient_temp": round(t_amb, 2),
            "isa_pressure_kpa": round(isa["ambient_pressure_kpa"], 2),
            "air_density_kg_m3": round(isa["air_density_kg_m3"], 3)
        }

    def auto_adjust(self, actual_cht: float, actual_egt: float, physics_baseline: dict):
        """
        Self-adjusting online filter. Dynamically tracks baseline shifts 
        without freezing during initial startup or small operating drifts.
        """
        # Current prediction error
        err_cht = actual_cht - physics_baseline["physics_cht"]
        err_egt = actual_egt - physics_baseline["physics_egt"]

        # If error is massive (e.g. artificial thermal spike > 60°C), treat as hard anomaly & don't adapt
        if abs(err_cht) > 60.0 or abs(err_egt) > 80.0:
            return

        # Smooth continuous online tracking (Exponential Moving Average update)
        self.cht_bias += self.lr * err_cht
        self.egt_bias += self.lr * err_egt

    def load_calibration(self, db_url: str = None):
        url = db_url or os.getenv("DATABASE_URL", "postgresql://uav_user:uav_password@127.0.0.1:5432/uav_telemetry")
        try:
            with psycopg.connect(url) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT cht_bias, egt_bias FROM physics_calibration_state WHERE model_name = 'rotax_914_0d';")
                    row = cur.fetchone()
                    if row and row[0] is not None:
                        self.cht_bias, self.egt_bias = float(row[0]), float(row[1])
                        logging.info(f"[+] Loaded Self-Adjusted Offsets -> CHT Bias: {self.cht_bias:.2f} | EGT Bias: {self.egt_bias:.2f}")
        except Exception as e:
            logging.warning(f"[!] Could not load calibration state ({e}). Self-adjusting from initial state.")

    def save_calibration(self, db_url: str = None):
        url = db_url or os.getenv("DATABASE_URL", "postgresql://uav_user:uav_password@127.0.0.1:5432/uav_telemetry")
        try:
            with psycopg.connect(url) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO physics_calibration_state (model_name, cht_bias, egt_bias, updated_at)
                        VALUES ('rotax_914_0d', %s, %s, CURRENT_TIMESTAMP)
                        ON CONFLICT (model_name) DO UPDATE 
                        SET cht_bias = EXCLUDED.cht_bias, egt_bias = EXCLUDED.egt_bias, updated_at = CURRENT_TIMESTAMP;
                        """,
                        (self.cht_bias, self.egt_bias)
                    )
                    conn.commit()
        except Exception as e:
            logging.error(f"[!] Failed to save calibration state: {e}")