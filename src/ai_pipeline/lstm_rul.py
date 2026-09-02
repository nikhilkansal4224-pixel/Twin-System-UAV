import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class RULPredictorEngine:
    """
    RUL Predictor Engine for Rotax 914.
    Maps physical model residual deltas (ΔY) to health decay curves.
    """

    def __init__(self, nominal_max_life_hours: float = 1200.0, sequence_length: int = 10):
        self.max_life = nominal_max_life_hours
        self.seq_len = sequence_length
        self.history = []

    def add_telemetry_step(self, residuals: dict):
        d_cht = float(residuals.get("Delta_CHT", residuals.get("residual_cht", 0.0)))
        d_egt = float(residuals.get("Delta_EGT", residuals.get("residual_egt", 0.0)))
        d_oil = float(residuals.get("Delta_Oil_P", residuals.get("residual_oil", 0.0)))

        # Normalized step error (scaled appropriately without harsh oil multipliers)
        step_delta = (d_cht * 0.45) + (d_egt * 0.45) + (d_oil * 2.0 * 0.1)

        # Immediate sequence reset if live deltas are nominal but buffer holds legacy errors
        if d_cht < 2.0 and d_egt < 3.0:
            if len(self.history) > 0 and np.mean(self.history) > 3.0:
                logging.info("[+] Live deltas nominal (<2°C). Clearing pre-calibration history buffer.")
                self.history.clear()

        self.history.append(step_delta)

        if len(self.history) > self.seq_len:
            self.history.pop(0)

    def predict_rul(self) -> dict:
        if not self.history:
            return {
                "predicted_rul_hours": self.max_life,
                "health_index_pct": 100.0,
                "maintenance_urgency": "NOMINAL"
            }

        avg_residual = float(np.mean(self.history))

        # Health percentage mapping
        health_pct = max(0.0, min(100.0, 100.0 * np.exp(-0.02 * max(0.0, avg_residual - 0.5))))
        predicted_rul = (health_pct / 100.0) * self.max_life

        if health_pct > 80.0:
            urgency = "NOMINAL"
        elif health_pct > 55.0:
            urgency = "MONITOR"
        elif health_pct > 30.0:
            urgency = "WARNING"
        else:
            urgency = "CRITICAL"

        return {
            "predicted_rul_hours": round(predicted_rul, 2),
            "health_index_pct": round(health_pct, 2),
            "maintenance_urgency": urgency
        }