import numpy as np

class ResidualCalculator:
    def __init__(self, warning_thresholds=None):
        """
        Initializes physical residual thresholds for micro-anomaly detection.
        Default thresholds represent maximum acceptable physical deviations before flagging.
        """
        self.thresholds = warning_thresholds or {
            "Delta_CHT": 12.0,     # Max 12.0 °C deviation allowed for Cylinder Head Temp
            "Delta_EGT": 35.0,     # Max 35.0 °C deviation allowed for Exhaust Gas Temp
            "Delta_Oil_P": 0.35,   # Max 0.35 bar deviation allowed for Oil Pressure
            "Delta_MAP": 5.0       # Max 5.0 kPa deviation allowed for Manifold Pressure
        }

    def compute_residuals(self, actual: dict, baseline: dict) -> dict:
        actual_cht = float(actual.get("CHT", 115.0))
        actual_egt = float(actual.get("EGT", 820.0))

        physics_cht = float(baseline.get("Physics_CHT", 115.0))
        physics_egt = float(baseline.get("Physics_EGT", 820.0))

        # Auto-convert Kelvin to Celsius if baseline is > 200
        if physics_cht > 200.0:
            physics_cht -= 273.15
        if physics_egt > 500.0:
            physics_egt -= 273.15

        delta_cht = abs(actual_cht - physics_cht)
        delta_egt = abs(actual_egt - physics_egt)

        return {
            "Delta_CHT": round(delta_cht, 2),
            "Delta_EGT": round(delta_egt, 2)
        }


# =====================================================================
# MODULE VERIFICATION & TEST DRIVE
# =====================================================================
if __name__ == "__main__":
    calculator = ResidualCalculator()

    # 1. Healthy Flight Condition Test
    healthy_telemetry = {"CHT": 118.2, "EGT": 822.0, "Oil_Pressure": 4.45}
    healthy_baseline  = {"Physics_CHT": 115.0, "Physics_EGT": 820.0, "Physics_Oil_P": 4.50}

    result_healthy = calculator.compute_residuals(healthy_telemetry, healthy_baseline)
    print("--- [TEST 1: HEALTHY ENGINE STATE] ---")
    print(f"Residuals (ΔY): {result_healthy['residuals']}")
    print(f"Anomaly Flagged: {result_healthy['anomaly_detected']}\n")

    # 2. Early-Stage Injector Clog Anomaly Test (EGT Spikes + CHT Drift)
    faulty_telemetry = {"CHT": 132.5, "EGT": 875.0, "Oil_Pressure": 4.42}
    faulty_baseline  = {"Physics_CHT": 115.0, "Physics_EGT": 820.0, "Physics_Oil_P": 4.50}

    result_faulty = calculator.compute_residuals(faulty_telemetry, faulty_baseline)
    print("--- [TEST 2: DEGRADATION / ANOMALY STATE] ---")
    print(f"Residuals (ΔY): {result_faulty['residuals']}")
    print(f"Parameter Flags: {result_faulty['flags']}")
    print(f"Anomaly Flagged: {result_faulty['anomaly_detected']}")
    print(f"Vector for PyTorch PINN: {result_faulty['pinn_input_vector']}")