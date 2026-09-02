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
        # Note: ZeroEngineModel.compute_physics_baseline() always returns Celsius,
        # so no Kelvin->Celsius conversion is needed here.

        delta_cht = round(abs(actual_cht - physics_cht), 2)
        delta_egt = round(abs(actual_egt - physics_egt), 2)

        residuals = {
            "Delta_CHT": delta_cht,
            "Delta_EGT": delta_egt
        }

        # Use the configured thresholds to flag anomalies
        flags = {
            "CHT_exceeded": delta_cht > self.thresholds["Delta_CHT"],
            "EGT_exceeded": delta_egt > self.thresholds["Delta_EGT"]
        }
        anomaly_detected = any(flags.values())

        return {
            "residuals": residuals,
            "residual_deltas": residuals,        # kept for main.py's .get() call
            "flags": flags,
            "anomaly_detected": anomaly_detected,
            "anomaly_flagged": anomaly_detected,  # alias, main.py checks both
            "pinn_input_vector": [
                delta_cht, delta_egt,
                0.0,  # Delta_Oil_P placeholder — wire in real value if available
                0.0   # Delta_MAP placeholder
            ]
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
