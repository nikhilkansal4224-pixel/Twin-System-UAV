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

    def compute_residuals(self, telemetry_actual: dict, physics_baseline: dict) -> dict:
    # Convert Physics outputs from Kelvin to Celsius if applicable
        physics_cht_c = physics_baseline.get("Physics_CHT", 388.15) - 273.15
        physics_egt_c = physics_baseline.get("Physics_EGT", 1093.15) - 273.15

        actual_cht = telemetry_actual.get("CHT", 115.0)
        actual_egt = telemetry_actual.get("EGT", 820.0)

        delta_cht = abs(actual_cht - physics_cht_c)
        delta_egt = abs(actual_egt - physics_egt_c)

        return {
        "Delta_CHT": round(delta_cht, 2),
        "Delta_EGT": round(delta_egt, 2)
    }

        for actual_key, physics_key in param_mapping.items():
            if actual_key in telemetry_data and physics_key in physics_baseline:
                actual_val = float(telemetry_data[actual_key])
                baseline_val = float(physics_baseline[physics_key])

                # Calculate absolute delta ΔY = |Actual - Baseline|
                delta = round(abs(actual_val - baseline_val), 3)
                delta_key = f"Delta_{actual_key}"
                residuals[delta_key] = delta

                # Evaluate against dynamic tolerance limits
                max_tolerance = self.thresholds.get(delta_key, 10.0)
                is_exceeded = delta > max_tolerance
                flags[f"{delta_key}_Flag"] = is_exceeded

                if is_exceeded:
                    anomaly_detected = True

        return {
            "residuals": residuals,
            "flags": flags,
            "anomaly_detected": anomaly_detected,
            "pinn_input_vector": np.array(list(residuals.values()), dtype=np.float32)
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