import numpy as np


class ResidualCalculator:
    def __init__(self, warning_thresholds: dict = None):
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
        """
        Calculates physical residuals (actual - baseline) for CHT, EGT, Oil Pressure, and MAP.
        
        :param actual: Dictionary containing actual sensor measurements (CHT, EGT, Oil_Pressure, MAP)
        :param baseline: Dictionary containing expected physics model baselines (Physics_CHT, Physics_EGT, Physics_Oil, Physics_MAP)
        :return: Dictionary containing deltas, anomaly flags, and PINN input vector
        """
        # 1. Read Actual Sensor Values with fallback defaults
        actual_cht = float(actual.get("CHT", actual.get("actual_cht", 115.0)))
        actual_egt = float(actual.get("EGT", actual.get("actual_egt", 820.0)))
        actual_oil = float(actual.get("Oil_Pressure", actual.get("actual_oil", 4.50)))
        actual_map = float(actual.get("MAP", actual.get("actual_map", 101.3)))

        # 2. Read Physics Model Baseline Values
        physics_cht = float(baseline.get("Physics_CHT", baseline.get("physics_cht", 115.0)))
        physics_egt = float(baseline.get("Physics_EGT", baseline.get("physics_egt", 820.0)))
        physics_oil = float(baseline.get("Physics_Oil_P", baseline.get("physics_oil", 4.50)))
        physics_map = float(baseline.get("Physics_MAP", baseline.get("physics_map", 101.3)))

        # 3. Compute Absolute Deviations (Residual Deltas)
        delta_cht = round(abs(actual_cht - physics_cht), 2)
        delta_egt = round(abs(actual_egt - physics_egt), 2)
        delta_oil = round(abs(actual_oil - physics_oil), 3)
        delta_map = round(abs(actual_map - physics_map), 2)

        residuals = {
            "Delta_CHT": delta_cht,
            "Delta_EGT": delta_egt,
            "Delta_Oil_P": delta_oil,
            "Delta_MAP": delta_map
        }

        # 4. Evaluate Anomaly Threshold Flags
        flags = {
            "CHT_exceeded": delta_cht > self.thresholds["Delta_CHT"],
            "EGT_exceeded": delta_egt > self.thresholds["Delta_EGT"],
            "Oil_P_exceeded": delta_oil > self.thresholds["Delta_Oil_P"],
            "MAP_exceeded": delta_map > self.thresholds["Delta_MAP"]
        }
        anomaly_detected = any(flags.values())

        # 5. Formulate 4-Feature Input Tensor Vector for PyTorch PINN
        pinn_input_vector = [delta_cht, delta_egt, delta_oil, delta_map]

        return {
            "residuals": residuals,
            "residual_deltas": residuals,        # Alias for main pipeline compatibility
            "flags": flags,
            "anomaly_detected": anomaly_detected,
            "anomaly_flagged": anomaly_detected,  # Alias for main pipeline compatibility
            "pinn_input_vector": pinn_input_vector
        }


# =====================================================================
# MODULE VERIFICATION & TEST DRIVE
# =====================================================================
if __name__ == "__main__":
    calculator = ResidualCalculator()

    # 1. Healthy Flight Condition Test
    healthy_telemetry = {"CHT": 115.56, "EGT": 819.74, "Oil_Pressure": 4.495, "MAP": 101.24}
    healthy_baseline  = {"Physics_CHT": 115.0, "Physics_EGT": 820.0, "Physics_Oil_P": 4.50, "Physics_MAP": 101.3}

    result_healthy = calculator.compute_residuals(healthy_telemetry, healthy_baseline)
    print("--- [TEST 1: HEALTHY ENGINE STATE] ---")
    print(f"Residuals (ΔY): {result_healthy['residuals']}")
    print(f"Anomaly Flagged: {result_healthy['anomaly_detected']}")
    print(f"PINN Input Vector: {result_healthy['pinn_input_vector']}\n")

    # 2. Thermal & Oil Degradation Anomaly Test
    faulty_telemetry = {"CHT": 137.02, "EGT": 828.61, "Oil_Pressure": 4.05, "MAP": 101.72}
    faulty_baseline  = {"Physics_CHT": 110.66, "Physics_EGT": 820.0, "Physics_Oil_P": 4.50, "Physics_MAP": 101.3}

    result_faulty = calculator.compute_residuals(faulty_telemetry, faulty_baseline)
    print("--- [TEST 2: DEGRADATION / ANOMALY STATE] ---")
    print(f"Residuals (ΔY): {result_faulty['residuals']}")
    print(f"Parameter Flags: {result_faulty['flags']}")
    print(f"Anomaly Flagged: {result_faulty['anomaly_detected']}")
    print(f"PINN Input Vector: {result_faulty['pinn_input_vector']}")