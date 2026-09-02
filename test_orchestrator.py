import json
import time
from src.ai_pipeline.orchestrator import DigitalTwinOrchestrator

def run_dry_run_test():
    print("[1/3] Initializing DigitalTwinOrchestrator...")
    orchestrator = DigitalTwinOrchestrator(
        pinn_path="pinn_model_latest.pth",
        lstm_path="lstm_rul_latest.pth"
    )

    # ------------------------------------------------------------------
    # Test Frame 1: Nominal Flight Operations
    # ------------------------------------------------------------------
    nominal_frame = {
        "timestamp": time.time(),
        "can_id": "0x100",
        "data": {
            "RPM": 5000.0,
            "MAP": 95.0,
            "CHT": 135.0,  # Expected nominal range
            "EGT": 820.0,  # Expected nominal range
            "Oil_Pressure": 4.2
        }
    }

    print("\n[2/3] Processing Nominal Telemetry Frame...")
    nominal_state = orchestrator.process_telemetry_frame(nominal_frame)
    print("------------------------------------------------------------------")
    print(json.dumps(nominal_state, indent=2))
    print("------------------------------------------------------------------")

    # ------------------------------------------------------------------
    # Test Frame 2: Thermal Overheat Anomaly Event
    # ------------------------------------------------------------------
    anomaly_frame = {
        "timestamp": time.time(),
        "can_id": "0x101",
        "data": {
            "RPM": 5800.0,
            "MAP": 101.3,
            "CHT": 185.0,  # Significant spike above 0D baseline
            "EGT": 930.0,  # Significant spike above 0D baseline
            "Oil_Pressure": 2.1
        }
    }

    print("\n[3/3] Processing Anomaly Telemetry Frame...")
    anomaly_state = orchestrator.process_telemetry_frame(anomaly_frame)
    print("------------------------------------------------------------------")
    print(json.dumps(anomaly_state, indent=2))
    print("------------------------------------------------------------------")

    # Assertions to confirm expectations
    assert nominal_state["maintenance_urgency"] in ["NOMINAL", "WARNING"], "Nominal frame should not trigger critical alert"
    assert anomaly_state["anomaly_flagged"] is True, "Anomaly frame must set anomaly_flagged=True"
    assert anomaly_state["maintenance_urgency"] == "CRITICAL", "Anomaly frame must set maintenance_urgency='CRITICAL'"
    
    print("\n[✔] ALL DRY-RUN TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_dry_run_test()