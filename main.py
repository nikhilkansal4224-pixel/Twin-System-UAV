#!/usr/bin/env python3
"""
Indigenous Digital Twin Framework for MALE UAV Aero-Piston Engines
Ground Control Station (GCS) Edge Orchestration Executable
"""
import torch
import sys
import os
import time
import json
import logging
from datetime import datetime

# Guarantee project root is in Python path regardless of execution directory
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import modules from src subpackages
from src.ingestion.dbc_decoder import DBCDecoder
from src.physics_engine.thermodynamics import ZeroEngineModel
from src.physics_engine.residual_calculator import ResidualCalculator
from src.ai_pipeline.pinn_model import PhysicsInformedNN
from src.ai_pipeline.lstm_rul import LSTMRULEstimator, RULPredictorEngine
from src.db.postgres_writer import PostgresWriter

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Detect and map global edge processing accelerator targets cleanly
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

class UAVEngineDigitalTwinApp:
    def __init__(self):
        logging.info("=" * 70)
        logging.info("Initializing Indigenous Digital Twin System for UAV Aero-Piston Engines")
        logging.info("Target Application: MALE UAV Ground Control Station (GCS) Edge Deployment")
        logging.info(f"Active Core Accelerator Target Engine: {device}")
        logging.info("=" * 70)

        # 1. Initialize Ingestion Subsystem
        logging.info("[1/5] Loading DBC Telemetry Decoder...")
        self.decoder = DBCDecoder()

        # 2. Initialize 0D Physics Engine Subsystem
        logging.info("[2/5] Initializing 0D Thermodynamic Reference Model (SciPy solve_ivp)...")
        self.physics_model = ZeroEngineModel()
        self.residual_calc = ResidualCalculator()

        # 3. Initialize PyTorch AI Subsystem & Load Pre-Trained Weights
        logging.info("[3/5] Loading PyTorch PINN Architecture & LSTM RUL Engine...")
        
        # A. Initialize PINN Model & Safe Device-Mapped Weights Loading
        self.pinn_model = PhysicsInformedNN().to(device)
        pinn_weights_path = os.path.join(project_root, "models", "saved_weights", "pinn_weights.pth")
        if os.path.exists(pinn_weights_path):
            self.pinn_model.load_state_dict(torch.load(pinn_weights_path, map_location=device, weights_only=True))
            logging.info(f"      --> Pre-trained PINN weights loaded from: '{pinn_weights_path}'")
        else:
            logging.info("      --> Warning: No PINN weights found. Running with initial weights.")
        self.pinn_model.eval()

        # B. Initialize LSTM RUL Engine & Safe Device-Mapped Weights Loading
        lstm_weights_path = os.path.join(project_root, "models", "saved_weights", "lstm_rul_weights.pth")
        self.rul_engine = RULPredictorEngine(nominal_max_life_hours=1200.0, sequence_length=10, weights_path=lstm_weights_path)

        # 4. Initialize Database Persistence Layer
        logging.info("[4/5] Connecting to SQLite Database...")
        self.sqlite_writer = SQLiteWriter()

        logging.info("[5/5] All Edge Subsystems Successfully Online!\n")

    def process_telemetry_frame(self, can_id: int, payload_bytes: bytes) -> dict:
        """
        Processes a single CAN-bus frame through the full Digital Twin pipeline.
        """
        # Step A: Decode raw CAN hex frame
        decoded_signals = self.decoder.decode_frame(can_id, payload_bytes)
        if not decoded_signals:
            return None

        rpm = float(decoded_signals.get("RPM", 5800.0))
        map_kpa = float(decoded_signals.get("MAP", 101.3))

        # Step B: Solve 0D Thermodynamics for theoretical healthy baseline
        physics_baseline = self.physics_model.compute_physics_baseline(rpm=rpm, map_kpa=map_kpa)

        # Step C: Compute absolute mathematical residuals ΔY = |Actual - Baseline|
        residual_results = self.residual_calc.compute_residuals(decoded_signals, physics_baseline)
        residuals = residual_results.get("residual_deltas", residual_results)

        # Step C.5: Align and assign your full 4-feature tensor array directly on device
        pinn_vec = [
            float(residuals.get("Delta_CHT", 0.0)),
            float(residuals.get("Delta_EGT", 0.0)),
            float(residuals.get("Delta_Oil_P", 0.0)),
            float(residuals.get("Delta_MAP", 0.0))
        ]
        
        input_tensor = torch.tensor([pinn_vec], dtype=torch.float32, device=device)
        
        with torch.no_grad():
            pinn_out = self.pinn_model(input_tensor)
        
        # Unpack indices cleanly depending on output dim dimensions
        pinn_severity_score = round(float(pinn_out[0, 2].item()), 4) if pinn_out.shape[1] >= 3 else round(float(pinn_out[0, 0].item()), 4)

        # Step D: Update sequence buffer & estimate Remaining Useful Life (RUL)
        self.rul_engine.add_telemetry_step(residuals)
        rul_assessment = self.rul_engine.predict_rul()

        # Determine anomaly status based on delta thresholds or return dictionary
        is_anomaly = residual_results.get("anomaly_detected", 
                    residual_results.get("anomaly_flagged", 
                    any(v > 15.0 for v in residuals.values() if isinstance(v, (int, float)))))

        # Step E: Construct unified Digital Twin state packet
        twin_state = {
            "timestamp": time.time(),
            "can_id": hex(can_id),
            "telemetry_actual": decoded_signals,
            "physics_baseline": physics_baseline,
            "residual_deltas": residuals,
            "anomaly_flagged": is_anomaly,
            "pinn_severity_score": pinn_severity_score,
            "rul_hours": rul_assessment["predicted_rul_hours"],
            "health_index_pct": rul_assessment["health_index_pct"],
            "maintenance_urgency": rul_assessment["maintenance_urgency"]
        }

        # Step F: Persist state to SQLite database
        self.sqlite_writer.write_twin_state(twin_state)

        return twin_state

    def run_live_simulation(self, loop_iterations=20, tick_interval=0.5):
        """
        Simulates real-time CAN-bus execution loop on local hardware.
        """
        import struct
        import random

        logging.info("Starting Edge Execution Loop (Simulating 10Hz CAN-bus Telemetry)...")
        logging.info("-" * 75)

        try:
            for iteration in range(1, loop_iterations + 1):
                # 1. Simulate Frame 0x100 (RPM, MAP, Oil Pressure)
                rpm_raw = int(5800 + random.uniform(-30, 30))
                map_raw = int((101.3 + random.uniform(-1, 1)) / 0.1)
                oil_raw = int((4.5 + random.uniform(-0.05, 0.05)) / 0.01)
                payload_100 = struct.pack("<HHH", rpm_raw, map_raw, oil_raw) + b'\x00\x00'

                self.process_telemetry_frame(0x100, payload_100)

                # 2. Simulate Frame 0x200 (CHT, EGT) with injected thermal drift at step 10
                cht_val = 115.0 if iteration < 10 else (115.0 + (iteration - 9) * 4.5)  # Inject thermal drift
                egt_val = 820.0 if iteration < 10 else (820.0 + (iteration - 9) * 12.0) # Inject EGT spike

                cht_raw = int((cht_val - (-40.0)) / 0.1)
                egt_raw = int(egt_val / 0.1)
                payload_200 = struct.pack("<HH", cht_raw, egt_raw) + b'\x00\x00\x00\x00'

                state_200 = self.process_telemetry_frame(0x200, payload_200)

                if state_200:
                    status_str = f"HEALTH: {state_200['health_index_pct']}% | RUL: {state_200['rul_hours']} hrs | STATUS: {state_200['maintenance_urgency']}"
                    anomaly_str = " [!] ANOMALY FLAGGED" if state_200['anomaly_flagged'] else " [OK] NOMINAL"
                    
                    logging.info(f"Iter {iteration:02d} | CHT: {state_200['telemetry_actual'].get('CHT', 0):.1f}°C (Δ {state_200['residual_deltas'].get('Delta_CHT', 0):.1f}) | EGT: {state_200['telemetry_actual'].get('EGT', 0):.1f}°C (Δ {state_200['residual_deltas'].get('Delta_EGT', 0):.1f}){anomaly_str}")
                    logging.info(f"        --> {status_str}")

                time.sleep(tick_interval)

            logging.info("-" * 75)
        except KeyboardInterrupt:
            logging.info("[!] Live simulation halted gracefully by Ground Control operator commands.")

if __name__ == "__main__":
    # Boot application engine instance
    app = UAVEngineDigitalTwinApp()
    app.run_live_simulation(loop_iterations=20, tick_interval=0.2)
