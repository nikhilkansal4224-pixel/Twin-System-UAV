import os
import sys
import logging
import torch
import numpy as np
from collections import deque

# Dynamic root resolution to prevent ModuleNotFoundError
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.zero_d_engine import ZeroEngineModel
from src.models.pinn_model import PhysicsInformedNN
from src.models.lstm_rul_model import RULPredictorEngine

logging.basicConfig(level=logging.INFO)

class DigitalTwinOrchestrator:
    def __init__(self, sequence_length: int = 10):
        self.seq_len = sequence_length
        self.history_buffer = deque(maxlen=self.seq_len)

        # 1. Initialize 0D Thermodynamic Physics Model
        self.physics_engine = ZeroEngineModel()

        # 2. Load PINN Model Weights
        self.pinn_model = PhysicsInformedNN()
        pinn_path = os.getenv("PINN_MODEL_PATH", "pinn_model_latest.pth")
        if os.path.exists(pinn_path):
            try:
                self.pinn_model.load_state_dict(torch.load(pinn_path, map_location=torch.device('cpu')))
                self.pinn_model.eval()
                logging.info(f"[+] Loaded PINN weights from {pinn_path}")
            except Exception as e:
                logging.warning(f"[!] Error loading PINN weights: {e}")

        # 3. Load LSTM RUL Model Weights
        self.rul_engine = RULPredictorEngine()
        lstm_path = os.getenv("LSTM_MODEL_PATH", "lstm_rul_latest.pth")
        if os.path.exists(lstm_path):
            self.rul_engine.load_weights(lstm_path)
            logging.info(f"[+] Loaded LSTM RUL weights from {lstm_path}")

    def process_telemetry_frame(self, raw_packet: dict) -> dict:
        """Runs the complete hybrid AI/physics evaluation pipeline on a single CAN frame."""
        data = raw_packet.get("data", raw_packet)
        timestamp = raw_packet.get("timestamp")
        can_id = raw_packet.get("can_id", "0x100")

        # Step A: 0D Physics Baseline Evaluation
        rpm = float(data.get("RPM", 5800.0))
        map_kpa = float(data.get("MAP", 101.3))
        physics_base = self.physics_engine.compute_baseline(rpm=rpm, map_kpa=map_kpa)

        # Step B: Compute Residual Deltas
        actual_cht = float(data.get("CHT", 135.0))
        actual_egt = float(data.get("EGT", 825.0))
        actual_oil = float(data.get("Oil_Pressure", 4.0))

        delta_cht = actual_cht - physics_base.get("physics_cht", actual_cht)
        delta_egt = actual_egt - physics_base.get("physics_egt", actual_egt)
        delta_oil = actual_oil - physics_base.get("physics_oil", actual_oil)
        delta_map = 0.0  # MAP baseline tracking delta

        residual_deltas = {
            "Delta_CHT": round(delta_cht, 2),
            "Delta_EGT": round(delta_egt, 2),
            "Delta_Oil_P": round(delta_oil, 2),
            "Delta_MAP": round(delta_map, 2)
        }

        # Step C: Buffer Sequence for LSTM RUL Evaluation (4 input features)
        feature_vector = [delta_cht, delta_egt, delta_oil, delta_map]
        self.history_buffer.append(feature_vector)

        # Pad sequence buffer if warming up
        while len(self.history_buffer) < self.seq_len:
            self.history_buffer.appendleft(feature_vector)

        seq_tensor = torch.tensor([list(self.history_buffer)], dtype=torch.float32)

        # Step D: Inference via LSTM & PINN
        rul_hours, health_idx = self.rul_engine.predict_rul(seq_tensor)
        
        if hasattr(self.pinn_model, 'detect_anomaly'):
            anomaly_flag = self.pinn_model.detect_anomaly(delta_cht, delta_egt, delta_oil, delta_map)
        else:
            anomaly_flag = health_idx < 70.0

        # Categorize Maintenance Urgency
        if health_idx > 80.0:
            urgency = "NOMINAL"
        elif health_idx > 50.0:
            urgency = "WARNING"
        else:
            urgency = "CRITICAL"

        # Step E: Consolidated Output Digital Twin State
        return {
            "timestamp": timestamp,
            "can_id": can_id,
            "telemetry_actual": data,
            "physics_baseline": physics_base,
            "residual_deltas": residual_deltas,
            "health_index_pct": round(health_idx, 2),
            "rul_hours": round(rul_hours, 2),
            "maintenance_urgency": urgency,
            "anomaly_flagged": anomaly_flag
        }