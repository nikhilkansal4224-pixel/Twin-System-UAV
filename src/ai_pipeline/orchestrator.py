import os
import time
import logging
import torch
import torch.nn as nn
import numpy as np

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# PyTorch Model Architectures
# ---------------------------------------------------------------------------
class UAVEnginePINN(nn.Module):
    """Physics-Informed Neural Network for thermal residual analysis."""
    def __init__(self, input_dim=5, hidden_dim=64, output_dim=4):
        super(UAVEnginePINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class EngineRUL_LSTM(nn.Module):
    """LSTM sequence model for Remaining Useful Life (RUL) estimation."""
    def __init__(self, input_dim=4, hidden_dim=32, num_layers=2):
        super(EngineRUL_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


# ---------------------------------------------------------------------------
# 0D Thermodynamic First-Principles Model Baseline
# ---------------------------------------------------------------------------
class ThermodynamicEngine0D:
    """0D Steady-State First-Principles Baseline Model (Rotax 914 Aero Engine)."""
    def predict_baseline(self, rpm: float, map_kpa: float) -> dict:
        # Standard physics equations mapping manifold pressure and RPM to baseline temps
        normalized_load = (map_kpa / 100.0) * (rpm / 5800.0)
        
        physics_cht = 90.0 + (55.0 * normalized_load)
        physics_egt = 650.0 + (220.0 * normalized_load)
        physics_oil = 5.5 - (1.8 * (rpm / 5800.0))

        return {
            "physics_cht": round(float(physics_cht), 2),
            "physics_egt": round(float(physics_egt), 2),
            "physics_oil": round(float(physics_oil), 2)
        }


# ---------------------------------------------------------------------------
# Digital Twin Orchestrator
# ---------------------------------------------------------------------------
class DigitalTwinOrchestrator:
    def __init__(
        self,
        pinn_path: str = "pinn_model_latest.pth",
        lstm_path: str = "lstm_rul_latest.pth"
    ):
        self.pinn_path = pinn_path
        self.lstm_path = lstm_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Instantiate models & physics baseline
        self.physics_engine = ThermodynamicEngine0D()
        self.pinn_model = UAVEnginePINN(input_dim=5, output_dim=4).to(self.device)
        self.lstm_model = EngineRUL_LSTM(input_dim=4).to(self.device)

        self.last_pinn_mtime = 0
        self.last_lstm_mtime = 0

        # Load state checkpoints
        self.load_models()

        # Sliding window buffer for LSTM time-series input
        self.sequence_buffer = []
        self.seq_length = 10

    def load_models(self):
        """Loads or hot-reloads model checkpoints dynamically."""
        # PINN Hot-Reloading
        if os.path.exists(self.pinn_path):
            mtime = os.path.getmtime(self.pinn_path)
            if mtime > self.last_pinn_mtime:
                try:
                    self.pinn_model.load_state_dict(
                        torch.load(self.pinn_path, map_location=self.device)
                    )
                    self.pinn_model.eval()
                    self.last_pinn_mtime = mtime
                    logging.info(f"[+] Loaded/Reloaded PINN model checkpoint from {self.pinn_path}")
                except Exception as e:
                    logging.error(f"[!] Error loading PINN checkpoint: {e}")

        # LSTM Hot-Reloading
        if os.path.exists(self.lstm_path):
            mtime = os.path.getmtime(self.lstm_path)
            if mtime > self.last_lstm_mtime:
                try:
                    self.lstm_model.load_state_dict(
                        torch.load(self.lstm_path, map_location=self.device)
                    )
                    self.lstm_model.eval()
                    self.last_lstm_mtime = mtime
                    logging.info(f"[+] Loaded/Reloaded LSTM model checkpoint from {self.lstm_path}")
                except Exception as e:
                    logging.error(f"[!] Error loading LSTM checkpoint: {e}")

    def process_telemetry_frame(self, raw_frame: dict) -> dict:
        """Processes an incoming raw telemetry frame into a full Digital Twin state."""
        # Check for dynamic weight updates from retraining tasks
        self.load_models()

        timestamp = raw_frame.get("timestamp", time.time())
        can_id = raw_frame.get("can_id", "0x100")
        data = raw_frame.get("data", raw_frame)

        # Extract Telemetry Features
        rpm = float(data.get("RPM", 5000.0))
        map_kpa = float(data.get("MAP", 95.0))
        actual_cht = float(data.get("CHT", data.get("cht_c", 135.0)))
        actual_egt = float(data.get("EGT", data.get("egt_c", 820.0)))
        actual_oil = float(data.get("Oil_Pressure", data.get("oil_pressure_bar", 4.2)))

        # Step 1: Compute 0D Physics Baseline
        physics_baseline = self.physics_engine.predict_baseline(rpm, map_kpa)
        p_cht = physics_baseline["physics_cht"]
        p_egt = physics_baseline["physics_egt"]
        p_oil = physics_baseline["physics_oil"]

        # Step 2: Compute Thermodynamic Residual Deltas
        delta_cht = round(actual_cht - p_cht, 2)
        delta_egt = round(actual_egt - p_egt, 2)
        delta_oil = round(actual_oil - p_oil, 2)

        # Step 3: PINN Anomaly Detection Inference
        pinn_input = torch.tensor(
            [[actual_cht, actual_egt, actual_oil, p_cht, p_egt]],
            dtype=torch.float32
        ).to(self.device)

        with torch.no_grad():
            pinn_output = self.pinn_model(pinn_input).cpu().numpy()[0]

        # Evaluate Thermal Anomaly Conditions
        anomaly_flagged = bool(abs(delta_cht) > 15.0 or abs(delta_egt) > 35.0)

        # Step 4: LSTM Remaining Useful Life (RUL) Inference
        if self.last_lstm_mtime > 0:
            # Trained model available: Run inference
            feature_vector = [delta_cht, delta_egt, delta_oil, rpm / 5800.0]
            self.sequence_buffer.append(feature_vector)
            if len(self.sequence_buffer) > self.seq_length:
                self.sequence_buffer.pop(0)

            while len(self.sequence_buffer) < self.seq_length:
                self.sequence_buffer.insert(0, feature_vector)

            lstm_input = torch.tensor(
                [self.sequence_buffer], dtype=torch.float32
            ).to(self.device)

            with torch.no_grad():
                rul_pred = self.lstm_model(lstm_input).cpu().item()

            rul_hours = max(0.0, round(float(rul_pred), 1))
            health_index_pct = round(min(100.0, max(0.0, (rul_hours / 1200.0) * 100.0)), 2)
        else:
            # Fallback when weights are not yet loaded
            residual_penalty = (abs(delta_cht) * 0.5) + (abs(delta_egt) * 0.1)
            health_index_pct = round(max(0.0, min(100.0, 100.0 - residual_penalty)), 2)
            rul_hours = round((health_index_pct / 100.0) * 1200.0, 1)

        # Step 5: Urgency Matrix Evaluation
        if anomaly_flagged or health_index_pct < 40.0:
            urgency = "CRITICAL"
        elif health_index_pct < 70.0 or abs(delta_cht) > 8.0:
            urgency = "WARNING"
        else:
            urgency = "NOMINAL"

        # Return structured twin state dictionary
        return {
            "timestamp": timestamp,
            "can_id": can_id,
            "telemetry_actual": {
                "RPM": rpm,
                "MAP": map_kpa,
                "CHT": actual_cht,
                "EGT": actual_egt,
                "Oil_Pressure": actual_oil
            },
            "physics_baseline": physics_baseline,
            "residual_deltas": {
                "Delta_CHT": delta_cht,
                "Delta_EGT": delta_egt,
                "Delta_Oil_P": delta_oil
            },
            "health_index_pct": health_index_pct,
            "rul_hours": rul_hours,
            "maintenance_urgency": urgency,
            "anomaly_flagged": anomaly_flagged
        }