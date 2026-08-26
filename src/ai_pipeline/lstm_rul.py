import torch
import torch.nn as nn
import numpy as np

# =====================================================================
# 1. LSTM REMAINING USEFUL LIFE (RUL) ARCHITECTURE
# =====================================================================
class LSTMRULEstimator(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=2, output_size=1):
        """
        Initializes the LSTM sequence model for RUL estimation.
        
        :param input_size: Number of features per time step (e.g., [Delta_CHT, Delta_EGT, Delta_Oil_P, Delta_MAP])
        :param hidden_size: Number of hidden units per LSTM layer
        :param num_layers: Number of stacked LSTM layers
        :param output_size: Output value (Predicted Remaining Useful Life in Hours)
        """
        super(LSTMRULEstimator, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Stacked LSTM Layers for temporal sequence modeling
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )

        # Fully Connected Regression Head
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        """
        Forward pass for time-series input sequence.
        
        :param x: Tensor of shape (batch_size, sequence_length, input_size)
        :return: Predicted RUL tensor of shape (batch_size, 1)
        """
        # Initialize hidden states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


# =====================================================================
# 2. RUL PREDICTION & HEALTH INDEX EVALUATOR
# =====================================================================
class RULPredictorEngine:
    def __init__(self, nominal_max_life_hours=1200.0, sequence_length=30):
        """
        Helper class to format streaming sequence buffers and compute health index scores.
        
        :param nominal_max_life_hours: Factory-rated engine component TBO (Time Between Overhauls)
        :param sequence_length: Number of time steps required for LSTM input sequence window
        """
        self.max_life = nominal_max_life_hours
        self.seq_len = sequence_length
        self.buffer = []
        self.model = LSTMRULEstimator()
        self.model.eval()  # Set model to evaluation mode

    def add_telemetry_step(self, residual_dict: dict):
        """
        Appends a new residual measurement step to the rolling sequence buffer.
        """
        features = [
            residual_dict.get("Delta_CHT", 0.0),
            residual_dict.get("Delta_EGT", 0.0),
            residual_dict.get("Delta_Oil_P", 0.0),
            residual_dict.get("Delta_MAP", 0.0)
        ]
        self.buffer.append(features)

        # Maintain fixed rolling sequence buffer length
        if len(self.buffer) > self.seq_len:
            self.buffer.pop(0)

    def predict_rul(self) -> dict:
        """
        Evaluates current sequence buffer and outputs remaining life hours and health percentage.
        """
        if len(self.buffer) < self.seq_len:
            # Pad buffer if sequence length is not yet met
            padded_buffer = [self.buffer[0] if self.buffer else [0.0]*4] * (self.seq_len - len(self.buffer)) + self.buffer
        else:
            padded_buffer = self.buffer

        # Convert to PyTorch Tensor: shape (1, sequence_length, input_size)
        input_tensor = torch.tensor([padded_buffer], dtype=torch.float32)

        with torch.no_grad():
            predicted_rul_hours = self.model(input_tensor).item()

        # Clamp output RUL between 0 and Nominal Max Life
        predicted_rul_hours = max(0.0, min(self.max_life, predicted_rul_hours + self.max_life * 0.75))
        
        # Calculate Health Index Percentage (100% = Brand New, 0% = Critical Replacement Needed)
        health_index_pct = (predicted_rul_hours / self.max_life) * 100.0

        return {
            "predicted_rul_hours": round(predicted_rul_hours, 1),
            "health_index_pct": round(health_index_pct, 2),
            "maintenance_urgency": "CRITICAL" if health_index_pct < 15.0 else ("WARNING" if health_index_pct < 35.0 else "NOMINAL")
        }


# =====================================================================
# MODULE VERIFICATION & TEST DRIVE
# =====================================================================
if __name__ == "__main__":
    print("[+] Testing LSTM Remaining Useful Life (RUL) Estimator Engine...")
    rul_engine = RULPredictorEngine(nominal_max_life_hours=1200.0, sequence_length=10)

    # 1. Simulate 15 steps of degraded thermal sequence data
    print("\n[+] Streaming Residual Sequence to LSTM Buffer...")
    for step in range(15):
        # Gradually increase residual deltas to simulate component wear
        simulated_residuals = {
            "Delta_CHT": 2.0 + (step * 0.8),
            "Delta_EGT": 8.0 + (step * 2.5),
            "Delta_Oil_P": 0.02 + (step * 0.015),
            "Delta_MAP": 0.1 + (step * 0.05)
        }
        rul_engine.add_telemetry_step(simulated_residuals)

    # 2. Evaluate RUL Prediction
    rul_results = rul_engine.predict_rul()

    print("\n--- [RUL & HEALTH INDEX PREDICTION SUMMARY] ---")
    print(f"Predicted RUL (Hours) : {rul_results['predicted_rul_hours']} hrs")
    print(f"Engine Health Index   : {rul_results['health_index_pct']}%")
    print(f"Maintenance Status    : {rul_results['maintenance_urgency']}")