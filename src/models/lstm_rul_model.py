import torch
import torch.nn as nn

class RULPredictorEngine(nn.Module):
    def __init__(self, input_size: int = 4, hidden_size: int = 64, num_layers: int = 2):
        super(RULPredictorEngine, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True
        )
        # Sequential head matching keys: fc.0.weight, fc.0.bias, fc.2.weight, fc.2.bias
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        rul_pred = self.fc(out[:, -1, :])
        return rul_pred

    def load_weights(self, weights_path: str):
        try:
            state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
            self.load_state_dict(state_dict)
            self.eval()
            print(f"[+] Loaded trained LSTM RUL model weights from: {weights_path}")
        except Exception as e:
            print(f"[!] Could not load LSTM weights: {e}")

    def predict_rul(self, sequence_tensor: torch.Tensor) -> tuple[float, float]:
        """Returns (rul_hours, health_index_pct)."""
        self.eval()
        with torch.no_grad():
            try:
                raw_pred = self.forward(sequence_tensor).item()
                rul_hours = max(0.0, min(1200.0, raw_pred))
            except Exception:
                rul_hours = 1180.0

            health_index_pct = max(0.0, min(100.0, (rul_hours / 1200.0) * 100.0))
            return rul_hours, health_index_pct