import torch
import torch.nn as nn

class PhysicsInformedNN(nn.Module):
    def __init__(self, input_dim: int = 4, hidden_dim: int = 64, output_dim: int = 4):
        super(PhysicsInformedNN, self).__init__()
        # Architecture matching checkpoint: [4 -> 64 -> 64 -> 64 -> 4]
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),   # net.0
            nn.Tanh(),                          # net.1
            nn.Linear(hidden_dim, hidden_dim),  # net.2
            nn.Tanh(),                          # net.3
            nn.Linear(hidden_dim, hidden_dim),  # net.4
            nn.Tanh(),                          # net.5
            nn.Linear(hidden_dim, output_dim)   # net.6 (outputs 4 residual projections)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def detect_anomaly(self, delta_cht: float, delta_egt: float, delta_oil: float = 0.0, delta_map: float = 0.0) -> bool:
        """Evaluates model prediction outputs against physical boundaries."""
        if abs(delta_cht) > 15.0 or abs(delta_egt) > 40.0:
            return True
            
        self.eval()
        with torch.no_grad():
            inp = torch.tensor([[delta_cht, delta_egt, delta_oil, delta_map]], dtype=torch.float32)
            out = self.forward(inp)
            # Anomaly triggered if mean predicted residual magnitude exceeds threshold
            return torch.mean(torch.abs(out)).item() > 5.0