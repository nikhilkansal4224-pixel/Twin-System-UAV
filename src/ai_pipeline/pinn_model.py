import torch
import torch.nn as nn
import torch.optim as optim

# =====================================================================
# 1. PYTORCH PINN NEURAL NETWORK ARCHITECTURE
# =====================================================================
class PhysicsInformedNN(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=3):
        """
        Input Vector : [Delta_CHT, Delta_EGT, Delta_Oil_P, Delta_MAP]
        Output Vector: [P_pred, T_pred, Degradation_Severity_Score]
        """
        super(PhysicsInformedNN, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),  # Smooth activation function for physical derivatives
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


# =====================================================================
# 2. CUSTOM PHYSICS-INFORMED LOSS FUNCTION
# =====================================================================
class PINNLoss(nn.Module):
    def __init__(self, lambda_physics=0.25, R_air=287.05):
        super(PINNLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.lambda_phys = lambda_physics
        self.R_air = R_air

    def forward(self, predictions, targets, volume_tensor, mass_tensor):
        """
        Calculates L_total = L_data + lambda * L_physics
        """
        # Data loss (MSE against ground truth/labels)
        loss_data = self.mse(predictions, targets)

        # Extract predicted physical states
        p_pred = predictions[:, 0]  # Predicted Pressure (Pa)
        t_pred = predictions[:, 1]  # Predicted Temperature (K)

        # Physics Residual: Ideal Gas Law (PV - mRT = 0)
        # L_physics penalizes outputs violating gas state equations
        ideal_gas_residual = (p_pred * volume_tensor) - (mass_tensor * self.R_air * t_pred)
        loss_physics = torch.mean(ideal_gas_residual ** 2)

        # Combined Loss
        loss_total = loss_data + (self.lambda_phys * loss_physics)
        return loss_total, loss_data, loss_physics


# =====================================================================
# MODULE VERIFICATION & TEST DRIVE
# =====================================================================
if __name__ == "__main__":
    print("[+] Initializing PyTorch Physics-Informed Neural Network Pipeline...")
    model = PhysicsInformedNN()
    criterion = PINNLoss(lambda_physics=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Simulated Batch Data (Batch size = 4)
    # Inputs: Residual deltas [Delta_CHT, Delta_EGT, Delta_Oil_P, Delta_MAP]
    dummy_input_deltas = torch.tensor([
        [1.2, 5.0, 0.05, 0.2],    # Normal state
        [18.5, 52.0, 0.10, 1.2],  # Thermal anomaly (injector/cooling fault)
        [2.1, 8.0, 0.65, 0.1],    # Lubrication drift
        [14.2, 41.0, 0.08, 0.9]   # Severe heat accumulation
    ], dtype=torch.float32)

    dummy_targets = torch.tensor([
        [101325.0, 350.0, 0.02],
        [150000.0, 420.0, 0.85],
        [102000.0, 355.0, 0.45],
        [140000.0, 410.0, 0.75]
    ], dtype=torch.float32)

    dummy_volume = torch.tensor([0.0003, 0.0003, 0.0003, 0.0003], dtype=torch.float32)
    dummy_mass = torch.tensor([0.00035, 0.00035, 0.00035, 0.00035], dtype=torch.float32)

    # Forward Pass & Physics Loss Evaluation
    optimizer.zero_grad()
    outputs = model(dummy_input_deltas)
    loss_total, l_data, l_phys = criterion(outputs, dummy_targets, dummy_volume, dummy_mass)
    
    loss_total.backward()
    optimizer.step()

    print(f"[+] Total Loss: {loss_total.item():.4f} | Data Loss: {l_data.item():.4f} | Physics Penalty: {l_phys.item():.4f}")
    print(f"[+] Inference Predictions Matrix:\n{outputs.detach().numpy()}")