import torch
import torch.nn as nn

class PhysicsInformedLoss(nn.Module):
    def __init__(self, lambda_physics=0.2, R_air=287.05, Cv_air=718.0):
        """
        Initializes the Physics-Informed Loss Module.
        
        :param lambda_physics: Weighting factor lambda for the physics penalty term
        :param R_air: Specific Gas Constant for air (J/kg*K)
        :param Cv_air: Specific Heat capacity at constant volume (J/kg*K)
        """
        super(PhysicsInformedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.lambda_phys = lambda_physics
        self.R = R_air
        self.Cv = Cv_air

    def compute_ideal_gas_residual(self, p_pred, t_pred, volume_tensor, mass_tensor):
        """
        Calculates Ideal Gas Law Physics Penalty: L_gas = mean((P * V - m * R * T)^2)
        """
        gas_residual = (p_pred * volume_tensor) - (mass_tensor * self.R * t_pred)
        return torch.mean(gas_residual ** 2)

    def compute_first_law_residual(self, p_pred, t_pred, dt_dt_pred, dq_in_tensor, dq_wall_tensor, dv_dt_tensor, mass_tensor):
        """
        Calculates First Law Energy Conservation Penalty:
        L_energy = mean((m * Cv * dT/dt - (dQ_in/dt - dQ_wall/dt - P * dV/dt))^2)
        """
        internal_energy_rate = mass_tensor * self.Cv * dt_dt_pred
        work_rate = p_pred * dv_dt_tensor
        energy_balance_residual = internal_energy_rate - (dq_in_tensor - dq_wall_tensor - work_rate)
        return torch.mean(energy_balance_residual ** 2)

    def forward(self, predictions, targets, physics_context):
        """
        Evaluates the combined total PINN loss.
        
        :param predictions: Model tensor [Batch, 4] -> [P_pred, T_pred, dT_dt_pred, Severity_score]
        :param targets: Ground truth labels [Batch, 4]
        :param physics_context: Dictionary containing volume, mass, dQ_in, dQ_wall, and dV_dt tensors
        :return: Tuple (L_total, L_data, L_physics_gas, L_physics_energy)
        """
        # 1. Standard Supervised Data Loss (MSE)
        loss_data = self.mse_loss(predictions, targets)

        # Extract predictions for physical boundary validation
        p_pred = predictions[:, 0]        # Predicted Pressure (Pa)
        t_pred = predictions[:, 1]        # Predicted Temperature (K)
        dt_dt_pred = predictions[:, 2]    # Predicted Temperature derivative (K/s)

        # Unpack physical state context tensors
        vol = physics_context["volume"]
        mass = physics_context["mass"]
        dq_in = physics_context["dq_in"]
        dq_wall = physics_context["dq_wall"]
        dv_dt = physics_context["dv_dt"]

        # 2. Physics Constraint Penalty Computations
        loss_phys_gas = self.compute_ideal_gas_residual(p_pred, t_pred, vol, mass)
        loss_phys_energy = self.compute_first_law_residual(p_pred, t_pred, dt_dt_pred, dq_in, dq_wall, dv_dt, mass)

        # Total Physics Residual Penalty
        loss_physics_total = loss_phys_gas + (0.001 * loss_phys_energy)

        # 3. Combined Total Loss: L_total = L_data + lambda * L_physics
        loss_total = loss_data + (self.lambda_phys * loss_physics_total)

        return loss_total, loss_data, loss_phys_gas, loss_phys_energy


# =====================================================================
# MODULE VERIFICATION & TEST DRIVE
# =====================================================================
if __name__ == "__main__":
    print("[+] Testing Physics-Informed Custom Loss Module...")
    loss_calculator = PhysicsInformedLoss(lambda_physics=0.25)

    batch_size = 4
    # Dummy Network Output Predictions: [P_pred, T_pred, dT_dt_pred, Severity]
    dummy_preds = torch.tensor([
        [101325.0, 350.0, 120.0, 0.05],
        [150000.0, 420.0, 450.0, 0.85],
        [102000.0, 355.0, 140.0, 0.40],
        [140000.0, 410.0, 380.0, 0.75]
    ], dtype=torch.float32, requires_grad=True)

    dummy_targets = torch.tensor([
        [101325.0, 350.0, 115.0, 0.00],
        [148000.0, 415.0, 430.0, 0.80],
        [101500.0, 352.0, 135.0, 0.35],
        [138000.0, 405.0, 370.0, 0.70]
    ], dtype=torch.float32)

    # Physical Context Input Tensors
    dummy_context = {
        "volume": torch.tensor([0.0003, 0.0003, 0.0003, 0.0003], dtype=torch.float32),
        "mass": torch.tensor([0.00035, 0.00035, 0.00035, 0.00035], dtype=torch.float32),
        "dq_in": torch.tensor([1500.0, 3500.0, 1600.0, 3200.0], dtype=torch.float32),
        "dq_wall": torch.tensor([200.0, 600.0, 220.0, 550.0], dtype=torch.float32),
        "dv_dt": torch.tensor([0.001, 0.002, 0.001, 0.002], dtype=torch.float32)
    }

    # Evaluate Loss Components
    l_total, l_data, l_gas, l_energy = loss_calculator(dummy_preds, dummy_targets, dummy_context)

    print("\n--- [PINN LOSS EVALUATION SUMMARY] ---")
    print(f"Total Combined Loss (L_total) : {l_total.item():.4f}")
    print(f"Supervised Data Loss (L_data)  : {l_data.item():.4f}")
    print(f"Gas Law Residual (L_gas)       : {l_gas.item():.4f}")
    print(f"1st Law Residual (L_energy)   : {l_energy.item():.4f}")