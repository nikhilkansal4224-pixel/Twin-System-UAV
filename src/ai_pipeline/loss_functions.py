import os
import sys
import torch
import torch.nn as nn

# Ensure project root is accessible
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class PhysicsInformedLoss(nn.Module):
    def __init__(
        self, 
        lambda_physics: float = 0.25, 
        R_air: float = 287.05, 
        Cv_air: float = 718.0,
        p_scale: float = 1e5,    # Standard pressure scaling constant (100 kPa / 1 bar)
        e_scale: float = 1e4     # Energy rate scaling constant (10 kW)
    ):
        """
        Initializes the Physics-Informed Loss Module for PINN training.
        
        :param lambda_physics: Multiplier weighting factor for total physics residual
        :param R_air: Specific Gas Constant for air (J/kg*K)
        :param Cv_air: Specific Heat capacity at constant volume (J/kg*K)
        :param p_scale: Normalization factor for Ideal Gas pressure-volume residuals
        :param e_scale: Normalization factor for First Law power/energy balance residuals
        """
        super(PhysicsInformedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.lambda_phys = lambda_physics
        self.R = R_air
        self.Cv = Cv_air
        self.p_scale = p_scale
        self.e_scale = e_scale

    def compute_ideal_gas_residual(
        self, 
        p_pred: torch.Tensor, 
        t_pred: torch.Tensor, 
        vol: torch.Tensor, 
        mass: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates normalized Ideal Gas Law Physics Penalty:
        L_gas = mean( ((P * V - m * R * T) / P_scale * V)^2 )
        """
        gas_residual = (p_pred * vol) - (mass * self.R * t_pred)
        # Normalize residual relative to pressure scale to keep gradients numerically stable
        normalized_residual = gas_residual / (self.p_scale * vol + 1e-8)
        return torch.mean(normalized_residual ** 2)

    def compute_first_law_residual(
        self, 
        p_pred: torch.Tensor, 
        t_pred: torch.Tensor, 
        dt_dt_pred: torch.Tensor, 
        dq_in: torch.Tensor, 
        dq_wall: torch.Tensor, 
        dv_dt: torch.Tensor, 
        mass: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates normalized First Law Energy Conservation Penalty:
        L_energy = mean( ((m * Cv * dT/dt - (dQ_in/dt - dQ_wall/dt - P * dV/dt)) / E_scale)^2 )
        """
        internal_energy_rate = mass * self.Cv * dt_dt_pred
        work_rate = p_pred * dv_dt
        
        energy_balance_residual = internal_energy_rate - (dq_in - dq_wall - work_rate)
        # Scale down large thermal power values (Watts) to O(1)
        normalized_residual = energy_balance_residual / self.e_scale
        return torch.mean(normalized_residual ** 2)

    def forward(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor, 
        physics_context: dict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluates combined total PINN loss.
        
        :param predictions: Model tensor [Batch, 4] -> [P_pred, T_pred, dT_dt_pred, Severity_score]
        :param targets: Ground truth labels [Batch, 4]
        :param physics_context: Dictionary containing volume, mass, dq_in, dq_wall, and dv_dt tensors
        :return: Tuple (L_total, L_data, L_physics_gas, L_physics_energy)
        """
        # 1. Supervised Data Loss (MSE)
        loss_data = self.mse_loss(predictions, targets)

        # Extract predictions (flattened to 1D to prevent broadcasting bugs)
        p_pred = predictions[:, 0].view(-1)
        t_pred = predictions[:, 1].view(-1)
        dt_dt_pred = predictions[:, 2].view(-1)

        # Extract & squeeze physical context tensors
        vol = physics_context["volume"].view(-1)
        mass = physics_context["mass"].view(-1)
        dq_in = physics_context["dq_in"].view(-1)
        dq_wall = physics_context["dq_wall"].view(-1)
        dv_dt = physics_context["dv_dt"].view(-1)

        # 2. Compute Physics Residual Losses
        loss_phys_gas = self.compute_ideal_gas_residual(p_pred, t_pred, vol, mass)
        loss_phys_energy = self.compute_first_law_residual(p_pred, t_pred, dt_dt_pred, dq_in, dq_wall, dv_dt, mass)

        # Total Physics Loss Component
        loss_physics_total = loss_phys_gas + loss_phys_energy

        # 3. Combined Loss
        loss_total = loss_data + (self.lambda_phys * loss_physics_total)

        return loss_total, loss_data, loss_phys_gas, loss_phys_energy


# =====================================================================
# MODULE VERIFICATION & TEST DRIVE
# =====================================================================
if __name__ == "__main__":
    print("[+] Testing Physics-Informed Custom Loss Module...")
    loss_calculator = PhysicsInformedLoss(lambda_physics=0.25)

    # Simulated Batch of Network Output Predictions: [P_pred, T_pred, dT_dt_pred, Severity]
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
    print(f"Total Combined Loss (L_total) : {l_total.item():.6f}")
    print(f"Supervised Data Loss (L_data)  : {l_data.item():.6f}")
    print(f"Gas Law Residual (L_gas)       : {l_gas.item():.6f}")
    print(f"1st Law Residual (L_energy)   : {l_energy.item():.6f}")