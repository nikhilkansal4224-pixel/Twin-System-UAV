import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# Ensure project root is accessible
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.ai_pipeline.pinn_model import PhysicsInformedNN
from src.ai_pipeline.loss_functions import PhysicsInformedLoss
from src.physics_engine.telemetry_physics import TelemetryPhysicsEngine
from src.physics_engine.kinematics import CrankSliderKinematics

# =====================================================================
# 1. PYTORCH DATASET CLASS FOR PINN
# =====================================================================
class TelemetryPINNDataset(Dataset):
    def __init__(self, processed_df: pd.DataFrame):
        """
        Extracts 4D input feature deltas, prediction targets, and 
        thermodynamic context tensors for physics loss evaluation.
        """
        # 1. Input Features: [Delta_CHT, Delta_EGT, Delta_Oil_P, Delta_MAP]
        self.inputs = torch.tensor(
            processed_df[['residual_cht', 'residual_egt', 'residual_oil', 'map_kpa']].values,
            dtype=torch.float32
        )

        # 2. Training Targets: [Pressure_Pa, Temp_K, dT_dt, Health_Severity]
        p_pa = processed_df['map_kpa'].values * 1000.0
        t_k = processed_df['cht_c'].values + 273.15
        dt_dt = np.gradient(t_k, processed_df['timestamp_s'].values if 'timestamp_s' in processed_df.columns else 0.1)
        severity = 1.0 - (processed_df['health_index_pct'].values / 100.0)

        self.targets = torch.tensor(
            np.column_stack([p_pa, t_k, dt_dt, severity]),
            dtype=torch.float32
        )

        # 3. Thermodynamic Physics Context Parameters
        kinematics = CrankSliderKinematics()
        v_curr, dv_dt, _ = kinematics.compute_vectorized_kinematics(
            theta_array=np.zeros(len(processed_df)),  # Evaluated at peak combustion angle
            rpm_array=processed_df['rpm'].values
        )

        # Estimate mass of trapped air/fuel mixture (PV = mRT)
        r_air = 287.05
        m_gas = p_pa * v_curr / (r_air * np.maximum(t_k, 200.0))

        self.physics_context = {
            "volume": torch.tensor(v_curr, dtype=torch.float32),
            "mass": torch.tensor(m_gas, dtype=torch.float32),
            "dq_in": torch.tensor(processed_df['physics_dq_in'].values, dtype=torch.float32),
            "dq_wall": torch.tensor(processed_df['physics_dq_wall'].values, dtype=torch.float32),
            "dv_dt": torch.tensor(dv_dt, dtype=torch.float32)
        }

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        ctx = {k: v[idx] for k, v in self.physics_context.items()}
        return self.inputs[idx], self.targets[idx], ctx


# =====================================================================
# 2. TRAINING ROUTINE
# =====================================================================
def train_pinn_model(
    data_path: str = None, 
    epochs: int = 15, 
    batch_size: int = 32, 
    lr: float = 1e-3, 
    lambda_phys: float = 0.25
):
    print("=" * 70)
    print("[+] Starting Physics-Informed Neural Network (PINN) Training Workflow")
    print("=" * 70)

    # 1. Load Data
    if data_path is None:
        data_path = os.path.join(project_root, "data", "uav_telemetry_sample.xlsx")
        if not os.path.exists(data_path):
            data_path = os.path.join(project_root, "uav_telemetry_sample.xlsx")

    print(f"[1/4] Loading telemetry dataset from: {data_path}")
    raw_df = pd.read_excel(data_path, header=1) if data_path.endswith('.xlsx') else pd.read_csv(data_path)

    # 2. Preprocess via Physics Engine
    print("[2/4] Preprocessing baselines and calculating thermodynamic rates...")
    engine = TelemetryPhysicsEngine()
    processed_df = engine.process_dataframe(raw_df)

    # 3. Create Dataset & DataLoader
    dataset = TelemetryPINNDataset(processed_df)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 4. Initialize Network, Loss, and Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[3/4] Initializing PINN Architecture on Device: {device}")

    model = PhysicsInformedNN(input_dim=4, hidden_dim=64, output_dim=4).to(device)
    criterion = PhysicsInformedLoss(lambda_physics=lambda_phys).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 5. Optimization Loop
    print(f"[4/4] Executing Training Optimization Loop ({epochs} Epochs)...")
    print("-" * 70)
    print(f"{'Epoch':<8} | {'Total Loss':<12} | {'Data Loss':<12} | {'Gas Loss':<12} | {'Energy Loss':<12}")
    print("-" * 70)

    model.train()
    for epoch in range(1, epochs + 1):
        running_total, running_data, running_gas, running_energy = 0.0, 0.0, 0.0, 0.0

        for batch_inputs, batch_targets, batch_ctx in dataloader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            batch_ctx = {k: v.to(device) for k, v in batch_ctx.items()}

            optimizer.zero_grad()

            predictions = model(batch_inputs)
            loss_total, loss_data, loss_gas, loss_energy = criterion(predictions, batch_targets, batch_ctx)

            loss_total.backward()
            optimizer.step()

            running_total += loss_total.item()
            running_data += loss_data.item()
            running_gas += loss_gas.item()
            running_energy += loss_energy.item()

        num_batches = len(dataloader)
        print(
            f"{epoch:<8} | "
            f"{running_total / num_batches:<12.4f} | "
            f"{running_data / num_batches:<12.4f} | "
            f"{running_gas / num_batches:<12.4f} | "
            f"{running_energy / num_batches:<12.4f}"
        )

    # 6. Save Trained Artifacts
    output_model_path = os.path.join(project_root, "pinn_model_latest.pth")
    torch.save(model.state_dict(), output_model_path)
    print("-" * 70)
    print(f"[+] PINN Training Complete. Weights saved to: {output_model_path}\n")


if __name__ == "__main__":
    train_pinn_model()