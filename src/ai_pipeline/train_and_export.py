import os
import sys
import torch
import torch.optim as optim
import numpy as np

# Ensure project root is in Python search path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Handle cross-module architecture components
from src.models.pinn_model import PhysicsInformedNN
from src.ai_pipeline.loss_functions import PhysicsInformedLoss
from src.ai_pipeline.lstm_rul import LSTMRULEstimator
from src.ai_pipeline.fault_generator import SyntheticFaultInverter

# =====================================================================
# GLOBAL HARDWARE DEVICE CONTEXT DETECTION
# =====================================================================
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"[+] Directing compute workloads to accelerator target: {device}")

# Create directory structures to store serialized weights
WEIGHTS_DIR = os.path.join(project_root, "models", "saved_weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

PINN_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "pinn_weights.pth")
LSTM_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "lstm_rul_weights.pth")


# =====================================================================
# 1. PINN OFFLINE TRAINING ROUTINE
# =====================================================================
def train_pinn_model(epochs=200, lr=0.001):
    print("\n[+] Initializing Offline PINN Training Routine...")
    pinn = PhysicsInformedNN(input_dim=4, hidden_dim=64, output_dim=3).to(device)
    loss_calculator = PhysicsInformedLoss(lambda_physics=0.25)
    optimizer = optim.Adam(pinn.parameters(), lr=lr)

    # Generate synthetic training batches
    fault_inverter = SyntheticFaultInverter()
    fault_df = fault_inverter.generate_fault_dataset(num_samples_per_fault=500)

    # Force continuous 32-bit allocation layout right at numpy concatenation phase
    inputs_np = np.column_stack([
        fault_df["CHT"].values - 115.0,
        fault_df["EGT"].values - 820.0,
        fault_df["Oil_Pressure"].values - 4.5,
        np.zeros(len(fault_df))
    ]).astype(np.float32)
    
    targets_np = np.column_stack([
        101325.0 + inputs_np[:, 3] * 1000.0,
        350.0 + inputs_np[:, 0] * 2.0,
        120.0 + inputs_np[:, 1] * 1.5
    ]).astype(np.float32)

    # Fast memory zero-copy assignment to target compute accelerator hardware
    inputs_tensor = torch.from_numpy(inputs_np).to(device)
    targets_tensor = torch.from_numpy(targets_np).to(device)

    # Physical Context Input Tensors allocated natively on device path
    num_samples = len(fault_df)
    physics_context = {
        "volume": torch.full((num_samples,), 0.0003, dtype=torch.float32, device=device),
        "mass": torch.full((num_samples,), 0.00035, dtype=torch.float32, device=device),
        "dq_in": torch.full((num_samples,), 2500.0, dtype=torch.float32, device=device),
        "dq_wall": torch.full((num_samples,), 400.0, dtype=torch.float32, device=device),
        "dv_dt": torch.full((num_samples,), 0.0015, dtype=torch.float32, device=device)
    }

    # Training Loop Execution
    pinn.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        predictions = pinn(inputs_tensor)

        # Match dimensions to [Batch, 4] with unified runtime tracking allocations
        padded_preds = torch.cat([predictions, torch.zeros((num_samples, 1), device=device)], dim=1)
        padded_targets = torch.cat([targets_tensor, torch.zeros((num_samples, 1), device=device)], dim=1)

        l_total, l_data, l_gas, l_energy = loss_calculator(padded_preds, padded_targets, physics_context)
        l_total.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == epochs:
            print(f"    Epoch [{epoch:03d}/{epochs:03d}] | Total Loss: {l_total.item():.4f} | Data Loss: {l_data.item():.4f} | Physics Penalty: {(l_gas + l_energy).item():.4f}")

    # Serialize weights state dict
    torch.save(pinn.state_dict(), PINN_WEIGHTS_PATH)
    print(f"[+] PINN Weights successfully exported to: '{PINN_WEIGHTS_PATH}'")
    return pinn


# =====================================================================
# 2. LSTM RUL OFFLINE TRAINING ROUTINE
# =====================================================================
def train_lstm_rul_model(epochs=150, lr=0.001, sequence_length=10):
    print("\n[+] Initializing Offline LSTM RUL Training Routine...")
    # LSTMRULEstimator now handles batch_first natively internally
    lstm = LSTMRULEstimator(input_size=4, hidden_size=64, num_layers=2, output_size=1).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(lstm.parameters(), lr=lr)

    # Generate synthetic degradation flight sequences
    num_sequences = 200
    sequences = []
    rul_labels = []

    for _ in range(num_sequences):
        start_rul = np.random.uniform(200.0, 1200.0)
        degradation_rate = np.random.uniform(0.5, 2.0)
        
        seq = []
        for step in range(sequence_length):
            delta_cht = (step * 0.5) * degradation_rate
            delta_egt = (step * 1.5) * degradation_rate
            delta_oil = (step * 0.01) * degradation_rate
            delta_map = (step * 0.02) * degradation_rate
            seq.append([delta_cht, delta_egt, delta_oil, delta_map])
            
        # Target output calculation matching clean evaluation scales
        target_rul = max(0.0, start_rul - (sequence_length * degradation_rate * 5.0))
        sequences.append(seq)
        rul_labels.append([target_rul])

    # Construct and pipe tensors directly to active runtime device target
    seq_tensor = torch.tensor(sequences, dtype=torch.float32, device=device)
    labels_tensor = torch.tensor(rul_labels, dtype=torch.float32, device=device)

    # Training Loop Execution
    lstm.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        predictions = lstm(seq_tensor)
        loss = criterion(predictions, labels_tensor)
        loss.backward()
        optimizer.step()

        if epoch % 30 == 0 or epoch == epochs:
            print(f"    Epoch [{epoch:03d}/{epochs:03d}] | RUL MSE Loss: {loss.item():.4f}")

    # Serialize weights state dict
    torch.save(lstm.state_dict(), LSTM_WEIGHTS_PATH)
    print(f"[+] LSTM RUL Weights successfully exported to: '{LSTM_WEIGHTS_PATH}'")
    return lstm


# =====================================================================
# MAIN EXECUTION ENTRY-POINT
# =====================================================================
if __name__ == "__main__":
    print("======================================================================")
    print("UAV Engine Digital Twin — PyTorch Model Training & Weight Exporter")
    print("======================================================================")
    
    train_pinn_model(epochs=200, lr=0.001)
    train_lstm_rul_model(epochs=150, lr=0.001)
    
    print("\n[+] Weight pre-training complete. Weights ready for fast loading in main.py.")
