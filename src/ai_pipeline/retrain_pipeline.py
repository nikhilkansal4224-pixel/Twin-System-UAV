import os
import logging
import psycopg
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)

DB_URL = os.getenv(
    "DATABASE_URL", 
    f"postgresql://{os.getenv('POSTGRES_USER', 'grafana')}:{os.getenv('POSTGRES_PASSWORD', 'Grafana@123')}@{os.getenv('POSTGRES_HOST', 'localhost')}:{os.getenv('POSTGRES_PORT', '5432')}/{os.getenv('POSTGRES_DB', 'grafana')}"
)

# ---------------------------------------------------------------------------
# Optuna-Optimized Parametric PINN (num_layers=3, hidden_dim=32)
# ---------------------------------------------------------------------------
class TunedUAVEnginePINN(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=32, num_layers=3, output_dim=4):
        super(TunedUAVEnginePINN, self).__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class RetrainingWorker:
    def __init__(self, min_samples_required=1000):
        self.min_samples = min_samples_required
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Optimal Hyperparameters from Optuna Trial 11
        self.best_lr = 9.87e-05
        self.best_hidden_dim = 32
        self.best_num_layers = 3
        self.best_batch_size = 32

    def fetch_training_data(self) -> pd.DataFrame:
        query = """
        SELECT 
            actual_cht, actual_egt, actual_oil,
            physics_cht, physics_egt,
            residual_cht, residual_egt, residual_oil,
            health_index_pct
        FROM uav_aero_engine_metrics
        ORDER BY created_at DESC
        LIMIT 50000;
        """
        try:
            with psycopg.connect(DB_URL) as conn:
                df = pd.read_sql_query(query, conn)
                logging.info(f"[+] Loaded {len(df)} records from PostgreSQL.")
                return df
        except Exception as e:
            logging.error(f"[!] Database query failed: {e}")
            return pd.DataFrame()

    def retrain_and_promote(self, df: pd.DataFrame, epochs=25, checkpoint_path="pinn_model_latest.pth"):
        if len(df) < self.min_samples:
            logging.warning(f"[!] Insufficient records for retraining ({len(df)}/{self.min_samples}). Skipping.")
            return

        X = df[['actual_cht', 'actual_egt', 'actual_oil', 'physics_cht', 'physics_egt']].values
        Y = df[['residual_cht', 'residual_egt', 'residual_oil', 'health_index_pct']].values

        X_tensor = torch.tensor(X, dtype=torch.float32)
        Y_tensor = torch.tensor(Y, dtype=torch.float32)

        dataset = TensorDataset(X_tensor, Y_tensor)
        loader = DataLoader(dataset, batch_size=self.best_batch_size, shuffle=True)

        model = TunedUAVEnginePINN(
            input_dim=5, 
            hidden_dim=self.best_hidden_dim, 
            num_layers=self.best_num_layers, 
            output_dim=4
        ).to(self.device)

        # Optuna Best Optimizer: Adam
        optimizer = optim.Adam(model.parameters(), lr=self.best_lr)
        criterion = nn.MSELoss()

        model.train()
        logging.info("[+] Retraining PINN with Optuna-tuned parameters...")

        for epoch in range(epochs):
            total_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * batch_x.size(0)

            epoch_loss = total_loss / len(dataset)
            if (epoch + 1) % 5 == 0:
                logging.info(f"    Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.6f}")

        # Atomic Swap Checkpoint Promotion
        temp_checkpoint = "pinn_model_tuned_temp.pth"
        torch.save(model.state_dict(), temp_checkpoint)
        os.replace(temp_checkpoint, checkpoint_path)
        logging.info(f"[✔] Successfully deployed tuned PINN checkpoint to '{checkpoint_path}'!")


if __name__ == "__main__":
    worker = RetrainingWorker(min_samples_required=100)
    data = worker.fetch_training_data()
    if not data.empty:
        worker.retrain_and_promote(data, epochs=25)