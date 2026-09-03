import os
import logging
import numpy as np
import pandas as pd
import psycopg
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from dotenv import load_dotenv
import optuna

load_dotenv()
logging.basicConfig(level=logging.INFO)

# Database Connection Info
# src/ai_pipeline/retrain_pipeline.py

DB_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://uav_user:uav_password@127.0.0.1:5432/uav_telemetry"
)

# PINN Architecture (Parameterized for Tuning)
class ParametricPINN(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=2, output_dim=4):
        super(ParametricPINN, self).__init__()
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


class OptunaRetrainer:
    def __init__(self, db_url=DB_URL):
        self.db_url = db_url
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fetch_historical_dataset(self) -> pd.DataFrame:
        """Loads telemetry metrics from PostgreSQL."""
        query = """
        SELECT 
            actual_cht, actual_egt, actual_oil,
            physics_cht, physics_egt,
            residual_cht, residual_egt, residual_oil,
            health_index_pct
        FROM uav_aero_engine_metrics
        ORDER BY created_at DESC
        LIMIT 20000;
        """
        try:
            with psycopg.connect(self.db_url) as conn:
                df = pd.read_sql_query(query, conn)
                logging.info(f"[+] Loaded {len(df)} records from PostgreSQL for Optuna optimization.")
                return df
        except Exception as e:
            logging.error(f"[!] Database read error: {e}")
            return pd.DataFrame()

    def objective(self, trial: optuna.Trial, df: pd.DataFrame) -> float:
        """Optuna trial objective function."""
        # 1. Hyperparameter Search Space Definition
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
        num_layers = trial.suggest_int("num_layers", 1, 4)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "AdamW"])

        # 2. Data Preparation & Train/Val Split
        X = df[['actual_cht', 'actual_egt', 'actual_oil', 'physics_cht', 'physics_egt']].values
        Y = df[['residual_cht', 'residual_egt', 'residual_oil', 'health_index_pct']].values

        split_idx = int(len(df) * 0.8)
        X_train, X_val = torch.tensor(X[:split_idx], dtype=torch.float32), torch.tensor(X[split_idx:], dtype=torch.float32)
        Y_train, Y_val = torch.tensor(Y[:split_idx], dtype=torch.float32), torch.tensor(Y[split_idx:], dtype=torch.float32)

        train_dataset = TensorDataset(X_train, Y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model = ParametricPINN(
            input_dim=5, 
            hidden_dim=hidden_dim, 
            num_layers=num_layers, 
            output_dim=4
        ).to(self.device)

        # 3. Dynamic Optimizer Selection
        if optimizer_name == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == "RMSprop":
            optimizer = optim.RMSprop(model.parameters(), lr=lr)
        else:
            optimizer = optim.AdamW(model.parameters(), lr=lr)

        criterion = nn.MSELoss()

        # 4. Training & Validation Loop with Pruning
        epochs = 15
        for epoch in range(epochs):
            model.train()
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()

            # Validation Evaluation
            model.eval()
            with torch.no_grad():
                val_x, val_y = X_val.to(self.device), Y_val.to(self.device)
                val_preds = model(val_x)
                val_loss = criterion(val_preds, val_y).item()

            # Optuna Pruning Trigger for Non-Performing Trials
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return val_loss

    def run_study(self, n_trials=25):
        """Executes Optuna optimization study."""
        df = self.fetch_historical_dataset()
        if len(df) < 100:
            logging.error("[!] Not enough historical metrics available for hyperparameter tuning.")
            return

        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        )
        
        logging.info(f"[+] Starting Optuna optimization study across {n_trials} trials...")
        study.optimize(lambda trial: self.objective(trial, df), n_trials=n_trials)

        logging.info("\n" + "=" * 60)
        logging.info("[✔] OPTUNA HYPERPARAMETER TUNING COMPLETE")
        logging.info(f"    Best Validation Loss: {study.best_value:.6f}")
        logging.info("    Optimal Parameters:")
        for k, v in study.best_params.items():
            logging.info(f"      - {k}: {v}")
        logging.info("=" * 60)

        return study.best_params


if __name__ == "__main__":
    retrainer = OptunaRetrainer()
    best_hyperparams = retrainer.run_study(n_trials=20)