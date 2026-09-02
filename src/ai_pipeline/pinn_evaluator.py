import os
import torch
import logging
from src.models.pinn_model import PhysicsInformedNN

logging.basicConfig(level=logging.INFO)

class PINNEvaluator:
    def __init__(self, model_path: str = None):
        """
        AI Pipeline wrapper for the Physics-Informed Neural Network (PINN).
        Handles normalization, physical constraint evaluation, and anomaly inference.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pinn = PhysicsInformedNN().to(self.device)
        
        weights = model_path or os.getenv("PINN_MODEL_PATH", "pinn_model_latest.pth")
        if os.path.exists(weights):
            try:
                self.pinn.load_state_dict(torch.load(weights, map_location=self.device))
                self.pinn.eval()
                logging.info(f"[+] PINNEvaluator loaded weights from: {weights}")
            except Exception as e:
                logging.warning(f"[!] Could not load PINN state dict: {e}")
        else:
            logging.warning(f"[!] PINN weights file '{weights}' not found. Initializing unweighted inference.")

    def evaluate_residuals(self, delta_cht: float, delta_egt: float, delta_oil: float = 0.0) -> dict:
        """
        Evaluates physical residual deltas through the PINN model.
        Returns anomaly confidence score and binary flag.
        """
        # Feature vector: [ΔCHT, ΔEGT, ΔOil_Pressure]
        input_tensor = torch.tensor([[delta_cht, delta_egt, delta_oil]], dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            anomaly_score = self.pinn(input_tensor).item()
            
        # Hard rule override + PINN neural score decision
        physical_limit_exceeded = abs(delta_cht) > 15.0 or abs(delta_egt) > 40.0
        is_anomaly = physical_limit_exceeded or (anomaly_score > 0.5)

        return {
            "pinn_anomaly_score": round(anomaly_score, 4),
            "anomaly_flag": is_anomaly,
            "physical_override": physical_limit_exceeded
        }