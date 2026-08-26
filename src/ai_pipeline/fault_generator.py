import numpy as np
import pandas as pd

class SyntheticFaultInverter:
    def __init__(self, baseline_cht=115.0, baseline_egt=820.0, baseline_oil_p=4.5):
        self.base_cht = baseline_cht
        self.base_egt = baseline_egt
        self.base_oil_p = baseline_oil_p

    def inject_injector_clog(self, severity: float) -> dict:
        """
        Simulates fuel injector restriction.
        Physics Impact: Lean mixture causes combustion temperature spike and EGT rise.
        """
        # Severity from 0.0 (Healthy) to 1.0 (100% Clogged)
        egt_spike = 120.0 * severity
        cht_drift = 25.0 * severity
        
        return {
            "CHT": self.base_cht + cht_drift + np.random.normal(0, 0.5),
            "EGT": self.base_egt + egt_spike + np.random.normal(0, 2.0),
            "Oil_Pressure": self.base_oil_p + np.random.normal(0, 0.02),
            "Fault_Label": "Fuel_Injector_Clog",
            "Severity": round(severity, 2)
        }

    def inject_cooling_blockage(self, severity: float) -> dict:
        """
        Simulates coolant passage restriction or radiator degradation.
        Physics Impact: Reduces wall heat transfer coefficient h_c, causing CHT thermal runaway.
        """
        cht_runaway = 50.0 * severity
        egt_secondary = 15.0 * severity
        
        return {
            "CHT": self.base_cht + cht_runaway + np.random.normal(0, 0.8),
            "EGT": self.base_egt + egt_secondary + np.random.normal(0, 1.5),
            "Oil_Pressure": self.base_oil_p - (0.2 * severity) + np.random.normal(0, 0.02),
            "Fault_Label": "Cooling_Blockage",
            "Severity": round(severity, 2)
        }

    def generate_fault_dataset(self, num_samples_per_fault=100) -> pd.DataFrame:
        """Generates a complete synthetic telemetry dataset for model training."""
        data_records = []
        
        for _ in range(num_samples_per_fault):
            severity = np.random.uniform(0.1, 1.0)
            data_records.append(self.inject_injector_clog(severity))
            data_records.append(self.inject_cooling_blockage(severity))
            
            # Add Healthy Baseline Samples
            data_records.append({
                "CHT": self.base_cht + np.random.normal(0, 0.5),
                "EGT": self.base_egt + np.random.normal(0, 1.5),
                "Oil_Pressure": self.base_oil_p + np.random.normal(0, 0.02),
                "Fault_Label": "Nominal_Healthy",
                "Severity": 0.0
            })
            
        return pd.DataFrame(data_records)


# =====================================================================
# MODULE VERIFICATION & TEST DRIVE
# =====================================================================
if __name__ == "__main__":
    print("[+] Initializing Synthetic Fault Inversion Engine...")
    generator = SyntheticFaultInverter()
    
    # Generate 300 synthetic telemetry samples
    dataset_df = generator.generate_fault_dataset(num_samples_per_fault=100)
    
    print("\n--- [SYNTHETIC FAULT TELEMETRY DATASET SUMMARY] ---")
    print(dataset_df.groupby("Fault_Label").describe().T.iloc[:6])
    
    # Save synthetic dataset to data folder
    dataset_df.to_csv("data/synthetic_faults/uav_engine_fault_dataset.csv", index=False)
    print("\n[+] Synthetic Fault Dataset exported to 'data/synthetic_faults/uav_engine_fault_dataset.csv'")