import numpy as np
import pandas as pd
import os

class SyntheticFaultInverter:
    def __init__(self, baseline_cht=115.0, baseline_egt=820.0, baseline_oil_p=4.5, default_telemetry_path=None):
        self.base_cht = baseline_cht
        self.base_egt = baseline_egt
        self.base_oil_p = baseline_oil_p
        
        # Resolve target project directories dynamically relative to file layer
        if default_telemetry_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
            project_root = os.path.abspath(os.path.join(current_dir, "../..")) if os.path.basename(current_dir) != "Twin-System-UAV" else current_dir
            self.telemetry_path = os.path.join(project_root, "data", "raw_telemetry", "uav_telemetry_sample.csv")
        else:
            self.telemetry_path = default_telemetry_path

    def load_flight_telemetry(self) -> pd.DataFrame:
        """
        Loads the generated time-series flight telemetry CSV and standardizes
        column keys for sequential LSTM modeling or time-domain tracking.
        """
        if not os.path.exists(self.telemetry_path):
            raise FileNotFoundError(
                f"[!] Flight telemetry file missing at: '{self.telemetry_path}'. "
                f"Please execute your telemetry generator script first to build this source file."
            )
        
        df = pd.read_csv(self.telemetry_path)
        
        # Bridge column keys between lower_case files and uppercase PyTorch pipeline shapes
        df = df.rename(columns={
            "cht_c": "CHT",
            "egt_c": "EGT",
            "oil_pressure_bar": "Oil_Pressure",
            "map_kpa": "MAP"
        })
        
        print(f"[+] Loaded and standardized {len(df)} flight sequence logs from: '{os.path.basename(self.telemetry_path)}'")
        return df

    def inject_injector_clog(self, severity: float) -> dict:
        """Lean mixture causes combustion temperature spike and EGT rise."""
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
        """Reduces wall heat transfer coefficient, causing CHT thermal runaway."""
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
        """Generates a complete balanced static synthetic dataset for static PINN training calibration."""
        data_records = []
        
        for _ in range(num_samples_per_fault):
            severity = np.random.uniform(0.1, 1.0)
            data_records.append(self.inject_injector_clog(severity))
            data_records.append(self.inject_cooling_blockage(severity))
            
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
    print("[+] Initializing Unified Synthetic Fault Inversion Engine...")
    generator = SyntheticFaultInverter()
    
    # 1. Test Static Data Generation Loop
    dataset_df = generator.generate_fault_dataset(num_samples_per_fault=100)
    print(f"[+] Static Balanced Matrix generated. Total records: {len(dataset_df)}")
    
    # Create directory tree context if needed
    os.makedirs("data/synthetic_faults", exist_ok=True)
    dataset_df.to_csv("data/synthetic_faults/uav_engine_fault_dataset.csv", index=False)
    print("[+] Balanced Matrix exported to 'data/synthetic_faults/uav_engine_fault_dataset.csv'")
    
    # 2. Test Time-Series Ingestion Path
    try:
        flight_df = generator.load_flight_telemetry()
        print(f"    Sample Row Columns: {list(flight_df.columns)}")
    except FileNotFoundError as e:
        print(f"\n[!] Notice: Time-series ingestion test skipped until data folder is generated.\n    ({e})")
