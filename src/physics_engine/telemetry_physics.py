import os
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Support cross-module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.physics_engine.combustion import WiebeCombustionModel
from src.physics_engine.heat_transfer import WoschniHeatTransferModel
from src.physics_engine.kinematics import CrankSliderKinematics


class TelemetryPhysicsEngine:
    def __init__(self):
        self.cht_model = LinearRegression()
        self.egt_model = LinearRegression()
        self.oil_model = LinearRegression()
        
        self.combustion_model = WiebeCombustionModel()
        self.heat_transfer_model = WoschniHeatTransferModel()
        self.kinematics_model = CrankSliderKinematics()
        
        self.is_calibrated = False

    def calibrate_models(self, nominal_df: pd.DataFrame):
        """Fits empirical regression models on nominal telemetry data."""
        X = nominal_df[['rpm', 'map_kpa', 'ambient_temp_c', 'altitude_m']]
        self.cht_model.fit(X, nominal_df['cht_c'])
        self.egt_model.fit(X, nominal_df['egt_c'])
        self.oil_model.fit(X, nominal_df['oil_pressure_bar'])
        self.is_calibrated = True

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes empirical baselines, kinematic states, thermodynamic heat rates (dQ_in, dQ_wall),
        EMA residuals, Health Index, and RUL.
        """
        # Ensure column standardizations for case-insensitive inputs
        column_map = {col: col.lower() for col in df.columns}
        df = df.rename(columns=column_map)

        if not self.is_calibrated:
            nom_data = df[df['fault_injected'] == 0] if 'fault_injected' in df.columns else df
            self.calibrate_models(nom_data)

        X = df[['rpm', 'map_kpa', 'ambient_temp_c', 'altitude_m']]

        # 1. Empirical Baselines
        df['physics_cht'] = self.cht_model.predict(X)
        df['physics_egt'] = self.egt_model.predict(X)
        df['physics_oil'] = self.oil_model.predict(X)

        # 2. Vectorized Kinematics & Thermodynamic Heat Rates
        rpm_vals = df['rpm'].values
        map_pa_vals = df['map_kpa'].values * 1000.0
        cht_k_vals = df['cht_c'].values + 273.15

        df['physics_dq_in'] = self.combustion_model.compute_vectorized_heat_release(rpm_vals)
        df['physics_dq_wall'] = self.heat_transfer_model.compute_vectorized_wall_heat_loss(
            p_pa_array=map_pa_vals,
            t_k_array=cht_k_vals,
            rpm_array=rpm_vals
        )

        # 3. Raw Residual Deltas
        df['residual_cht_raw'] = df['cht_c'] - df['physics_cht']
        df['residual_egt_raw'] = df['egt_c'] - df['physics_egt']
        df['residual_oil_raw'] = df['oil_pressure_bar'] - df['physics_oil']

        # 4. Exponential Moving Average (EMA) Filtering (20-sample window @ 10Hz = 2.0s)
        df['residual_cht'] = df['residual_cht_raw'].ewm(span=20).mean()
        df['residual_egt'] = df['residual_egt_raw'].ewm(span=20).mean()
        df['residual_oil'] = df['residual_oil_raw'].ewm(span=20).mean()

        # 5. Composite Health Index (%)
        cht_pen = np.maximum(0.0, df['residual_cht']) * 2.5
        egt_pen = np.maximum(0.0, df['residual_egt']) * 1.0
        oil_pen = np.maximum(0.0, -df['residual_oil']) * 120.0

        df['health_index_pct'] = np.maximum(0.0, 100.0 - (cht_pen + egt_pen + oil_pen))

        # 6. Calibrated Remaining Useful Life (RUL) Hours
        df['rul_hours'] = 1200.0 * ((df['health_index_pct'] / 100.0) ** 2.0)

        # 7. Urgency Classification & Anomaly Flags
        def assign_urgency(row):
            if row['health_index_pct'] < 50.0 or row['residual_cht'] > 15.0:
                return 'CRITICAL'
            elif row['health_index_pct'] < 80.0 or row['residual_cht'] > 8.0:
                return 'WARNING'
            elif row['health_index_pct'] < 95.0 or row['residual_cht'] > 4.0:
                return 'ELEVATED'
            return 'NOMINAL'

        df['maintenance_urgency'] = df.apply(assign_urgency, axis=1)
        df['can_id'] = '0x100'
        df['anomaly_flag'] = df['fault_injected'].astype(int) if 'fault_injected' in df.columns else 0

        return df


# =====================================================================
# VERIFICATION TEST
# =====================================================================
if __name__ == "__main__":
    print("[+] Verification testing Telemetry Physics Engine...")
    
    # Generate dummy telemetry frame
    sample_data = {
        "timestamp_s": [0.1, 0.2, 0.3],
        "rpm": [5800.0, 5810.0, 5790.0],
        "map_kpa": [101.3, 101.5, 101.1],
        "cht_c": [115.2, 115.5, 116.0],
        "egt_c": [820.1, 821.0, 819.5],
        "oil_pressure_bar": [4.50, 4.49, 4.51],
        "ambient_temp_c": [15.0, 15.0, 15.0],
        "altitude_m": [0.0, 0.0, 0.0],
        "fault_injected": [0, 0, 0]
    }
    
    dummy_df = pd.DataFrame(sample_data)
    engine = TelemetryPhysicsEngine()
    processed = engine.process_dataframe(dummy_df)
    
    print("[+] Successfully processed telemetry frame. Output columns:")
    print(processed[['physics_cht', 'physics_dq_in', 'physics_dq_wall', 'health_index_pct', 'maintenance_urgency']])