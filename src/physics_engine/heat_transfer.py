import numpy as np


class WoschniHeatTransferModel:
    def __init__(self, bore: float = 0.0795, stroke: float = 0.061, t_wall_k: float = 420.0, woschni_constant: float = 3.2):
        """
        Initializes heat transfer parameters for aero-piston cylinder walls.
        
        :param bore: Cylinder Bore diameter (m) [default: Rotax 914 bore: 79.5 mm]
        :param stroke: Piston Stroke length (m) [default: 61 mm]
        :param t_wall_k: Cylinder wall boundary temperature (K) [default: 420 K / 146.8 °C]
        :param woschni_constant: Empirical scaling constant C for Woschni formula
        """
        self.bore = bore
        self.stroke = stroke
        self.t_wall = t_wall_k
        self.C = woschni_constant
        self.piston_area = np.pi * (bore / 2.0) ** 2

    def compute_gas_velocity(self, rpm: float, p_bar: float, p_motored_bar: float = 1.0) -> float:
        """Calculates characteristic gas velocity inside cylinder (v_gas) in m/s."""
        mean_piston_speed = 2.0 * self.stroke * (rpm / 60.0)
        
        c1 = 2.28      # Constant during compression/expansion
        c2 = 0.00324   # Combustion pressure rise velocity factor
        
        pressure_diff = max(0.0, p_bar - p_motored_bar)
        v_gas = (c1 * mean_piston_speed) + (c2 * (self.piston_area * self.stroke * 298.0 / (1.013e5 * self.piston_area)) * pressure_diff)
        return float(max(v_gas, mean_piston_speed))

    def compute_heat_transfer_coefficient(self, p_pa: float, t_k: float, rpm: float) -> float:
        """Calculates Woschni heat transfer coefficient h_c (W / (m^2 * K))."""
        p_bar = max(p_pa / 1e5, 0.1)
        t_k = max(t_k, 200.0)
        
        v_gas = self.compute_gas_velocity(rpm, p_bar)
        
        # Woschni Correlation: h_c = C * B^(-0.2) * P^(0.8) * T^(-0.53) * v_gas^(0.8)
        h_c = self.C * (self.bore ** -0.2) * (p_bar ** 0.8) * (t_k ** -0.53) * (v_gas ** 0.8)
        return float(h_c)

    def compute_wall_heat_loss(self, p_pa: float, t_k: float, v_curr_m3: float, rpm: float) -> dict:
        """Computes instantaneous wall heat loss rate dQ_wall/dt (Watts or J/s)."""
        stroke_height = v_curr_m3 / self.piston_area
        exposed_area = (2.0 * self.piston_area) + (np.pi * self.bore * stroke_height)

        h_c = self.compute_heat_transfer_coefficient(p_pa, t_k, rpm)
        dq_wall_dt = h_c * exposed_area * (t_k - self.t_wall)

        return {
            "dq_wall_dt": round(float(dq_wall_dt), 3),
            "h_c": round(float(h_c), 2),
            "exposed_area_m2": round(float(exposed_area), 5)
        }

    def compute_vectorized_wall_heat_loss(
        self, 
        p_pa_array: np.ndarray, 
        t_k_array: np.ndarray, 
        rpm_array: np.ndarray,
        v_curr_m3: float = 0.0003
    ) -> np.ndarray:
        """
        Fast vectorized computation of wall heat loss dQ_wall/dt across an entire DataFrame column.
        """
        p_bar_array = np.maximum(p_pa_array / 1e5, 0.1)
        t_k_clamped = np.maximum(t_k_array, 200.0)
        
        mean_piston_speed = 2.0 * self.stroke * (rpm_array / 60.0)
        v_gas = 2.28 * mean_piston_speed
        
        h_c_array = self.C * (self.bore ** -0.2) * (p_bar_array ** 0.8) * (t_k_clamped ** -0.53) * (v_gas ** 0.8)
        
        stroke_height = v_curr_m3 / self.piston_area
        exposed_area = (2.0 * self.piston_area) + (np.pi * self.bore * stroke_height)
        
        dq_wall_array = h_c_array * exposed_area * (t_k_clamped - self.t_wall)
        return np.round(dq_wall_array, 3)