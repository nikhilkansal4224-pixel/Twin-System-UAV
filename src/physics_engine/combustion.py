import numpy as np
import pandas as pd


class WiebeCombustionModel:
    def __init__(self, lhv_fuel: float = 44.0e6, a_wiebe: float = 5.0, m_wiebe: float = 2.0):
        """
        Initializes chemical combustion parameters for aero-piston engines.
        
        :param lhv_fuel: Lower Heating Value of aviation fuel (J/kg) [default: 44 MJ/kg]
        :param a_wiebe: Efficiency form factor parameter
        :param m_wiebe: Combustion shape factor parameter
        """
        self.lhv_fuel = lhv_fuel
        self.a = a_wiebe
        self.m = m_wiebe

    def mass_fraction_burned(self, theta: float, theta_start: float, delta_theta: float) -> float:
        """Calculates cumulative mass fraction of fuel burned x_b(theta)."""
        if theta < theta_start:
            return 0.0
        elif theta > (theta_start + delta_theta):
            return 1.0

        y = (theta - theta_start) / delta_theta
        xb = 1.0 - np.exp(-self.a * (y ** (self.m + 1.0)))
        return float(xb)

    def burn_rate_derivative(self, theta: float, theta_start: float, delta_theta: float) -> float:
        """Calculates instantaneous burn rate derivative d(x_b)/d(theta)."""
        if theta < theta_start or theta > (theta_start + delta_theta):
            return 0.0

        y = (theta - theta_start) / delta_theta
        dxb_dtheta = (self.a * (self.m + 1.0) / delta_theta) * (y ** self.m) * np.exp(-self.a * (y ** (self.m + 1.0)))
        return float(dxb_dtheta)

    def compute_heat_release_rate(
        self, 
        theta: float, 
        rpm: float, 
        fuel_mass_per_cycle: float = 0.000035, 
        spark_advance_deg: float = -15.0, 
        combustion_duration_deg: float = 60.0
    ) -> dict:
        """Computes instantaneous chemical heat release rate dQ_in/dt (Joules/sec or Watts)."""
        theta_start = np.radians(spark_advance_deg)
        delta_theta = np.radians(combustion_duration_deg)
        omega = (2.0 * np.pi * rpm) / 60.0  # Angular velocity (rad/s)

        xb = self.mass_fraction_burned(theta, theta_start, delta_theta)
        dxb_dtheta = self.burn_rate_derivative(theta, theta_start, delta_theta)

        dq_in_dt = fuel_mass_per_cycle * self.lhv_fuel * dxb_dtheta * omega
        total_heat_capacity = fuel_mass_per_cycle * self.lhv_fuel

        return {
            "dq_in_dt": round(float(dq_in_dt), 3),
            "mass_fraction_burned": round(xb, 4),
            "dxb_dtheta": round(dxb_dtheta, 4),
            "cumulative_heat_released": round(xb * total_heat_capacity, 2)
        }

    def compute_vectorized_heat_release(
        self, 
        rpm_array: np.ndarray, 
        fuel_mass_per_cycle: float = 0.000035, 
        spark_advance_deg: float = -15.0, 
        combustion_duration_deg: float = 60.0
    ) -> np.ndarray:
        """
        Fast vectorized computation of peak dQ_in/dt across an entire telemetry DataFrame column.
        Evaluates at peak heat release crank angle (theta = 0.0 rad).
        """
        theta_start = np.radians(spark_advance_deg)
        delta_theta = np.radians(combustion_duration_deg)
        theta_peak = 0.0  # Top Dead Center (TDC) peak burn point

        dxb_dtheta = self.burn_rate_derivative(theta_peak, theta_start, delta_theta)
        omega_array = (2.0 * np.pi * rpm_array) / 60.0

        dq_in_array = fuel_mass_per_cycle * self.lhv_fuel * dxb_dtheta * omega_array
        return np.round(dq_in_array, 3)