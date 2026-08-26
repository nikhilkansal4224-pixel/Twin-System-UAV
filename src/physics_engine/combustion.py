import numpy as np

class WiebeCombustionModel:
    def __init__(self, lhv_fuel=44.0e6, a_wiebe=5.0, m_wiebe=2.0):
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
        """
        Calculates cumulative mass fraction of fuel burned x_b(theta).
        
        :param theta: Current crank angle (radians)
        :param theta_start: Spark ignition start angle (radians)
        :param delta_theta: Total duration of combustion (radians)
        :return: Mass fraction burned value between 0.0 and 1.0
        """
        if theta < theta_start:
            return 0.0
        elif theta > (theta_start + delta_theta):
            return 1.0

        y = (theta - theta_start) / delta_theta
        xb = 1.0 - np.exp(-self.a * (y ** (self.m + 1.0)))
        return float(xb)

    def burn_rate_derivative(self, theta: float, theta_start: float, delta_theta: float) -> float:
        """
        Calculates instantaneous burn rate derivative d(x_b)/d(theta).
        
        :param theta: Current crank angle (radians)
        :param theta_start: Spark ignition start angle (radians)
        :param delta_theta: Total duration of combustion (radians)
        :return: Derivative dxb/dtheta
        """
        if theta < theta_start or theta > (theta_start + delta_theta):
            return 0.0

        y = (theta - theta_start) / delta_theta
        dxb_dtheta = (self.a * (self.m + 1.0) / delta_theta) * (y ** self.m) * np.exp(-self.a * (y ** (self.m + 1.0)))
        return float(dxb_dtheta)

    def compute_heat_release_rate(
        self, 
        theta: float, 
        rpm: float, 
        fuel_mass_per_cycle: float, 
        spark_advance_deg: float = -15.0, 
        combustion_duration_deg: float = 60.0
    ) -> dict:
        """
        Computes instantaneous chemical heat release rate dQ_in/dt (Joules/sec).
        
        :param theta: Current crank angle (radians)
        :param rpm: Engine rotational speed
        :param fuel_mass_per_cycle: Ingested fuel mass per stroke (kg)
        :param spark_advance_deg: Ignition angle relative to TDC (degrees)
        :param combustion_duration_deg: Angular burn window (degrees)
        :return: Dictionary with dQ_in/dt, cumulative heat released, and burned fraction
        """
        # Convert angles from degrees to radians
        theta_start = np.radians(spark_advance_deg)
        delta_theta = np.radians(combustion_duration_deg)
        omega = (2.0 * np.pi * rpm) / 60.0  # Angular velocity (rad/s)

        # 1. Mass fraction burned
        xb = self.mass_fraction_burned(theta, theta_start, delta_theta)

        # 2. Derivative dxb/dtheta
        dxb_dtheta = self.burn_rate_derivative(theta, theta_start, delta_theta)

        # 3. Chemical Heat Release Rate: dQ_in/dt = m_fuel * LHV * (dxb/dtheta) * omega
        dq_in_dt = fuel_mass_per_cycle * self.lhv_fuel * dxb_dtheta * omega
        total_heat_capacity = fuel_mass_per_cycle * self.lhv_fuel

        return {
            "dq_in_dt": round(float(dq_in_dt), 3),             # Thermal power release (Watts / J/s)
            "mass_fraction_burned": round(xb, 4),               # Cumulative x_b
            "dxb_dtheta": round(dxb_dtheta, 4),                 # Instantaneous burn rate
            "cumulative_heat_released": round(xb * total_heat_capacity, 2)  # Total Joules
        }


# =====================================================================
# MODULE VERIFICATION & TEST DRIVE
# =====================================================================
if __name__ == "__main__":
    print("[+] Testing 0D Combustion & Wiebe Heat Release Engine...")
    combustion_model = WiebeCombustionModel()

    rpm_test = 5800.0
    fuel_per_cycle = 0.000035  # ~35 mg of fuel per stroke

    # Evaluate across combustion phase (-20° to +50° Crank Angle)
    test_angles_deg = [-20.0, -15.0, 0.0, 15.0, 30.0, 45.0, 60.0]

    print(f"\n--- [WIEBE BURN PROFILE @ {rpm_test} RPM] ---")
    print(f"{'Crank Angle (deg)':<20} | {'xb (Burned)':<12} | {'dxb/dtheta':<12} | {'dQ_in/dt (kW)':<15}")
    print("-" * 68)

    for deg in test_angles_deg:
        rad = np.radians(deg)
        results = combustion_model.compute_heat_release_rate(
            theta=rad, 
            rpm=rpm_test, 
            fuel_mass_per_cycle=fuel_per_cycle
        )
        dq_kw = results["dq_in_dt"] / 1000.0  # Convert Watts to kW
        print(f"{deg:<20.1f} | {results['mass_fraction_burned']:<12.4f} | {results['dxb_dtheta']:<12.4f} | {dq_kw:<15.2f}")