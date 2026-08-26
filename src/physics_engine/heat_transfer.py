import numpy as np

class WoschniHeatTransferModel:
    def __init__(self, bore=0.0795, stroke=0.061, t_wall_k=420.0, woschni_constant=3.2):
        """
        Initializes heat transfer parameters for aero-piston cylinder walls.
        
        :param bore: Cylinder Bore diameter (m) [default: Rotax 914 bore]
        :param stroke: Piston Stroke length (m)
        :param t_wall_k: Cylinder wall boundary temperature (K) [default: 420 K / 146.8 °C]
        :param woschni_constant: Empirical scaling constant C for Woschni formula
        """
        self.bore = bore
        self.stroke = stroke
        self.t_wall = t_wall_k
        self.C = woschni_constant
        self.piston_area = np.pi * (bore / 2.0)**2

    def compute_gas_velocity(self, rpm: float, p_bar: float, p_motored_bar: float = 1.0) -> float:
        """
        Calculates characteristic gas velocity inside cylinder (v_gas) in m/s.
        """
        mean_piston_speed = 2.0 * self.stroke * (rpm / 60.0)
        
        # Pressure spike velocity factor during combustion phase
        c1 = 2.28  # Constant during compression/expansion
        c2 = 0.00324  # Combustion pressure rise velocity factor
        
        pressure_diff = max(0.0, p_bar - p_motored_bar)
        v_gas = (c1 * mean_piston_speed) + (c2 * (self.piston_area * self.stroke * 298.0 / (1.013e5 * self.piston_area)) * pressure_diff)
        return float(max(v_gas, mean_piston_speed))

    def compute_heat_transfer_coefficient(self, p_pa: float, t_k: float, rpm: float) -> float:
        """
        Calculates Woschni heat transfer coefficient h_c (W / (m^2 * K)).
        """
        p_bar = max(p_pa / 1e5, 0.1)  # Convert Pa to bar
        t_k = max(t_k, 200.0)          # Clamp minimum temperature
        
        v_gas = self.compute_gas_velocity(rpm, p_bar)
        
        # Woschni Correlation: h_c = C * B^(-0.2) * P^(0.8) * T^(-0.53) * v_gas^(0.8)
        h_c = self.C * (self.bore ** -0.2) * (p_bar ** 0.8) * (t_k ** -0.53) * (v_gas ** 0.8)
        return float(h_c)

    def compute_wall_heat_loss(self, p_pa: float, t_k: float, v_curr_m3: float, rpm: float) -> dict:
        """
        Computes instantaneous wall heat loss rate dQ_wall/dt (Watts or J/s).
        
        :param p_pa: In-cylinder pressure (Pa)
        :param t_k: In-cylinder gas temperature (K)
        :param v_curr_m3: Instantaneous cylinder volume (m^3)
        :param rpm: Engine rotational speed
        :return: Dictionary containing dQ_wall/dt, h_c, and wall surface area
        """
        # 1. Calculate exposed cylinder wall surface area A(theta)
        # Total area = 2 * Piston Face Area + Cylinder Liner Surface Area
        stroke_height = v_curr_m3 / self.piston_area
        exposed_area = (2.0 * self.piston_area) + (np.pi * self.bore * stroke_height)

        # 2. Compute heat transfer coefficient h_c
        h_c = self.compute_heat_transfer_coefficient(p_pa, t_k, rpm)

        # 3. Newton's Law of Cooling: dQ_wall/dt = h_c * A * (T_gas - T_wall)
        dq_wall_dt = h_c * exposed_area * (t_k - self.t_wall)

        return {
            "dq_wall_dt": round(float(dq_wall_dt), 3),      # Wall loss rate (Watts / J/s)
            "h_c": round(float(h_c), 2),                    # Heat transfer coefficient (W/m^2*K)
            "exposed_area_m2": round(float(exposed_area), 5)# Surface area exposed to combustion
        }


# =====================================================================
# MODULE VERIFICATION & TEST DRIVE
# =====================================================================
if __name__ == "__main__":
    print("[+] Testing 0D Wall Heat Transfer Engine (Woschni Correlation)...")
    heat_transfer_model = WoschniHeatTransferModel()

    rpm_test = 5800.0
    piston_area = np.pi * (0.0795 / 2.0)**2
    clearance_vol = (piston_area * 0.061) / (9.0 - 1.0)

    # Test cases simulating compression, combustion, and exhaust strokes
    test_conditions = [
        {"state": "Compression Onset", "P_Pa": 2.5e5, "T_K": 450.0, "V_m3": clearance_vol * 4.0},
        {"state": "Peak Combustion",   "P_Pa": 65.0e5, "T_K": 2200.0, "V_m3": clearance_vol * 1.2},
        {"state": "Expansion Phase",   "P_Pa": 15.0e5, "T_K": 1300.0, "V_m3": clearance_vol * 3.5},
        {"state": "Exhaust Phase",     "P_Pa": 2.0e5,  "T_K": 850.0,  "V_m3": clearance_vol * 7.0}
    ]

    print(f"\n--- [WOSCHNI HEAT LOSS EVALUATION @ {rpm_test} RPM] ---")
    print(f"{'Engine State':<20} | {'P (bar)':<8} | {'T (K)':<8} | {'h_c (W/m²K)':<14} | {'dQ_wall/dt (kW)':<15}")
    print("-" * 75)

    for condition in test_conditions:
        res = heat_transfer_model.compute_wall_heat_loss(
            p_pa=condition["P_Pa"],
            t_k=condition["T_K"],
            v_curr_m3=condition["V_m3"],
            rpm=rpm_test
        )
        p_bar = condition["P_Pa"] / 1e5
        dq_kw = res["dq_wall_dt"] / 1000.0  # Convert Watts to kW
        print(f"{condition['state']:<20} | {p_bar:<8.1f} | {condition['T_K']:<8.1f} | {res['h_c']:<14.2f} | {dq_kw:<15.2f}")