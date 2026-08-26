import numpy as np
from scipy.integrate import solve_ivp

# =====================================================================
# 1. ENGINE GEOMETRY & PHYSICAL CONSTANTS (e.g., Rotax 914 Aero Engine)
# =====================================================================
BORE = 0.0795         # Cylinder Bore diameter (m)
STROKE = 0.061        # Piston Stroke length (m)
CRANK_RADIUS = STROKE / 2.0  # Crankshaft radius r (m)
CON_ROD = 0.112       # Connecting Rod length l (m)
CR = 9.0              # Compression Ratio
PISTON_AREA = np.pi * (BORE / 2.0)**2
DISPLACEMENT = PISTON_AREA * STROKE
CLEARANCE_VOL = DISPLACEMENT / (CR - 1.0)
LHV_FUEL = 44.0e6     # Lower Heating Value of aviation fuel (J/kg)
R_AIR = 287.05        # Specific Gas Constant for air (J/kg*K)
Cv_AIR = 718.0        # Specific Heat at constant volume (J/kg*K)
V_DISPLACEMENT = (np.pi / 4.0) * (BORE ** 2) * STROKE
V_CLEARANCE = V_DISPLACEMENT / (CR - 1.0)  # Clearance Volume (V_c)

# =====================================================================
# 2. CRANK-SLIDER KINEMATICS & WIEBE HEAT RELEASE MODELS
# =====================================================================
def get_cylinder_volume(theta):
    """Calculates instantaneous cylinder volume V(theta) based on crank angle."""
    lambda_param = CRANK_RADIUS / CON_ROD
    # Kinematic equation for piston displacement
    term = 1.0 + np.cos(theta) + (1.0 / lambda_param) * (1.0 - np.sqrt(1.0 - lambda_param**2 * np.sin(theta)**2))
    s_theta = CRANK_RADIUS * (1.0 - term)
    v_theta = CLEARANCE_VOL + PISTON_AREA * s_theta
    return v_theta

def wiebe_heat_release(theta, theta_start=-10.0*np.pi/180.0, delta_theta=60.0*np.pi/180.0, a=5.0, m=2.0):
    """Calculates instantaneous heat release rate dQ_in/dt using the Wiebe function."""
    if theta < theta_start or theta > (theta_start + delta_theta):
        return 0.0
    
    y = (theta - theta_start) / delta_theta
    # Mass fraction burned derivative dx_b/dtheta
    dxb_dtheta = a * (m + 1.0) / delta_theta * (y**m) * np.exp(-a * (y**(m + 1.0)))
    return dxb_dtheta

def woschni_heat_transfer(p, t, v_theta, rpm):
    """Calculates wall heat loss rate dQ_wall/dt using Woschni correlation."""
    mean_piston_speed = 2.0 * STROKE * (rpm / 60.0)
    # Woschni heat transfer coefficient h_c
    p_abs = np.maximum(p, 1e2)  # Prevent negative pressure during ODE step integration
    t_abs = np.maximum(t, 200.0)

    h_c = 3.2 * (BORE**-0.2) * ((p_abs / 1e5)**0.8) * (t_abs**-0.53) * (mean_piston_speed**0.8)
    
    # Instantaneous wall surface area
    area_wall = 2.0 * PISTON_AREA + np.pi * BORE * (v_theta / PISTON_AREA)
    t_wall = 420.0  # Nominal cylinder wall temperature (K)
    
    dq_wall = h_c * area_wall * (t - t_wall)
    return dq_wall

# =====================================================================
# 3. 0D THERMODYNAMIC ODE GOVERNING EQUATION (1st Law)
# =====================================================================
def first_law_ode(t, y_state, rpm, fuel_mass_per_cycle):
    """
    Solves dT/dt using First Law of Thermodynamics:
    dU/dt = dQ_in/dt - dQ_wall/dt - P * dV/dt
    """
    temp = y_state[0]
    omega = (2.0 * np.pi * rpm) / 60.0  # Angular velocity (rad/s)
    theta = omega * t                    # Crank angle (rad)
    
    v_curr = get_cylinder_volume(theta)
    
    # Calculate mass of trapped gas via Ideal Gas Law (PV = mRT)
    p_intake = 101325.0  # Nominal ambient intake pressure (Pa)
    m_gas = (p_intake * v_curr) / (R_AIR * 300.0)
    
    # Calculate current cylinder pressure
    v_curr = np.maximum(v_curr, V_CLEARANCE)  # Prevent zero or negative volume
    p_curr = (m_gas * R_AIR * temp) / v_curr
    
    # 1. Heat Input Rate (dQ_in/dt)
    dq_in_dtheta = wiebe_heat_release(theta) * (fuel_mass_per_cycle * LHV_FUEL)
    dq_in_dt = dq_in_dtheta * omega
    
    # 2. Wall Loss Rate (dQ_wall/dt)
    dq_wall_dt = woschni_heat_transfer(p_curr, temp, v_curr, rpm)
    
    # 3. Piston Work Rate (dW/dt = P * dV/dt)
    # Analytical derivative of volume dV/dt
    dv_dt = PISTON_AREA * CRANK_RADIUS * omega * np.sin(theta)
    work_rate = p_curr * dv_dt
    
    # First Law: m * Cv * dT/dt = dQ_in/dt - dQ_wall/dt - P * dV/dt
    dtemp_dt = (dq_in_dt - dq_wall_dt - work_rate) / (m_gas * Cv_AIR)
    
    return [dtemp_dt]

# =====================================================================
# 4. 0D BASELINE ENGINE SOLVER & RESIDUAL COMPUTATION
# =====================================================================
class ZeroEngineModel:
    def __init__(self):
        self.t_span = (0.0, 0.04)  # ~2 engine revolutions at 3000 RPM

    def compute_physics_baseline(self, rpm, map_kpa):
        """Solves 0D ODEs to calculate theoretical healthy EGT and CHT values."""
        t_init = [350.0]  # Initial compression temperature (K)
        fuel_mass = 0.00003  # ~30mg per stroke
        
        # Solve ODE using SciPy solve_ivp
        sol = solve_ivp(
            first_law_ode, 
            self.t_span, 
            t_init, 
            args=(rpm, fuel_mass),
            method='RK45', 
            max_step=0.0001
        )
        
        # Calculate theoretical equilibrium outputs
        computed_cht_c = np.mean(sol.y[0]) - 273.15  # Convert K to °C
        computed_egt_c = np.max(sol.y[0]) * 0.75 - 273.15
        
        return {
            "Physics_CHT": round(computed_cht_c, 2),
            "Physics_EGT": round(computed_egt_c, 2)
        }

def calculate_residuals(telemetry_data, physics_baseline):
    """
    Computes absolute mathematical residuals ΔY = |Actual - Physics Baseline|
    """
    residuals = {}
    if "CHT" in telemetry_data and "Physics_CHT" in physics_baseline:
        residuals["Delta_CHT"] = round(abs(telemetry_data["CHT"] - physics_baseline["Physics_CHT"]), 2)
        
    if "EGT" in telemetry_data and "Physics_EGT" in physics_baseline:
        residuals["Delta_EGT"] = round(abs(telemetry_data["EGT"] - physics_baseline["Physics_EGT"]), 2)
        
    return residuals

# =====================================================================
# 5. EXECUTION & TEST DRIVE
# =====================================================================
if __name__ == "__main__":
    print("[+] Initializing 0D Thermodynamic Reference Model...")
    physics_twin = ZeroEngineModel()
    
    # Simulated CAN-bus live telemetry
    live_can_telemetry = {"RPM": 5800.0, "MAP": 101.3, "CHT": 135.5, "EGT": 880.0}
    
    print(f"[+] Live CAN Telemetry Ingested: {live_can_telemetry}")
    
    # Calculate theoretical healthy baseline
    baseline = physics_twin.compute_physics_baseline(
        rpm=live_can_telemetry["RPM"], 
        map_kpa=live_can_telemetry["MAP"]
    )
    print(f"[+] 0D Physics Model Baseline: {baseline}")
    
    # Calculate Residual Deltas ΔY
    deltas = calculate_residuals(live_can_telemetry, baseline)
    print(f"[+] Computed Residual Deltas (ΔY): {deltas}")
    
    # Micro-Anomaly Flag Evaluation
    if deltas.get("Delta_CHT", 0) > 15.0 or deltas.get("Delta_EGT", 0) > 40.0:
        print("\n[!] WARNING: Physical Residual Delta Threshold Exceeded!")
        print("[!] Micro-Anomaly Flagged -> Routing Residuals to PyTorch PINN Pipeline...")