import numpy as np


class CrankSliderKinematics:
    def __init__(self, bore: float = 0.0795, stroke: float = 0.061, con_rod: float = 0.112, compression_ratio: float = 9.0):
        """
        Initializes piston kinematics parameters.
        Default values map to Rotax 914 aero-piston engine specifications.
        """
        self.bore = bore
        self.stroke = stroke
        self.con_rod = con_rod
        self.cr = compression_ratio
        
        self.r = stroke / 2.0                             # Crank radius (m)
        self.lambda_param = self.r / con_rod              # Geometry ratio (r/l)
        self.piston_area = np.pi * (bore / 2.0) ** 2      # Cross-sectional face area (m^2)
        
        self.v_swept = self.piston_area * stroke          # Displacement volume V_d (m^3)
        self.v_clearance = self.v_swept / (self.cr - 1.0) # Clearance volume V_c (m^3)

    def get_piston_displacement(self, theta: float) -> float:
        """Calculates instantaneous piston position s(theta) from Top Dead Center (m)."""
        sin_t = np.sin(theta)
        cos_t = np.cos(theta)
        
        # Exact crank-slider displacement formula
        radical = np.sqrt(max(0.0, 1.0 - (self.lambda_param ** 2) * (sin_t ** 2)))
        s_theta = self.r * ((1.0 - cos_t) + (1.0 / self.lambda_param) * (1.0 - radical))
        return float(s_theta)

    def get_volume(self, theta: float) -> float:
        """Calculates instantaneous cylinder volume V(theta) in m^3."""
        s_theta = self.get_piston_displacement(theta)
        return float(self.v_clearance + (self.piston_area * s_theta))

    def get_volume_derivative_dtheta(self, theta: float) -> float:
        """Calculates exact derivative dV/dtheta (m^3/rad)."""
        sin_t = np.sin(theta)
        cos_t = np.cos(theta)
        
        denom = np.sqrt(max(1e-6, 1.0 - (self.lambda_param ** 2) * (sin_t ** 2)))
        dv_dtheta = self.piston_area * self.r * sin_t * (1.0 + (self.lambda_param * cos_t) / denom)
        return float(dv_dtheta)

    def compute_kinematic_state(self, theta: float, rpm: float) -> dict:
        """
        Computes cylinder geometric parameters and volume expansion rates.
        
        :param theta: Current crankshaft angle (radians)
        :param rpm: Engine rotational speed
        :return: Dictionary containing volume, surface area, and dV/dt expansion work rate
        """
        omega = (2.0 * np.pi * rpm) / 60.0  # Angular velocity (rad/s)
        
        v_curr = self.get_volume(theta)
        dv_dtheta = self.get_volume_derivative_dtheta(theta)
        dv_dt = dv_dtheta * omega           # Volume rate of change dV/dt (m^3/s)
        
        # Calculate cylinder wall area exposed to combustion gases
        stroke_height = v_curr / self.piston_area
        exposed_surface_area = (2.0 * self.piston_area) + (np.pi * self.bore * stroke_height)

        return {
            "v_curr_m3": round(v_curr, 8),
            "v_curr_cc": round(v_curr * 1e6, 2),
            "dv_dt": round(dv_dt, 6),
            "exposed_area_m2": round(exposed_surface_area, 6),
            "piston_speed_mps": round(dv_dt / self.piston_area, 3)
        }

    def compute_vectorized_kinematics(
        self, 
        theta_array: np.ndarray, 
        rpm_array: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fast vectorized computation of V(theta), dV/dt, and exposed wall area
        across an entire telemetry DataFrame column.
        
        :return: Tuple of (v_curr_m3_array, dv_dt_array, exposed_area_array)
        """
        sin_t = np.sin(theta_array)
        cos_t = np.cos(theta_array)
        omega_array = (2.0 * np.pi * rpm_array) / 60.0

        radical = np.sqrt(np.maximum(0.0, 1.0 - (self.lambda_param ** 2) * (sin_t ** 2)))
        s_theta = self.r * ((1.0 - cos_t) + (1.0 / self.lambda_param) * (1.0 - radical))
        v_curr_array = self.v_clearance + (self.piston_area * s_theta)

        denom = np.sqrt(np.maximum(1e-6, 1.0 - (self.lambda_param ** 2) * (sin_t ** 2)))
        dv_dtheta = self.piston_area * self.r * sin_t * (1.0 + (self.lambda_param * cos_t) / denom)
        dv_dt_array = dv_dtheta * omega_array

        stroke_height = v_curr_array / self.piston_area
        exposed_area_array = (2.0 * self.piston_area) + (np.pi * self.bore * stroke_height)

        return v_curr_array, dv_dt_array, exposed_area_array