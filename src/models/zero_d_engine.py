import logging

class ZeroEngineModel:
    def __init__(self):
        # Rotax 914 nominal reference constants
        self.nominal_rpm = 5800.0
        self.nominal_map = 101.3  # kPa

    def compute_baseline(self, rpm: float, map_kpa: float) -> dict:
        """
        Computes expected theoretical baselines for CHT, EGT, and Oil Pressure
        using 0D thermodynamic scaling laws.
        """
        rpm_ratio = rpm / self.nominal_rpm
        map_ratio = map_kpa / self.nominal_map

        # Thermodynamic baseline estimations
        expected_cht = 135.0 * (rpm_ratio ** 0.8) * (map_ratio ** 0.3)
        expected_egt = 825.0 * (rpm_ratio ** 0.5) * (map_ratio ** 0.2)
        expected_oil_p = 4.0 * (rpm_ratio ** 0.9)

        return {
            "physics_cht": round(expected_cht, 2),
            "physics_egt": round(expected_egt, 2),
            "physics_oil": round(expected_oil_p, 2)
        }