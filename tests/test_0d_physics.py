import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.physics_engine.thermodynamics import ZeroEngineModel, calculate_residuals


def test_energy_balance():
    """Baseline CHT/EGT at nominal cruise RPM/MAP should sit in a physically realistic band."""
    model = ZeroEngineModel()
    baseline = model.compute_physics_baseline(rpm=5800.0, map_kpa=101.3)

    assert 100.0 < baseline["Physics_CHT"] < 160.0
    assert 750.0 < baseline["Physics_EGT"] < 900.0


def test_baseline_scales_with_rpm():
    """Higher RPM should not produce a lower baseline temperature."""
    model = ZeroEngineModel()
    low_rpm = model.compute_physics_baseline(rpm=5000.0, map_kpa=101.3)
    high_rpm = model.compute_physics_baseline(rpm=6500.0, map_kpa=101.3)

    assert high_rpm["Physics_EGT"] >= low_rpm["Physics_EGT"]


def test_residual_calculation():
    """A telemetry reading well above baseline should show a nonzero residual."""
    baseline = {"Physics_CHT": 115.0, "Physics_EGT": 820.0}
    telemetry = {"CHT": 132.5, "EGT": 875.0}

    deltas = calculate_residuals(telemetry, baseline)
    assert deltas["Delta_CHT"] == 17.5
    assert deltas["Delta_EGT"] == 55.0
