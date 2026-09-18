from typing import Dict, Any, Callable
from models import UAVTelemetry

class FailureScenarioInjector:
    """
    Simulated Failure Scenarios for AeroCortex UAV Testing.
    Applies realistic physical and sensor deviations to UAV telemetry.
    """

    @staticmethod
    def apply_scenario(telemetry: UAVTelemetry, scenario_name: str) -> UAVTelemetry:
        t = telemetry.model_copy(deep=True)
        scenario = scenario_name.upper()

        if scenario == "GPS_INTERFERENCE":
            t.gps_status = "degraded"
            t.gps_accuracy = 6.4
            # Slight velocity drift caused by degraded Kalman filter
            t.velocity = max(2.0, t.velocity - 2.5)

        elif scenario == "GPS_LOSS":
            t.gps_status = "lost"
            t.gps_accuracy = 99.9
            t.velocity = max(1.0, t.velocity - 4.0)

        elif scenario == "BATTERY_DEGRADATION":
            # Sudden voltage drop below nominal under load
            t.battery_voltage = 13.75
            t.battery_level = max(5.0, t.battery_level - 15.0)

        elif scenario == "LOW_BATTERY":
            t.battery_level = 13.5
            t.battery_voltage = 13.6

        elif scenario == "COMMUNICATION_LOSS":
            t.communication_status = "lost"

        elif scenario == "STRONG_WIND":
            t.wind_speed = 16.8
            t.wind_direction = 225.0
            # Aerodynamic buffeting causes velocity and heading deviation
            t.velocity = max(4.0, t.velocity - 3.8)
            t.heading = (t.heading + 15.0) % 360.0
            t.imu_acceleration = [2.5, -3.1, 11.8]

        elif scenario == "SENSOR_ANOMALY":
            # Erratic sensor divergence
            t.imu_acceleration = [18.5, -24.2, 32.1]
            t.imu_gyroscope = [4.2, -6.1, 5.8]

        elif scenario == "COMBINED_FAILURE":
            # Severe simultaneous GPS degradation and high wind
            t.gps_status = "degraded"
            t.gps_accuracy = 7.8
            t.wind_speed = 17.5
            t.battery_voltage = 13.9
            t.communication_status = "degraded"
            t.imu_acceleration = [4.2, -3.8, 14.5]

        return t

    @staticmethod
    def list_available_scenarios() -> list[str]:
        return [
            "NORMAL",
            "GPS_INTERFERENCE",
            "GPS_LOSS",
            "BATTERY_DEGRADATION",
            "LOW_BATTERY",
            "COMMUNICATION_LOSS",
            "STRONG_WIND",
            "SENSOR_ANOMALY",
            "COMBINED_FAILURE"
        ]
