import time
from typing import Dict, Any, Tuple
from config import config
from models import UAVTelemetry, SituationReport, FailureType, SeverityLevel

class SituationAgent:
    """
    Situation Agent:
    Deterministic real-time anomaly detector and failure classifier.
    Continuously evaluates sensor streams against physical threshold envelopes.
    Does NOT use an LLM for anomaly detection to ensure safety and determinism.
    """
    def __init__(self):
        self.limits = config.flight_envelope
        self.last_telemetry: UAVTelemetry = None
        self.last_battery_time: float = 0.0

    def assess_situation(self, telemetry: UAVTelemetry) -> SituationReport:
        detected_anomalies = []
        failure_type = "NONE"
        severity = "NOMINAL"
        confidence = 1.0
        description = "Nominal operating parameters."

        # 1. GPS Checks
        gps_interference = False
        gps_loss = False
        if telemetry.gps_status == "lost" or telemetry.gps_accuracy > 15.0:
            gps_loss = True
            detected_anomalies.append("GPS_LOSS")
        elif telemetry.gps_status == "degraded" or telemetry.gps_accuracy > self.limits.max_gps_hdop:
            gps_interference = True
            detected_anomalies.append("GPS_INTERFERENCE")

        # 2. Battery Checks
        low_battery = False
        battery_degradation = False
        if telemetry.battery_level < self.limits.min_battery_percent:
            low_battery = True
            detected_anomalies.append("LOW_BATTERY")
        elif telemetry.battery_voltage < 14.2 and telemetry.battery_level > 25.0:
            # Voltage sag disproportionate to remaining capacity
            battery_degradation = True
            detected_anomalies.append("BATTERY_DEGRADATION")

        # 3. Communication Checks
        comms_loss = False
        if telemetry.communication_status in ("lost", "disconnected"):
            comms_loss = True
            detected_anomalies.append("COMMUNICATION_LOSS")

        # 4. Wind Speed Checks
        strong_wind = False
        if telemetry.wind_speed > self.limits.max_wind_speed_mps:
            strong_wind = True
            detected_anomalies.append("STRONG_WIND")

        # 5. Sensor IMU Checks
        sensor_anomaly = False
        ax, ay, az = telemetry.imu_acceleration
        gx, gy, gz = telemetry.imu_gyroscope
        # Abnormal acceleration spike (excluding 1g gravity) or erratic angular rate
        total_accel = (ax**2 + ay**2 + az**2)**0.5
        total_gyro = (gx**2 + gy**2 + gz**2)**0.5
        if total_accel > 30.0 or total_gyro > 5.0:
            sensor_anomaly = True
            detected_anomalies.append("SENSOR_ANOMALY")

        # Classification and Severity Assignment
        num_anomalies = len(detected_anomalies)
        anomaly_detected = num_anomalies > 0

        if num_anomalies > 1:
            failure_type = "COMBINED_FAILURE"
            severity = "CRITICAL"
            confidence = 0.96
            description = f"Multiple simultaneous critical anomalies detected: {', '.join(detected_anomalies)}"
            
        elif num_anomalies == 1:
            failure_type = detected_anomalies[0]
            
            if failure_type == "LOW_BATTERY":
                severity = "CRITICAL" if telemetry.battery_level < self.limits.critical_battery_percent else "HIGH"
                confidence = 0.98
                description = f"Battery level at {telemetry.battery_level:.1f}% below {self.limits.min_battery_percent}% threshold"
                
            elif failure_type == "BATTERY_DEGRADATION":
                severity = "HIGH"
                confidence = 0.92
                description = f"Abnormal voltage sag ({telemetry.battery_voltage:.2f}V) indicates cell degradation"
                
            elif failure_type == "GPS_LOSS":
                severity = "HIGH"
                confidence = 0.95
                description = f"Complete GPS satellite lock lost (Accuracy: {telemetry.gps_accuracy:.1f}m)"
                
            elif failure_type == "GPS_INTERFERENCE":
                severity = "MEDIUM" if telemetry.gps_accuracy < 5.0 else "HIGH"
                confidence = 0.91
                description = f"GPS interference detected (Status: {telemetry.gps_status}, Accuracy: {telemetry.gps_accuracy:.1f}m)"
                
            elif failure_type == "COMMUNICATION_LOSS":
                severity = "HIGH"
                confidence = 0.94
                description = "Ground station telemetry communication link severed"
                
            elif failure_type == "STRONG_WIND":
                severity = "HIGH" if telemetry.wind_speed > 16.0 else "MEDIUM"
                confidence = 0.90
                description = f"Wind velocity ({telemetry.wind_speed:.1f} m/s) exceeds flight envelope ({self.limits.max_wind_speed_mps} m/s)"
                
            elif failure_type == "SENSOR_ANOMALY":
                severity = "HIGH"
                confidence = 0.88
                description = f"IMU dynamic sensor anomaly detected (Accel: {total_accel:.1f} m/s2, Gyro: {total_gyro:.1f} rad/s)"

        self.last_telemetry = telemetry

        return SituationReport(
            anomaly_detected=anomaly_detected,
            failure_type=failure_type,
            severity=severity,
            confidence=confidence,
            current_state={
                "altitude": telemetry.altitude,
                "velocity": telemetry.velocity,
                "battery_level": telemetry.battery_level,
                "battery_voltage": telemetry.battery_voltage,
                "gps_accuracy": telemetry.gps_accuracy,
                "wind_speed": telemetry.wind_speed,
                "active_anomalies": detected_anomalies
            },
            mission_phase=telemetry.mission_state,
            timestamp=time.time(),
            description=description
        )
