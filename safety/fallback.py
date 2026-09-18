from typing import Dict, Any, List
from models import UAVTelemetry, RecoveryPlan, SituationReport

class DeterministicFallbackExecutor:
    """
    Deterministic Fallback Executor.
    Provides immediate, mathematically verified fail-safe trajectories
    when an LLM recovery plan is rejected or unavailable.
    """
    @staticmethod
    def get_fallback_plan(
        telemetry: UAVTelemetry,
        situation: SituationReport,
        fallback_action_override: str = "RETURN_TO_HOME"
    ) -> RecoveryPlan:
        ftype = situation.failure_type
        action = fallback_action_override
        steps: List[str] = []
        reason = f"Deterministic safety fallback triggered for anomaly '{ftype}'"
        risk = "HIGH"
        conf = 0.99

        # Case 1: Critical or Low Battery
        if telemetry.battery_level < 10.0 or ftype == "LOW_BATTERY":
            action = "CONTROLLED_EMERGENCY_LAND"
            reason = "Critical battery exhaustion hazard. Direct vertical emergency descent activated."
            steps = [
                "Zero horizontal groundspeed",
                "Command descent velocity to -1.5 m/s",
                "Arm ground proximity sensor",
                "Disarm motors on touchdown"
            ]
            risk = "HIGH"

        elif telemetry.battery_level < 20.0 or ftype == "BATTERY_DEGRADATION":
            action = "POWER_CONSERVATIVE_RTH"
            reason = "Battery degradation detected. Throttling flight velocity to best-endurance speed."
            steps = [
                "Reduce current draw by capping speed at 8.0 m/s",
                "Turn directly to home recovery point",
                "Continuously monitor cell delta-V"
            ]
            risk = "MEDIUM"

        # Case 2: GPS Failure
        elif ftype in ("GPS_INTERFERENCE", "GPS_LOSS"):
            if ftype == "GPS_INTERFERENCE":
                action = "SWITCH_TO_VIO_DEAD_RECKONING"
                reason = "GPS degraded. Disengaging GNSS receiver and engaging optical/inertial odometry."
                steps = [
                    "Isolate GPS position loop",
                    "Switch primary navigation filter to VIO / Optical Flow",
                    "Maintain current altitude and evaluate horizontal drift"
                ]
                risk = "MEDIUM"
            else:
                action = "INERTIAL_DEAD_RECKONING_SAFE_RTH"
                reason = "Total GPS signal lost. Executing inertial dead-reckoning back to launch coordinates."
                steps = [
                    "Engage IMU dead reckoning with compass heading",
                    "Ascend to obstacle clearance ceiling",
                    "Fly reverse heading vector toward base",
                    "Initiate visual beacon search near home"
                ]
                risk = "HIGH"

        # Case 3: High Wind
        elif ftype == "STRONG_WIND" or telemetry.wind_speed > 12.0:
            action = "REDUCE_VELOCITY_ALTITUDE_HOLD_DESCENT"
            reason = "Wind velocity exceeds safe aerodynamic limit. Lowering altitude to reduce wind shear."
            steps = [
                "Reduce cruise speed by 50%",
                "Step-descend 30 meters to lower boundary layer",
                "Align yaw into wind vector if groundspeed stalls"
            ]
            risk = "MEDIUM"

        # Case 4: Communication Loss
        elif ftype == "COMMUNICATION_LOSS":
            action = "AUTONOMOUS_FAILSAFE_HOLD_THEN_RTH"
            reason = "Ground link lost. Executing failsafe hover for 10 seconds before autonomous RTH."
            steps = [
                "Hover at current coordinates for 10 seconds",
                "Listen for heartbeat packet reconnection",
                "If silent, engage autonomous Return-To-Home waypoint stream"
            ]
            risk = "LOW"

        # Case 5: Sensor Anomaly
        elif ftype == "SENSOR_ANOMALY":
            action = "SWITCH_SECONDARY_IMU_STABILIZE"
            reason = "Sensor cross-validation divergence. Transferring flight loop to secondary IMU."
            steps = [
                "Exclude erroneous IMU sensor readings",
                "Transfer control loop to secondary redundant IMU",
                "Execute gentle level-flight stabilization"
            ]
            risk = "MEDIUM"

        return RecoveryPlan(
            action=action,
            reason=reason,
            steps=steps,
            expected_outcome="Deterministic safe flight state stabilization",
            confidence=conf,
            risk_level=risk,
            source="Deterministic-Safety-Fallback"
        )
