from typing import List, Dict, Any, Tuple
from config import config
from models import UAVTelemetry, RecoveryPlan, SafetyVerdict

class FlightSafetyConstraints:
    """
    Deterministic Safety Constraints Engine.
    Hard aerodynamic, electrical, and spatial envelope rules that can never be violated
    by an LLM-generated recovery plan.
    """
    def __init__(self):
        self.limits = config.flight_envelope
        self.min_confidence = config.llm.min_confidence_threshold

    def evaluate(self, telemetry: UAVTelemetry, plan: RecoveryPlan) -> SafetyVerdict:
        violated_constraints: List[str] = []
        is_approved = True
        risk_level = "LOW"
        fallback_action = "RETURN_TO_HOME"
        reason = "All deterministic safety criteria satisfied."

        # 1. Planner Confidence Constraint
        if plan.confidence < self.min_confidence:
            violated_constraints.append(f"Planner confidence ({plan.confidence:.2f}) below safe threshold ({self.min_confidence:.2f})")
            is_approved = False
            risk_level = "HIGH"

        # 2. Critical Battery Constraints
        if telemetry.battery_level < self.limits.critical_battery_percent:
            violated_constraints.append(f"Battery critically depleted ({telemetry.battery_level:.1f}% < {self.limits.critical_battery_percent:.1f}%)")
            is_approved = False
            risk_level = "CRITICAL"
            fallback_action = "CONTROLLED_EMERGENCY_LAND"
            
        elif telemetry.battery_level < self.limits.min_battery_percent:
            # Rejects continuing mission under low battery
            if plan.action in ("CONTINUE_MISSION", "WAYPOINT_NAV"):
                violated_constraints.append(f"Cannot continue mission with battery ({telemetry.battery_level:.1f}%) below minimum ({self.limits.min_battery_percent:.1f}%)")
                is_approved = False
                risk_level = "HIGH"
                fallback_action = "POWER_CONSERVATIVE_RTH"

        # 3. GPS Dependency Constraints
        gps_compromised = (telemetry.gps_status in ("degraded", "lost") or telemetry.gps_accuracy > self.limits.max_gps_hdop)
        if gps_compromised:
            # Action cannot rely on standard GPS navigation
            if plan.action in ("CONTINUE_MISSION", "GPS_WAYPOINT_FOLLOW", "PRECISION_GPS_HOVER"):
                violated_constraints.append(f"Proposed action '{plan.action}' requires healthy GPS, but GPS is {telemetry.gps_status} (HDOP: {telemetry.gps_accuracy})")
                is_approved = False
                risk_level = "HIGH"
                fallback_action = "SWITCH_TO_VIO_DEAD_RECKONING"

        # 4. Wind Tolerance Constraints
        if telemetry.wind_speed > self.limits.max_wind_speed_mps:
            if plan.action in ("HIGH_SPEED_TRANSIT", "RAPID_CLIMB"):
                violated_constraints.append(f"Severe wind ({telemetry.wind_speed:.1f} m/s) exceeds maximum tolerance ({self.limits.max_wind_speed_mps:.1f} m/s)")
                is_approved = False
                risk_level = "HIGH"
                fallback_action = "REDUCE_VELOCITY_ALTITUDE_HOLD_DESCENT"

        # 5. Altitude Envelope Constraints
        if telemetry.altitude > self.limits.max_altitude_m:
            violated_constraints.append(f"Altitude ({telemetry.altitude:.1f}m) exceeds ceiling limit ({self.limits.max_altitude_m:.1f}m)")
            is_approved = False
            risk_level = "HIGH"
            fallback_action = "REDUCE_VELOCITY_ALTITUDE_HOLD_DESCENT"
        elif telemetry.altitude < self.limits.min_altitude_m and plan.action == "CONTINUE_MISSION":
            violated_constraints.append(f"Altitude ({telemetry.altitude:.1f}m) below safe floor ({self.limits.min_altitude_m:.1f}m)")
            is_approved = False
            risk_level = "HIGH"
            fallback_action = "CONTROLLED_EMERGENCY_LAND"

        # 6. Geofence Boundary Check
        gf = self.limits.geofence
        if not (gf["min_lat"] <= telemetry.latitude <= gf["max_lat"] and
                gf["min_lon"] <= telemetry.longitude <= gf["max_lon"]):
            violated_constraints.append(f"UAV position ({telemetry.latitude:.4f}, {telemetry.longitude:.4f}) breached operational geofence")
            is_approved = False
            risk_level = "CRITICAL"
            fallback_action = "RETURN_TO_HOME"

        if not is_approved:
            reason = f"Safety Agent rejected plan '{plan.action}': " + "; ".join(violated_constraints)

        return SafetyVerdict(
            approved=is_approved,
            reason=reason,
            risk_level=risk_level,
            fallback_action=fallback_action,
            violated_constraints=violated_constraints
        )
