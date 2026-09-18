import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from models import SemanticRule, UAVTelemetry
from config import config, PROJECT_ROOT
from memory.document_store import get_document_store

class SemanticMemory:
    """
    Semantic Memory Layer:
    Maintains generalized operational rules, flight domain heuristics,
    and IF-THEN recovery procedures.
    Can be dynamically evaluated against telemetry and updated by the Learning Agent.
    """
    def __init__(self, file_path: Optional[str] = None):
        if file_path:
            self.file_path = Path(file_path)
        else:
            self.file_path = PROJECT_ROOT / config.memory.semantic_rules_path
        
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.rules: Dict[str, SemanticRule] = {}
        self._load_or_initialize()

    def _load_or_initialize(self) -> None:
        try:
            store = get_document_store()
            if store.available:
                mongo_rules = store.get_rules_sync()
                if mongo_rules:
                    for item in mongo_rules:
                        payload = dict(item)
                        if "rule_id" not in payload and payload.get("_id"):
                            payload["rule_id"] = str(payload["_id"])
                        payload.pop("_id", None)
                        rule = SemanticRule(**{
                            k: v for k, v in payload.items()
                            if k in SemanticRule.model_fields
                        })
                        self.rules[rule.rule_id] = rule
                    return
        except Exception:
            pass
        if self.file_path.exists():
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        rule = SemanticRule(**item)
                        self.rules[rule.rule_id] = rule
                    return
            except Exception:
                pass
        self._seed_default_rules()
        self.save()

    def _seed_default_rules(self) -> None:
        default_rules = [
            SemanticRule(
                rule_id="RULE_GPS_01",
                trigger="GPS_INTERFERENCE",
                condition="gps_accuracy > 3.0 or gps_status == 'degraded'",
                action="SWITCH_TO_VIO_DEAD_RECKONING",
                confidence=0.92,
                description="IF GPS interference occurs THEN switch to Visual-Inertial Odometry / Dead Reckoning"
            ),
            SemanticRule(
                rule_id="RULE_GPS_02",
                trigger="GPS_LOSS",
                condition="gps_status == 'lost' or gps_accuracy > 10.0",
                action="INERTIAL_DEAD_RECKONING_SAFE_RTH",
                confidence=0.95,
                description="IF complete GPS loss occurs THEN hold altitude and dead-reckon Return-to-Home"
            ),
            SemanticRule(
                rule_id="RULE_BATTERY_01",
                trigger="BATTERY_DEGRADATION",
                condition="battery_voltage < 14.2 or voltage_sag_detected",
                action="POWER_CONSERVATIVE_RTH",
                confidence=0.90,
                description="IF rapid battery voltage sag occurs THEN throttle power consumption and RTH"
            ),
            SemanticRule(
                rule_id="RULE_BATTERY_02",
                trigger="LOW_BATTERY",
                condition="battery_level < 15.0",
                action="CONTROLLED_EMERGENCY_LAND",
                confidence=0.98,
                description="IF battery is under critical 15% threshold THEN execute controlled vertical landing"
            ),
            SemanticRule(
                rule_id="RULE_WIND_01",
                trigger="STRONG_WIND",
                condition="wind_speed > 12.0",
                action="REDUCE_VELOCITY_ALTITUDE_HOLD_DESCENT",
                confidence=0.88,
                description="IF wind speeds exceed 12 m/s THEN reduce airspeed and descend to lower shear boundary"
            ),
            SemanticRule(
                rule_id="RULE_COMMS_01",
                trigger="COMMUNICATION_LOSS",
                condition="communication_status == 'lost'",
                action="AUTONOMOUS_FAILSAFE_HOLD_THEN_RTH",
                confidence=0.89,
                description="IF ground telemetry link is severed THEN loiter 10s to reconnect, then execute autonomous RTH"
            ),
            SemanticRule(
                rule_id="RULE_SENSOR_01",
                trigger="SENSOR_ANOMALY",
                condition="imu_divergence > threshold",
                action="SWITCH_SECONDARY_IMU_STABILIZE",
                confidence=0.87,
                description="IF primary IMU reports divergent acceleration/gyro THEN switch to secondary IMU sensor"
            ),
            SemanticRule(
                rule_id="RULE_COMBINED_01",
                trigger="COMBINED_FAILURE",
                condition="multiple_anomalies_active",
                action="CONTROLLED_EMERGENCY_LAND",
                confidence=0.96,
                description="IF multiple simultaneous sensor/power failures occur THEN prioritize immediate safe landing"
            )
        ]
        for r in default_rules:
            self.rules[r.rule_id] = r

    def save(self) -> None:
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump([r.model_dump() for r in self.rules.values()], f, indent=2)
        except Exception:
            pass
        try:
            store = get_document_store()
            if store.available:
                for rule in self.rules.values():
                    doc = rule.model_dump()
                    doc["_id"] = rule.rule_id
                    doc["if_condition"] = rule.trigger
                    doc["then_action"] = rule.action
                    doc["hits"] = rule.success_count + rule.failure_count
                    doc["source"] = "semantic_memory"
                    store.upsert_rule_sync(doc)
        except Exception:
            pass

    def match_rules(self, failure_type: str, telemetry: Optional[UAVTelemetry] = None) -> List[SemanticRule]:
        matched = []
        for rule in self.rules.values():
            if rule.trigger == failure_type:
                matched.append(rule)
        # Sort by confidence descending
        matched.sort(key=lambda r: r.confidence, reverse=True)
        return matched

    def update_or_add_rule(self, rule: SemanticRule) -> None:
        if rule.rule_id in self.rules:
            existing = self.rules[rule.rule_id]
            existing.success_count += rule.success_count
            existing.failure_count += rule.failure_count
            # Adjust confidence with beta update
            total = existing.success_count + existing.failure_count
            existing.confidence = round(existing.success_count / max(1, total), 3)
            self.rules[rule.rule_id] = existing
        else:
            self.rules[rule.rule_id] = rule
        self.save()

    def reinforce_action(self, failure_type: str, action: str, success: bool) -> Optional[SemanticRule]:
        for rule in self.rules.values():
            if rule.trigger == failure_type and rule.action == action:
                if success:
                    rule.success_count += 1
                else:
                    rule.failure_count += 1
                total = rule.success_count + rule.failure_count
                rule.confidence = round(rule.success_count / max(1, total), 3)
                self.save()
                return rule
        # If no rule existed for this action, create a new learned rule
        if success:
            new_id = f"LEARNED_{failure_type}_{len(self.rules)+1}"
            new_rule = SemanticRule(
                rule_id=new_id,
                trigger=failure_type,
                condition="auto_learned_from_mission_success",
                action=action,
                confidence=0.85,
                success_count=1,
                failure_count=0,
                description=f"Auto-learned operational rule from successful recovery of {failure_type}"
            )
            self.rules[new_id] = new_rule
            self.save()
            return new_rule
        return None

    def get_all_rules(self) -> List[SemanticRule]:
        return list(self.rules.values())
