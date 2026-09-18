import time
import math
import random
from typing import List, Dict, Any, Optional
from models import UAVTelemetry

class TelemetryGenerator:
    """
    Generates realistic 6-DOF flight telemetry for a multirotor UAV.
    Computes smooth position, velocity, orientation, battery discharge,
    environmental wind, and realistic sensor noise.
    """
    def __init__(self, mission_id: str = "MISSION_001"):
        self.mission_id = mission_id
        self.step_idx = 0
        self.lat = 11.0168
        self.lon = 76.9558
        self.altitude = 120.0
        self.velocity = 12.5
        self.heading = 90.0
        self.battery_level = 95.0
        self.battery_voltage = 16.4
        self.waypoint = 1
        self.mission_state = "CRUISE"
        self.wind_speed = 6.5
        self.wind_direction = 180.0
        self.gps_status = "healthy"
        self.gps_accuracy = 1.2
        self.comms_status = "connected"
        self.payload_status = "nominal"

    def step(self, dt: float = 1.0) -> UAVTelemetry:
        self.step_idx += 1
        
        # Advance position along heading (simple planar approximation)
        rad = math.radians(self.heading)
        # 1 deg lat ~ 111,000m, 1 deg lon ~ 111,000m * cos(lat)
        d_north = self.velocity * math.cos(rad) * dt
        d_east = self.velocity * math.sin(rad) * dt
        self.lat += d_north / 111000.0
        self.lon += d_east / (111000.0 * math.cos(math.radians(self.lat)))

        # Battery decay
        self.battery_level = max(0.0, self.battery_level - 0.05 * dt)
        # Nominal 4S LiPo voltage: 16.8V at 100%, 14.8V nominal, 13.6V empty
        self.battery_voltage = round(13.6 + (self.battery_level / 100.0) * 3.2, 2)

        # Micro-fluctuations
        accel_z = 9.81 + random.gauss(0, 0.05)
        accel_x = random.gauss(0, 0.03)
        accel_y = random.gauss(0, 0.03)
        gyro_x = random.gauss(0, 0.01)
        gyro_y = random.gauss(0, 0.01)
        gyro_z = random.gauss(0, 0.01)

        return UAVTelemetry(
            timestamp=time.time(),
            mission_id=self.mission_id,
            latitude=round(self.lat, 6),
            longitude=round(self.lon, 6),
            altitude=round(self.altitude + random.gauss(0, 0.2), 2),
            velocity=round(self.velocity + random.gauss(0, 0.1), 2),
            battery_level=round(self.battery_level, 2),
            battery_voltage=self.battery_voltage,
            gps_status=self.gps_status,
            gps_accuracy=round(self.gps_accuracy + random.gauss(0, 0.05), 2),
            imu_acceleration=[round(accel_x, 3), round(accel_y, 3), round(accel_z, 3)],
            imu_gyroscope=[round(gyro_x, 3), round(gyro_y, 3), round(gyro_z, 3)],
            wind_speed=round(self.wind_speed + random.gauss(0, 0.2), 2),
            wind_direction=round(self.wind_direction, 1),
            communication_status=self.comms_status,
            mission_state=self.mission_state,
            waypoint=self.waypoint,
            heading=self.heading,
            payload_status=self.payload_status
        )
