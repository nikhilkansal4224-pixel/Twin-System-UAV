import time
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

# =====================================================================
# 1. INFLUXDB DATABASE CONFIGURATION
# =====================================================================
INFLUX_URL = "http://localhost:8086"
INFLUX_TOKEN = "apiv3_NikhilGod4224"  # Set in docker-compose or environment
INFLUX_ORG = "uav_defense_org"
INFLUX_BUCKET = "engine_telemetry_bucket"

class InfluxDBWriter:
    def __init__(self, url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG, bucket=INFLUX_BUCKET):
        """
        Initializes the InfluxDB time-series database client.
        """
        self.bucket = bucket
        self.org = org
        self.client = None
        self.write_api = None

        try:
            self.client = InfluxDBClient(url=url, token=token, org=org)
            self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
            print(f"[+] Connected to InfluxDB Time-Series DB at {url} (Bucket: '{bucket}')")
        except Exception as e:
            print(f"[!] Warning: Failed to connect to InfluxDB ({e}). Writer standing by in dry-run mode.")

    def format_point(self, twin_state: dict) -> Point:
        """
        Formats a Digital Twin state packet into an InfluxDB Data Point.
        """
        timestamp = int(twin_state.get("timestamp", time.time()) * 1e9)  # Nanosecond timestamp
        can_id = str(twin_state.get("can_id", "0x100"))
        
        telemetry = twin_state.get("telemetry_actual", {})
        baseline = twin_state.get("physics_baseline", {})
        residuals = twin_state.get("residual_deltas", {})

        # Construct Time-Series Point
        point = Point("uav_aero_engine_metrics") \
            .tag("can_id", can_id) \
            .tag("maintenance_status", twin_state.get("maintenance_urgency", "NOMINAL")) \
            .time(timestamp, WritePrecision.NS)

        # 1. Add Live Telemetry Fields
        for key, val in telemetry.items():
            point.field(f"actual_{key.lower()}", float(val))

        # 2. Add 0D Physics Baseline Fields
        for key, val in baseline.items():
            point.field(f"physics_{key.lower()}", float(val))

        # 3. Add Residual Delta Fields (ΔY)
        for key, val in residuals.items():
            point.field(f"residual_{key.lower()}", float(val))

        # 4. Add RUL & Health Index Fields
        point.field("health_index_pct", float(twin_state.get("health_index_pct", 100.0)))
        point.field("rul_hours", float(twin_state.get("rul_hours", 1200.0)))
        point.field("anomaly_flag", 1.0 if twin_state.get("anomaly_flagged", False) else 0.0)

        return point

    def write_twin_state(self, twin_state: dict) -> bool:
        """
        Writes processed Digital Twin state metrics to InfluxDB.
        """
        point = self.format_point(twin_state)

        if self.write_api:
            try:
                self.write_api.write(bucket=self.bucket, org=self.org, record=point)
                return True
            except Exception as e:
                print(f"[!] Error writing record to InfluxDB: {e}")
                return False
        else:
            # Standalone print mode when database is not running
            print(f"| DRY-RUN INFLUX POINT | Line Protocol: {point.to_line_protocol()}")
            return True

    def close(self):
        """Closes the active database connection."""
        if self.client:
            self.client.close()
            print("[-] InfluxDB Writer Client Disconnected.")


# =====================================================================
# MODULE VERIFICATION & TEST DRIVE
# =====================================================================
if __name__ == "__main__":
    print("[+] Testing InfluxDB Persistence Engine...")
    writer = InfluxDBWriter()

    # Sample Digital Twin State Packet
    sample_twin_state = {
        "timestamp": time.time(),
        "can_id": "0x200",
        "telemetry_actual": {"RPM": 5800.0, "MAP": 101.3, "CHT": 132.5, "EGT": 875.0},
        "physics_baseline": {"Physics_CHT": 115.0, "Physics_EGT": 820.0},
        "residual_deltas": {"Delta_CHT": 17.5, "Delta_EGT": 55.0},
        "anomaly_flagged": True,
        "rul_hours": 845.2,
        "health_index_pct": 70.43,
        "maintenance_urgency": "WARNING"
    }

    # Execute Persistence Test
    success = writer.write_twin_state(sample_twin_state)
    print(f"\n[+] Write Status: {'SUCCESS' if success else 'FAILED'}")

    writer.close()