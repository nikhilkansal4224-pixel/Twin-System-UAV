import json
import time
import sys
import os

# Add parent directory to path for cross-module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from kafka import KafkaConsumer  # Install via: pip install kafka-python
from src.physics_engine.thermodynamics import ZeroEngineModel
from src.physics_engine.residual_calculator import ResidualCalculator
from src.ai_pipeline.pinn_model import PhysicsInformedNN
from src.ai_pipeline.lstm_rul import RULPredictorEngine

# =====================================================================
# 1. KAFKA BROKER & TOPIC CONFIGURATION
# =====================================================================
KAFKA_BROKER = "localhost:9092"
INPUT_TOPIC = "uav-engine-telemetry"
CONSUMER_GROUP = "digital-twin-orchestrator"

def json_deserializer(payload_bytes):
    """Deserializes raw JSON byte payload into Python dictionary."""
    return json.loads(payload_bytes.decode("utf-8"))

def initialize_kafka_consumer():
    """Attempts connection to local Kafka broker consumer queue."""
    try:
        consumer = KafkaConsumer(
            INPUT_TOPIC,
            bootstrap_servers=[KAFKA_BROKER],
            group_id=CONSUMER_GROUP,
            value_deserializer=json_deserializer,
            auto_offset_reset="latest"
        )
        print(f"[+] Successfully Connected to Local Kafka Queue: '{INPUT_TOPIC}'")
        return consumer
    except Exception as e:
        print(f"[!] Warning: Unable to connect to Kafka Broker ({e}). Pipeline standing by.")
        return None

# =====================================================================
# 2. DIGITAL TWIN ORCHESTRATION PIPELINE
# =====================================================================
class DigitalTwinOrchestrator:
    def __init__(self):
        print("[+] Initializing Edge Digital Twin Subsystems...")
        self.physics_twin = ZeroEngineModel()
        self.residual_calc = ResidualCalculator()
        self.pinn_model = PhysicsInformedNN()
        self.rul_engine = RULPredictorEngine(nominal_max_life_hours=1200.0, sequence_length=10)

    def process_telemetry_packet(self, raw_packet: dict) -> dict:
        """
        Executes end-to-end Digital Twin processing pipeline on a single frame packet.
        """
        telemetry_data = raw_packet.get("data", {})
        rpm = float(telemetry_data.get("RPM", 5800.0))
        map_kpa = float(telemetry_data.get("MAP", 101.3))

        # STEP 1: Compute 0D Thermodynamic Physical Baseline
        physics_baseline = self.physics_twin.compute_physics_baseline(rpm=rpm, map_kpa=map_kpa)

        # STEP 2: Compute Absolute Residual Deltas ΔY = |Actual - Baseline|
        residual_analysis = self.residual_calc.compute_residuals(telemetry_data, physics_baseline)
        residuals = residual_analysis["residuals"]

        # STEP 3: Pass Residual Deltas to LSTM RUL Predictor Buffer
        self.rul_engine.add_telemetry_step(residuals)
        rul_assessment = self.rul_engine.predict_rul()

        # STEP 4: Format Unified Digital Twin Output Packet
        digital_twin_state = {
            "timestamp": raw_packet.get("timestamp", time.time()),
            "can_id": raw_packet.get("can_id", "0x100"),
            "telemetry_actual": telemetry_data,
            "physics_baseline": physics_baseline,
            "residual_deltas": residuals,
            "anomaly_flagged": residual_analysis["anomaly_detected"],
            "rul_hours": rul_assessment["predicted_rul_hours"],
            "health_index_pct": rul_assessment["health_index_pct"],
            "maintenance_urgency": rul_assessment["maintenance_urgency"]
        }

        return digital_twin_state

# =====================================================================
# 3. CONSUMER LOOP & EXECUTION ENTRY-POINT
# =====================================================================
def start_orchestration_service():
    orchestrator = DigitalTwinOrchestrator()
    consumer = initialize_kafka_consumer()

    print("\n[+] Edge Pipeline Operational. Ingesting Real-Time Telemetry Stream...\n" + "=" * 75)

    if consumer:
        try:
            for message in consumer:
                packet = message.value
                twin_state = orchestrator.process_telemetry_packet(packet)
                
                print(f"| TIMESTAMP: {twin_state['timestamp']:.2f} | CAN ID: {twin_state['can_id']}")
                print(f"  --> Telemetry Actual : {twin_state['telemetry_actual']}")
                print(f"  --> 0D Physics Base  : {twin_state['physics_baseline']}")
                print(f"  --> Deltas (ΔY)      : {twin_state['residual_deltas']}")
                print(f"  --> Health / RUL     : {twin_state['health_index_pct']}% ({twin_state['rul_hours']} hrs remaining)")
                print(f"  --> Status           : {twin_state['maintenance_urgency']}")
                print("-" * 75)
        except KeyboardInterrupt:
            print("\n[-] Pipeline Consumer Stopped.")
            consumer.close()
    else:
        # Fallback Test Mode using Synthetic CAN Packet
        print("[!] Executing Fallback Direct Pipeline Verification Test...")
        test_packet = {
            "timestamp": time.time(),
            "can_id": "0x200",
            "data": {"RPM": 5800.0, "MAP": 101.3, "CHT": 132.5, "EGT": 875.0}
        }
        twin_state = orchestrator.process_telemetry_packet(test_packet)
        print(f"\n[+] Digital Twin Processed Output State Packet:\n{json.dumps(twin_state, indent=2)}")

if __name__ == "__main__":
    start_orchestration_service()