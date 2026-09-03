import json
import time
import sys
import os
import logging
import torch

# Ensure project root is in Python search path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from kafka import KafkaConsumer
from src.physics_engine.thermodynamics import ZeroEngineModel
from src.physics_engine.residual_calculator import ResidualCalculator
from src.models.pinn_model import PhysicsInformedNN
from src.ai_pipeline.lstm_rul import RULPredictorEngine
from src.db.postgres_writer import PostgresWriter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# =====================================================================
# 1. KAFKA BROKER & TOPIC CONFIGURATION
# =====================================================================
# Force IPv4 127.0.0.1 to avoid IPv6 (::1) socket disconnection issues on macOS
KAFKA_BROKER = os.getenv("KAFKA_BOOTSTRAP_SERVERS", os.getenv("KAFKA_BROKER", "127.0.0.1:9092"))
INPUT_TOPIC = os.getenv("KAFKA_TOPIC", "uav-engine-telemetry")
CONSUMER_GROUP = "digital-twin-orchestrator"
MODEL_WEIGHTS_PATH = os.getenv("PINN_MODEL_PATH", "pinn_model_latest.pth")


def initialize_kafka_consumer():
    """Attempts connection to local Kafka broker consumer queue."""
    try:
        consumer = KafkaConsumer(
            INPUT_TOPIC,
            bootstrap_servers=[KAFKA_BROKER],
            group_id=CONSUMER_GROUP,
            value_deserializer=lambda x: json.loads(x.decode("utf-8")),
            auto_offset_reset="latest",
            enable_auto_commit=True
        )
        logging.info(f"[+] Successfully Connected to Kafka Queue: '{INPUT_TOPIC}' on {KAFKA_BROKER}")
        return consumer
    except Exception as e:
        logging.warning(f"[!] Unable to connect to Kafka Broker on {KAFKA_BROKER} ({e}). Pipeline running in fallback mode.")
        return None


# =====================================================================
# 2. DIGITAL TWIN ORCHESTRATION PIPELINE
# =====================================================================
class DigitalTwinOrchestrator:
    def __init__(self, db_writer: PostgresWriter = None):
        logging.info("[+] Initializing Edge Digital Twin Subsystems...")
        
        # Initialize Self-Adjusting 0D Physics Twin and load saved calibration from Postgres
        self.physics_twin = ZeroEngineModel()
        self.physics_twin.load_calibration()

        self.residual_calc = ResidualCalculator()
        
        # UPDATED: PINN dimension updated to input_dim=6 (actual_cht, actual_egt, actual_oil, physics_cht, physics_egt, altitude_m)
        self.pinn_model = PhysicsInformedNN(input_dim=6, hidden_dim=32, output_dim=4)
        self.load_pinn_weights()

        self.rul_engine = RULPredictorEngine(nominal_max_life_hours=1200.0, sequence_length=10)
        self.db_writer = db_writer or PostgresWriter()
        
        self.frame_count = 0  # Counter for periodic DB calibration & model hot-reload checks

    def load_pinn_weights(self):
        """Loads or hot-reloads trained PINN weights if available on disk."""
        if os.path.exists(MODEL_WEIGHTS_PATH):
            try:
                self.pinn_model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=torch.device('cpu')))
                self.pinn_model.eval()
                logging.info(f"[+] Hot-reloaded PINN model weights from '{MODEL_WEIGHTS_PATH}'")
            except Exception as e:
                logging.warning(f"[!] Failed to load model weights from '{MODEL_WEIGHTS_PATH}': {e}")

    def process_telemetry_packet(self, raw_packet: dict) -> dict:
        """
        Executes end-to-end Digital Twin processing pipeline on a single frame packet.
        """
        telemetry_data = raw_packet.get("data", raw_packet.get("telemetry", {}))
        
        # Extract telemetry features with safe defaults
        rpm = float(telemetry_data.get("RPM", telemetry_data.get("rpm", 5800.0)))
        map_kpa = float(telemetry_data.get("MAP", telemetry_data.get("map_kpa", 101.3)))
        ambient_temp = float(telemetry_data.get("Ambient_Temp", telemetry_data.get("ambient_temp_c", 15.0)))
        altitude = float(telemetry_data.get("Altitude", telemetry_data.get("altitude_m", 0.0)))
        actual_cht = float(telemetry_data.get("CHT", telemetry_data.get("actual_cht", 135.0)))
        actual_egt = float(telemetry_data.get("EGT", telemetry_data.get("actual_egt", 820.0)))
        actual_oil = float(telemetry_data.get("Oil_Pressure", telemetry_data.get("actual_oil_pressure", 4.0)))

        # STEP 1: Compute Baseline using current self-adjusted bias offsets
        physics_baseline = self.physics_twin.compute_physics_baseline(
            rpm=rpm,
            map_kpa=map_kpa,
            ambient_temp_c=ambient_temp,
            altitude_m=altitude
        )
        physics_cht = float(physics_baseline.get("physics_cht", 135.0))
        physics_egt = float(physics_baseline.get("physics_egt", 820.0))

        # STEP 2: CONTINUOUS AUTO-ADJUSTMENT (Learns real sensor baseline dynamically)
        self.physics_twin.auto_adjust(actual_cht, actual_egt, physics_baseline)

        # STEP 3: Compute Residual Deltas ΔY = |Actual - Self-Adjusted Baseline|
        residual_analysis = self.residual_calc.compute_residuals(telemetry_data, physics_baseline)
        residuals = residual_analysis["residuals"]
        is_anomaly = residual_analysis["anomaly_detected"]

        # STEP 4: PINN Neural Inference Forward Pass (6 Inputs)
        pinn_prediction = None
        try:
            input_tensor = torch.tensor(
                [[actual_cht, actual_egt, actual_oil, physics_cht, physics_egt, altitude]], 
                dtype=torch.float32
            )
            with torch.no_grad():
                pinn_output = self.pinn_model(input_tensor).squeeze(0).tolist()
                pinn_prediction = {
                    "pinn_res_cht": pinn_output[0],
                    "pinn_res_egt": pinn_output[1],
                    "pinn_res_oil": pinn_output[2],
                    "pinn_health_index": pinn_output[3]
                }
        except Exception as e:
            logging.debug(f"[!] PINN Inference bypassed: {e}")

        # STEP 5: Periodically persist learned bias offsets & check for hot-reloaded model weights (~every 30 packets)
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            self.physics_twin.save_calibration()
            self.load_pinn_weights()

        # STEP 6: Pass Residual Deltas to LSTM RUL Predictor Engine
        self.rul_engine.add_telemetry_step(residuals)
        rul_assessment = self.rul_engine.predict_rul()

        # STEP 7: Format Unified Digital Twin State Output Packet
        digital_twin_state = {
            "timestamp": raw_packet.get("timestamp", time.time()),
            "can_id": raw_packet.get("can_id", "0x100"),
            "telemetry_actual": telemetry_data,
            "physics_baseline": physics_baseline,
            "residual_deltas": residuals,
            "pinn_prediction": pinn_prediction,
            "anomaly_flagged": is_anomaly,
            "rul_hours": rul_assessment["predicted_rul_hours"],
            "health_index_pct": rul_assessment["health_index_pct"],
            "maintenance_urgency": rul_assessment["maintenance_urgency"]
        }

        # STEP 8: Persist State Packet to PostgreSQL Database
        self.db_writer.write_metrics(digital_twin_state)

        return digital_twin_state


# =====================================================================
# 3. CONSUMER LOOP & EXECUTION ENTRY-POINT
# =====================================================================
def start_orchestration_service():
    orchestrator = DigitalTwinOrchestrator()
    consumer = initialize_kafka_consumer()

    logging.info("Edge Pipeline Operational. Ingesting Real-Time Telemetry Stream...\n" + "=" * 75)

    if consumer:
        try:
            for message in consumer:
                packet = message.value
                twin_state = orchestrator.process_telemetry_packet(packet)
                
                print(f"| TIMESTAMP: {twin_state['timestamp']:.2f} | CAN ID: {twin_state['can_id']}")
                print(f"  --> Telemetry Actual : {twin_state['telemetry_actual']}")
                print(f"  --> 0D Physics Base  : {twin_state['physics_baseline']}")
                print(f"  --> Deltas (ΔY)      : {twin_state['residual_deltas']}")
                print(f"  --> Health / RUL     : {twin_state['health_index_pct']:.2f}% ({twin_state['rul_hours']:.2f} hrs remaining)")
                print(f"  --> Urgency Status   : {twin_state['maintenance_urgency']}")
                print("-" * 75)
        except KeyboardInterrupt:
            logging.info("Pipeline Consumer Stopped by User.")
            orchestrator.physics_twin.save_calibration()
            consumer.close()
    else:
        # Fallback Direct Pipeline Verification Test using Sample Telemetry Packet
        logging.info("Executing Fallback Direct Pipeline Verification Test...")
        test_packet = {
            "timestamp": time.time(),
            "can_id": "0x100",
            "data": {
                "RPM": 5800.0,
                "MAP": 101.3,
                "CHT": 137.02,
                "EGT": 828.61,
                "Oil_Pressure": 4.05,
                "Ambient_Temp": 15.0,
                "Altitude": 250.0
            }
        }
        twin_state = orchestrator.process_telemetry_packet(test_packet)
        print(f"\n[+] Digital Twin Processed State Packet:\n{json.dumps(twin_state, indent=2)}")


if __name__ == "__main__":
    start_orchestration_service()