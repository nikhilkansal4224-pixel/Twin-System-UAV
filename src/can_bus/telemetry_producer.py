import os
import sys
import json
import time
import random
import logging
from dotenv import load_dotenv

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
TOPIC = os.getenv("KAFKA_TOPIC", "uav-engine-telemetry")


def json_serializer(data):
    return json.dumps(data).encode("utf-8")


def run_telemetry_producer():
    try:
        from kafka import KafkaProducer
        producer = KafkaProducer(
            bootstrap_servers=[KAFKA_BROKER],
            value_serializer=json_serializer
        )
        logging.info(f"[+] Connected to Kafka Broker at {KAFKA_BROKER}. Streaming to '{TOPIC}'...")
    except Exception as e:
        logging.error(f"[!] Failed to connect to Kafka broker: {e}")
        logging.info("[i] Ensure Docker or local Kafka is running on localhost:9092.")
        return

    print("\nStreaming UAV Rotax 914 CAN Telemetry (Press Ctrl+C to stop)...")
    print("=" * 70)

    step = 0
    try:
        while True:
            # Simulate realistic Rotax 914 operating telemetry with slight thermal drift
            rpm = 5800.0 + random.uniform(-40.0, 40.0)
            map_kpa = 101.3 + random.uniform(-1.2, 1.2)
            cht = 135.0 + (step * 0.02) + random.uniform(-0.8, 0.8)
            egt = 825.0 + random.uniform(-4.0, 4.0)
            oil_press = 4.05 + random.uniform(-0.08, 0.08)
            ambient_temp = 15.0 + random.uniform(-0.2, 0.2)
            altitude = 250.0 + random.uniform(-1.5, 1.5)

            payload = {
                "timestamp": time.time(),
                "can_id": "0x100",
                "data": {
                    "RPM": round(rpm, 2),
                    "MAP": round(map_kpa, 2),
                    "CHT": round(cht, 2),
                    "EGT": round(egt, 2),
                    "Oil_Pressure": round(oil_press, 2),
                    "Ambient_Temp": round(ambient_temp, 2),
                    "Altitude": round(altitude, 2)
                }
            }

            producer.send(TOPIC, value=payload)
            logging.info(f"Sent CAN Frame -> RPM: {payload['data']['RPM']} | CHT: {payload['data']['CHT']}°C | EGT: {payload['data']['EGT']}°C")

            step += 1
            time.sleep(1.0)

    except KeyboardInterrupt:
        logging.info("Stopping Telemetry Producer...")
        producer.flush()
        producer.close()


if __name__ == "__main__":
    run_telemetry_producer()