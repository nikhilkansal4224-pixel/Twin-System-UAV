import logging
import time
import json
from kafka import KafkaConsumer
from src.ai_pipeline.orchestrator import DigitalTwinOrchestrator
from src.db.postgres_writer import PostgresWriter

logging.basicConfig(level=logging.INFO)

def main():
    logging.info("[+] Starting UAV Digital Twin Service...")
    
    # Initialize orchestrator and database writer
    orchestrator = DigitalTwinOrchestrator(
        pinn_path="pinn_model_latest.pth",
        lstm_path="lstm_rul_latest.pth"
    )
    db_writer = PostgresWriter()

    # Retry loop to connect to Kafka while container boots
    consumer = None
    while not consumer:
        try:
            consumer = KafkaConsumer(
                "uav-engine-telemetry",
                bootstrap_servers=["kafka:9092"],
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                auto_offset_reset="latest",
                group_id="digital-twin-group"
            )
            logging.info("[+] Connected to Kafka broker!")
        except Exception as e:
            logging.warning(f"[!] Kafka not ready yet ({e}). Retrying in 5s...")
            time.sleep(5)

    logging.info("[+] Processing live telemetry stream...")
    for message in consumer:
        try:
            raw_frame = message.value
            twin_state = orchestrator.process_telemetry_frame(raw_frame)
            db_writer.write_metrics(twin_state)
        except Exception as e:
            logging.error(f"[!] Processing error: {e}")

if __name__ == "__main__":
    main()