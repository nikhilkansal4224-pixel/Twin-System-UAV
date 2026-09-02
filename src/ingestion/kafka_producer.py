import os
import sys
import json
import time
import logging
import can
from kafka import KafkaProducer  # Install via: pip install kafka-python python-can

# Ensure project root is in Python search path for cross-module execution
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.ingestion.dbc_decoder import DBCDecoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# =====================================================================
# CONFIGURATION & ENVIRONMENT OVERRIDES
# =====================================================================
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "uav-engine-telemetry")
CAN_CHANNEL = os.getenv("CAN_CHANNEL", "vcan0")
CAN_BUSTYPE = os.getenv("CAN_BUSTYPE", "virtual")


def json_serializer(data: dict) -> bytes:
    """Serializes Python dictionary objects into JSON byte arrays."""
    return json.dumps(data).encode("utf-8")


def initialize_kafka_producer() -> KafkaProducer:
    """Attempts to connect to the local Kafka broker with retry logic."""
    try:
        producer = KafkaProducer(
            bootstrap_servers=[KAFKA_BROKER],
            value_serializer=json_serializer,
            acks=1,  # Wait for broker acknowledgement
            retries=3
        )
        logging.info(f"[+] Connected to Local Kafka Broker at {KAFKA_BROKER}")
        return producer
    except Exception as e:
        logging.warning(f"[!] Unable to connect to Kafka Broker ({e}). Running in standalone console mode.")
        return None


# =====================================================================
# STREAMING PRODUCER CLASS
# =====================================================================
class TelemetryProducerService:
    def __init__(self, dbc_path: str = None, channel: str = CAN_CHANNEL, bustype: str = CAN_BUSTYPE):
        self.decoder = DBCDecoder(dbc_file_path=dbc_path)
        self.producer = initialize_kafka_producer()
        self.channel = channel
        self.bustype = bustype

    def start_telemetry_stream(self, rate_hz: float = 10.0):
        """
        Continuously ingests CAN bus frames, decodes engineering units,
        and streams JSON packets to Kafka at the specified sampling rate.
        """
        try:
            bus = can.interface.Bus(bustype=self.bustype, channel=self.channel, receive_own_messages=True)
            logging.info(f"[+] Virtual CAN-bus Interface Initialized ({self.channel})...")
        except Exception as e:
            logging.error(f"[!] Error initializing CAN interface '{self.channel}': {e}")
            return

        logging.info(f"[+] Starting Ingestion Stream -> Kafka Topic: '{KAFKA_TOPIC}' @ {rate_hz} Hz...\n")
        sleep_interval = 1.0 / rate_hz

        try:
            while True:
                rx_msg = bus.recv(timeout=0.1)

                if rx_msg:
                    # 1. Decode raw hex payload using DBCDecoder
                    decoded_telemetry = self.decoder.decode_frame(rx_msg.arbitration_id, rx_msg.data)

                    if decoded_telemetry:
                        # Construct standardized JSON payload packet
                        payload_packet = {
                            "timestamp": time.time(),
                            "can_id": f"0x{rx_msg.arbitration_id:03X}",
                            "data": decoded_telemetry
                        }

                        # 2. Publish packet to Kafka Queue
                        if self.producer:
                            self.producer.send(KAFKA_TOPIC, value=payload_packet)
                            self.producer.flush()

                        hex_str = " ".join(f"{b:02X}" for b in rx_msg.data)
                        print(f"| STREAMED | Topic: {KAFKA_TOPIC} | ID: {payload_packet['can_id']} | Payload: {payload_packet['data']}")

                time.sleep(sleep_interval)

        except KeyboardInterrupt:
            logging.info("[-] Telemetry Producer Pipeline Stopped by user.")
            if self.producer:
                self.producer.close()


# =====================================================================
# EXECUTION ENTRY-POINT
# =====================================================================
if __name__ == "__main__":
    producer_service = TelemetryProducerService()
    producer_service.start_telemetry_stream(rate_hz=10.0)