import os
import sys
import json
import time
import logging
import can
from kafka import KafkaProducer  # Install via: pip install kafka-python python-can

# Ensure project root is in Python search path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.can_bus.can_decoder import CANDecoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# =====================================================================
# CONFIGURATION
# =====================================================================
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "uav-engine-telemetry")
CAN_CHANNEL = os.getenv("CAN_CHANNEL", "vcan0")
CAN_BUSTYPE = os.getenv("CAN_BUSTYPE", "socketcan")


def json_serializer(data: dict) -> bytes:
    """Serializes Python dictionary to JSON byte payload."""
    return json.dumps(data).encode("utf-8")


def initialize_kafka_producer() -> KafkaProducer:
    """Initializes connection to Kafka Producer broker."""
    try:
        producer = KafkaProducer(
            bootstrap_servers=[KAFKA_BROKER],
            value_serializer=json_serializer,
            retries=3
        )
        logging.info(f"[+] Connected to Kafka Producer at {KAFKA_BROKER}")
        return producer
    except Exception as e:
        logging.warning(f"[!] Kafka Broker unavailable ({e}). Running listener in standalone mode.")
        return None


# =====================================================================
# CAN BUS LISTENER SERVICE
# =====================================================================
class CANListenerService:
    def __init__(self, channel: str = CAN_CHANNEL, bustype: str = CAN_BUSTYPE):
        self.decoder = CANDecoder()
        self.producer = initialize_kafka_producer()
        self.channel = channel
        self.bustype = bustype

    def start_listening(self):
        """Connects to CAN bus socket and continuously listens for telemetry frames."""
        try:
            bus = can.interface.Bus(channel=self.channel, bustype=self.bustype, receive_own_messages=True)
            logging.info(f"[+] Listening on CAN interface '{self.channel}' ({self.bustype})...")
        except Exception as e:
            logging.error(f"[!] Failed to bind to CAN bus '{self.channel}': {e}")
            logging.info("[*] Switching to simulated loopback mode...")
            bus = None

        logging.info(f"[+] Streaming decoded frames to Kafka topic '{KAFKA_TOPIC}'...\n" + "=" * 75)

        try:
            while True:
                if bus:
                    # Read frame from SocketCAN
                    rx_msg = bus.recv(timeout=1.0)
                    if rx_msg is None:
                        continue
                    can_id = rx_msg.arbitration_id
                    payload = rx_msg.data
                else:
                    # Fallback simulation if no active physical/virtual socket
                    time.sleep(0.5)
                    can_id = 0x100
                    payload = self.decoder.generate_raw_can_frame(can_id)

                # Decode CAN frame
                decoded_data = self.decoder.decode_can_message(can_id, payload)
                if not decoded_data:
                    continue

                packet = self.decoder.format_orchestrator_packet(can_id, decoded_data)

                # Publish to Kafka
                if self.producer:
                    self.producer.send(KAFKA_TOPIC, value=packet)

                hex_str = " ".join(f"{b:02X}" for b in payload)
                print(f"| ID: 0x{can_id:03X} | Raw: [{hex_str:<23}] | Decoded: {decoded_data}")

        except KeyboardInterrupt:
            logging.info("[-] CAN Listener Service stopped by user.")
            if self.producer:
                self.producer.flush()
                self.producer.close()


if __name__ == "__main__":
    listener = CANListenerService()
    listener.start_listening()