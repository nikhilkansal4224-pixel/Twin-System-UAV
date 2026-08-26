import time
import json
import can
from kafka import KafkaProducer  # Install via: pip install kafka-python python-can
from .dbc_decoder import DBCDecoder  # Import the DBC Decoder built previously

# =====================================================================
# 1. KAFKA PRODUCER CONFIGURATION
# =====================================================================
KAFKA_BROKER = "localhost:9092"
KAFKA_TOPIC = "uav-engine-telemetry"

def json_serializer(data):
    """Serializes Python dictionary objects into JSON byte arrays."""
    return json.dumps(data).encode("utf-8")

def initialize_kafka_producer():
    """Attempts to connect to the local Kafka broker."""
    try:
        producer = KafkaProducer(
            bootstrap_servers=[KAFKA_BROKER],
            value_serializer=json_serializer,
            acks=1,  # Wait for broker acknowledgement
            retries=3
        )
        print(f"[+] Connected to Local Kafka Broker at {KAFKA_BROKER}")
        return producer
    except Exception as e:
        print(f"[!] Warning: Unable to connect to Kafka Broker ({e}). Running in standalone console mode.")
        return None

# =====================================================================
# 2. STREAMING PRODUCER LOOP
# =====================================================================
def start_telemetry_stream():
    producer = initialize_kafka_producer()
    decoder = DBCDecoder()  # Initialize DBC translation engine
    
    # Initialize virtual CAN interface (or socketcan on Linux)
    try:
        bus = can.interface.Bus(bustype='virtual', channel='vcan0', receive_own_messages=True)
        print("[+] Virtual CAN-bus Interface Initialized (vcan0)...")
    except Exception as e:
        print(f"[!] Error initializing CAN interface: {e}")
        return

    print(f"[+] Starting Ingestion Stream -> Kafka Topic: '{KAFKA_TOPIC}'...\n")

    try:
        while True:
            # Receive raw message frame from CAN-bus
            rx_msg = bus.recv(timeout=1.0)
            
            if rx_msg:
                # 1. Decode raw hex payload using DBCDecoder
                decoded_telemetry = decoder.decode_frame(rx_msg.arbitration_id, rx_msg.data)
                
                if decoded_telemetry:
                    # Construct standardized JSON payload packet
                    payload_packet = {
                        "timestamp": time.time(),
                        "can_id": hex(rx_msg.arbitration_id),
                        "data": decoded_telemetry
                    }
                    
                    # 2. Publish packet to Kafka Queue
                    if producer:
                        producer.send(KAFKA_TOPIC, value=payload_packet)
                        producer.flush()
                        
                    # Print live stream confirmation
                    print(f"| STREAMED | Topic: {KAFKA_TOPIC} | Payload: {payload_packet}")

            time.sleep(0.1)  # 10Hz streaming loop rate

    except KeyboardInterrupt:
        print("\n[-] Telemetry Producer Pipeline Stopped.")
        if producer:
            producer.close()

# =====================================================================
# MODULE VERIFICATION
# =====================================================================
if __name__ == "__main__":
    start_telemetry_stream()