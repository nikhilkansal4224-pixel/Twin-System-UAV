import time
import struct
import random
import logging
import can  # Install via: pip install python-can

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# =====================================================================
# 1. CAN BUS TELEMETRY FRAME CONFIGURATIONS (.DBC MAPPING)
# =====================================================================
# Physical Value = (Raw Integer * Factor) + Offset
CAN_CONFIG = {
    0x100: {  # Engine Speed & Pressures
        "name": "Engine_Dynamics",
        "signals": {
            "RPM": {"start_bit": 0, "length": 16, "factor": 1.0, "offset": 0.0, "unit": "RPM"},
            "MAP": {"start_bit": 16, "length": 16, "factor": 0.1, "offset": 0.0, "unit": "kPa"},
            "Oil_Pressure": {"start_bit": 32, "length": 16, "factor": 0.01, "offset": 0.0, "unit": "bar"}
        }
    },
    0x200: {  # Thermal Profile
        "name": "Engine_Thermal",
        "signals": {
            "CHT": {"start_bit": 0, "length": 16, "factor": 0.1, "offset": -40.0, "unit": "degC"},
            "EGT": {"start_bit": 16, "length": 16, "factor": 0.1, "offset": 0.0, "unit": "degC"}
        }
    }
}


class CANDecoder:
    def __init__(self, config: dict = None):
        self.config = config or CAN_CONFIG

    def generate_raw_can_frame(self, can_id: int) -> bytes:
        """Simulates real-time sensor measurements and packs them into a CAN hex byte array."""
        if can_id == 0x100:
            rpm = int(5800 + random.uniform(-25, 25))
            map_val = int((101.3 + random.uniform(-1, 1)) / 0.1)
            oil_p = int((4.5 + random.uniform(-0.1, 0.1)) / 0.01)
            return struct.pack("<HHH", rpm, map_val, oil_p) + b"\x00\x00"

        elif can_id == 0x200:
            cht_raw = int((115.0 - (-40.0) + random.uniform(-0.5, 0.5)) / 0.1)
            egt_raw = int((820.0 + random.uniform(-2.0, 2.0)) / 0.1)
            return struct.pack("<HH", cht_raw, egt_raw) + b"\x00\x00\x00\x00"

        return b"\x00" * 8

    def decode_can_message(self, can_id: int, payload_bytes: bytes) -> dict:
        """Parses raw hexadecimal payloads and applies linear DBC transformations."""
        if can_id not in self.config:
            return {}

        cfg = self.config[can_id]
        decoded = {}

        if can_id == 0x100:
            rpm_raw, map_raw, oil_raw = struct.unpack("<HHH", payload_bytes[:6])
            decoded["RPM"] = (rpm_raw * cfg["signals"]["RPM"]["factor"]) + cfg["signals"]["RPM"]["offset"]
            decoded["MAP"] = (map_raw * cfg["signals"]["MAP"]["factor"]) + cfg["signals"]["MAP"]["offset"]
            decoded["Oil_Pressure"] = (oil_raw * cfg["signals"]["Oil_Pressure"]["factor"]) + cfg["signals"]["Oil_Pressure"]["offset"]

        elif can_id == 0x200:
            cht_raw, egt_raw = struct.unpack("<HH", payload_bytes[:4])
            decoded["CHT"] = (cht_raw * cfg["signals"]["CHT"]["factor"]) + cfg["signals"]["CHT"]["offset"]
            decoded["EGT"] = (egt_raw * cfg["signals"]["EGT"]["factor"]) + cfg["signals"]["EGT"]["offset"]

        return decoded

    def format_orchestrator_packet(self, can_id: int, decoded_data: dict) -> dict:
        """Formats decoded dictionary into standardized Digital Twin ingest payload."""
        return {
            "timestamp": time.time(),
            "can_id": f"0x{can_id:03X}",
            "data": decoded_data
        }


# =====================================================================
# LIVE VIRTUAL CAN STREAM SERVICE
# =====================================================================
def run_telemetry_stream(channel: str = "vcan0"):
    decoder = CANDecoder()
    try:
        bus = can.interface.Bus(bustype="virtual", channel=channel, receive_own_messages=True)
        logging.info(f"[+] Virtual CAN-bus Interface Initialized ({channel})...")
    except Exception as e:
        logging.warning(f"[!] Could not bind to virtual CAN bus ({e}). Running in loopback mode.")
        bus = None

    logging.info("[+] Starting Live UAV Aero-Piston Telemetry Stream...\n")

    try:
        while True:
            for can_id in [0x100, 0x200]:
                raw_payload = decoder.generate_raw_can_frame(can_id)

                if bus:
                    msg = can.Message(arbitration_id=can_id, data=raw_payload, is_extended_id=False)
                    bus.send(msg)
                    rx_msg = bus.recv(timeout=1.0)
                    payload_bytes = rx_msg.data if rx_msg else raw_payload
                else:
                    payload_bytes = raw_payload

                decoded_data = decoder.decode_can_message(can_id, payload_bytes)
                packet = decoder.format_orchestrator_packet(can_id, decoded_data)

                hex_str = " ".join(f"{b:02X}" for b in payload_bytes)
                print(f"| ID: {packet['can_id']} | Raw: [{hex_str}]")
                print(f"  --> Decoded Packet: {packet['data']}")

            print("-" * 70)
            time.sleep(0.5)

    except KeyboardInterrupt:
        logging.info("Telemetry Stream Terminated.")


if __name__ == "__main__":
    run_telemetry_stream()