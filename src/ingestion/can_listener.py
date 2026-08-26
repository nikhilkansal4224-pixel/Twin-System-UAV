import time
import struct
import random
import can  # Install via: pip install python-can cantools

# =====================================================================
# 1. DEFINE CAN BUS TELEMETRY FRAME CONFIGURATIONS (.DBC MAPPING)
# =====================================================================
# Physical Value = (Raw Integer * Factor) + Offset
CAN_CONFIG = {
    0x100: {  # Engine Speed & Pressures
        "name": "Engine_Dynamics",
        "signals": {
            "RPM": {"start_bit": 0, "length": 16, "factor": 1.0, "offset": 0, "unit": "RPM"},
            "MAP": {"start_bit": 16, "length": 16, "factor": 0.1, "offset": 0, "unit": "kPa"}, # Manifold Absolute Pressure
            "Oil_Pressure": {"start_bit": 32, "length": 16, "factor": 0.01, "offset": 0, "unit": "bar"}
        }
    },
    0x200: {  # Thermal Profile
        "name": "Engine_Thermal",
        "signals": {
            "CHT": {"start_bit": 0, "length": 16, "factor": 0.1, "offset": -40.0, "unit": "degC"}, # Cylinder Head Temp
            "EGT": {"start_bit": 16, "length": 16, "factor": 0.1, "offset": 0, "unit": "degC"}     # Exhaust Gas Temp
        }
    }
}

# =====================================================================
# 2. RAW HEX TELEMETRY ENCODER (SIMULATING UAV ENGINE SENSORS)
# =====================================================================
def generate_raw_can_frame(can_id):
    """Simulates real-time sensor measurements and packs them into a CAN hex byte array."""
    if can_id == 0x100:
        # Simulate baseline operational values with slight physical noise
        rpm = int(5800 + random.uniform(-25, 25))          # ~5800 RPM
        map_val = int((101.3 + random.uniform(-1, 1)) / 0.1) # ~101.3 kPa
        oil_p = int((4.5 + random.uniform(-0.1, 0.1)) / 0.01) # ~4.5 bar
        
        # Pack into 6 bytes (3 unsigned 16-bit integers, Little-Endian)
        data = struct.pack("<HHH", rpm, map_val, oil_p) + b'\x00\x00'
        return data

    elif can_id == 0x200:
        cht_raw = int((115.0 - (-40.0) + random.uniform(-0.5, 0.5)) / 0.1) # ~115 degC
        egt_raw = int((820.0 + random.uniform(-2.0, 2.0)) / 0.1)            # ~820 degC
        
        # Pack into 4 bytes (2 unsigned 16-bit integers, Little-Endian)
        data = struct.pack("<HH", cht_raw, egt_raw) + b'\x00\x00\x00\x00'
        return data

# =====================================================================
# 3. CAN DECODER (.DBC LINEAR TRANSFORMATION)
# =====================================================================
def decode_can_message(can_id, payload_bytes):
    """Parses raw hexadecimal payloads and applies DBC linear transformations."""
    if can_id not in CAN_CONFIG:
        return None

    config = CAN_CONFIG[can_id]
    decoded_telemetry = {}

    if can_id == 0x100:
        rpm_raw, map_raw, oil_raw = struct.unpack("<HHH", payload_bytes[:6])
        
        # Physical Value = (Raw * Factor) + Offset
        decoded_telemetry["RPM"] = (rpm_raw * config["signals"]["RPM"]["factor"]) + config["signals"]["RPM"]["offset"]
        decoded_telemetry["MAP"] = (map_raw * config["signals"]["MAP"]["factor"]) + config["signals"]["MAP"]["offset"]
        decoded_telemetry["Oil_Pressure"] = (oil_raw * config["signals"]["Oil_Pressure"]["factor"]) + config["signals"]["Oil_Pressure"]["offset"]

    elif can_id == 0x200:
        cht_raw, egt_raw = struct.unpack("<HH", payload_bytes[:4])
        
        decoded_telemetry["CHT"] = (cht_raw * config["signals"]["CHT"]["factor"]) + config["signals"]["CHT"]["offset"]
        decoded_telemetry["EGT"] = (egt_raw * config["signals"]["EGT"]["factor"]) + config["signals"]["EGT"]["offset"]

    return decoded_telemetry

# =====================================================================
# 4. MAIN LOOP (REAL-TIME SIMULATION & STREAMING)
# =====================================================================
def run_telemetry_stream():
    # Setup virtual CAN bus interface
    bus = can.interface.Bus(bustype='virtual', channel='vcan0', receive_own_messages=True)
    print("[+] Virtual CAN-bus Interface Initialized (vcan0)...")
    print("[+] Starting Live UAV Aero-Piston Telemetry Stream...\n")

    try:
        while True:
            for can_id in [0x100, 0x200]:
                # 1. Generate simulated sensor frame
                raw_payload = generate_raw_can_frame(can_id)
                msg = can.Message(arbitration_id=can_id, data=raw_payload, is_extended_id=False)
                bus.send(msg)

                # 2. Receive and decode frame
                rx_msg = bus.recv(timeout=1.0)
                if rx_msg:
                    decoded_data = decode_can_message(rx_msg.arbitration_id, rx_msg.data)
                    hex_str = ' '.join(f'{b:02X}' for b in rx_msg.data)
                    
                    print(f"| ID: 0x{rx_msg.arbitration_id:03X} | Raw Payload: [{hex_str}]")
                    print(f"  --> Decoded Units: {decoded_data}")
            
            print("-" * 70)
            time.sleep(0.5) # Simulate 2Hz telemetry transmission loop

    except KeyboardInterrupt:
        print("\n[-] Telemetry Stream Terminated.")

if __name__ == "__main__":
    run_telemetry_stream()