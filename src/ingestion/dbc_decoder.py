import struct
import cantools  # Install via: pip install cantools python-can

class DBCDecoder:
    def __init__(self, dbc_file_path=None):
        """
        Initializes the CAN DBC database parser.
        If a .DBC file path is provided, it loads via cantools.
        Otherwise, it falls back to hardcoded CAN signal mapping configurations.
        """
        self.db = None
        if dbc_file_path:
            try:
                self.db = cantools.database.load_file(dbc_file_path)
                print(f"[+] Successfully loaded CAN Database: {dbc_file_path}")
            except Exception as e:
                print(f"[!] Warning: Failed to load .DBC file ({e}). Using inline dictionary fallback.")

        # Hardcoded fallback signal configuration mapping
        self.fallback_config = {
            0x100: {  # Engine Speed & Pressures Frame
                "RPM": {"start_byte": 0, "length": 2, "fmt": "<H", "factor": 1.0, "offset": 0.0, "unit": "RPM"},
                "MAP": {"start_byte": 2, "length": 2, "fmt": "<H", "factor": 0.1, "offset": 0.0, "unit": "kPa"},
                "Oil_Pressure": {"start_byte": 4, "length": 2, "fmt": "<H", "factor": 0.01, "offset": 0.0, "unit": "bar"}
            },
            0x200: {  # Thermal Profile Frame
                "CHT": {"start_byte": 0, "length": 2, "fmt": "<H", "factor": 0.1, "offset": -40.0, "unit": "degC"},
                "EGT": {"start_byte": 2, "length": 2, "fmt": "<H", "factor": 0.1, "offset": 0.0, "unit": "degC"}
            }
        }

    def decode_frame(self, can_id: int, payload_bytes: bytes) -> dict:
        """
        Parses raw hex payload bytes into physical engineering units using arbitration ID.
        
        :param can_id: CAN Arbitration ID (e.g., 0x100, 0x200)
        :param payload_bytes: Raw CAN hex frame bytes (up to 8 bytes)
        :return: Dictionary containing decoded signal names and physical values
        """
        # Option 1: Use loaded .DBC Database
        if self.db:
            try:
                message = self.db.get_message_by_frame_id(can_id)
                decoded_signals = message.decode(payload_bytes)
                return decoded_signals
            except Exception as e:
                print(f"[!] DBC Decoding Error for ID 0x{can_id:03X}: {e}")
                return {}

        # Option 2: Fallback manual bit unpacking & linear scaling
        if can_id not in self.fallback_config:
            return {}

        decoded_signals = {}
        frame_schema = self.fallback_config[can_id]

        for signal_name, spec in frame_schema.items():
            start = spec["start_byte"]
            end = start + spec["length"]
            raw_slice = payload_bytes[start:end]

            if len(raw_slice) == spec["length"]:
                # Unpack raw byte stream into integer (Little-Endian)
                raw_int = struct.unpack(spec["fmt"], raw_slice)[0]
                
                # Apply linear conversion formula: Physical Value = (Raw * Factor) + Offset
                physical_value = (raw_int * spec["factor"]) + spec["offset"]
                decoded_signals[signal_name] = round(physical_value, 2)

        return decoded_signals


# =====================================================================
# MODULE VERIFICATION & TEST DRIVE
# =====================================================================
if __name__ == "__main__":
    print("[+] Testing DBC Decoder Engine...")
    decoder = DBCDecoder()  # Initialize decoder (runs fallback config)

    # 1. Simulate Raw Hex Payload for ID 0x100 (RPM = 5800, MAP = 101.3 kPa, Oil_P = 4.5 bar)
    # RPM: 5800 -> 0x16A8 | MAP: 1013 (raw) -> 0x03F5 | Oil_P: 450 (raw) -> 0x01C2
    raw_payload_100 = struct.pack("<HHH", 5800, 1013, 450) + b'\x00\x00'
    hex_str_100 = ' '.join(f'{b:02X}' for b in raw_payload_100)

    decoded_100 = decoder.decode_frame(0x100, raw_payload_100)
    print("\n--- [FRAME 0x100: ENGINE DYNAMICS] ---")
    print(f"Raw Hex Payload  : [{hex_str_100}]")
    print(f"Decoded Telemetry: {decoded_100}")

    # 2. Simulate Raw Hex Payload for ID 0x200 (CHT = 115.0 °C, EGT = 820.0 °C)
    # CHT: (115 - (-40))/0.1 = 1550 (raw) -> 0x060E | EGT: 820/0.1 = 8200 (raw) -> 0x2008
    raw_payload_200 = struct.pack("<HH", 1550, 8200) + b'\x00\x00\x00\x00'
    hex_str_200 = ' '.join(f'{b:02X}' for b in raw_payload_200)

    decoded_200 = decoder.decode_frame(0x200, raw_payload_200)
    print("\n--- [FRAME 0x200: THERMAL PROFILE] ---")
    print(f"Raw Hex Payload  : [{hex_str_200}]")
    print(f"Decoded Telemetry: {decoded_200}")