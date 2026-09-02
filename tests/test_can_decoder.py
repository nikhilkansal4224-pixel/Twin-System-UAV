import sys
import os
import struct

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingestion.dbc_decoder import DBCDecoder


def test_dbc_decoding():
    """Round-trip a known RPM/MAP/Oil_Pressure frame through the fallback decoder."""
    decoder = DBCDecoder()
    payload = struct.pack("<HHH", 5800, 1013, 450) + b"\x00\x00"

    decoded = decoder.decode_frame(0x100, payload)

    assert abs(decoded["RPM"] - 5800.0) < 1.0
    assert abs(decoded["MAP"] - 101.3) < 0.5
    assert abs(decoded["Oil_Pressure"] - 4.5) < 0.05


def test_thermal_frame_decoding():
    """Round-trip a known CHT/EGT frame through the fallback decoder."""
    decoder = DBCDecoder()
    payload = struct.pack("<HH", 1550, 8200) + b"\x00\x00\x00\x00"

    decoded = decoder.decode_frame(0x200, payload)

    assert abs(decoded["CHT"] - 115.0) < 0.5
    assert abs(decoded["EGT"] - 820.0) < 0.5


def test_unknown_can_id_returns_empty():
    decoder = DBCDecoder()
    assert decoder.decode_frame(0x999, b"\x00" * 8) == {}
