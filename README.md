# 🚁 Indigenous Digital Twin System for UAV Aero-Piston Engines

An end-to-end, edge-deployable **Digital Twin & Predictive Health Monitoring (PHM)** system for UAV aero-piston engines (e.g., Rotax 912 / 914 series).

This system ingests real-time CAN-bus engine telemetry, computes a **0D thermodynamic physics baseline**, calculates residual deviations (ΔY = Actual − Physics-Expected), scores anomalies via a **Physics-Informed Neural Network (PINN)**, and estimates **Remaining Useful Life (RUL)** using an LSTM sequence model.

---

## 🛠️ System Architecture & Data Pipeline

```
┌─────────────────┐   CAN Frame    ┌──────────────────────┐   Decoded Signals   ┌───────────────────────┐
│  UAV Engine /    ├───────────────►│  DBC Telemetry       ├─────────────────────►│  0D Thermodynamic     │
│  CAN Replay Log  │  (0x100/0x200) │  Decoder              │ (RPM, MAP, CHT, EGT) │  Reference Model       │
└─────────────────┘                 └──────────────────────┘                       └──────────┬────────────┘
                                                                                                │ Physics Baseline
                                                                                                ▼
┌─────────────────┐   Health & RUL  ┌──────────────────────┐   Residual Deltas    ┌───────────────────────┐
│ SQLite Database  │◄────────────────┤  LSTM RUL Engine &    │◄──────────────────────┤ Residual Calculator   │
│ (data/db_file)   │ & Anomaly State │  PINN Diagnostics     │  ΔCHT = |Actual-Phys| │                        │
└────────┬─────────┘                 └──────────────────────┘                       └───────────────────────┘
         │
         ▼
┌──────────────────┐
│ Grafana Dashboard │ (Live GCS Visualization)
└──────────────────┘
```

---

## 🔑 Key Features

- **Real-Time CAN Telemetry Decoding** — ingests raw binary CAN frames (`0x100`, `0x200`) using DBC signal definitions.
- **0D Thermodynamic Reference Model** — solves first-law energy-balance equations (`scipy.solve_ivp`) to establish a dynamic, physics-based healthy baseline for CHT and EGT.
- **Residual Analytics (ΔY)** — flags subtle thermal degradation by comparing actual telemetry against physics expectations, rather than fixed thresholds.
- **LSTM RUL Predictor** — sequence model estimating Remaining Useful Life (flight-hours) and a Health Index (%).
- **Physics-Informed Neural Network (PINN)** — scores fault severity, trained with a loss function that penalizes physically implausible predictions (ideal gas law, energy conservation), not just label mismatch.
- **Lightweight SQLite Persistence** — zero-dependency, edge-friendly logging with no background daemon.
- **Grafana Dashboard Integration** — pre-configured queries for live GCS telemetry plotting and anomaly alerting.

---

## 📂 Project Structure

```text
Twin-System-UAV/
├── main.py                          # Canonical entry point — live 10Hz simulation & orchestration loop
├── create_db.py                     # Initializes the SQLite schema
├── populate_db.py                   # Seeds synthetic test telemetry
├── data/
│   └── engine_telemetry.db          # SQLite time-series database (auto-generated)
├── config/
│   ├── engine_rotax914.json         # Rotax 914 engine geometry/constants
│   ├── kafka_config.json            # Kafka broker/topic config
│   └── uav_telemetry.dbc            # CAN signal matrix definition
├── models/saved_weights/
│   ├── pinn_weights.pth             # Pretrained PINN weights (used by main.py)
│   └── lstm_rul_weights.pth         # Pretrained LSTM weights (used by main.py)
├── src/
│   ├── ingestion/
│   │   ├── dbc_decoder.py           # CAN frame → physical value decoder
│   │   ├── can_listener.py          # Real CAN-bus hardware listener
│   │   └── kafka_producer.py        # Streams decoded telemetry to Kafka
│   ├── physics_engine/
│   │   ├── thermodynamics.py        # 0D ODE thermodynamic solver (used by main.py)
│   │   ├── residual_calculator.py   # Residual (ΔY) computation
│   │   ├── combustion.py            # Standalone Wiebe combustion model
│   │   ├── heat_transfer.py         # Standalone Woschni heat-transfer model
│   │   └── kinematics.py            # Standalone crank-slider kinematics
│   ├── ai_pipeline/
│   │   ├── pinn_model.py            # PINN architecture (used by main.py)
│   │   ├── lstm_rul.py              # LSTM RUL model (used by main.py)
│   │   ├── loss_functions.py        # Physics-informed loss (gas law + energy conservation)
│   │   ├── fault_generator.py       # Synthetic fault-injection training data generator
│   │   └── train_and_export.py      # Offline training script, exports .pth weights
│   └── orchestration/
│       ├── qlite_writer.py          # SQLite persistence layer (used by main.py)
│       └── kafka_consumer.py        # Kafka-based streaming orchestration (alternate path)
├── dashboard/grafana/
│   └── uav_engine_twin_dashboard.json
└── tests/
    ├── test_0d_physics.py
    └── test_can_decoder.py
```

> **Note on repo scope:** this repository also contains an in-progress secondary track (`src/main.py`, `src/db/postgres_writer.py`, `src/streaming/`, `src/models/`, `src/can_bus/`, `app_desktop.py`) exploring a Postgres/Kafka-backed desktop application version of this system. It is **not** the entry point described in this README and is not required to run the core digital twin pipeline documented here.

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- Python 3.9+
- Grafana Server (optional, for GCS visualization)

### 2. Clone & Install Dependencies

```bash
git clone https://github.com/nikhilkansal4224-pixel/Twin-System-UAV.git
cd Twin-System-UAV

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Initialize the SQLite Database

```bash
python3 create_db.py
```

Optionally, seed sample data:

```bash
python3 populate_db.py
```

---

## 🚀 Running the Digital Twin

```bash
python3 main.py
```

### Expected Console Output

```text
======================================================================
Initializing Indigenous Digital Twin System for UAV Aero-Piston Engines
Target Application: MALE UAV Ground Control Station (GCS) Edge Deployment
======================================================================
[1/5] Loading DBC Telemetry Decoder...
[2/5] Initializing 0D Thermodynamic Reference Model (SciPy solve_ivp)...
[3/5] Loading PyTorch PINN Architecture & LSTM RUL Engine...
      --> Pre-trained PINN weights loaded: 'models/saved_weights/pinn_weights.pth'
      --> Pre-trained LSTM RUL weights loaded: 'models/saved_weights/lstm_rul_weights.pth'
[4/5] Connecting to SQLite Persistence Database...
[+] SQLite Database initialized at: 'data/engine_telemetry.db'
[5/5] All Edge Subsystems Successfully Online!

Starting Edge Execution Loop (Simulating 10Hz CAN-bus Telemetry)...
---------------------------------------------------------------------------
Iter 01 | CHT: 115.0°C (Δ 0.0) | EGT: 820.0°C (Δ 0.0) [NOMINAL]
       --> HEALTH: 100.0% | RUL: 1200.0 hrs | STATUS: NOMINAL
```

---

## 📊 Grafana Visualization Setup

1. Launch Grafana (`http://127.0.0.1:3000`).
2. Navigate to **Connections → Data Sources → Add data source → SQLite**.
3. Set the database path to the absolute path of `data/engine_telemetry.db` on your machine.
4. Example query for CHT actual vs. physics baseline:

```sql
SELECT
  datetime(timestamp, 'unixepoch') AS time,
  actual_cht,
  physics_cht,
  residual_cht
FROM uav_aero_engine_metrics
ORDER BY id DESC;
```

---

## 📄 License

This project is licensed under the MIT License — see the `LICENSE` file for details.
