Here is the complete **`README.md`** file content formatted directly for copy-pasting into your project root:

```markdown
# 🚁 Indigenous Digital Twin System for UAV Aero-Piston Engines

An end-to-end, edge-deployable **Digital Twin & Predictive Health Monitoring (PHM)** system for UAV aero-piston engines (e.g., Rotax 912 / 914 series). 

Designed for deployment on Ground Control Station (GCS) edge hardware, this system ingests real-time CAN-bus engine telemetry, executes a **0D thermodynamic physics baseline**, computes residual deviations ($\Delta Y$), flags thermal anomalies via a **Physics-Informed Neural Network (PINN)**, and estimates **Remaining Useful Life (RUL)** using an LSTM sequence model.

---

## 🛠️ System Architecture & Data Pipeline


```

┌─────────────────┐    CAN Frame     ┌────────────────────────┐    Decoded Signals    ┌──────────────────────┐
│  UAV Engine /   ├─────────────────►│ DBC Telemetry Decoder  ├────────────────────►│  0D Thermodynamic    │
│ CAN Replay Log  │    (0x100/0x200) │ (src/1_ingestion)      │ (RPM, MAP, CHT, EGT)│  Reference Model     │
└─────────────────┘                  └────────────────────────┘                     └──────────┬───────────┘
│ Physics Baseline
▼
┌─────────────────┐   Health & RUL   ┌────────────────────────┐    Residual Deltas    ┌──────────────────────┐
│ SQLite Database ├──────────────────┤   LSTM RUL Engine &    │◄───────────────────┤ Residual Calculator  │
│ (data/db_file)  │  & Anomaly State │   PINN Diagnostics     │   ΔCHT = |Act-Phys|   │ (src/2_physics)      │
└────────┬────────┘                  └────────────────────────┘                       └──────────────────────┘
│
▼
┌─────────────────┐
│ Grafana Dashboard│ (Live GCS Visualization)
└─────────────────┘

```

---

## 🔑 Key Features

* **Real-Time CAN Telemetry Decoding:** Ingests raw binary CAN frames (`0x100`, `0x200`) using DBC signal definitions.
* **0D Thermodynamic Reference Model:** Solves First-Law differential equations (`SciPy solve_ivp`) to establish dynamic healthy baseline values for Cylinder Head Temperature (CHT) and Exhaust Gas Temperature (EGT).
* **Residual Analytics ($\Delta Y$):** Isolates subtle thermal degradation by evaluating absolute deviations between actual telemetry and physics expectations.
* **LSTM RUL & Degradation Predictor:** Multi-step time-series model predicting Remaining Useful Life in flight hours and Health Index percentage (%).
* **Lightweight SQLite Persistence:** Zero-dependency, edge-friendly local database logging without background daemon overhead.
* **Grafana Dashboard Integration:** Pre-configured SQL queries for live GCS telemetry plotting and anomaly alert triggering.

---

## 📂 Project Structure

```text
uav-engine-digital-twin/
├── create_db.py                  # Standalone script to initialize SQLite schema
├── populate_db.py                # Script to populate synthetic test telemetry
├── main.py                       # Core 10Hz live execution loop & orchestrator
├── data/
│   └── engine_telemetry.db       # SQLite time-series database (auto-generated)
├── dbc/
│   └── uav_engine.dbc            # CAN matrix definition file
├── models/
│   └── saved_weights/
│       ├── pinn_weights.pth      # Pre-trained PINN PyTorch model
│       └── lstm_rul_weights.pth  # Pre-trained LSTM sequence model
└── src/
    ├── 1_ingestion/
    │   └── dbc_decoder.py        # CAN ID parser
    ├── 2_physics_engine/
    │   ├── thermodynamics.py     # 0D ODE thermodynamic solver
    │   └── residual_calculator.py# Residual compute engine (ΔY)
    ├── 3_pinn_rul/
    │   └── lstm_rul_engine.py    # PyTorch inference module
    └── 4_orchestration/
        └── sqlite_writer.py      # Local database persistence layer

```

---

## ⚙️ Installation & Setup

### **1. Prerequisites**

* Python 3.9+
* Grafana Server (for GCS visualization)

### **2. Clone & Install Dependencies**

```bash
git clone [https://github.com/your-org/uav-engine-digital-twin.git](https://github.com/your-org/uav-engine-digital-twin.git)
cd uav-engine-digital-twin

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required Python packages
pip install torch numpy scipy python-can cantools

```

### **3. Initialize SQLite Database**

Run the initialization script to set up the local data directory and database schema:

```bash
python3 create_db.py

```

*(Optional)* Populate sample data to verify database access:

```bash
python3 populate_db.py

```

---

## 🚀 Running the Digital Twin

Execute the live simulation execution loop:

```bash
python3 main.py

```

### **Expected Console Output:**

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
2026-08-26 15:10:00 [INFO] Iter 01 | CHT: 115.0°C (Δ 0.0) | EGT: 820.0°C (Δ 0.0) [NOMINAL]
2026-08-26 15:10:00 [INFO]        --> HEALTH: 100.0% | RUL: 1200.0 hrs | STATUS: NOMINAL

```

---

## 📊 Grafana Visualization Setup

To set up real-time monitoring on Grafana:

1. Launch Grafana (`http://127.0.0.1:3000`).
2. Navigate to **Connections** $\rightarrow$ **Data Sources** $\rightarrow$ **Add data source** $\rightarrow$ **SQLite**.
3. Set the database path to:
```text
/Users/<your-username>/path-to-repo/data/engine_telemetry.db

```


4. Query CHT & EGT actuals vs physics baselines in Grafana panels:
```sql
SELECT 
  datetime(timestamp, 'unixepoch') AS time, 
  actual_cht, 
  physics_cht,
  residual_cht
FROM uav_aero_engine_metrics 
ORDER BY id DESC

```



---

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.
