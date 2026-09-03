import os
import sys
import time
import subprocess
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="UAV Digital Twin - Master Launchpad",
    page_icon="🎛️",
    layout="wide"
)

# 1. Resolve project root cleanly regardless of file location
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent if CURRENT_DIR.name == "src" else CURRENT_DIR

CONSUMER_PATH = BASE_DIR / "src" / "streaming" / "kafka_consumer.py"
PRODUCER_PATH = BASE_DIR / "src" / "can_bus" / "telemetry_producer.py"
RETRAIN_PATH = BASE_DIR / "src" / "ai_pipeline" / "retrain_pipeline.py"
REPORT_PATH = BASE_DIR / "src" / "reports" / "generate_report.py"

# 2. Configure Environment Dictionary with explicit PYTHONPATH pointing to BASE_DIR
ENV = os.environ.copy()
ENV["PYTHONPATH"] = str(BASE_DIR)

DB_URL = os.getenv("DATABASE_URL", "postgresql://uav_user:uav_password@127.0.0.1:5432/uav_telemetry")

# Process state handling
if "consumer_proc" not in st.session_state:
    st.session_state.consumer_proc = None
if "producer_proc" not in st.session_state:
    st.session_state.producer_proc = None

st.title("🎛️ UAV Digital Twin — Master Launchpad")
st.caption("Single-interface execution, ISA altitude flight controls, and PINN model management.")

st.divider()

# ==============================================================================
# SECTION 1: BACKGROUND PIPELINE MANAGER
# ==============================================================================
st.subheader("⚡ Background Pipeline Manager")

col_proc1, col_proc2 = st.columns(2)

with col_proc1:
    st.markdown("### 📥 Telemetry Consumer & AI Twin")
    consumer_active = st.session_state.consumer_proc is not None and st.session_state.consumer_proc.poll() is None
    
    if consumer_active:
        st.success("🟢 Consumer Engine RUNNING")
        if st.button("⏹️ Stop Consumer Worker", key="stop_consumer", use_container_width=True):
            st.session_state.consumer_proc.terminate()
            st.session_state.consumer_proc = None
            st.rerun()
    else:
        st.error("🔴 Consumer Engine STOPPED")
        if st.button("▶️ Start Consumer Worker", key="start_consumer", use_container_width=True):
            if not CONSUMER_PATH.exists():
                st.error(f"File not found: {CONSUMER_PATH}")
            else:
                st.session_state.consumer_proc = subprocess.Popen(
                    [sys.executable, str(CONSUMER_PATH)],
                    cwd=str(BASE_DIR),
                    env=ENV
                )
                time.sleep(1)
                st.rerun()

with col_proc2:
    st.markdown("### 📡 Flight Telemetry Producer")
    producer_active = st.session_state.producer_proc is not None and st.session_state.producer_proc.poll() is None
    
    if producer_active:
        st.success("🟢 Flight Producer STREAMING")
        if st.button("⏹️ Stop Producer Stream", key="stop_producer", use_container_width=True):
            st.session_state.producer_proc.terminate()
            st.session_state.producer_proc = None
            st.rerun()
    else:
        st.error("🔴 Flight Producer STOPPED")
        if st.button("▶️ Start Producer Stream", key="start_producer", use_container_width=True):
            if not PRODUCER_PATH.exists():
                st.error(f"File not found: {PRODUCER_PATH}")
            else:
                st.session_state.producer_proc = subprocess.Popen(
                    [sys.executable, str(PRODUCER_PATH)],
                    cwd=str(BASE_DIR),
                    env=ENV
                )
                time.sleep(1)
                st.rerun()

st.divider()

# ==============================================================================
# SECTION 2: LIVE METRICS HUD
# ==============================================================================
st.subheader("📊 Live Telemetry & Digital Twin Metrics")

@st.fragment(run_every=2.0)
def render_live_telemetry():
    try:
        df = pd.read_sql_query(
            """
            SELECT id,
                   timestamp, 
                   COALESCE(actual_cht, 0.0) as actual_cht, 
                   COALESCE(physics_cht, 0.0) as physics_cht, 
                   COALESCE(residual_cht, 0.0) as residual_cht, 
                   COALESCE(actual_egt, 0.0) as actual_egt, 
                   COALESCE(physics_egt, 0.0) as physics_egt, 
                   COALESCE(residual_egt, 0.0) as residual_egt, 
                   COALESCE(health_index_pct, 100.0) as health_index_pct, 
                   COALESCE(rul_hours, 1200.0) as rul_hours, 
                   COALESCE(maintenance_urgency, 'NOMINAL') as maintenance_urgency, 
                   COALESCE(altitude_m, 0.0) as altitude_m 
            FROM uav_aero_engine_metrics 
            ORDER BY id DESC LIMIT 60;
            """,
            con=DB_URL
        )
        if not df.empty:
            df = df.iloc[::-1]  # Chronological order for plots
            latest = df.iloc[-1]
            
            def safe_float(val, default=0.0):
                try:
                    return float(val) if pd.notnull(val) else default
                except (ValueError, TypeError):
                    return default

            health_val = safe_float(latest.get('health_index_pct'), 100.0)
            urgency_val = str(latest.get('maintenance_urgency', 'NOMINAL'))
            alt_val = safe_float(latest.get('altitude_m'), 0.0)
            rul_val = safe_float(latest.get('rul_hours'), 1200.0)

            # Top-level metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Engine Health", f"{health_val:.1f}%")
            m2.metric("Urgency Level", urgency_val)
            m3.metric("Live Altitude", f"{alt_val:.1f} m")
            m4.metric("Remaining Life", f"{rul_val:.1f} Hours")

            # Side-by-side interactive charts
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                st.markdown("**Thermal Tracking (Actual vs Physics CHT °C)**")
                st.line_chart(
                    df, 
                    y=["actual_cht", "physics_cht"], 
                    height=250
                )
                
            with chart_col2:
                st.markdown("**Flight Profile (Altitude m)**")
                st.area_chart(
                    df, 
                    y="altitude_m", 
                    height=250, 
                    color="#3b82f6"
                )
                
        else:
            st.info("No data in database yet. Click 'Start Consumer Worker' and 'Start Producer Stream' above!")
    except Exception as e:
        st.warning(f"Database Read Status: {e}")

render_live_telemetry()

st.divider()

# ==============================================================================
# SECTION 3: MLOPS & RETRAINING
# ==============================================================================
st.subheader("🤖 MLOps Model Management & Diagnostic Reports")

col_ml1, col_ml2 = st.columns(2)

with col_ml1:
    st.markdown("### 🔄 PINN Retraining Loop")
    if st.button("🔄 Trigger PINN Model Retraining Pipeline", use_container_width=True, type="primary"):
        if not RETRAIN_PATH.exists():
            st.error(f"Retrain script not found: {RETRAIN_PATH}")
        else:
            try:
                proc = subprocess.Popen(
                    [sys.executable, "-m", "src.ai_pipeline.retrain_pipeline"],
                    cwd=str(BASE_DIR),
                    env=ENV,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                st.success(f"PINN retraining worker launched in background (PID: {proc.pid})")
                st.toast("Retraining initiated successfully!", icon="🚀")
            except Exception as e:
                st.error(f"Retraining execution error: {e}")

with col_ml2:
    st.markdown("### 📄 Diagnostic Reporting")
    if st.button("📄 Generate PDF Post-Flight Summary", use_container_width=True):
        if not REPORT_PATH.exists():
            st.error(f"Report script not found: {REPORT_PATH}")
        else:
            with st.spinner("Generating diagnostic PDF summary from database records..."):
                try:
                    result = subprocess.run(
                        [sys.executable, str(REPORT_PATH)],
                        capture_output=True, text=True, check=True, cwd=str(BASE_DIR), env=ENV
                    )
                    st.success("PDF Diagnostic Summary exported successfully!")
                    st.code(result.stdout, language="text")
                except Exception as e:
                    st.error(f"Report generation error: {e}")