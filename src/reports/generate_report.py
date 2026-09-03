import os
import sys
import psycopg
import pandas as pd
from datetime import datetime
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = BASE_DIR / "reports"
OUTPUT_DIR.mkdir(exist_ok=True)

# Inside src/reports/generate_report.py
DB_URL = os.getenv("DATABASE_URL","postgresql://uav_user:uav_password@127.0.0.1:5432/uav_telemetry")

# 1. Pull data based on TIME WINDOW (e.g. last 30 minutes) or increase row limit
def fetch_mission_data(minutes_back: int = 30):
    """Queries engine telemetry recorded within the specified flight duration."""
    query = """
        SELECT created_at, rpm, map_kpa, actual_cht, physics_cht, residual_cht,
               actual_egt, physics_egt, residual_egt, actual_oil_pressure,
               health_index_pct, rul_hours, maintenance_urgency, anomaly_flag
        FROM uav_aero_engine_metrics
        WHERE created_at >= NOW() - (INTERVAL '1 minute' * %s)
        ORDER BY created_at DESC;
    """
    with psycopg.connect(DB_URL) as conn:
        df = pd.read_sql_query(query, conn, params=(minutes_back,))
    return df

def generate_pdf_report(minutes_back: int = 30, display_rows: int = 30):
    df = fetch_mission_data(minutes_back=minutes_back)
    if df.empty:
        print("[!] No metrics found in database for the requested time window.")
        return None

    # Compute Summary Statistics over the FULL pulled dataset
    total_frames = len(df)
    avg_health = df["health_index_pct"].mean()
    min_health = df["health_index_pct"].min()
    est_rul = df["rul_hours"].iloc[0]
    anomaly_count = df["anomaly_flag"].sum()
    max_cht = df["actual_cht"].max()
    max_egt = df["actual_egt"].max()
    avg_oil = df["actual_oil_pressure"].mean() if "actual_oil_pressure" in df else 4.0
    overall_status = df["maintenance_urgency"].iloc[0]

    report_filename = OUTPUT_DIR / f"Mission_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    doc = SimpleDocTemplate(str(report_filename), pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=18, leading=22, textColor=colors.HexColor("#1e293b"))
    subtitle_style = ParagraphStyle('SubTitleStyle', parent=styles['Normal'], fontSize=9, leading=12, textColor=colors.HexColor("#64748b"))
    section_style = ParagraphStyle('SectionStyle', parent=styles['Heading2'], fontSize=12, leading=16, textColor=colors.HexColor("#0f172a"))
    cell_bold = ParagraphStyle('CellBold', parent=styles['Normal'], fontSize=8, leading=10, fontName="Helvetica-Bold")
    cell_norm = ParagraphStyle('CellNorm', parent=styles['Normal'], fontSize=8, leading=10)

    elements = []

    # Header
    elements.append(Paragraph("<b>ROTAX 914 DIGITAL TWIN — POST-FLIGHT DIAGNOSTIC REPORT</b>", title_style))
    elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC | Mission Records Evaluated: {total_frames} (Last {minutes_back} Mins)", subtitle_style))
    elements.append(Spacer(1, 8))
    elements.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#0284c7"), spaceAfter=12))

    # Executive Summary
    status_color = "#ef4444" if anomaly_count > 0 or overall_status == "CRITICAL" else "#10b981"
    
    summary_data = [
        [Paragraph("Overall Engine Status", cell_bold), Paragraph(f"<font color='{status_color}'><b>{overall_status}</b></font>", cell_norm),
         Paragraph("Total Anomalies Flagged", cell_bold), Paragraph(str(anomaly_count), cell_norm)],
        [Paragraph("Average Health Index", cell_bold), Paragraph(f"{avg_health:.1f}% (Min: {min_health:.1f}%)", cell_norm),
         Paragraph("Estimated RUL", cell_bold), Paragraph(f"{est_rul:.1f} Hours", cell_norm)],
        [Paragraph("Peak Cylinder Head Temp", cell_bold), Paragraph(f"{max_cht:.1f} °C", cell_norm),
         Paragraph("Peak Exhaust Gas Temp", cell_bold), Paragraph(f"{max_egt:.1f} °C", cell_norm)],
        [Paragraph("Average Oil Pressure", cell_bold), Paragraph(f"{avg_oil:.2f} bar", cell_norm),
         Paragraph("Total Flight Packets", cell_bold), Paragraph(str(total_frames), cell_norm)]
    ]

    t_summary = Table(summary_data, colWidths=[130, 140, 130, 140])
    t_summary.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ('PADDING', (0,0), (-1,-1), 5),
    ]))
    
    elements.append(Paragraph("<b>Executive Health Summary</b>", section_style))
    elements.append(Spacer(1, 4))
    elements.append(t_summary)
    elements.append(Spacer(1, 12))

    # Detailed Telemetry Sample Table
    elements.append(Paragraph(f"<b>Telemetry Log Sample (Showing {min(display_rows, total_frames)} of {total_frames} Packets)</b>", section_style))
    elements.append(Spacer(1, 4))

    telemetry_rows = [
        [Paragraph("Time", cell_bold), Paragraph("RPM", cell_bold), Paragraph("Act CHT", cell_bold),
         Paragraph("Phys CHT", cell_bold), Paragraph("Δ CHT", cell_bold), Paragraph("Act EGT", cell_bold),
         Paragraph("Phys EGT", cell_bold), Paragraph("Δ EGT", cell_bold)]
    ]

    # 2. INCREASED ROW COUNT FOR PRINTED TABLE
    for _, row in df.head(display_rows).iterrows():
        ts = row['created_at'].strftime('%H:%M:%S') if pd.notnull(row['created_at']) else "N/A"
        telemetry_rows.append([
            Paragraph(ts, cell_norm),
            Paragraph(f"{row['rpm']:.0f}", cell_norm),
            Paragraph(f"{row['actual_cht']:.1f}", cell_norm),
            Paragraph(f"{row['physics_cht']:.1f}", cell_norm),
            Paragraph(f"{row['residual_cht']:.2f}", cell_norm),
            Paragraph(f"{row['actual_egt']:.1f}", cell_norm),
            Paragraph(f"{row['physics_egt']:.1f}", cell_norm),
            Paragraph(f"{row['residual_egt']:.2f}", cell_norm),
        ])

    t_telemetry = Table(telemetry_rows, colWidths=[60, 55, 60, 60, 60, 60, 60, 65])
    t_telemetry.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e293b")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f1f5f9")]),
        ('PADDING', (0,0), (-1,-1), 4),
    ]))

    elements.append(t_telemetry)

    doc.build(elements)
    print(f"[+] Post-Flight Report Generated: {report_filename} ({total_frames} records evaluated)")
    return report_filename

if __name__ == "__main__":
    generate_pdf_report(minutes_back=30, display_rows=30)