import sys
import subprocess
import time
from PyQt6.QtCore import QUrl
from PyQt6.QtWidgets import QApplication, QMainWindow, QTabWidget
from PyQt6.QtWebEngineWidgets import QWebEngineView


class DigitalTwinDesktopApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UAV Digital Twin — Ground Control Station")
        self.setGeometry(100, 100, 1440, 900)

        # Tab bar for embedded dashboards
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Streamlit Tab
        self.streamlit_view = QWebEngineView()
        self.streamlit_view.setUrl(QUrl("http://localhost:8501"))
        self.tabs.addTab(self.streamlit_view, "Streamlit Launchpad")

        # Grafana Tab
        self.grafana_view = QWebEngineView()
        self.grafana_view.setUrl(QUrl("http://localhost:3000"))
        self.tabs.addTab(self.grafana_view, "Grafana Analytics")


def main():
    # Start backend services using Docker Compose
    print("[+] Launching background services...")
    subprocess.Popen(["docker-compose", "up", "-d"])
    
    # Allow background services a few seconds to warm up
    time.sleep(3)

    app = QApplication(sys.argv)
    window = DigitalTwinDesktopApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()