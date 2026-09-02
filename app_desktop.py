import sys
import subprocess
import time
from pathlib import Path

from PyQt6.QtCore import QUrl, Qt
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import QApplication, QMainWindow, QTabWidget, QSplashScreen, QProgressBar
from PyQt6.QtWebEngineWidgets import QWebEngineView

# Resolve asset paths (works both in development and inside PyInstaller bundle)
BASE_DIR = Path(__file__).resolve().parent
ICON_PATH = str(BASE_DIR / "assets" / "app_icon.png")
SPLASH_PATH = str(BASE_DIR / "assets" / "splash.png")


class DigitalTwinDesktopApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UAV Digital Twin — Ground Control Station")
        self.setGeometry(100, 100, 1440, 900)
        
        # Set Window Icon
        if Path(ICON_PATH).exists():
            self.setWindowIcon(QIcon(ICON_PATH))

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
    app = QApplication(sys.argv)
    
    # Set Application Icon
    if Path(ICON_PATH).exists():
        app.setWindowIcon(QIcon(ICON_PATH))

    # 1. Display Splash Screen
    splash = None
    if Path(SPLASH_PATH).exists():
        pixmap = QPixmap(SPLASH_PATH)
        splash = QSplashScreen(pixmap, Qt.WindowType.WindowStaysOnTopHint)
        splash.show()
        splash.showMessage(
            "Initializing Docker Subsystems & Telemetry Engine...",
            Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter,
            Qt.GlobalColor.white
        )
        app.processEvents()

    # 2. Launch Background Docker Services
    try:
        subprocess.Popen(["docker-compose", "up", "-d"])
    except Exception as e:
        print(f"[!] Warning launching docker-compose: {e}")

    # Allow containerized services to warm up
    for i in range(5):
        time.sleep(1)
        if splash:
            splash.showMessage(
                f"Starting services... ({5 - i}s)",
                Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter,
                Qt.GlobalColor.white
            )
            app.processEvents()

    # 3. Launch Main Window and Close Splash
    main_window = DigitalTwinDesktopApp()
    main_window.show()

    if splash:
        splash.finish(main_window)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()