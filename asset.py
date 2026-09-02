import os
from pathlib import Path
import subprocess
from PIL import Image, ImageDraw

def create_assets():
    assets_dir = Path("assets")
    assets_dir.mkdir(exist_ok=True)
    iconset_dir = assets_dir / "app_icon.iconset"
    iconset_dir.mkdir(exist_ok=True)

    print("[+] Generating base high-res icon PNG...")
    # 1. Base App Icon (512x512)
    img = Image.new("RGBA", (512, 512), (15, 23, 42, 255))
    draw = ImageDraw.Draw(img)
    draw.ellipse([40, 40, 472, 472], outline="#0284c7", width=12)
    draw.ellipse([70, 70, 442, 442], outline="#38bdf8", width=4)
    draw.ellipse([226, 226, 286, 286], fill="#38bdf8")
    draw.line([100, 256, 412, 256], fill="#f8fafc", width=16)
    draw.line([256, 100, 256, 412], fill="#f8fafc", width=16)
    draw.rectangle([216, 160, 296, 210], fill="#3b82f6")
    draw.rectangle([216, 302, 296, 352], fill="#3b82f6")
    
    icon_png = assets_dir / "app_icon.png"
    img.save(icon_png)

    # 2. Windows .ico
    ico_path = assets_dir / "app_icon.ico"
    img.save(ico_path, format="ICO", sizes=[(16,16), (32,32), (48,48), (64,64), (128,128), (256,256)])
    print(f"[+] Created {ico_path}")

    # 3. macOS .icns via sips and iconutil
    sizes = [
        (16, "icon_16x16.png"), (32, "icon_16x16@2x.png"),
        (32, "icon_32x32.png"), (64, "icon_32x32@2x.png"),
        (128, "icon_128x128.png"), (256, "icon_128x128@2x.png"),
        (256, "icon_256x256.png"), (512, "icon_256x256@2x.png"),
        (512, "icon_512x512.png"), (1024, "icon_512x512@2x.png")
    ]
    
    for size, filename in sizes:
        subprocess.run([
            "sips", "-z", str(size), str(size),
            str(icon_png), "--out", str(iconset_dir / filename)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    icns_path = assets_dir / "app_icon.icns"
    subprocess.run(["iconutil", "-c", "icns", str(iconset_dir), "-o", str(icns_path)])
    
    # Clean up temporary iconset directory
    for f in iconset_dir.glob("*.png"):
        f.unlink()
    iconset_dir.rmdir()
    print(f"[+] Created {icns_path}")

    # 4. Splash Screen (600x400)
    print("[+] Generating splash screen...")
    splash = Image.new("RGBA", (600, 400), (15, 23, 42, 255))
    s_draw = ImageDraw.Draw(splash)
    s_draw.rectangle([0, 385, 600, 400], fill="#0284c7")
    s_draw.rectangle([0, 380, 600, 385], fill="#38bdf8")

    for i in range(0, 600, 40):
        s_draw.line([i, 0, i, 400], fill="#1e293b", width=1)
    for j in range(0, 400, 40):
        s_draw.line([0, j, 600, j], fill="#1e293b", width=1)

    s_draw.text((40, 140), "UAV DIGITAL TWIN", fill="#f8fafc")
    s_draw.text((40, 180), "ROTAX 914 GROUND CONTROL STATION", fill="#0284c7")
    s_draw.text((40, 220), "v1.0.0 - Telemetry & Diagnostics", fill="#94a3b8")

    splash_path = assets_dir / "splash.png"
    splash.save(splash_path)
    print(f"[+] Created {splash_path}")

    print("\n[=] All assets generated successfully in assets/ directory!")

if __name__ == "__main__":
    create_assets()