"""
predict_and_act.py
==================
Run YOLO11 inference on a printer image → send action command to printer.

Usage:
  python predict_and_act.py path/to/frame.jpg
  python predict_and_act.py path/to/frame.jpg --backend octoprint
  python predict_and_act.py path/to/frame.jpg --backend serial
  python predict_and_act.py path/to/frame.jpg --show          # visualise detections

Printer backends:
  console    — dry run, prints commands only (default for testing)
  octoprint  — OctoPrint REST API
  serial     — USB serial (Marlin / Klipper)
  moonraker  — Klipper / Moonraker REST API
"""

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from paths import default_model_path

# ─── Printer backend configuration ───────────────────────────────────────────
BACKEND      = "console"           # override via --backend flag

OCTO_URL     = "http://octopi.local"
OCTO_KEY     = "YOUR_OCTOPRINT_API_KEY"

SERIAL_PORT  = "/dev/tty.usbmodem1101"   # run: ls /dev/tty.usb* to find yours
SERIAL_BAUD  = 115200

MOON_URL     = "http://klipper.local"

# ─── Model path ──────────────────────────────────────────────────────────────
MODEL_PATH   = default_model_path()

# ─── Detection thresholds ────────────────────────────────────────────────────
CONF_THRESH  = 0.40   # minimum confidence to act
IOU_THRESH   = 0.45

# ─── Action map: class name → (description, gcode, octoprint_cmd, severity) ─
#   severity: 0=info, 1=warning, 2=critical
ACTION_MAP = {
    "spaghetti_failure": {
        "description": "🚨 PAUSE immediately — spaghetti detected",
        "gcode":       ["M0"],
        "octoprint":   "pause",
        "severity":    2,
    },
    "warping": {
        "description": "⚠️  Slow 70%, raise bed +5 °C, alert",
        "gcode":       ["M220 S70", "M140 S{{bed+5}}"],
        "octoprint":   None,
        "severity":    1,
    },
    "first_layer_failure": {
        "description": "🚨 PAUSE — first layer failed, alert operator",
        "gcode":       ["M0"],
        "octoprint":   "pause",
        "severity":    2,
    },
    "under_extrusion": {
        "description": "⚠️  Slow 80%, raise nozzle +5 °C",
        "gcode":       ["M220 S80", "M104 S{{nozzle+5}}"],
        "octoprint":   None,
        "severity":    1,
    },
    "stringing": {
        "description": "⚠️  Lower nozzle -3 °C, tune retraction later",
        "gcode":       ["M104 S{{nozzle-3}}"],
        "octoprint":   None,
        "severity":    1,
    },
    "blob_or_zit": {
        "description": "⚠️  Slow 85%, alert",
        "gcode":       ["M220 S85"],
        "octoprint":   None,
        "severity":    1,
    },
    "layer_shift": {
        "description": "🚨 PAUSE — mechanical failure (layer shift)",
        "gcode":       ["M0"],
        "octoprint":   "pause",
        "severity":    2,
    },
    "nozzle_clog": {
        "description": "🚨 PAUSE — nozzle clog, run recovery",
        "gcode":       ["M0"],
        "octoprint":   "pause",
        "severity":    2,
    },
    "good_print": {
        "description": "✅ CONTINUE — print looks healthy",
        "gcode":       [],
        "octoprint":   None,
        "severity":    0,
    },
}

# ─── Priority: if multiple classes detected, act on highest severity ──────────
def top_detection(results):
    """Return (class_name, confidence) of the highest-severity, highest-conf detection."""
    best = None
    best_sev = -1
    best_conf = 0.0

    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_id  = int(box.cls.item())
            conf    = float(box.conf.item())
            name    = r.names[cls_id]
            if conf < CONF_THRESH:
                continue
            action = ACTION_MAP.get(name, ACTION_MAP["good_print"])
            sev    = action["severity"]
            if sev > best_sev or (sev == best_sev and conf > best_conf):
                best      = name
                best_sev  = sev
                best_conf = conf

    if best is None:
        return "good_print", 1.0
    return best, best_conf

# ─── Printer backends ─────────────────────────────────────────────────────────

def send_console(action):
    for g in action["gcode"]:
        print(f"  [GCODE]   {g}")

def send_octoprint(action):
    try:
        import requests
        h = {"X-Api-Key": OCTO_KEY, "Content-Type": "application/json"}
        if action["octoprint"] == "pause":
            r = requests.post(f"{OCTO_URL}/api/job",
                              json={"command": "pause", "action": "pause"},
                              headers=h, timeout=5)
            print(f"  [OCTO] pause → HTTP {r.status_code}")
        for g in action["gcode"]:
            r = requests.post(f"{OCTO_URL}/api/printer/command",
                              json={"command": g}, headers=h, timeout=5)
            print(f"  [OCTO] {g} → HTTP {r.status_code}")
    except Exception as e:
        print(f"  [OCTO ERROR] {e}")

def send_serial(action):
    try:
        import serial, time
        with serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=2) as ser:
            time.sleep(2)
            for g in action["gcode"]:
                ser.write((g + "\n").encode())
                print(f"  [SERIAL] sent: {g}")
                time.sleep(0.2)
    except Exception as e:
        print(f"  [SERIAL ERROR] {e}")

def send_moonraker(action):
    try:
        import requests
        if action["octoprint"] == "pause":
            r = requests.post(f"{MOON_URL}/printer/print/pause", timeout=5)
            print(f"  [MOON] pause → HTTP {r.status_code}")
        for g in action["gcode"]:
            r = requests.post(f"{MOON_URL}/printer/gcode/script",
                              json={"script": g}, timeout=5)
            print(f"  [MOON] {g} → HTTP {r.status_code}")
    except Exception as e:
        print(f"  [MOON ERROR] {e}")

BACKENDS = {
    "console":   send_console,
    "octoprint": send_octoprint,
    "serial":    send_serial,
    "moonraker": send_moonraker,
}

# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="3D Print Failure Detector")
    parser.add_argument("image", help="Path to printer frame image")
    parser.add_argument("--backend", default=BACKEND,
                        choices=["console", "octoprint", "serial", "moonraker"])
    parser.add_argument("--show", action="store_true",
                        help="Show detection visualisation")
    parser.add_argument("--model", default=str(MODEL_PATH))
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        print("  Run:  python train_yolo.py  first.")
        sys.exit(1)

    from ultralytics import YOLO
    model = YOLO(str(model_path))

    print(f"\n[PREDICT] {args.image}")
    results = model.predict(
        source  = args.image,
        conf    = CONF_THRESH,
        iou     = IOU_THRESH,
        verbose = False,
        show    = args.show,
    )

    # Print all detections
    print("\n  All detections:")
    has_det = False
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        for box in r.boxes:
            cls_name = r.names[int(box.cls.item())]
            conf     = float(box.conf.item())
            bar = "█" * int(conf * 20)
            print(f"    {cls_name:<25} {conf:5.1%}  {bar}")
            has_det = True
    if not has_det:
        print("    (no detections above threshold)")

    # Pick highest-priority detection and act
    cls_name, conf = top_detection(results)
    action = ACTION_MAP.get(cls_name, ACTION_MAP["good_print"])
    sev_label = ["INFO", "WARNING", "CRITICAL"][action["severity"]]

    print(f"\n  ── Action Decision ──────────────────────────")
    print(f"  Detected : {cls_name}  ({conf:.1%})")
    print(f"  Severity : {sev_label}")
    print(f"  Action   : {action['description']}")

    send_fn = BACKENDS.get(args.backend, send_console)
    send_fn(action)

    # Save annotated image
    for r in results:
        out_path = Path("detection_result.jpg")
        r.save(filename=str(out_path))
        print(f"\n  Annotated image saved → {out_path.resolve()}")
        break

if __name__ == "__main__":
    main()
