"""
pc_inference_server.py
======================
Mac-side Flask server for Precint3D.
"""
import io
import base64
import time
import json
import requests
from flask import Flask, request, jsonify, render_template_string
from PIL import Image
import numpy as np
import torch

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from paths import default_model_path

# ─── Configuration ────────────────────────────────────────────────────────────
ESP32_IP     = "192.168.1.XXX"          # ← Change to your ESP32-S3's IP address
ESP32_PORT   = 80
MODEL_PATH   = str(default_model_path())
SERVER_PORT  = 5001

# PER-CLASS CONFIDENCE THRESHOLDS
CLASS_CONF_THRESHOLDS = {
    "spaghetti_failure": 0.40,    # Lowered from 0.55 to 0.40 to catch real spaghetti, while dodging the 0.38 false positive
    "under_extrusion": 0.15,      
    "layer_shift": 0.15,          
    "stringing": 0.10,            
    "warping": 0.20,
    "blob_or_zit": 0.15,
    "first_layer_failure": 0.20,
    "nozzle_clog": 0.15,
    "good_print": 0.50            
}
# ─────────────────────────────────────────────────────────────────────────────

# Action map
ACTION_MAP = {
    "spaghetti_failure": {"action": "🚨 PAUSE immediately",            "severity": "critical", "color": "#ff4444"},
    "warping":           {"action": "⚠️  Slow 70%, raise bed +5°C",    "severity": "warning",  "color": "#ff9900"},
    "first_layer_failure":{"action":"🚨 PAUSE — first layer failed",    "severity": "critical", "color": "#ff4444"},
    "under_extrusion":   {"action": "⚠️  Slow 80%, raise nozzle +5°C", "severity": "warning",  "color": "#ff9900"},
    "stringing":         {"action": "⚠️  Lower nozzle -3°C",           "severity": "warning",  "color": "#ff9900"},
    "blob_or_zit":       {"action": "⚠️  Slow 85%, alert",             "severity": "warning",  "color": "#ff9900"},
    "layer_shift":       {"action": "🚨 PAUSE — layer shift",          "severity": "critical", "color": "#ff4444"},
    "nozzle_clog":       {"action": "🚨 PAUSE — nozzle clog",          "severity": "critical", "color": "#ff4444"},
    "good_print":        {"action": "✅ Continue — print looks healthy", "severity": "ok",       "color": "#00cc66"},
}

app = Flask(__name__)
model = None

def load_model():
    global model
    from ultralytics import YOLO
    print(f"[MODEL] Loading {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    print("[MODEL] Ready ✅")

def fetch_from_esp32():
    url = f"http://{ESP32_IP}:{ESP32_PORT}/capture"
    print(f"[ESP32] Fetching {url}")
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.content   # raw JPEG bytes

def run_inference(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # Fetch all detections down to 1% confidence just to see what it's thinking
    results = model.predict(source=img, conf=0.01, verbose=False)
    r = results[0]

    raw_detections = []
    valid_detections = []
    keep_indices = []

    if r.boxes is not None:
        for i, box in enumerate(r.boxes):
            cls_id = int(box.cls.item())
            conf   = float(box.conf.item())
            name   = r.names[cls_id]
            
            # Deduplicate classes - only keep the highest confidence for each class to avoid UI spam
            if not any(d['class'] == name for d in raw_detections):
                raw_detections.append({"class": name, "confidence": round(conf, 3)})
            
            thresh = CLASS_CONF_THRESHOLDS.get(name, 0.30)
            if conf >= thresh:
                keep_indices.append(i)
                valid_detections.append({"class": name, "confidence": round(conf, 3)})

        # Filter the boxes so .plot() only draws the ones that passed our thresholds
        if keep_indices:
            r.boxes = r.boxes[keep_indices]
        else:
            r.boxes = r.boxes.__class__(torch.empty((0, 6)), r.boxes.orig_shape)

    raw_detections.sort(key=lambda x: -x["confidence"])
    valid_detections.sort(key=lambda x: -x["confidence"])

    annotated = r.plot()  # numpy array
    pil_img = Image.fromarray(annotated[..., ::-1])  # BGR→RGB
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    return valid_detections, raw_detections, img_b64

@app.route("/")
def index():
    return render_template_string(HTML_DASHBOARD, esp32_ip=ESP32_IP, action_map_js=json.dumps(ACTION_MAP))

@app.route("/trigger", methods=["POST"])
def trigger():
    try:
        t0 = time.time()
        image_bytes = fetch_from_esp32()
        return process_inference(image_bytes, t0)
    except requests.exceptions.ConnectionError:
        return jsonify({"success": False, "error": f"Cannot reach ESP32 at {ESP32_IP}:{ESP32_PORT}."}), 503
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/upload", methods=["POST"])
def upload():
    try:
        t0 = time.time()
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file part"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"success": False, "error": "No selected file"}), 400
        image_bytes = file.read()
        return process_inference(image_bytes, t0)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

def process_inference(image_bytes, t0):
    valid_detections, raw_detections, img_b64 = run_inference(image_bytes)

    if valid_detections:
        top = valid_detections[0]
        cls_name = top["class"]
        confidence = top["confidence"]
    else:
        cls_name = "good_print"
        confidence = 1.0

    action_info = ACTION_MAP.get(cls_name, ACTION_MAP["good_print"])
    elapsed = round(time.time() - t0, 2)

    return jsonify({
        "success":     True,
        "class":       cls_name,
        "confidence":  confidence,
        "action":      action_info["action"],
        "severity":    action_info["severity"],
        "color":       action_info["color"],
        "all_detections": valid_detections,
        "raw_detections": raw_detections,
        "image_b64":   img_b64,
        "elapsed_s":   elapsed,
    })

@app.route("/esp32_status")
def esp32_status():
    try:
        r = requests.get(f"http://{ESP32_IP}:{ESP32_PORT}/status", timeout=3)
        return jsonify({"online": True, **r.json()})
    except Exception as e:
        return jsonify({"online": False, "error": str(e)})

HTML_DASHBOARD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Precint3D — 3D Print Monitor</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg:       #0a0a0f;
    --surface:  #13131a;
    --border:   #1e1e2e;
    --accent:   #6c63ff;
    --ok:       #00cc66;
    --warn:     #ff9900;
    --crit:     #ff4444;
    --text:     #e0e0f0;
    --muted:    #666688;
  }
  body { font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; display: flex; flex-direction: column; }
  header { padding: 1.5rem 2rem; border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 1rem; }
  header h1 { font-size: 1.4rem; font-weight: 800; letter-spacing: -0.5px; }
  header h1 span { color: var(--accent); }
  .badge { font-size: 0.7rem; padding: 0.2rem 0.6rem; border-radius: 99px; background: var(--border); color: var(--muted); }
  #esp-badge { background: #1a1a2e; }
  #esp-badge.online { background: rgba(0,204,102,0.15); color: var(--ok); }
  #esp-badge.offline { background: rgba(255,68,68,0.15); color: var(--crit); }
  main { flex: 1; display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; padding: 2rem; max-width: 1200px; margin: 0 auto; width: 100%; }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 16px; padding: 1.5rem; }
  .card h2 { font-size: 0.85rem; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 1rem; }
  button.action-btn { width: 100%; padding: 1.2rem; font-size: 1.1rem; font-weight: 700; border: none; border-radius: 12px; cursor: pointer; color: white; transition: all 0.2s; letter-spacing: 0.5px; margin-bottom: 0.5rem; }
  #capture-btn { background: linear-gradient(135deg, var(--accent), #9c6cff); }
  #upload-btn { background: linear-gradient(135deg, #444, #666); }
  button.action-btn:hover:not(:disabled) { transform: translateY(-2px); box-shadow: 0 8px 32px rgba(108,99,255,0.4); }
  button.action-btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
  button.action-btn.loading { background: var(--border); }
  #file-input { display: none; }
  .or-divider { text-align: center; color: var(--muted); font-size: 0.9rem; margin: 0.5rem 0; font-weight: 600; }
  #result-box { margin-top: 1.5rem; padding: 1.5rem; border-radius: 12px; border: 2px solid var(--border); display: none; }
  #result-box.visible { display: block; }
  #result-class { font-size: 1.8rem; font-weight: 800; margin-bottom: 0.3rem; }
  #result-conf { font-size: 0.9rem; color: var(--muted); margin-bottom: 1rem; }
  #result-action { font-size: 1rem; font-weight: 600; padding: 0.8rem 1rem; border-radius: 8px; background: rgba(255,255,255,0.05); }
  #detections-list { margin-top: 1rem; }
  .det-row { display: flex; justify-content: space-between; align-items: center; padding: 0.4rem 0; border-bottom: 1px solid var(--border); font-size: 0.85rem; }
  .det-bar { height: 4px; border-radius: 2px; margin-top: 2px; }
  #preview-img { width: 100%; border-radius: 12px; display: none; }
  #preview-img.visible { display: block; }
  #preview-placeholder { width: 100%; aspect-ratio: 4/3; border-radius: 12px; background: var(--border); display: flex; align-items: center; justify-content: center; color: var(--muted); font-size: 0.9rem; }
  #timing { font-size: 0.75rem; color: var(--muted); margin-top: 0.5rem; text-align: right; }
  #error-box { margin-top: 1rem; padding: 1rem; background: rgba(255,68,68,0.1); border: 1px solid var(--crit); border-radius: 8px; color: var(--crit); font-size: 0.9rem; display: none; }
  #error-box.visible { display: block; }
  @media (max-width: 700px) { main { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<header>
  <div><h1>Precint<span>3D</span></h1></div>
  <span class="badge" id="esp-badge">ESP32 checking...</span>
  <span class="badge">YOLO11s</span>
  <span class="badge">{{ esp32_ip }}</span>
</header>
<main>
  <div>
    <div class="card">
      <h2>Analyze Image</h2>
      <button id="capture-btn" class="action-btn" onclick="captureAndAnalyze()">📸 Capture from ESP32</button>
      <div class="or-divider">— OR —</div>
      <button id="upload-btn" class="action-btn" onclick="document.getElementById('file-input').click()">📁 Upload from Mac</button>
      <input type="file" id="file-input" accept="image/*" onchange="uploadAndAnalyze(event)">
      
      <div id="error-box"></div>
      <div id="result-box">
        <div id="result-class">—</div>
        <div id="result-conf">—</div>
        <div id="result-action">—</div>
      </div>
    </div>
    <div class="card" style="margin-top:1.5rem">
      <h2>All Detections</h2>
      <div id="detections-list"><p style="color:var(--muted);font-size:0.85rem">Click capture or upload to see detections</p></div>
      <div id="timing"></div>
    </div>
  </div>
  <div class="card">
    <h2>Camera Feed</h2>
    <div id="preview-placeholder">No image yet — capture or upload</div>
    <img id="preview-img" alt="Annotated detection">
  </div>
</main>
<script>
const ACTION_MAP = {{ action_map_js | safe }};

async function checkESP32() {
  try {
    const r = await fetch('/esp32_status');
    const d = await r.json();
    const badge = document.getElementById('esp-badge');
    if (d.online) {
      badge.textContent = `ESP32 ✓ RSSI ${d.rssi} dBm`;
      badge.className = 'badge online';
    } else {
      badge.textContent = 'ESP32 offline';
      badge.className = 'badge offline';
    }
  } catch {}
}

function displayResults(data) {
  const errBox = document.getElementById('error-box');
  if (!data.success) {
    errBox.textContent = '❌ ' + data.error;
    errBox.className = 'visible';
    return;
  }
  errBox.className = 'badge';

  const img = document.getElementById('preview-img');
  const placeholder = document.getElementById('preview-placeholder');
  img.src = 'data:image/jpeg;base64,' + data.image_b64;
  img.className = 'visible';
  placeholder.style.display = 'none';

  const box = document.getElementById('result-box');
  box.style.borderColor = data.color;
  box.className = 'visible';
  document.getElementById('result-class').textContent = data.class.replace(/_/g, ' ').toUpperCase();
  document.getElementById('result-class').style.color = data.color;
  document.getElementById('result-conf').textContent = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;
  document.getElementById('result-action').textContent = data.action;
  document.getElementById('result-action').style.borderLeft = `4px solid ${data.color}`;

  const list = document.getElementById('detections-list');
  if (data.raw_detections.length === 0) {
    list.innerHTML = '<p style="color:var(--muted);font-size:0.85rem">No detections found by AI</p>';
  } else {
    list.innerHTML = data.raw_detections.map(d => {
      const color = (ACTION_MAP[d.class] || {color:'#888'}).color;
      const passed = data.all_detections.some(v => v.class === d.class && v.confidence === d.confidence);
      const opacity = passed ? '1' : '0.4';
      const statusText = passed ? '' : '<span style="font-size:10px; color:#ff4444; margin-left:5px">(ignored by threshold)</span>';
      return `<div class="det-row" style="opacity:${opacity}">
        <span>${d.class.replace(/_/g,' ')} ${statusText}</span>
        <span style="color:${color};font-weight:600">${(d.confidence*100).toFixed(1)}%</span>
      </div>
      <div class="det-bar" style="width:${(d.confidence*100).toFixed(0)}%;background:${color};opacity:${opacity}"></div>`;
    }).join('');
  }
  document.getElementById('timing').textContent = `Analysis took ${data.elapsed_s}s`;
}

function setLoading(isLoading) {
  const btnCapture = document.getElementById('capture-btn');
  const btnUpload = document.getElementById('upload-btn');
  if (isLoading) {
    btnCapture.disabled = true;
    btnUpload.disabled = true;
  } else {
    btnCapture.disabled = false;
    btnUpload.disabled = false;
    btnCapture.textContent = '📸 Capture from ESP32';
    btnUpload.textContent = '📁 Upload from Mac';
  }
}

async function captureAndAnalyze() {
  const errBox = document.getElementById('error-box');
  errBox.className = 'badge';
  setLoading(true);
  document.getElementById('capture-btn').textContent = '⏳ Capturing...';
  
  try {
    const resp = await fetch('/trigger', { method: 'POST' });
    const data = await resp.json();
    displayResults(data);
  } catch (e) {
    errBox.textContent = '❌ Server error: ' + e.message;
    errBox.className = 'visible';
  } finally {
    setLoading(false);
  }
}

async function uploadAndAnalyze(event) {
  const file = event.target.files[0];
  if (!file) return;

  const errBox = document.getElementById('error-box');
  errBox.className = 'badge';
  setLoading(true);
  document.getElementById('upload-btn').textContent = '⏳ Analyzing...';

  const formData = new FormData();
  formData.append('file', file);

  try {
    const resp = await fetch('/upload', { method: 'POST', body: formData });
    const data = await resp.json();
    displayResults(data);
  } catch (e) {
    errBox.textContent = '❌ Server error: ' + e.message;
    errBox.className = 'visible';
  } finally {
    setLoading(false);
    event.target.value = ''; // Reset input
  }
}

checkESP32();
setInterval(checkESP32, 10000);
</script>
</body>
</html>"""

if __name__ == "__main__":
    load_model()
    print(f"\n{'='*50}")
    print(f"  Precint3D Inference Server")
    print(f"  Dashboard: http://localhost:{SERVER_PORT}")
    print(f"  ESP32-S3:  http://{ESP32_IP}/capture")
    print(f"{'='*50}\n")
    app.run(host="0.0.0.0", port=SERVER_PORT, debug=False)
