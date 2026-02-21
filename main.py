import os, io, base64, time, threading
from datetime import datetime

import cv2
import numpy as np
import requests
from flask import Flask, render_template, jsonify, request, Response
from PIL import Image
from werkzeug.utils import secure_filename

# ── Optional YOLO ──────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    import supervision as sv
    YOLO_OK = True
except ImportError:
    YOLO_OK = False

# ══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════

ESP32_URL        = "http://10.13.106.254/capture"
ESP32_CONFIG_URL = "http://10.13.106.254/control?var=framesize&val={}"

FIREBASE_HOST        = "https://newcam-19ef1-default-rtdb.firebaseio.com/"
FIREBASE_AUTH        = "0njZXc3wlhf62RfoqLOlZhKNdDQCBp0NFQxRrKIB"
FIREBASE_INPUT_PATH  = "/captured_images"
FIREBASE_OUTPUT_PATH = "/MVR"

MODEL_PATH            = "best.pt"
AUTO_CONFIDENCE       = 0.1
PIXEL_TO_METER        = 0.5
AUTO_CAPTURE_INTERVAL = 10          # seconds between auto-captures
UPLOAD_FOLDER         = "uploads"
RESULTS_FOLDER        = "results"
MAX_CONTENT_LENGTH    = 16 * 1024 * 1024

RESOLUTIONS = {
    "QQVGA (160x120)":  0, "QVGA (320x240)":   1,
    "VGA (640x480)":    2, "SVGA (800x600)":   3,
    "XGA (1024x768)":   4, "SXGA (1280x1024)": 5,
    "UXGA (1600x1200)": 6,
}

# ══════════════════════════════════════════════════════════════════════════
#  APP SETUP
# ══════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
app.config["UPLOAD_FOLDER"]     = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Global state
model               = None
latest_result       = {}          # last detection result served to UI
latest_raw_frame_b64 = ""         # latest ESP32 frame (base64 JPEG)
activity_log        = []          # list of {time, msg} dicts (capped at 200)
auto_capture_on     = True
last_auto_ts        = 0

def _log(msg: str):
    entry = {"time": datetime.now().strftime("%H:%M:%S"), "msg": msg}
    activity_log.append(entry)
    if len(activity_log) > 200:
        activity_log.pop(0)
    print(f"[{entry['time']}] {msg}")

# ══════════════════════════════════════════════════════════════════════════
#  MODEL
# ══════════════════════════════════════════════════════════════════════════

def load_model():
    global model
    if not YOLO_OK:
        _log("⚠️  ultralytics not installed – detection disabled")
        return
    if not os.path.exists(MODEL_PATH):
        _log(f"⚠️  {MODEL_PATH} not found – detection disabled")
        return
    try:
        _log(f"⏳ Loading {MODEL_PATH}…")
        model = YOLO(MODEL_PATH)
        _log(f"✅ Model ready")
    except Exception as e:
        _log(f"❌ Model load error: {e}")

# ══════════════════════════════════════════════════════════════════════════
#  CARBON
# ══════════════════════════════════════════════════════════════════════════

def calculate_carbon(area_px: float) -> dict:
    area_m2      = area_px * (PIXEL_TO_METER ** 2)
    area_ha      = area_m2 / 10_000
    carbon_tons  = area_ha * 388
    co2_tons     = carbon_tons * 3.67
    trees_equiv  = round(co2_tons * 1_000 / 20)
    return {
        "area_m2":     round(area_m2,     2),
        "area_ha":     round(area_ha,     4),
        "carbon_tons": round(carbon_tons, 2),
        "co2_tons":    round(co2_tons,    2),
        "trees_equiv": trees_equiv,
    }

# ══════════════════════════════════════════════════════════════════════════
#  FIREBASE
# ══════════════════════════════════════════════════════════════════════════

def _fb(method: str, path: str, data: dict) -> bool:
    url = f"{FIREBASE_HOST}{path}.json?auth={FIREBASE_AUTH}"
    try:
        fn  = {"POST": requests.post, "PUT": requests.put,
               "PATCH": requests.patch}[method]
        r   = fn(url, json=data, timeout=10)
        return r.status_code == 200
    except Exception as e:
        _log(f"❌ Firebase {method} error: {e}")
        return False

def firebase_upload_raw(b64: str):
    ok = _fb("POST", FIREBASE_INPUT_PATH, {
        "timestamp": datetime.now().isoformat(), "image": b64})
    _log("✅ Raw → /captured_images" if ok else "❌ Raw upload failed")

def firebase_upload_result(result: dict, ann_b64: str):
    c   = result["carbon"]
    ts  = f"detection_{int(time.time()*1000)}"
    payload = {
        "timestamp": datetime.now().isoformat(),
        "detection_summary": {
            "total_mangroves": result["total"],
            "area_m2":         c["area_m2"],
            "area_hectares":   c["area_ha"],
        },
        "carbon_sequestration": {
            "carbon_stock_tons":         c["carbon_tons"],
            "co2_equivalent_tons":       c["co2_tons"],
            "trees_equivalent_per_year": c["trees_equiv"],
        },
        "annotated_image": ann_b64,
        "detections":      result["detections"],
        "parameters": {
            "confidence_threshold": result["conf"],
            "pixel_to_meter":       PIXEL_TO_METER,
        },
        "source": result.get("source", "flask_app"),
    }
    ok1 = _fb("PUT",   f"{FIREBASE_OUTPUT_PATH}/latest_detection", payload)
    ok2 = _fb("PATCH", f"{FIREBASE_OUTPUT_PATH}/history", {ts: payload})
    _log(f"✅ Results → /MVR/latest_detection" if ok1 else "❌ Result upload failed")
    if ok2: _log(f"✅ History → /MVR/history/{ts}")

# ══════════════════════════════════════════════════════════════════════════
#  DETECTION
# ══════════════════════════════════════════════════════════════════════════

def run_detection(frame_bgr: np.ndarray, conf: float, source: str = "flask") -> dict:
    """Run YOLO on frame; returns result dict with annotated_b64."""
    results      = model.predict(frame_bgr, conf=conf, verbose=False)
    det_list     = []
    total_area_px = 0
    annotated    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        sv_det  = sv.Detections.from_ultralytics(r)
        box_ann = sv.BoxAnnotator(thickness=3, color=sv.Color(r=0, g=220, b=80))
        lbl_ann = sv.LabelAnnotator(
            text_color=sv.Color(r=255, g=255, b=255),
            text_scale=0.5, text_thickness=2)
        labels  = [f"{r.names[int(cid)]} {c:.0%}"
                   for cid, c in zip(sv_det.class_id, sv_det.confidence)]
        annotated = box_ann.annotate(annotated, sv_det)
        annotated = lbl_ann.annotate(annotated, sv_det, labels)

        for i, box in enumerate(r.boxes):
            cid   = int(box.cls[0])
            bbox  = box.xyxy[0].cpu().numpy()
            w, h  = bbox[2]-bbox[0], bbox[3]-bbox[1]
            total_area_px += w * h
            det_list.append({
                "id": i+1, "class": r.names[cid],
                "confidence": round(float(box.conf[0])*100, 1),
                "bbox": {"x1":int(bbox[0]),"y1":int(bbox[1]),
                         "x2":int(bbox[2]),"y2":int(bbox[3])},
                "area_pixels": int(w*h),
            })

    ann_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    _, buf  = cv2.imencode(".jpg", ann_bgr)
    ann_b64 = base64.b64encode(buf).decode()

    return {
        "total":      len(det_list),
        "detections": det_list,
        "carbon":     calculate_carbon(total_area_px),
        "conf":       conf,
        "source":     source,
        "annotated_b64": ann_b64,
        "timestamp":  datetime.now().isoformat(),
    }

# ══════════════════════════════════════════════════════════════════════════
#  PIPELINE  (capture → detect → upload)
# ══════════════════════════════════════════════════════════════════════════

def full_pipeline(frame_bgr: np.ndarray, conf: float, source: str):
    global latest_result

    # Encode raw
    _, raw_buf = cv2.imencode(".jpg", frame_bgr)
    raw_b64    = base64.b64encode(raw_buf).decode()

    # Upload raw in background
    threading.Thread(target=firebase_upload_raw, args=(raw_b64,), daemon=True).start()

    if model is None:
        _log("⚠️  Model not loaded – skipping detection")
        return

    _log(f"🔍 Detecting (conf={conf:.2f})…")
    try:
        result = run_detection(frame_bgr, conf, source)
    except Exception as e:
        _log(f"❌ Detection error: {e}")
        return

    latest_result = result
    c = result["carbon"]
    _log(f"✅ {result['total']} mangrove(s) | "
         f"CO₂: {c['co2_tons']} t | Trees: {c['trees_equiv']:,}")

    # Upload results in background
    threading.Thread(target=firebase_upload_result,
                     args=(result, result["annotated_b64"]), daemon=True).start()

# ══════════════════════════════════════════════════════════════════════════
#  AUTO-CAPTURE BACKGROUND THREAD
# ══════════════════════════════════════════════════════════════════════════

def auto_capture_loop():
    global last_auto_ts, latest_raw_frame_b64
    _log("🤖 Auto-capture loop started")
    while True:
        time.sleep(1)
        if not auto_capture_on:
            continue
        now = time.time()
        if now - last_auto_ts < AUTO_CAPTURE_INTERVAL:
            continue
        last_auto_ts = now
        _log("📸 Auto-capture triggered")
        try:
            r = requests.get(ESP32_URL, timeout=5)
            if r.status_code != 200:
                _log(f"❌ ESP32 HTTP {r.status_code}")
                continue
            arr   = np.frombuffer(r.content, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                _log("❌ Frame decode failed")
                continue
            # Cache latest raw frame for stream endpoint
            _, raw = cv2.imencode(".jpg", frame)
            latest_raw_frame_b64 = base64.b64encode(raw).decode()

            full_pipeline(frame, AUTO_CONFIDENCE, "auto_capture")

        except Exception as e:
            _log(f"❌ Auto-capture error: {e}")

# ══════════════════════════════════════════════════════════════════════════
#  MJPEG PROXY STREAM  (proxies ESP32 JPEG stream frame by frame)
# ══════════════════════════════════════════════════════════════════════════

def esp32_frame_generator():
    while True:
        try:
            r = requests.get(ESP32_URL, timeout=3)
            if r.status_code == 200:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" +
                       r.content + b"\r\n")
        except Exception:
            pass
        time.sleep(0.15)

# ══════════════════════════════════════════════════════════════════════════
#  FLASK ROUTES
# ══════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html",
                           auto_capture=auto_capture_on,
                           auto_conf=AUTO_CONFIDENCE,
                           resolutions=list(RESOLUTIONS.keys()),
                           capture_interval=AUTO_CAPTURE_INTERVAL)

# ── Live stream proxy ───────────────────────────────────────────────────
@app.route("/stream")
def stream():
    return Response(esp32_frame_generator(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# ── Manual capture ──────────────────────────────────────────────────────
@app.route("/capture_now", methods=["POST"])
def capture_now():
    conf = float(request.json.get("conf", AUTO_CONFIDENCE))
    try:
        r = requests.get(ESP32_URL, timeout=5)
        if r.status_code != 200:
            return jsonify({"success": False, "error": f"ESP32 HTTP {r.status_code}"})
        arr   = np.frombuffer(r.content, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"success": False, "error": "Frame decode failed"})
        threading.Thread(target=full_pipeline,
                         args=(frame, conf, "manual"), daemon=True).start()
        return jsonify({"success": True, "msg": "Pipeline started"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# ── Manual file upload ──────────────────────────────────────────────────
@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file"})
    f    = request.files["file"]
    conf = float(request.form.get("conf", 0.3))
    name = secure_filename(f.filename)
    path = os.path.join(UPLOAD_FOLDER,
                        f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{name}")
    f.save(path)
    frame = cv2.imread(path)
    if frame is None:
        return jsonify({"success": False, "error": "Cannot read image"})
    threading.Thread(target=full_pipeline,
                     args=(frame, conf, "manual_upload"), daemon=True).start()
    return jsonify({"success": True, "msg": "Processing…"})

# ── Latest result ────────────────────────────────────────────────────────
@app.route("/latest")
def latest():
    if not latest_result:
        return jsonify({"success": False, "msg": "No detection yet"})
    return jsonify({"success": True, "data": latest_result})

# ── Activity log ─────────────────────────────────────────────────────────
@app.route("/logs")
def logs():
    return jsonify(activity_log[-50:])

# ── Toggle auto-capture ───────────────────────────────────────────────────
@app.route("/toggle_auto", methods=["POST"])
def toggle_auto():
    global auto_capture_on
    auto_capture_on = not auto_capture_on
    _log(f"🔄 Auto-capture {'ON' if auto_capture_on else 'OFF'}")
    return jsonify({"auto_capture": auto_capture_on})

# ── Change ESP32 resolution ───────────────────────────────────────────────
@app.route("/set_resolution", methods=["POST"])
def set_resolution():
    res_name = request.json.get("resolution", "QVGA (320x240)")
    val      = RESOLUTIONS.get(res_name, 1)
    try:
        r = requests.get(ESP32_CONFIG_URL.format(val), timeout=3)
        _log(f"📐 Resolution → {res_name}")
        return jsonify({"success": r.status_code == 200})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# ── Health ────────────────────────────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify({
        "model_loaded":   model is not None,
        "auto_capture":   auto_capture_on,
        "auto_conf":      AUTO_CONFIDENCE,
        "yolo_available": YOLO_OK,
    })

# ══════════════════════════════════════════════════════════════════════════
#  STARTUP
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    load_model()
    threading.Thread(target=auto_capture_loop, daemon=True).start()

    port = int(os.environ.get("PORT", 5100))
    print("=" * 60)
    print("🌳 MANGROVE DETECTION SYSTEM")
    print("=" * 60)
    print(f"🌐  http://localhost:{port}")
    print(f"📷  ESP32  → {ESP32_URL}")
    print(f"🔥  Firebase → {FIREBASE_HOST}")
    print(f"🤖  Auto-capture every {AUTO_CAPTURE_INTERVAL}s  (conf={AUTO_CONFIDENCE})")
    print("=" * 60)
    app.run(host="0.0.0.0", port=port, debug=False)