from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import requests
import base64
from ultralytics import YOLO
import supervision as sv
from datetime import datetime

# ================= CONFIG =================

ESP32_CAPTURE_URL = "http://10.13.106.254/capture"
ESP32_CONTROL_URL = "http://10.13.106.254/control?var=framesize&val={}"

FIREBASE_HOST = "https://your-project-default-rtdb.firebaseio.com/"
FIREBASE_AUTH = "YOUR_FIREBASE_AUTH"
FIREBASE_PATH = "/MVR/latest_detection.json"

PIXEL_TO_METER = 0.5
CARBON_PER_HA = 388

# ESP32 frame size mapping
RESOLUTIONS = {
    "QQVGA (160x120)": 0,
    "QVGA (320x240)": 1,
    "VGA (640x480)": 2,
    "SVGA (800x600)": 3,
    "XGA (1024x768)": 4,
    "SXGA (1280x1024)": 5,
    "UXGA (1600x1200)": 6
}

# ==========================================

app = Flask(__name__)
model = YOLO("best.pt")


# -------- Carbon Calculation --------
def calculate_carbon(area_m2):
    area_ha = area_m2 / 10000
    carbon = area_ha * CARBON_PER_HA
    co2 = carbon * 3.67
    trees = round(co2 * 1000 / 20)

    return {
        "area_m2": round(area_m2, 2),
        "area_ha": round(area_ha, 4),
        "carbon_tons": round(carbon, 2),
        "co2_tons": round(co2, 2),
        "trees": trees
    }


# -------- Fetch ESP32 Image --------
def get_esp32_frame():
    response = requests.get(ESP32_CAPTURE_URL, timeout=3)
    img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return frame


# -------- Detection Core --------
def run_detection(conf):
    frame = get_esp32_frame()
    results = model.predict(frame, conf=conf)

    detections_list = []
    total_area_pixels = 0
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    for r in results:
        detections = sv.Detections.from_ultralytics(r)
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        labels = []

        for box in r.boxes:
            bbox = box.xyxy[0].cpu().numpy()
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            area = width * height
            total_area_pixels += area

            class_id = int(box.cls[0])
            conf_score = float(box.conf[0])
            class_name = r.names[class_id]

            detections_list.append({
                "class": class_name,
                "confidence": round(conf_score * 100, 2),
                "area_pixels": int(area)
            })

            labels.append(f"{class_name} {conf_score:.2f}")

        annotated = box_annotator.annotate(image_rgb.copy(), detections)
        annotated = label_annotator.annotate(annotated, detections, labels)

    total_area_m2 = total_area_pixels * (PIXEL_TO_METER ** 2)
    carbon_data = calculate_carbon(total_area_m2)

    _, buffer = cv2.imencode(".jpg", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    img_base64 = base64.b64encode(buffer).decode()

    return {
        "image": img_base64,
        "total_detections": len(detections_list),
        "detections": detections_list,
        "carbon": carbon_data,
        "timestamp": datetime.now().isoformat()
    }


# -------- Routes --------

@app.route("/")
def index():
    return render_template("index.html", resolutions=RESOLUTIONS.keys())


@app.route("/detect", methods=["POST"])
def detect():
    confidence = float(request.json.get("confidence", 0.3))
    result = run_detection(confidence)
    return jsonify(result)


@app.route("/set_resolution", methods=["POST"])
def set_resolution():
    resolution_name = request.json.get("resolution")
    if resolution_name in RESOLUTIONS:
        value = RESOLUTIONS[resolution_name]
        requests.get(ESP32_CONTROL_URL.format(value))
        return jsonify({"status": "changed"})
    return jsonify({"status": "invalid"}), 400


@app.route("/send_to_firebase", methods=["POST"])
def send_to_firebase():
    data = request.json
    url = f"{FIREBASE_HOST}{FIREBASE_PATH}?auth={FIREBASE_AUTH}"
    response = requests.put(url, json=data)
    return jsonify({"status": response.status_code})


if __name__ == "__main__":
    app.run(debug=True)