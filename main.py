import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import base64
import io
import os
from datetime import datetime

import cv2
import numpy as np
import requests
from PIL import Image, ImageTk

try:
    from ultralytics import YOLO
    import supervision as sv
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("ultralytics / supervision not installed – detection disabled")

ESP32_URL = "http://10.13.106.254/capture"
ESP32_CONFIG_URL = "http://10.13.106.254/control?var=framesize&val={}"

FIREBASE_HOST = "https://newcam-19ef1-default-rtdb.firebaseio.com/"
FIREBASE_AUTH = "0njZXc3wlhf62RfoqLOlZhKNdDQCBp0NFQxRrKIB"
FIREBASE_INPUT_PATH = "/captured_images"
FIREBASE_OUTPUT_PATH = "/MVR"

MODEL_PATH = "best.pt"
DEFAULT_CONFIDENCE = 0.1
PIXEL_TO_METER = 0.5
CAPTURE_INTERVAL_MS = 10000
STREAM_REFRESH_MS = 150

RESOLUTIONS = {
    "QQVGA (160×120)": 0,
    "QVGA  (320×240)": 1,
    "VGA   (640×480)": 2,
    "SVGA  (800×600)": 3,
    "XGA   (1024×768)": 4,
    "SXGA  (1280×1024)": 5,
    "UXGA  (1600×1200)": 6,
}

def calculate_carbon(total_area_m2: float) -> dict:
    area_ha = total_area_m2 / 10000
    carbon_tons = area_ha * 388
    co2_tons = carbon_tons * 3.67
    trees_equiv = round(co2_tons * 1000 / 20)
    return {
        "area_m2": round(total_area_m2, 2),
        "area_ha": round(area_ha, 4),
        "carbon_tons": round(carbon_tons, 2),
        "co2_tons": round(co2_tons, 2),
        "trees_equiv": trees_equiv,
    }

def firebase_post(path: str, data: dict, log=print) -> bool:
    url = f"{FIREBASE_HOST}{path}.json?auth={FIREBASE_AUTH}"
    try:
        r = requests.post(url, json=data, timeout=10)
        return r.status_code == 200
    except Exception as e:
        log(f"Firebase POST error: {e}")
        return False

def firebase_put(path: str, data: dict, log=print) -> bool:
    url = f"{FIREBASE_HOST}{path}.json?auth={FIREBASE_AUTH}"
    try:
        r = requests.put(url, json=data, timeout=10)
        return r.status_code == 200
    except Exception as e:
        log(f"Firebase PUT error: {e}")
        return False

def firebase_patch(path: str, data: dict, log=print) -> bool:
    url = f"{FIREBASE_HOST}{path}.json?auth={FIREBASE_AUTH}"
    try:
        r = requests.patch(url, json=data, timeout=10)
        return r.status_code == 200
    except Exception as e:
        log(f"Firebase PATCH error: {e}")
        return False

def upload_raw_image(img_b64: str, log=print) -> bool:
    payload = {
        "timestamp": datetime.now().isoformat(),
        "image": img_b64,
    }
    ok = firebase_post(FIREBASE_INPUT_PATH, payload, log)
    if ok:
        log("Raw image uploaded")
    return ok

def upload_detection_result(result: dict, img_b64: str, log=print) -> bool:
    carbon = result["carbon"]
    payload = {
        "timestamp": datetime.now().isoformat(),
        "detection_summary": {
            "total_mangroves": result["total"],
            "area_m2": carbon["area_m2"],
            "area_hectares": carbon["area_ha"],
        },
        "carbon_sequestration": {
            "carbon_stock_tons": carbon["carbon_tons"],
            "co2_equivalent_tons": carbon["co2_tons"],
            "trees_equivalent_per_year": carbon["trees_equiv"],
        },
        "annotated_image": img_b64,
        "detections": result["detections"],
        "parameters": {
            "confidence_threshold": result["conf"],
            "pixel_to_meter": PIXEL_TO_METER,
        },
        "source": "tkinter_app",
    }

    ok1 = firebase_put(f"{FIREBASE_OUTPUT_PATH}/latest_detection", payload, log)
    ts_key = f"detection_{int(time.time()*1000)}"
    ok2 = firebase_patch(f"{FIREBASE_OUTPUT_PATH}/history", {ts_key: payload}, log)
    return ok1

def run_detection(frame_bgr: np.ndarray, model, conf: float):
    results = model.predict(frame_bgr, conf=conf, verbose=False)
    detections_list = []
    total_area_px = 0
    annotated = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue

        sv_det = sv.Detections.from_ultralytics(r)
        box_ann = sv.BoxAnnotator(thickness=3, color=sv.Color(r=0, g=255, b=0))
        lbl_ann = sv.LabelAnnotator(
            text_color=sv.Color(r=255, g=255, b=255),
            text_scale=0.5, text_thickness=2,
        )
        labels = [
            f"{r.names[int(cid)]} {c:.0%}"
            for cid, c in zip(sv_det.class_id, sv_det.confidence)
        ]
        annotated = box_ann.annotate(annotated, sv_det)
        annotated = lbl_ann.annotate(annotated, sv_det, labels)

        for i, box in enumerate(r.boxes):
            cid = int(box.cls[0])
            cname = r.names[cid]
            conf_ = float(box.conf[0])
            bbox = box.xyxy[0].cpu().numpy()
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            total_area_px += w * h
            detections_list.append({
                "id": i + 1,
                "class": cname,
                "confidence": round(conf_ * 100, 1),
                "bbox": {
                    "x1": int(bbox[0]),
                    "y1": int(bbox[1]),
                    "x2": int(bbox[2]),
                    "y2": int(bbox[3])
                },
                "area_pixels": int(w * h),
            })

    annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    carbon = calculate_carbon(total_area_px * (PIXEL_TO_METER ** 2))
    result = {
        "total": len(detections_list),
        "detections": detections_list,
        "carbon": carbon,
        "conf": conf,
    }
    return annotated_bgr, result

class MangroveApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Mangrove Detection")
        self.root.configure(bg="#1e3c1e")
        self.root.resizable(True, True)

        self.model = None
        self.last_frame_bgr = None
        self.auto_capture = tk.BooleanVar(value=True)
        self.conf_var = tk.DoubleVar(value=DEFAULT_CONFIDENCE)
        self.res_var = tk.StringVar(value="QVGA  (320×240)")
        self.status_var = tk.StringVar(value="Initialising")
        self._stream_job = None
        self._capture_job = None

        self._build_ui()
        self._load_model_async()
        self._start_stream()
        self._schedule_capture()

    def _build_ui(self):
        self.stream_lbl = tk.Label(self.root, bg="black", width=60, height=20)
        self.stream_lbl.pack()

        self.det_lbl = tk.Label(self.root, bg="black", width=60, height=20)
        self.det_lbl.pack()

        tk.Button(self.root, text="Capture & Detect",
                  command=self._manual_capture).pack()

    def log(self, msg: str):
        print(msg)

    def set_status(self, txt: str):
        self.root.after(0, lambda: self.status_var.set(txt))

    def _load_model_async(self):
        def _load():
            global YOLO_AVAILABLE
            if not YOLO_AVAILABLE or not os.path.exists(MODEL_PATH):
                YOLO_AVAILABLE = False
                return
            try:
                self.model = YOLO(MODEL_PATH)
            except Exception:
                YOLO_AVAILABLE = False
        threading.Thread(target=_load, daemon=True).start()

    def _start_stream(self):
        self._fetch_and_show_frame()

    def _fetch_and_show_frame(self):
        def _fetch():
            try:
                r = requests.get(ESP32_URL, timeout=2)
                if r.status_code == 200:
                    arr = np.frombuffer(r.content, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        self.last_frame_bgr = frame.copy()
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        self._update_stream_label(rgb)
            except Exception:
                pass

        threading.Thread(target=_fetch, daemon=True).start()
        self._stream_job = self.root.after(STREAM_REFRESH_MS, self._fetch_and_show_frame)

    def _update_stream_label(self, rgb: np.ndarray):
        img = Image.fromarray(rgb).resize((480, 320), Image.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)

        def _update():
            self.stream_lbl.imgtk = imgtk
            self.stream_lbl.configure(image=imgtk)
        self.root.after(0, _update)

    def _schedule_capture(self):
        self._capture_job = self.root.after(
            CAPTURE_INTERVAL_MS, self._auto_capture_tick)

    def _auto_capture_tick(self):
        if self.auto_capture.get():
            threading.Thread(target=self._do_capture_pipeline,
                             daemon=True).start()
        self._capture_job = self.root.after(
            CAPTURE_INTERVAL_MS, self._auto_capture_tick)

    def _manual_capture(self):
        threading.Thread(target=self._do_capture_pipeline,
                         daemon=True).start()

    def _do_capture_pipeline(self):
        try:
            r = requests.get(ESP32_URL, timeout=5)
            if r.status_code != 200:
                return
            arr = np.frombuffer(r.content, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                return
        except Exception:
            return

        self.last_frame_bgr = frame.copy()
        _, buf = cv2.imencode(".jpg", frame)
        raw_b64 = base64.b64encode(buf).decode()
        threading.Thread(target=upload_raw_image,
                         args=(raw_b64, self.log), daemon=True).start()

        if not YOLO_AVAILABLE or self.model is None:
            return

        try:
            conf = self.conf_var.get()
            annotated_bgr, result = run_detection(frame, self.model, conf)
        except Exception:
            return

        _, abuf = cv2.imencode(".jpg", annotated_bgr)
        ann_b64 = base64.b64encode(abuf).decode()

        ann_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        ann_img = Image.fromarray(ann_rgb).resize((480, 320), Image.LANCZOS)
        ann_imgtk = ImageTk.PhotoImage(image=ann_img)

        def _update_det():
            self.det_lbl.imgtk = ann_imgtk
            self.det_lbl.configure(image=ann_imgtk)
        self.root.after(0, _update_det)

        upload_detection_result(result, ann_b64, self.log)

    def on_close(self):
        if self._stream_job:
            self.root.after_cancel(self._stream_job)
        if self._capture_job:
            self.root.after_cancel(self._capture_job)
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = MangroveApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()