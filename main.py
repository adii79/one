import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import requests
import base64
import time
from datetime import datetime
from ultralytics import YOLO
import supervision as sv
import io

# ==============================
# CONFIGURATION
# ==============================

ESP32_URL = "http://10.13.106.254/capture"

FIREBASE_HOST = "https://your-project-default-rtdb.firebaseio.com/"
FIREBASE_AUTH = "YOUR_FIREBASE_AUTH_TOKEN"
FIREBASE_OUTPUT_PATH = "/MVR"

MODEL_PATH = "best.pt"
CONFIDENCE_THRESHOLD = 0.3
PIXEL_TO_METER = 0.5

CARBON_PER_HECTARE = 388  # tons C/ha


# ==============================
# LOAD MODEL
# ==============================

try:
    model = YOLO(MODEL_PATH)
    print("✅ YOLO model loaded")
except Exception as e:
    print("❌ Model load error:", e)
    model = None


# ==============================
# CARBON CALCULATION
# ==============================

def calculate_carbon(area_m2):
    area_ha = area_m2 / 10000
    carbon_tons = area_ha * CARBON_PER_HECTARE
    co2_tons = carbon_tons * 3.67
    trees_equiv = round(co2_tons * 1000 / 20)

    return {
        "area_m2": round(area_m2, 2),
        "area_ha": round(area_ha, 4),
        "carbon_tons": round(carbon_tons, 2),
        "co2_tons": round(co2_tons, 2),
        "trees_equivalent": trees_equiv
    }


# ==============================
# TKINTER APPLICATION
# ==============================

class MangroveApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Mangrove Detection - ESP32")

        self.video_label = tk.Label(root)
        self.video_label.pack()

        self.detect_button = tk.Button(root, text="Run Detection", command=self.run_detection)
        self.detect_button.pack(pady=5)

        self.upload_button = tk.Button(root, text="Send Detection to Firebase", command=self.send_to_firebase)
        self.upload_button.pack(pady=5)

        self.status_label = tk.Label(root, text="Status: Waiting...")
        self.status_label.pack()

        self.current_frame = None
        self.last_result = None

        self.update_frame()

    # ==============================
    # STREAM FRAME
    # ==============================

    def update_frame(self):
        try:
            response = requests.get(ESP32_URL, timeout=2)
            if response.status_code == 200:
                img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                if frame is not None:
                    self.current_frame = frame.copy()
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    imgtk = ImageTk.PhotoImage(image=img)

                    self.video_label.imgtk = imgtk
                    self.video_label.configure(image=imgtk)

        except Exception as e:
            print("Stream error:", e)

        self.root.after(100, self.update_frame)

    # ==============================
    # RUN YOLO DETECTION
    # ==============================

    def run_detection(self):
        if model is None:
            messagebox.showerror("Error", "Model not loaded")
            return

        if self.current_frame is None:
            messagebox.showwarning("Warning", "No frame available")
            return

        self.status_label.config(text="Status: Running detection...")

        frame = self.current_frame.copy()

        results = model.predict(frame, conf=CONFIDENCE_THRESHOLD)

        detections_list = []
        total_area_pixels = 0

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        for r in results:
            detections = sv.Detections.from_ultralytics(r)

            box_annotator = sv.BoxAnnotator(thickness=3)
            label_annotator = sv.LabelAnnotator()

            labels = []
            for i, box in enumerate(r.boxes):
                bbox = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = r.names[class_id]

                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                area = width * height
                total_area_pixels += area

                detections_list.append({
                    "id": i + 1,
                    "class": class_name,
                    "confidence": round(conf * 100, 2),
                    "area_pixels": int(area)
                })

                labels.append(f"{class_name} {conf:.2f}")

            annotated = box_annotator.annotate(image_rgb.copy(), detections)
            annotated = label_annotator.annotate(annotated, detections, labels)

        # convert back for display
        annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

        total_area_m2 = total_area_pixels * (PIXEL_TO_METER ** 2)
        carbon_data = calculate_carbon(total_area_m2)

        self.last_result = {
            "total_detections": len(detections_list),
            "detections": detections_list,
            "carbon": carbon_data,
            "timestamp": datetime.now().isoformat()
        }

        img = Image.fromarray(annotated)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.status_label.config(
            text=f"Detected: {len(detections_list)} | CO2: {carbon_data['co2_tons']} tons"
        )

    # ==============================
    # SEND ONLY DATA TO FIREBASE
    # ==============================

    def send_to_firebase(self):

        if not self.last_result:
            messagebox.showwarning("Warning", "Run detection first")
            return

        try:
            firebase_data = {
                "timestamp": self.last_result["timestamp"],
                "detection_summary": {
                    "total_mangroves": self.last_result["total_detections"],
                    "area_m2": self.last_result["carbon"]["area_m2"],
                    "area_hectares": self.last_result["carbon"]["area_ha"]
                },
                "carbon_sequestration": {
                    "carbon_stock_tons": self.last_result["carbon"]["carbon_tons"],
                    "co2_equivalent_tons": self.last_result["carbon"]["co2_tons"],
                    "trees_equivalent_per_year": self.last_result["carbon"]["trees_equivalent"]
                },
                "detections": self.last_result["detections"],
                "source": "esp32_local_detection"
            }

            url = f"{FIREBASE_HOST}{FIREBASE_OUTPUT_PATH}/latest_detection.json?auth={FIREBASE_AUTH}"

            response = requests.put(url, json=firebase_data, timeout=10)

            if response.status_code == 200:
                messagebox.showinfo("Success", "Detection data sent to Firebase")
            else:
                messagebox.showerror("Error", f"Firebase error: {response.status_code}")

        except Exception as e:
            messagebox.showerror("Error", str(e))


# ==============================
# RUN APP
# ==============================

if __name__ == "__main__":
    root = tk.Tk()
    app = MangroveApp(root)
    root.mainloop()