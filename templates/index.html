import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import requests
import threading
import time
from ultralytics import YOLO
import io

# =========================
# CONFIG
# =========================
ESP32_STREAM_URL = "http://192.168.4.1:81/stream"
ESP32_CONTROL_URL = "http://192.168.4.1/control?var=framesize&val="
MODEL_PATH = "best.pt"

RESOLUTION_MAP = {
    "QQVGA (160x120)": 10,
    "QVGA (320x240)": 8,
    "VGA (640x480)": 6,
    "SVGA (800x600)": 5,
    "XGA (1024x768)": 4,
}

# =========================
# LOAD MODEL
# =========================
model = YOLO(MODEL_PATH)

# =========================
# MAIN APP
# =========================
class YOLOApp:

    def __init__(self, root):
        self.root = root
        self.root.title("ESP32-CAM Object Detection System")
        self.root.geometry("1400x850")
        self.root.configure(bg="#1e1e1e")

        self.running = False
        self.auto_detect = False
        self.current_frame = None

        self.create_ui()

    # =========================
    # UI DESIGN
    # =========================
    def create_ui(self):

        title = tk.Label(
            self.root,
            text="ESP32-CAM YOLO Detection Dashboard",
            font=("Arial", 24, "bold"),
            bg="#1e1e1e",
            fg="white"
        )
        title.pack(pady=10)

        main_frame = tk.Frame(self.root, bg="#1e1e1e")
        main_frame.pack(fill="both", expand=True)

        # LEFT - Video
        self.video_label = tk.Label(main_frame, bg="black")
        self.video_label.pack(side="left", padx=20, pady=20, expand=True)

        # RIGHT - Controls
        control_frame = tk.Frame(main_frame, bg="#2d2d2d", width=400)
        control_frame.pack(side="right", fill="y", padx=10, pady=10)

        # Confidence slider
        tk.Label(control_frame, text="Confidence Threshold",
                 font=("Arial", 14), bg="#2d2d2d", fg="white").pack(pady=10)

        self.conf_slider = tk.Scale(
            control_frame,
            from_=0.1,
            to=1.0,
            resolution=0.05,
            orient="horizontal",
            length=300
        )
        self.conf_slider.set(0.5)
        self.conf_slider.pack()

        # Auto detect interval
        tk.Label(control_frame, text="Auto Detect Interval (sec)",
                 font=("Arial", 14), bg="#2d2d2d", fg="white").pack(pady=10)

        self.interval_entry = tk.Entry(control_frame, font=("Arial", 14))
        self.interval_entry.insert(0, "5")
        self.interval_entry.pack(pady=5)

        # Resolution dropdown
        tk.Label(control_frame, text="ESP32 Resolution",
                 font=("Arial", 14), bg="#2d2d2d", fg="white").pack(pady=10)

        self.resolution_var = tk.StringVar()
        self.resolution_var.set("VGA (640x480)")
        resolution_menu = ttk.Combobox(
            control_frame,
            textvariable=self.resolution_var,
            values=list(RESOLUTION_MAP.keys()),
            state="readonly"
        )
        resolution_menu.pack(pady=5)

        tk.Button(
            control_frame,
            text="Set Resolution",
            command=self.set_resolution,
            font=("Arial", 14),
            bg="#444",
            fg="white",
            width=20
        ).pack(pady=10)

        # Buttons
        tk.Button(control_frame, text="Start Stream",
                  command=self.start_stream,
                  font=("Arial", 14),
                  bg="#28a745", fg="white",
                  width=20).pack(pady=5)

        tk.Button(control_frame, text="Stop Stream",
                  command=self.stop_stream,
                  font=("Arial", 14),
                  bg="#dc3545", fg="white",
                  width=20).pack(pady=5)

        tk.Button(control_frame, text="Detect Now",
                  command=self.detect_objects,
                  font=("Arial", 14),
                  bg="#007bff", fg="white",
                  width=20).pack(pady=5)

        tk.Button(control_frame, text="Toggle Auto Detect",
                  command=self.toggle_auto_detect,
                  font=("Arial", 14),
                  bg="#ffc107", fg="black",
                  width=20).pack(pady=5)

        # Results panel
        tk.Label(control_frame, text="Detection Results",
                 font=("Arial", 16, "bold"),
                 bg="#2d2d2d", fg="white").pack(pady=15)

        self.result_text = tk.Text(
            control_frame,
            height=12,
            font=("Arial", 12)
        )
        self.result_text.pack(padx=10)

    # =========================
    # STREAM
    # =========================
    def start_stream(self):
        self.running = True
        threading.Thread(target=self.stream_video, daemon=True).start()

    def stop_stream(self):
        self.running = False

    def stream_video(self):
        stream = requests.get(ESP32_STREAM_URL, stream=True)
        bytes_data = b''

        for chunk in stream.iter_content(chunk_size=1024):
            if not self.running:
                break

            bytes_data += chunk
            a = bytes_data.find(b'\xff\xd8')
            b = bytes_data.find(b'\xff\xd9')

            if a != -1 and b != -1:
                jpg = bytes_data[a:b+2]
                bytes_data = bytes_data[b+2:]

                image = cv2.imdecode(
                    np.frombuffer(jpg, dtype=np.uint8),
                    cv2.IMREAD_COLOR
                )
                self.current_frame = image
                self.display_frame(image)

    def display_frame(self, frame):
        frame = cv2.resize(frame, (900, 650))
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(img))
        self.video_label.configure(image=img)
        self.video_label.image = img

    # =========================
    # DETECTION
    # =========================
    def detect_objects(self):
        if self.current_frame is None:
            return

        conf = self.conf_slider.get()
        results = model(self.current_frame, conf=conf)

        annotated = results[0].plot()
        self.display_frame(annotated)

        self.result_text.delete(1.0, tk.END)

        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf_val = float(box.conf[0])
            name = model.names[cls]
            self.result_text.insert(tk.END,
                                    f"{name} - {conf_val:.2f}\n")

    # =========================
    # AUTO DETECT
    # =========================
    def toggle_auto_detect(self):
        self.auto_detect = not self.auto_detect
        if self.auto_detect:
            threading.Thread(target=self.auto_loop, daemon=True).start()

    def auto_loop(self):
        while self.auto_detect:
            time.sleep(int(self.interval_entry.get()))
            self.detect_objects()

    # =========================
    # RESOLUTION
    # =========================
    def set_resolution(self):
        res_key = self.resolution_var.get()
        val = RESOLUTION_MAP[res_key]
        requests.get(ESP32_CONTROL_URL + str(val))
        messagebox.showinfo("Success", "Resolution Updated!")

# =========================
# RUN
# =========================
if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOApp(root)
    root.mainloop()