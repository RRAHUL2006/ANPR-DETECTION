import cv2
import numpy as np
import os
import time
from ultralytics import YOLO
from paddleocr import PaddleOCR
from collections import defaultdict, Counter
import re
import csv
from datetime import datetime 

log_file = open("plate_logs.csv", "a", newline="")
writer = csv.writer(log_file)

if log_file.tell() == 0:
    writer.writerow(["timestamp", "plate", "confidence", "time_ms"])

# =========================
# 🔥 CONFIG
# =========================
MODEL_PATH = r"C:\\Users\\R RAHUL\\OneDrive\\Desktop\\AIPLANTO\\best8s.pt"
RTSP_URL = "rtsp://service:Admin123$@10.92.42.203/axis-media/media.amp"

os.makedirs("plates", exist_ok=True)

# =========================
# 🔥 NORMALIZE
# =========================
def normalize_plate_text(text):
    text = text.upper().replace(" ", "")
    text = re.sub(r"[^A-Z0-9]", "", text)
    text = text.replace("O", "0").replace("I", "1")
    return text

# =========================
# 🔥 OCR CLASS
# =========================
class PlateOCRer:
    def __init__(self):
        self.ocr = PaddleOCR(lang='en')

    def ocr_plate(self, img_bgr):
        if len(img_bgr.shape) == 2:
            img = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        result = self.ocr.ocr(img)

        if not result or result[0] is None:
            return "", 0.0

        texts, confs = [], []

        for line in result:
            if line is None:
                continue

            for item in line:
                if item is None or len(item) < 2:
                    continue

                txt, conf = item[1]
                texts.append(txt)
                confs.append(float(conf))

        if not texts:
            return "", 0.0

        text = normalize_plate_text("".join(texts))
        avg_conf = float(np.mean(confs)) if confs else 0.0

        return text, avg_conf

# =========================
# 🔥 UTILS
# =========================
def iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])

    inter = max(0, xB-xA) * max(0, yB-yA)
    areaA = (a[2]-a[0])*(a[3]-a[1])
    areaB = (b[2]-b[0])*(b[3]-b[1])

    return inter / (areaA + areaB - inter + 1e-6)

def get_final_plate(texts):
    return Counter(texts).most_common(1)[0][0] if texts else ""

# =========================
# 🔥 INIT
# =========================
model = YOLO(MODEL_PATH)
ocr_engine = PlateOCRer()

cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

plate_memory = defaultdict(list)
tracks = []
track_id_counter = 0
saved_plates = set()

last_save_time = 0
save_interval = 2  # seconds

# =========================
# 🔥 LOOP
# =========================
while True:
    ret, frame = cap.read()

    if not ret:
        print("Reconnecting stream...")
        cap.release()
        time.sleep(1)
        cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
        continue

    frame = cv2.resize(frame, (960, 540))

    # 🔥 TOTAL TIMER START
    total_start = time.perf_counter()

    # 🔥 YOLO TIMER
    yolo_start = time.perf_counter()
    results = model(frame, conf=0.5, iou=0.4, verbose=False)
    yolo_end = time.perf_counter()
    yolo_time = (yolo_end - yolo_start) * 1000

    new_tracks = []
    ocr_total_time = 0
    ocr_count = 0

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        current_box = [x1, y1, x2, y2]

        assigned_id = None
        for tid, tbox in tracks:
            if iou(current_box, tbox) > 0.3:
                assigned_id = tid
                break

        if assigned_id is None:
            assigned_id = track_id_counter
            track_id_counter += 1

        new_tracks.append((assigned_id, current_box))

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2)

        # 🔥 OCR TIMER
        ocr_start = time.perf_counter()
        text, conf = ocr_engine.ocr_plate(gray)
        ocr_end = time.perf_counter()

        ocr_time = (ocr_end - ocr_start) * 1000
        ocr_total_time += ocr_time
        ocr_count += 1

        if conf > 0.6 and len(text) >= 6:
            plate_memory[assigned_id].append(text)

        final_text = get_final_plate(plate_memory[assigned_id])

        now = time.time()
        if final_text and len(plate_memory[assigned_id]) >= 5:
            if final_text not in saved_plates and (now - last_save_time > save_interval):

                saved_plates.add(final_text)
                last_save_time = now

                filename = f"plates/{final_text}_{int(now)}.jpg"
                cv2.imwrite(filename, crop)

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                writer.writerow([timestamp, final_text, round(conf, 3), round(yolo_time, 2)])
                log_file.flush()

                print(f"[STREAM DETECTED] {final_text} | conf={conf:.2f} | {yolo_time:.2f} ms")

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        if final_text:
            label = f"{final_text} ({conf:.2f})"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0,255,0), 2)

    tracks = new_tracks

    # 🔥 TOTAL TIMER END
    total_end = time.perf_counter()
    total_time = (total_end - total_start) * 1000

    avg_ocr_time = (ocr_total_time / ocr_count) if ocr_count > 0 else 0

    # 🔥 DISPLAY TIMINGS
    cv2.putText(frame, f"Total: {total_time:.2f} ms", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    cv2.putText(frame, f"YOLO: {yolo_time:.2f} ms", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

    cv2.putText(frame, f"OCR(avg): {avg_ocr_time:.2f} ms", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

    cv2.imshow("ANPR STREAM mir", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
log_file.close()