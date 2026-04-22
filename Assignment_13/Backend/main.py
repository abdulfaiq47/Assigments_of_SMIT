# ==========
# main.py
# ==========

from ultralytics import YOLO
import cv2
import numpy as np
from sort.sort import Sort
from util import (
    get_car,
    read_license_plate,
    write_csv,
    is_in_detection_zone,
    detect_seatbelt,
    draw_detection_zone
)
import util

import sys

input_video = sys.argv[1]
output_csv = sys.argv[2]

# ── Load Models ────────────────────────────────────────────────────────────────
coco_model             = YOLO('yolov8n.pt')
license_plate_detector = YOLO('License_plate_detection_model (1).pt')
seatbelt_model         = YOLO('seat_belt_detection_model (2).pt')        # <-- update filename

# ── Load Video ─────────────────────────────────────────────────────────────────
cap      = cv2.VideoCapture(input_video)
vehicles = [2, 3, 5, 7]   # COCO: car, motorcycle, bus, truck

# ── State ─────────────────────────────────────────────────────────────────────
results     = {}
mot_tracker = Sort()
frame_nmr   = -1

print("=" * 60)
print("Starting detection pipeline...")
print("=" * 60)

# ── Frame Loop ─────────────────────────────────────────────────────────────────
while True:
    frame_nmr += 1
    ret, frame = cap.read()

    if not ret: 
        print(f"\nEnd of video at frame {frame_nmr}.")
        break

    if util.DETECTION_ZONE is None:
        h, w = frame.shape[:2]

        util.DETECTION_ZONE = np.array([
            [int(0.35 * w), int(0.45 * h)],
            [int(0.65 * w), int(0.45 * h)],
            [int(0.95 * w), int(0.95 * h)],
            [int(0.05 * w), int(0.95 * h)]
        ], dtype=np.int32)
    
    results[frame_nmr] = {}

    # Draw zone overlay (comment out after tuning)
    frame = draw_detection_zone(frame, util.DETECTION_ZONE)

    print(f"\n── Frame {frame_nmr} ──────────────────────────────────────────")

    # ── 1. Detect Vehicles ─────────────────────────────────────────────────────
    coco_results = coco_model(frame)[0]
    detections_  = [
        [x1, y1, x2, y2, score]
        for x1, y1, x2, y2, score, class_id in coco_results.boxes.data.tolist()
        if int(class_id) in vehicles
    ]
    print(f"  Vehicles detected : {len(detections_)}")

    # ── 2. Track Vehicles ──────────────────────────────────────────────────────
    track_ids = (
        mot_tracker.update(np.asarray(detections_))
        if detections_
        else np.empty((0, 5))
    )

    # ── 3. Detect License Plates ───────────────────────────────────────────────
    lp_results = license_plate_detector(frame)[0]
    print(f"  License plates    : {len(lp_results.boxes)}")

    for lp in lp_results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = lp

        # ── 4. Match Plate → Car ───────────────────────────────────────────────
        xcar1, ycar1, xcar2, ycar2, car_id = get_car(lp, track_ids)
        if car_id == -1:
            continue

        car_bbox = [xcar1, ycar1, xcar2, ycar2]

        # ── 5. Zone Check ──────────────────────────────────────────────────────
        if not is_in_detection_zone(car_bbox, util.DETECTION_ZONE):
            print(f"  Car {int(car_id):>4} | Outside zone — skip")
            continue

        # ── 6. Seatbelt Check ──────────────────────────────────────────────────
        seatbelt_status = detect_seatbelt(frame, car_bbox, seatbelt_model)
        print(f"  Car {int(car_id):>4} | Seatbelt: {seatbelt_status}")

        if seatbelt_status == 'wearing':
            print(f"           | Compliant — skip")
            continue

        # ── 7. Crop & Pre-process Plate ────────────────────────────────────────
        lp_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
        if lp_crop.size == 0:
            print(f"           | Empty crop — skip")
            continue

        gray   = cv2.cvtColor(lp_crop, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY_INV)

        # ── 8. OCR ─────────────────────────────────────────────────────────────
        lp_text, lp_score = read_license_plate(thresh)
        if lp_text is None:
            print(f"           | OCR failed — skip")
            continue

        print(f"  Car {int(car_id):>4} | ⚠ NO SEATBELT | Plate: {lp_text} ({lp_score:.2f})")

        # ── 9. Store ───────────────────────────────────────────────────────────
        results[frame_nmr][car_id] = {
            'car':           {'bbox': [xcar1, ycar1, xcar2, ycar2]},
            'license_plate': {
                'bbox':       [x1, y1, x2, y2],
                'text':       lp_text,
                'bbox_score': score,
                'text_score': lp_score
            }
        }

# ── Save ───────────────────────────────────────────────────────────────────────
cap.release()
write_csv(results, output_csv)
print(f"\nDone. Results written → {output_csv}")