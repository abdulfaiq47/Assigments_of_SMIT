# ==========
# util.py
# ==========

import cv2
import numpy as np
import easyocr

# ── OCR Reader ─────────────────────────────────────────────────────────────────
reader = easyocr.Reader(['en'], gpu=False)

# ── Detection Zone (adjust to your video resolution) ──────────────────────────
DETECTION_ZONE = None
# ── Character Correction Maps ──────────────────────────────────────────────────
dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}
dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S'}


# ══════════════════════════════════════════════════════════════════════════════
# CSV
# ══════════════════════════════════════════════════════════════════════════════

def write_csv(results, output_path):
    """Write detection results to a CSV file."""
    with open(output_path, 'w') as f:
        f.write('frame_nmr,car_id,car_bbox,license_plate_bbox,'
                'license_plate_bbox_score,license_number,license_number_score\n')

        for frame_nmr in results:
            for car_id in results[frame_nmr]:
                entry = results[frame_nmr][car_id]
                if ('car' in entry
                        and 'license_plate' in entry
                        and 'text' in entry['license_plate']):

                    cb  = entry['car']['bbox']
                    lpb = entry['license_plate']['bbox']

                    f.write('{},{},{},{},{},{},{}\n'.format(
                        frame_nmr,
                        car_id,
                        '[{} {} {} {}]'.format(*cb),
                        '[{} {} {} {}]'.format(*lpb),
                        entry['license_plate']['bbox_score'],
                        entry['license_plate']['text'],
                        entry['license_plate']['text_score']
                    ))


# ══════════════════════════════════════════════════════════════════════════════
# LICENSE PLATE OCR
# ══════════════════════════════════════════════════════════════════════════════

def license_complies_format(text):
    """Accept plates between 3–12 characters (relaxed for broad compatibility)."""
    return 3 <= len(text) <= 12


def format_license(text):
    """
    Apply character-correction mapping to a 7-character plate string.
    Only called when plate length is exactly 7.
    """
    mapping = {
        0: dict_int_to_char, 1: dict_int_to_char,
        4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
        2: dict_char_to_int, 3: dict_char_to_int
    }
    return ''.join(
        mapping[j].get(text[j], text[j]) for j in range(7)
    )


def read_license_plate(license_plate_crop):
    """
    Run EasyOCR on a cropped license plate image.

    Returns:
        (text, score) or (None, None)
    """
    detections = reader.readtext(license_plate_crop)

    for (bbox, text, score) in detections:
        text = text.upper().replace(' ', '')
        print(f"  [OCR] '{text}'  score={score:.2f}")

        if license_complies_format(text):
            return text, score

    return None, None


# ══════════════════════════════════════════════════════════════════════════════
# VEHICLE TRACKING
# ══════════════════════════════════════════════════════════════════════════════

def get_car(license_plate, vehicle_track_ids):
    """
    Match a license plate bounding box to a tracked vehicle.

    Args:
        license_plate : [x1, y1, x2, y2, score, class_id]
        vehicle_track_ids : array of [x1, y1, x2, y2, track_id]

    Returns:
        (xcar1, ycar1, xcar2, ycar2, car_id)  or  (-1,-1,-1,-1,-1)
    """
    x1, y1, x2, y2, score, class_id = license_plate

    for j, (xcar1, ycar1, xcar2, ycar2, car_id) in enumerate(vehicle_track_ids):
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            return vehicle_track_ids[j]

    print("  [TRACK] Plate not matched to any vehicle.")
    return -1, -1, -1, -1, -1


# ══════════════════════════════════════════════════════════════════════════════
# DETECTION ZONE
# ══════════════════════════════════════════════════════════════════════════════

def is_in_detection_zone(bbox, zone=DETECTION_ZONE):
    """
    Check if the bottom-center of a bounding box is inside the zone polygon.

    Args:
        bbox : [x1, y1, x2, y2]

    Returns:
        bool
    """
    x1, y1, x2, y2 = bbox
    cx = int((x1 + x2) / 2)
    cy = int(y2)
    return cv2.pointPolygonTest(zone, (float(cx), float(cy)), False) >= 0


def draw_detection_zone(frame, zone=DETECTION_ZONE, color=(0, 255, 255), thickness=2):
    """Draw the detection zone polygon on a frame (for debugging)."""
    cv2.polylines(frame, [zone], isClosed=True, color=color, thickness=thickness)
    return frame


# ══════════════════════════════════════════════════════════════════════════════
# SEATBELT DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_seatbelt(frame, car_bbox, seatbelt_model, confidence_threshold=0.5):
    """
    Crop the upper half of a car bounding box and run seatbelt detection.

    Args:
        frame            : full BGR video frame
        car_bbox         : [x1, y1, x2, y2]
        seatbelt_model   : loaded YOLO model
        confidence_threshold : min score to accept detection

    Returns:
        'wearing' | 'not_wearing' | 'unknown'
    """
    x1, y1, x2, y2 = [int(v) for v in car_bbox]
    mid_y = int((y1 + y2) / 2)

    driver_crop = frame[y1:mid_y, x1:x2]
    if driver_crop.size == 0:
        return 'unknown'

    detections = seatbelt_model(driver_crop)[0]

    for det in detections.boxes.data.tolist():
        _, _, _, _, score, class_id = det
        class_name = seatbelt_model.names[int(class_id)].lower()

        if score >= confidence_threshold:
            if any(kw in class_name for kw in ('no', 'without', 'not')):
                return 'not_wearing'
            if any(kw in class_name for kw in ('belt', 'wear', 'with')):
                return 'wearing'

    return 'unknown'