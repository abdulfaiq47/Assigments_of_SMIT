# ==========
# visualize.py
# ==========

import ast
import cv2
import numpy as np
import pandas as pd



# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def parse_bbox(bbox_str):
    """
    Parse bbox strings in either '[x1 x2 x3 x4]' or 'x1 x2 x3 x4' format.
    Returns (x1, y1, x2, y2) as floats.
    """
    s = str(bbox_str).strip()
    if s.startswith('['):
        # Handle space-separated values inside brackets
        s = s.replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ',')
        return ast.literal_eval(s)
    else:
        vals = s.split()
        return tuple(map(float, vals))


def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=2, corner_radius=0.15):
    """Fancy corner-only border with semi-transparent fill."""
    x1, y1 = top_left
    x2, y2 = bottom_right
    w, h   = x2 - x1, y2 - y1
    lx     = int(w * corner_radius)
    ly     = int(h * corner_radius)

    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    img = cv2.addWeighted(overlay, 0.15, img, 0.85, 0)

    for px, py in [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]:
        sx = 1 if px == x1 else -1
        sy = 1 if py == y1 else -1
        cv2.line(img, (px, py), (px + sx * lx, py),        color, thickness)
        cv2.line(img, (px, py), (px,            py + sy * ly), color, thickness)

    return img


def draw_plate_overlay(frame, car_bbox, license_crop, plate_number):
    """
    Paste the license plate crop image + white text banner above the car bbox.
    """
    car_x1, car_y1, car_x2, car_y2 = [int(v) for v in car_bbox]
    H, W = license_crop.shape[:2]
    fH, fW = frame.shape[:2]

    # Positions
    crop_y1 = car_y1 - H - 100
    crop_y2 = car_y1 - 100
    ban_y1  = car_y1 - H - 400
    ban_y2  = car_y1 - H - 100
    cx_l    = int((car_x1 + car_x2 - W) / 2)
    cx_r    = cx_l + W

    # Bounds check
    if any(v < 0 for v in [crop_y1, ban_y1, cx_l]) or cx_r > fW:
        return frame

    # Paste plate image
    frame[crop_y1:crop_y2, cx_l:cx_r] = license_crop

    # White banner
    frame[ban_y1:ban_y2, cx_l:cx_r] = (255, 255, 255)

    # Plate number text on banner
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 4.3, 17
    (tw, th), _ = cv2.getTextSize(plate_number, font, scale, thick)
    tx = int((car_x1 + car_x2 - tw) / 2)
    ty = int(car_y1 - H - 250 + th / 2)
    cv2.putText(frame, plate_number, (tx, ty), font, scale, (0, 0, 0), thick)

    return frame


def draw_violation_badge(frame, car_bbox, plate_number):
    """
    Red 'NO SEATBELT | PLATE' banner drawn just below the car bbox.
    """
    x1, y1, x2, y2 = [int(v) for v in car_bbox]
    fH = frame.shape[0]
    badge_h = 55
    by1 = y2
    by2 = min(y2 + badge_h, fH)

    cv2.rectangle(frame, (x1, by1), (x2, by2), (0, 0, 200), -1)

    label = f"NO SEATBELT  |  {plate_number}"
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
    (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
    tx = x1 + max(0, (x2 - x1 - tw) // 2)
    ty = by1 + (badge_h + th) // 2

    cv2.putText(frame, label, (tx, ty), font, scale, (255, 255, 255), thick)
    return frame


def draw_plate_label(frame, lp_bbox, plate_number):
    """
    Small label drawn directly above the license plate bounding box.
    """
    x1, y1, x2, y2 = [int(v) for v in lp_bbox]

    # Orange box around the plate
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 3)

    # Text above the box
    label = plate_number
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2
    (tw, th), _ = cv2.getTextSize(label, font, scale, thick)

    # Background pill
    pad = 4
    cv2.rectangle(frame,
                  (x1 - pad, y1 - th - pad * 2 - 10),
                  (x1 + tw + pad, y1 - 10),
                  (0, 165, 255), -1)
    cv2.putText(frame, label,
                (x1, y1 - 14),
                font, scale, (255, 255, 255), thick)
    return frame


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

# ── Load CSV ───────────────────────────────────────────────────────────────────

import sys

csv_file = sys.argv[1]
output_video = sys.argv[2]
video_path = sys.argv[3]
cap = cv2.VideoCapture(video_path)

results = pd.read_csv(csv_file)

# ── Open Video ─────────────────────────────────────────────────────────────────
cap    = cv2.VideoCapture(video_path)
fps    = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out    = cv2.VideoWriter(output_video,
                         cv2.VideoWriter_fourcc(*'mp4v'),
                         fps, (width, height))

# ── Pre-load Best Plate Crop per Car ──────────────────────────────────────────
print("Pre-loading license plate crops...")
license_plate = {}

for car_id in np.unique(results['car_id']):
    try:
        # Get rows for this car
        car_rows = results[results['car_id'] == car_id]

        # ✅ filter valid scores
        car_rows = car_rows[
            (car_rows['license_number_score'] > 0) &
            (car_rows['license_number_score'] <= 1)
        ]

        # ✅ filter clean plate text (VERY IMPORTANT)
        car_rows = car_rows[
            car_rows['license_number'].str.match(r'^[A-Z0-9]+$', na=False)
        ]

        if car_rows.empty:
            continue

        # ✅ get best row directly (no duplicate max logic)
        best_row = car_rows.loc[
            car_rows['license_number_score'].idxmax()
        ]

        max_score = best_row['license_number_score']

    except Exception as e:
        print(f"  [ERROR] Car {car_id}: {e}")
        continue

    # Only use rows with a valid score
    valid_rows = car_rows[car_rows['license_number_score'] == max_score]
    if valid_rows.empty:
        continue

    best_row = valid_rows.iloc[0]

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(best_row['frame_nmr']))
    ret, frame = cap.read()
    if not ret:
        continue

    x1, y1, x2, y2 = parse_bbox(best_row['license_plate_bbox'])
    raw = frame[int(y1):int(y2), int(x1):int(x2)]

    if raw.size == 0:
        continue

    target_h = 400
    target_w = max(1, int((x2 - x1) * target_h / (y2 - y1)))
    resized  = cv2.resize(raw, (target_w, target_h))

    plate_num = str(best_row['license_number'])
    license_plate[car_id] = {
        'license_crop':         resized,
        'license_plate_number': plate_num
    }
    print(f"  Car {int(car_id):>4} → '{plate_num}'  (score={max_score:.3f})")

# ── Render Loop ────────────────────────────────────────────────────────────────
print("\nRendering video...")
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
frame_nmr = -1

while True:
    ret, frame = cap.read()
    frame_nmr += 1

    if not ret:
        break

    df_ = results[results['frame_nmr'] == frame_nmr]

    for _, row in df_.iterrows():
        car_id = row['car_id']

        car_x1, car_y1, car_x2, car_y2 = parse_bbox(row['car_bbox'])
        lp_x1,  lp_y1,  lp_x2,  lp_y2  = parse_bbox(row['license_plate_bbox'])

        car_bbox = (car_x1, car_y1, car_x2, car_y2)
        lp_bbox  = (lp_x1,  lp_y1,  lp_x2,  lp_y2)

        # Red corner border on the car
        frame = draw_border(frame,
                            (int(car_x1), int(car_y1)),
                            (int(car_x2), int(car_y2)),
                            color=(0, 0, 255), thickness=4)

        if car_id not in license_plate:
            continue

        plate_number = license_plate[car_id]['license_plate_number']
        crop         = license_plate[car_id]['license_crop']

        # Small label on the plate box itself
        frame = draw_plate_label(frame, lp_bbox, plate_number)

        # Large overlay (crop + text banner) above the car
        try:
            frame = draw_plate_overlay(frame, car_bbox, crop, plate_number)
        except Exception as e:
            print(f"  [overlay] Frame {frame_nmr}, Car {car_id}: {e}")

        # Red badge below the car
        frame = draw_violation_badge(frame, car_bbox, plate_number)

    out.write(frame)

    if frame_nmr % 30 == 0:
        print(f"  Frame {frame_nmr} processed...")

# ── Cleanup ────────────────────────────────────────────────────────────────────
cap.release()
out.release()
print("\nDone → out.mp4")