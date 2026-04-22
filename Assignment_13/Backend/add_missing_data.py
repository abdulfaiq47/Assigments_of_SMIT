# ==========
# add_missing_data.py
# ==========

import csv
import numpy as np
from scipy.interpolate import interp1d


def interpolate_bounding_boxes(data):
    """
    For each car, linearly interpolate bounding boxes across frames
    where detections were missed.
    """
    frame_numbers         = np.array([int(row['frame_nmr'])         for row in data])
    car_ids               = np.array([int(float(row['car_id']))      for row in data])
    car_bboxes            = np.array([list(map(float, row['car_bbox'].split()))            for row in data])
    license_plate_bboxes  = np.array([list(map(float, row['license_plate_bbox'].split()))  for row in data])

    interpolated_data = []

    for car_id in np.unique(car_ids):
        mask               = car_ids == car_id
        car_frame_numbers  = frame_numbers[mask]
        first_frame        = car_frame_numbers[0]

        # Original frame list for this car (for lookup)
        orig_frames = [row['frame_nmr'] for row in data if int(float(row['car_id'])) == car_id]

        interp_car_bboxes = []
        interp_lp_bboxes  = []

        for i, frame_number in enumerate(car_frame_numbers):
            car_bbox = car_bboxes[mask][i]
            lp_bbox  = license_plate_bboxes[mask][i]

            if i > 0:
                prev_frame     = car_frame_numbers[i - 1]
                gap            = frame_number - prev_frame

                if gap > 1:
                    x_endpoints = np.array([prev_frame, frame_number])
                    x_fill      = np.linspace(prev_frame, frame_number, num=gap, endpoint=False)

                    car_interp = interp1d(x_endpoints,
                                          np.vstack((interp_car_bboxes[-1], car_bbox)),
                                          axis=0)(x_fill)
                    lp_interp  = interp1d(x_endpoints,
                                          np.vstack((interp_lp_bboxes[-1], lp_bbox)),
                                          axis=0)(x_fill)

                    interp_car_bboxes.extend(car_interp[1:])
                    interp_lp_bboxes.extend(lp_interp[1:])

            interp_car_bboxes.append(car_bbox)
            interp_lp_bboxes.append(lp_bbox)

        # Build output rows
        for i, (cb, lb) in enumerate(zip(interp_car_bboxes, interp_lp_bboxes)):
            frame_num = str(first_frame + i)

            row = {
                'frame_nmr':           frame_num,
                'car_id':              str(car_id),
                'car_bbox':            ' '.join(map(str, cb)),
                'license_plate_bbox':  ' '.join(map(str, lb)),
            }

            if frame_num in orig_frames:
                orig = next(r for r in data
                            if r['frame_nmr'] == frame_num
                            and int(float(r['car_id'])) == car_id)
                row['license_plate_bbox_score'] = orig.get('license_plate_bbox_score', '0')
                row['license_number']            = orig.get('license_number',            '0')
                row['license_number_score']      = orig.get('license_number_score',      '0')
            else:
                row['license_plate_bbox_score'] = '0'
                row['license_number']            = '0'
                row['license_number_score']      = '0'

            interpolated_data.append(row)

    return interpolated_data


def create_perfect_csv(data):
    """
    For each car, keep only the single row with the highest valid OCR score (0–1).
    """
    best_rows = {}

    for row in data:
        try:
            car_id = int(float(row['car_id']))
            score  = float(row.get('license_number_score', 0))
            if score > 1.0:          # discard invalid scores
                score = 0.0
        except (ValueError, TypeError):
            continue

        if car_id not in best_rows or score > best_rows[car_id]['score']:
            best_rows[car_id] = {'score': score, 'row': row}

    return [item['row'] for item in best_rows.values()]


import sys

input_csv = sys.argv[1]
output_interpolated = sys.argv[2]
output_perfect = sys.argv[3]
# ── Run ────────────────────────────────────────────────────────────────────────
HEADER = ['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox',
          'license_plate_bbox_score', 'license_number', 'license_number_score']

with open(input_csv, 'r') as f:
    raw_data = list(csv.DictReader(f))

# Fix bbox strings: strip brackets so split() works cleanly
for row in raw_data:
    row['car_bbox']           = row['car_bbox'].strip('[]').strip()
    row['license_plate_bbox'] = row['license_plate_bbox'].strip('[]').strip()

print(f"Loaded {len(raw_data)} rows from {input_csv}")

interpolated = interpolate_bounding_boxes(raw_data)

with open(output_interpolated, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=HEADER)
    writer.writeheader()
    writer.writerows(interpolated)

print(f"Interpolated → {output_interpolated}  ({len(interpolated)} rows)")

perfect = create_perfect_csv(interpolated)

with open(output_perfect, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=HEADER)
    writer.writeheader()
    writer.writerows(perfect)

print(f"Best-score   → {output_perfect}            ({len(perfect)} rows)")