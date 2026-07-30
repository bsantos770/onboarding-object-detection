"""
Measures the size of each detected pumpkin from its bounding box, in pixels
and in cm (see calibration.py).

Assumptions:
- The trough is 100cm wide.
- The cm conversion is only valid for objects on the same plane as the trough.
- The images used didn't go through any resize.
"""

import csv
import math
from pathlib import Path

import supervision as sv
from ultralytics import YOLO

import calibration

MODEL_PATH = "runs/pumpkins/lr01_batch16/weights/best.pt"
TEST_IMAGES_DIR = "dataset/Pumpkins detection.v2i.yolov12/test/images"
OUTPUT_CSV = "pumpkin_sizes.csv"
CONF_THRESHOLD = 0.5


def main():
    scale_cm_per_px = calibration.SCALE_CM_PER_PX

    model = YOLO(MODEL_PATH)
    results = model.predict(source=TEST_IMAGES_DIR, conf=CONF_THRESHOLD, stream=True)

    rows = []
    for result in results:
        detections = sv.Detections.from_ultralytics(result)
        image_name = Path(result.path).name

        for x1, y1, x2, y2 in detections.xyxy:
            width = x2 - x1
            height = y2 - y1
            area_bbox = width * height
            # A pumpkin is round, not rectangular: the bbox overestimates its
            # real area by a factor of 4/pi (~27%). We approximate the pumpkin
            # as an ellipse inscribed in the bbox to correct for that bias.
            area_ellipse = (math.pi / 4) * area_bbox
            rows.append(
                {
                    "image": image_name,
                    "width_px": round(width, 1),
                    "height_px": round(height, 1),
                    "area_bbox_px": round(area_bbox, 1),
                    "area_ellipse_px": round(area_ellipse, 1),
                    "width_cm": round(width * scale_cm_per_px, 1),
                    "height_cm": round(height * scale_cm_per_px, 1),
                    "area_ellipse_cm2": round(area_ellipse * scale_cm_per_px**2, 1),
                }
            )

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "width_px",
                "height_px",
                "area_bbox_px",
                "area_ellipse_px",
                "width_cm",
                "height_cm",
                "area_ellipse_cm2",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    areas_cm2 = [row["area_ellipse_cm2"] for row in rows]
    print(f"Detected pumpkins: {len(rows)}")
    print(f"Scale: {scale_cm_per_px:.4f} cm/px")
    print(f"Average area (ellipse): {sum(areas_cm2) / len(areas_cm2):.1f} cm²")
    print(f"Min area (ellipse): {min(areas_cm2):.1f} cm²")
    print(f"Max area (ellipse): {max(areas_cm2):.1f} cm²")
    print(f"Results saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
