import csv
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from ultralytics import SAM, YOLO

import calibration

DETECTION_MODEL_PATH = "runs/pumpkins/lr01_batch16/weights/best.pt"
SEGMENTATION_MODEL_PATH = "sam2.1_b.pt"
TEST_IMAGES_DIR = "dataset/Pumpkins detection.v2i.yolov12/test/images"
OUTPUT_CSV = "pumpkin_segmentation.csv"
CONF_THRESHOLD = 0.5


def main():
    scale_cm_per_px = calibration.SCALE_CM_PER_PX

    detection_model = YOLO(DETECTION_MODEL_PATH)
    segmentation_model = SAM(SEGMENTATION_MODEL_PATH)

    detection_results = detection_model.predict(
        source=TEST_IMAGES_DIR, conf=CONF_THRESHOLD, stream=True
    )

    rows = []
    for result in detection_results:
        detections = sv.Detections.from_ultralytics(result)
        if len(detections) == 0:
            continue

        image_path = result.path
        image_name = Path(image_path).name

        # Use YOLO's boxes as prompts so SAM knows where each pumpkin is
        sam_result = segmentation_model(
            image_path, bboxes=detections.xyxy, verbose=False
        )[0]
        masks = sv.Detections.from_ultralytics(sam_result).mask

        for mask in masks:
            area = int(mask.sum())

            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            largest_contour = max(contours, key=cv2.contourArea)
            _, (w, h), _ = cv2.minAreaRect(largest_contour)

            longest_side_px = max(w, h)
            rows.append(
                {
                    "image": image_name,
                    "area_px": area,
                    "longest_side_px": round(longest_side_px, 1),
                    "area_cm2": round(area * scale_cm_per_px**2, 1),
                    "longest_side_cm": round(longest_side_px * scale_cm_per_px, 1),
                }
            )

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "area_px",
                "longest_side_px",
                "area_cm2",
                "longest_side_cm",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    areas_cm2 = [row["area_cm2"] for row in rows]
    sides_cm = [row["longest_side_cm"] for row in rows]
    print(f"Segmented pumpkins: {len(rows)}")
    print(f"Average area: {sum(areas_cm2) / len(areas_cm2):.1f} cm²")
    print(f"Average longest side: {sum(sides_cm) / len(sides_cm):.1f} cm")
    print(f"Results saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
