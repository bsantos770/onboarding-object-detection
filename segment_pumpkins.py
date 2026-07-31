import csv
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from ultralytics import SAM, YOLO

import calibration
from config import MODEL_PATH, TEST_IMAGES_DIR

SEGMENTATION_MODEL_PATH = "sam2.1_b.pt"
OUTPUT_CSV = "pumpkin_segmentation.csv"
CONF_THRESHOLD = 0.5


def main():
    scale_cm_per_px = calibration.SCALE_CM_PER_PX

    detection_model = YOLO(MODEL_PATH)
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

        # Use YOLO's boxes as prompts so SAM knows where each pumpkin is.
        # conf=0.0: SAM scores its own mask quality and drops low-scoring masks
        # by default. We keep every mask instead and record its quality score,
        # so a bad mask stays visible in the data (and traceable to its box)
        # instead of silently disappearing.
        sam_result = segmentation_model(
            image_path, bboxes=detections.xyxy, conf=0.0, verbose=False
        )[0]
        sam_detections = sv.Detections.from_ultralytics(sam_result)
        masks = sam_detections.mask
        mask_quality = sam_detections.confidence

        # With conf=0.0, SAM can no longer drop masks, so this should be
        # structurally impossible, if it still happens, something is broken.
        if masks is None or len(masks) != len(detections.xyxy):
            raise RuntimeError(f"{image_name}: expected one mask per box")

        for detection_index, (bbox, mask, quality) in enumerate(
            zip(detections.xyxy, masks, mask_quality)
        ):
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue
            largest_contour = max(contours, key=cv2.contourArea)
            _, (w, h), _ = cv2.minAreaRect(largest_contour)
            longest_side_px = max(w, h)
            area = int(mask.sum())

            x1, y1, x2, y2 = bbox
            rows.append(
                {
                    "image": image_name,
                    "detection_index": detection_index,
                    "bbox_x1": round(float(x1), 1),
                    "bbox_y1": round(float(y1), 1),
                    "bbox_x2": round(float(x2), 1),
                    "bbox_y2": round(float(y2), 1),
                    "mask_quality": round(float(quality), 4),
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
                "detection_index",
                "bbox_x1",
                "bbox_y1",
                "bbox_x2",
                "bbox_y2",
                "mask_quality",
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
