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
# Small slack for masks that extend a few pixels past the box that prompted them.
BBOX_MATCH_TOLERANCE = 5


def mask_fits_in_bbox(mask_rect, bbox):
    mx, my, mw, mh = mask_rect
    bx1, by1, bx2, by2 = bbox
    return (
        mx >= bx1 - BBOX_MATCH_TOLERANCE
        and my >= by1 - BBOX_MATCH_TOLERANCE
        and mx + mw <= bx2 + BBOX_MATCH_TOLERANCE
        and my + mh <= by2 + BBOX_MATCH_TOLERANCE
    )


def match_masks_to_boxes(masks, boxes):
    """Pair each mask with the box that produced it.

    If SAM dropped some masks (it scores its own mask quality and can drop
    low-scoring ones), the counts no longer match and position alone can't
    be trusted, fall back to walking both lists in order, matching each
    mask to the next box that contains it.
    """
    if len(masks) == len(boxes):
        return list(zip(boxes, masks))

    matched = []
    box_index = 0
    for mask in masks:
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        mask_rect = cv2.boundingRect(max(contours, key=cv2.contourArea))

        while box_index < len(boxes) and not mask_fits_in_bbox(
            mask_rect, boxes[box_index]
        ):
            box_index += 1
        if box_index >= len(boxes):
            break

        matched.append((boxes[box_index], mask))
        box_index += 1

    return matched


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
        sam_result = segmentation_model(
            image_path, bboxes=detections.xyxy, verbose=False
        )[0]
        masks = sv.Detections.from_ultralytics(sam_result).mask

        for detection_index, (bbox, mask) in enumerate(
            match_masks_to_boxes(masks, detections.xyxy)
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
