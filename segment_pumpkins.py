import csv
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from ultralytics import SAM, YOLO

DETECTION_MODEL_PATH = "runs/pumpkins/lr01_batch16/weights/best.pt"
SEGMENTATION_MODEL_PATH = "sam2.1_b.pt"
VALID_IMAGES_DIR = "dataset/Pumpkins detection.v2i.yolov12/valid/images"
OUTPUT_CSV = "pumpkin_segmentation.csv"


def main():
    detection_model = YOLO(DETECTION_MODEL_PATH)
    segmentation_model = SAM(SEGMENTATION_MODEL_PATH)

    detection_results = detection_model.predict(source=VALID_IMAGES_DIR, stream=True)

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
        # masks will be a list of boolean arrays, one for each detected pumpkin,
        # true where the pumpkin is and false elsewhere
        masks = sv.Detections.from_ultralytics(sam_result).mask

        for mask in masks:
            # Compute the area of the mask (number of pixels that are True)
            area = int(mask.sum())

            # Find the largest contour in the mask and compute its minimum area rectangle
            # findContours returns a list of contours, each of which is an array of points
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            # largest_contour is the contour with the largest area
            largest_contour = max(contours, key=cv2.contourArea)
            # minAreaRect returns a rotated rectangle that bounds the contour, represented by its center, size (width, height), and rotation angle
            _, (w, h), _ = cv2.minAreaRect(largest_contour)

            rows.append(
                {
                    "image": image_name,
                    "area_px": area,
                    "longest_side_px": round(max(w, h), 1),
                }
            )

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "area_px", "longest_side_px"])
        writer.writeheader()
        writer.writerows(rows)

    areas = [row["area_px"] for row in rows]
    sides = [row["longest_side_px"] for row in rows]
    print(f"Calabazas segmentadas: {len(rows)}")
    print(f"Área promedio: {sum(areas) / len(areas):.1f} px²")
    print(f"Lado más largo promedio: {sum(sides) / len(sides):.1f} px")
    print(f"Resultados guardados en {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
