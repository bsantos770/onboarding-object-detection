import argparse

import cv2
import supervision as sv
from ultralytics import YOLO

MODEL_PATH = "runs/pumpkins/lr01_batch16/weights/best.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    args = parser.parse_args()

    model = YOLO(MODEL_PATH)

    image = cv2.imread(args.image_path)
    results = model(image)[0]
    detections = sv.Detections.from_ultralytics(results)

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    annotated_image = box_annotator.annotate(scene=image, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections
    )

    sv.plot_image(annotated_image)


if __name__ == "__main__":
    main()
