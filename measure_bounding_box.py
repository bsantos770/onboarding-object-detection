import csv
from pathlib import Path
import supervision as sv
from ultralytics import YOLO

MODEL_PATH = "runs/pumpkins/lr01_batch16/weights/best.pt"
VALID_IMAGES_DIR = "dataset/Pumpkins detection.v2i.yolov12/valid/images"
OUTPUT_CSV = "pumpkin_sizes.csv"

model = YOLO(MODEL_PATH)
results = model.predict(source=VALID_IMAGES_DIR, stream=True)

rows = []
for result in results:
    # convert to supervision format
    detections = sv.Detections.from_ultralytics(result)
    image_name = Path(result.path).name

    for x1, y1, x2, y2 in detections.xyxy:
        width = x2 - x1
        height = y2 - y1
        rows.append(
            {
                "image": image_name,
                "width_px": round(width, 1),
                "height_px": round(height, 1),
                "area_px": round(width * height, 1),
            }
        )

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["image", "width_px", "height_px", "area_px"])
    writer.writeheader()
    writer.writerows(rows)

areas = [row["area_px"] for row in rows]
print(f"Calabazas detectadas: {len(rows)}")
print(f"Área promedio: {sum(areas) / len(areas):.1f} px²")
print(f"Área mínima: {min(areas):.1f} px²")
print(f"Área máxima: {max(areas):.1f} px²")
print(f"Resultados guardados en {OUTPUT_CSV}")
