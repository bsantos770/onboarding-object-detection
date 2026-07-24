from pathlib import Path
import wandb
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
DATA_YAML = BASE_DIR / "dataset" / "Pumpkins detection.v2i.yolov12" / "data.yaml"
DEVICE = "mps"
PROJECT_DIR = BASE_DIR / "runs" / "pumpkins"
WANDB_ENTITY = "bsantos7-eagerworks"
WANDB_PROJECT = "pumpkin-detection"

# ---- Experimentos a correr ----
experiments = [
    {"name": "lr01_batch16", "lr0": 0.01, "batch": 16, "epochs": 10},
    {"name": "lr001_batch16", "lr0": 0.001, "batch": 16, "epochs": 10},
    {
        "name": "lr01_batch16_aug",
        "lr0": 0.01,
        "batch": 16,
        "epochs": 10,
        "mosaic": 1.0,
        "degrees": 10.0,
    },
]

for cfg in experiments:
    name = cfg.pop("name")
    print(f"\n=== Corriendo experimento: {name} ===")

    # Un run de W&B por experimento
    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=name,
        config=cfg,
        reinit=True,
    )

    model = YOLO("yolo11n.pt")
    model.train(
        data=DATA_YAML,
        imgsz=640,
        device=DEVICE,
        project=PROJECT_DIR,
        name=name,
        **cfg,
    )

    run.finish()

# print("\nListo. Para ver los experimentos en tu dashboard de W&B:")
# print(f"https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}")
