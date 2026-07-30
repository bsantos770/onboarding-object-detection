from pathlib import Path

from ultralytics import YOLO, settings

import wandb

BASE_DIR = Path(__file__).resolve().parent
DATA_YAML = BASE_DIR / "dataset" / "Pumpkins detection.v2i.yolov12" / "data.yaml"
PROJECT_DIR = BASE_DIR / "runs" / "pumpkins"
WANDB_ENTITY = "bsantos7-eagerworks"
WANDB_PROJECT = "pumpkin-detection"

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
        "close_mosaic": 5,
        "scale": 1.0,
    },
    {
        "name": "lr01_batch16_aug_scale",
        "lr0": 0.01,
        "batch": 16,
        "epochs": 10,
        "scale": 1.0,
    },
]


def main():
    settings.update({"wandb": True})

    for cfg in experiments:
        name = cfg.pop("name")
        print(f"\n=== Running experiment: {name} ===")

        run = wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            name=name,
            config=cfg,
            reinit=True,
        )

        model = YOLO("yolo11n.pt")

        def log_full_config(trainer, run=run):
            run.config.update(vars(trainer.args), allow_val_change=True)

        model.add_callback("on_pretrain_routine_start", log_full_config)

        model.train(
            data=DATA_YAML,
            imgsz=640,
            project=PROJECT_DIR,
            name=name,
            **cfg,
        )


if __name__ == "__main__":
    main()
