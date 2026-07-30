"""
Evaluates the winning model on test/, which never took part in training or model selection.
"""

import sys
from pathlib import Path

from ultralytics import YOLO

import wandb

BASE_DIR = Path(__file__).resolve().parent
DATA_YAML = BASE_DIR / "dataset" / "Pumpkins detection.v2i.yolov12" / "data.yaml"
PROJECT_DIR = BASE_DIR / "runs" / "pumpkins"
WANDB_ENTITY = "bsantos7-eagerworks"
WANDB_PROJECT = "pumpkin-detection"


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: uv run evaluate_final.py <winning_experiment_name>")

    winner_name = sys.argv[1]
    weights_path = PROJECT_DIR / winner_name / "weights" / "best.pt"
    if not weights_path.exists():
        sys.exit(
            f"Couldn't find {weights_path}. Did you run train.py with that experiment name?"
        )

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=f"{winner_name}_test_final",
        reinit=True,
    )

    model = YOLO(weights_path)
    # split="test" instead of the default ("val") -- this is what makes the
    # metric come from data that never took part in any decision
    metrics = model.val(data=DATA_YAML, split="test")

    summary = {
        "precision": metrics.box.mp,
        "recall": metrics.box.mr,
        "mAP50": metrics.box.map50,
        "mAP50-95": metrics.box.map,
    }
    run.log(summary)
    run.finish()

    print(f"\n=== Final metrics for '{winner_name}' on test/ ===")
    for key, value in summary.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
