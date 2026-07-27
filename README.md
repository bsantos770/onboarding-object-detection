# Pumpkin Object Detection

Onboarding exercise: train an object detection model to identify pumpkins and measure the size of each one from its detected bounding box.

## Overview

1. Trained **YOLO11n** (Ultralytics) on a pumpkin dataset annotated in Roboflow
   (YOLOv12 format, 1 class: `pumpkins`).
2. Ran 3 experiments with different hyperparameters and tracked each one in
   **Weights & Biases**.
3. With the best model, ran inference over the whole validation set and measured
   the size (width, height, area) of each detected pumpkin using **supervision**.


### Dataset

The dataset is not included in the repo (see `.gitignore`). Download it from
[Google Drive](https://drive.google.com/file/d/15LYAlhzyoWVSGfVevL6DFimHxnEQV9Ir/view)
and unzip it into `dataset/`, resulting in the following path:

```
dataset/Pumpkins detection.v2i.yolov12/data.yaml
```

## Training

```bash
uv run train.py
```

Runs 3 experiments (defined in `experiments` inside `train.py`), each with a fresh
YOLO11n model, and logs every run separately to W&B:
[wandb.ai/bsantos7-eagerworks/pumpkin-detection](https://wandb.ai/bsantos7-eagerworks/pumpkin-detection).

Weights and metrics for each run are saved to `runs/pumpkins/<experiment_name>/`.

A `lr01_batch32` configuration was also tried, but it was dropped from the comparison:
each epoch took noticeably longer with no clear benefit over `batch=16`, so it wasn't
worth the extra training time at this stage.

Metrics below are W&B's summary for each run (evaluated on the `best.pt` checkpoint):

| Experiment | lr0 | batch | augmentation | precision | recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|---|
| `lr01_batch16` | 0.01 | 16 | no | **0.9473** | 0.9642 | 0.98824 | **0.8775** |
| `lr001_batch16` | 0.001 | 16 | no | 0.9402 | 0.96381 | 0.98697 | 0.87613 |
| `lr01_batch16_aug` | 0.01 | 16 | yes (mosaic + rotation) | 0.94983 | **0.96552** | **0.99073** | 0.82636 |

**`lr01_batch16`** was picked as the best model, it has the highest mAP50-95 and the
highest precision of the three, and its mAP50 is within 0.002 of the top score. The
augmentation used in `lr01_batch16_aug` gives the best mAP50 and recall, but its
mAP50-95 drops noticeably (0.826 vs 0.877), at this low epoch count the augmented
model likely needs more epochs to converge, so it isn't the better pick yet.

## Inference on a single image

```bash
uv run main.py "dataset/Pumpkins detection.v2i.yolov12/valid/images/<name>.jpg"
```

Runs the best model (`runs/pumpkins/lr01_batch16/weights/best.pt`) on the given image
and displays the annotated result (bounding boxes + labels) in a window using supervision.

## Size measurement

```bash
uv run measure_bounding_box.py
```

Runs the best model over **all** images in `valid/`, and for each detected pumpkin
computes width, height, and area in pixels from the bounding box (`detections.xyxy`).
Results are saved to [`pumpkin_sizes.csv`](pumpkin_sizes.csv) (one row per detected
pumpkin).

## Repo structure

```
train.py                  # trains and tracks the 3 experiments in W&B
main.py                   # inference + visualization on a single image
measure_bounding_box.py   # inference over validation + size measurement -> pumpkin_sizes.csv
pumpkin_sizes.csv          # measurement results (one row per pumpkin)
dataset/                  # downloaded dataset (gitignored)
runs/                     # weights and metrics per experiment (gitignored)
```
