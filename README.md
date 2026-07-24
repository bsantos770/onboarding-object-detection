# Pumpkin Object Detection

Onboarding exercise: train an object detection model to identify pumpkins and measure the size of each one from its detected bounding box.

## What was done

1. Trained **YOLO11n** (Ultralytics) on a pumpkin dataset annotated in Roboflow
   (YOLOv12 format, 1 class: `pumpkins`).
2. Ran 3 experiments with different hyperparameters and tracked each one in
   **Weights & Biases**.
3. With the best model, ran inference over the whole validation set and measured
   the size (width, height, area) of each detected pumpkin using **supervision**.


### Dataset

The dataset is not included in the repo (see `.gitignore`). Download it from
[Google Drive](https://drive.google.com/file/d/15LYAlhzyoWVSGfVevL6DFimHxnEQV9Ir/view)
and unzip it into `dataset/`, so it looks like:

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

Weights and metrics for each run are saved to `runs/pumpkins/<experiment_name>/`
(gitignored, regenerated on training).

| Experiment | lr0 | batch | augmentation | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| `lr01_batch16` | 0.01 | 16 | no | 0.9905 | **0.8743** |
| `lr001_batch16` | 0.001 | 16 | no | 0.9870 | 0.8767 |
| `lr01_batch16_aug` | 0.01 | 16 | yes (mosaic + rotation) | **0.9907** | 0.8264 |

All three are essentially tied on mAP50. **`lr01_batch16`** was picked as the best
model: its mAP50-95 is nearly identical to `lr001_batch16` (0.002 difference) but with better recall, while the augmentation used in `lr01_batch16_aug` hurts mAP50-95 rather than helping at this low epoch count. This is a judgment call, not a decisive gap — with more training epochs it would be worth re-running the comparison.

## Inference on a single image

```bash
uv run main.py "dataset/Pumpkins detection.v2i.yolov12/valid/images/<name>.jpg"
```

Runs the best model (`runs/pumpkins/lr01_batch16/weights/best.pt`) on the given image
and displays the annotated result (bounding boxes + labels) in a window.

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
