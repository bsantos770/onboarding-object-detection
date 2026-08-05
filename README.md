# Pumpkin Object Detection

Onboarding exercise: train an object detection model to identify pumpkins and measure the size of each one from its detected bounding box.

## Overview

1. Trained **YOLO11n** (Ultralytics) on a pumpkin dataset annotated in Roboflow
   (YOLOv12 format, 1 class: `pumpkins`).
2. Ran 3 experiments with different hyperparameters and tracked each one in
   **Weights & Biases**.
3. With the best model, ran inference over the whole validation set and measured
   the size (width, height, area) of each detected pumpkin using **supervision**.
4. Segmented each pumpkin two ways: prompting **SAM2** with the detected boxes, and
   fine-tuning an **RF-DETR-Seg-Nano** model to predict masks directly.


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

Runs the experiments defined in `experiments` inside `train.py`, each with a fresh
YOLO11n model, and logs every run separately to W&B:
[wandb.ai/bsantos7-eagerworks/pumpkin-detection](https://wandb.ai/bsantos7-eagerworks/pumpkin-detection).

Weights and metrics for each run are saved to `runs/pumpkins/<experiment_name>/`.

A `lr01_batch32` configuration was also tried, but it was dropped from the comparison:
each epoch took noticeably longer with no clear benefit over `batch=16`, so it wasn't
worth the extra training time at this stage.

Model selection is based on each run's metrics on `valid/` (W&B's summary, evaluated
on the `best.pt` checkpoint) — this is only used to **pick** a winner, not to report
final performance:

| Experiment | lr0 | batch | augmentation | precision | recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|---|
| `lr01_batch16` | 0.01 | 16 | default (mosaic never active) | 0.934 | **0.985** | 0.990 | **0.879** |
| `lr001_batch16` | 0.001 | 16 | default (mosaic never active) | 0.950 | 0.976 | 0.990 | 0.873 |
| `lr01_batch16_aug` | 0.01 | 16 | default + mosaic (epochs 1-5) + rotation | **0.962** | 0.943 | 0.977 | 0.802 |
| `lr01_batch16_aug_scale` | 0.01 | 16 | default + scale=1.0 (mosaic never active) | 0.927 | 0.946 | 0.976 | 0.839 |

All four use Ultralytics' default augmentation (hsv jitter, scale=0.5, fliplr) — "default"
here just means no extra params were set on top of that.

Results are close between `lr01_batch16` and `lr001_batch16`, so we go with
**`lr01_batch16`** — it uses Ultralytics' default `lr0`. The augmented runs score lower on mAP50-95, so augmentation hurts at this epoch count.

### Final metrics (held out test set)

```bash
uv run evaluate_final.py <winning_experiment_name>
```

Once a winner is picked from the table above, this loads its `best.pt` and evaluates it
on `test/` — the 25 images set aside by `split_val_test.py` that never influenced training
or model selection. These are the numbers that should be reported as the project's actual
performance, not the `valid/` ones used to pick the winner.

| precision | recall | mAP50 | mAP50-95 |
|---|---|---|---|
| 0.9534 | 0.9746 | 0.9900 | 0.8881 |

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

Runs the best model over **all** images in `test/` and
for each detected pumpkin computes width, height, and area from the bounding box, both
in pixels and in cm (see [calibration.py](calibration.py)).

Two area columns are saved: `area_bbox_px` (the raw `width * height`) and
`area_ellipse_px` (`area_bbox_px * π/4`). A pumpkin is round, not rectangular, so the
bounding box overestimates its real area by a factor of `4/π` (~27%), approximating it
as an ellipse inscribed in the box corrects for that.

**Known limitations:**
- The cm conversion assumes the trough is 100cm wide and that pumpkins sit on the same
  plane as it.
- We assume the images didn't go through any resize before we got them.

Results are saved to [`pumpkin_sizes.csv`](pumpkin_sizes.csv) (one row per detected
pumpkin).

## Segmentation

Two approaches were tried to get a pixel-level mask for each pumpkin, beyond the
detection model's bounding box.

### SAM2 (prompted by YOLO boxes)

```bash
uv run segment_pumpkins.py
```

For each image in `test/`, runs the best YOLO model to get bounding boxes, then
prompts **SAM2** (`sam2.1_b.pt`) with those boxes (with `conf=0.0`, so SAM never drops
a low-quality mask) so it knows where to segment. For each resulting mask, computes
area and longest side (via `cv2.minAreaRect`, which captures the pumpkin's true size
regardless of orientation), in pixels and in cm (see [calibration.py](calibration.py)),
along with `mask_quality`, SAM's own confidence score for that mask, so a bad
segmentation stays visible and traceable to its box instead of silently disappearing.

Results are saved to [`pumpkin_segmentation.csv`](pumpkin_segmentation.csv) (one row
per segmented pumpkin).

### RF-DETR-Seg (fine-tuned)

The original dataset only has bounding box annotations, so masks were auto-labeled instead of
hand-drawn, this runs the trained YOLO detector (`lr01_batch16/weights/best.pt`) over every
image (`train/`, `valid/`, `test/`) and prompts SAM with each detected box to get a
mask. The largest contour of each mask is extracted and saved in YOLO-segmentation format to
`roboflow_export/`, which was zipped and uploaded to Roboflow, reviewed, and exported as
`pumpkin-segmentation.coco-segmentation`.

```bash
uv run train_segmentation.py
```

Fine-tunes `RFDETRSegNano` for 10 epochs. Results obtained on the test set were:

| mAP50 (bbox) | mAP50-95 (bbox) | precision | recall | segm mAP50 | segm mAP50-95 |
|---|---|---|---|---|---|
| 0.975 | 0.870 | 0.966 | 0.955 | 0.978 | 0.834 |

## Repo structure

```
split_val_test.py         # one-time setup: carves test/ out of valid/ (fixed seed)
train.py                  # trains and tracks the experiments in W&B, model selection on valid/
evaluate_final.py         # evaluates the winning experiment on test/ -> final reported metrics
main.py                   # inference + visualization on a single image
measure_bounding_box.py   # inference over validation + size measurement -> pumpkin_sizes.csv
segment_pumpkins.py       # inference + SAM2 segmentation -> pumpkin_segmentation.csv
train_segmentation.py     # fine-tunes RF-DETR-Seg-Nano on the segmentation dataset, tracked in W&B
pumpkin_sizes.csv         # bounding box measurement results (one row per pumpkin)
pumpkin_segmentation.csv  # segmentation measurement results (one row per pumpkin)
dataset/                  # downloaded dataset (gitignored)
runs/                     # weights and metrics per experiment (gitignored)
```
