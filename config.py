from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_ROOT = BASE_DIR / "dataset" / "Pumpkins detection.v2i.yolov12"
DATA_YAML = DATASET_ROOT / "data.yaml"
TEST_IMAGES_DIR = DATASET_ROOT / "test" / "images"
PROJECT_DIR = BASE_DIR / "runs" / "pumpkins"
MODEL_PATH = PROJECT_DIR / "lr01_batch16" / "weights" / "best.pt"
