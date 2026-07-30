"""
Splits the original `valid/` split into `valid/` and `test/`.

"""

import random
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_ROOT = BASE_DIR / "dataset" / "Pumpkins detection.v2i.yolov12"
VALID_DIR = DATASET_ROOT / "valid"
TEST_DIR = DATASET_ROOT / "test"
SEED = 42
TEST_FRACTION = 0.5


def main():
    if TEST_DIR.exists():
        print(f"{TEST_DIR} already exists.")
        return

    image_paths = sorted((VALID_DIR / "images").iterdir())
    stems = [p.stem for p in image_paths]

    random.Random(SEED).shuffle(stems)
    n_test = round(len(stems) * TEST_FRACTION)
    test_stems = set(stems[:n_test])

    (TEST_DIR / "images").mkdir(parents=True)
    (TEST_DIR / "labels").mkdir(parents=True)

    moved = 0
    for image_path in image_paths:
        stem = image_path.stem
        if stem not in test_stems:
            continue
        label_path = VALID_DIR / "labels" / f"{stem}.txt"

        shutil.move(str(image_path), TEST_DIR / "images" / image_path.name)
        shutil.move(str(label_path), TEST_DIR / "labels" / label_path.name)
        moved += 1

    remaining = len(list((VALID_DIR / "images").iterdir()))
    print(f"Moved {moved} images from valid/ to test/.")
    print(f"valid/ now has {remaining} images, test/ has {moved} images.")


if __name__ == "__main__":
    main()
