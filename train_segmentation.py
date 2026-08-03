from rfdetr import RFDETRSegNano

from config import RFDETR_DATASET_DIR, RFDETR_OUTPUT_DIR

EPOCHS = 1  # minimal test run, not a real training
BATCH_SIZE = 1  # kept small, batch_size=4 ran MPS out of memory on masks
GRAD_ACCUM_STEPS = 1  # matches the minimal test, not a real training
NUM_WORKERS = 2
WANDB_PROJECT = "pumpkin-segmentation"
RUN_NAME = "rfdetr_seg_nano"


def main():
    model = RFDETRSegNano(amp=True)
    model.train(
        dataset_dir=str(RFDETR_DATASET_DIR),
        output_dir=str(RFDETR_OUTPUT_DIR),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        grad_accum_steps=GRAD_ACCUM_STEPS,
        num_workers=NUM_WORKERS,
        wandb=True,
        project=WANDB_PROJECT,
        run=RUN_NAME,
    )


if __name__ == "__main__":
    main()
