from rfdetr import RFDETRSegNano

from config import RFDETR_DATASET_DIR, RFDETR_OUTPUT_DIR

EPOCHS = 10
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8
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
        progress_bar="rich",
        run_test=True,
    )


if __name__ == "__main__":
    main()
