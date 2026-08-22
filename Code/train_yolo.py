"""
train_yolo.py
=============
Train a YOLO11s model on the consolidated 3D print failure dataset.
Uses PyTorch MPS (Apple Metal) automatically on M2 Pro.

Run:
  python train_yolo.py

Output:
  runs/detect/print_guard/weights/best.pt   ← best model
  runs/detect/print_guard/weights/last.pt   ← last epoch
"""

from ultralytics import YOLO
import pathlib
import json

# ─── Configuration ────────────────────────────────────────────────────────────
DATA_YAML   = pathlib.Path("./dataset_yolo/data.yaml")   # produced by prepare_dataset.py
MODEL_BASE  = "yolo11s.pt"      # yolo11s — very high accuracy, trains overnight ~12 hrs
PROJECT     = "runs/detect"
RUN_NAME    = "print_guard_overnight"

EPOCHS      = 100               # with early stopping; usually converges by ~50
IMG_SIZE    = 640
BATCH       = 16                # optimal for M2 Pro 32 GB unified memory
WORKERS     = 4
PATIENCE    = 20                # early stop if no improvement for 20 epochs
CONF_THRES  = 0.25              # confidence threshold for detection
IOU_THRES   = 0.45              # NMS IoU threshold
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if not DATA_YAML.exists():
        print(f"[ERROR] {DATA_YAML} not found.")
        print("  Run:  python prepare_dataset.py  first.")
        return

    print("=" * 60)
    print("  3D Print Failure Detector — YOLO11 Training")
    print("=" * 60)
    print(f"\n  Model   : {MODEL_BASE}")
    print(f"  Data    : {DATA_YAML.resolve()}")
    print(f"  Epochs  : {EPOCHS} (early stop patience={PATIENCE})")
    print(f"  Batch   : {BATCH}")
    print(f"  Device  : auto (will use M2 Metal MPS)\n")

    # Load pretrained YOLO11s (downloads ~18 MB if not cached)
    model = YOLO(MODEL_BASE)

    # Train
    results = model.train(
        data     = str(DATA_YAML.resolve()),
        epochs   = EPOCHS,
        imgsz    = IMG_SIZE,
        batch    = BATCH,
        workers  = WORKERS,
        patience = PATIENCE,
        project  = PROJECT,
        name     = RUN_NAME,
        device   = "mps",          # Apple Metal GPU — M2 Pro
        exist_ok = True,
        # Augmentation — strong, helps with rare classes like nozzle_clog
        hsv_h    = 0.015,
        hsv_s    = 0.7,
        hsv_v    = 0.4,
        degrees  = 10.0,
        translate= 0.1,
        scale    = 0.5,
        shear    = 3.0,
        perspective = 0.0003,
        flipud   = 0.05,
        fliplr   = 0.5,
        mosaic   = 1.0,
        mixup    = 0.15,
        copy_paste = 0.1,          # helps rare classes like nozzle_clog
        # Optimizer — cosine LR schedule for best convergence
        optimizer= "AdamW",
        lr0      = 0.001,
        lrf      = 0.01,
        weight_decay = 0.0005,
        warmup_epochs= 3,
        cos_lr   = True,           # cosine annealing
        label_smoothing = 0.05,    # reduces overconfidence
        # Confidence / NMS
        conf     = CONF_THRES,
        iou      = IOU_THRES,
        # Save
        save     = True,
        save_period = 5,           # checkpoint every 5 epochs
        plots    = True,
        verbose  = True,
    )

    best_pt = pathlib.Path(PROJECT) / RUN_NAME / "weights" / "best.pt"
    print("\n" + "=" * 60)
    print("  Training complete!")
    print("=" * 60)
    print(f"\n  Best model : {best_pt.resolve()}")

    # Validate on test set
    print("\n  Running validation on test set...")
    metrics = model.val(
        data   = str(DATA_YAML.resolve()),
        split  = "test",
        conf   = CONF_THRES,
        iou    = IOU_THRES,
    )

    # Save class names alongside best.pt for the inference script
    names_path = best_pt.parent / "class_names.json"
    with open(names_path, "w") as f:
        json.dump(model.names, f, indent=2)
    print(f"  Class names saved → {names_path}")

    print(f"\n  mAP50     : {metrics.box.map50:.4f}")
    print(f"  mAP50-95  : {metrics.box.map:.4f}")
    print(f"\nNext step:\n  python predict_and_act.py path/to/printer_frame.jpg")

if __name__ == "__main__":
    main()
