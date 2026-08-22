"""
prepare_dataset.py
==================
Consolidates the 3 Roboflow YOLO datasets into one clean 9-class dataset.

Source class mess → clean 9-class action mapping:
  spaghetti_failure  (0) : all spaghetti variants
  warping            (1) : warping
  first_layer_failure(2) : bed adhesion failures
  under_extrusion    (3) : fail, printfail
  stringing          (4) : Stringing, stringing
  blob_or_zit        (5) : Blobs
  layer_shift        (6) : Crack
  nozzle_clog        (7) : nozzle blob failure
  good_print         (8) : printpass
"""

import os
import shutil
import pathlib
from collections import defaultdict

# ── Target classes (maps to printer actions) ────────────────────────────────
TARGET_CLASSES = [
    "spaghetti_failure",   # 0
    "warping",             # 1
    "first_layer_failure", # 2
    "under_extrusion",     # 3
    "stringing",           # 4
    "blob_or_zit",         # 5
    "layer_shift",         # 6
    "nozzle_clog",         # 7
    "good_print",          # 8
]

# ── Source datasets to merge ─────────────────────────────────────────────────
DATASETS = [
    {
        "name": "3d-defects-starter (MAIN)",
        "path": pathlib.Path("./3d-defects-starter.yolov11"),
        "classes": [
            '-spaghetti-printpass-prin-6ROn',  # 0  → spaghetti_failure
            'Blobs',                            # 1  → blob_or_zit
            'Crack',                            # 2  → layer_shift
            'Spagatti',                         # 3  → spaghetti_failure
            'Spaghetti',                        # 4  → spaghetti_failure
            'Stringing',                        # 5  → stringing
            'bed adhesion failure',             # 6  → first_layer_failure
            'fail',                             # 7  → under_extrusion
            'nozzle blob failure',              # 8  → nozzle_clog
            'poor initial layer bed adhesion faiure',  # 9 → first_layer_failure
            'printfail',                        # 10 → under_extrusion
            'printpass',                        # 11 → good_print
            'spaghetti',                        # 12 → spaghetti_failure
            'spahgetti',                        # 13 → spaghetti_failure
            'stringing',                        # 14 → stringing
            'warping',                          # 15 → warping
        ],
        "remap": {
            0: 0,  # -spaghetti... → spaghetti_failure
            1: 5,  # Blobs → blob_or_zit
            2: 6,  # Crack → layer_shift
            3: 0,  # Spagatti → spaghetti_failure
            4: 0,  # Spaghetti → spaghetti_failure
            5: 4,  # Stringing → stringing
            6: 2,  # bed adhesion failure → first_layer_failure
            7: 3,  # fail → under_extrusion
            8: 7,  # nozzle blob failure → nozzle_clog
            9: 2,  # poor initial layer bed adhesion → first_layer_failure
            10: 3, # printfail → under_extrusion
            11: 8, # printpass → good_print
            12: 0, # spaghetti → spaghetti_failure
            13: 0, # spahgetti → spaghetti_failure
            14: 4, # stringing → stringing
            15: 1, # warping → warping
        },
        "weight": 1,  # use all images
    },
    {
        "name": "printer-spaghetti",
        "path": pathlib.Path("./printer spaghetti.yolov11"),
        "classes": [
            'Blobs',                            # 0  → blob_or_zit
            'Crack',                            # 1  → layer_shift
            'Spagatti',                         # 2  → spaghetti_failure
            'Spaghetti',                        # 3  → spaghetti_failure
            'Stringing',                        # 4  → stringing
            'bed adhesion failure',             # 5  → first_layer_failure
            'fail',                             # 6  → under_extrusion
            'nozzle blob failure',              # 7  → nozzle_clog
            'poor initial layer bed adhesion faiure',  # 8 → first_layer_failure
            'spaghetti',                        # 9  → spaghetti_failure
            'spahgetti',                        # 10 → spaghetti_failure
            'stringing',                        # 11 → stringing
        ],
        "remap": {
            0: 5,  # Blobs → blob_or_zit
            1: 6,  # Crack → layer_shift
            2: 0,  # Spagatti → spaghetti_failure
            3: 0,  # Spaghetti → spaghetti_failure
            4: 4,  # Stringing → stringing
            5: 2,  # bed adhesion failure → first_layer_failure
            6: 3,  # fail → under_extrusion
            7: 7,  # nozzle blob failure → nozzle_clog
            8: 2,  # poor initial layer bed adhesion → first_layer_failure
            9: 0,  # spaghetti → spaghetti_failure
            10: 0, # spahgetti → spaghetti_failure
            11: 4, # stringing → stringing
        },
        "weight": 1,
    },
    {
        "name": "3d-print (pass/fail)",
        "path": pathlib.Path("./3d print.yolov11"),
        "classes": ['printfail', 'printpass'],
        "remap": {
            0: 3,  # printfail → under_extrusion
            1: 8,  # printpass → good_print
        },
        "weight": 1,
    },
]

OUT_DIR = pathlib.Path("./dataset_yolo")

def remap_label_file(src_label, dst_label, remap):
    """Read a YOLO label file, remap class IDs, write to dst."""
    lines_out = []
    with open(src_label) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            old_cls = int(parts[0])
            new_cls = remap.get(old_cls)
            if new_cls is None:
                continue  # skip unknown classes
            parts[0] = str(new_cls)
            lines_out.append(" ".join(parts))
    if lines_out:
        dst_label.parent.mkdir(parents=True, exist_ok=True)
        with open(dst_label, "w") as f:
            f.write("\n".join(lines_out) + "\n")
        return True
    return False

def process_split(dataset, split, counts):
    src_images = dataset["path"] / split / "images"
    src_labels = dataset["path"] / split / "labels"
    if not src_images.exists():
        return 0

    dst_images = OUT_DIR / split / "images"
    dst_labels = OUT_DIR / split / "labels"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    ds_name = dataset["name"].replace(" ", "_").replace("/", "_")
    copied = 0

    for img_path in src_images.iterdir():
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            continue

        label_path = src_labels / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        # Unique filename to avoid collisions across datasets
        unique_stem = f"{ds_name}_{img_path.name}"
        dst_img = dst_images / unique_stem
        dst_lbl = dst_labels / (pathlib.Path(unique_stem).stem + ".txt")

        ok = remap_label_file(label_path, dst_lbl, dataset["remap"])
        if ok:
            shutil.copy2(img_path, dst_img)
            # Count by class
            with open(dst_lbl) as f:
                for line in f:
                    cls = int(line.split()[0])
                    counts[split][cls] += 1
            copied += 1

    return copied

def write_data_yaml():
    yaml_content = f"""train: {(OUT_DIR / 'train' / 'images').resolve()}
val:   {(OUT_DIR / 'valid' / 'images').resolve()}
test:  {(OUT_DIR / 'test'  / 'images').resolve()}

nc: {len(TARGET_CLASSES)}
names: {TARGET_CLASSES}
"""
    yaml_path = OUT_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    return yaml_path

def main():
    print("=" * 60)
    print("  3D Print Failure Dataset Consolidation")
    print("=" * 60)

    if OUT_DIR.exists():
        print(f"\n⚠  Removing existing {OUT_DIR} ...")
        shutil.rmtree(OUT_DIR)

    counts = {
        "train": defaultdict(int),
        "valid": defaultdict(int),
        "test":  defaultdict(int),
    }

    for ds in DATASETS:
        if not ds["path"].exists():
            print(f"\n⚠  Skipping '{ds['name']}' — path not found: {ds['path']}")
            continue
        print(f"\n📦 Processing: {ds['name']}")
        for split in ["train", "valid", "test"]:
            n = process_split(ds, split, counts)
            print(f"   {split:6s}: {n:5d} images")

    yaml_path = write_data_yaml()

    print("\n" + "=" * 60)
    print("  Class distribution in merged dataset")
    print("=" * 60)
    print(f"\n  {'Class':<25} {'Train':>7} {'Valid':>7} {'Test':>7}")
    print("  " + "-" * 46)
    for i, name in enumerate(TARGET_CLASSES):
        t = counts["train"][i]
        v = counts["valid"][i]
        te = counts["test"][i]
        warn = "  ⚠ LOW" if t < 50 else ""
        print(f"  {name:<25} {t:>7} {v:>7} {te:>7}{warn}")

    total = sum(counts["train"].values())
    print(f"\n  Total training annotations: {total}")
    print(f"\n✅ Dataset ready at: {OUT_DIR.resolve()}")
    print(f"   YAML config    : {yaml_path.resolve()}")
    print("\nNext step: python train_yolo.py")

if __name__ == "__main__":
    main()
