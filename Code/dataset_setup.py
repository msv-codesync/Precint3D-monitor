#!/usr/bin/env python3
"""
dataset_setup.py  —  Organise your raw images into the required folder layout.

Usage:
  1. Put all your images anywhere, named like:
       spaghetti_failure_001.jpg, warping_032.jpg, good_print_005.jpg, etc.
     OR keep them in a flat folder and this script will copy them.

  2. Edit SOURCE_DIR and (optionally) DEST_DIR below.

  3. Run:  python dataset_setup.py
"""

import shutil
import pathlib

# ─── EDIT THESE ──────────────────────────────────────
SOURCE_DIR = pathlib.Path("./raw_images")   # your flat folder with all images
DEST_DIR   = pathlib.Path("./dataset")      # where the class folders will go
# ─────────────────────────────────────────────────────

CLASSES = [
    "spaghetti_failure",
    "warping",
    "first_layer_failure",
    "under_extrusion",
    "stringing",
    "blob_or_zit",
    "layer_shift",
    "nozzle_clog",
    "good_print",
]

def create_class_folders():
    for cls in CLASSES:
        folder = DEST_DIR / cls
        folder.mkdir(parents=True, exist_ok=True)
        print(f"  Created: {folder}")

def auto_sort_by_filename():
    """
    Copies images from SOURCE_DIR into DEST_DIR/<class>/
    if the filename starts with a known class name.
    Example: 'spaghetti_failure_001.jpg' → dataset/spaghetti_failure/
    """
    if not SOURCE_DIR.exists():
        print(f"[SKIP] SOURCE_DIR '{SOURCE_DIR}' not found. Create folders manually.")
        return

    moved = 0
    unmatched = []
    for img in SOURCE_DIR.iterdir():
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            continue
        matched = False
        for cls in CLASSES:
            if img.name.lower().startswith(cls):
                dest = DEST_DIR / cls / img.name
                shutil.copy2(img, dest)
                moved += 1
                matched = True
                break
        if not matched:
            unmatched.append(img.name)

    print(f"\n  Auto-sorted {moved} images.")
    if unmatched:
        print(f"  Could not match {len(unmatched)} files (move them manually):")
        for u in unmatched[:10]:
            print(f"    {u}")
        if len(unmatched) > 10:
            print(f"    ... and {len(unmatched)-10} more")

def count_dataset():
    print("\n  Dataset summary:")
    total = 0
    for cls in CLASSES:
        folder = DEST_DIR / cls
        count = len(list(folder.glob("*"))) if folder.exists() else 0
        bar = "█" * (count // 5)
        status = "✓" if count >= 50 else "⚠ (add more images)" if count > 0 else "✗ EMPTY"
        print(f"    {cls:<25} {count:>4} images  {bar}  {status}")
        total += count
    print(f"\n  Total: {total} images across {len(CLASSES)} classes")
    if total < 500:
        print("  ⚠  Aim for at least 100 images per class for decent accuracy.")

if __name__ == "__main__":
    print("=== Dataset Setup ===\n")
    print("Creating class folders in:", DEST_DIR)
    create_class_folders()
    auto_sort_by_filename()
    count_dataset()
    print("\nDone. Now run: python train_model.py")
