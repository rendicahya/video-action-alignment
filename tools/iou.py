import sys

sys.path.append(".")

from pathlib import Path

import click
import cv2
import numpy as np
from tqdm import tqdm

from assertpy.assertpy import assert_that
from config import settings as conf

ROOT = Path.cwd()
DATASET = conf.active.dataset
N_FILES = conf.datasets[DATASET].n_videos
DETECTOR = conf.active.detector
DET_CONFIDENCE = conf.detect[DETECTOR].confidence
MASK_DIR = ROOT / "data" / DATASET / DETECTOR / str(DET_CONFIDENCE) / "detect" / "mask"
file_list = []
file2class_map = {}

print("Input:", MASK_DIR)
print("n videos:", N_FILES)

assert_that(MASK_DIR).is_directory().is_readable()

if not click.confirm("\nDo you want to continue?", show_default=True):
    exit("Aborted.")

print("Building file list...")

for file in MASK_DIR.glob("**/*.npz"):
    if file.stem == "iou":
        continue

    file_list.append(file.stem)
    file2class_map[file.stem] = file.parent

bar = tqdm(total=N_FILES, dynamic_ncols=True)
data = np.zeros((N_FILES, N_FILES), np.float16)

for i, file1 in enumerate(file_list):
    bar.set_description(file1)

    path1 = (MASK_DIR / file2class_map[file1] / file1).with_suffix(".npz")
    mask1 = np.load(path1)["arr_0"]
    file2_list = file_list[i + 1 :]
    iou_list = []

    for j, file2 in enumerate(file2_list):
        bar.set_description(f"{file1} ({j}/{len(file2_list)})")

        path2 = (MASK_DIR / file2class_map[file2] / file2).with_suffix(".npz")
        mask2 = np.load(path2)["arr_0"]
        mask2_len = len(mask2)

        for f, frame1 in enumerate(mask1):
            frame2 = mask2[f % mask2_len]

            if frame1.shape != frame2.shape:
                frame2 = cv2.resize(frame2, frame1.shape[::-1])

            intersection = np.logical_and(frame1, frame2).sum()
            union = np.logical_or(frame1, frame2).sum()
            iou = 0.0 if union == 0 else intersection / union

            iou_list.append(iou)

        data[i, i + 1 + j] = np.mean(iou_list)

    bar.update(1)

bar.close()
np.savez_compressed(MASK_DIR / "iou.npz", data)
