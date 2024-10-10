import sys

sys.path.append(".")

from concurrent.futures import ThreadPoolExecutor
from os.path import splitext
from pathlib import Path

import click
import cv2
import numpy as np
from tqdm import tqdm

from assertpy.assertpy import assert_that
from config import settings as conf


def compute_iou(file1, file2):
    mask1 = MASK_BANK[file1]
    mask2 = MASK_BANK[file2]

    mask2_len = len(mask2)
    iou_list = []

    for f, frame1 in enumerate(mask1):
        frame2 = mask2[f % mask2_len]

        if frame1.shape != frame2.shape:
            frame2 = cv2.resize(frame2, frame1.shape[::-1])

        intersection = np.logical_and(frame1, frame2).sum()
        union = np.logical_or(frame1, frame2).sum()
        iou = 0.0 if union == 0 else intersection / union
        iou_list.append(iou)

    return np.mean(iou_list)


ROOT = Path.cwd()
DATASET = conf.active.dataset
DATASET_DIR = ROOT / conf.datasets[DATASET].path
N_FILES = conf.datasets[DATASET].n_videos
DETECTOR = conf.active.detector
DET_CONFIDENCE = conf.detect[DETECTOR].confidence
MASK_DIR = ROOT / "data" / DATASET / DETECTOR / str(DET_CONFIDENCE) / "detect" / "mask"
IOU_PATH = MASK_DIR / "iou.npz"
MAX_WORKERS = conf.active.max_workers
MASK_BANK = {}
RESIZE_FACTOR = conf.iou.resize[DATASET]
file_list = []
file2class_map = {}

print("Input:", MASK_DIR)
print("n videos:", N_FILES)

assert_that(MASK_DIR).is_directory().is_readable()

if not click.confirm("\nDo you want to continue?", show_default=True):
    exit("Aborted.")

print("Loading masks...")

with open(DATASET_DIR / "list.txt") as f:
    for line in tqdm(f, total=N_FILES, dynamic_ncols=True):
        action, filename = line.split()[0].split("/")
        stem = splitext(filename)[0]

        file_list.append(stem)
        file2class_map[stem] = action

        mask_path = (MASK_DIR / action / filename).with_suffix(".npz")
        mask = np.load(mask_path)["arr_0"]

        if RESIZE_FACTOR < 1:
            height, width = mask.shape[1:]
            new_width = int(width * RESIZE_FACTOR)
            new_height = int(height * RESIZE_FACTOR)

            mask = cv2.resize(mask, (new_width, new_height))

        MASK_BANK[stem] = mask

if IOU_PATH.exists():
    data = np.load(IOU_PATH)["arr_0"]
    START_IDX = np.where(np.all(data == 0, axis=1))[0][0]
else:
    data = np.zeros((N_FILES, N_FILES), np.float16)
    START_IDX = 0

print(f"Working on {MAX_WORKERS} max workers...")

for file1_idx, file1 in enumerate(file_list):
    if file1_idx < START_IDX:
        continue

    file2_list = file_list[file1_idx + 1 :]
    jobs = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for file2 in file2_list:
            file2_idx = file_list.index(file2)
            jobs[(file1_idx, file2_idx)] = executor.submit(compute_iou, file1, file2)

        print(f"({file1_idx+1}/{len(file_list)})")

        for (i, j), job in tqdm(jobs.items(), total=len(jobs), dynamic_ncols=True):
            data[i, j] = job.result()

    if file1_idx % 10 == 0:
        print("Saving checkpoint...")
        np.savez_compressed(IOU_PATH, data)

np.savez_compressed(IOU_PATH, data)
