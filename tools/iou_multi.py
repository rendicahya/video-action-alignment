import sys

sys.path.append(".")

import concurrent.futures
from pathlib import Path

import click
import cv2
import numpy as np
import torch.multiprocessing as mp
from tqdm import tqdm

from assertpy.assertpy import assert_that
from config import settings as conf


def compute_iou(file1, file2):
    path1 = (MASK_DIR / file2class_map[file1] / file1).with_suffix(".npz")
    path2 = (MASK_DIR / file2class_map[file2] / file2).with_suffix(".npz")

    mask1 = np.load(path1)["arr_0"]
    mask2 = np.load(path2)["arr_0"]

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
N_FILES = conf.datasets[DATASET].n_videos
N_FILES = 100
DETECTOR = conf.active.detector
DET_CONFIDENCE = conf.detect[DETECTOR].confidence
MASK_DIR = ROOT / "data" / DATASET / DETECTOR / str(DET_CONFIDENCE) / "detect" / "mask"
N_CORES = mp.cpu_count()
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

file_list = file_list[:N_FILES]
data = np.zeros((N_FILES, N_FILES), np.float16)

with concurrent.futures.ThreadPoolExecutor(max_workers=N_CORES) as executor:
    futures = {}

    print("Preparing thread jobs...")

    for i, file1 in tqdm(enumerate(file_list), total=len(file_list)):
        file2_list = file_list[i + 1 : N_FILES]

        for file2 in file2_list:
            futures[(file1, file2)] = executor.submit(compute_iou, file1, file2)

    print(f"Executing on {N_CORES} threads...")

    for (file1, file2), future in tqdm(futures.items(), total=len(futures)):
        i = file_list.index(file1)
        j = file_list.index(file2)
        data[i, j] = future.result()

np.savez_compressed(MASK_DIR / "iou.npz", data)
np.save(MASK_DIR / "iou.npy", data)
