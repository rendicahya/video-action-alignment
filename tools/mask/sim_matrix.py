import sys

sys.path.append(".")

import datetime
import os
import time
from concurrent.futures import ThreadPoolExecutor
from os.path import splitext
from pathlib import Path
from tempfile import mkdtemp

import click
import cv2
import numpy as np
import psutil
from tqdm import tqdm

from assertpy.assertpy import assert_that
from config import settings as conf


def compute_sim(file1, file2, method):
    mask1 = mask_bank[file1]
    mask2 = mask_bank[file2]

    mask2_len = len(mask2)
    score_list = []

    for f, frame1 in enumerate(mask1):
        frame2 = mask2[f % mask2_len]

        if frame1.shape != frame2.shape:
            frame2 = cv2.resize(frame2, frame1.shape[::-1])

        intersection = np.logical_and(frame1, frame2).sum()
        union = np.logical_or(frame1, frame2).sum()
        iou = 0.0 if union == 0 else intersection / union

        score_list.append(iou)

    return np.mean(score_list)


def pack_temporal(mask_array, length=5):
    T, H, W = mask_array.shape
    full_groups = T // length
    remainder = T % length

    if remainder != 0:
        padding = (length - remainder, H, W)
        padded = np.pad(
            mask_array,
            ((0, padding[0]), (0, 0), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        T = padded.shape[0]
    else:
        padded = mask_array

    return np.max(padded.reshape(T // length, length, H, W), axis=1)


ROOT = Path.cwd()
DATASET = conf.active.dataset
DATASET_DIR = ROOT / conf.datasets[DATASET].path
N_FILES = conf.datasets[DATASET].n_videos
DETECTOR = conf.active.detector
DET_CONFIDENCE = conf.detect[DETECTOR].confidence
METHOD = conf.mask_sim.method
MASK_DIR = ROOT / "data" / DATASET / DETECTOR / str(DET_CONFIDENCE) / "detect/mask"
OUT_PATH = MASK_DIR / f"{METHOD}.npz"
MAX_WORKERS = conf.active.max_workers
RESIZE_FACTOR = conf.mask_sim.resize[DATASET]
MIN_MEMORY = conf.mask_sim.min_memory
PACK_TEMPORAL = conf.mask_sim.pack_temporal
mask_bank = {}
file_list = []
stem2action = {}

assert_that(MASK_DIR).is_directory().is_readable()
assert_that(METHOD).is_in("iou", "fou")

if OUT_PATH.exists():
    data = np.load(OUT_PATH)["arr_0"]
    START_IDX = np.where(np.all(data == 0, axis=1))[0][0]
else:
    data = np.zeros((N_FILES, N_FILES), np.float16)
    START_IDX = 0

print("Input:", MASK_DIR.relative_to(ROOT))
print("n videos:", N_FILES)
print("Method:", METHOD)
print("Resize factor:", RESIZE_FACTOR)
print("Max workers:", MAX_WORKERS)
print("Starting row:", START_IDX)
print("Temporal pack:", PACK_TEMPORAL.enabled, PACK_TEMPORAL.length)

if not click.confirm("\nDo you want to continue?", show_default=True):
    exit("Aborted.")

print("Loading masks into memory...")

with open(DATASET_DIR / "list.txt") as f:
    lines = f.readlines()
    lines.reverse()

    for line in tqdm(lines, total=N_FILES, dynamic_ncols=True):
        action, filename = line.split()[0].split("/")
        stem = splitext(filename)[0]

        file_list.append(stem)
        stem2action[stem] = action

        mask_path = (MASK_DIR / action / filename).with_suffix(".npz")
        mask = np.load(mask_path)["arr_0"]

        if RESIZE_FACTOR < 1:
            height, width = mask.shape[1:]
            new_width = int(width * RESIZE_FACTOR)
            new_height = int(height * RESIZE_FACTOR)

            mask = np.stack(
                [cv2.resize(layer, (new_width, new_height)) for layer in mask]
            )

        if PACK_TEMPORAL.enabled:
            mask = pack_temporal(mask)

        free_memory = psutil.virtual_memory().available / (1024**3)

        if free_memory < MIN_MEMORY:
            print("Memory limit exceeded.")
            exit()

        mask_bank[stem] = mask

print(f"Working with {MAX_WORKERS} max workers...")

bar = tqdm(
    total=len(file_list),
    dynamic_ncols=True,
)

bar.update(START_IDX)

for file1_idx, file1 in enumerate(file_list):
    if file1_idx < START_IDX:
        continue

    file2_list = file_list[file1_idx + 1 :]
    jobs = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for file2 in file2_list:
            file2_idx = file_list.index(file2)
            jobs[(file1_idx, file2_idx)] = executor.submit(
                compute_sim, file1, file2, METHOD
            )

        for (i, j), job in jobs.items():
            data[i, j] = job.result()

    if (file1_idx + 1) % 10 == 0:
        bar.set_description("Saving matrix...")
        np.savez_compressed(OUT_PATH, data)

    bar.update(1)

np.savez_compressed(OUT_PATH, data)
