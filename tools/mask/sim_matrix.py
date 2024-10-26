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


def compute_sim(fg_file, bg_file, method):
    fg_mask = mask_bank[fg_file]
    bg_mask = mask_bank[bg_file]

    bg_len = len(bg_mask)
    score_list = []

    for f, fg_frame in enumerate(fg_mask):
        bg_frame = bg_mask[f % bg_len]

        if fg_frame.shape != bg_frame.shape:
            bg_frame = cv2.resize(bg_frame, fg_frame.shape[::-1])

        intersection = np.logical_and(fg_frame, bg_frame).sum()

        if method == "iou":
            union = np.logical_or(fg_frame, bg_frame).sum()
            score = 0.0 if union == 0 else intersection / union
        elif method == "bao":
            bg_area = bg_frame.sum()
            score = 0.0 if bg_area == 0 else intersection / bg_area

        score_list.append(score)

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
assert_that(METHOD).is_in("iou", "bao")

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

for fg_idx, fg_file in enumerate(file_list):
    if fg_idx < START_IDX:
        continue

    bg_file_list = file_list[fg_idx + 1 :]
    jobs = {}

    bar.set_description(fg_file[:30])

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for bg_file in bg_file_list:
            bg_idx = file_list.index(bg_file)
            jobs[(fg_idx, bg_idx)] = executor.submit(
                compute_sim, fg_file, bg_file, METHOD
            )

        for (f, b), job in jobs.items():
            data[f, b] = job.result()

    if (fg_idx + 1) % 10 == 0:
        bar.set_description("Saving matrix...")
        np.savez_compressed(OUT_PATH, data)
        bar.set_description("Saved")

    bar.update(1)

np.savez_compressed(OUT_PATH, data)
