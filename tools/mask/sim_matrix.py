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
from lib_sim import *
from tqdm import tqdm

from assertpy.assertpy import assert_that
from config import settings as conf


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
MASK_DIR = ROOT / "data" / DATASET / DETECTOR / str(DET_CONFIDENCE) / "detect/mask"
IOU_OUT_PATH = MASK_DIR / f"iou-std.npz"
BAO_OUT_PATH = MASK_DIR / f"bao-std.npz"
RESIZE_FACTOR = conf.mask_sim.resize[DATASET]
MIN_MEMORY = conf.mask_sim.min_memory
MULTITHREADING = conf.mask_sim.multithreading
PACK_TEMPORAL = conf.mask_sim.pack_temporal
STANDARD_SIZE = (conf.datasets[DATASET].standard.h, conf.datasets[DATASET].standard.w)

mask_bank = {}
file_list = []
stem2action = {}

assert_that(MASK_DIR).is_directory().is_readable()

if BAO_OUT_PATH.exists():
    iou_data = np.load(IOU_OUT_PATH)["arr_0"]
    bao_data = np.load(BAO_OUT_PATH)["arr_0"]
    START_IDX = np.where(np.all(iou_data == 0, axis=1))[0][0]
else:
    iou_data = np.zeros((N_FILES, N_FILES), np.float16)
    bao_data = np.zeros((N_FILES, N_FILES), np.float16)
    START_IDX = 0

START_IDX = 0

print("Input:", MASK_DIR.relative_to(ROOT))
print("n videos:", N_FILES)
print("Standard size:", STANDARD_SIZE)
print("Starting row:", START_IDX)
print(
    "Output:",
    IOU_OUT_PATH.relative_to(ROOT),
    "(exists)" if IOU_OUT_PATH.exists() else "(not exists)",
)
print(
    "Output:",
    BAO_OUT_PATH.relative_to(ROOT),
    "(exists)" if BAO_OUT_PATH.exists() else "(not exists)",
)
print("Temporal pack:", PACK_TEMPORAL.enabled, PACK_TEMPORAL.length)

if not click.confirm("\nDo you want to continue?", show_default=True):
    exit("Aborted.")

print("Loading masks into memory...")

with open(DATASET_DIR / "list.txt") as f:
    lines = f.readlines()

bar = tqdm(total=N_FILES, dynamic_ncols=True)

for line in lines:
    action, filename = line.split()[0].split("/")
    stem = splitext(filename)[0]
    stem2action[stem] = action
    mask_path = (MASK_DIR / action / filename).with_suffix(".npz")
    mask = np.load(mask_path)["arr_0"]
    free_memory = psutil.virtual_memory().available / (1024**3)

    if mask.shape[1:] != STANDARD_SIZE:
        mask = np.stack([cv2.resize(layer, STANDARD_SIZE[::-1]) for layer in mask])

    if PACK_TEMPORAL.enabled:
        mask = pack_temporal(mask, PACK_TEMPORAL.length)

    if free_memory < MIN_MEMORY:
        print("Memory limit exceeded.")
        exit()

    file_list.append(stem)
    bar.set_description(f"{free_memory:.1f} GB free memory")
    bar.update(1)

    mask_bank[stem] = mask

bar.close()

if MULTITHREADING.enabled:
    print(f"Working with {MULTITHREADING.max_workers} max workers...")

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

    if MULTITHREADING.enabled:
        with ThreadPoolExecutor(max_workers=MULTITHREADING.max_workers) as executor:
            for bg_file in bg_file_list:
                bg_idx = file_list.index(bg_file)
                jobs[(fg_idx, bg_idx)] = executor.submit(
                    compute_sim_bool_std, mask_bank[fg_file], mask_bank[bg_file]
                )

            for (f, b), job in jobs.items():
                iou, iob, iof = job.result()
                iou_data[f, b] = iou
                bao_data[f, b] = iob
                bao_data[b, f] = iof
    else:
        for bg_file in bg_file_list:
            bg_idx = file_list.index(bg_file)
            iou, iob, iof = compute_sim_cupy_std(mask_bank[fg_file], mask_bank[bg_file])
            iou_data[fg_idx, bg_idx] = iou
            bao_data[fg_idx, bg_idx] = iob
            bao_data[bg_idx, fg_idx] = iof

    if (fg_idx + 1) % 100 == 0:
        bar.set_description("Saving matrix...")
        np.savez_compressed(BAO_OUT_PATH, bao_data)
        np.savez_compressed(IOU_OUT_PATH, iou_data)

    bar.update(1)

bar.close()
np.savez_compressed(BAO_OUT_PATH, bao_data)
np.savez_compressed(IOU_OUT_PATH, iou_data)
