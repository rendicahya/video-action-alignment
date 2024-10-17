import os
import sys

import cv2
import mmcv
import numpy as np

sys.path.append(".")

import random
from collections import defaultdict
from pathlib import Path

import click
from scipy.ndimage import binary_closing
from tqdm import tqdm

from assertpy.assertpy import assert_that
from config import settings as conf
from python_video import frames_to_video


def main():
    ROOT = Path.cwd()
    DATASET = conf.active.DATASET
    N_VIDEOS = conf.datasets[DATASET].n_videos
    DETECTOR = conf.active.DETECTOR
    DET_CONFIDENCE = conf.detect[DETECTOR].confidence
    smooth_edge = conf.cutmix.smooth_edge
    MASK_DIR = ROOT / "data" / DATASET / DETECTOR / str(DET_CONFIDENCE) / "detect/mask"
    TEMPORAL_CLOSING_LENGTH = conf.temporal_closing.length
    OUT_DIR = MASK_DIR.parent / "mask-closing"

    print("Input:", MASK_DIR.relative_to(ROOT))
    print("Output:", OUT_DIR.relative_to(ROOT))

    if not click.confirm("\nDo you want to continue?", show_default=True):
        exit("Aborted.")

    kernel = np.ones((TEMPORAL_CLOSING_LENGTH, 1, 1))

    for file in tqdm(MASK_DIR.glob(f"**/*.npz"), total=N_VIDEOS, dynamic_ncols=True):
        if file.parent == MASK_DIR:
            continue

        mask = np.load(file)["arr_0"]
        closed = binary_closing(mask == 255, structure=kernel).astype(np.uint8) * 255
        action = file.parent.name
        out_path = OUT_DIR / action / f"{file.stem}.npz"

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, closed)


if __name__ == "__main__":
    main()
