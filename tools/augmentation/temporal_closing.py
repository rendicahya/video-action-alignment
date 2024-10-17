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
    OUT_DIR = MASK_DIR.parent / "mask-closing"

    print("Input:", MASK_DIR.relative_to(ROOT))
    print("Output:", OUT_DIR.relative_to(ROOT))

    if not click.confirm("\nDo you want to continue?", show_default=True):
        exit("Aborted.")

    structuring_element = np.ones((5, 1, 1))

    for file in tqdm(MASK_DIR.glob(f"**/*.npz"), total=N_VIDEOS, dynamic_ncols=True):
        if file.stem != "v_HorseRiding_g01_c01":
            continue

        mask = np.load(file)["arr_0"]
        closed = binary_closing(mask, structure=structuring_element)
        action = file.parent.name
        out_path = OUT_DIR / action / f"{file.stem}.npz"

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, closed)
        break


if __name__ == "__main__":
    main()
