import sys

sys.path.append(".")

import json
import random
from collections import defaultdict
from math import sqrt
from os.path import splitext
from pathlib import Path

import click
import cv2
import mmcv
import numpy as np
from library import videomix_fn
from tqdm import tqdm

from assertpy.assertpy import assert_that
from config import settings as conf


def rand_bbox(W, H, lambda_):
    cut_ratio = np.sqrt(1.0 - lambda_)
    cut_w = int(W * cut_ratio)
    cut_h = int(H * cut_ratio)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    return x1, y1, x2, y2


def main():
    ROOT = Path.cwd()
    DATASET = conf.active.dataset
    EXT = conf.datasets[DATASET].ext
    N_VIDEOS = conf.datasets[DATASET].N_VIDEOS
    RANDOM_SEED = conf.active.RANDOM_SEED
    VIDEO_DIR = ROOT / "data" / DATASET / "videos"
    OUT_DIR = ROOT / "data" / DATASET / "videomix/mask"
    ALPHA = 1

    random.seed(RANDOM_SEED)

    print("n videos:", N_VIDEOS)
    print("Input:", VIDEO_DIR.relative_to(ROOT))
    print(
        "Output:",
        OUT_DIR.relative_to(ROOT),
        "(exists)" if OUT_DIR.exists() else "(not exists)",
    )

    assert_that(VIDEO_DIR).is_directory().is_readable()

    if not click.confirm("\nDo you want to continue?", show_default=True):
        exit("Aborted.")

    for file in tqdm(VIDEO_DIR.glob(f"**/*{EXT}"), total=N_VIDEOS, dynamic_ncols=True):
        video_reader = mmcv.VideoReader(str(file))
        w, h = video_reader.resolution

        lambda_ = np.random.beta(ALPHA, ALPHA)
        x1, y1, x2, y2 = rand_bbox(w, h, lambda_)

        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        label = file.parent.name
        mask_out_path = OUT_DIR / label / file.stem

        mask_out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(mask_out_path, mask)


if __name__ == "__main__":
    main()
