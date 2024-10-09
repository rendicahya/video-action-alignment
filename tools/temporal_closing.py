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
    root = Path.cwd()
    dataset = conf.active.dataset
    detector = conf.active.detector
    det_confidence = conf.detect[detector].confidence
    smooth_edge = conf.cutmix.smooth_edge
    video_ext = conf.datasets[dataset].ext
    mask_dir = (
        root / "data" / dataset / detector / str(det_confidence) / "detect" / "mask"
    )
    out_dir = (
        root
        / "data"
        / dataset
        / detector
        / str(det_confidence)
        / "detect"
        / "mask-close"
    )

    if not click.confirm("\nDo you want to continue?", show_default=True):
        exit("Aborted.")

    structuring_element = np.ones((5, 1, 1))

    for file in mask_dir.glob(f"**/*{video_ext}"):
        if file.stem != "v_HorseRiding_g01_c01":
            continue

        action = file.parent.name
        out_path = out_dir / action / f"{file.stem}.npz"


if __name__ == "__main__":
    main()
