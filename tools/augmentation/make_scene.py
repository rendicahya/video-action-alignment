import sys

sys.path.append(".")

import json
import os
import random
from pathlib import Path

import click
import cv2
import mmcv
import numpy as np
from tqdm import tqdm

from assertpy.assertpy import assert_that
from config import settings as conf
from python_video import frames_to_video


def main():
    ROOT = Path.cwd()
    DATASET = conf.active.dataset
    N_VIDEOS = conf.datasets[DATASET].N_VIDEOS
    EXT = conf.datasets[DATASET].ext
    DETECTOR = conf.active.detector
    DET_CONF = str(conf.detect[DETECTOR].confidence)
    VIDEO_IN_DIR = ROOT / "data" / DATASET / "videos"
    MASK_DIR = ROOT / "data" / DATASET / DETECTOR / DET_CONF / "detect/mask-dilation"
    VIDEO_OUT_DIR = VIDEO_IN_DIR.parent / "scene"

    print("Input:", VIDEO_IN_DIR.relative_to(ROOT))
    print("n videos:", N_VIDEOS)
    print(
        "Output:",
        VIDEO_OUT_DIR.relative_to(ROOT),
        "(exists)" if VIDEO_OUT_DIR.exists() else "(not exists)",
    )

    assert_that(VIDEO_IN_DIR).is_directory().is_readable()
    assert_that(MASK_DIR).is_directory().is_readable()

    if not click.confirm("\nDo you want to continue?", show_default=True):
        exit("Aborted.")

    for file in tqdm(
        VIDEO_IN_DIR.glob(f"**/*{EXT}"), total=N_VIDEOS, dynamic_ncols=True
    ):
        action = file.parent.name
        mask_path = MASK_DIR / action / file.with_suffix(".npz").name
        mask_cube = np.load(mask_path)["arr_0"]
        video_reader = mmcv.VideoReader(str(file))
        out_frames = []

        for mask, frame in zip(mask_cube, video_reader):
            frame[mask == 255] = 0
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            out_frames.append(frame)

        fps = video_reader.fps
        video_out_path = VIDEO_OUT_DIR / action / file.with_suffix(".mp4").name

        video_out_path.parent.mkdir(parents=True, exist_ok=True)
        frames_to_video(
            out_frames,
            video_out_path,
            writer="moviepy",
            fps=fps,
        )


if __name__ == "__main__":
    main()
