import sys

sys.path.append(".")

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

ROOT = Path.cwd()
DATASET = conf.active.dataset
EXT = conf.datasets[DATASET].ext
VIDEO_IN_DIR = ROOT / "data" / DATASET / "videos"
VIDEO_OUT_DIR = ROOT / "data" / DATASET / "random-corrupt"
N_VIDEOS = conf.datasets[DATASET].N_VIDEOS
min_area_ratio = 0.1
max_area_ratio = 0.3

print("Input:", VIDEO_IN_DIR.relative_to(ROOT))
print("Output:", VIDEO_OUT_DIR.relative_to(ROOT))

assert_that(VIDEO_IN_DIR).is_directory().is_readable()

if not click.confirm("\nDo you want to continue?", show_default=True):
    exit("Aborted.")

for file in tqdm(VIDEO_IN_DIR.glob(f"**/*{EXT}"), total=N_VIDEOS, dynamic_ncols=True):
    video_reader = mmcv.VideoReader(str(file))
    action = file.parent.name
    out_path = VIDEO_OUT_DIR / action / file.with_suffix(".mp4").name

    H, W = video_reader.resolution
    fps = video_reader.fps
    total_area = H * W

    target_area = np.random.uniform(min_area_ratio, max_area_ratio) * total_area

    rect_w = np.random.randint(1, W)
    rect_h = int(target_area / rect_w)

    if rect_h > H:
        rect_h = H

    top_left_x = np.random.randint(0, W - rect_w + 1)
    top_left_y = np.random.randint(0, H - rect_h + 1)
    out_frames = []

    for frame in video_reader:
        frame[top_left_y : top_left_y + rect_h, top_left_x : top_left_x + rect_w] = 0
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        out_frames.append(frame)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames_to_video(out_frames, out_path, writer="moviepy", fps=fps)
