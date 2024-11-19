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
from python_video import frames_to_video


def main():
    ROOT = Path.cwd()
    DATASET = conf.active.dataset
    EXT = conf.datasets[DATASET].ext
    N_VIDEOS = conf.datasets[DATASET].N_VIDEOS
    MULTIPLICATION = conf.cutmix.multiplication
    RANDOM_SEED = conf.active.RANDOM_SEED
    VIDEO_DIR = ROOT / "data" / DATASET / "videos"
    OUT_DIR = ROOT / "data" / DATASET / "videomix"

    random.seed(RANDOM_SEED)

    print("n videos:", N_VIDEOS)
    print("Multiplication:", MULTIPLICATION)
    print("Input:", VIDEO_DIR.relative_to(ROOT))
    print(
        "Output:",
        OUT_DIR.relative_to(ROOT),
        "(exists)" if OUT_DIR.exists() else "(not exists)",
    )

    assert_that(VIDEO_DIR).is_directory().is_readable()

    if not click.confirm("\nDo you want to continue?", show_default=True):
        exit("Aborted.")

    with open(VIDEO_DIR / "list.txt") as f:
        file_list = f.readlines()

    label2scenes = defaultdict(list)
    bar = tqdm(total=N_VIDEOS * MULTIPLICATION, dynamic_ncols=True)
    n_written = 0

    for i, line in enumerate(file_list):
        path, label_idx = line.split()
        label, filename = path.split("/")
        stem = splitext(filename)[0]

        label2scenes[label].append(stem)

    for action_line in file_list:
        action_file, action_label_idx = action_line.split()
        action_path = Path(VIDEO_DIR / action_file)
        action_label = action_path.parent.name
        i = 0
        scene_label_options = [s for s in label2scenes.keys() if s != action_label]
        fps = mmcv.VideoReader(str(action_path)).fps

        while i < MULTIPLICATION:
            scene_label = random.choice(scene_label_options)
            scene_options = label2scenes[scene_label]
            scene_stem = random.choice(scene_options)
            scene_path = (VIDEO_DIR / scene_label / scene_stem).with_suffix(EXT)
            out_frames = videomix_fn(action_path, scene_path)
            video_out_path = (
                OUT_DIR / action_label / f"{action_path.stem}-{i}-{scene_label}"
            ).with_suffix(".mp4")

            if not out_frames:
                continue

            scene_label_options.remove(scene_label)
            video_out_path.parent.mkdir(parents=True, exist_ok=True)

            frames_to_video(
                out_frames,
                video_out_path,
                writer="moviepy",
                fps=fps,
            )

            n_written += 1
            i += 1

            bar.update(1)

    bar.close()
    print("Written videos:", n_written)


if __name__ == "__main__":
    main()
