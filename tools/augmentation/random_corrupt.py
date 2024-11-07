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


@click.command()
@click.argument("ratio", nargs=1, required=True, type=float)
def main(ratio):
    ROOT = Path.cwd()
    DATASET = conf.active.dataset
    EXT = conf.datasets[DATASET].ext
    IN_DIR = ROOT / "data" / DATASET / "videos"
    OUT_DIR = ROOT / "data" / DATASET / f"random-corrupt-{ratio}"
    N_VIDEOS = conf.datasets[DATASET].N_VIDEOS

    print("Input:", IN_DIR.relative_to(ROOT))
    print("Output:", OUT_DIR.relative_to(ROOT))

    assert_that(IN_DIR).is_directory().is_readable()

    if not click.confirm("\nDo you want to continue?", show_default=True):
        exit("Aborted.")

    for file in tqdm(IN_DIR.glob(f"**/*{EXT}"), total=N_VIDEOS, dynamic_ncols=True):
        video_reader = mmcv.VideoReader(str(file))
        action = file.parent.name
        out_path = OUT_DIR / action / file.with_suffix(".mp4").name

        H, W = video_reader.resolution
        fps = video_reader.fps

        total_area = H * W
        target_area = int(ratio * total_area)
        side = int(np.sqrt(target_area))
        side = min(side, H, W)

        top_left_x = np.random.randint(0, W - side + 1)
        top_left_y = np.random.randint(0, H - side + 1)
        out_frames = []

        for frame in video_reader:
            frame[
                top_left_y : top_left_y + side,
                top_left_x : top_left_x + side,
            ] = 0
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            out_frames.append(frame)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        frames_to_video(out_frames, out_path, writer="moviepy", fps=fps)


if __name__ == "__main__":
    main()
