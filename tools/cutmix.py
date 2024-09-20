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
from tqdm import tqdm

from assertpy.assertpy import assert_that
from config import settings as conf
from python_video import frames_to_video


def cutmix_fn(actor_path, scene_path, mask_bundle, scene_replace, scene_mask):
    if not actor_path.is_file() or not actor_path.exists():
        print("Not a file or not exists:", actor_path)
        return None

    if not scene_path.is_file() or not scene_path.exists():
        print("Not a file or not exists:", scene_path)
        return None

    actor_reader = mmcv.VideoReader(str(actor_path))
    w, h = actor_reader.resolution
    scene_frame = None
    blank = np.zeros((h, w), np.uint8)

    for f, actor_frame in enumerate(actor_reader):
        if f == len(mask_bundle) - 1:
            return

        if scene_frame is None:
            scene_reader = mmcv.VideoReader(str(scene_path))
            scene_n_frames = scene_reader.frame_cnt
            scene_frame = scene_reader.read()

        if scene_frame.shape[:2] != (h, w):
            scene_frame = cv2.resize(scene_frame, (w, h))

        if scene_replace == "black":
            is_foreground = scene_mask[f % scene_n_frames] == 255
            scene_frame[is_foreground] = 0

        actor_mask = mask_bundle[f]

        if actor_mask is None:
            actor_mask = blank

        actor = cv2.bitwise_and(actor_frame, actor_frame, mask=actor_mask)
        scene = cv2.bitwise_and(scene_frame, scene_frame, mask=255 - actor_mask)

        mix = cv2.add(actor, scene)
        scene_frame = scene_reader.read()

        yield cv2.cvtColor(mix, cv2.COLOR_BGR2RGB)


def main():
    root = Path.cwd()
    dataset = conf.active.dataset
    detector = conf.active.detector
    det_confidence = conf.detect[detector].confidence
    smooth_edge = conf.cutmix.smooth_edge
    scene_replace = conf.cutmix.scene_replace
    use_REPP = conf.active.use_REPP
    multiplication = conf.cutmix.multiplication
    video_ext = conf.datasets[dataset].ext
    n_videos = conf.datasets[dataset].n_videos
    random_seed = conf.active.random_seed
    video_in_dir = root / "data" / dataset / "videos"
    mask_dir = (
        root / "data" / dataset / detector / str(det_confidence) / "detect" / "mask"
    )
    video_out_dir = (
        root / "data" / dataset / detector / str(det_confidence) / "mix" / scene_replace
    )
    out_ext = conf.cutmix.output.ext
    scene_dict = defaultdict(list)

    print("Σ videos:", n_videos)
    print("Multiplication:", multiplication)
    print("Smooth edge:", smooth_edge)
    print("Input:", video_in_dir.relative_to(root))
    print("Mask:", mask_dir.relative_to(root))
    print("Output:", video_out_dir.relative_to(root))

    assert_that(video_in_dir).is_directory().is_readable()
    assert_that(mask_dir).is_directory().is_readable()

    if not click.confirm("\nDo you want to continue?", show_default=True):
        exit("Aborted.")

    random.seed(random_seed)

    with open(video_in_dir / "list.txt") as f:
        for line in f:
            action, filename = line.split()[0].split("/")
            scene_dict[action].append(filename)

    bar = tqdm(total=n_videos * multiplication, dynamic_ncols=True)
    n_written = 0

    for file in video_in_dir.glob(f"**/*{video_ext}"):
        action = file.parent.name
        video_mask_path = mask_dir / action / file.with_suffix(".npz").name

        if not video_mask_path.is_file() or not video_mask_path.exists():
            continue

        mask_bundle = np.load(video_mask_path)["arr_0"]
        fps = mmcv.VideoReader(str(file)).fps
        scene_action_options = [s for s in scene_dict.keys() if s != action]

        for i in range(multiplication):
            scene_action_pick = random.choice(scene_action_options)
            video_out_path = (
                video_out_dir / action / f"{file.stem}-{scene_action_pick}"
            ).with_suffix(out_ext)

            scene_action_options.remove(scene_action_pick)

            if (
                video_out_path.exists()
                and mmcv.VideoReader(str(video_out_path)).frame_cnt > 0
            ):
                bar.update(1)
                continue

            scene_options = scene_dict[scene_action_pick]
            scene_pick = random.choice(scene_options)

            if scene_replace == "noop":
                scene_mask = None
            else:
                scene_pick_base = os.path.splitext(scene_pick)[0]
                scene_mask_path = (
                    mask_dir / scene_action_pick / f"{scene_pick_base}.npz"
                )
                scene_mask = np.load(scene_mask_path)["arr_0"]

            scene_path = video_in_dir / scene_action_pick / scene_pick
            out_frames = cutmix_fn(
                file, scene_path, mask_bundle, scene_replace, scene_mask
            )

            if out_frames:
                video_out_path.parent.mkdir(parents=True, exist_ok=True)
                frames_to_video(
                    out_frames,
                    video_out_path,
                    writer=conf.cutmix.output.writer,
                    fps=fps,
                )

                n_written += 1
            else:
                print("out_frames None: ", file.name)

            bar.update(1)

    bar.close()
    print("Written videos:", n_written)


if __name__ == "__main__":
    main()
