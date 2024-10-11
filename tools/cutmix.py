import sys

sys.path.append(".")
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import click
import cv2
import mmcv
import numpy as np
from tqdm import tqdm

from assertpy.assertpy import assert_that
from config import settings as conf
from python_video import frames_to_video


def cutmix_fn(
    actor_path, scene_path, action_mask, scene_replace, scene_transform, scene_mask
):
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

    if scene_mask.shape[:2] != (h, w) and scene_replace in ("white", "black"):
        scene_mask = np.moveaxis(scene_mask, 0, -1)
        scene_mask = cv2.resize(scene_mask, dsize=(w, h))
        scene_mask = np.moveaxis(scene_mask, -1, 0)

    for f, actor_frame in enumerate(actor_reader):
        if f == len(action_mask) - 1:
            return

        if scene_frame is None:
            scene_reader = mmcv.VideoReader(str(scene_path))
            scene_n_frames = scene_reader.frame_cnt
            scene_frame = scene_reader.read()

        if scene_frame.shape[:2] != (h, w):
            scene_frame = cv2.resize(scene_frame, (w, h))

        if scene_replace in ("white", "black"):
            is_foreground = scene_mask[f % scene_n_frames] == 255

        if scene_replace == "white":
            scene_frame[is_foreground] = 255
        elif scene_replace == "black":
            scene_frame[is_foreground] = 0

        actor_mask = action_mask[f]

        if actor_mask is None:
            actor_mask = blank

        if scene_transform == "hflip":
            scene_frame = cv2.flip(scene_frame, 1)

        actor = cv2.bitwise_and(actor_frame, actor_frame, mask=actor_mask)
        scene = cv2.bitwise_and(scene_frame, scene_frame, mask=255 - actor_mask)

        mix = cv2.add(actor, scene)
        scene_frame = scene_reader.read()

        yield cv2.cvtColor(mix, cv2.COLOR_BGR2RGB)


def main():
    ROOT = Path.cwd()
    DATASET = conf.active.dataset
    DETECTOR = conf.active.detector
    DET_CONFIDENCE = conf.detect[DETECTOR].confidence
    SMOOTH_EDGE = conf.cutmix.smooth_edge
    SCENE_REPLACE = conf.cutmix.scene.replace
    scene_transform = conf.cutmix.scene.transform
    scene_selection_method = conf.cutmix.scene.selection.method
    scene_selection_tolerance = conf.cutmix.scene.selection.tolerance
    multiplication = conf.cutmix.multiplication
    EXT = conf.datasets[DATASET].ext
    N_VIDEOS = conf.datasets[DATASET].N_VIDEOS
    RANDOM_SEED = conf.active.RANDOM_SEED
    VIDEO_IN_DIR = ROOT / "data" / DATASET / "videos"
    VIDEO_WRITER = conf.cutmix.output.writer
    MASK_DIR = (
        ROOT / "data" / DATASET / DETECTOR / str(DET_CONFIDENCE) / "detect" / "mask"
    )
    VIDEO_OUT__DIR = (
        ROOT
        / "data"
        / DATASET
        / DETECTOR
        / str(DET_CONFIDENCE)
        / "mix"
        / scene_selection_method
        / scene_transform
    )
    action2scenes_dict = defaultdict(list)
    scene2action_dict = {}

    print("Σ videos:", N_VIDEOS)
    print("Multiplication:", multiplication)
    print("Smooth edge:", SMOOTH_EDGE)
    print("Input:", VIDEO_IN_DIR.relative_to(ROOT))
    print("Mask:", MASK_DIR.relative_to(ROOT))
    print("Output:", VIDEO_OUT__DIR.relative_to(ROOT))
    print("Scene selection:", scene_selection_method)
    print("Scene transform:", scene_transform)

    assert_that(VIDEO_IN_DIR).is_directory().is_readable()
    assert_that(MASK_DIR).is_directory().is_readable()
    assert_that(scene_selection_method).is_in("random", "area", "iou")
    assert_that(SCENE_REPLACE).is_in("noop", "white", "black", "inpaint")
    assert_that(scene_transform).is_in("notransform", "hflip")

    if not click.confirm("\nDo you want to continue?", show_default=True):
        exit("Aborted.")

    random.seed(RANDOM_SEED)

    with open(VIDEO_IN_DIR / "list.txt") as f:
        for line in f:
            action, filename = line.split()[0].split("/")
            stem = os.path.splitext(filename)[0]

            if scene_selection_method == "random":
                action2scenes_dict[action].append(stem)
            elif scene_selection_method == "area":
                scene2action_dict[stem] = action

    if scene_selection_method == "area":
        with open(MASK_DIR / "ratio.json") as f:
            ratio_json = json.load(f)

    bar = tqdm(total=N_VIDEOS * multiplication, dynamic_ncols=True)
    n_written = 0

    for file in VIDEO_IN_DIR.glob(f"**/*{EXT}"):
        action = file.parent.name
        video_mask_path = MASK_DIR / action / file.with_suffix(".npz").name

        if not video_mask_path.is_file() or not video_mask_path.exists():
            continue

        if scene_selection_method == "random":
            scene_class_options = [s for s in action2scenes_dict.keys() if s != action]
        if scene_selection_method == "area":
            action_mask = np.load(video_mask_path)["arr_0"]
            action_mask_ratio = np.count_nonzero(action_mask) / action_mask.size

        i = 0

        while i < multiplication:
            bar.set_description(f"{file.stem} ({i+1}/{multiplication})")

            if scene_selection_method == "random":
                scene_class = random.choice(scene_class_options)
                scene_options = action2scenes_dict[scene_class]
                scene = random.choice(scene_options)

                scene_class_options.remove(scene_class)
            elif scene_selection_method == "area":
                mask_ratio_lower = action_mask_ratio * (1 - scene_selection_tolerance)
                mask_ratio_upper = action_mask_ratio * (1 + scene_selection_tolerance)
                scene_options = [
                    filename
                    for filename, ratio in ratio_json.items()
                    if mask_ratio_lower < ratio < mask_ratio_upper
                    and scene2action_dict[filename] != action
                ]

                if len(scene_options) <= 1:
                    break

                scene = random.choice(scene_options)
                scene_class = scene2action_dict[scene]

            video_out_path = (
                VIDEO_OUT__DIR / action / f"{file.stem}-{i}-{scene_class}"
            ).with_suffix(".mp4")

            if (
                video_out_path.exists()
                and mmcv.VideoReader(str(video_out_path)).frame_cnt > 0
            ):
                bar.update(1)
                continue

            if SCENE_REPLACE == "noop":
                scene_mask = None
            else:
                scene_mask_path = (MASK_DIR / scene_class / scene).with_suffix(".npz")
                scene_mask = np.load(scene_mask_path)["arr_0"]

                if len(scene_mask) > 500:
                    continue

            scene_path = (VIDEO_IN_DIR / scene_class / scene).with_suffix(EXT)
            out_frames = cutmix_fn(
                file,
                scene_path,
                action_mask,
                SCENE_REPLACE,
                scene_transform,
                scene_mask,
            )

            if out_frames:
                fps = mmcv.VideoReader(str(file)).fps

                video_out_path.parent.mkdir(parents=True, exist_ok=True)
                frames_to_video(
                    out_frames,
                    video_out_path,
                    writer="moviepy",
                    fps=fps,
                )

                n_written += 1
                i += 1
            else:
                print("out_frames None: ", file.name)

            bar.update(1)

    bar.close()
    print("Written videos:", n_written)


if __name__ == "__main__":
    main()
