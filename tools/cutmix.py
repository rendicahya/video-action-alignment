import os
import sys

import cv2
import mmcv
import numpy as np

sys.path.append(".")
import json

import random
from collections import defaultdict
from pathlib import Path

import click
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
    root = Path.cwd()
    dataset = conf.active.dataset
    detector = conf.active.detector
    det_confidence = conf.detect[detector].confidence
    smooth_edge = conf.cutmix.smooth_edge
    scene_replace = conf.cutmix.scene.replace
    scene_transform = conf.cutmix.scene.transform
    scene_selection_method = conf.cutmix.scene.selection.method
    scene_selection_tolerance = conf.cutmix.scene.selection.tolerance
    multiplication = conf.cutmix.multiplication
    video_ext = conf.datasets[dataset].ext
    n_videos = conf.datasets[dataset].n_videos
    random_seed = conf.active.random_seed
    video_in_dir = root / "data" / dataset / "videos"
    video_writer = conf.cutmix.output.writer
    mask_dir = (
        root / "data" / dataset / detector / str(det_confidence) / "detect" / "mask"
    )
    video_out_dir = (
        root
        / "data"
        / dataset
        / detector
        / str(det_confidence)
        / "mix"
        / scene_selection_method
        / scene_transform
    )
    out_ext = conf.cutmix.output.ext
    scene_dict = defaultdict(list)

    print("Σ videos:", n_videos)
    print("Multiplication:", multiplication)
    print("Smooth edge:", smooth_edge)
    print("Input:", video_in_dir.relative_to(root))
    print("Mask:", mask_dir.relative_to(root))
    print("Output:", video_out_dir.relative_to(root))
    print("Writer:", video_writer)
    print("Scene selection:", scene_selection_method)
    print("Scene transform:", scene_transform)

    assert_that(video_in_dir).is_directory().is_readable()
    assert_that(mask_dir).is_directory().is_readable()
    assert_that(scene_selection_method).is_in("random", "area", "iou")
    assert_that(scene_replace).is_in("noop", "white", "black", "inpaint")
    assert_that(scene_transform).is_in("noop", "hflip")

    if not click.confirm("\nDo you want to continue?", show_default=True):
        exit("Aborted.")

    random.seed(random_seed)

    if scene_selection_method == "random":
        with open(video_in_dir / "list.txt") as f:
            for line in f:
                action, filename = line.split()[0].split("/")
                scene_dict[action].append(filename)
    elif scene_selection_method == "area":
        with open(mask_dir / "ratio.json") as f:
            ratio_json = json.load(f)

    bar = tqdm(total=n_videos * multiplication, dynamic_ncols=True)
    n_written = 0

    for file in video_in_dir.glob(f"**/*{video_ext}"):
        action = file.parent.name
        video_mask_path = mask_dir / action / file.with_suffix(".npz").name

        if not video_mask_path.is_file() or not video_mask_path.exists():
            continue

        action_mask = np.load(video_mask_path)["arr_0"]
        fps = mmcv.VideoReader(str(file)).fps
        i = 0

        if scene_selection_method == "random":
            scene_class_options = [s for s in scene_dict.keys() if s != action]
        if scene_selection_method == "area":
            action_mask_ratio = np.count_nonzero(action_mask) / action_mask.size

        while i < multiplication:
            if scene_selection_method == "random":
                scene_class = random.choice(scene_class_options)
                scene_options = scene_dict[scene_class]
                scene_pick = random.choice(scene_options)

                scene_class_options.remove(scene_class)
            elif scene_selection_method == "area":
                mask_ratio_lower = action_mask_ratio * (1 - scene_selection_tolerance)
                mask_ratio_upper = action_mask_ratio * (1 + scene_selection_tolerance)
                scene_options = [
                    filename
                    for filename, ratio in ratio_json.items()
                    if mask_ratio_lower < ratio < mask_ratio_upper
                    and filename.split("_")[1] != action
                ]

                if len(scene_options) <= 1:
                    continue

                scene_pick = random.choice(scene_options) + video_ext
                scene_class = scene_pick.split("_")[1]

            video_out_path = (
                video_out_dir / action / f"{file.stem}-{scene_class}"
            ).with_suffix(out_ext)

            if (
                video_out_path.exists()
                and mmcv.VideoReader(str(video_out_path)).frame_cnt > 0
            ):
                bar.update(1)
                continue

            if scene_replace == "noop":
                scene_mask = None
            else:
                scene_pick_base = os.path.splitext(scene_pick)[0]
                scene_mask_path = mask_dir / scene_class / f"{scene_pick_base}.npz"
                scene_mask = np.load(scene_mask_path)["arr_0"]

                if len(scene_mask) > 500:
                    continue

            scene_path = video_in_dir / scene_class / scene_pick
            out_frames = cutmix_fn(
                file,
                scene_path,
                action_mask,
                scene_replace,
                scene_transform,
                scene_mask,
            )

            if out_frames:
                video_out_path.parent.mkdir(parents=True, exist_ok=True)
                frames_to_video(
                    out_frames,
                    video_out_path,
                    writer=video_writer,
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
