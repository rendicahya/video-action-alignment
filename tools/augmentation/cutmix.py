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


def add_suffix(path: Path, suffix: str):
    return path.parent / (path.stem + suffix)


def cutmix_fn(
    actor_path,
    scene_path,
    action_mask,
    scene_replace,
    scene_mask,
    scene_transform,
    soft_edge,
    scene_transform_rand,
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

    if scene_transform:
        do_scene_transform = scene_transform_rand.random() <= scene_transform["prob"]

    if scene_mask.shape[:2] != (h, w) and scene_replace in ("white", "black"):
        scene_mask = np.moveaxis(scene_mask, 0, -1)
        scene_mask = cv2.resize(scene_mask, dsize=(w, h))
        scene_mask = np.moveaxis(scene_mask, -1, 0)

    if soft_edge:
        actor_mask_normal = action_mask / 255.0
        scene_mask_normal = scene_mask / 255.0

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

        if not soft_edge:
            if scene_replace == "white":
                scene_frame[is_foreground] = 255
            elif scene_replace == "black":
                scene_frame[is_foreground] = 0

        actor_mask = action_mask[f]

        if actor_mask is None:
            actor_mask = blank

        if scene_transform and do_scene_transform:
            scene_frame = scene_transform["fn"](scene_frame)

        if soft_edge:
            actor_mask_3 = np.repeat(
                np.expand_dims(actor_mask_normal[f], axis=2), 3, axis=2
            )
            scene_mask_3 = np.repeat(
                np.expand_dims(1 - scene_mask_normal[f % scene_n_frames], axis=2),
                3,
                axis=2,
            )
            actor = (actor_frame * actor_mask_3).astype(np.uint8)
            scene = (scene_frame * scene_mask_3 * (1 - actor_mask_3)).astype(np.uint8)
        else:
            actor = cv2.bitwise_and(actor_frame, actor_frame, mask=actor_mask)
            scene = cv2.bitwise_and(scene_frame, scene_frame, mask=255 - actor_mask)

        mix = cv2.add(actor, scene)

        scene_frame = scene_reader.read()

        yield cv2.cvtColor(mix, cv2.COLOR_BGR2RGB)


def main():
    ROOT = Path.cwd()
    DATASET = conf.active.dataset
    DETECTOR = conf.active.detector
    DET_CONF = conf.detect[DETECTOR].confidence
    SOFT_EDGE_ENABLED = conf.cutmix.soft_edge.enabled
    TEMPORAL_MORPHOLOGY_ENABLED = conf.cutmix.temporal_morphology.enabled
    SCENE_REPLACE = conf.cutmix.scene.replace
    SCENE_TRANSFORM_ENABLED = conf.cutmix.scene.transform.enabled
    SCENE_TRANSFORM_OP = conf.cutmix.scene.transform.op
    SCENE_SELECTION_METHOD = conf.cutmix.scene.selection.method
    MULTIPLICATION = conf.cutmix.multiplication
    EXT = conf.datasets[DATASET].ext
    N_VIDEOS = conf.datasets[DATASET].N_VIDEOS
    RANDOM_SEED = conf.active.RANDOM_SEED
    VIDEO_IN_DIR = ROOT / "data" / DATASET / "videos"
    MASK_DIR = ROOT / "data" / DATASET / DETECTOR / str(DET_CONF) / "detect/mask"
    VIDEO_OUT_DIR = (
        ROOT
        / "data"
        / DATASET
        / DETECTOR
        / str(DET_CONF)
        / "mix"
        / SCENE_SELECTION_METHOD
    )

    if TEMPORAL_MORPHOLOGY_ENABLED:
        TEMPORAL_MORPHOLOGY_OP = conf.cutmix.temporal_morphology.op
        MASK_DIR = add_suffix(MASK_DIR, "-" + TEMPORAL_MORPHOLOGY_OP)
        VIDEO_OUT_DIR = add_suffix(VIDEO_OUT_DIR, "-" + TEMPORAL_MORPHOLOGY_OP)

        assert_that(TEMPORAL_MORPHOLOGY_OP).is_in("dilation", "opening", "closing")

    if SOFT_EDGE_ENABLED:
        MASK_DIR = add_suffix(MASK_DIR, "-soft")
        VIDEO_OUT_DIR = add_suffix(VIDEO_OUT_DIR, "-soft")

    if SCENE_TRANSFORM_ENABLED:
        if SCENE_TRANSFORM_OP == "hflip":
            scene_transform = {"fn": lambda frame: cv2.flip(frame, 1), "prob": 0.5}
            VIDEO_OUT_DIR = add_suffix(VIDEO_OUT_DIR, "-hflip")
    else:
        scene_transform = None

    print("n videos:", N_VIDEOS)
    print("Multiplication:", MULTIPLICATION)
    print("Input:", VIDEO_IN_DIR.relative_to(ROOT))
    print("Mask:", MASK_DIR.relative_to(ROOT))
    print(
        "Output:",
        VIDEO_OUT_DIR.relative_to(ROOT),
        "(exists)" if VIDEO_OUT_DIR.exists() else "(not exists)",
    )

    if SCENE_SELECTION_METHOD in ("iou-v", "iou-m", "bao-v", "bao-m"):
        MATRIX_PATH = MASK_DIR.parent / f"mask/{SCENE_SELECTION_METHOD[:3]}.npz"

        assert_that(MATRIX_PATH).is_file().is_readable()

        MATRIX = np.load(MATRIX_PATH)["arr_0"]
        check_value = MATRIX[-2, -1]

        assert_that(check_value).is_not_equal_to(0.0)
        print("Scene selection:", SCENE_SELECTION_METHOD)
        print("Matrix cell check:", check_value)

    assert_that(VIDEO_IN_DIR).is_directory().is_readable()
    assert_that(MASK_DIR).is_directory().is_readable()
    assert_that(SCENE_SELECTION_METHOD).is_in(
        "random", "iou-v", "iou-m", "bao-v", "bao-m"
    )
    assert_that(SCENE_REPLACE).is_in("noop", "white", "black", "inpaint")
    assert_that(SCENE_TRANSFORM_OP).is_in("hflip")

    if not click.confirm("\nDo you want to continue?", show_default=True):
        exit("Aborted.")

    random.seed(RANDOM_SEED)

    action2scenes = defaultdict(list)
    scene2action = {}
    action_list = np.zeros(N_VIDEOS, np.uint8)
    scene_transform_rand = random.Random()
    idx2stem = []
    action_name2idx = {}

    with open(VIDEO_IN_DIR / "list.txt") as f:
        for i, line in enumerate(f):
            path, action_idx = line.split()
            action, filename = path.split("/")
            stem = os.path.splitext(filename)[0]

            if SCENE_SELECTION_METHOD == "random":
                action2scenes[action].append(stem)
            elif SCENE_SELECTION_METHOD in ("iou-v", "iou-m", "bao-v", "bao-m"):
                scene2action[stem] = action
                action_list[i] = int(action_idx)
                idx2stem.append(stem)

                if action not in action_name2idx:
                    action_name2idx[action] = int(action_idx)

    bar = tqdm(total=N_VIDEOS * MULTIPLICATION, dynamic_ncols=True)
    n_written = 0

    with open(VIDEO_IN_DIR / "list.txt") as f:
        file_list = f.readlines()

    for file_idx, line in enumerate(file_list):
        path, action_idx = line.split()
        file = Path(VIDEO_IN_DIR / path)
        action = file.parent.name
        video_mask_path = (MASK_DIR / action / file.name).with_suffix(".npz")
        used_actions = [int(action_idx)]
        used_videos = [int(file_idx)]

        if not video_mask_path.is_file() or not video_mask_path.exists():
            continue

        action_mask = np.load(video_mask_path)["arr_0"]

        if SCENE_SELECTION_METHOD == "random":
            scene_class_options = [s for s in action2scenes.keys() if s != action]
        elif SCENE_SELECTION_METHOD in ("iou-v", "iou-m", "bao-v", "bao-m"):
            iou_row = MATRIX[file_idx][file_idx:]
            iou_col = MATRIX[:, file_idx][:file_idx]
            iou_merge = np.concatenate((iou_col, iou_row))
            all_options = np.argsort(iou_merge)[::-1]

        i = 0

        while i < MULTIPLICATION:
            bar.set_description(f"{file.stem[:20]} ({i+1}/{MULTIPLICATION})")

            if SCENE_SELECTION_METHOD == "random":
                scene_class = random.choice(scene_class_options)
                scene_options = action2scenes[scene_class]
                scene = random.choice(scene_options)

                scene_class_options.remove(scene_class)

            elif SCENE_SELECTION_METHOD in ("iou-v", "iou-m", "bao-v", "bao-m"):
                # Videos having the same action
                same_action = np.where(np.isin(action_list, used_actions))

                # Exclude from sorting
                eligible = np.setdiff1d(all_options, same_action, assume_unique=True)

                if SCENE_SELECTION_METHOD in ("iou-m", "bao-m"):
                    eligible = np.setdiff1d(eligible, used_videos, assume_unique=True)

                # Get scene with max score
                scene_id = eligible[i]
                scene = idx2stem[scene_id]
                scene_class = scene2action[scene]

                # V-variants: varying action class
                if SCENE_SELECTION_METHOD in ("iou-v", "bao-v"):
                    scene_class_idx = action_name2idx[scene_class]

                    # Remember the action of the selected video
                    used_actions.append(scene_class_idx)

                # M-variants: max score without varying action class
                elif SCENE_SELECTION_METHOD in ("iou-m", "bao-m"):
                    # Remember the selected video
                    used_videos.append(scene_id)

            # Obsolete
            elif SCENE_SELECTION_METHOD == "iou-10":
                used_videos = np.where(np.isin(action_list, used_actions))
                eligible = np.setdiff1d(all_options, used_videos, assume_unique=True)
                # Get a random scene from top-10 IoU
                scene_id = random.choice(eligible[:10])
                scene = idx2stem[scene_id]
                scene_class = scene2action[scene]
                scene_class_idx = action_name2idx[scene_class]

                used_actions.append(scene_class_idx)

            video_out_path = (
                VIDEO_OUT_DIR / action / f"{file.stem}-{i}-{scene_class}"
            ).with_suffix(".mp4")

            if video_out_path.exists():
                bar.update(1)
                continue

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
                scene_mask,
                scene_transform,
                SOFT_EDGE_ENABLED,
                scene_transform_rand,
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
