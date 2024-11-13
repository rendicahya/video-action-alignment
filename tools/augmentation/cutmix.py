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
from library import compute_artifact, cutmix_fn
from tqdm import tqdm

from assertpy.assertpy import assert_that
from config import settings as conf
from python_video import frames_to_video


def append_name(path: Path, name: str):
    return path.parent / (path.stem + name + path.suffix)


def main():
    ROOT = Path.cwd()
    DATASET = conf.active.dataset
    EXT = conf.datasets[DATASET].ext
    N_VIDEOS = conf.datasets[DATASET].N_VIDEOS
    DETECTOR = conf.active.detector
    DET_CONF = str(conf.detect[DETECTOR].confidence)
    SOFT_EDGE = conf.cutmix.soft_edge
    TEMPORAL_MORPHOLOGY = conf.cutmix.morphology.temporal
    SPATIAL_MORPHOLOGY = conf.cutmix.morphology.spatial
    SCENE_TRANSFORM = conf.cutmix.scene.transform
    SCENE_SELECTION = conf.cutmix.scene.selection.method
    MULTIPLICATION = conf.cutmix.multiplication
    RANDOM_SEED = conf.active.RANDOM_SEED
    COMPUTE_ARTIFACT = conf.cutmix.compute_artifact
    WRITE_VIDEOS = conf.cutmix.write_videos
    VIDEO_DIR = ROOT / "data" / DATASET / "videos"
    MASK_DIR = ROOT / "data" / DATASET / DETECTOR / DET_CONF / "detect/mask"
    OUT_DIR = ROOT / "data" / DATASET / DETECTOR / DET_CONF / "mix" / SCENE_SELECTION

    if TEMPORAL_MORPHOLOGY.enabled:
        MASK_DIR = append_name(MASK_DIR, "-" + TEMPORAL_MORPHOLOGY.op)
        OUT_DIR = append_name(OUT_DIR, "-" + TEMPORAL_MORPHOLOGY.op)

        assert_that(TEMPORAL_MORPHOLOGY.op).is_in("dilation", "opening", "closing")

    if SPATIAL_MORPHOLOGY.enabled:
        OUT_DIR = append_name(OUT_DIR, "-enlarged")

    if SOFT_EDGE.enabled:
        MASK_DIR = append_name(MASK_DIR, "-soft")
        OUT_DIR = append_name(OUT_DIR, "-soft")

    if SCENE_TRANSFORM.enabled:
        if SCENE_TRANSFORM.op == "hflip":
            scene_transform = {"fn": lambda frame: cv2.flip(frame, 1), "prob": 0.5}
            OUT_DIR = append_name(OUT_DIR, "-hflip")
    else:
        scene_transform = None

    if SCENE_SELECTION != "random":
        OUT_DIR = append_name(OUT_DIR, "-std")

    print("n videos:", N_VIDEOS)
    print("Multiplication:", MULTIPLICATION)
    print("Input:", VIDEO_DIR.relative_to(ROOT))
    print("Mask:", MASK_DIR.relative_to(ROOT))
    print("Scene selection:", SCENE_SELECTION)

    if SCENE_SELECTION in ("iou-v", "iou-m", "bao-v", "bao-m"):
        MATRIX_PATH = MASK_DIR.parent / f"mask/{SCENE_SELECTION[:3]}.npz"
        MATRIX_PATH = append_name(MATRIX_PATH, "-std")

        assert_that(MATRIX_PATH).is_file().is_readable()

        MATRIX = np.load(MATRIX_PATH)["arr_0"]
        check_value = MATRIX[-2, -1]

        assert_that(check_value).is_not_equal_to(0.0)
        print("Matrix:", MATRIX_PATH.relative_to(ROOT))
        print("Matrix cell check:", check_value)

    print(
        "Output:",
        OUT_DIR.relative_to(ROOT),
        "(exists)" if OUT_DIR.exists() else "(not exists)",
    )
    print("Write videos:", WRITE_VIDEOS)
    print("Compute artifact:", COMPUTE_ARTIFACT)

    assert_that(VIDEO_DIR).is_directory().is_readable()
    assert_that(MASK_DIR).is_directory().is_readable()
    assert_that(SCENE_SELECTION).is_in("random", "iou-v", "iou-m", "bao-v", "bao-m")
    assert_that(SCENE_TRANSFORM.op).is_in("hflip")

    if not click.confirm("\nDo you want to continue?", show_default=True):
        exit("Aborted.")

    random.seed(RANDOM_SEED)

    action2scenes = defaultdict(list)
    scene2action = {}
    action_list = np.zeros(N_VIDEOS, np.uint8)
    scene_transform_rand = random.Random()
    idx2stem = []
    action_name2idx = {}

    with open(VIDEO_DIR / "list.txt") as f:
        file_list = f.readlines()

    for i, line in enumerate(file_list):
        path, action_idx = line.split()
        action, filename = path.split("/")
        stem = splitext(filename)[0]

        if SCENE_SELECTION == "random":
            action2scenes[action].append(stem)
        else:
            scene2action[stem] = action
            action_list[i] = int(action_idx)
            idx2stem.append(stem)

            if action not in action_name2idx:
                action_name2idx[action] = int(action_idx)

    bar = tqdm(total=N_VIDEOS * MULTIPLICATION, dynamic_ncols=True)
    n_written = 0
    artifact_list = np.zeros(N_VIDEOS * MULTIPLICATION, dtype=np.float32)

    for file_idx, line in enumerate(file_list):
        path, action_idx = line.split()
        file = Path(VIDEO_DIR / path)
        action = file.parent.name
        video_mask_path = (MASK_DIR / action / file.name).with_suffix(".npz")
        fps = mmcv.VideoReader(str(file)).fps

        if not video_mask_path.is_file() or not video_mask_path.exists():
            continue

        action_mask = np.load(video_mask_path)["arr_0"]

        if SPATIAL_MORPHOLOGY.enabled and SPATIAL_MORPHOLOGY.op == "dilation":
            T, h, w = action_mask.shape
            hypotenuse = sqrt(h**2 + w**2)
            kernel_size = int(hypotenuse * SPATIAL_MORPHOLOGY.ratio)
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (kernel_size, kernel_size)
            )
            action_mask = np.array(
                [cv2.dilate(action_mask[t], kernel) for t in range(T)]
            )

        if SCENE_SELECTION == "random":
            scene_class_options = [s for s in action2scenes.keys() if s != action]
        else:
            if SCENE_SELECTION.startswith("iou"):
                row = MATRIX[file_idx][file_idx:]
                col = MATRIX[:, file_idx][:file_idx]
                sim_data = np.concatenate((col, row))
            else:
                sim_data = MATRIX[file_idx]

            all_videos = np.argsort(sim_data)

            # Videos from the current action
            videos_of_same_action = np.where(np.isin(action_list, [int(action_idx)]))

            # Exclude them from sorting
            eligible = np.setdiff1d(
                all_videos, videos_of_same_action, assume_unique=True
            )

        i = 0

        while i < MULTIPLICATION:
            bar.set_description(f"{file.stem[:20]} ({i+1}/{MULTIPLICATION})")

            if SCENE_SELECTION == "random":
                scene_class = random.choice(scene_class_options)
                scene_options = action2scenes[scene_class]
                scene_stem = random.choice(scene_options)

                scene_class_options.remove(scene_class)

            else:
                scene_id = eligible[-1]
                scene_stem = idx2stem[scene_id]
                scene_class = scene2action[scene_stem]

                if SCENE_SELECTION in ("iou-v", "bao-v"):
                    scene_class_idx = action_name2idx[scene_class]
                    videos_of_selected_action = np.where(
                        np.isin(action_list, [scene_class_idx])
                    )

                    # Remove the videos of the selected action from the eligible list
                    eligible = np.setdiff1d(
                        eligible, videos_of_selected_action, assume_unique=True
                    )

                elif SCENE_SELECTION in ("iou-m", "bao-m"):
                    # Remove the selected from the eligible list
                    eligible = np.setdiff1d(eligible, [scene_id], assume_unique=True)

            video_out_path = (
                OUT_DIR / action / f"{file.stem}-{i}-{scene_class}"
            ).with_suffix(".mp4")

            if video_out_path.exists():
                bar.update(1)
                i += 1
                continue

            scene_mask_path = (MASK_DIR / scene_class / scene_stem).with_suffix(".npz")
            scene_mask = np.load(scene_mask_path)["arr_0"]

            if len(scene_mask) > 500:
                continue

            if COMPUTE_ARTIFACT:
                artifact_list[file_idx * MULTIPLICATION + i] = compute_artifact(
                    action_mask, scene_mask
                )

            scene_path = (VIDEO_DIR / scene_class / scene_stem).with_suffix(EXT)

            if WRITE_VIDEOS:
                out_frames = cutmix_fn(
                    file,
                    action_mask,
                    scene_path,
                    scene_mask,
                    scene_transform,
                    SOFT_EDGE.enabled,
                    scene_transform_rand,
                )

                if not out_frames:
                    continue

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

    if COMPUTE_ARTIFACT:
        print("Artifact ratio:", artifact_list.mean())


if __name__ == "__main__":
    main()
