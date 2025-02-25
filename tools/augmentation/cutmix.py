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

    if SCENE_SELECTION == "videomix":
        MASK_DIR = ROOT / "data" / DATASET / "videomix/mask"
        OUT_DIR = ROOT / "data" / DATASET / "videomix/videos"

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

    if SCENE_SELECTION in ("iou-v", "iou-m", "bao-v", "bao-m"):
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
    assert_that(SCENE_SELECTION).is_in(
        "random", "iou-v", "iou-m", "bao-v", "bao-m", "videomix"
    )
    assert_that(SCENE_TRANSFORM.op).is_in("hflip")

    if not click.confirm("\nDo you want to continue?", show_default=True):
        exit("Aborted.")

    random.seed(RANDOM_SEED)

    label2scenes = defaultdict(list)
    scene2label = {}
    label_list = np.zeros(N_VIDEOS, np.uint8)
    scene_transform_rand = random.Random()
    idx2stem = []
    label2idx = {}

    with open(VIDEO_DIR / "list.txt") as f:
        file_list = f.readlines()

    for i, line in enumerate(file_list):
        path, label_idx = line.split()
        label, filename = path.split("/")
        stem = splitext(filename)[0]

        if SCENE_SELECTION in ("random", "videomix"):
            label2scenes[label].append(stem)
        else:
            scene2label[stem] = label
            label_list[i] = int(label_idx)
            idx2stem.append(stem)

            if label not in label2idx:
                label2idx[label] = int(label_idx)

    bar = tqdm(total=N_VIDEOS * MULTIPLICATION, dynamic_ncols=True)
    n_written = 0
    artifact_list = np.zeros(N_VIDEOS * MULTIPLICATION, dtype=np.float32)

    for file_idx, line in enumerate(file_list):
        # if file_idx != 231:
        #     continue

        path, label_idx = line.split()
        file = Path(VIDEO_DIR / path)
        label = file.parent.name
        video_mask_path = (MASK_DIR / label / file.name).with_suffix(".npz")
        action_mask = np.load(video_mask_path)["arr_0"]

        if WRITE_VIDEOS:
            fps = mmcv.VideoReader(str(file)).fps

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

        if SCENE_SELECTION in ("random", "videomix"):
            scene_label_options = [s for s in label2scenes.keys() if s != label]
        else:
            if SCENE_SELECTION.startswith("iou"):
                row = MATRIX[file_idx][file_idx:]
                col = MATRIX[:, file_idx][:file_idx]
                sim_data = np.concatenate((col, row))
            else:
                sim_data = MATRIX[file_idx]

            all_videos = np.argsort(sim_data)

            # Videos from the current action
            videos_same_label = np.where(np.isin(label_list, [int(label_idx)]))

            # Exclude them from sorting
            eligible = np.setdiff1d(all_videos, videos_same_label, assume_unique=True)

        i = 0

        while i < MULTIPLICATION:
            bar.set_description(f"{file.stem[:20]} ({i+1}/{MULTIPLICATION})")

            if SCENE_SELECTION in ("random", "videomix"):
                scene_label = random.choice(scene_label_options)
                scene_options = label2scenes[scene_label]
                scene_stem = random.choice(scene_options)

                scene_label_options.remove(scene_label)
            else:
                scene_id = eligible[-1]
                scene_stem = idx2stem[scene_id]
                scene_label = scene2label[scene_stem]

                if SCENE_SELECTION in ("iou-v", "bao-v"):
                    scene_label_idx = label2idx[scene_label]
                    videos_selected_label = np.where(
                        np.isin(label_list, [scene_label_idx])
                    )

                    # Remove the videos of the selected action from the eligible list
                    eligible = np.setdiff1d(
                        eligible, videos_selected_label, assume_unique=True
                    )
                elif SCENE_SELECTION in ("iou-m", "bao-m"):
                    # Remove the selected from the eligible list
                    eligible = np.setdiff1d(eligible, [scene_id], assume_unique=True)

            if SCENE_SELECTION == "videomix":
                scene_mask = None
            else:
                scene_mask_path = (MASK_DIR / scene_label / scene_stem).with_suffix(
                    ".npz"
                )
                scene_mask = np.load(scene_mask_path)["arr_0"]

            if len(scene_mask) > 500:
                continue

            if COMPUTE_ARTIFACT:
                artifact_list[file_idx * MULTIPLICATION + i] = compute_artifact(
                    action_mask, scene_mask
                )

            if WRITE_VIDEOS:
                video_out_path = (
                    OUT_DIR / label / f"{file.stem}-{i}-{scene_label}"
                ).with_suffix(".mp4")

                if video_out_path.exists():
                    bar.update(1)
                    i += 1
                    continue

                scene_path = (VIDEO_DIR / scene_label / scene_stem).with_suffix(EXT)
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
        OUT_DIR.mkdir(parents=True, exist_ok=True)

        with open(OUT_DIR / "artifact.json", "w") as f:
            json.dump(artifact_list.tolist(), f, indent=2)

        print("Artifact ratio:", artifact_list.mean())


if __name__ == "__main__":
    main()
