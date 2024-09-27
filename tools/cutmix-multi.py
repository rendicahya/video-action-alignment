import multiprocessing
import os
import sys
from threading import Thread

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


def thread_job(
    f,
    actor_frame,
    actor_mask,
    scene_frame,
    scene_mask_cube,
    scene_n_frames,
    blank,
    scene_replace,
    dim,
    result_dict,
):
    # print("Frame", f)

    if scene_frame.shape[:2] != dim[::-1]:
        scene_frame = cv2.resize(scene_frame, dim)

    if scene_replace in ("white", "black"):
        is_foreground = scene_mask_cube[f % scene_n_frames] == 255

    if scene_replace == "white":
        scene_frame[is_foreground] = 255
    elif scene_replace == "black":
        scene_frame[is_foreground] = 0

    if actor_mask is None:
        actor_mask = blank

    actor = cv2.bitwise_and(actor_frame, actor_frame, mask=actor_mask)
    scene = cv2.bitwise_and(scene_frame, scene_frame, mask=255 - actor_mask)
    mix = cv2.add(actor, scene)
    mix = cv2.cvtColor(mix, cv2.COLOR_BGR2RGB)

    result_dict.update({f: mix})


def cutmix_fn(actor_path, scene_path, actor_mask_cube, scene_replace, scene_mask_cube):
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
    result_dict = {}

    if scene_mask_cube.shape[:2] != (h, w) and scene_replace in ("white", "black"):
        scene_mask_cube = np.moveaxis(scene_mask_cube, 0, -1)
        scene_mask_cube = cv2.resize(scene_mask_cube, dsize=(w, h))
        scene_mask_cube = np.moveaxis(scene_mask_cube, -1, 0)

    threads = []

    for f, actor_frame in enumerate(actor_reader):
        if f == len(actor_mask_cube) - 1:
            return

        if scene_frame is None:
            scene_reader = mmcv.VideoReader(str(scene_path))
            scene_n_frames = scene_reader.frame_cnt
            scene_frame = scene_reader.read()

        actor_mask = actor_mask_cube[f]

        thread = Thread(
            target=thread_job,
            args=(
                f,
                actor_frame,
                actor_mask,
                scene_frame,
                scene_mask_cube,
                scene_n_frames,
                blank,
                scene_replace,
                actor_reader.resolution,
                result_dict,
            ),
        )

        thread.start()
        threads.append(thread)

        scene_frame = scene_reader.read()

    print("Finish")
    for thread in threads:
        thread.join()

    # print(len(result_dict))
    # result_dict = dict(sorted(result_dict.items()))
    # result_dict = np.stack(result_dict.values())

    # return result_dict


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
    video_writer = conf.cutmix.output.writer
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
    print("Video writer:", video_writer)

    assert_that(video_in_dir).is_directory().is_readable()
    assert_that(mask_dir).is_directory().is_readable()
    assert_that(scene_replace).is_in("white", "black", "inpaint")

    # if not click.confirm("\nDo you want to continue?", show_default=True):
    #     exit("Aborted.")

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

        actor_mask_cube = np.load(video_mask_path)["arr_0"]
        fps = mmcv.VideoReader(str(file)).fps
        scene_action_options = [s for s in scene_dict.keys() if s != action]
        i = 0

        while i < multiplication:
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
                scene_mask_cube = None
            else:
                scene_pick_base = os.path.splitext(scene_pick)[0]
                scene_mask_cube_path = (
                    mask_dir / scene_action_pick / f"{scene_pick_base}.npz"
                )
                scene_mask_cube = np.load(scene_mask_cube_path)["arr_0"]

                if len(scene_mask_cube) > 500:
                    continue

            scene_path = video_in_dir / scene_action_pick / scene_pick
            out_frames = cutmix_fn(
                file, scene_path, actor_mask_cube, scene_replace, scene_mask_cube
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
            break
        break

    bar.close()
    print("Written videos:", n_written)


if __name__ == "__main__":
    main()
