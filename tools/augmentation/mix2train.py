import sys

sys.path.append(".")

import pickle
from collections import defaultdict
from os.path import splitext
from pathlib import Path

import click
import mmcv
import numpy as np
from lib_cutmix import cutmix_fn
from tqdm import tqdm

from assertpy.assertpy import assert_that
from config import settings as conf
from python_video import frames_to_video


@click.command()
@click.argument(
    "dump-path",
    nargs=1,
    required=True,
    type=click.Path(
        exists=True,
        readable=True,
        file_okay=True,
        dir_okay=False,
        path_type=Path,
    ),
)
def main(dump_path):
    ROOT = Path.cwd()
    DATASET = conf.active.dataset
    DETECTOR = conf.active.detector
    DET_CONF = str(conf.detect[DETECTOR].confidence)
    DET_CONFIDENCE = str(conf.detect[DETECTOR].confidence)
    MULTIPLICATION = conf.cutmix.multiplication
    ACTION_DIR = ROOT / "data" / DATASET / "videos"
    WORK_DIR = ROOT / "mmaction2/work_dirs"
    OUT_DIR = ROOT / "data" / DATASET / "mix2train"
    EXT = conf.datasets[DATASET].ext
    N_VIDEOS = conf.datasets[DATASET].N_VIDEOS
    SCENE_DIR = ROOT / "data" / DATASET / "scene"
    MASK_DIR = ROOT / "data" / DATASET / DETECTOR / DET_CONF / "detect/mask-dilation"
    MATRIX_PATH = ROOT / "data" / DATASET / DETECTOR / DET_CONF / "detect/mask/iou.npz"
    MATRIX = np.load(MATRIX_PATH)["arr_0"]
    check_value = MATRIX[-2, -1]

    print("n videos:", N_VIDEOS)
    print("Action:", ACTION_DIR.relative_to(ROOT))
    print("Scene:", SCENE_DIR.relative_to(ROOT))
    print("Mask dir:", MASK_DIR.relative_to(ROOT))
    print("Matrix:", MATRIX_PATH.relative_to(ROOT))
    print("Output:", OUT_DIR.relative_to(ROOT))

    assert_that(ACTION_DIR).is_directory().is_readable()
    assert_that(SCENE_DIR).is_directory().is_readable()
    assert_that(SCENE_DIR / "list.txt").is_file().is_readable()
    assert_that(ACTION_DIR / "list.txt").is_file().is_readable()
    assert_that(check_value).is_not_equal_to(0.0)

    if not click.confirm("\nDo you want to continue?", show_default=True):
        exit("Aborted.")

    idx2stem = []
    stem2class = {}
    scene_class_list = np.zeros(N_VIDEOS, np.uint8)

    with open(dump_path, "rb") as file:
        dump_data = pickle.load(file)

    with open(ACTION_DIR / "list.txt") as file:
        for line in file:
            subpath, class_idx = line.split()
            class_name, filename = subpath.split("/")
            stem = splitext(filename)[0]

            stem2class[stem] = class_name
            idx2stem.append(stem)

    with open(SCENE_DIR / "list.txt") as f:
        scene_file_list = f.readlines()

    for i, line in enumerate(scene_file_list):
        subpath, class_idx = line.split()
        class_name, filename = subpath.split("/")
        scene_class_list[i] = int(class_idx)

    bar = tqdm(total=len(dump_data) * MULTIPLICATION, dynamic_ncols=True)
    n_skipped = 0
    n_written = 0

    for scene_file_idx, line in enumerate(scene_file_list):
        subpath, class_idx = line.split()
        used_videos = [int(scene_file_idx)]
        info = dump_data[scene_file_idx]

        if info["pred_label"] != info["gt_label"]:
            bar.update(MULTIPLICATION)
            continue

        scene_class, scene_file = subpath.split("/")
        scene_stem = splitext(scene_file)[0]
        scene_path = SCENE_DIR / subpath
        fps = mmcv.VideoReader(str(scene_path)).fps

        # Get similarity scores from matrix
        matrix_row = MATRIX[scene_file_idx][scene_file_idx:]
        matrix_col = MATRIX[:, scene_file_idx][:scene_file_idx]
        matrix_merge = np.concatenate((matrix_col, matrix_row))
        all_options = np.argsort(matrix_merge)

        # Videos having the same action
        same_class = np.where(np.isin(scene_class_list, [int(class_idx)]))

        # Exclude from sorting
        eligible = np.setdiff1d(all_options, same_class, assume_unique=True)

        for _ in range(MULTIPLICATION):
            eligible = np.setdiff1d(eligible, used_videos, assume_unique=True)

            # Get action video with max score
            action_id = eligible[-1]
            action_stem = idx2stem[action_id]
            action_class = stem2class[action_stem]
            action_path = (ACTION_DIR / action_class / action_stem).with_suffix(EXT)
            mask_path = MASK_DIR / action_class / (action_stem + ".npz")
            out_path = (
                OUT_DIR / action_class / f"{scene_stem}-{action_stem}"
            ).with_suffix(".mp4")

            if not mask_path.exists():
                bar.update(1)
                continue

            action_mask = np.load(mask_path)["arr_0"]

            if out_path.exists() and mmcv.VideoReader(str(out_path)).frame_cnt > 0:
                bar.update(1)
                continue

            used_videos.append(action_id)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            out_frames = cutmix_fn(action_path, action_mask, scene_path)

            if out_frames:
                frames_to_video(
                    out_frames,
                    out_path,
                    writer="moviepy",
                    fps=fps,
                )

                n_written += 1
            else:
                print("out_frames None: ", filename)

            bar.update(1)
            # break

        break

    bar.close()
    print("Written videos:", n_written)
    print("Skipped videos:", n_skipped)


if __name__ == "__main__":
    main()
