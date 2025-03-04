import sys

sys.path.append(".")

import pickle
from collections import defaultdict
from os.path import splitext
from pathlib import Path

import click
import mmcv
import numpy as np
from library import cutmix_fn
from tqdm import tqdm

from assertpy.assertpy import assert_that
from config import settings as conf
from python_video import frames_to_video


def append_name(path: Path, name: str):
    return path.parent / (path.stem + name + path.suffix)


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
    EXT = conf.datasets[DATASET].ext
    N_VIDEOS = conf.datasets[DATASET].N_VIDEOS
    DETECTOR = conf.active.detector
    DET_CONF = str(conf.detect[DETECTOR].confidence)
    MULTIPLICATION = conf.cutmix.multiplication
    SCENE_SELECTION = conf.cutmix.scene.selection.method
    ACTION_DIR = ROOT / "data" / DATASET / "videos"
    SCENE_DIR = ROOT / "data" / DATASET / "scene"
    MASK_DIR = ROOT / "data" / DATASET / DETECTOR / DET_CONF / "detect/mask-dilation"
    MATRIX_PATH = (
        ROOT / "data" / DATASET / DETECTOR / DET_CONF / "detect/mask/iou-std.npz"
    )
    MATRIX = np.load(MATRIX_PATH)["arr_0"]
    check_value = MATRIX[-2, -1]
    WORK_DIR = ROOT / "mmaction2/work_dirs"
    OUT_DIR = ROOT / "data" / DATASET / "mix2train" / SCENE_SELECTION

    print("n videos:", N_VIDEOS)
    print("Action:", ACTION_DIR.relative_to(ROOT))
    print("Scene:", SCENE_DIR.relative_to(ROOT))
    print("Mask:", MASK_DIR.relative_to(ROOT))
    print("Matrix:", MATRIX_PATH.relative_to(ROOT))
    print(
        "Output:",
        OUT_DIR.relative_to(ROOT),
        "(exists)" if OUT_DIR.exists() else "(not exists)",
    )

    assert_that(ACTION_DIR).is_directory().is_readable()
    assert_that(SCENE_DIR).is_directory().is_readable()
    assert_that(MASK_DIR).is_directory().is_readable()
    assert_that(SCENE_DIR / "list.txt").is_file().is_readable()
    assert_that(ACTION_DIR / "list.txt").is_file().is_readable()
    assert_that(dump_path).is_file().is_readable()
    assert_that(dump_path).is_named("dump.pkl")
    assert_that(check_value).is_not_equal_to(0.0)

    if not click.confirm("\nDo you want to continue?", show_default=True):
        exit("Aborted.")

    idx2stem = []
    stem2label = {}
    scene_label_list = np.zeros(N_VIDEOS, np.uint8)
    label2idx = {}

    with open(dump_path, "rb") as file:
        dump_data = pickle.load(file)

    with open(ACTION_DIR / "list.txt") as file:
        for line in file:
            subpath, label_idx = line.split()
            label, filename = subpath.split("/")
            stem = splitext(filename)[0]

            stem2label[stem] = label
            idx2stem.append(stem)

    with open(SCENE_DIR / "list.txt") as f:
        scene_file_list = f.readlines()

    for i, line in enumerate(scene_file_list):
        subpath, label_idx = line.split()
        label, filename = subpath.split("/")
        scene_label_list[i] = int(label_idx)

        if label not in label2idx:
            label2idx[label] = int(label_idx)

    bar = tqdm(total=len(dump_data) * MULTIPLICATION, dynamic_ncols=True)
    n_written = 0

    for scene_file_idx, line in enumerate(scene_file_list):
        subpath, label_idx = line.split()
        info = dump_data[scene_file_idx]

        if info["pred_label"] != info["gt_label"]:
            bar.update(MULTIPLICATION)
            continue

        scene_label, scene_file = subpath.split("/")
        scene_stem = splitext(scene_file)[0]
        scene_path = SCENE_DIR / subpath
        fps = mmcv.VideoReader(str(scene_path)).fps

        # Get similarity scores from matrix
        matrix_row = MATRIX[scene_file_idx][scene_file_idx:]
        matrix_col = MATRIX[:, scene_file_idx][:scene_file_idx]
        matrix_merge = np.concatenate((matrix_col, matrix_row))
        all_videos = np.argsort(matrix_merge)

        # Videos having the same action
        same_label = np.where(np.isin(scene_label_list, [int(label_idx)]))

        # Exclude from sorting
        eligible = np.setdiff1d(all_videos, same_label, assume_unique=True)

        for i in range(MULTIPLICATION):
            # Get action video with max score
            action_id = eligible[-1]
            action_stem = idx2stem[action_id]
            action_label = stem2label[action_stem]
            action_path = (ACTION_DIR / action_label / action_stem).with_suffix(EXT)
            mask_path = MASK_DIR / action_label / (action_stem + ".npz")
            out_path = (
                OUT_DIR
                / action_label
                / f'{scene_stem.replace("-", "")}-{action_stem.replace("-", "")}'
            ).with_suffix(".mp4")

            if not mask_path.exists():
                bar.update(1)
                continue

            if out_path.exists() and mmcv.VideoReader(str(out_path)).frame_cnt > 0:
                bar.update(1)
                continue

            if SCENE_SELECTION in ("iou-v", "bao-v"):
                action_label_idx = label2idx[action_label]
                videos_selected_label = np.where(
                    np.isin(scene_label_list, [action_label_idx])
                )
                eligible = np.setdiff1d(
                    eligible, videos_selected_label, assume_unique=True
                )
            elif SCENE_SELECTION in ("iou-m", "bao-m"):
                eligible = np.setdiff1d(eligible, [action_id], assume_unique=True)

            out_path.parent.mkdir(parents=True, exist_ok=True)

            action_mask = np.load(mask_path)["arr_0"]
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

    bar.close()
    print("Written videos:", n_written)


if __name__ == "__main__":
    main()
