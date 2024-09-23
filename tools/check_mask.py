import sys

sys.path.append(".")
import numpy as np

import random
from pathlib import Path
from tqdm import tqdm

import mmcv

from assertpy.assertpy import assert_that
from config import settings as conf


def main():
    root = Path.cwd()
    dataset = conf.active.dataset
    detector = conf.active.detector
    det_confidence = conf.detect[detector].confidence
    video_ext = conf.datasets[dataset].ext
    n_videos = conf.datasets[dataset].n_videos
    random_seed = conf.active.random_seed
    video_in_dir = root / "data" / dataset / "videos"
    mask_dir = (
        root / "data" / dataset / detector / str(det_confidence) / "detect" / "mask"
    )

    print("Σ videos:", n_videos)
    print("Input:", video_in_dir.relative_to(root))
    print("Mask:", mask_dir.relative_to(root))

    assert_that(video_in_dir).is_directory().is_readable()
    assert_that(mask_dir).is_directory().is_readable()

    random.seed(random_seed)

    bar = tqdm(total=n_videos, dynamic_ncols=True)

    for file in video_in_dir.glob(f"**/*{video_ext}"):
        action = file.parent.name
        video_mask_path = mask_dir / action / file.with_suffix(".npz").name

        if not video_mask_path.is_file() or not video_mask_path.exists():
            continue

        mask_bundle = np.load(video_mask_path)["arr_0"]
        mask_shape = mask_bundle.shape

        video_info = mmcv.VideoReader(str(file))
        n_frames = video_info.frame_cnt
        res = video_info.resolution[::-1]

        if mask_shape[0] != n_frames or mask_shape[1:] != res:
            print(file.name)

        bar.update(1)


if __name__ == "__main__":
    main()
