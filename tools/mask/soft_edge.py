import sys

import cv2
import numpy as np

sys.path.append(".")

from pathlib import Path

import click
from tqdm import tqdm

from assertpy.assertpy import assert_that
from config import settings as conf


def add_suffix(path: Path, suffix: str):
    return path.parent / (path.stem + suffix)


def main():
    ROOT = Path.cwd()
    DATASET = conf.active.DATASET
    N_VIDEOS = conf.datasets[DATASET].n_videos
    DETECTOR = conf.active.DETECTOR
    DET_CONFIDENCE = conf.detect[DETECTOR].confidence
    TEMPORAL_MORPHOLOGY_ENABLED = conf.cutmix.temporal_morphology.enabled
    KERNEL_SIZE = conf.cutmix.soft_edge.kernel_size
    MASK_DIR = ROOT / "data" / DATASET / DETECTOR / str(DET_CONFIDENCE) / "detect/mask"

    if TEMPORAL_MORPHOLOGY_ENABLED:
        TEMPORAL_MORPHOLOGY_OP = conf.cutmix.temporal_morphology.op
        MASK_DIR = add_suffix(MASK_DIR, "-" + TEMPORAL_MORPHOLOGY_OP)

    OUT_DIR = add_suffix(MASK_DIR, "-soft")

    print("Input:", MASK_DIR.relative_to(ROOT))
    print(
        "Output:",
        OUT_DIR.relative_to(ROOT),
        "(exists)" if OUT_DIR.exists() else "(not exists)",
    )

    assert_that(MASK_DIR).is_directory().exists()

    if not click.confirm("\nDo you want to continue?", show_default=True):
        exit("Aborted.")

    for file in tqdm(MASK_DIR.glob(f"**/*.npz"), total=N_VIDEOS, dynamic_ncols=True):
        if file.parent == MASK_DIR:
            continue

        mask = np.load(file)["arr_0"]
        soft = np.stack(
            [cv2.GaussianBlur(frame, (KERNEL_SIZE, KERNEL_SIZE), 0) for frame in mask]
        )
        action = file.parent.name
        out_path = OUT_DIR / action / f"{file.stem}.npz"

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, soft)


if __name__ == "__main__":
    main()
