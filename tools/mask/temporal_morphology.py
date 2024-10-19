import sys

import numpy as np

sys.path.append(".")

from pathlib import Path

import click
from scipy.ndimage import binary_closing, binary_dilation, binary_opening
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
    MASK_DIR = ROOT / "data" / DATASET / DETECTOR / str(DET_CONFIDENCE) / "detect/mask"
    MORPHOLOGY_OP = conf.cutmix.temporal_morphology.op
    MORPHOLOGY_LENGTH = conf.cutmix.temporal_morphology.length
    OUT_DIR = add_suffix(MASK_DIR, "-" + MORPHOLOGY_OP)

    print("Operation:", MORPHOLOGY_OP)
    print("Input:", MASK_DIR.relative_to(ROOT))
    print("Output:", OUT_DIR.relative_to(ROOT))

    if not click.confirm("\nDo you want to continue?", show_default=True):
        exit("Aborted.")

    assert_that(MORPHOLOGY_OP).is_in("dilation", "opening", "closing")

    kernel = np.ones((MORPHOLOGY_LENGTH, 1, 1))
    morphology_fn = {
        "dilation": binary_dilation,
        "opening": binary_opening,
        "closing": binary_closing,
    }

    for file in tqdm(MASK_DIR.glob(f"**/*.npz"), total=N_VIDEOS, dynamic_ncols=True):
        if file.parent == MASK_DIR:
            continue

        mask = np.load(file)["arr_0"]
        mask = morphology_fn[MORPHOLOGY_OP](mask == 255, structure=kernel)
        mask = mask.astype(np.uint8) * 255
        action = file.parent.name
        out_path = OUT_DIR / action / f"{file.stem}.npz"

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, mask)


if __name__ == "__main__":
    main()
