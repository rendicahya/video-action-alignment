import sys

sys.path.append(".")

from pathlib import Path

from assertpy.assertpy import assert_that
from config import settings as conf


def main():
    ROOT = Path.cwd()
    DATASET = conf.active.dataset
    DATASET_DIR = ROOT / conf.datasets[DATASET].path
    OUT_PATH = DATASET_DIR.parent / "annotations/classInd.txt"

    assert_that(DATASET_DIR).is_directory().exists()

    subdirs = [subdir for subdir in DATASET_DIR.iterdir() if subdir.is_dir()]

    subdirs.sort(key=lambda x: x.name)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_PATH, "w") as f:
        for i, subdir in enumerate(subdirs, start=1):
            f.write(f"{i} {subdir.name}\n")


if __name__ == "__main__":
    main()
