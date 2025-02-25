import sys

sys.path.append(".")

import json
from pathlib import Path

from assertpy.assertpy import assert_that
from config import settings as conf


def main():
    ROOT = Path.cwd()
    DATASET = conf.active.dataset
    DETECTOR = conf.active.detector
    DET_CONF = str(conf.detect[DETECTOR].confidence)
    TEMPORAL_MORPHOLOGY = conf.cutmix.morphology.temporal
    MASK_DIR = (
        ROOT
        / "data"
        / DATASET
        / DETECTOR
        / DET_CONF
        / "detect"
        / ("mask-dilation" if TEMPORAL_MORPHOLOGY.enabled else "mask")
    )
    RATIO_FILE = MASK_DIR / "ratio.json"

    with open(RATIO_FILE) as f:
        ratio_list = json.load(f)

    avg_ratio = sum([ratio for file, ratio in ratio_list.items()]) / len(ratio_list)

    print("File:", RATIO_FILE.relative_to(ROOT))
    print("Average ratio:", round(avg_ratio * 100, 2))


if __name__ == "__main__":
    main()
