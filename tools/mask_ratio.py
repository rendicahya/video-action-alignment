import sys

sys.path.append(".")

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import click
import numpy as np
from tqdm import tqdm

from assertpy.assertpy import assert_that
from config import settings as conf


def calc_ratio(path):
    mask = np.load(path)["arr_0"]
    ratio = np.count_nonzero(mask) / mask.size

    return path.stem, round(ratio, 4)


ROOT = Path.cwd()
DATASET = conf.active.dataset
DETECTOR = conf.active.detector
DET_CONFIDENCE = conf.detect[DETECTOR].confidence
N_FILES = conf.datasets[DATASET].n_videos
MASK_DIR = ROOT / "data" / DATASET / DETECTOR / str(DET_CONFIDENCE) / "detect/mask"
MAX_WORKERS = conf.active.max_workers
OUT_PATH = MASK_DIR / "ratio.json"
data = {}

print("Input:", MASK_DIR.relative_to(ROOT))
print("Output:", OUT_PATH.relative_to(ROOT))
print("n videos:", N_FILES)

if not click.confirm("\nDo you want to continue?", show_default=True):
    exit("Aborted.")

assert_that(MASK_DIR).is_directory().is_readable()

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exec:
    jobs = {exec.submit(calc_ratio, path): path for path in MASK_DIR.glob("**/*.npz")}

    for future in tqdm(as_completed(jobs), total=N_FILES, dynamic_ncols=True):
        path_stem, ratio = future.result()
        data[path_stem] = ratio

with open(OUT_PATH, "w") as f:
    json.dump(data, f)
