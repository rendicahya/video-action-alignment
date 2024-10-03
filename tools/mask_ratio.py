import sys

sys.path.append(".")

import json
from pathlib import Path

import click
import numpy as np
from tqdm import tqdm

from assertpy.assertpy import assert_that
from config import settings as conf

root = Path.cwd()
dataset = conf.active.dataset
detector = conf.active.detector
det_confidence = conf.detect[detector].confidence
n_files = conf.datasets[dataset].n_videos
mask_dir = root / "data" / dataset / detector / str(det_confidence) / "detect" / "mask"
json_out_path = mask_dir / "ratio.json"

print("Input:", mask_dir)
print("Output:", json_out_path)
print("Σ videos:", n_files)

if not click.confirm("\nDo you want to continue?", show_default=True):
    exit("Aborted.")

assert_that(mask_dir).is_directory().is_readable()

data = {}
bar = tqdm(total=n_files, dynamic_ncols=True)

for mask_path in mask_dir.glob("**/*.npz"):
    mask_bundle = np.load(mask_path)["arr_0"]
    mask_ratio = np.count_nonzero(mask_bundle) / mask_bundle.size
    data[mask_path.stem] = mask_ratio

    bar.update(1)

bar.close()

with open(json_out_path, "w") as f:
    json.dump(data, f)
