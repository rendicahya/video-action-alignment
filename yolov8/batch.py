import sys

sys.path.append(".")

import json
from pathlib import Path

import click
import numpy as np
from config import settings as conf
from tqdm import tqdm
from ultralytics import YOLO

from assertpy.assertpy import assert_that

root = Path.cwd()
dataset = conf.active.dataset
detector = conf.active.detector
ext = conf[dataset].ext
video_in_dir = root / conf[dataset].path
generate_video = conf.yolov8.generate_videos
confidence = conf.yolov8.confidence
checkpoint = conf.yolov8.checkpoint
json_out_dir = root / f"data/{dataset}/{detector}/detect/{confidence}/json"

print("Dataset:", dataset)
print("Dataset path:", video_in_dir.relative_to(root))
print("Generate video:", generate_video)
print("Checkpoint:", checkpoint)

assert_that(video_in_dir).is_directory().is_readable()
assert_that(checkpoint).is_file().is_readable()

if not click.confirm("\nDo you want to continue?", show_default=True):
    exit("Aborted.")

model = YOLO(checkpoint)
bar = tqdm(total=conf[dataset].n_videos)

for file in video_in_dir.glob(f"**/*{ext}"):
    action = file.parent.name
    json_out_path = json_out_dir / action / file.with_suffix(".json").name

    if json_out_path.exists() and json_out_path.stat().st_size:
        bar.update(1)
        continue

    detection_data = {}
    results = model(file, stream=True, conf=confidence, device="cuda:0", verbose=False)

    for i, result in enumerate(results):
        detection_data.update(
            {
                i: [
                    (
                        box.xywh.cpu().numpy().astype(int).tolist(),
                        round(box.conf.cpu().numpy().astype(float)[0], 2),
                        int(box.cls.cpu().numpy()[0]),
                    )
                    for box in result.boxes
                    if box.cls == 0
                ]
            }
        )

    json_out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_out_path, "w") as json_file:
        json.dump(detection_data, json_file)

    bar.update(1)

bar.close()
