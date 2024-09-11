import sys

sys.path.append(".")

import pickle
from pathlib import Path

import click
import cv2
import mmcv
import numpy as np
from config import settings as conf
from moviepy.editor import ImageSequenceClip
from tqdm import tqdm
from ultralytics import YOLO

from assertpy.assertpy import assert_that

root = Path.cwd()
dataset = conf.active.dataset
detector = conf.active.detector
ext = conf[dataset].ext
video_in_dir = root / conf[dataset].path
generate_video = conf.detect.generate_videos
det_confidence = conf.detect[detector].confidence
checkpoint = conf.detect[detector].checkpoint
mask_out_dir = root / f"data/{dataset}/{detector}/detect/{det_confidence}/mask"
dump_out_dir = root / f"data/{dataset}/{detector}/detect/{det_confidence}/dump"
video_out_dir = root / f"data/{dataset}/{detector}/detect/{det_confidence}/video"
human_class = conf.detect[detector].human_class

print("Input:", video_in_dir.relative_to(root))
print("Output mask:", mask_out_dir.relative_to(root))
print("Output dump:", dump_out_dir.relative_to(root))
print("Checkpoint:", checkpoint)

if generate_video:
    print("Output video:", video_out_dir.relative_to(root))

assert_that(video_in_dir).is_directory().is_readable()
assert_that(checkpoint).is_file().is_readable()

if not click.confirm("\nDo you want to continue?", show_default=True):
    exit("Aborted.")

model = YOLO(checkpoint)
bar = tqdm(total=conf[dataset].n_videos)

for file in video_in_dir.glob(f"**/*{ext}"):
    action = file.parent.name
    mask_out_path = mask_out_dir / action / file.stem
    dump_out_path = dump_out_dir / action / file.with_suffix(".pckl").name

    if mask_out_path.exists() and mask_out_path.stat().st_size:
        bar.update(1)
        continue

    if generate_video:
        output_frames = []

    det_results = model(
        file, stream=True, conf=det_confidence, device="cuda:0", verbose=False
    )
    video = mmcv.VideoReader(str(file))
    n_frames = video.frame_cnt
    mask_cube = np.zeros((n_frames, video.height, video.width), np.uint8)
    dump_data = {}

    for i, result in enumerate(det_results):
        boxes = np.rint(result.boxes.xywh.cpu().numpy()).astype(int)
        classes = result.boxes.cls
        confidences = result.boxes.conf
        frame_id = "%06d" % int(i)
        frame_dump = []

        for cls, confidence, (x, y, w, h) in zip(classes, confidences, boxes):
            if cls not in human_class:
                continue

            half_w = w // 2
            half_h = h // 2

            x1 = x - half_w
            y1 = y - half_h
            x2 = x + half_w
            y2 = y + half_h

            mask_cube[int(i), y1:y2, x1:x2] = 255

            frame_dump.append(
                {
                    "image_id": frame_id,
                    "bbox": [x1, y1, w, h],
                    "scores": confidence,
                    "bbox_center": [x, y],
                }
            )

        dump_data[frame_id] = frame_dump

        if generate_video:
            frame = result.plot()[..., ::-1]

            output_frames.append(frame)

    mask_out_path.parent.mkdir(parents=True, exist_ok=True)
    dump_out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(mask_out_path, mask_cube)

    with open(dump_out_path, "wb") as f:
        pickle.dump((file.name, dump_data), f)

    if generate_video:
        video_out_path = video_out_dir / action / file.with_suffix(".mp4").name
        fps = video.fps

        video_out_path.parent.mkdir(parents=True, exist_ok=True)
        ImageSequenceClip(output_frames, fps=fps).write_videofile(
            str(video_out_path), logger=None
        )

    bar.update(1)

bar.close()
