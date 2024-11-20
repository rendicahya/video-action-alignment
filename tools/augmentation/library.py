import random

import cupy as cp
import cv2
import mmcv
import numpy as np


def videomix_fn(actor_path, scene_path):
    actor_reader = mmcv.VideoReader(str(actor_path))
    scene_frame = None
    alpha = 1
    w, h = actor_reader.resolution
    lambda_ = np.random.beta(alpha, alpha)
    x1, y1, x2, y2 = rand_bbox(w, h, lambda_)

    for f, actor_frame in enumerate(actor_reader):
        if scene_frame is None:
            scene_reader = mmcv.VideoReader(str(scene_path))
            scene_n_frames = scene_reader.frame_cnt
            scene_frame = scene_reader.read()

        if scene_frame.shape[:2] != (h, w):
            scene_frame = cv2.resize(scene_frame, (w, h))

        actor_frame[y1:y2, x1:x2] = scene_frame[y1:y2, x1:x2]

        yield cv2.cvtColor(actor_frame, cv2.COLOR_BGR2RGB)


def cutmix_fn(
    actor_path,
    action_mask,
    scene_path,
    scene_mask=None,
    scene_transform=None,
    soft_edge=False,
    scene_transform_rand=random.Random(),
):
    actor_reader = mmcv.VideoReader(str(actor_path))
    w, h = actor_reader.resolution
    scene_frame = None
    blank = np.zeros((h, w), np.uint8)

    if scene_transform:
        do_scene_transform = scene_transform_rand.random() <= scene_transform["prob"]

    if scene_mask is not None and scene_mask.shape[1:] != (h, w):
        scene_mask = np.moveaxis(scene_mask, 0, -1)
        scene_mask = cv2.resize(scene_mask, dsize=(w, h))
        scene_mask = np.moveaxis(scene_mask, -1, 0)

    if soft_edge:
        actor_mask_norm = action_mask / 255.0
        scene_mask_norm = scene_mask / 255.0

    for f, actor_frame in enumerate(actor_reader):
        if f == len(action_mask) - 1:
            return

        if scene_frame is None:
            scene_reader = mmcv.VideoReader(str(scene_path))
            scene_n_frames = scene_reader.frame_cnt
            scene_frame = scene_reader.read()

        if scene_frame.shape[:2] != (h, w):
            scene_frame = cv2.resize(scene_frame, (w, h))

        if scene_mask is not None and not soft_edge:
            is_foreground = scene_mask[f % scene_n_frames] == 255
            scene_frame[is_foreground] = 0

        actor_mask = action_mask[f] if action_mask.ndim == 3 else action_mask

        if actor_mask is None:
            actor_mask = blank

        if scene_transform and do_scene_transform:
            scene_frame = scene_transform["fn"](scene_frame)

        if soft_edge:
            actor_mask_3 = np.repeat(
                np.expand_dims(actor_mask_norm[f], axis=2), 3, axis=2
            )
            scene_mask_3 = np.repeat(
                np.expand_dims(1 - scene_mask_norm[f % scene_n_frames], axis=2),
                3,
                axis=2,
            )
            actor = (actor_frame * actor_mask_3).astype(np.uint8)
            scene = (scene_frame * scene_mask_3 * (1 - actor_mask_3)).astype(np.uint8)
        else:
            actor = cv2.bitwise_and(actor_frame, actor_frame, mask=actor_mask)
            scene = cv2.bitwise_and(scene_frame, scene_frame, mask=255 - actor_mask)

        mix = cv2.add(actor, scene)

        scene_frame = scene_reader.read()

        yield cv2.cvtColor(mix, cv2.COLOR_BGR2RGB)


def compute_artifact(fg_mask, bg_mask) -> float:
    fg_len = len(fg_mask)
    bg_len = len(bg_mask)
    h, w = fg_mask.shape[1:]

    if bg_mask.shape[1:] != (h, w):
        bg_mask = np.moveaxis(bg_mask, 0, -1)
        bg_mask = cv2.resize(bg_mask, dsize=(w, h))
        bg_mask = np.moveaxis(bg_mask, -1, 0)

    fg_mask = cp.array(fg_mask, dtype=cp.bool_)
    bg_mask = cp.array(bg_mask, dtype=cp.bool_)

    bg_mask_repeated = (
        bg_mask[:fg_len]
        if fg_len <= bg_len
        else cp.tile(bg_mask, (fg_len // bg_len + 1, 1, 1))[:fg_len]
    )

    bg_area = bg_mask_repeated.sum(axis=(1, 2), dtype=cp.float32)
    intersection = (fg_mask & bg_mask_repeated).sum(axis=(1, 2), dtype=cp.float32)
    diff_ratio = cp.where(bg_area == 0, 0, (bg_area - intersection) / bg_area).mean()

    return diff_ratio
