import random

import cv2
import mmcv
import numpy as np


def cutmix_fn(
    actor_path,
    action_mask,
    scene_path,
    scene_mask=None,
    scene_transform=None,
    soft_edge=False,
    scene_transform_rand=random.Random(),
):
    if not actor_path.is_file() or not actor_path.exists():
        print("Not a file or not exists:", actor_path)
        return None

    if not scene_path.is_file() or not scene_path.exists():
        print("Not a file or not exists:", scene_path)
        return None

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

        actor_mask = action_mask[f]

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
