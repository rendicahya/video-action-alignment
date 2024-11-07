import cupy as cp
import cv2
import numpy as np
from numba import njit, prange


def compute_sim_uint8(fg_mask, bg_mask):
    bg_len = len(bg_mask)
    fg_len = len(fg_mask)
    score_list = np.empty((3, fg_len), np.float32)

    if fg_mask[0].shape != bg_mask[0].shape:
        bg_mask = [cv2.resize(frame, fg_mask[0].shape[::-1]) for frame in bg_mask]

    for f, fg_frame in enumerate(fg_mask):
        bg_frame = bg_mask[f % bg_len]

        intersection = np.logical_and(fg_frame, bg_frame).sum()
        union = np.logical_or(fg_frame, bg_frame).sum()

        fg_area = np.count_nonzero(fg_frame)
        bg_area = np.count_nonzero(bg_frame)

        iou = 0.0 if union == 0 else intersection / union
        iob = 0.0 if bg_area == 0 else intersection / bg_area
        iof = 0.0 if fg_area == 0 else intersection / fg_area

        score_list[..., f] = iou, iob, iof

    return np.mean(score_list, axis=1)


def compute_sim_uint8_std(fg_mask, bg_mask):
    bg_len = len(bg_mask)
    fg_len = len(fg_mask)
    score_list = np.empty((3, fg_len), np.float32)

    for f, fg_frame in enumerate(fg_mask):
        bg_frame = bg_mask[f % bg_len]

        intersection = np.logical_and(fg_frame, bg_frame).sum()
        union = np.logical_or(fg_frame, bg_frame).sum()

        fg_area = np.count_nonzero(fg_frame)
        bg_area = np.count_nonzero(bg_frame)

        iou = 0.0 if union == 0 else intersection / union
        iob = 0.0 if bg_area == 0 else intersection / bg_area
        iof = 0.0 if fg_area == 0 else intersection / fg_area

        score_list[..., f] = iou, iob, iof

    return np.mean(score_list, axis=1)


def compute_sim_bool(fg_mask, bg_mask):
    bg_len = len(bg_mask)
    fg_len = len(fg_mask)
    score_list = np.empty((3, fg_len), np.float32)

    if fg_mask[0].shape != bg_mask[0].shape:
        bg_mask = [
            cv2.resize(frame.astype(np.uint8), fg_mask[0].shape[::-1]).astype(bool)
            for frame in bg_mask
        ]

    for f, fg_frame in enumerate(fg_mask):
        bg_frame = bg_mask[f % bg_len]

        intersection = (fg_frame & bg_frame).sum()
        union = (fg_frame | bg_frame).sum()

        fg_area = fg_frame.sum()
        bg_area = bg_frame.sum()

        iou = 0.0 if union == 0 else intersection / union
        iob = 0.0 if bg_area == 0 else intersection / bg_area
        iof = 0.0 if fg_area == 0 else intersection / fg_area

        score_list[..., f] = iou, iob, iof

    return np.mean(score_list, axis=1)


def compute_sim_bool_std(fg_mask, bg_mask):
    bg_len = len(bg_mask)
    fg_len = len(fg_mask)
    score_list = np.empty((3, fg_len), np.float32)

    for f, fg_frame in enumerate(fg_mask):
        bg_frame = bg_mask[f % bg_len]

        intersection = (fg_frame & bg_frame).sum()
        union = (fg_frame | bg_frame).sum()

        fg_area = fg_frame.sum()
        bg_area = bg_frame.sum()

        iou = 0.0 if union == 0 else intersection / union
        iob = 0.0 if bg_area == 0 else intersection / bg_area
        iof = 0.0 if fg_area == 0 else intersection / fg_area

        score_list[..., f] = iou, iob, iof

    return np.mean(score_list, axis=1)


def compute_sim_cupy(fg_mask, bg_mask):
    fg_len = len(fg_mask)
    bg_len = len(bg_mask)

    fg_mask = cp.array(fg_mask, dtype=cp.bool_)

    if fg_mask.shape[1:] != bg_mask.shape[1:]:
        bg_mask = cp.array(
            [cv2.resize(frame, fg_mask.shape[1:][::-1]) for frame in bg_mask],
            dtype=cp.bool_,
        )
    else:
        bg_mask = cp.array(bg_mask, dtype=cp.bool_)

    bg_mask_repeated = (
        bg_mask[:fg_len]
        if fg_len <= bg_len
        else cp.tile(bg_mask, (fg_len // bg_len + 1, 1, 1))[:fg_len]
    )

    intersection = (fg_mask & bg_mask_repeated).sum(axis=(1, 2), dtype=cp.float16)
    union = (fg_mask | bg_mask_repeated).sum(axis=(1, 2), dtype=cp.float16)

    bg_area = bg_mask_repeated.sum(axis=(1, 2), dtype=cp.float16)
    fg_area = fg_mask.sum(axis=(1, 2), dtype=cp.float16)

    iou = cp.where(union == 0, 0, intersection / union).mean().item()
    iob = cp.where(bg_area == 0, 0, intersection / bg_area).mean().item()
    iof = cp.where(fg_area == 0, 0, intersection / fg_area).mean().item()

    return iou, iob, iof


def compute_sim_cupy_std(fg_mask, bg_mask):
    fg_len = len(fg_mask)
    bg_len = len(bg_mask)

    fg_mask = cp.array(fg_mask, dtype=cp.bool_)
    bg_mask = cp.array(bg_mask, dtype=cp.bool_)

    bg_mask_repeated = (
        bg_mask[:fg_len]
        if fg_len <= bg_len
        else cp.tile(bg_mask, (fg_len // bg_len + 1, 1, 1))[:fg_len]
    )

    intersection = (fg_mask & bg_mask_repeated).sum(axis=(1, 2), dtype=cp.float16)
    union = (fg_mask | bg_mask_repeated).sum(axis=(1, 2), dtype=cp.float16)

    bg_area = bg_mask_repeated.sum(axis=(1, 2), dtype=cp.float16)
    fg_area = fg_mask.sum(axis=(1, 2), dtype=cp.float16)

    iou = cp.where(union == 0, 0, intersection / union).mean().item()
    iob = cp.where(bg_area == 0, 0, intersection / bg_area).mean().item()
    iof = cp.where(fg_area == 0, 0, intersection / fg_area).mean().item()

    return iou, iob, iof
