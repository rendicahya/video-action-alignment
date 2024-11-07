from pathlib import Path

import numpy as np
from lib_sim import *
from tqdm import tqdm


def pack_temporal(mask_array, length=5):
    T, H, W = mask_array.shape
    full_groups = T // length
    remainder = T % length

    if remainder != 0:
        padding = (length - remainder, H, W)
        padded = np.pad(
            mask_array,
            ((0, padding[0]), (0, 0), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        T = padded.shape[0]
    else:
        padded = mask_array

    return np.max(padded.reshape(T // length, length, H, W), axis=1)


base_path = "data/hmdb51/yolov8-coco/0.25/detect/mask/"
mask1_path = base_path + "brush_hair/April_09_brush_hair_u_nm_np1_ba_goo_0.npz"
# mask2_path = (
#     base_path
#     + "shake_hands/40_Trofeo_S_A_R__Princesa_Sofia_Price_Giving_Ceremony_shake_hands_f_cm_np2_le_bad_2.npz"
# )
mask2_path = (
    base_path + "brush_hair/atempting_to_brush_my_hair_brush_hair_u_nm_np2_le_goo_0.npz"
)

mask1_uint8 = np.load(mask1_path)["arr_0"]
mask2_uint8 = np.load(mask2_path)["arr_0"]

mask1_bool = mask1_uint8.astype(bool)
mask2_bool = mask2_uint8.astype(bool)

mask1_packed = pack_temporal(mask1_bool, length=5)
mask2_packed = pack_temporal(mask2_bool, length=5)

iou_uint8 = compute_sim_uint8(mask1_uint8, mask2_uint8)
iou_uint8_std = compute_sim_uint8_std(mask1_uint8, mask2_uint8)
iou_bool = compute_sim_bool(mask1_bool, mask2_bool)
iou_bool_std = compute_sim_bool_std(mask1_bool, mask2_bool)
iou_cupy = compute_sim_cupy(mask1_bool, mask2_bool)
iou_cupy_std = compute_sim_cupy_std(mask1_uint8, mask2_uint8)
# iou_cupy_packed = compute_sim_cupy(mask1_packed, mask2_packed)

print("mask1.shape:", mask1_uint8.shape)
print("mask2.shape:", mask2_uint8.shape)

print("IoU uint8:", iou_uint8)
print("IoU uint8 std:", iou_uint8_std)
print("IoU bool:", iou_bool)
print("IoU bool std:", iou_bool_std)
print("IoU CuPy:", iou_cupy)
print("IoU CuPy std:", iou_cupy_std)
# print("IoU v5 packed:", iou_cupy_packed)

# exit()

loop = 1000

for i in tqdm(range(loop)):
    compute_sim_uint8(mask1_uint8, mask2_uint8)

for i in tqdm(range(loop)):
    compute_sim_bool(mask1_bool, mask2_bool)

for i in tqdm(range(loop)):
    compute_sim_cupy(mask1_uint8, mask2_uint8)
