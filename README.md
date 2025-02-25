# Action-Aligned Video Pairing

## 1. Preparation

1. Clone this repository.

```shell
git clone --recursive https://github.com/rendicahya/video-action-alignment.git
cd video-action-alignment/
```

2. Create virtual environment.

```shell
python3 -m venv ~/venv/video-action-alignment/
source ~/venv/video-action-alignment/bin/activate
pip install -U pip
```

3. Install MMAction2.

```shell
pip install openmim
mim install mmengine mmcv
pip install -v -e mmaction2/
```

4. Install other required packages.

```shell
pip install click tqdm dynaconf av decord moviepy
```

## 2. Download datasets

Link project's `data/` directory with MMAction2's `data/` directory.

```shell
cd mmaction2/
ln -s ../data/ ./
cd -
```

### a. UCF101

1. Download videos.

```shell
cd mmaction2/tools/data/ucf101/
bash download_videos.sh
```

2. Verify the number of videos. Expected: 13,320.

```shell
find videos/ -type f | wc -l
```

3. Download annotations.

```shell
bash download_annotations.sh
cd -
```

4. Generate splits.

```shell
python3 mmaction2/tools/data/build_file_list.py ucf101 data/ucf101/videos/ --format videos --shuffle --seed 0
```

5. Verify structure.

```shell
intercutmix/data/ucf101/
├── annotations/
│   ├── classInd.txt
│   ├── testlist01.txt
│   ├── testlist02.txt
│   ├── testlist03.txt
│   ├── trainlist01.txt
│   ├── trainlist02.txt
│   └── trainlist03.txt
├── videos/
│   ├── ApplyEyeMakeup/
│   │   ├── v_ApplyEyeMakeup_g01_c01.avi
│   │   ├── v_ApplyEyeMakeup_g01_c02.avi
│   │   ├── v_ApplyEyeMakeup_g01_c03.avi
│   │   └── ...
│   └── ...
├── ucf101_train_split_1_videos.txt
├── ucf101_train_split_2_videos.txt
├── ucf101_train_split_3_videos.txt
├── ucf101_val_split_1_videos.txt
├── ucf101_val_split_2_videos.txt
└── ucf101_val_split_3_videos.txt
```

### b. HMDB51

1. Download videos.

```shell
cd mmaction2/tools/data/hmdb51/
bash download_videos.sh
```

2. Verify the number of videos. Expected: 6,766.

```shell
find videos/ -type f | wc -l
```

3. Download annotations.

```shell
bash download_annotations.sh
cd -
```

4. Generate splits.

```shell
python3 mmaction2/tools/data/build_file_list.py hmdb51 data/hmdb51/videos/ --format videos --shuffle --seed 0
```

5. Verify structure.

```shell
intercutmix/data/hmdb51/
├── annotations/
│   ├── brush_hair_test_split1.txt
│   ├── brush_hair_test_split2.txt
│   ├── brush_hair_test_split3.txt
│   └── ...
├── frames/
│   ├── brush_hair/
│   │   ├── April_09_brush_hair_u_nm_np1_ba_goo_0/
│   │   │   ├── img_00001.png
│   │   │   ├── img_00002.png
│   │   │   ├── img_00003.png
│   │   │   └── ...
│   │   └── ...
│   └── ...
├─── videos/
│   ├── brush_hair/
│   │   ├── April_09_brush_hair_u_nm_np1_ba_goo_0.avi
│   │   ├── April_09_brush_hair_u_nm_np1_ba_goo_1.avi
│   │   ├── April_09_brush_hair_u_nm_np1_ba_goo_2.avi
│   │   └── ...
│   └── ...
├── hmdb51_train_split_1_videos.txt
├── hmdb51_train_split_2_videos.txt
├── hmdb51_train_split_3_videos.txt
├── hmdb51_val_split_1_videos.txt
├── hmdb51_val_split_2_videos.txt
└── hmdb51_val_split_3_videos.txt
```

### c. Kinetics100

1. Download videos.

```shell
wget https://some-google-drive-url/file
```

2. Verify the number of videos. Expected: 9,999.

```shell
find videos/ -type f | wc -l
```

3. Generate annotations.

```shell
python3 tools/data/make_classind.py
```

## 3. Run Inference

1. Download checkpoints.

```shell
pip install gdown
mkdir checkpoints/
gdown -O checkpoints/ <download-key>
```

| **Dataset** | **Top-1** | **Top-5** | **Config Path**          | **Checkpoint Path**                  | **Download Key**                  |
|-------------|-----------|-----------|--------------------------|--------------------------------------|-----------------------------------|
| UCF101      |   84.93%  |   96.27%  | mmaction2/configs/action-alignment/demo/c3d_sports1m-pretrained_8xb64-16x1x1-100e_ucf101-rgb-iou-s-dilation.py | checkpoints/ucf101-iou-s-td.pth      | 1c6bC4U6XSNHWX6eedR4QyCF5X46tfv8c |
| HMDB51      |   56.21%  |   85.42%  | mmaction2/configsaction-alignment/demo/c3d_sports1m-pretrained_8xb64-16x1x1-100e_hmdb51-rgb-bao-s-dilation.py | checkpoints/hmdb51-bao-s-td.pth      | 1_-e_Ww9oR5zu8xNmmTmBlfIhN2SbvMDd |
| Kinetics100 |   60.52%  |   84.92%  | mmaction2/configs/action-alignment/demo/c3d_sports1m-pretrained_8xb64-16x1x1-100e_kinetics100-rgb-bao-s-dilation.py | checkpoints/kinetics100-bao-s-td.pth | 10uO-gwM3v2x1enxWUAEp_sMfGqP5TUrr |

2. Run inference.

```shell
python3 mmaction/tools/test.py <config-path> <checkpoint-path>
```