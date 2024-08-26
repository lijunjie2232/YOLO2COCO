# YOLO2COCO
YOLOv8 dataset to COCO json format

---
### requirements

```(shell)
pip install tqdm opencv-python numpy
```

## YOLO v8 Dataset Structure

```shell
coco/
│
├── images/
│   ├── train/
│   └── val/
│
├── labels/
│   ├── train/
│   └── val/
│
└── data.yaml
```

```shell
#data.yaml of coco dataset in yolo v8 fomat

path: /home/user/dataset/coco/
train:
- images/train
val:
- images/val
names:
  0: person
  1: bicycle
  2: car
  ......
  77: teddy bear
  78: hair drier
  79: toothbrush
```

## Usage

[yolo2coco.py](https://github.com/lijunjie2232/YOLO2COCO/blob/master/yolo2coco.py) is for yolov8 dataset and also partially compatible with yolov5 dataset which uses txt to store images path but is not recommended.

[yolov5_2_coco.py](https://github.com/lijunjie2232/YOLO2COCO/blob/master/yolov5_2_coco.py) is for yolov5 (old) dataset, and is modified from https://github.com/RapidAI/YOLO2COCO, partially compatible with yaml file

1. on shell
```shell
python yolo2coco.py -c $YAML_FILE

# for more help
python yolo2coco.py -h
```

2.  use python

```python
import subprocess

subprocess.run(
    [
        "python",
        "yolo2coco.py",
        "-c",
        "coco/data.yaml",
        "-p",
        "16",
        "-o",
        "coco_format/",
        "--copy"
    ],
    check=True
)
```

3. old script usage for yolov5 dataset

```shell
python yolov5_2_coco_new.py --data_dir $DATA_DIR --mode_list train,test,val 
```



 
