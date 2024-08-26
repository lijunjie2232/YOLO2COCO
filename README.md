# YOLO2COCO
YOLOv8 dataset to COCO json format

---
### requirements

```(shell)
pip install tqdm opencv-python numpy
```



## Usage

[yolo2coco.py](https://github.com/lijunjie2232/YOLO2COCO/blob/master/yolo2coco.py) is for yolov8 dataset and also could use for yolov5 dataset which uses txt to store images path but is not recommended.

[yolov5_2_coco.py](https://github.com/lijunjie2232/YOLO2COCO/blob/master/yolov5_2_coco.py) is for yolov5 (old) dataset

1. on shell
```shell
python yolo2coco.py -c $YAML_FILE
```

2.  use python

```python
import subprocess

subprocess.run(["python", "yolo2coco.py", "-c", "data.yaml"], check=True)
```

3. old script usage for yolov5 dataset

```shell
python yolov5_2_coco_new.py --data_dir $DATA_DIR --mode_list train,test,val 
```



 
