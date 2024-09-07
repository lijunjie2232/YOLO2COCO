# -*- encoding: utf-8 -*-
# @File: yolo2coco.py
# @Author: lijunjie2232
# @Contact: git@lijunjie2232

import argparse
import json
import shutil
import time
import warnings
from pathlib import Path
import os
import numpy as np

import cv2
from tqdm import tqdm
import yaml
from PIL import Image
from multiprocessing.pool import ThreadPool
from itertools import repeat

import logging

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)
LG = logging.getLogger("YOLO2COCO")


def check_yolo_yaml(file):
    if not isinstance(file, Path):
        file = Path(file)
    assert file.is_file(), f"yaml file {file} not exists or is not a file"
    data = None
    with open(file, "r", encoding="utf8") as f:
        data = yaml.safe_load(f)
    assert (
        "path" in data
    ), f"""there is no key "path" in yaml file which means path of dataset"""
    if not Path(data["path"]).is_absolute():
        data["path"] = Path(file).parent / data["path"]
    assert "names" in data, f"no classes information in yaml file"
    if isinstance(data["names"], list):
        data["names"] = dict(zip(data["names"], range(len(data["names"]))))
    assert isinstance(
        data["names"], dict
    ), "classes format not support, please use list in yaml file to store classes information"
    return data


def read_txt(txt_path):
    with open(str(txt_path), "r", encoding="utf-8") as f:
        data = f.read().strip("\n").split("\n")
    return data


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    y = []
    y.append((x[0] + x[2]) / 2)  # x center
    y.append((x[1] + x[3]) / 2)  # y center
    y.append(x[2] - x[0])  # width
    y.append(x[3] - x[1])  # height
    return y


def verify_exists(file_path: str | Path, exists: str = "exists", raise_if_ne=True):
    """_summary_

    Args:
        file_path (str | Path): file path to verify
        exists (str, optional): verify method. in ["exists", "is_file", "is_dir", ...] Defaults to "exists".
        raise_if_ne (bool, optional): while file invalid and this param if True, raise FileNotFoundError. Defaults to True.

    Returns:
        bool: if raise_if_ne is False, return bool value
    """
    file_path = Path(file_path)
    assert hasattr(file_path, exists)
    if raise_if_ne:
        assert getattr(file_path, exists)(), FileNotFoundError(
            f"The {file_path} not {exists}!!!"
        )
    else:
        return getattr(file_path, exists)()
    return True


def write_json(json_path: str | Path, content: dict):
    """_summary_

    Args:
        json_path (str | Path): output json path
        content (dict): json data
    """
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(content, f, ensure_ascii=False)


def img_checker(img: str | Path):
    """_summary_

    Args:
        img (str | Path): img path

    Returns:
        object: return None if invalid image
    """
    try:
        Image.open(img).load()
        return img
    except:
        return None


class YOLO2COCO:

    # PIC_SUFFIX = [".png", ".jpg", ".jpeg", ".raw", ".gif", ".bmp"]

    def __init__(
        self, yaml_file: str, output: str = "", img_cp: bool = False, mt: int = 4
    ):
        """_summary_

        Args:
            yaml_file (str): yaml file of yolo dataset
            output (str, optional): output directory. Defaults to "".
            img_cp (bool, optional): is copy image file to output directory. Defaults to False.
            mt (int, optional): number of multiprocess. Defaults to 4.
        """
        self.METAINFO = check_yolo_yaml(yaml_file)
        self._root_ = Path(self.METAINFO["path"])
        self.modes = []
        self.single_dir = True
        for t in ["train", "val", "test"]:
            if t in self.METAINFO:
                if not isinstance(self.METAINFO[t], list):
                    self.METAINFO[t] = [self.METAINFO[t]]
                for i in self.METAINFO[t]:
                    verify_exists(self._root_ / i)
                self.single_dir &= len(self.METAINFO[t]) == 1
                self.modes.append(t)

        if self.single_dir:
            if output:
                self.output_dir = Path(output)
                LG.info(f"set output path: {self.output_dir}")
            else:
                self.output_dir = self._root_
                LG.warning(
                    f"not set output path, use original dataset directory: {self.output_dir} as output path"
                )
        else:
            if output:
                self.output_dir = Path(output)
                if self.output_dir.is_dir() and self.output_dir.samefile(self._root_):
                    self.output_dir = (
                        self.output_dir.parent / f"{self.output_dir.stem}_coco_format"
                    )
                    LG.warning(
                        f"not single directory of data but output directory is the same to dataset directory, force use directory: {self.output_dir} as output path"
                    )
                else:
                    LG.info(f"set output path: {self.output_dir}")
            else:
                self.output_dir = self._root_.parent / f"{self._root_.stem}_coco_format"
                LG.warning(
                    f"not set output path but not single directory of data, use directory: {self.output_dir} as output path"
                )

        self.json_dir = self.output_dir / "annotations"
        self.img_cp = img_cp
        os.makedirs(self.json_dir, exist_ok=True)
        if self.output_dir.samefile(self._root_):
            LG.warning(
                "not copy image for output directory is the same to dataset directory"
            )
            self.img_cp = False
        else:
            if self.single_dir:
                LG.info(
                    "img will%scopy into output derectory"
                    % (" " if self.img_cp else " not ")
                )
            else:
                self.img_cp = True
                LG.warning(
                    "img will be force to copy into output derectory due to not single directory of data"
                )

        self.mt = 1 if mt <= 1 else mt
        LG.info(f"use {self.mt} thread{'s'if self.mt > 1 else ''}")
        self._init_json()

    def get_img_list(self, mode):
        """_summary_
        get image list from path in yaml file, support directorys, support txt too but not recommend

        Args:
            mode (str): type of set, in ['train', 'val', 'test']

        Returns:
            list: img list of all {mode} set
        """
        total_img = []

        for idx, path in enumerate(self.METAINFO[mode]):
            use_absolute = False
            path = Path(path)
            if not path.is_absolute():
                path = (self._root_ / path).absolute()
            img_list = None
            if path.__str__().endswith(".txt"):  # img path stores in a file
                if not self.img_cp:
                    LG.warning(
                        "it is not recommended to use txt file if not copy image, or just use script for yolov5"
                    )
                use_absolute = True
                with open(path, "r", encoding="utf8") as f:
                    img_list = f.read().strip("\n").split("\n")
                img_list = [
                    Path(i) if Path(i).is_absolute() else self._root_ / i
                    for i in img_list
                ]
            elif path.is_dir():
                img_list = [path / i for i in os.listdir(self._root_ / path)]
            else:
                raise (Exception(f"format of path: {path} not support"))
            img_list_valid = []
            with ThreadPool(self.mt) as pool:
                results = pool.imap(
                    func=img_checker,
                    iterable=img_list,
                )
                for i in tqdm(
                    results,
                    total=len(img_list),
                    desc=f"{mode} image " + f"{idx }" if idx > 0 else "" + "check",
                ):
                    if not i:
                        continue
                    else:
                        img_list_valid.append(
                            i.relative_to((self._root_.absolute()).__str__())
                            if not use_absolute
                            else i
                        )
            total_img.extend(img_list_valid)
        return total_img

    def __call__(self):

        for mode in self.modes:
            # Read the image txt.
            img_list = self.get_img_list(mode)

            # Create the directory of saving the new image.
            if self.img_cp:
                save_img_dir = self.output_dir / f"{mode}{self.cur_year}"
                save_img_dir.mkdir(exist_ok=True)
                LG.info(f"image of new {mode} set will be save into {save_img_dir}")
            else:
                LG.info(f"{mode} image will not copy")
                save_img_dir = None

            # Generate json file.
            save_json_path = self.json_dir / f"instances_{mode}{self.cur_year}.json"

            json_data = self.convert(img_list, mode, save_img_dir=save_img_dir)

            write_json(save_json_path, json_data)
            LG.info(f"annotation file has witen to {save_json_path}")
        LG.info(f"Successfully convert, detail in {self.output_dir}")

    def _init_json(self):
        classes = self.METAINFO["names"]

        self.type = "instances"
        self.annotation_id = 1

        self.cur_year = time.strftime("%Y", time.localtime(time.time()))
        self.info = {
            "year": int(self.cur_year),
            "version": "1.0",
            "description": "For object detection",
            "date_created": time.strftime("%Y/%m/%d", time.localtime()),
        }

        self.licenses = [
            {
                "id": 1,
                "name": "Apache License v2.0",
                "url": "https://github.com/lijunjie2232/YOLO2COCO/LICENSE",
            }
        ]

    def append_bg_img(self, img_list):
        bg_dir = self._root_ / "background_images"
        if bg_dir.exists():
            bg_img_list = list(bg_dir.iterdir())
            for bg_img_path in bg_img_list:
                img_list.append(str(bg_img_path))
        return img_list

    @property
    def categories(self):
        """_summary_

        Returns:
            dict: return categories of dataset
        """
        return [
            {
                "supercategory": v,
                "id": k + 1,
                "name": v,
            }
            for k, v in self.METAINFO["names"].items()
        ]

    def data_handler(self, args):
        img_id, img_path, save_img_dir = args

        use_absolute = img_path.is_absolute()

        image_dict = self.get_image_info(
            img_path, img_id, save_img_dir=save_img_dir, use_absolute=use_absolute
        )

        label_path = (
            Path(img_path.parent.__str__().replace("images", "labels"))
            / f"{Path(img_path).stem}.txt"
        )
        annotations = self.get_annotation(
            label_path,
            img_id,
            image_dict["height"],
            image_dict["width"],
            use_absolute=use_absolute,
        )
        return image_dict, annotations

    def convert(self, img_list: list, mode: str, save_img_dir=None):
        """_summary_

        Args:
            img_list (list): img list of {mode}
            mode (str): type of set
            save_img_dir (_type_, optional): cp image into output save_img_dir from origin dataset. Defaults to None.

        Returns:
            dict: json annotation dict of {mode}
        """
        images, annotations = [], []
        anno_nums = 0

        with ThreadPool(self.mt) as pool:
            results = pool.imap(
                func=self.data_handler,
                iterable=zip(
                    range(1, len(img_list) + 1),
                    img_list,
                    repeat(save_img_dir),
                ),
            )
            cvt_process = tqdm(
                results,
                total=len(img_list),
                desc=f"{mode} image and annotation process",
            )
            for image_dict, anns in cvt_process:
                images.append(image_dict)
                for ann in anns:
                    ann["id"] = self.annotation_id
                    self.annotation_id += 1
                    annotations.append(ann)
                anno_nums += len(anns)

                cvt_process.set_postfix(
                    {
                        "processed": image_dict["file_name"],
                        "anno numbers": anno_nums,
                    }
                )
        # for img_id, img_path, save_img_dir in zip(
        #     range(1, len(cvt_process) + 1), cvt_process, repeat(save_img_dir)
        # ):

        #     image_dict, anns = self.data_handler(img_id, img_path, save_img_dir)
        #     images.append(image_dict)
        #     for ann in anns:
        #         ann["id"] = self.annotation_id
        #         self.annotation_id += 1
        #         annotations.append(ann)
        #     anno_nums += len(anns)

        #     cvt_process.set_postfix(
        #         {
        #             "processed": Path(img_path).name,
        #             "anno numbers": anno_nums,
        #         }
        #     )

        json_data = {
            "info": self.info,
            "images": images,
            "licenses": self.licenses,
            "type": self.type,
            "annotations": annotations,
            "categories": self.categories,
        }
        return json_data

    def get_image_info(
        self, img_path: str | Path, img_id: int, save_img_dir=None, use_absolute=False
    ):
        """_summary_

        Args:
            img_path (str | Path): path of current image
            img_id (int): id of current image
            save_img_dir (_type_, optional): .... Defaults to None.
            use_absolute (bool, optional): .... Defaults to False.

        Returns:
            dict: image dict of image in coco image format
        """
        new_img_name = img_path
        if not use_absolute:
            img_path = self._root_ / Path(img_path)
            new_img_name = new_img_name.name

        img_src = cv2.imread(str(img_path))

        if save_img_dir:
            new_img_name = f"{img_id:012d}.jpg"
            save_img_path = save_img_dir / new_img_name
            if img_path.suffix.lower() == ".jpg":
                shutil.copyfile(img_path, save_img_path)
            else:
                cv2.imwrite(str(save_img_path), img_src)

        height, width = img_src.shape[:2]
        image_info = {
            "date_captured": self.cur_year,
            "file_name": new_img_name,
            "id": img_id,
            "height": height,
            "width": width,
        }
        return image_info

    def get_annotation(
        self,
        label_path: Path,
        img_id: int,
        height: int,
        width: float,
        use_absolute=False,
    ):
        """_summary_

        Args:
            label_path (Path): path of current label
            img_id (int): id of current image
            height (int): height of current image
            width (int): width of current image
            use_absolute (bool, optional): ... Defaults to False.
        """

        def get_box_info(vertex_info: list, height: int, width: int):
            """_summary_

            Args:
                vertex_info (list): one line of yolo format annotation
            height (int): height of current image
            width (int): width of current image

            Returns:
                (segmentation:list(list), bbox:list, area:int) of a annotation
            """
            segment = False
            vertex_info = [float(i) for i in vertex_info]
            if len(vertex_info) > 4:
                x, y = vertex_info[::2], vertex_info[1::2]
                x_min, y_min, x_max, y_max = min(x), min(y), max(x), max(y)
                cx, cy, w, h = xyxy2xywh([x_min, y_min, x_max, y_max])
                segment = [
                    (np.array(vertex_info).reshape(-1, 2) * np.array([w, h]))
                    .flatten()
                    .tolist()
                ]
            elif len(vertex_info) == 4:
                box = np.array(vertex_info, dtype=np.float64)
                box[:2] -= box[2:] / 2
                cx, cy, w, h = box
            else:
                return [], [], 0

            cx = cx * width
            cy = cy * height
            box_w = w * width
            box_h = h * height

            # left top
            x0 = max(cx - box_w / 2, 0)
            y0 = max(cy - box_h / 2, 0)

            # right bottom
            x1 = min(x0 + box_w, width)
            y1 = min(y0 + box_h, height)

            segmentation = segment if segment else [[x0, y0, x1, y0, x1, y1, x0, y1]]
            bbox = [x0, y0, box_w, box_h]
            area = box_w * box_h
            return segmentation, bbox, area

        if not use_absolute:
            label_path = self._root_ / Path(label_path)
        if not label_path.is_file():
            return []

        annotation = []
        label_list = read_txt(str(label_path))
        for i, one_line in enumerate(label_list):
            label_info = one_line.split(" ")
            if len(label_info) < 5:
                warnings.warn(f"The {i+1} line of the {label_path} has been corrupted.")
                continue

            category_id, vertex_info = label_info[0], label_info[1:]
            segmentation, bbox, area = get_box_info(vertex_info, height, width)
            if bbox:
                annotation.append(
                    {
                        "segmentation": segmentation,
                        "area": area,
                        "iscrowd": 0,
                        "image_id": img_id,
                        "bbox": bbox,
                        "category_id": int(category_id) + 1,
                        "id": 0,
                    }
                )
        return annotation


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Datasets converter from YOLOV8 to COCO")
    parser.add_argument(
        "-c", "--conf", type=str, default="data.yaml", help="yaml file of dataset"
    )
    parser.add_argument("-o", "--output", type=str, default="", help="output directory")
    parser.add_argument(
        "-p", "--parallel", type=int, default=4, help="numbers of multiple thread"
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        default=False,
        help="copy image into new directory, auto disabled while using origin directory as output",
    )
    args = parser.parse_args()

    converter = YOLO2COCO(
        yaml_file=args.conf, output=args.output, img_cp=args.copy, mt=args.parallel
    )
    converter()
