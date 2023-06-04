"""
This file used to format PASCAL VOC label to mAP ground-truth label.
"""
import os
import glob
from typing import List
import xml.etree.ElementTree as ET


def detect_PASCAL(file_path: str) -> List[List[str]]:
    """
    :param file_path: PASCAL VOC xml filepath
    :return: [[name, xmin, ymin, max, ymax, difficult], ...]
    """
    assert os.path.exists(file_path), f"{file_path} not found."
    root = ET.parse(file_path).getroot()
    if root.find("object").find("name") is not None:
        return [[
            obj.find("name").text,
            obj.find("bndbox").find("xmin").text,
            obj.find("bndbox").find("ymin").text,
            obj.find("bndbox").find("xmax").text,
            obj.find("bndbox").find("ymax").text,
            obj.find("bndbox").find("difficult") if obj.find("bndbox").find("difficult") is not None else "0"
        ] for obj in root.findall("object")]


def write_ground_truth(bndboxs: List[List[str]], export_path: str):
    if bndboxs is not None:
        with open(export_path, "w") as f:
            for bndbox in bndboxs:
                f.write(f"{bndbox[0]} {bndbox[1]} {bndbox[2]} {bndbox[3]} {bndbox[4]} {'' if bndbox[5] == '0' else 'difficult'}")


def PASCAL_to_mAP(PASCAL_list: List[str], export_path: str = "output", skip_check: bool = False):
    """
    :param PASCAL_list: filepath list, PASCAL VOC label files
    :param export_path: mAP ground-truth label export path
    :param skip_check: skip checking export_path Whether is empty
    """
    if os.path.exists(export_path):
        if skip_check:
            import shutil
            shutil.rmtree(export_path)
        assert len(os.listdir(export_path)) != 0, f"{export_path} not empty, Check it"
    os.makedirs(export_path, exist_ok=True)
    for filepath in PASCAL_list:
        write_ground_truth(
            bndboxs=detect_PASCAL(filepath),
            export_path=os.path.join(export_path, f"{os.path.splitext(os.path.basename(filepath))[0]}.txt")
        )


if __name__ == '__main__':
    PASCAL_to_mAP(
        PASCAL_list=glob.glob("../infer/images/Annotations/*.xml"),
        export_path="../mAP/input/ground-truth/",
        skip_check=True
    )
