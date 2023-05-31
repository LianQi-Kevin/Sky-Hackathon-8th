import os
import glob
from typing import List

from PIL import Image
from kitti_to_PASCAL_VOC import create_PASCAL_VOC, kitti_decode


def png_to_jpg(png_path: str) -> Image:
    return Image.open(png_path).convert("RGB")


def Combined_dataset(root_path: List[str], image_path_tag: str, label_path_tag: str, export_path: str = "output"):
    for dataset_index, path in enumerate(root_path):
        rgb_path = os.path.join(path, image_path_tag, "*.png")
        label_path = os.path.join(path, label_path_tag)
        export_label_path = os.path.join(export_path, "Annotations")
        export_image_path = os.path.join(export_path, "JPEGImages")
        for img_index, img_path in enumerate(glob.glob(rgb_path)):
            os.makedirs(os.path.join(export_image_path), exist_ok=True)
            os.makedirs(os.path.join(export_label_path), exist_ok=True)
            basename, img_suffix = os.path.splitext(os.path.basename(img_path))
            try:
                img = png_to_jpg(img_path) if img_suffix == ".png" else Image.open(img_path)
            except Exception:
                continue
            data = kitti_decode(
                filepath=os.path.join(label_path, f"{basename}.txt"),
                img_size=(img.width, img.height, 3),
                folder="Sky-Hackathon-8th"
            )
            # 使用新的信息覆盖 kitti 读出信息
            new_basename = f"{dataset_index}_{img_index}"
            data["filename"] = f"{new_basename}{'.jpeg' if img_suffix == '.png' else img_suffix}"
            if len(data["object"]) != 0:
                create_PASCAL_VOC(data, True).write(os.path.join(export_label_path, f"{new_basename}.xml"))
                img.save(os.path.join(export_image_path, data["filename"]))
            print(data)


if __name__ == '__main__':
    Combined_dataset(
        root_path=glob.glob("../Dataset/_output*"),
        image_path_tag="Camera/rgb",
        label_path_tag="Camera/object_detection",
        export_path="../Dataset/output"
    )
