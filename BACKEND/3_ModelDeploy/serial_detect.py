import glob
import logging
import os
from typing import List

import cv2
import numpy as np

from utils.TRTDetection import TRTDetection
from utils.pre_process import pre_process_batch_yolov5


def get_frames(video_path: str) -> List[np.ndarray]:
    video = cv2.VideoCapture(video_path)
    export = []
    while True:
        _, img = video.read()
        if img is not None:
            export.append(img)
        else:
            break
    return export


def main():
    # init
    trt_detection = TRTDetection(
        trt_engine_path="models/yolov5n_best_640_batch_1.trt",
        class_list=["box"],
        batch_size=1,
        exp_size=(640, 640)
    )

    # detect
    images_path_list = glob.glob("infer/images/*.jpeg")
    for padded_img, raw_images, area in pre_process_batch_yolov5(
            image_list=[cv2.imread(img_path) for img_path in images_path_list],
            max_batch=1,
            exp_size=(640, 640),
            swap=(2, 0, 1)
    ):
        host_outputs = trt_detection.infer(padded_img)
        postprocess_result = trt_detection.post_process_batch(host_outputs=host_outputs, batch_size=8, conf=0.05, nms=0.45)
        for num in range(len(postprocess_result)):
            postprocess_result[num] = postprocess_result[num].tolist() if postprocess_result[num] is not None else []
        img_path_list = images_path_list[area[0]: area[1] if area[1] <= len(images_path_list) else -1]
        logging.warning(postprocess_result)
        for result, raw_img, img_path in zip(postprocess_result, raw_images, img_path_list):
            visualized_img = trt_detection.visual(result, raw_img, cls_conf=0.05)
            cv2.imwrite(os.path.basename(img_path), visualized_img)


if __name__ == '__main__':
    from utils.logging_utils import log_set
    log_set(logging.DEBUG)
    main()
