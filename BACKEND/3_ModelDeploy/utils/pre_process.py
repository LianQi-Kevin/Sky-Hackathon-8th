from math import ceil
import time
import numpy as np
import cv2
import logging
from typing import Tuple, List, Union


def padding_img(image_raw: np.ndarray, exp_size: Union[Tuple[int, int], int] = (640, 640),
                swap: Tuple[int, int, int] = (2, 0, 1), normalization: bool = True) -> np.ndarray:
    """Resize raw image and Padding to exp_size (Left fill)
    Args:
        image_raw: image raw, np.ndarray, cv2.imread(img_path)
        exp_size: exp size, padding target shape
        swap: channel swap mode, default CHW
        normalization: whether to normalize
    Return:
        padded_img
    """
    h, w, _ = image_raw.shape
    # exp_size
    exp_height = exp_size[0] if isinstance(exp_size, tuple) else exp_size
    exp_width = exp_size[1] if isinstance(exp_size, tuple) else exp_size
    # 创建一个(640, 640, 3)的数组
    padded_img = np.ones((exp_height, exp_width, 3), dtype=np.uint8) * 114
    # 计算图片实际大小和预期大小插值
    r = min(exp_height / h, exp_width / image_raw.shape[1])
    # resize图片
    resized_img = cv2.resize(cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB), (int(w * r), int(h * r)),
                             interpolation=cv2.INTER_LINEAR).astype(np.uint8)
    # 填充resized图片到padded_img
    padded_img[: int(h * r), : int(w * r)] = resized_img
    # normalization
    if normalization:
        padded_img = padded_img.astype(np.float32)
        padded_img /= 255.0
    # HWC to CHW
    return np.transpose(padded_img, swap)


def pre_process_batch_yolov5(image_list: list, max_batch: int = 1,
                             exp_size: Union[Tuple[int, int], int] = (640, 640)) -> List[list]:
    """from raw images list, get batch preprocessed data
    Args:
        image_list: image row data list.
        max_batch: model batch-size
        exp_size: exp size, model shape
    Return:
        [[pre-processed images], [raw images]]
    """
    exp_height = exp_size[0] if isinstance(exp_size, tuple) else exp_size
    exp_width = exp_size[1] if isinstance(exp_size, tuple) else exp_size
    for num in range(ceil(len(image_list) / max_batch)):
        ST_time = time.time()
        output = [np.ones((3, exp_height, exp_width), dtype=np.float32) * 114 for _ in range(max_batch)]
        raw_batch = image_list[num * max_batch: (num * max_batch) + max_batch]
        for index, image_raw in enumerate(raw_batch):
            image, _ = padding_img(image_raw, exp_size=exp_size, swap=(2, 0, 1), normalization=True)
            # CHW to NCHW format
            output[index] = np.expand_dims(image, axis=0)
        logging.debug(f"preprocess batch: {time.time() - ST_time}s")
        # to C order and return
        yield [np.ascontiguousarray(np.array(output), dtype=np.float32), raw_batch]


def pre_process_batch_yolox(image_list, max_batch=1, img_size=(640, 640), swap=(2, 0, 1), un_read=False):
    """
    return: [[preprocessed images], [source images]]
    """
    exp_width, exp_height = img_size[0], img_size[1]
    for num in range(ceil(len(image_list) / max_batch)):
        ST_time = time.time()
        output = [np.ones((3, exp_height, exp_width), dtype=np.float32) * 114 for _ in range(max_batch)]
        for index, img in enumerate(image_list[num * max_batch: (num * max_batch) + max_batch]):
            if un_read:
                img = cv2.imread(img, cv2.IMREAD_COLOR)
            # 创建一个(640, 640, 3)的数组
            padded_img = np.ones((exp_height, exp_width, 3), dtype=np.uint8) * 114
            # 计算图片实际大小和预期大小插值
            r = min(exp_height / img.shape[0], exp_width / img.shape[1])
            # resize图片
            resized_img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)),
                                     interpolation=cv2.INTER_LINEAR).astype(np.uint8)
            # 填充resized图片到padded_img
            padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
            # 转换成(3, 640, 640的数组)
            padded_img = padded_img.transpose(swap)
            output[index] = padded_img
        output = np.array(output)
        # 转换数组位置到内存连续， 加速调用
        logging.debug("preprocess batch: {}s".format(time.time() - ST_time))
        yield [np.ascontiguousarray(output, dtype=np.float32), image_list[num * max_batch: (num * max_batch) + max_batch]]


def preprocess_yolov7(img: np.array, exp_height=640, exp_width=640, swap=(2, 0, 1)) -> np.array:
    # 创建一个(640, 640, 3)的数组
    padded_img = np.full((exp_height, exp_width, 3), fill_value=128, dtype=np.uint8) * 114
    # 计算图片实际大小和预期大小插值
    r = min(exp_height / img.shape[0], exp_width / img.shape[1])
    # resize图片
    resized_img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)),
                             interpolation=cv2.INTER_LINEAR).astype(np.uint8)
    # 填充resized图片到padded_img
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    # 转换数据类型
    padded_img = padded_img.astype(np.float32)
    # 归一化
    padded_img /= 255.0
    # 转换成(3, 640, 640的数组) 即CHW格式
    return padded_img.transpose(swap)


def preprocess_yolov7_batch_images(image_list: list, max_batch=1, exp_size=(640, 640), swap=(2, 0, 1), un_read=False) -> list:
    """
    :param image_list: a list, [img-path or np.array]
    :param max_batch: model input batch-size
    :param exp_size: model input img size
    :param swap: Img channel swap, to CHW
    :param un_read:
    :return yield [[preprocessed images], [source images]]
    """
    exp_width, exp_height = exp_size[0], exp_size[1]
    group_num = ceil(len(image_list) / max_batch)
    for num in range(group_num):
        ST_time = time.time()
        output = [np.ones((3, exp_height, exp_width), dtype=np.float32) * 114 for _ in range(max_batch)]
        for index, img in enumerate(image_list[num * max_batch: (num * max_batch) + max_batch]):
            if un_read:
                img = cv2.imread(img, cv2.IMREAD_COLOR)
            output[index] = preprocess_yolov7(img, exp_height, exp_width, swap)
        # CHW 到 NCHW 格式
        output = [np.expand_dims(out, axis=0) for out in output]
        # list to array
        output = np.array(output, dtype=np.float32)
        # 转换数组位置到内存连续, 加速调用
        logging.debug("preprocess batch: {}s".format(time.time() - ST_time))
        yield [np.ascontiguousarray(output, dtype=np.float32), image_list[num * max_batch: (num * max_batch) + max_batch]]
