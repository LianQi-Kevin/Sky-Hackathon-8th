import os
import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
from torch import cat as torch_cat
from torch import max as torch_max
from torch import Tensor
from torchvision.ops import batched_nms
from loguru import logger
import logging
import threading
from typing import List, Tuple

# Global var
_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)


@logger.catch
class TRTDetection:
    def __init__(self, engine_file_path: str, cls_list: List[str], batch_size: int = 1, exp_size: Tuple[int, int] = (640, 640)):
        # threading context
        # 创建一个上下文堆栈
        # https://documen.tician.de/pycuda/driver.html#pycuda.driver.Context
        self.cfx = cuda.Device(0).make_context()

        # Load engine
        self.engine_file_path = engine_file_path
        self.engine = self._load_engine()
        logging.info("Successful load {}".format(os.path.basename(self.engine_file_path)))

        # basic var
        self.cls_list = cls_list
        self.num_classes = len(self.cls_list)
        self.batch_size = batch_size
        self.exp_height = exp_size[1]
        self.exp_width = exp_size[0]

        # detect var
        self.host_inputs = []
        self.cuda_inputs = []
        self.host_outputs = []
        self.cuda_outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        self.context = self._create_context()

    def detect(self, img_resized):
        threading.Thread.__init__(self)
        # 推到栈顶
        # https://documen.tician.de/pycuda/driver.html#pycuda.driver.Context.push
        self.cfx.push()
        np.copyto(self.host_inputs[0], img_resized.ravel())
        # 将处理好的图片从CPU内存中复制到GPU显存
        cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)
        # 开始执行推理任务
        self.context.execute_async(
            batch_size=self.batch_size,
            bindings=self.bindings,
            stream_handle=self.stream.handle)
        # 将推理结果输出从GPU显存复制到CPU内存
        cuda.memcpy_dtoh_async(self.host_outputs[0], self.cuda_outputs[0], self.stream)
        self.stream.synchronize()
        return self.host_outputs[0]

    def visual(self, output, img, cls_conf=0.35):
        """
        visual single img, put text and conf
        """
        if len(output) == 0:
            return img
        else:
            bandboxes, scores, classes = self.remapping_result(output, img)
            for i in range(len(bandboxes)):
                box = bandboxes[i]
                cls_id = int(classes[i])
                score = scores[i]
                if score < cls_conf:
                    continue
                x0 = int(box[0])
                y0 = int(box[1])
                x1 = int(box[2])
                y1 = int(box[3])

                color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
                text = '{}:{:.1f}%'.format(self.cls_list[cls_id], score * 100)
                txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
                font = cv2.FONT_HERSHEY_SIMPLEX

                text_size = cv2.getTextSize(text, font, 0.4, 1)[0]
                cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
                txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
                cv2.rectangle(img, (x0, y0 + 1), (x0 + text_size[0] + 1, y0 + int(1.5 * text_size[1])), txt_bk_color, -1)
                cv2.putText(img, text, (x0, y0 + text_size[1]), font, 0.4, txt_color, thickness=1)
            return img

    # 重映射推理结果
    def remapping_result(self, output, img):
        """
        remapping result for single img
        :param output: a list, postprocess_outputs[index]
        :param img: a np array, cv2 img
        """
        output = np.array(output, dtype=object)
        ratio = min(self.exp_height / img.shape[0], self.exp_width / img.shape[1])
        bandboxes = output[:, 0:4]
        # preprocessing: resize
        bandboxes /= ratio
        scores = output[:, 4]
        classes = output[:, 5]
        return bandboxes, scores, classes

    # 反序列化引擎
    def _load_engine(self):
        assert os.path.exists(self.engine_file_path), "{} not found".format(self.engine_file_path)
        logging.info("Load engine from {}".format(self.engine_file_path))
        with open(self.engine_file_path, "rb") as f, trt.Runtime(trt.Logger()) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    # 通过加载的引擎，生成可执行的上下文
    def _create_context(self):
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            # 注意：这里的host_mem需要时用pagelocked memory，以免内存被释放
            host_mem = cuda.pagelocked_empty(size, np.float32)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(cuda_mem))
            if self.engine.binding_is_input(binding):
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(cuda_mem)
            else:
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append(cuda_mem)
        return self.engine.create_execution_context()

    def post_process_batch(self, host_outputs, batch_size=1, conf=0.3, nms=0.45, result_path=None):
        """
        :param result_path:
        :param conf:
        :param nms:
        :param host_outputs: x, y, w, h, conf, cls1, cls2, cls3, cls4 ······
        :param batch_size:
        :return [[x1, y1, x2, y2, scores, cls_name], [x1, y1, x2, y2, scores, cls_name], ···]
        """
        if result_path is not None:
            np.save("{}.npy".format(self.engine_file_path), host_outputs)

        # xywh2xyxy (4ms)
        team_num = self.num_classes + 5
        prediction = host_outputs.reshape(batch_size, int(host_outputs.shape[0] / team_num / batch_size), team_num)
        box_corner = np.zeros(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        prediction = Tensor(prediction)

        # get detections
        output = [None for _ in range(len(prediction))]
        print(len(output))
        for i, image_pred in enumerate(prediction):
            # torch.max方法 提取类conf最高的值及其索引位置
            class_conf, class_pred = torch_max(image_pred[:, 5: team_num], dim=1, keepdim=True)
            # conf * class_conf, squeeze() 消除所有空白维度 eg:(2, 1, 2, 2, 1) -> (2, 2, 2)
            # 判断该项是否大于conf 返回True/False
            conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf).squeeze()
            # 拆分维度并整理
            # int(host_outputs.shape[0] / team_num / batch_size) 从 1*25200*85 还原到 25200
            # 转成[[x1, y1, x2, y2, score, idx], [x1, y1, x2, y2, score, idx], ...]
            detections = torch_cat([image_pred[:, :4],
                                    image_pred[:, 4].reshape(int(host_outputs.shape[0] / team_num / batch_size), 1) * class_conf,
                                    class_pred.float()], dim=1)
            # 如果conf_mask为True, 则使用, 否则None
            detections = detections[conf_mask]

            # iou nms
            nms_out_index = batched_nms(
                boxes=detections[:, :4],
                scores=detections[:, 4],
                idxs=detections[:, 5],
                iou_threshold=nms)

            output[i] = detections[nms_out_index]
        return output

    # 释放引擎，释放GPU显存，释放CUDA流
    def __del__(self):
        """Free CUDA memories."""
        del self.stream
        del self.cuda_outputs
        del self.cuda_inputs
        del self.cfx

    # 销毁存储的可推理上下文
    def destroy(self):
        # 销毁上下文
        # https://documen.tician.de/pycuda/driver.html#pycuda.driver.Context.detach
        self.cfx.detach()


if __name__ == '__main__':
    detection = TRTDetection(
        engine_file_path="../models/yolov5n_best_640_batch_8.trt",
        cls_list=["box"],
        batch_size=8,
        exp_size=(640, 640)
    )

