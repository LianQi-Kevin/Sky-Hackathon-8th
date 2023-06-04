import os
import shutil
import threading

import cv2
from utils.TRTDetection import YoLov5TRT, get_img_path_batches


class InferThread(threading.Thread):
    def __init__(self, wrapper, image_path_batch):
        threading.Thread.__init__(self)
        self.yolov5_wrapper = wrapper
        self.image_path_batch = image_path_batch

    def run(self):
        batch_image_raw, use_time, _ = self.yolov5_wrapper.infer(
            self.yolov5_wrapper.get_raw_image(self.image_path_batch))
        for index, img_path in enumerate(self.image_path_batch):
            save_name = os.path.join('output', os.path.basename(img_path))
            # Save image
            cv2.imwrite(save_name, batch_image_raw[index])
        print('input->{}, time->{:.2f}ms, saving into output/'.format(self.image_path_batch, use_time * 1000))


class GetDetectionResult(threading.Thread):
    def __init__(self, wrapper, image_path_batch, detection_results):
        threading.Thread.__init__(self)
        self.yolov5_wrapper = wrapper
        self.image_path_batch = image_path_batch
        self.detection_results = detection_results
        self.categories = self.yolov5_wrapper.categories

        if os.path.exists(self.detection_results):
            shutil.rmtree(self.detection_results)
        os.makedirs(self.detection_results, exist_ok=True)

    def run(self) -> None:
        export_path = self.detection_results
        batch_image_raw, use_time, bndboxes = self.yolov5_wrapper.infer(
            self.yolov5_wrapper.get_raw_image(self.image_path_batch))
        for index, img_path in enumerate(self.image_path_batch):
            print(f"index: {index}", f"img_path: {img_path}")
            export_path = os.path.join(self.detection_results, f"{os.path.splitext(os.path.basename(img_path))[0]}.txt")
            with open(export_path, "w") as f:
                for bndbox, scores, cls_id in zip(bndboxes[index][0], bndboxes[index][1], bndboxes[index][2]):
                    f.write(f"{self.categories[int(cls_id)]} {scores} {bndbox[0]} {bndbox[1]} {bndbox[2]} {bndbox[3]}\n")
        print('input->{}, time->{:.2f}ms, saving into {}'.format(self.image_path_batch, use_time * 1000, export_path))


class WarmUpThread(threading.Thread):
    def __init__(self, wrapper):
        threading.Thread.__init__(self)
        self.yolov5_wrapper = wrapper

    def run(self):
        batch_image_raw, use_time, _ = self.yolov5_wrapper.infer(self.yolov5_wrapper.get_raw_image_zeros())
        print('warm_up->{}, time->{:.2f}ms'.format(batch_image_raw[0].shape, use_time * 1000))


if __name__ == "__main__":
    # load custom plugin and engine
    PLUGIN_LIBRARY = "plugins/libplugins_416.so"
    engine_file_path = "models/yolov5n6_416.trt"

    # load labels
    categories = ["box"]

    # export path
    if os.path.exists('output/'):
        shutil.rmtree('output/')
    os.makedirs('output/')

    # a YoLov5TRT instance
    yolov5_wrapper = YoLov5TRT(
        engine_file_path=engine_file_path,
        plugin=PLUGIN_LIBRARY,
        categories=categories
    )
    try:
        print('batch size is', yolov5_wrapper.batch_size)

        image_dir = "infer/images/"
        image_path_batches = get_img_path_batches(yolov5_wrapper.batch_size, image_dir)

        for i in range(10):
            thread1 = WarmUpThread(yolov5_wrapper)
            thread1.start()
            thread1.join()
        # for batch in image_path_batches:
        #     thread1 = InferThread(yolov5_wrapper, batch)
        #     thread1.start()
        #     thread1.join()
        for batch in image_path_batches:
            thread1 = GetDetectionResult(yolov5_wrapper, batch, "mAP/input/detection-results")
            thread1.start()
            thread1.join()

    finally:
        # destroy the instance
        yolov5_wrapper.destroy()
