import os
import shutil
import threading

import cv2
from utils.TRTDetection import YoLov5TRT, get_img_path_batches


class InferThread(threading.Thread):
    def __init__(self, wrapper, image_path_batch, ):
        threading.Thread.__init__(self)
        self.yolov5_wrapper = wrapper
        self.image_path_batch = image_path_batch

    def run(self):
        batch_image_raw, use_time, bndboxes = self.yolov5_wrapper.infer(
            self.yolov5_wrapper.get_raw_image(self.image_path_batch))
        for index, img_path in enumerate(self.image_path_batch):
            print(f"index: {index}", f"img_path: {img_path}", bndboxes[index][0])
            save_name = os.path.join('output', os.path.basename(img_path))
            # Save image
            cv2.imwrite(save_name, batch_image_raw[index])
        print('input->{}, time->{:.2f}ms, saving into output/'.format(self.image_path_batch, use_time * 1000))


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

        image_dir = "infer/"
        image_path_batches = get_img_path_batches(yolov5_wrapper.batch_size, image_dir)

        for i in range(10):
            # create a new thread to do warm_up
            thread1 = WarmUpThread(yolov5_wrapper)
            thread1.start()
            thread1.join()
        for batch in image_path_batches:
            # create a new thread to do inference
            thread1 = InferThread(yolov5_wrapper, batch)
            thread1.start()
            thread1.join()
    finally:
        # destroy the instance
        yolov5_wrapper.destroy()
