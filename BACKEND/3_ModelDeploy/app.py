import subprocess
import os
import sys
import shutil
import time
import asyncio

import cv2
from PIL import Image
from datetime import datetime

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from utils.TRTDetection import YoLov5TRT, get_img_path_batches
from detect import InferThread

sys.path.append('/home/jetson/Sky-Hackathon-8th/BACKEND/3_ModelDeploy')

app = Flask("Sky-Hackathon-8th", static_folder='')
CORS(app)
uploadPath = 'uploads/'
engine_file_path = "models/yolov5n6_640.trt"
plugin = "plugins/libplugins_640.so"
categories = ["box"]

# export path
if os.path.exists('output/'):
    shutil.rmtree('output/')
os.makedirs('output/')


@app.route('/')
def home():
    return render_template('sky.html', template_folder='templates')


@app.route("/cv/upload", methods=['POST'])
def cvUpload():
    if request.method == 'POST':
        f = request.files['file']
        print('image', f, f.filename)

        if not 'image' in f.headers.get('Content-Type'):
            return '图片有错误', 400

        original = f'{uploadPath}original.jpg'

        try:
            # Convert image to jpeg
            im = Image.open(f)
            rgb_im = im.convert('RGB')
            rgb_im.save(original)

            # Add timestamp
            dt = datetime.now()
            ts = str(int(datetime.timestamp(dt)))
            return jsonify(original + '?t=' + ts)

        except Exception as _:
            return '有错误', 400


@app.route("/api/detect/image")
def detectImage():
    yolov5_wrapper = YoLov5TRT(
        engine_file_path=engine_file_path,
        plugin=plugin,
        categories=categories
    )
    try:
        thread1 = InferThread(yolov5_wrapper, ["uploads/original.jpg"])
        thread1.start()
        thread1.join()
    finally:
        yolov5_wrapper.destroy()
        yolov5_wrapper = ""
    result = {
        "detection_result_image_path": f'/output/original.jpg?t={str(int(datetime.timestamp(datetime.now())))}'
    }
    return jsonify(result)


@app.route("/api/detect/fps")
def detectFPS():
    # Code here
    # fps_results = subprocess.Popen('python3 /home/nvidia/8th_CV/cv_fps.py', shell=True, stdout=subprocess.PIPE,
    #                                stderr=subprocess.STDOUT)
    # fps_results = str(fps_results.stdout.read()).split('\\n')[-2]
    # fps_results = fps_results.split(" ")[-1]
    yolov5_wrapper = YoLov5TRT(
        engine_file_path=engine_file_path,
        plugin=plugin,
        categories=categories
    )
    try:
        async def infer(raw_images):
            return yolov5_wrapper.infer(raw_images)
        # init
        Write = False
        fps = 0.0
        tic = time.time()
        video = cv2.VideoCapture("infer/videos/test.mp4")
        fps = video.get(cv2.CAP_PROP_FPS)
        videoWriter = None
        if Write:
            frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
            videoWriter = cv2.VideoWriter("output/result.mp4", fourcc, fps, (frame_width, frame_height))
        # load and infer
        # tasks = []
        while True:
            _, frame = video.read()
            if frame is not None:
                # asyncio.wait(tasks)
                batch_image_raw, _, _ = yolov5_wrapper.infer([frame])
                if Write and batch_image_raw[0] is not None:
                    videoWriter.write(batch_image_raw[0])
                toc = time.time()
                curr_fps = 1.0 / (toc - tic)
                fps = curr_fps if fps == 0.0 else (fps * 0.95 + curr_fps * 0.05)
                tic = toc
            else:
                break
        video.release()
        cv2.destroyAllWindows()
    finally:
        yolov5_wrapper.destroy()
        yolov5_wrapper = ""

    result = {
        "detection_FPS": fps,
    }
    return jsonify(result)


@app.route("/api/detect/map")
def detectMAP():
    yolov5_wrapper = YoLov5TRT(
        engine_file_path=engine_file_path,
        plugin=plugin,
        categories=categories
    )
    try:
        image_path = "/home/jetson/Sky-Hackathon-8th/BACKEND/3_ModelDeploy/mAP/input/images-optional"
        export_root_path = "/home/jetson/Sky-Hackathon-8th/BACKEND/3_ModelDeploy/mAP/input/detection-results"
        for batch in get_img_path_batches(yolov5_wrapper.batch_size, image_path):
            batch_image_raw, use_time, bndboxes = yolov5_wrapper.infer(
                yolov5_wrapper.get_raw_image(batch))
            export_path = ""
            for index, img_path in enumerate(batch):
                # print(f"index: {index}", f"img_path: {img_path}")
                export_path = os.path.join(export_root_path, f"{os.path.splitext(os.path.basename(img_path))[0]}.txt")
                with open(export_path, "w") as f:
                    for bndbox, scores, cls_id in zip(bndboxes[index][0], bndboxes[index][1], bndboxes[index][2]):
                        f.write(f"box {scores} {bndbox[0]} {bndbox[1]} {bndbox[2]} {bndbox[3]}\n")
            print('input->{}, time->{:.2f}ms, saving into {}'.format(batch, use_time * 1000, export_path))

        mAP = subprocess.Popen('python3 mAP/main.py -na -np -q', shell=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
        map_results = mAP.stdout.read().decode().split(" = ")[1]
        print(f"mAP: {map_results}")
        cv2.destroyAllWindows()
    finally:
        yolov5_wrapper.destroy()
        yolov5_wrapper = ""

    result = {
        "detection_mAP": map_results,
    }
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
