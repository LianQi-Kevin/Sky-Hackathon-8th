# Yolov4-tiny Train notes (yolov4-tiny relu jedi pretrained ver.)

## 1. Clone Darknet repository

```shell
git clone -b yolov4 https://github.com/AlexeyAB/darknet.git && cd darknet
```

## 2. Change Makefile and Build darknet

### 2.1 Change Makefile to enabled opencv and GPU 

```shell
sed -i 's/OPENCV=0/OPENCV=1/' Makefile
sed -i 's/GPU=0/GPU=1/' Makefile
sed -i 's/CUDNN=0/CUDNN=1/' Makefile
sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile
sed -i 's/LIBSO=0/LIBSO=1/' Makefile
```

### 2.2 Verify cuda

```shell
nvcc --version
```

### 2.3 Build Darknet

```shell
make
```

## 3. Download yolov4-tiny cfg and pretrained weights

* for other custom dataset, please check [configure-files-for-training](https://haobin-tan.netlify.app/ai/computer-vision/object-detection/train-yolo-v4-custom-dataset/#configure-files-for-training) to change yolov4-tiny.cfg with your self

```shell
mkdir -p workspace/weights
mkdir -p workspace/cfg
wget -P workspace/weights https://github.com/cap-lab/jedi/releases/download/jedi_legacy/yolo4tiny_relu.zip
unzip -d workspace/weights/ workspace/weights/yolo4tiny_relu.zip && rm workspace/weights/yolo4_relu.zip
#wget -P workspace/cfg https://github.com/cap-lab/jedi/releases/download/jedi_legacy/yolo4tiny_relu.cfg
wget -P workspace/cfg 
```

## 4. Download Dataset

```shell
mkdir -p workspace/dataset
wget -O workspace/dataset/data.zip <Roboflow download Raw URL>
unzip -d workspace/dataset workspace/dataset/data.zip && rm workspace/dataset/data.zip
```

* `<Roboflow download Raw URL>`: 请自行替换为roboflow数据集下载地址, 下载格式请选择 `YOLO Darknet`
* 请访问 <a href="https://universe.roboflow.com/hackathon-8th/box-detect-omniverse"><img src="https://app.roboflow.com/images/download-dataset-badge.svg"></img></a> , 并点击`Download this Dataset`获取下载链接


---

### Reference

* [train-yolo-v4-custom-dataset](https://haobin-tan.netlify.app/ai/computer-vision/object-detection/train-yolo-v4-custom-dataset/)