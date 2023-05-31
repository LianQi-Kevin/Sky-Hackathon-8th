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

### 2.2 Build Darknet

```shell
make
```

## 3. Download yolov4-tiny cfg and pretrained weights

```shell
mkdir -p workspace/weights
mkdir -p workspace/models
wget -P workspace/weights https://github.com/cap-lab/jedi/releases/download/jedi_legacy/yolo4tiny_relu.zip
unzip -d workspace/weights/ workspace/weights/yolo4tiny_relu.zip && rm workspace/weights/yolo4_relu.zip
wget -P workspace/models https://github.com/cap-lab/jedi/releases/download/jedi_legacy/yolo4tiny_relu.cfg
```
