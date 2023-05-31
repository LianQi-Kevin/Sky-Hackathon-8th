# Yolov5n6 Train notes

## 1. install required
```shell
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
```

## 2. Clone Yolov5-v6.0 and download pretrained models

### 2.1 Clone Yolov6-v6.0 repository

```shell
git clone -b v6.0 https://github.com/ultralytics/yolov5.git && cd yolov5
```

### 2.2 Download pretrained models

```shell
mkdir -p workspace/weights
wget -P workspace/weights https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5n6.pt
```

### 2.3 copy and change model_name.yaml

```shell
mkdir -p workspace/models
cp models/hub/yolov5n6.yaml workspace/models/yolov5n6.yaml
```

修改`workspace/models/yolov5s6.yaml`的第四行`nc`的值为`1`

```yaml
# nc: 80  # number of classes
nc: 1
```

## 3. Download Dataset

```shell
mkdir -p workspace/dataset
wget -O workspace/dataset/data.zip <Roboflow download Raw URL>
unzip -d workspace/dataset workspace/dataset/data.zip && rm workspace/dataset/data.zip
```

* `<Roboflow download Raw URL>`: 请自行替换为roboflow数据集下载地址, 下载格式请选择 `YOLOv5`
* 请访问 <a href="https://universe.roboflow.com/hackathon-8th/box-detect-omniverse"><img src="https://app.roboflow.com/images/download-dataset-badge.svg"></img></a> , 并点击`Download this Dataset`获取下载链接

下载后修改`workspace/dataset/data.yaml`前三行的路径

```yaml
#train: ../train/images
#val: ../valid/images
#test: ../test/images
train: workspace/dataset/train/images
val: workspace/dataset/valid/images
test: workspace/dataset/test/images
```


## 4. Train Yolov5

```shell
python train.py --data workspace/dataset/data.yaml --weights workspace/weights/yolov5n6.pt --cfg workspace/models/yolov5n6.yaml --batch-size 128 --epochs 500 
```

## 5. Tensorboard show

```shell
tensorboard --logdir runs/train --port 6006 --bind_all
```