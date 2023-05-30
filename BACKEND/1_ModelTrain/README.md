# Yolov5 Train notes

## 1. install required
```shell
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
```

## 2. Clone Yolov5-v6.0 and download pretrained models

### 2.1 Clone Yolov6-v6.0

```shell
git clone -b v6.0 https://github.com/ultralytics/yolov5.git && cd yolov5
```

### 2.2 Download pretrained models

```shell
mkdir -p workspace/weights
wget -P workspace/weights https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s6.pt
wget -P workspace/weights https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5n6.pt
```

### 2.3 copy and change model_name.yaml

```shell
mkdir -p workspace/models
cp models/hub/yolov5s6.yaml workspace/models/yolov5s6.yaml
cp models/hub/yolov5n6.yaml workspace/models/yolov5n6.yaml
```

修改 `workspace/models/yolov5s6.yaml` and `workspace/models/yolov5s6.yaml`两个文件的第四行`nc`的值为`1`

```yaml
# nc: 80  # number of classes
nc: 1
```

## 3. Download Dataset and Train the model

```shell
mkdir -p workspace/dataset
wget -P workdpace/dataset ""
```