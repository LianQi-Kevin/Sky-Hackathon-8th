# Export Yolov5-v6.0

## 1. Install requirements

```shell
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
```

## 2. Clone Yolov5 repository and Change

### 2.1 Clone repository

```shell
git clone -b v6.0 https://github.com/ultralytics/yolov5.git && cd yolov5
```

### 2.2 Modify the code for dynamic batchsize

> https://github.com/shouxieai/tensorRT_Pro#guide-for-different-tasksmodel-support

```python
# line 56 forward function in yolov5/models/yolo.py 
# bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
# x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
# modified into:

bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
bs = -1
ny = int(ny)
nx = int(nx)
x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

# line 70 in yolov5/models/yolo.py
#  z.append(y.view(bs, -1, self.no))
# modified into：
z.append(y.view(bs, self.na * ny * nx, self.no))

############# for yolov5-6.0 #####################
# line 60 in yolov5/models/yolo.py
# if self.grid[i].shape[2:4] != x[i].shape[2:4] or self.onnx_dynamic:
#    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
# modified into:
if self.grid[i].shape[2:4] != x[i].shape[2:4] or self.onnx_dynamic:
    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

# disconnect for pytorch trace
anchor_grid = (self.anchors[i].clone() * self.stride[i]).view(1, -1, 1, 1, 2)

# line 65 in yolov5/models/yolo.py
# y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
# modified into:
y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid  # wh

# line 69 in yolov5/models/yolo.py
# wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
# modified into:
wh = (y[..., 2:4] * 2) ** 2 * anchor_grid  # wh
############# for yolov5-6.0 #####################


# line 78 in yolov5/export.py
# torch.onnx.export(dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},  # shape(1,3,640,640)
#                                'output': {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)  修改为
# modified into:
torch.onnx.export(dynamic_axes={'images': {0: 'batch'},  # shape(1,3,640,640)
                                'output': {0: 'batch'}  # shape(1,25200,85) 
```

The changed file is [export.py](./replacement/export.py) and [yolo.py](./replacement/yolo.py). 
You can replace these files in the yolov5 folder

```shell
cp {Sky-Hackathon-8th}/2_ModelExport/replacement/export.py {yolov5}/export.py
cp {Sky-Hackathon-8th}/2_ModelExport/replacement/yolo.py {yolov5}/models/yolo.py
```

## 3. Export onnx models

```shell
#python export.py --imgsz=640 --weights=yolov5n.pt --include=onnx --dynamic
python export.py --imgsz=416 --weights=yolov5n.pt --include=onnx --batch-size 8
```

---

### reference

* [tensorRT_Pro](https://github.com/shouxieai/tensorRT_Pro)