# ONNX TO TRT (切换到 tensorrtx/yolov5 方案，trtexec 搁置)

## 1. Build trtexec 

* Our Deploy platform is Jetson Xavier NX. So the tensorRT is already installed
* Just need link to the `/usr/bin` path

```shell
sudo ln -s /usr/src/tensorrt/bin/trtexec /usr/bin/trtexec
```

## 2. ONNX to TensorRT engine

```shell
./trtexec --onnx=models/yolov5n_best_640_batch_8.onnx --saveEngine=models/yolov5n_best_640_batch_8.trt --fp16
```

* 当前模型在使用`yolov5/export.py`导出为`onnx`格式的时候指定了静态的`batch-size`, 故此处无需二次指定`batch-size`.
* 如果要使用动态`batch-size`, 请使用该格式

```shell
# 生成动态batchsize的engine
./trtexec  --onnx=<onnx_file> \			        #指定onnx模型文件
           --minShapes=input:<shape_of_min_batch> \ 	#最小的batchsize x 通道数 x 输入尺寸x x 输入尺寸y
           --optShapes=input:<shape_of_opt_batch> \  	#最佳输入维度，跟maxShapes一样就好
           --maxShapes=input:<shape_of_max_batch> \ 	#最大输入维度
           --workspace=<size_in_megabytes> \ 		#设置工作空间大小单位是MB(默认为16MB)
           --saveEngine=<engine_file> \   		#输出engine
           --fp16
        	
# example
./trtexec  --onnx=yolov5n_best_640.onnx \
           --minShapes=input:1x3x640x640 \
           --optShapes=input:8x3x640x640 \
           --maxShapes=input:8x3x640x640 \
           --workspace=4096 \
           --saveEngine=yolov5n_best_640.trt \
           --fp16
```

---

### Reference

* [tensorRT_document](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)
* [trtexec_document](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#trtexec)