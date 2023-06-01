# ONNX TO TRT

## 1. Build trtexec 

* Our Deploy platform is Jetson Xavier NX. So the tensorRT is already installed
* Just need link to the `/usr/bin` path

```shell
sudo ln -s /usr/src/tensorrt/bin/trtexec /usr/bin/trtexec
```

## 2. ONNX to TensorRT engine

```shell
#生成静态batchsize的engine
./trtexec 	--onnx=<onnx_file> \ 						#指定onnx模型文件
        	--explicitBatch \ 							#在构建引擎时使用显式批大小(默认=隐式)显示批处理
        	--saveEngine=<tensorRT_engine_file> \ 		#输出engine
        	--workspace=<size_in_megabytes> \ 			#设置工作空间大小单位是MB(默认为16MB)
        	--fp16 										#除了fp32之外，还启用fp16精度(默认=禁用)
        
#生成动态batchsize的engine
./trtexec 	--onnx=<onnx_file> \						#指定onnx模型文件
        	--minShapes=input:<shape_of_min_batch> \ 	#最小的batchsize x 通道数 x 输入尺寸x x 输入尺寸y
        	--optShapes=input:<shape_of_opt_batch> \  	#最佳输入维度，跟maxShapes一样就好
        	--maxShapes=input:<shape_of_max_batch> \ 	#最大输入维度
        	--workspace=<size_in_megabytes> \ 			#设置工作空间大小单位是MB(默认为16MB)
        	--saveEngine=<engine_file> \   				#输出engine
        	--fp16   									#除了fp32之外，还启用fp16精度(默认=禁用)
```

---

### Reference

* [tensorRT_document](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)
* [trtexec_document](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#trtexec)