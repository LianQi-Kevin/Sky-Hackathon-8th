# 链接 OpenCV 和 TensorRT 到 conda 环境

## 1. 重新安装和编译 OpenCV

* Jetson Xavier NX, Jetpack 4.6.1, OpenCV 4.1.1, TensorRT 8.2.1.8

### 1.1 卸载现有的 OpenCV

```shell
sudo sudo apt-get purge *libopencv*
sudo apt autoremove
```

### 1.2 安装依赖项

```shell
sudo apt-get update
sudo apt-get install -y build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
sudo apt-get install -y python2.7-dev python3.6-dev python-dev python-numpy python3-numpy
sudo apt-get install -y libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev
sudo apt-get install -y libv4l-dev v4l-utils qv4l2 v4l2ucp
sudo apt-get install -y wget
```

### 1.3 下载并解压 opencv-4.1.1 和 opencv_contrib-4.1.1

```shell
wget https://github.com/opencv/opencv/archive/4.1.1.zip -O opencv-4.1.1.zip && unzip opencv-4.1.1.zip
wget https://github.com/opencv/opencv_contrib/archive/4.1.1.zip -O opencv_contrib-4.1.1.zip && unzip opencv_contrib-4.1.1.zip
```

### 1.4 下载 boostdesc_bgm

* 访问 [boostdesc_bgm_eg.zip](https://pan.baidu.com/s/1jHBV7Kv3kWrpHl0UYQ9FjQ?pwd=b9ao)，下载`boostdesc_bgm_eg.zip`并上传到`opencv_contrib-4.1.1`文件夹

```shell
unzip boostdesc_bgm_eg.zip -d modules/xfeatures2d/src/ && rm boostdesc_bgm_eg.zip
```

### 1.5 编译 OpenCV 4.1.1

#### 1.5.1 创建并进入opencv-4.1.1/build文件夹

```shell
mkdir -p opencv-4.1.1/build && cd opencv-4.1.1/build
```

* 当前的文件夹结构应为

```
workspace
├─opencv-4.1.1
│  └─build    (当前的位置)
└─opencv_contrib-4.1.1
   └─modules
```

#### 1.5.2 编译 OpenCV 到 conda 环境

* 需要配置链接 OpenCV 的 conda 环境的路径为`/home/jetson/miniconda/envs/Yolov5`, 请自行替换对应环境的路径

```shell
cmake \
-DBUILD_opencv_python3=ON \
-DBUILD_opencv_python2=OFF \
-DPYTHON3_EXECUTABLE=/home/jetson/miniconda3/envs/Yolov5/bin/python3.6m \
-DPYTHON_INCLUDE_DIR=/home/jetson/miniconda3/envs/Yolov5/include/python3.6m \
-DPYTHON_LIBRARY=/home/jetson/miniconda3/envs/Yolov5/lib/libpython3.6m.so \
-DPYTHON_NUMPY_INCLUDE_DIRS=/home/jetson/miniconda3/envs/Yolov5/lib/python3.6/site-packages/numpy/core/include \
-DPYTHON_PACKAGES_PATH=/home/jetson/miniconda3/envs/Yolov5/lib/python3.6/site-packages \
-DPYTHON_DEFAULT_EXECUTABLE=/home/jetson/miniconda3/envs/Yolov5/bin/python3.6m \
-DCMAKE_INSTALL_PREFIX=/usr/local \
-DBUILD_EXAMPLES=OFF \
-DOPENCV_GENERATE_PKGCONFIG=ON \
-DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
-DCMAKE_BUILD_TYPE=RELEASE \
-DOPENCV_ENABLE_NONFREE=1 \
-DWITH_FFMPEG=1 \
-DCUDA_ARCH_BIN=7.2 \
-DCUDA_ARCH_PTX=7.2 \
-DWITH_CUDA=1 \
-DENABLE_FAST_MATH=1 \
-DCUDA_FAST_MATH=1 \
-DWITH_CUBLAS=1 \
-DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.1.1/modules \
..
```

### 1.5.3 编译

```shell
make -j6
sudo make install
```

* [OpenCV编译问题笔记](#OpenCV编译问题笔记)

### 3. 编译 OpenCV 到 conda 环境

```shell
cmake \
-DBUILD_opencv_python3=ON \
-DBUILD_opencv_python2=OFF \
-DPYTHON3_EXECUTABLE=/home/nvidia/archiconda3/envs/yolo5/bin/python3.6m \
-DPYTHON_INCLUDE_DIR=/home/nvidia/archiconda3/envs/yolo5/include/python3.6m \
-DPYTHON_LIBRARY=/home/nvidia/archiconda3/envs/yolo5/lib/libpython3.6m.so \
-DPYTHON_NUMPY_INCLUDE_DIRS=/home/nvidia/archiconda3/envs/yolo5/lib/python3.6/site-packages/numpy/core/include \
-DPYTHON_PACKAGES_PATH=/home/nvidia/archiconda3/envs/yolo5/lib/python3.6/site-packages \
-DPYTHON_DEFAULT_EXECUTABLE=/home/nvidia/archiconda3/envs/yolo5/bin/python3.6m \
-DCMAKE_INSTALL_PREFIX=/usr/local \
-DBUILD_EXAMPLES=OFF \
-DOPENCV_GENERATE_PKGCONFIG=ON \
-DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
-DCMAKE_BUILD_TYPE=RELEASE \
-DOPENCV_ENABLE_NONFREE=1 \
-DWITH_FFMPEG=1 \
-DCUDA_ARCH_BIN=7.2 \
-DCUDA_ARCH_PTX=7.2 \
-DWITH_CUDA=1 \
-DENABLE_FAST_MATH=1 \
-DCUDA_FAST_MATH=1 \
-DWITH_CUBLAS=1 \
-DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.1.1/modules ..
```

---

### OpenCV编译问题笔记

#### 1. liblibprotobuf.io

* 报错信息

  ```
  [ 26%] Linking CXX static library ../lib/liblibprotobuf.a 
  [ 26%] Built target libprotobuf Makefile:162: recipe for target 'all' failed 
  make: *** [all] Error 2
  ```

* 解决方法

  因为 eigen 库默认安装在了`/usr/include/eigen3/Eigen`路径下, 需使用下面命令映射到`/usr/include`路径下

  ```shell
  sudo apt-get install libeigen3-dev
  sudo ln -s /usr/include/eigen3/Eigen /usr/include/Eigen
  ```

---

### Reference

* [Jetson Xavier NX设备将opencv和tensorrt链接到conda环境](https://blog.csdn.net/weixin_46151178/article/details/129037080)
* [Opencv之undefined reference to cv::face::BasicFaceRecognizer](https://mp.weixin.qq.com/s/L7Dln5y0HgjGMCOomei4kQ)
* [Jetson tx2 nano nx xavier 安装opencv4.1.1和OpenCV_contrib-4.1.1](https://blog.csdn.net/weixin_46151178/article/details/116133598)
