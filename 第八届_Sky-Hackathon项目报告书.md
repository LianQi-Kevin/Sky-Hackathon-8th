<center><h1>第八届 Sky Hackathon</h1></center>

<center><h2>参赛项目书</h2></center>



参赛学校：北京建筑大学

参赛队名：无辑

指导老师：孙绪华

团队成员：赵泽宇、刘学恺、王成儒、劳钧彦、单坤

---

## 一、场景建模部分

*   使用 NVIDIA Omniverse Code 完成了场景的构建，共使用各类构件 35 个，合成了11000张图片，人工筛选后保留10000张图片进入数据增强阶段，除此之外还完成了PASCAL VOC 格式的自定义 Writer

    *   合成脚本：[FactoryScene1.py](https://github.com/LianQi-Kevin/Sky-Hackathon-8th/blob/main/BACKEND/0_Omniverse/FactoryScene1.py)

    *   PASCAL_Writer：[PascalWriter.py](https://github.com/LianQi-Kevin/Sky-Hackathon-8th/blob/main/BACKEND/0_Omniverse/tools/PascalWriter.py)

*   借助 roboflow 完成了数据集的格式转换、数据增强和快速共享

    *   添加$2\%$的椒盐噪声、$±15$ 度的图片旋转和 $± 25\%$ 的亮度调整
    *   按照95%、4%、1%分割了训练、验证和测试集
    *   增强后的图片训练集26K张，验证集1.1K张，测试集190张
    *   下载数据集：<a href="https://universe.roboflow.com/hackathon-8th/box-detect-omniverse">
            <img src="https://cdn.jsdelivr.net/gh/LianQi-Kevin/Markdown_Image_Hosting@main/images/202306041635567.svg+xml"></img>
        </a>

*   部分训练集展示：

    <img src="https://cdn.jsdelivr.net/gh/LianQi-Kevin/Markdown_Image_Hosting@main/images/202306041637096.jpg" alt="train_batch0" style="zoom: 25%;" />

    <img src="https://cdn.jsdelivr.net/gh/LianQi-Kevin/Markdown_Image_Hosting@main/images/202306041637032.jpg" alt="train_batch1" style="zoom: 25%;" />

## 二、模型训练

*   本次赛事使用 [Yolov5-v6.0](https://github.com/ultralytics/yolov5/releases/tag/v6.0) 完成了模型的训练任务

*   共训练了四个模型，分别是`yolov5n-416`、`yolov5n6-416`、`yolov5n-640`、`yolov5n6-640`

*   训练用数据集均采用前文提到托管在 roboflow 的数据集

*   yolov5n

    ![results](https://cdn.jsdelivr.net/gh/LianQi-Kevin/Markdown_Image_Hosting@main/images/202306041641187.png)

*   yolov5n6

    ![results](https://cdn.jsdelivr.net/gh/LianQi-Kevin/Markdown_Image_Hosting@main/images/202306041641280.png)

## 三、模型导出与部署

*   本次比赛参考了 [tensorrtx](https://github.com/wang-xinyu/tensorrtx) 来完成部署端的脚本
*   编译 tensorrtx/yolov5 后将`.pt`转换到`.wts`，再在部署端对模型进行 TRT 加速并保存序列化的文件
*   [yolov5/README.md](https://github.com/wang-xinyu/tensorrtx/blob/yolov5-v6.0/yolov5/README.md)

## 四、UI优化

*   本次采用了 VUE3 + axios 来完成前端部分

## 五、团队收获与杂项

### 5.1 参赛感想与收获

在比赛为期五周的时间里，我成功部署了YOLOV5, 并且学会了如何基于ONNX序列化 TensorRT engine。
 深入了解了GPU异步和CPU多线程操作，制作了异步的推理脚本，提高了模型推理速度，也提高了我们的编程水平。
 令我充分认识到了项目式学习的高效，在认真阅读开源项目的过程中受益匪浅，深刻领悟到了高性能、低耦合的重要性。
 同时，我们也深受开源精神的感召，感谢行业先驱们不仅设计出了优质、高效的算法，还将该算法无私地提供给所有人，为了帮助我们学习该算法甚至免费提供平台和算力，供我们实验，激励着我们不断学习新知识，并在将来为开源世界添砖加瓦。

### 5.2 团队工作照片

<img src="https://cdn.jsdelivr.net/gh/LianQi-Kevin/Markdown_Image_Hosting@main/images/202306041651513.jpg" alt="40b5ff69581903ae85655c39048f8dc" style="zoom: 33%;" />

<img src="https://cdn.jsdelivr.net/gh/LianQi-Kevin/Markdown_Image_Hosting@main/images/202306041651094.jpg" alt="cc3cd1a0e4ef7dc4bc7b72e98a0f6f1" style="zoom: 33%;" />

### 5.3 参考和完成的相关库

*   https://github.com/LianQi-Kevin/Sky-Hackathon-8th
*   https://github.com/wang-xinyu/tensorrtx
*   [链接 OpenCV 和 TensorRT 到 conda 环境](https://blog.csdn.net/liuliu123456790/article/details/131004744)