# Sky-Hackathon 8th Back-end

该文件夹为赛事后端部分，计划使用 Flask 完成服务封装。模型则采用 Yolov5n

---

### 关于 omni.replicator.core 的代码补全方法

* 暂时未找到官方的相关方案，使用`mklink`命令解决

```shell
cd {Sky-Hackathon-8th}/BACKEND
mklink /J ./omni {OMNIVERSE CODE安装路径}/extscache/omni.replicator.core-1.7.7+104.2.wx64.r.cp37/omni
```

> 此处仅解决代码补全，运行脚本仍需要使用`code-2022.3.3/omni.code.bat`