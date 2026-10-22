建筑物损伤检测数据集，![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7acd075f42684b3cb40a7a22edca8a2d.png)
14304张，yolo，voc，coco三种标注方式
图像尺寸:640*640
类别数量:5类![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0ed3242ec77d4d73a5eaab9626f07964.png)

训练集图像数量:11443; 验证集图像数量:1430； 测试集图像数量:1431
类别名称: 每一类图像数 ，每一类标注数
crackrotation - 裂缝：6253，17044
leakagerotation - 渗漏：2464，6642
abscissionrotation - 脱落：5165，22700
corrosionrotation - 锈蚀：2102，9207
bulgerotation - 鼓包：1170，1851
image num: 14304

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3d20c89750144bad818e279f2f106f39.png)

模型与源代码
采用 YOLOv11n 网络训练![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3b96c8ba034f4aea929034e0e519e11e.png)

训练轮次：80 个 epoch![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/af76aa1803fa4e788372134c01cca7ed.png)

提供训练好的yolo11n模型，全部训练 + 测试源代码
训练精度 mAP 效果如图所示![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a5dd0ff04a4c4777bb349a0520507508.png)

PyQt5 界面功能![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e4e30b006a4d474294388b57cf912c5e.png)

界面使用 PyQt5 开发，提供全部源码（.ui、.qrc、.py 及图标文件）
支持图片检测、视频检测、摄像头实时检测![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/647a91fd61ef4af1be1114afd295f1c2.png)

界面实时显示：目标位置、目标总数、置信度等信息
支持检测结果保存导出
操作简单直观，无需命令行
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2af28a8bf2a74fee97444495fe8e5ff2.png)



4、
运行环境：Python=3.8、opencv-python、PyQt5、torch
支持 Windows、Linux 系统

# 建筑物损伤检测数据集
## 数据集信息总表
|项目|参数|
| ---- | ---- |
|数据集名称|建筑物损伤检测数据集|
|总图像数量|14304张|
|图像尺寸|640 × 640|
|标注格式|YOLO、VOC、COCO|
|类别数量|5类|
|训练集|11443张|
|验证集|1430张|
|测试集|1431张|

## 类别分布表
|类别英文|类别中文|含该类图像数|标注框总数|
| ---- | ---- |---:|---:|
|crackrotation|裂缝|6253|17044|
|leakagerotation|渗漏|2464|6642|
|abscissionrotation|脱落|5165|22700|
|corrosionrotation|锈蚀|2102|9207|
|bulgerotation|鼓包|1170|1851|

## 数据集标签&关键词
**标签**：裂缝、渗漏、脱落、锈蚀、鼓包
**关键词**：建筑物缺陷检测、混凝土病害、YOLOv11、目标检测、建筑巡检、无人机巡检、结构损伤识别

## 应用场景
1. 建筑物/桥梁结构安全定期巡检，无人机航拍自动识别墙体病害
2. 老旧小区外立面安全隐患排查，自动标记裂缝、墙体脱落风险区域
3. 市政工程、高架、廊道构筑物健康监测
4. 高校科研、毕业设计、深度学习算法训练验证
5. 建筑运维平台，自动检测图片/视频/摄像头实时画面的结构损伤

## 1. data.yaml 数据集配置文件
```yaml
path: ./building_damage_dataset
train: images/train
val: images/val
test: images/test

names:
  0: crackrotation
  1: leakagerotation
  2: abscissionrotation
  3: corrosionrotation
  4: bulgerotation
```

## 2. YOLOv11n 训练代码
```python
from ultralytics import YOLO

# 加载yolov11n预训练权重
model = YOLO("yolo11n.pt")

# 开始训练
train_results = model.train(
    data="data.yaml",
    epochs=80,
    imgsz=640,
    batch=8,
    device=0,
    workers=2,
    project="building_damage",
    name="yolo11n_building",
    conf=0.25
)

# 验证模型精度
metrics = model.val()
print(f"mAP@0.5 = {metrics.box.map50:.4f}")

# 单图推理
model.predict("test.jpg", save=True, conf=0.3)

# 导出onnx部署模型
model.export(format="onnx")
```

## 3. PyQt5简易GUI推理界面代码
```python
import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from ultralytics import YOLO

class DetWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("建筑物损伤检测系统 YOLOv11")
        self.resize(1000,700)
        self.model = YOLO("runs/building_damage/yolo11n_building/weights/best.pt")

        self.btn_img = QPushButton("图片检测",self)
        self.btn_img.move(20,20)
        self.btn_img.clicked.connect(self.detect_img)
        self.label_show = QLabel(self)
        self.label_show.setGeometry(20,80,800,550)

    def detect_img(self):
        fname,_ = QFileDialog.getOpenFileName()
        if not fname:
            return
        res = self.model(fname,conf=0.3)
        img = res[0].plot()
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        h,w,c = img.shape
        qimg = QImage(img.data,w,h,c*w,QImage.Format_RGB888)
        self.label_show.setPixmap(QPixmap.fromImage(qimg))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = DetWindow()
    win.show()
    sys.exit(app.exec_())
```

## 环境依赖
```
python==3.8
torch
ultralytics
opencv-python
pyqt5
numpy
```

