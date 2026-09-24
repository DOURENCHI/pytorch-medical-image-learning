# PyTorch Medical Image Learning

## 1. 项目简介

本项目用于学习和实践基于 PyTorch 的医学影像深度学习方法。

项目主要围绕医学 X 射线图像展开，逐步实现从数据读取、数据预处理、Dataset / DataLoader、模型构建、损失函数、训练与验证，到医学图像分割实验的完整流程。

本项目同时作为后续医学影像 AI、模型部署和相关工程实践的基础代码仓库。

## 2. 技术栈

* Python
* PyTorch
* NumPy
* Pillow
* OpenCV
* Matplotlib

## 3. 项目结构

```text
pytorch-medical-image-learning/
│
├── data/                  # 数据集及相关文件
├── learning_start/        # 学习过程代码
│
├── dataset.py             # 自定义 Dataset
├── make_mask.py           # Mask 处理相关代码
│
├── test_dataset.py        # Dataset 测试
├── test_dataloader.py     # DataLoader 测试
├── test_model.py          # 模型测试
├── test_loss.py           # 损失函数测试
├── test_cnn.py            # CNN 测试
│
├── requirements.txt       # Python依赖
├── .gitignore             # Git忽略文件
└── README.md              # 项目说明
```

## 4. 环境配置

建议使用 Conda 创建独立 Python 环境。

安装依赖：

```bash
pip install -r requirements.txt
```

检查 PyTorch：

```python
import torch

print(torch.__version__)
print(torch.cuda.is_available())
```

## 5. 数据集

项目使用医学 X 射线图像进行学习和实验。

数据目录及具体数据组织方式根据实际实验数据进行配置。

原始数据集不建议直接上传到 GitHub，而是在 README 中说明数据集来源、下载方式以及目录结构。

## 6. 当前学习内容

目前项目主要完成以下内容：

* Python 医学图像处理基础
* X 射线图像读取
* 图像预处理
* NumPy 图像操作
* PyTorch Tensor 基础
* 自定义 Dataset
* DataLoader
* CNN 基础
* 模型结构理解
* Loss 函数基础
* PyTorch 基础训练流程

## 7. 实验流程

整体流程：

```text
医学影像
   ↓
图像读取
   ↓
图像预处理
   ↓
Dataset
   ↓
DataLoader
   ↓
深度学习模型
   ↓
Loss计算
   ↓
反向传播
   ↓
模型更新
   ↓
验证
   ↓
评价指标
```

## 8. 当前进展

项目目前处于 PyTorch 医学影像深度学习基础阶段。

后续将逐步加入：

* U-Net
* 医学图像分割
* Dice / IoU 等评价指标
* 完整训练与验证流程
* 实验结果可视化
* 模型保存与加载
* 医学影像 AI 实验

## 9. 后续计划

后续计划逐步扩展到：

1. 医学图像分割
2. MONAI 医学影像深度学习框架
3. 更完整的医学影像 AI 实验
4. 模型训练与性能优化
5. 模型部署
6. C++ / Python 推理
7. 端侧 AI 部署

## 10. 项目定位

本项目主要用于记录个人学习过程、实验代码和工程实践。

随着项目逐步完善，将持续整理代码结构、实验结果和项目文档，使其具备可复现性和工程展示价值。
