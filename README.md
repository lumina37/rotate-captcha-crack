# Rotate-Captcha-Crack

CNN预测图片旋转角度，可用于破解图像旋转验证码

搞这个项目的主要目的是练手[`torchdata`](https://pytorch.org/data/beta/index.html)，数据集的预处理与构建也是本项目的精髓所在

因为现有的旋图验证码破解方法大多基于[`RotNet (ICLR 2018)`](https://arxiv.org/abs/1803.07728)，我就做了这个Up-to-Date的方案：用[`regnet (CVPR 2020)`](https://arxiv.org/abs/2003.13678)做backbone；设计了一个损失函数`RotationLoss`，和RotNet的那个[`angle_error_regression`](https://github.com/d4nst/RotNet/blob/a56ea59818bbdd76d4dd8d83b8bbbaae6a802310/utils.py#L30-L36)的思路差不多，就是预测出一个角度然后和`ground-truth label`作差，具体点就是在`MSELoss`的基础上加了个余弦系数来缩小真实值的`± n * 360°`与真实值之间的度量距离

## 准备环境

+ 一张4G以上显存的GPU，不然就只能设很小的`batch_size`，会导致训练效果很垃圾
+ 确保你的`Python`版本`>=3.7`

+ 拉取代码并安装依赖库

```bash
git clone https://github.com/Starry-OvO/Rotate-Captcha-Crack.git
cd ./Rotate-Captcha-Crack
pip install -r requirements.txt
```

## 偷数据集

+ 我这里直接偷的[`Landscape-Dataset`](https://github.com/yuweiming70/Landscape-Dataset)，你也可以自己扒拉一些风景照放到任意一个文件夹里，因为是自监督学习，所以不限制图像尺寸也不需要标注
+ 在`config.yaml`里配置`dataset:root`字段指向装有图片的文件夹
+ 运行`prepare.py`准备数据集

```bash
python prepare.py
```

## 训练

```bash
python train.py
```

## 在测试集上验证模型

```bash
python evaluate.py
```

## 扔一张图进去看看实际效果

Linux环境需要配置GUI或者自己把debug函数从显示图像改成保存图像

```bash
python test.py
```
