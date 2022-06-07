# Rotate-Captcha-Crack

CNN预测图片旋转角度

在下文提到的数据集上训练28个epoch（耗时24min）得到的平均预测误差为`22.06°`，模型文件大小`39.7MB`，可以轻松破解某度的旋图验证码

## 准备环境

+ 一张显存4G以上的GPU

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

## 设计细节

+ 搞这个项目的主要目的是练手[`torchdata`](https://pytorch.org/data/beta/index.html)，数据集的预处理与构建流程是本项目的精髓所在

+ 现有的旋图验证码破解方法大多基于[`RotNet (ICLR2018)`](https://arxiv.org/abs/1803.07728)，其backbone为`ResNet50`，将角度预测视作分类问题，并计算交叉熵损失，本项目的`RotationNet`是对`RotNet`的简单改进

+ backbone为[`regnet (CVPR2020)`](https://arxiv.org/abs/2003.13678)的`RegNetY 1.6GFLOPs`

+ 损失函数`RotationLoss`和RotNet仓库中给出的[`angle_error_regression`](https://github.com/d4nst/RotNet/blob/a56ea59818bbdd76d4dd8d83b8bbbaae6a802310/utils.py#L30-L36)思路接近，使用最后的全连接层预测一个角度值并与`ground-truth`作差，最后在`MSELoss`的基础上加了个余弦因子来缩小真实值的`± n * 360°`与真实值之间的度量距离

+ 该损失函数可导且几乎为凸，为什么说是几乎，因为当`lambda_cos>0.25`时在`predict=±1`的地方可能出现局部最小

+ 下面比较一下`RotationLoss`和`angle_error_regression`的函数图像

- angle_error_regression

![RotNet-angle_error_regression](https://user-images.githubusercontent.com/48282276/172344856-e3904a62-a099-40af-86cd-6174a6bf5e3f.png)

- RotationLoss

![This-RotationLoss](https://user-images.githubusercontent.com/48282276/172344938-fd2f5991-152a-49b0-a0bd-5a7f6402c947.png)
