# Rotate-Captcha-Crack

CNN预测图片旋转角度

在下文提到的数据集上训练35个epoch（耗时30min）得到的平均预测误差为`19.98°`，模型文件大小`3.8MB`，可以轻松破解某度的旋图验证码

## 准备环境

+ 一张显存4G以上的GPU

+ 确保你的`Python`版本`>=3.7`

+ 确保你的`PyTorch`版本`>=1.11`

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

Linux环境需要配置GUI或者自己把debug方法从显示图像改成保存图像

```bash
python test.py
```

## 设计细节

+ 搞这个项目的主要目的是练手[`torchdata`](https://pytorch.org/data/beta/index.html)，数据集的预处理与构建流程是本项目的精髓所在

+ 现有的旋图验证码破解方法大多基于[`RotNet (ICLR2018)`](https://arxiv.org/abs/1803.07728)，其backbone为`ResNet50`，将角度预测视作360分类问题，并计算交叉熵损失，本项目的`RotationNet`是对`RotNet`的简单改进

+ backbone为[`regnet (CVPR2020)`](https://arxiv.org/abs/2003.13678)的`RegNetX 1.6GFLOPs`

+ `RotNet`中使用的交叉熵损失会令`1°`和`359°`之间的度量距离接近一个类似`358°`的较大值，这显然是一个违背常识的结果，它们之间的度量距离应当是一个类似`2°`的极小值。而`RotNet`仓库（并未写入论文）给出的[`angle_error_regression`](https://github.com/d4nst/RotNet/blob/a56ea59818bbdd76d4dd8d83b8bbbaae6a802310/utils.py#L30-L36)损失函数效果较差，因为该损失函数在应对离群值时梯度方向存在明显问题，你可以在后续的损失函数图像比对中轻松看出这一点

+ 本人设计的损失函数`RotationLoss`和`angle_error_regression`的思路相近，我使用最后的全连接层预测出一个角度值并与`ground-truth`作差，然后在`MSELoss`的基础上加了个余弦约束项来缩小真实值的`±k*360°`与真实值之间的度量距离

+ 为什么这里使用`MSELoss`，因为自监督学习生成的`label`可以保证不含有任何离群值，因此损失函数设计不需要考虑离群值的问题，同时`MSELoss`不破坏损失函数的可导性

+ 该损失函数在整个实数域可导且几乎为凸，为什么说是几乎，因为当`lambda_cos>0.25`时在`predict=±1`的地方会出现局部极小

+ 最后直观比较一下`RotationLoss`和`angle_error_regression`的函数图像

- angle_error_regression

![RotNet-angle_error_regression](https://github.com/Starry-OvO/Rotate-Captcha-Crack/blob/master/figures/RotNet-angle_error_regression.png)

- RotationLoss

![This-RotationLoss](https://github.com/Starry-OvO/Rotate-Captcha-Crack/blob/master/figures/This-RotationLoss.png)
