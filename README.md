# Rotate-Captcha-Crack

中文 | [English](README_EN.md)

CNN预测图片旋转角度，可用于破解百度旋转验证码

测试效果：

![test_result](https://user-images.githubusercontent.com/48282276/221872572-7dfc7fcc-5bda-43e8-bee7-3a55ffd6e8a9.png)

本仓库实现了三类模型：

| 名称        | Backbone          | 损失函数     | 跨域测试误差（越小越好） | 大小（MB） |
| ----------- | ----------------- | ------------ | ------------------------ | ---------- |
| RotNet      | ResNet50          | 交叉熵       | **1.3002°**              | 92.7       |
| RotNetR     | RegNetY 3.2GFLOPs | 交叉熵       | **1.3089°**              | 69.8       |
| RCCNet_v0_4 | RegNetY 3.2GFLOPs | MSE+余弦修正 | 44.8499°                 | 70.8       |

注：
- `RotNet`为[`d4nst/RotNet`](https://github.com/d4nst/RotNet/blob/master/train/train_street_view.py)的PyTorch实现
- `RotNetR`仅在`RotNet`的基础上替换了backbone，并将分类数减少至180。其在[谷歌街景数据集](https://www.crcv.ucf.edu/data/GMCP_Geolocalization/)上训练9个epoch（耗时30min）得到的平均预测误差为`1.3089°`
- 跨域测试使用[谷歌街景](https://www.crcv.ucf.edu/data/GMCP_Geolocalization/)/[Landscape-Dataset](https://github.com/yuweiming70/Landscape-Dataset)作为训练集，百度验证码作为测试集（特别鸣谢@xiangbei1997）
- 演示用到的百度验证码图片来自[RotateCaptchaBreak](https://github.com/chencchen/RotateCaptchaBreak/tree/master/data/baiduCaptcha)

## 体验已有模型

### 准备环境

+ 支持CUDA10+的GPU（如需训练则显存还需要不少于4G）

+ 确保你的`Python`版本`>=3.8 <3.11`

+ 确保你的`PyTorch`版本`>=1.11`

+ 拉取代码并安装依赖库

```shell
git clone --depth=1 https://github.com/Starry-OvO/rotate-captcha-crack.git
cd ./rotate-captcha-crack
pip install .
```

注意不要漏了`install`后面那个`.`

+ 或者，使用虚拟环境

```shell
git clone --depth=1 https://github.com/Starry-OvO/rotate-captcha-crack.git
python -m venv ./rotate-captcha-crack --system-site-packages
cd ./rotate-captcha-crack
# 根据你的Shell类型挑选一个合适的脚本激活虚拟环境 例如./Script/Active.ps1
python -m pip install -U pip
pip install .
```

### 下载预训练模型

下载[Release](https://github.com/Starry-OvO/rotate-captcha-crack/releases)中的压缩包并解压到`./models`文件夹下

文件目录结构类似`./models/RCCNet_v0_4/230228_20_07_25_000/best.pth`

本项目仍处于beta阶段，模型名称会频繁发生变更，因此出现任何`FileNotFoundError`请先尝试用git回退到对应的tag

### 输入一个验证码图像并查看旋转效果

如果你的系统没有GUI，尝试把debug方法从显示图像改成保存图像

```bash
python test_captcha.py
```

### 使用http服务端

+ 安装额外依赖

```shell
pip install aiohttp httpx[cli]
```

+ 运行服务端

```shell
python server.py
```

+ 另开一命令行窗口发送图像

```shell
 httpx -m POST http://127.0.0.1:4396 -f img ./test.jpg
```

## 训练新模型

### 准备数据集

+ 我这里直接扒的谷歌街景和[Landscape-Dataset](https://github.com/yuweiming70/Landscape-Dataset)，你也可以自己收集一些风景照并放到一个文件夹里，图像没有尺寸要求

+ 在`train.py`里配置`dataset_root`变量指向装有图片的文件夹

+ 不需要手动标注，dataset会在读取图片的同时自动完成矩形裁剪、缩放旋转等工作

### 训练

```bash
python train_RotNet.py
```

### 在测试集上验证模型

```bash
python test_RotNet.py
```

## 设计细节

现有的旋图验证码破解方法大多基于[`d4nst/RotNet`](https://github.com/d4nst/RotNet)，其backbone为`ResNet50`，将角度预测视作360分类问题，并计算交叉熵损失。

`RotNet`中使用的交叉熵损失会令 $1°$ 和 $359°$ 之间的度量距离接近一个类似 $358°$ 的较大值，这显然是一个违背常识的结果。它们之间的度量距离应当是一个类似 $2°$ 的极小值。

同时，[d4nst/RotNet](https://github.com/d4nst/RotNet)给出的[`angle_error_regression`](https://github.com/d4nst/RotNet/blob/a56ea59818bbdd76d4dd8d83b8bbbaae6a802310/utils.py#L30-L36)损失函数效果较差。这是因为该损失函数在应对离群值时的梯度将导致不收敛的结果，你可以在后续的损失函数比较中轻松理解这一点。

本人设计的回归损失函数`RotationLoss`在`MSELoss`的基础上加了个余弦约束项来缩小真实值的 $±k*360°$ 与真实值之间的度量距离。

$$ \mathcal{L}(dist) = {dist}^{2} + \lambda_{cos} (1 - \cos(2\pi*{dist})) $$

为什么这里使用`MSELoss`，因为自监督学习生成的`label`可以保证不含有任何离群值，因此损失函数设计不需要考虑离群值的问题，同时`MSELoss`不破坏损失函数的可导性。

该损失函数在整个实数域可导且几乎为凸，为什么说是几乎，因为当 $\lambda_{cos} \gt 0.25$ 时在 $predict = \pm 1$ 的地方会出现局部极小值。

最后直观比较一下`RotationLoss`和`angle_error_regression`的函数图像。

![loss](https://user-images.githubusercontent.com/48282276/223087577-fe054521-36c4-4665-9132-2ca7dd2270f8.png)
