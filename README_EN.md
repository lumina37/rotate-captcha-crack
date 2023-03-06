# Rotate-Captcha-Crack

[中文](README.md) | English

Predict the rotation angle of given picture through CNN. It can be used for rotate-captcha cracking.

Test result:

![test_result](https://user-images.githubusercontent.com/48282276/221872572-7dfc7fcc-5bda-43e8-bee7-3a55ffd6e8a9.png)

Two kinds of models are implemented, as shown below.

| Name        | Backbone          | Loss                       | Cross-Domain Diff (less is better) | Size (MB) |
| ----------- | ----------------- | -------------------------- | ---------------------------------- | --------- |
| RotNet      | ResNet50          | CrossEntropy               | **1.3002°**                        | 92.7      |
| RCCNet_v0_4 | RegNetY 3.2GFLOPs | MSE with Cosine Correction | 44.9389°                           | 70.8      |

Note:
- RotNet is the implementation of [d4nst/RotNet](https://github.com/d4nst/RotNet/blob/master/train/train_street_view.py) with PyTorch. The average prediction error is `1.3002°`, obtained by 64 epochs of training (costs 2 hrs) on the [Google Street View](https://www.crcv.ucf.edu/data/GMCP_Geolocalization/) dataset.
- About the Cross-Domain Test: [Google Street View](https://www.crcv.ucf.edu/data/GMCP_Geolocalization/) and [Landscape-Dataset](https://github.com/yuweiming70/Landscape-Dataset) for training set, and Captcha Pictures from Baidu for testing set (special thx for @xiangbei1997 )
- The captcha picture used in the demo above comes from [RotateCaptchaBreak](https://github.com/chencchen/RotateCaptchaBreak/tree/master/data/baiduCaptcha)

## Try it!

### Prepare

+ GPU supporting CUDA10+ (mem>=4G if you wanna train your own model)

+ Python>=3.8 <3.11

+ PyTorch>=1.11

+ Clone the repository and install all requiring dependencies

```shell
git clone --depth=1 https://github.com/Starry-OvO/rotate-captcha-crack.git
cd ./rotate-captcha-crack
pip install .
```

**DONT** miss the `.` after `install`

+ Or, if you prefer `venv`

```shell
git clone --depth=1 https://github.com/Starry-OvO/rotate-captcha-crack.git
python -m venv ./rotate-captcha-crack --system-site-packages
cd ./rotate-captcha-crack
# Choose the proper script to acivate venv according to your shell type. e.g. ./Script/active*
python -m pip install -U pip
pip install .
```

### Download the Pretrained Models

Download the zip files in [Release](https://github.com/Starry-OvO/rotate-captcha-crack/releases) and unzip them to the `./models` dir.

The directory structure will be like `./models/RCCNet_v0_4/230228_20_07_25_000/best.pth`

The models' names will be modified frequently as the project is still in beta status. So, if any `FileNotFoundError` occurs, plz try to rollback to the corresponding tag first.

### Test the Rotation Effect by Inputting a Captcha Picture

If no GUI is presented, try to change the debugging behavior from showing images to saving them.

```bash
python test_captcha.py
```

### Use HTTP Server

+ Install extra dependencies

```shell
pip install aiohttp httpx[cli]
```

+ Launch server
  
```shell
python server.py
```

+ Another Shell to Send Images

```shell
 httpx -m POST http://127.0.0.1:4396 -f img ./test.jpg
```

## Train Your Own Model

### Prepare Datasets

+ For this project I'm using [Google Street View](https://www.crcv.ucf.edu/data/GMCP_Geolocalization/) and [Landscape-Dataset](https://github.com/yuweiming70/Landscape-Dataset) for training. You can collect some photos and leave them in one directory. Without any size requirement.

+ Modify the `dataset_root` variable in `train.py`, let it points to the directory containing images.

+ No manual labeling is required. All the cropping, rotation and resizing will be done soon after the image is loaded.

### Train

```bash
python train_RotNet.py
```

### Validate the Model on Testing-Dataset

```bash
python test.py
```

## 设计细节

现有的旋图验证码破解方法大多基于[`d4nst/RotNet`](https://github.com/d4nst/RotNet)，其backbone为`ResNet50`，将角度预测视作360分类问题，并计算交叉熵损失，本项目的`RCCNet`是对`RotNet`的简单改进

要特别注意，搜索RotNet搜出来的模型和论文是两码事，模型是我上面提到的`d4nst/RotNet`，论文是[*Unsupervised Representation Learning by Predicting Image Rotations (ICLR2018)*](https://arxiv.org/abs/1803.07728)，论文对应的开源仓库是[`FeatureLearningRotNet`](https://github.com/gidariss/FeatureLearningRotNet)。不要傻傻地用论文里的4/8分类来做旋转检测，那个是用来做内容分类的，在旋转角分类上效果很差，360分类才是做旋转角分类应该用的。下文的`RotNet`都是指代[`d4nst/RotNet`](https://github.com/d4nst/RotNet)

`RCCNet`的backbone是[`regnet (CVPR2020)`](https://arxiv.org/abs/2003.13678)的`RegNetY 3.2GFLOPs`

`RotNet`中使用的交叉熵损失会令 $1°$ 和 $359°$ 之间的度量距离接近一个类似 $358°$ 的较大值，这显然是一个违背常识的结果，它们之间的度量距离应当是一个类似 $2°$ 的极小值。而[d4nst](https://github.com/d4nst)给出的[`angle_error_regression`](https://github.com/d4nst/RotNet/blob/a56ea59818bbdd76d4dd8d83b8bbbaae6a802310/utils.py#L30-L36)损失函数效果较差，因为该损失函数在应对离群值时的梯度方向存在明显问题，你可以在后续的损失函数图像比对中轻松看出这一点

本人设计的损失函数`RotationLoss`和`angle_error_regression`的思路相近，我使用最后的全连接层预测出一个角度值并与`ground-truth`作差，然后在`MSELoss`的基础上加了个余弦约束项来缩小真实值的 $±k*360°$ 与真实值之间的度量距离

$$\mathcal{L}(dist) = {dist}^{2} + \lambda_{cos} (1 - \cos(2\pi*{dist})) $$

为什么这里使用`MSELoss`，因为自监督学习生成的`label`可以保证不含有任何离群值，因此损失函数设计不需要考虑离群值的问题，同时`MSELoss`不破坏损失函数的可导性

该损失函数在整个实数域可导且几乎为凸，为什么说是几乎，因为当 $\lambda_{cos} \gt 0.25$ 时在 $predict = \pm 1$ 的地方会出现局部极小

最后直观比较一下`RotationLoss`和`angle_error_regression`的函数图像

![loss](https://user-images.githubusercontent.com/48282276/223087577-fe054521-36c4-4665-9132-2ca7dd2270f8.png)
