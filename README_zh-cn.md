# Rotate-Captcha-Crack

中文 | [English](https://github.com/lumina37/rotate-captcha-crack)

CNN预测图片旋转角度，可用于破解旋转验证码。

测试效果：

![test_result](https://user-images.githubusercontent.com/48282276/224320691-a8eefd23-392b-4580-a729-7869fa237eaa.png)

本仓库实现了三类模型：

| 名称    | Backbone       | 跨域测试误差（越小越好） | 参数量  | MACs  |
| ------- | -------------- | ------------------------ | ------- | ----- |
| RotNet  | ResNet50       | 53.4684°                 | 24.246M | 4.09G |
| RotNetR | RegNet_Y_3_2GF | 6.5922°                  | 18.117M | 3.18G |

`RotNet`为[`d4nst/RotNet`](https://github.com/d4nst/RotNet/blob/master/train/train_street_view.py)的PyTorch实现。`RotNetR`仅在`RotNet`的基础上将backbone替换为[`RegNet_Y_3_2GF`](https://arxiv.org/abs/2101.00590)，并将分类数减少至128。其在[COCO 2017 (Unlabeled) 数据集](https://pan.baidu.com/s/1iAZmJkaq_raJdKJDVLe6rQ?pwd=fsn9)上训练128个epoch（耗时3.4小时）得到的平均预测误差为`7.1818°`。

跨域测试使用[COCO 2017 (Unlabeled) 数据集](https://pan.baidu.com/s/1iAZmJkaq_raJdKJDVLe6rQ?pwd=fsn9)作为训练集，百度验证码作为测试集（感谢@xiangbei1997）。

演示用到的百度验证码图片来自[RotateCaptchaBreak](https://github.com/chencchen/RotateCaptchaBreak/tree/master/data/baiduCaptcha)。

## 体验已有模型

### 准备环境

+ 内存大小不少于16G的CUDA设备（如显存不足请酌减batch size）

+ 确保你的`Python`版本`>=3.11,<3.15`

+ 确保你的`PyTorch`版本`>=2.0`

+ 拉取代码

```shell
git clone https://github.com/lumina37/rotate-captcha-crack.git --depth 1
cd ./rotate-captcha-crack
```

+ 安装依赖

强烈推荐使用[`uv>=0.5.3`](https://docs.astral.sh/uv/)作为包管理工具。如果你已经安装了`uv`，请执行以下命令：

```shell
uv sync
```

或者，如果你喜欢用`conda`：以下步骤会在项目文件夹下创建一个虚拟环境。你也可以使用具名环境。

```shell
conda create -p .conda
conda activate ./.conda
conda install matplotlib tqdm
conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia
```

或者，如果你喜欢直接使用`pip`：

```shell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install .
```

### 下载预训练模型

下载[Release](https://github.com/lumina37/rotate-captcha-crack/releases)中的压缩包并解压到`./models`文件夹下。

文件目录结构类似`./models/RotNetR/230228_20_07_25_000/best.pth`。

本项目仍处于beta阶段，模型名称会频繁发生变更，因此出现任何`FileNotFoundError`请先尝试用git回退到对应的tag。

### 输入一个验证码图像并查看旋转效果

```shell
uv run test_captcha.py
```

打开`./debug.jpg`查看结果。

如果你没有安装`uv`的话，请使用：

```shell
python test_captcha.py
```

### 使用http服务端

+ 安装额外依赖

使用`uv`：

```shell
uv pip install .[server]
```

或者使用`conda`：

```shell
conda install aiohttp
```

或者使用`pip`：

```shell
pip install .[server]
```

+ 运行服务端

使用`uv`：

```shell
uv run server.py
```

如果你没有安装`uv`的话，请使用：

```shell
python server.py
```

+ 另开一命令行窗口发送图像

使用curl:

```shell
curl -X POST --data-binary @test.jpg http://127.0.0.1:4396
```

或使用Windows PowerShell:

```shell
irm -Uri http://127.0.0.1:4396 -Method Post -InFile test.jpg
```

## 训练新模型

### 准备数据集

+ 我这里直接扒的[谷歌街景](https://www.crcv.ucf.edu/data/GMCP_Geolocalization/)和[Landscape-Dataset](https://github.com/yuweiming70/Landscape-Dataset)，你也可以自己收集一些风景照并放到一个文件夹里，图像没有尺寸要求

+ 在`train.py`里配置`dataset_root`变量指向装有图片的文件夹

+ 不需要手动标注，dataset会在读取图片的同时自动完成矩形裁剪、缩放旋转等工作

### 训练

```shell
uv run train_RotNetR.py
```

### 在测试集上验证模型

```shell
uv run test_RotNetR.py
```

## 相关文章

[吾爱破解 - 简单聊聊旋转验证码攻防](https://www.52pojie.cn/thread-1754224-1-1.html)
