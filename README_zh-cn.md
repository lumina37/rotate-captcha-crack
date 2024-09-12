# Rotate-Captcha-Crack

中文 | [English](https://github.com/lumina37/rotate-captcha-crack)

CNN预测图片旋转角度，可用于破解旋转验证码。

测试效果：

![test_result](https://user-images.githubusercontent.com/48282276/224320691-a8eefd23-392b-4580-a729-7869fa237eaa.png)

本仓库实现了三类模型：

| 名称        | Backbone          | 跨域测试误差（越小越好） | 参数量  | MACs  |
| ----------- | ----------------- | ------------------------ | ------- | ----- |
| RotNet      | ResNet50          | 75.6512°                 | 24.246M | 4.09G |
| RotNetR     | RegNetY 3.2GFLOPs | 15.1818°                 | 18.117M | 3.18G |
| RCCNet_v0_5 | RegNetY 3.2GFLOPs | 56.8515°                 | 20.212M | 3.18G |

`RotNet`为[`d4nst/RotNet`](https://github.com/d4nst/RotNet/blob/master/train/train_street_view.py)的PyTorch实现。`RotNetR`仅在`RotNet`的基础上替换了backbone，并将分类数减少至128。其在[谷歌街景数据集](https://www.crcv.ucf.edu/data/GMCP_Geolocalization/)上训练64个epoch（耗时3小时）得到的平均预测误差为`15.1818°`。

跨域测试使用[谷歌街景](https://www.crcv.ucf.edu/data/GMCP_Geolocalization/)/[Landscape-Dataset](https://github.com/yuweiming70/Landscape-Dataset)作为训练集，百度验证码作为测试集（感谢@xiangbei1997）。

演示用到的百度验证码图片来自[RotateCaptchaBreak](https://github.com/chencchen/RotateCaptchaBreak/tree/master/data/baiduCaptcha)。

## 体验已有模型

### 准备环境

+ 支持CUDA10+的计算设备（如需训练则显存还需要不少于4G）

+ 确保你的`Python`版本`>=3.8,<3.13`

+ 确保你的`PyTorch`版本`>=1.11`

+ 拉取代码

```shell
git clone https://github.com/lumina37/rotate-captcha-crack.git --depth=1
cd ./rotate-captcha-crack
```

+ 安装依赖

强烈推荐使用[`rye`](https://rye-up.com/)作为包管理工具。如果你已经安装了`rye`，请执行以下命令：

```shell
rye pin 3.12
rye sync
```

或者，如果你喜欢用`conda`：以下步骤会在项目文件夹下创建一个虚拟环境。你也可以使用具名环境。

```shell
conda create -p .conda
conda activate ./.conda
conda install matplotlib tqdm tomli
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
```

或者，如果你喜欢直接使用`pip`：

```shell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

### 下载预训练模型

下载[Release](https://github.com/lumina37/rotate-captcha-crack/releases)中的压缩包并解压到`./models`文件夹下。

文件目录结构类似`./models/RotNetR/230228_20_07_25_000/best.pth`。

本项目仍处于beta阶段，模型名称会频繁发生变更，因此出现任何`FileNotFoundError`请先尝试用git回退到对应的tag。

### 输入一个验证码图像并查看旋转效果

如果你的系统没有GUI，尝试把debug方法从显示图像改成保存图像。

```shell
rye run python test_captcha.py
```

如果你没有安装`rye`的话，去掉前缀的`rye run`即可。

### 使用http服务端

+ 安装额外依赖

使用`rye`：

```shell
rye sync --features=server
```

或者使用`conda`：

```shell
conda install aiohttp httpx[cli]
```

或者使用`pip`：

```shell
pip install -e .[server]
```

+ 运行服务端

使用`rye`：

```shell
rye run python server.py
```

或者使用其他：

```shell
python server.py
```

+ 另开一命令行窗口发送图像

```shell
rye run httpx -m POST http://127.0.0.1:4396 -f img ./test.jpg
```

## 训练新模型

### 准备数据集

+ 我这里直接扒的[谷歌街景](https://www.crcv.ucf.edu/data/GMCP_Geolocalization/)和[Landscape-Dataset](https://github.com/yuweiming70/Landscape-Dataset)，你也可以自己收集一些风景照并放到一个文件夹里，图像没有尺寸要求

+ 在`train.py`里配置`dataset_root`变量指向装有图片的文件夹

+ 不需要手动标注，dataset会在读取图片的同时自动完成矩形裁剪、缩放旋转等工作

### 训练

```shell
rye run python train_RotNetR.py
```

### 在测试集上验证模型

```shell
rye run python test_RotNetR.py
```

## 相关文章

[吾爱破解 - 简单聊聊旋转验证码攻防](https://www.52pojie.cn/thread-1754224-1-1.html)
