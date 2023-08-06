# Rotate-Captcha-Crack

因为准备发论文，故优先更新英文文档，中文文档仅保留使用方法就不介绍原理了

中文 | [English](https://github.com/Starry-OvO/rotate-captcha-crack)

CNN预测图片旋转角度，可用于破解旋转验证码。

测试效果：

![test_result](https://user-images.githubusercontent.com/48282276/224320691-a8eefd23-392b-4580-a729-7869fa237eaa.png)

本仓库实现了三类模型：

| 名称        | Backbone          | 跨域测试误差（越小越好） | 参数量  | FLOPs  |
| ----------- | ----------------- | ------------------------ | ------- | ------ |
| RotNet      | ResNet50          | 71.7920°                 | 24.246M | 4.132G |
| RotNetR     | RegNetY 3.2GFLOPs | 19.1594°                 | 18.468M | 3.223G |
| RCCNet_v0_5 | RegNetY 3.2GFLOPs | 42.7774°                 | 17.923M | 3.223G |

`RotNet`为[`d4nst/RotNet`](https://github.com/d4nst/RotNet/blob/master/train/train_street_view.py)的PyTorch实现。`RotNetR`仅在`RotNet`的基础上替换了backbone，并将分类数减少至180。其在[谷歌街景数据集](https://www.crcv.ucf.edu/data/GMCP_Geolocalization/)上训练64个epoch（耗时2小时）得到的平均预测误差为`19.1594°`。

跨域测试使用[谷歌街景](https://www.crcv.ucf.edu/data/GMCP_Geolocalization/)/[Landscape-Dataset](https://github.com/yuweiming70/Landscape-Dataset)作为训练集，百度验证码作为测试集（感谢@xiangbei1997）。

演示用到的百度验证码图片来自[RotateCaptchaBreak](https://github.com/chencchen/RotateCaptchaBreak/tree/master/data/baiduCaptcha)。

## 体验已有模型

### 准备环境

+ 支持CUDA10+的计算设备（如需训练则显存还需要不少于4G）

+ 确保你的`Python`版本`>=3.8 <3.12`

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

下载[Release](https://github.com/Starry-OvO/rotate-captcha-crack/releases)中的压缩包并解压到`./models`文件夹下。

文件目录结构类似`./models/RCCNet_v0_5/230228_20_07_25_000/best.pth`。

本项目仍处于beta阶段，模型名称会频繁发生变更，因此出现任何`FileNotFoundError`请先尝试用git回退到对应的tag。

### 输入一个验证码图像并查看旋转效果

如果你的系统没有GUI，尝试把debug方法从显示图像改成保存图像。

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

+ 我这里直接扒的[谷歌街景](https://www.crcv.ucf.edu/data/GMCP_Geolocalization/)和[Landscape-Dataset](https://github.com/yuweiming70/Landscape-Dataset)，你也可以自己收集一些风景照并放到一个文件夹里，图像没有尺寸要求

+ 在`train.py`里配置`dataset_root`变量指向装有图片的文件夹

+ 不需要手动标注，dataset会在读取图片的同时自动完成矩形裁剪、缩放旋转等工作

### 训练

```bash
python train_RotNetR.py
```

### 在测试集上验证模型

```bash
python test_RotNetR.py
```

## 相关文章

[吾爱破解 - 简单聊聊旋转验证码攻防](https://www.52pojie.cn/thread-1754224-1-1.html)
