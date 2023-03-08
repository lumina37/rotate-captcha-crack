# Rotate-Captcha-Crack

[中文](https://github.com/Starry-OvO/rotate-captcha-crack) | English

Predict the rotation angle of given picture through CNN. It can be used for rotate-captcha cracking.

Test result:

![test_result](https://user-images.githubusercontent.com/48282276/221872572-7dfc7fcc-5bda-43e8-bee7-3a55ffd6e8a9.png)

Three kinds of models are implemented, as shown below.

| Name        | Backbone          | Loss                       | Cross-Domain Loss (less is better) | Size (MB) |
| ----------- | ----------------- | -------------------------- | ---------------------------------- | --------- |
| RotNet      | ResNet50          | CrossEntropy               | 1.1548°  **                        | 92.7      |
| RotNetR     | RegNetY 3.2GFLOPs | CrossEntropy               | 1.2825°                            | 69.8      |
| RCCNet_v0_5 | RegNetY 3.2GFLOPs | MSE with Cosine-Correction | 44.8499°                           | 70.8      |

Note:
- RotNet is the implementation of [`d4nst/RotNet`](https://github.com/d4nst/RotNet/blob/master/train/train_street_view.py) over PyTorch.
- `RotNetR` is based on `RotNet`. It just renew the backbone and reduce the class number to 180. It's average prediction error is `1.2825°`, obtained by 64 epochs of training (2hours) on the [Google Street View](https://www.crcv.ucf.edu/data/GMCP_Geolocalization/) dataset.
- About the Cross-Domain Test: [Google Street View](https://www.crcv.ucf.edu/data/GMCP_Geolocalization/) and [Landscape-Dataset](https://github.com/yuweiming70/Landscape-Dataset) for training, and Captcha Pictures from Baidu for testing (special thx to @xiangbei1997)
- The captcha picture used in the demo above comes from [RotateCaptchaBreak](https://github.com/chencchen/RotateCaptchaBreak/tree/master/data/baiduCaptcha)

## Try it!

### Prepare

+ GPU supporting CUDA10+ (mem>=4G for training)

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

The directory structure will be like `./models/RCCNet_v0_5/230228_20_07_25_000/best.pth`

The models' names will change frequently as the project is still in beta status. So, if any `FileNotFoundError` occurs, please try to rollback to the corresponding tag first.

### Test the Rotation Effect by a Single Captcha Picture

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

+ For this project I'm using [Google Street View](https://www.crcv.ucf.edu/data/GMCP_Geolocalization/) and [Landscape-Dataset](https://github.com/yuweiming70/Landscape-Dataset) for training. You can collect some photos and leave them in one directory. Without any size or shape requirement.

+ Modify the `dataset_root` variable in `train.py`, let it points to the directory containing images.

+ No manual labeling is required. All the cropping, rotation and resizing will be done soon after the image is loaded.

### Train

```bash
python train_RotNetR.py
```

### Validate the Model on Test Set

```bash
python test_RotNetR.py
```

## Details of Design

Most of the rotate-captcha cracking methods are based on [`d4nst/RotNet`](https://github.com/d4nst/RotNet), with `ResNet50` as its backbone. `RotNet` treat the angle prediction as a classification task with 360 classes, then use `CrossEntropy` to compute the loss.

Yet `CrossEntropy` will bring a sizeable metric distance of about $358°$ between $1°$ and $359°$, clearly defies common sense, it should be a small value like $2°$. Meanwhile, the [`angle_error_regression`](https://github.com/d4nst/RotNet/blob/a56ea59818bbdd76d4dd8d83b8bbbaae6a802310/utils.py#L30-L36) given by [d4nst/RotNet](https://github.com/d4nst/RotNet) is less effective. That's because when dealing with outliers, the gradient will lead to a non-convergence result. You can easily understand this through the subsequent comparison between loss functions.

My regression loss function `RotationLoss` is based on `MSELoss`, with an extra cosine-correction to decrease the metric distance between $±k*360°$.

$$ \mathcal{L}(dist) = {dist}^{2} + \lambda_{cos} (1 - \cos(2\pi*{dist})) $$

Why `MSELoss` here? Because the `label` generated by 
self-supervised method is guaranteed not to contain any outliers. So our design does not need to consider the outliers. Also, `MSELoss` won't break the derivability of loss function.

The loss function is derivable and *almost* convex over the entire $\mathbb{R}$. Why say *almost*? Because there will be local minimum at $predict = \pm 1$ when $\lambda_{cos} \gt 0.25$.

Finally, let's take a look at the figure of two loss functions:

![loss](https://user-images.githubusercontent.com/48282276/223087577-fe054521-36c4-4665-9132-2ca7dd2270f8.png)
