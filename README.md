# Rotate-Captcha-Crack

[中文](README_zh-cn.md) | English

Predict the rotation angle of given picture through CNN. This project can be used for rotate-captcha cracking.

Test result:

![test_result](https://user-images.githubusercontent.com/48282276/224320691-a8eefd23-392b-4580-a729-7869fa237eaa.png)

Three kinds of model are implemented, as shown in the table below.

| Name    | Backbone       | Cross-Domain Loss (less is better) | Params  | MACs  |
| ------- | -------------- | ---------------------------------- | ------- | ----- |
| RotNet  | ResNet50       | 53.4684°                           | 24.246M | 4.09G |
| RotNetR | RegNet_Y_3_2GF | 6.5922°                            | 18.117M | 3.18G |

RotNet is the implementation of [`d4nst/RotNet`](https://github.com/d4nst/RotNet/blob/master/train/train_street_view.py) over PyTorch. `RotNetR` is based on `RotNet`, with [`RegNet_Y_3_2GF`](https://arxiv.org/abs/2101.00590) as its backbone and class number of 128. The average prediction error is `7.1818°`, obtained by 128 epochs of training (3.4 hours) on the [COCO 2017 (Unlabeled) Dataset](http://images.cocodataset.org/zips/unlabeled2017.zip).

The Cross-Domain Test uses [COCO 2017 (Unlabeled) Dataset](http://images.cocodataset.org/zips/unlabeled2017.zip) for training, and Captcha Pictures from Baidu (thanks to @xiangbei1997) for testing.

The captcha picture used in the demo above comes from [RotateCaptchaBreak](https://github.com/chencchen/RotateCaptchaBreak/tree/master/data/baiduCaptcha)

## Try it!

### Prepare

+ CUDA device with mem>=16G for training (reduce the batch size if necessary)

+ Python>=3.11,<3.15

+ PyTorch>=2.0

+ Clone the repository.

```shell
git clone https://github.com/lumina37/rotate-captcha-crack.git --depth 1
cd ./rotate-captcha-crack
```

+ Install all requiring dependencies.

This project strongly suggest you to use [`uv>=0.5.3`](https://docs.astral.sh/uv/) for package management. Run the following commands if you already have `uv`:

```shell
uv sync
```

Or, if you prefer `conda`: The following steps will create a virtual env under the working directory. You can also use a named env.

```shell
conda create -p .conda
conda activate ./.conda
conda install matplotlib tqdm
conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia
```

Or, if you prefer a direct `pip`:

```shell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install .
```

### Download the Pretrained Models

Download the `*.zip` files in [Release](https://github.com/lumina37/rotate-captcha-crack/releases) and unzip them all to the `./models` dir.

The directory structure will be like `./models/RotNetR/230228_20_07_25_000/best.pth`

The names of models will change frequently as the project is still in beta status. So, if any `FileNotFoundError` occurs, please try to rollback to the corresponding tag first.

### Test the Rotation Effect by a Single Captcha Picture

```shell
uv run test_captcha.py
```

Open `./debug.jpg` to check the result.

If you do not have `uv`, please use:

```shell
python test_captcha.py
```

### Use HTTP Server

+ Install extra dependencies

With `uv`:

```shell
uv pip install .[server]
```

or with `conda`:

```shell
conda install aiohttp
```

or with `pip`:

```shell
pip install .[server]
```

+ Launch server

```shell
uv run server.py
```

If you do not have `uv`, just use:

```shell
python server.py
```

+ Another Shell to Send Images

Use curl:

```shell
curl -X POST --data-binary @test.jpg http://127.0.0.1:4396
```

Or use Windows PowerShell:

```shell
irm -Uri http://127.0.0.1:4396 -Method Post -InFile test.jpg
```

## Train Your Own Model

### Prepare Datasets

+ For this project I'm using [Google Street View](https://www.crcv.ucf.edu/data/GMCP_Geolocalization/) and [Landscape-Dataset](https://github.com/yuweiming70/Landscape-Dataset) for training. You can collect some photos and leave them in one directory. Without any size or shape requirement.

+ Modify the `dataset_root` variable in `train.py`, let it points to the directory containing images.

+ No manual labeling is required. All the cropping, rotation and resizing will be done soon after the image is loaded.

### Train


```shell
uv run train_RotNetR.py
```

### Validate the Model on Test Set

```shell
uv run test_RotNetR.py
```

## Details of Design

Most of the rotate-captcha cracking methods are based on [`d4nst/RotNet`](https://github.com/d4nst/RotNet), with `ResNet50` as its backbone. `RotNet` regards the angle prediction as a classification task with 360 classes, then uses cross entropy to compute the loss.

Yet `KLDivLoss` with one-hot labels will bring a uniform metric distance between all angles (e.g. $\mathrm{dist}(1°, 2°) = \mathrm{dist}(1°, 180°)$ ), clearly defies the common sense. *[Arbitrary-Oriented Object Detection with Circular Smooth Label (ECCV'20)](https://www.researchgate.net/publication/343636147_Arbitrary-Oriented_Object_Detection_with_Circular_Smooth_Label)* introduces an interesting trick, by smoothing the one-hot label, e.g. `[0,1,0,0] -> [0.1,0.8,0.1,0]`, CSL provides a loss measurement closer to our intuition, such that $\mathrm{dist}(1°,180°) \gt \mathrm{dist}(1°,3°)$.

Meanwhile, the [`angle_error_regression`](https://github.com/d4nst/RotNet/blob/a56ea59818bbdd76d4dd8d83b8bbbaae6a802310/utils.py#L30-L36) proposed by [d4nst/RotNet](https://github.com/d4nst/RotNet) is less effective. That's because when dealing with outliers, the gradient leads to a non-convergence result. It's better to use a `SmoothL1Loss` for regression.
