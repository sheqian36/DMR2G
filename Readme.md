This is the official repo for the paper DMR$^2$G: Diffusion Model for Radiology Report Generation

## Environment

Before running our code, you may setting the environments using the following lines.

```{bash}
conda create -n dmr2g python=3.8
pip install -r requirements.txt
```

## Datasets

We use two datasets (IU X-Ray and MIMIC-CXR) in our paper.

For `IU X-Ray`, you can download the dataset from [here](https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg/view?usp=sharing) and then put the files in `data/iu_xray`.

For `MIMIC-CXR`, you can download the dataset from [here](https://drive.google.com/file/d/1DS6NYirOXQf8qYieSVMvqNwuOlgAbM_E/view?usp=sharing) and then put the files in `data/mimic_cxr`. You can apply the dataset [here](https://drive.google.com/file/d/1DS6NYirOXQf8qYieSVMvqNwuOlgAbM_E/view?usp=sharing) with your license of [PhysioNet](https://physionet.org/content/mimic-cxr-jpg/2.0.0/).

## Download pretained model

You can download the models we trained for each dataset from [here](https://drive.google.com/drive/folders/17P-B-zdQIQ5dms6PvznDhNfTNlhIzjBq?usp=drive_link).

## Train

cd train_scripts and run `bash iu_xray.sh` to train a model on the IU X-Ray data.

cd train_scripts and run `mimic_cxr.sh` to train a model on the MIMIC-CXR data.

## Test

cd inference_scripts and run `bash iu_xray.sh` to test a model on the IU X-Ray data.

cd inference_scripts and run `bash mimic_cxr.sh` to test a model on the MIMIC-CXR data.
