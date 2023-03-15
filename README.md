# Teeth Segmentation Using UNet and Its Variants

This repository contains the code for training and evaluating various UNet models for teeth segmentation. The models implemented include the original UNet, as well as some of its variants such as UNet++, ResUNet, and Attention UNet.

## Data
The data used for this project is not publicly available, but you can request it by contacting me through the email address provided on the profile page. Once you have the data, make sure to organize it in the following directory structure:

```bash
data-directory
├── train
│   ├── images
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   ├── ...
│   ├── masks
│   │   ├── 1.bmp
│   │   ├── 2.bmp
│   │   ├── ...
├── val
│   ├── images
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   ├── ...
│   ├── masks
│   │   ├── 1.bmp
│   │   ├── 2.bmp
│   │   ├── ...
├── test
│   ├── images
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   ├── ...
│   ├── masks
│   │   ├── 1.bmp
│   │   ├── 2.bmp
│   │   ├── ...

```

## Usage
Before running the code, make sure to modify the config.py file to match your directory structure and preferences.

To train the model, simply run the train.py script:

```bash
python train.py 
```


