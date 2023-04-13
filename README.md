# Teeth Segmentation Using UNet and Its Variants

This repository contains the code for training and evaluating various UNet models for teeth segmentation. The models implemented include the original UNet, as well as some of its variants such as UNet++, ResUNet, and Attention UNet.

## Data
The data used for this project is not publicly available, but you can request it by contacting me through the email address provided on the profile page. Once you have the data, make sure to update the paths accordingly. 

## Usage
Before running the code, make sure to modify the train.py file to match your customized settings and model executation.

To train the model, simply run the train.py script:

```bash
python train.py 
```

## Results for the test data

| UNet Variants | Test Accurary | Test Dice Score |
|----------|----------|----------|
| Base UNet                 |  96.10         |   90.47       |
| UNet with GN              |  96.71         |   91.53       |
| Attention UNet            |  96.40         |   91.01       |
| Spatial Attention UNet    |  96.45         |   91.09       |
| Inception UNet            |  96.29         |   90.69       |
| Residual UNet             |  96.16         |   90.06      |
| UNet++                    |  96.11         |   90.33      |
| Dense UNet with GN        |  96.77         |   91.88       |
| **Spatial Attention UNet2 ${\color{red}\^*}$**  |  **97.32**        |  **93.12**        |

${\color{red}\*}$ increase the resolution from 256\*256 to 768\*512, reduce the batch size from 16 to 2, used Group Normalization and  Custom spatial attention module with base UNet 




