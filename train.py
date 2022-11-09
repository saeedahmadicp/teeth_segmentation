import torch
from torchvision import transforms as t
import torch.nn as nn
import torch.optim as optim
import numpy as np


from utils import get_loaders, Fit
from model import UNET, Attention_UNET
from focal_loss import FocalLoss

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

LEARNING_RATE = 1e-4
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 30
IMAGE_HEIGHT = 256 # 1127 originally
IMAGE_WIDTH = 256 # 1991 originally
TRAIN_IMG_DIR = "./train-val/train2018/"
TRAIN_MASK_DIR = "./train-val/masks/"
TEST_IMG_DIR = "./test/test2018/" 
TEST_MASK_DIR = "./test/mask/"




def main():

    

    train_images_transform = t.Compose(
        [
            t.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
            t.ToTensor(),
            t.Normalize(
                mean = [0.477, 0.451, 0.411],
                std = [0.284, 0.280, 0.292],
            ),
            
        ]
    )

    train_masks_transform = t.Compose(
        [
            t.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
            t.ToTensor(),
        ]
    )

    test_images_transform, test_masks_transform =  train_images_transform, train_masks_transform

    train_dl, test_dl = get_loaders(
        train_dir= TRAIN_IMG_DIR,
        train_maskdir= TRAIN_MASK_DIR,
        test_dir= TEST_IMG_DIR,
        test_maskdir= TEST_MASK_DIR,
        batch_size= BATCH_SIZE,
        train_images_transform= train_images_transform,
        train_masks_transform= train_masks_transform,
        test_images_transform= test_images_transform,
        test_masks_transform= test_masks_transform,
    )

    print("Simple Unet")
    model = UNET(in_channels=3, out_channels=1)
    model.to(device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    Fit(model=model,train_dl=train_dl, test_dl=test_dl, loss_fn=loss_fn, optimizer=optimizer, epochs=NUM_EPOCHS, device=DEVICE)
    
    print("Attention U-Net")
    model = Attention_UNET(in_channels=3, out_channels=1)
    model.to(device=DEVICE)
    Fit(model=model,train_dl=train_dl, test_dl=test_dl, loss_fn=loss_fn, optimizer=optimizer, epochs=NUM_EPOCHS, device=DEVICE)




if __name__ == "__main__":
    main()