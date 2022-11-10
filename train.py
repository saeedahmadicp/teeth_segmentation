import torch
from torchvision import transforms as t
import torch.nn as nn
import torch.optim as optim
import numpy as np


from utils import get_loaders, Fit
from model import UNET, Attention_UNET, Inception_UNET, Inception_Attention_UNET, ResUNET
from focal_loss import FocalLoss

from lookahead import Lookahead

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

    loss_fn = nn.BCEWithLogitsLoss()

    
    print("Deep Residual UNET")
    model = ResUNET(in_channels=3, out_channels=1)
    model.to(device=DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE,)
    #lookahead = Lookahead(optimizer, k=8, alpha=0.2) 
    Fit(model=model,train_dl=train_dl, test_dl=test_dl, loss_fn=loss_fn, optimizer=optimizer, epochs=NUM_EPOCHS, device=DEVICE)


if __name__ == "__main__":
    main()