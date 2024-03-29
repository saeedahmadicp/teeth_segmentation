import torch
from torchvision import transforms as t
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter


from utils import get_loaders, Fit, check_accuracy, plot_history, visualize_random_image
from models import UNET, Attention_UNET, Inception_UNET, Inception_Attention_UNET, ResUNET, ResUNETPlus, ResUNET_with_GN, ResUNET_with_CBAM, UNET_GN, CustomAttention_UNET
from models.unetplusplus import  NestedUNet as UNET_Plus
from dataset import split_data, split_category
#from focal_loss import FocalLoss
from lookahead import Lookahead
from models.dense_unet import DenseUNet
from losses import DiceBCELossLogitsLoss



import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

LEARNING_RATE = 1e-4
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
NUM_EPOCHS = 30
IMAGE_HEIGHT = 512 #256 # 1127 originally
IMAGE_WIDTH = 768 #256 # 1991 originally
TRAIN_IMG_DIR = "./train-val/train2018/"
TRAIN_MASK_DIR = "./train-val/masks/"
TEST_IMG_DIR = "./test/test2018/" 
TEST_MASK_DIR = "./test/mask/"
MODEL_PATH = "./saved_models/customSpatialAttentionUnet2.pth"



def main():

    
    ## transforms for train images
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

    ## transforms for train masks
    train_masks_transform = t.Compose(
        [
            t.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
            t.ToTensor(),
        ]
    )

    ## transforms for test images and masks
    test_images_transform, test_masks_transform =  train_images_transform, train_masks_transform

    ##spliting the data into train, validation and test subset
    cat_wise_images, cat_wise_masks = split_category(TEST_IMG_DIR, TEST_MASK_DIR)
    data_dict = split_data(cat_wise_images, cat_wise_masks, test_train_ratio=0.7, train_valid_ratio=0.9)

    train_dl, validation_dl, test_dl = get_loaders(
        #train_dir= TRAIN_IMG_DIR,
        #train_maskdir= TRAIN_MASK_DIR,
        images_dir= TEST_IMG_DIR,
        masks_dir= TEST_MASK_DIR,
        batch_size= BATCH_SIZE,
        train_images_transform= train_images_transform,
        train_masks_transform= train_masks_transform,
        test_images_transform= test_images_transform,
        test_masks_transform= test_masks_transform,
        data_dict = data_dict,
    )

    
    #loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = DiceBCELossLogitsLoss()

    
    print("CustomAttentionwithGN_DL")
   # writer = SummaryWriter("runs/CustomAttentionwithGN_DL3")  
    model = CustomAttention_UNET(in_channels=3, out_channels=1)
    model.to(device=DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE,)
    #lookahead = Lookahead(optimizer, k=3, alpha=0.6) 
    history = Fit(model=model,train_dl=train_dl, validation_dl=validation_dl, loss_fn=loss_fn, optimizer=optimizer, epochs=NUM_EPOCHS, device=DEVICE, writer=writer)

    torch.save(model.state_dict(),MODEL_PATH )
    #model.load_state_dict(torch.load(MODEL_PATH))

    visualize_random_image(model=model, loader=test_dl, device=DEVICE, threshold=0.85, width=IMAGE_WIDTH, height=IMAGE_HEIGHT)

    dict = check_accuracy(test_dl, model, device=DEVICE, threshold=0.85, test=True)

    print(f"\n\ntest_accuracy:  {dict['accuracy']:.2f}")
    print(f"test dice score:  {dict['dice_score']:.2f}")
    print(f"test precision:  {dict['precision']:.2f}")
    print(f"test recall:  {dict['recall']:.2f}")
    print(f"test specificity:  {dict['specificity']:.2f}")
    print(f"test f1_score:  {dict['f1_score']:.2f}")

    
    ### ploting graphs
   # plot_history(history)

    print("Completed")
   


if __name__ == "__main__":
    main()


