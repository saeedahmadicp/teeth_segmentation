import torch
from torchvision import transforms as t
import torch.nn as nn
import torch.optim as optim
from model import UNET
import numpy as np


from utils import get_loaders, check_accuracy,  train_fn, plot_graph, check_test_accuracy


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

    train_accuracies = []
    validation_accuracies = []
    train_dice_scores = []
    validation_dice_scores = []
    train_losses = []
    epochs = np.arange(0, NUM_EPOCHS, 1).tolist()

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

    model = UNET(in_channels=3, out_channels=1)
    model.to(device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-7)

    print("Training started ::: **************** ")
    for epoch in range(NUM_EPOCHS):
        print("\nEpoch: ", epoch)
        train_loss = train_fn(
            loader=train_dl,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=DEVICE,
        )
        ## Training accuracy
        train_accuracy, train_ds = check_accuracy(
            loader=train_dl,
            model=model,
            device=DEVICE,
            validation= False,
        )

        ## Validation accuracy
        validation_accuracy, validation_ds = check_accuracy(
            loader=train_dl,
            model=model,
            device=DEVICE,
            validation= True,
        )
        
        train_accuracies.append(train_accuracy)
        validation_accuracies.append(validation_accuracy)

        train_dice_scores.append(train_ds)
        validation_dice_scores.append(validation_ds)

        train_losses.append(train_loss)


    
    print("Done")

    check_test_accuracy(
        loader= test_dl,
        model=model,
        device=DEVICE,
    )

    plot_graph(
            x=epochs,
            y=train_losses,
            x_label= "No. of Epochs",
            y_label= "Train Losses",
            title= "Training losses vs no. of epochs",
        )

    plot_graph(
            x=epochs,
            y=train_accuracies,
            x_label= "No. of Epochs",
            y_label= "Train Accuracies",
            title= "Training accuracies vs no. of epochs",
        )
    plot_graph(
            x=epochs,
            y= train_dice_scores,
            x_label= "No. of Epochs",
            y_label= "Training dice scores",
            title= "Training dice scores vs no. of epochs",
        )
    plot_graph(
            x=epochs,
            y= validation_accuracies,
            x_label= "No. of Epochs",
            y_label= "validation dice scores",
            title= "validation accuracies vs no. of epochs",
        )
    plot_graph(
            x=epochs,
            y= validation_dice_scores,
            x_label= "No. of Epochs",
            y_label= "validation dice scores",
            title="validation dice scores vs no. of epochs",
        )

if __name__ == "__main__":
    main()