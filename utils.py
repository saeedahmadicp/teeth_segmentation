import torch
from dataset import Teeth_Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


def get_loaders(
    images_dir,
    masks_dir,
    batch_size,
    train_images_transform,
    train_masks_transform,
    test_images_transform,
    test_masks_transform,
    data_dict,
    ):

    train_ds = Teeth_Dataset(
        images_dir = images_dir, 
        masks_dir = masks_dir,
        data_dict=data_dict,
        data_type='train',
        transform = train_images_transform, 
        target_transform = train_masks_transform)

    validation_ds = Teeth_Dataset(
        images_dir = images_dir, 
        masks_dir = masks_dir,
        data_dict=data_dict,
        data_type='validation',
        transform = test_images_transform, 
        target_transform = test_masks_transform)
    
    test_ds = Teeth_Dataset(
        images_dir = images_dir, 
        masks_dir = masks_dir,
        data_dict=data_dict,
        data_type='test',
        transform = test_images_transform, 
        target_transform = test_masks_transform)

    train_dl = DataLoader(
        dataset = train_ds,
        batch_size = batch_size,
        shuffle = True, 
        )
    
    test_dl = DataLoader(
        dataset = test_ds,
        batch_size = batch_size,
        shuffle = False, 
        )
    validation_dl = DataLoader(
        dataset = validation_ds,
        batch_size = batch_size,
        shuffle = False, 
        )
    
    return train_dl, validation_dl, test_dl

def dice_coeff(pred, target):
        smooth = 1.
        num = pred.size(0)
        m1 = pred.view(num, -1)  # Flatten
        m2 = target.view(num, -1)  # Flatten
        intersection = (m1 * m2).sum()

        return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

def check_accuracy(loader, model, device="cuda", threshold=0.5):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for _, (x, y) in enumerate(loader):
            
            x = x.to(device)
            y = y.to(device) #.unsqueeze(1)

            ## for unet plus plus  
            preds = torch.sigmoid((model(x)[-1]))

            # for unet

            preds = (preds > threshold).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += dice_coeff(preds, y)
            
        accuracy = num_correct/num_pixels*100
        dice_score = (dice_score/(len(loader)))*100

        print(
             f"Got {num_correct}/{num_pixels} with acc {accuracy:.2f}"
            )
        print(f"Dice score: {dice_score :.2f}")


        accuracy, dice_score = accuracy.detach().cpu().item() , dice_score.detach().cpu().item() 

    return accuracy, dice_score


def validation_loss(model, validation_dl, loss_fn, device):
    total_loss = 0.0

    for x, y in validation_dl:
        x, y = x.to(device), y.to(device)

        ## for unet plus plus 
        preds = model(x)[-1]
        loss = loss_fn(preds, y)

        total_loss += loss.detach().cpu().item()
    
    return total_loss/len(validation_dl)

    
def train_fn(train_dl, model, optimizer, loss_fn, device):
    mean_loss = 0
   
    for _, (data, targets) in enumerate(train_dl):
        
        data = data.to(device=device)
        targets = targets.to(device=device)

        optimizer.zero_grad()
        # forward   
        predictions = model(data)
        #print("Prediction shape: ", predictions.shape)
        
        ### Only for UNET plus plus
        loss = 0.0
        for output in predictions:
            loss += loss_fn(output, targets)
        loss /= (len(predictions))

        #loss = loss_fn(predictions, targets)
        # backward
        loss.backward()
        optimizer.step()
        mean_loss += loss.detach().cpu().item()

    return mean_loss/(len(train_dl))


def Fit(model, train_dl, validation_dl, loss_fn, optimizer, epochs, device, writer):
    train_accuracies = []
    validation_accuracies = []
    train_dice_scores = []
    validation_dice_scores = []
    train_losses = []
    validation_losses = []
    
        
    print("Training started ::: **************** ")
    for epoch in range(epochs):
        print("\nEpoch: ", epoch)
        train_loss = train_fn(
            train_dl=train_dl,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )

        ## Training accuracy
        print("\nResults for Training data: ")
        train_accuracy, train_ds = check_accuracy(
            loader=train_dl,
            model=model,
            device=device,
            threshold=0.5,
        )

        ## Validation accuracy
        print("\nResults for Validation data: ")
        validation_accuracy, validation_ds = check_accuracy(
            loader=validation_dl,
            model=model,
            device=device,
            threshold=0.5,
        )

        validation_loss_ = validation_loss(model, validation_dl, loss_fn, device)

        writer.add_scalar('Training Loss', train_loss, epoch)
        writer.add_scalar('Validation Loss', validation_loss_, epoch)
        writer.add_scalar('Training Accuracy', train_accuracy, epoch)
        writer.add_scalar('Validation Accuracy', validation_accuracy, epoch)
        writer.add_scalar('Training Dice Score', train_ds, epoch)
        writer.add_scalar('Validation Dice Score', validation_ds, epoch)


        train_accuracies.append(train_accuracy)
        validation_accuracies.append(validation_accuracy)

        train_dice_scores.append(train_ds)
        validation_dice_scores.append(validation_ds)

        train_losses.append(train_loss)
        validation_losses.append(validation_loss_)

    history = {
        'model': model,
        'epochs': epochs,
        'train_losses':train_losses,
        'validation_losses': validation_losses,
        'train_accuracies': train_accuracies,
        'train_dice_scores':train_dice_scores,
        'validation_accuracies': validation_accuracies,
        'validation_dice_scores': validation_dice_scores
    }
    
    print("Done")

    return history

def plot_graph(x, y1, y2, x_label, y_label, title):

    plt.title(title)
    plt.plot(x, y1, '-b', label='train')
    plt.plot(x, y2, '-r', label='validation')
    plt.xlabel(x_label)
    plt.legend()
    #plt.ylabel(y_label)
    plt.savefig(f'{title}.png')
    plt.show()



def plot_history(history):
    epochs_list = np.arange(0, history['epochs'], 1).tolist()

    plot_graph(
            x = epochs_list,
            y1 = history['train_losses'], 
            y2 = history['validation_losses'],
            x_label= "n iterations", 
            y_label= "losses",
            title= "Iteration vs losses",
    )

    plot_graph(
            x = epochs_list,
            y1 = history['train_accuracies'], 
            y2 = history['validation_accuracies'],
            x_label= "n iterations", 
            y_label= "accuracies",
            title= "Iteration vs accuracies",
    )

    plot_graph(
            x = epochs_list,
            y1 = history['train_dice_scores'], 
            y2 = history['validation_dice_scores'],
            x_label= "n iterations", 
            y_label= "dice scores",
            title= "Iteration vs dice scores",
    )