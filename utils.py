import torch
from dataset import Teeth_Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt



def get_loaders(
    train_dir,
    train_maskdir,
    test_dir,
    test_maskdir,
    batch_size,
    train_images_transform,
    train_masks_transform,
    test_images_transform,
    test_masks_transform,
    ):

    train_ds = Teeth_Dataset(
        images_dir = train_dir, 
        masks_dir = train_maskdir,
        transform = train_images_transform, 
        target_transform = train_masks_transform)

    test_ds = Teeth_Dataset(
        images_dir = test_dir, 
        masks_dir = test_maskdir,
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
    
    return train_dl, test_dl

def dice_coeff(pred, target):
        smooth = 1.
        num = pred.size(0)
        m1 = pred.view(num, -1)  # Flatten
        m2 = target.view(num, -1)  # Flatten
        intersection = (m1 * m2).sum()

        return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

def check_accuracy(loader, model, device="cuda", validation=True):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        if validation:
            for idx, (x, y) in enumerate(loader):
                if idx >= 14:
                    x = x.to(device)
                    y = y.to(device) #.unsqueeze(1)
                
                    preds = torch.sigmoid(model(x))
                
                    preds = (preds > 0.5).float()
                    num_correct += (preds == y).sum()
                    num_pixels += torch.numel(preds)
                    dice_score += dice_coeff(preds, y)
                    #dice_score += (2 * (preds * y).sum()) / ( (preds + y).sum() + 1e-8)
            
            accuracy = num_correct/num_pixels*100
            dice_score = (dice_score/4)*100

            print("\nResults for validation data: ")
            print(
                f"Got {num_correct}/{num_pixels} with acc {accuracy:.2f}"
            )
            print(f"Dice score: {dice_score :.2f}")

        else:
            for idx, (x, y) in enumerate(loader):
                if idx < 14:
                    x = x.to(device)
                    y = y.to(device) #.unsqueeze(1)
                
                    preds = torch.sigmoid(model(x))
                
                    preds = (preds > 0.5).float()
                    num_correct += (preds == y).sum()
                    num_pixels += torch.numel(preds)
                    dice_score += dice_coeff(preds, y)
                    #dice_score += (2 * (preds * y).sum()) / ( (preds + y).sum() + 1e-8)
            
            accuracy = num_correct/num_pixels*100
            dice_score = (dice_score/14)*100

            print("\nResults for Training data: ")
            print(
                f"Got {num_correct}/{num_pixels} with acc {accuracy:.2f}"
            )
            print(f"Dice score: {dice_score :.2f}")

        accuracy, dice_score = accuracy.detach().cpu().item() , dice_score.detach().cpu().item() 

    return accuracy, dice_score


    
def train_fn(loader, model, optimizer, loss_fn, device):
    mean_loss = 0
    for batch_idx, (data, targets) in enumerate(loader):
        if batch_idx <= 14:
            data = data.to(device=device)
            targets = targets.to(device=device)

            optimizer.zero_grad()

            # forward   
            predictions = model(data)

            #print("Prediction shape: ", predictions.shape)
            loss = loss_fn(predictions, targets)

            # backward
            loss.backward()
            optimizer.step()

            mean_loss += loss.detach().cpu().item()
    return mean_loss/14.0

def plot_graph(x, y, x_label, y_label, title):

    plt.title(title)
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(f'{title}.png')
    plt.show()


def check_test_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for idx, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device) #.unsqueeze(1)
                
            preds = torch.sigmoid(model(x))
                
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += dice_coeff(preds, y)
            #dice_score += (2 * (preds * y).sum()) / ( (preds + y).sum() + 1e-8)
            
        accuracy = num_correct/num_pixels*100
        dice_score = (dice_score/(len(loader)))*100

        print("\nResults for test data: ")
        print(
            f"Got {num_correct}/{num_pixels} with acc {accuracy:.2f}"
        )
        print(f"Dice score: {dice_score :.2f}")


    