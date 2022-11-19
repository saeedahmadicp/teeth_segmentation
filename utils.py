import torch
from dataset import Teeth_Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms as t

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


def evaluate(preds, targets):
    """ 
      Returns specificty, precision, recall and f1_score   

    """

    confusion_vector = preds / targets
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    ### precision, recall, f1_score and specificity
    specificity = true_negatives / (true_negatives + false_positives)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = (2.0 * (recall*precision)) / (recall + precision)

    dict = {
        'specificity': specificity,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }

    return dict

def dice_coeff(pred, target):
        smooth = 1.
        num = pred.size(0)
        m1 = pred.view(num, -1)  # Flatten
        m2 = target.view(num, -1)  # Flatten
        intersection = (m1 * m2).sum()

        return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

def check_accuracy(loader, model, device="cuda", threshold=0.5, test=False):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()
    if test:
        f1_score, precision, recall, specificity = 0.0, 0.0 , 0.0 , 0.0

    with torch.no_grad():
        for _, (x, y) in enumerate(loader):
            
            x = x.to(device)
            y = y.to(device) #.unsqueeze(1)

            ## for unet plus plus  
            preds = torch.sigmoid((model(x)))

            # for unet

            preds = (preds > threshold).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += dice_coeff(preds, y)

            if test:
                temp_dict = evaluate(preds, y)
                f1_score += temp_dict['f1_score']
                precision += temp_dict['precision']
                recall += temp_dict['recall']
                specificity += temp_dict['specificity']



        

        accuracy = num_correct/num_pixels*100
        dice_score = (dice_score/(len(loader)))*100
        
        accuracy, dice_score = accuracy.detach().cpu().item() , dice_score.detach().cpu().item() 

    if test: 
        f1_score = (f1_score/(len(loader)))*100
        precision = (precision/(len(loader)))*100
        recall = (recall/(len(loader)))*100
        specificity = (specificity/(len(loader)))*100

        dict = {
                'specificity': specificity,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'accuracy':accuracy,
                'dice_score': dice_score,
                }
        return dict

    else:
        print(f"Got {num_correct}/{num_pixels} with acc {accuracy:.2f}" )
        print(f"Dice score: {dice_score :.2f}")
    
        return accuracy, dice_score


def validation_loss(model, validation_dl, loss_fn, device):
    total_loss = 0.0

    for x, y in validation_dl:
        x, y = x.to(device), y.to(device)


        preds = model(x)
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

        loss = loss_fn(predictions, targets)
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


def visualize_random_image(model, loader, device, threshold, width, height):

    rand_batch = torch.randint(0, len(loader), (1,)).item()
    
    for batch, (x, y) in enumerate(loader): 

        if batch == rand_batch:
            x = x.to(device)
            y = y.to(device) #.unsqueeze(1)

            preds = torch.sigmoid((model(x)))

            preds = (preds > threshold).float() * 255.0
            y = y * 255.0


            preds = preds[0].view(height, width)
            y = y[0].view(height, width)

            y, preds = y.detach().cpu(), preds.detach().cpu()

            
            figure = plt.figure(figsize=(4,4))
            plt.title(f'test image plot batch size {rand_batch}, first sample. (orignal, predictions)')
            figure.add_subplot(1,2, 1)
            plt.imshow(y)
            figure.add_subplot(1,2, 2)
            plt.imshow(preds)

            plt.savefig(f'batch_{rand_batch}_sample_0 (orignal, predictions).png')
            plt.show()

            

