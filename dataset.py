import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import re
from sklearn.model_selection import train_test_split

class Teeth_Dataset(Dataset):
    def __init__(self, images_dir, masks_dir,data_dict, data_type, transform=None, target_transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transfrom = transform
        self.target_transform = target_transform


        self.images = data_dict[f'{data_type}_images'] #os.listdir(images_dir)
        self.masks = data_dict[f'{data_type}_masks'] #os.listdir(masks_dir)


    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        img_path = os.path.join(self.images_dir, str(self.images[index]))
        mask_path = os.path.join(self.masks_dir, str(self.masks[index]))

        image = Image.open(img_path)
        mask = Image.open(mask_path)

        if self.transfrom:
            image = self.transfrom(image)
        
        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask

def split_category(images_path, masks_path):
    '''
    This function read all the images and split the images category wise
    '''
    a = os.listdir(images_path)
    b = os.listdir(masks_path)

    all_images = [image for image in a if image.endswith('.jpg')]
    all_masks = [mask for mask in b if mask.endswith('.bmp')]
    
    catogry_wise_images = [[] for _ in range(10)]
    catogry_wise_masks = [[] for _ in range(10)]

    for image, mask in zip(all_images, all_masks):
        image_cat = image.split('-')[0]
        image_cat = int((re.search(r"[0-9]+", image_cat)).group())
        catogry_wise_images[image_cat-1].append(image)
        catogry_wise_masks[image_cat-1].append(mask)
    return catogry_wise_images, catogry_wise_masks


def split_data(category_images, category_masks, test_train_ratio=0.7, train_valid_ratio=0.9):
    train_images, train_labels = [], []
    validation_images, validation_labels = [], []
    test_images, test_labels = [], []

    for cat_images, cat_masks in zip(category_images, category_masks):
        train_valid_images, test_images_, train_valid_labels, test_labels_ = train_test_split(cat_images, cat_masks, test_size=test_train_ratio, shuffle=True, random_state=15)
        train_images_, valid_images_, train_labels_, valid_labels_ = train_test_split(train_valid_images, train_valid_labels, train_size=train_valid_ratio, shuffle=True,random_state=30)

        train_images.extend(train_images_)
        train_labels.extend(train_labels_)
        validation_images.extend(valid_images_)
        validation_labels.extend(valid_labels_)
        test_images.extend(test_images_)
        test_labels.extend(test_labels_)

        data_dict = {
            'train_images': train_images,
            'train_masks': train_labels,
            'validation_images': validation_images,
            'validation_masks': validation_labels,
            'test_images': test_images,
            'test_masks': test_labels,
        }

    return data_dict





def test():
    TEST_IMG_DIR = "./test/test2018/" 
    TEST_MASK_DIR = "./test/mask/"
    cat_wise_images, cat_wise_masks = split_category(TEST_IMG_DIR, TEST_MASK_DIR)
    data_dict = split_data(cat_wise_images, cat_wise_images)
    print("hello")
    #ds = Teeth_Dataset("./train-val/train2018/", "./train-val/masks/")
    

if __name__ == "__main__":
    test()


