import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class Teeth_Dataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, target_transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transfrom = transform
        self.target_transform = target_transform

        self.images = os.listdir(images_dir)
        self.masks = os.listdir(masks_dir)


    def __len__(self):
        return len(self.masks)


    def __getitem__(self, index):
        img_path = os.path.join(self.images_dir, (str(self.masks[index])).replace(".bmp", ".jpg"))
        mask_path = os.path.join(self.masks_dir, str(self.masks[index]))

        image = Image.open(img_path)
        mask = Image.open(mask_path)

        if self.transfrom:
            image = self.transfrom(image)
        
        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask


def test():
    ds = Teeth_Dataset("./train-val/train2018/", "./train-val/masks/")
    ds[0]

if __name__ == "__main__":
    test()


