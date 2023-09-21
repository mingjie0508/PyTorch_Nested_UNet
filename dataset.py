####################
# dataset.py
#
# Description: Contains custom image dataset class
####################

import os
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFilter

class ImageDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        # a list of all filenames, e.g. "1.jpg", "10.jpg"
        self.filenames = os.listdir(img_dir)
        self.transform = transform
    def __len__(self):
        return len(self.filenames)
    def __getitem__(self, idx):
        # read image and sharpen
        img_filename = self.filenames[idx]
        img_path = os.path.join(self.img_dir, img_filename)
        img = Image.open(img_path).convert("RGB")
        img = img.filter(ImageFilter.SHARPEN)
        size = img.size
        img = np.array(img)
        # read mask
        mask_filename = self.filenames[idx]
        mask_path = os.path.join(self.mask_dir, mask_filename)
        mask = Image.open(mask_path).convert("L")
        mask = np.array(mask)
        mask = np.float32(np.where(mask > 255.0 / 2, 1.0, 0.0))
        # transform image and mask
        if self.transform is not None:
            # apply transformation, convert to tensor
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        return img, mask, {'filename': img_filename, 'size': size}


if __name__ == '__main__':
    TRAIN_IMAGES_DIR = "./dataset/train/images"     # directory for training images
    TRAIN_MASKS_DIR = "./dataset/train/masks"       # directory for training masks
    TEST_IMAGES_DIR = "./dataset/test/images"       # directory for test images
    TEST_MASKS_DIR = "./dataset/test/masks"         # direcotry for test masks

    # define augmentation
    train_transform = A.Compose([
        A.Resize(136, 136),
        A.RandomCrop(128, 128),
        A.RandomRotate90(),
        A.Flip(),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    test_transform = A.Compose([
        A.Resize(128, 128),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # create dataset objects
    train_dataset = ImageDataset(
        img_dir=TRAIN_IMAGES_DIR,
        mask_dir=TRAIN_MASKS_DIR,
        transform=train_transform
    )
    test_dataset = ImageDataset(
        img_dir=TEST_IMAGES_DIR,
        mask_dir=TEST_MASKS_DIR,
        transform=test_transform
    )

    print("Size of training set:", len(train_dataset))
    img, mask, attributes = train_dataset[0]
    print("First image in training set:", attributes)
    print("Size of first augmented image in training set:", img.size())
    print("Size of first mask in training set:", mask.size())
    print("First augmented image in training set:")
    print(img)
    print("First mask in training set:")
    print(mask)
    print("Unique values in first mask:", torch.unique(mask))
