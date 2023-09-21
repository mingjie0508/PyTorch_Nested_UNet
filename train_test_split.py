####################
# train_test_split.py
#
# Description: Splits the image dataset into training (80%), validation (10%), and test (10%) sets
# Usage: python train_test_split.py
#
# Output directory structure:
#   train
#       images: training images, e.g. "1.jpg", "3.jpg"
#       masks: train masks, e.g. "1.jpg", "3.jpg"
#   validation
#       images: validation images, e.g. "4.jpg", "6.jpg"
#       masks: validation masks, e.g. "4.jpg", "6.jpg"
#   test
#       images: test images, e.g. "2.jpg", "5.jpg"
#       masks: test masks, e.g. "2.jpg", "5.jpg"
####################

import os
import shutil
from sklearn.model_selection import train_test_split

IMAGES_DIR = "./Water_Bodies_Dataset/Images"    # directory for all images
MASKS_DIR = "./Water_Bodies_Dataset/Masks"      # directory for all masks
TRAIN_IMAGES_DIR = "./dataset/train/images"     # directory for training images
TRAIN_MASKS_DIR = "./dataset/train/masks"       # directory for training masks
VAL_IMAGES_DIR = "./dataset/validation/images"  # directory for validation images
VAL_MASKS_DIR = "./dataset/validation/masks"    # directory for validation masks
TEST_IMAGES_DIR = "./dataset/test/images"       # directory for test images
TEST_MASKS_DIR = "./dataset/test/masks"         # direcotry for test masks
PRED_DIR = "./dataset/test/predictions"         # directory for predicted labels
TRAIN_SIZE = 0.9
VAL_SIZE = 0.1
TEST_SIZE = 0.1

if __name__ == '__main__':
    # create directories for training, validation, and test sets
    dirs = (
        TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR, 
        VAL_IMAGES_DIR, VAL_MASKS_DIR, 
        TEST_IMAGES_DIR, TEST_MASKS_DIR, 
        PRED_DIR
    )
    for train_test_dir in dirs:
        # create dir if not exist
        os.makedirs(train_test_dir, exist_ok=True)
        # remove any existing files
        for dirname, _, filenames in os.walk(train_test_dir):
            for f in filenames:
                path = os.path.join(dirname, f)
                if os.path.isfile(path):
                    os.remove(path)

    imgs_paths, masks_paths = [], []
    # iterate over all filenames in dataset
    for dirname, _, filenames in os.walk(IMAGES_DIR):
        for filename in filenames:
            # e.g. "./Water_Bodies_Dataset/Images/water_body_1"
            img_path = os.path.join(IMAGES_DIR, filename)
            # e.g. "./Water_Bodies_Dataset/Masks/water_body_1"
            mask_path = os.path.join(MASKS_DIR, filename)
            # append paths to variables
            imgs_paths.append(img_path)
            masks_paths.append(mask_path)
    
    # split training and test sets
    img_temp, img_test, mask_temp, mask_test = train_test_split(
        imgs_paths, masks_paths, test_size=TEST_SIZE, random_state=1
    )
    img_train, img_val, mask_train, mask_val = train_test_split(
        img_temp, mask_temp, test_size=VAL_SIZE/(1-TEST_SIZE), random_state=1
    )
    print("Number of training images:", len(img_train))
    print("Number of validation images:", len(img_val))
    print("Number of test images:", len(img_test))
    print("Number of training masks:", len(mask_train))
    print("Number of validation masks:", len(mask_val))
    print("Number of test masks:", len(mask_test))

    # copy files to training and test directories
    for path in img_train:
        filename = os.path.basename(path).split("_")[-1]  # e.g. "1.jpg"
        new_path = os.path.join(TRAIN_IMAGES_DIR, filename)
        shutil.copy(path, new_path)
    for path in img_val:
        filename = os.path.basename(path).split("_")[-1]  # e.g. "2.jpg"
        new_path = os.path.join(VAL_IMAGES_DIR, filename)
        shutil.copy(path, new_path)
    for path in img_test:
        filename = os.path.basename(path).split("_")[-1]  # e.g. "3.jpg"
        new_path = os.path.join(TEST_IMAGES_DIR, filename)
        shutil.copy(path, new_path)
    for path in mask_train:
        filename = os.path.basename(path).split("_")[-1]  # e.g. "1.jpg"
        new_path = os.path.join(TRAIN_MASKS_DIR, filename)
        shutil.copy(path, new_path)
    for path in mask_val:
        filename = os.path.basename(path).split("_")[-1]  # e.g. "2.jpg"
        new_path = os.path.join(VAL_MASKS_DIR, filename)
        shutil.copy(path, new_path)
    for path in mask_test:
        filename = os.path.basename(path).split("_")[-1]  # e.g. "3.jpg"
        new_path = os.path.join(TEST_MASKS_DIR, filename)
        shutil.copy(path, new_path)
    print("Finished splitting training, validation, and test sets")
