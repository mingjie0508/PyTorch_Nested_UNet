import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from unet import UNet, NestedUNet
from utils import (
    load_checkpoint,
    get_dataloaders,
    check_accuracy,
    save_pred
)

# configuration
SEED = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "UNET"
LOSS = "BCE_DICE_LOSS"
BATCH_SIZE = 48
NUM_WORKERS = 1
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = True
CHECKPOINT = f"./model/{MODEL}_{LOSS}_checkpoint.pth"       # path to model checkpoint
TRAIN_IMAGES_DIR = "./dataset/train/images"     # directory for training images
TRAIN_MASKS_DIR = "./dataset/train/masks"       # directory for training masks
VAL_IMAGES_DIR = "./dataset/validation/images"  # directory for validation images
VAL_MASKS_DIR = "./dataset/validation/masks"    # direcotry for validation masks
TEST_IMAGES_DIR = "./dataset/test/images"       # directory for test images
TEST_MASKS_DIR = "./dataset/test/masks"         # direcotry for test masks
PRED_DIR = "./dataset/test/predictions"         # directory for predicted labels for test


def config():
    # set seed
    torch.manual_seed(SEED)
    # define data augmentations for test sets
    test_transform = A.Compose([
        A.Resize(IMAGE_WIDTH, IMAGE_HEIGHT),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    # define model
    if MODEL == "UNET":
        model = UNet(num_classes=1, input_channels=3).to(DEVICE)
    elif MODEL == "NESTED_UNET":
        model = NestedUNet(num_classes=1, input_channels=3).to(DEVICE)
    else:
        raise NotImplementedError
    # load checkpoint
    if LOAD_MODEL:
        print("Loading checkpoint...")
        load_checkpoint(torch.load(CHECKPOINT), model)
    # training, validation, and test dataloaders
    train_dataloader, test_dataloader = get_dataloaders(
        train_img_dir=TRAIN_IMAGES_DIR,
        train_mask_dir=TRAIN_MASKS_DIR,
        test_img_dir=TEST_IMAGES_DIR,
        test_mask_dir=TEST_MASKS_DIR,
        batch_size=BATCH_SIZE,
        train_transform=test_transform,  # dummy, train dataloader is ignored
        test_transform=test_transform,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    return test_dataloader, model


def main():
    print("Configuring...")
    test_dataloader, model = config()

    # save predicted labels
    print("Testing...")
    check_accuracy(test_dataloader, model, device=DEVICE)
    print("Saving predictions...")
    scores = save_pred(test_dataloader, model, PRED_DIR, device=DEVICE)


if __name__ == "__main__":
    print("DEVICE:", DEVICE)
    print("MODEL:", MODEL)
    print("LOSS:", LOSS)
    print("BATCH_SIZE:", BATCH_SIZE)
    print("NUM_WORKERS:", NUM_WORKERS)
    print("IMAGE_HEIGHT:", IMAGE_HEIGHT)
    print("IMAGE_WIDTH:", IMAGE_WIDTH)
    print("LOAD_MODEL:", LOAD_MODEL)

    main()
