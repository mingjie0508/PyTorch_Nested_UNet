import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from unet import UNet, NestedUNet
from losses import BCEDiceLoss
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_dataloaders,
    check_accuracy,
    save_accuracy
)

# configuration
SEED = 1
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "UNET"
LOSS = "BCE_DICE_LOSS"
BATCH_SIZE = 48
NUM_EPOCHS = 50
NUM_WORKERS = 1
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = False
CHECKPOINT = f"./model/{MODEL}_{LOSS}_checkpoint.pth"       # path to model checkpoint
ACCURACY_PATH = f"./model/{MODEL}_{LOSS}_accuracy.csv"      # path to validation accuracies
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
    # define data augmentations for training and validation sets
    train_transform = A.Compose([
        A.Resize(IMAGE_WIDTH+16, IMAGE_HEIGHT+16),
        A.RandomCrop(IMAGE_WIDTH, IMAGE_HEIGHT),
        A.RandomRotate90(),
        A.Flip(),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
        #A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    val_transform = A.Compose([
        A.Resize(IMAGE_WIDTH, IMAGE_HEIGHT),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
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
    # define loss function
    if LOSS == "BCE":
        loss_fn = nn.BCEWithLogitsLoss()
    elif LOSS == "BCE_DICE_LOSS":
        loss_fn = BCEDiceLoss()
    else:
        raise NotImplementedError
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # training, validation, and test dataloaders
    train_dataloader, val_dataloader = get_dataloaders(
        train_img_dir=TRAIN_IMAGES_DIR,
        train_mask_dir=TRAIN_MASKS_DIR,
        test_img_dir=VAL_IMAGES_DIR,
        test_mask_dir=VAL_MASKS_DIR,
        batch_size=BATCH_SIZE,
        train_transform=train_transform,
        test_transform=val_transform,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    train_dataloader, test_dataloader = get_dataloaders(
        train_img_dir=TRAIN_IMAGES_DIR,
        train_mask_dir=TRAIN_MASKS_DIR,
        test_img_dir=TEST_IMAGES_DIR,
        test_mask_dir=TEST_MASKS_DIR,
        batch_size=BATCH_SIZE,
        train_transform=train_transform,
        test_transform=test_transform,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    dataloaders = (train_dataloader, val_dataloader, test_dataloader)
    # define scaler
    scaler = torch.cuda.amp.GradScaler()

    return dataloaders, model, optimizer, loss_fn, scaler


def train_fn(dataloader, model, optimizer, loss_fn, scaler):
    running_loss = 0.0

    for x, y, attributes in dataloader:
        x = x.to(device=DEVICE)
        y = y.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            preds = model(x)
            loss = loss_fn(preds, y)
        running_loss += loss.item()
        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    loss = running_loss / len(dataloader)
    return loss


def val_fn(dataloader, model, loss_fn):
    running_loss = 0.0
    model.eval()

    for x, y, attributes in dataloader:
        x = x.to(device=DEVICE)
        y = y.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.no_grad():
            preds = model(x)
            loss = loss_fn(preds, y)
        running_loss += loss.item()
    
    model.train()
    loss = running_loss / len(dataloader)
    return loss


def main():
    print("-"*10)
    print("Configuring...")
    dataloaders, model, optimizer, loss_fn, scaler = config()
    train_dataloader, val_dataloader, test_dataloader = dataloaders

    # train loop
    max_dice = 0.0
    accuracy_list = []
    print("Training...")
    for epoch in range(NUM_EPOCHS):
        print("-"*10)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        # train
        train_loss = train_fn(train_dataloader, model, optimizer, loss_fn, scaler)
        val_loss = val_fn(val_dataloader, model, loss_fn)
        print(f"Training loss: {train_loss:0.5f}")
        print(f"Validation loss: {val_loss:0.5f}")
        # check accuracy
        print("Checking validation accuracy...")
        scores = check_accuracy(val_dataloader, model, device=DEVICE)
        scores["Validation_Loss"] = val_loss    # append loss to accuracy results
        scores["Train_Loss"] = train_loss       # append loss to accuracy results
        accuracy_list.append(scores)
        # save model
        curr_dice = scores['Dice']
        if curr_dice > max_dice:
            print(f"Validation Dice increases: {max_dice:0.5f} -> {curr_dice:0.5f}")
            max_dice = curr_dice
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            print("Saving checkpoint...")
            save_checkpoint(checkpoint, filename=CHECKPOINT)

    # save all accuracies
    print("-"*10)
    print("Saving all train and validation accuracies...")
    save_accuracy(accuracy_list, filename=ACCURACY_PATH)
    # save predicted labels
    print("Testing...")
    check_accuracy(test_dataloader, model, device=DEVICE)


if __name__ == "__main__":
    print("SEED:", SEED)
    print("LEARNING_RATE:", f"{LEARNING_RATE:0.2E}")
    print("DEVICE:", DEVICE)
    print("MODEL:", MODEL)
    print("LOSS:", LOSS)
    print("BATCH_SIZE:", BATCH_SIZE)
    print("NUM_EPOCHS:", NUM_EPOCHS)
    print("NUM_WORKERS:", NUM_WORKERS)
    print("IMAGE_HEIGHT:", IMAGE_HEIGHT)
    print("IMAGE_WIDTH:", IMAGE_WIDTH)
    print("LOAD_MODEL:", LOAD_MODEL)

    main()
