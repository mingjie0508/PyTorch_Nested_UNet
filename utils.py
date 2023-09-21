import os
import csv
import numpy as np
import torch
import torchvision
from dataset import ImageDataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score


def save_checkpoint(state, filename="model_checkpoint.pth"):
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    model.load_state_dict(checkpoint['state_dict'])


def get_dataloaders(
    train_img_dir,
    train_mask_dir,
    test_img_dir,
    test_mask_dir,
    batch_size,
    train_transform,
    test_transform,
    num_workers=4,
    pin_memory=True
):
    # training set
    train_dataset = ImageDataset(
        img_dir=train_img_dir,
        mask_dir=train_mask_dir,
        transform=train_transform
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True
    )
    # test/validation set
    test_dataset = ImageDataset(
        img_dir=test_img_dir,
        mask_dir=test_mask_dir,
        transform=test_transform
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )

    return train_dataloader, test_dataloader


def check_accuracy(dataloader, model, device="cuda"):
    scores = []
    model.eval()

    with torch.no_grad():
        for x, y, attributes in dataloader:
            # make batch prediction
            x = x.to(device)
            y = y.to(device)
            y = y.unsqueeze(1).cpu().numpy()
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            preds = preds.cpu().numpy()
            # compute metrics for each image
            for pred1, y1 in zip(preds, y):
                pred1, y1 = pred1.flatten(), y1.flatten()
                acc_value = accuracy_score(pred1, y1)
                f1_value = f1_score(pred1, y1, labels=[0.0, 1.0], average="binary", zero_division=0.0)
                # sklearn jaccard doesn't have zero_division argument
                jac_value = jaccard_score(pred1, y1, labels=[0.0, 1.0], average="binary")
                dice_value = 2*jac_value / (jac_value+1)
                recall_value = recall_score(pred1, y1, labels=[0.0, 1.0], average="binary", zero_division=0.0)
                precision_value = precision_score(pred1, y1, labels=[0.0, 1.0], average="binary", zero_division=0.0)
                scores.append([acc_value, f1_value, jac_value, dice_value, recall_value, precision_value])

    # average score over all batches
    mean_scores = np.mean(scores, axis=0)
    scores = {
        "Accuracy": mean_scores[0],
        "F1": mean_scores[1],
        "Jaccard": mean_scores[2],
        "Dice": mean_scores[3],
        "Recall": mean_scores[4],
        "Precision": mean_scores[5]
    }
    # print scores, e.g. Accuracy: 0.83251
    for key, val in scores.items():
        print(f"{key:<10} {val:0.5f}")

    model.train()
    return(scores)


def save_accuracy(accuracy_list, filename="accuracy.csv"):
    # column names, e.g. Epoch, Accuracy, F1, Jaccard, Recall, Precision
    field_names = ["Epoch"]
    field_names.extend(list(accuracy_list[0].keys()))
    # table represented as a list of dicts, where each dict is a row
    # add Epoch index
    for i, row in enumerate(accuracy_list):
        row["Epoch"] = i+1
    # save all accuracies
    with open(filename, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = field_names)
        writer.writeheader()
        writer.writerows(accuracy_list)


def save_pred(dataloader, model, pred_dir, device="cuda"):
    model.eval()
    for x, y, attributes in dataloader:
        # make batch predictions
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
        # save each predicted label separately
        filenames = attributes['filename']
        heights, widths = attributes['size']
        heights, widths = heights.tolist(), widths.tolist()
        for pred, width, height, f in zip(preds, widths, heights, filenames):
            # convert back to original image size
            size = (width, height)
            resize = torchvision.transforms.Resize(size, antialias=True)
            pred = resize(pred)
            pred = pred.float()  # binarize the predicted labels
            # save
            pred_path = os.path.join(pred_dir, f)
            torchvision.utils.save_image(pred, pred_path)
    
    model.train()
            
