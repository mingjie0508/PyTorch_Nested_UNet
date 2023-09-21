# PyTorch_Nested_UNet

Unofficial PyTorch implementation of Nested U-Net (UNet++) for image segmentation. Original paper: [UNet++: A Nested U-Net Architecture for Medical Image Segmentation (Zhou et al., 2018)](https://arxiv.org/abs/1807.10165)

### Installation
Create conda environment
```
conda create -n=<env_name> python=3.8 anaconda
conda activate <env_name>
```

Install Pytorch
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

Install libraries
```
pip install -r requirements.txt
```

### Dataset

Satellite water bodies images. Each image comes with a binary mask where white represents water and black represents anything else but water.

Dataset source: https://www.kaggle.com/datasets/franciscoescobar/satellite-images-of-water-bodies

Original dataset structure
```
Water_Bodies_Dataset
├── Images
|   ├── water_body_1.jpg
|   └── ...
└── Masks
    ├── water_body_1.jpg
    └── ...
```

### Run

1. Split the dataset into training, validation, and test sets. Save the three sets in: `./dataset`
```sh
python train_test_split.py
```

2. Train U-Net using the training and validation sets.
```sh
python train.py
```

3. Make predictions for the test set.
```sh
python predict.py
```

### Output

Model checkpoint: `./model/{MODEL}_{LOSS}_checkpoint.pth`

Training/validation losses and accuracy: `./model/{MODEL}_{LOSS}_accuracy.csv`

Training log: `./model/{MODEL}_{LOSS}.log`

Predicted segmentation: `./test/predictions`
