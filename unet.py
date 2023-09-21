####################
# unet.py
#
# Description: Contains UNet and nested UNet architecture
####################

import torch
from torch import nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class UNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        feat_num = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = DoubleConv(input_channels, feat_num[0], feat_num[0])
        self.conv1_0 = DoubleConv(feat_num[0], feat_num[1], feat_num[1])
        self.conv2_0 = DoubleConv(feat_num[1], feat_num[2], feat_num[2])
        self.conv3_0 = DoubleConv(feat_num[2], feat_num[3], feat_num[3])
        self.conv4_0 = DoubleConv(feat_num[3], feat_num[4], feat_num[4])

        self.conv3_1 = DoubleConv(feat_num[3]+feat_num[4], feat_num[3], feat_num[3])
        self.conv2_2 = DoubleConv(feat_num[2]+feat_num[3], feat_num[2], feat_num[2])
        self.conv1_3 = DoubleConv(feat_num[1]+feat_num[2], feat_num[1], feat_num[1])
        self.conv0_4 = DoubleConv(feat_num[0]+feat_num[1], feat_num[0], feat_num[0])

        self.final = nn.Conv2d(feat_num[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output


class NestedUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        feat_num = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = DoubleConv(input_channels, feat_num[0], feat_num[0])
        self.conv1_0 = DoubleConv(feat_num[0], feat_num[1], feat_num[1])
        self.conv0_1 = DoubleConv(feat_num[0]+feat_num[1], feat_num[0], feat_num[0])

        self.conv2_0 = DoubleConv(feat_num[1], feat_num[2], feat_num[2])
        self.conv1_1 = DoubleConv(feat_num[1]+feat_num[2], feat_num[1], feat_num[1])
        self.conv0_2 = DoubleConv(feat_num[0]*2+feat_num[1], feat_num[0], feat_num[0])

        self.conv3_0 = DoubleConv(feat_num[2], feat_num[3], feat_num[3])
        self.conv2_1 = DoubleConv(feat_num[2]+feat_num[3], feat_num[2], feat_num[2])
        self.conv1_2 = DoubleConv(feat_num[1]*2+feat_num[2], feat_num[1], feat_num[1])
        self.conv0_3 = DoubleConv(feat_num[0]*3+feat_num[1], feat_num[0], feat_num[0])

        self.conv4_0 = DoubleConv(feat_num[3], feat_num[4], feat_num[4])
        self.conv3_1 = DoubleConv(feat_num[3]+feat_num[4], feat_num[3], feat_num[3])
        self.conv2_2 = DoubleConv(feat_num[2]*2+feat_num[3], feat_num[2], feat_num[2])
        self.conv1_3 = DoubleConv(feat_num[1]*3+feat_num[2], feat_num[1], feat_num[1])
        self.conv0_4 = DoubleConv(feat_num[0]*4+feat_num[1], feat_num[0], feat_num[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(feat_num[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(feat_num[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(feat_num[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(feat_num[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(feat_num[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output


if __name__ == '__main__':
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    IMAGE_HEIGHT = 128
    IMAGE_WIDTH = 128

    # create UNet model
    print("UNet model")
    unet = UNet(num_classes=1, input_channels=3).to(DEVICE)
    # create random input images, batch size 8
    img = torch.randn((8, 3, IMAGE_HEIGHT, IMAGE_WIDTH))
    img = img.to(DEVICE)
    # make predictions
    preds = unet(img)
    print("Size of input image batch:", img.size())
    print("Size of output prediction batch:", preds.size())

    print("-"*10)

    # create nested UNet model
    print("Nested UNet model")
    nested_unet = NestedUNet(num_classes=1, input_channels=3).to(DEVICE)
    # create random input images, batch size 8
    img = torch.randn((8, 3, IMAGE_HEIGHT, IMAGE_WIDTH))
    img = img.to(DEVICE)
    # make predictions
    preds = nested_unet(img)
    print("Size of input image batch:", img.size())
    print("Size of output prediction batch:", preds.size())
