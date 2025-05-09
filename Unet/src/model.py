"""
Module description:
    This module contains the implementation of a U-Net architecture for image segmentation.

It includes classes for building the network, such as Block, Encoder, Decoder, and UNet.
    These classes utilize PyTorch's neural network modules and torchvision for image processing.
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv_op(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        down = self.conv(x)  # For skip connection
        p = self.pool(down)
        p = self.dropout(p)  # Regularization

        return down, p


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.dropout = nn.Dropout2d(dropout)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], 1)
        x = self.dropout(x)  # Regularization
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels, num_classes, dropout=0):
        super().__init__()
        self.down_convolution_1 = Encoder(in_channels, 64, dropout)
        self.down_convolution_2 = Encoder(64, 128, dropout)
        self.down_convolution_3 = Encoder(128, 256, dropout)
        self.down_convolution_4 = Encoder(256, 512, dropout)

        self.bottle_neck = DoubleConv(512, 1024)

        self.up_convolution_1 = Decoder(1024, 512, dropout)
        self.up_convolution_2 = Decoder(512, 256, dropout)
        self.up_convolution_3 = Decoder(256, 128, dropout)
        self.up_convolution_4 = Decoder(128, 64, dropout)

        self.out = nn.Conv2d(
            in_channels=64, out_channels=num_classes, kernel_size=1
        )

    def forward(self, x):
        down_1, p1 = self.down_convolution_1(x)
        down_2, p2 = self.down_convolution_2(p1)
        down_3, p3 = self.down_convolution_3(p2)
        down_4, p4 = self.down_convolution_4(p3)

        b = self.bottle_neck(p4)

        up_1 = self.up_convolution_1(b, down_4)
        up_2 = self.up_convolution_2(up_1, down_3)
        up_3 = self.up_convolution_3(up_2, down_2)
        up_4 = self.up_convolution_4(up_3, down_1)

        out = self.out(up_4)
        return out
