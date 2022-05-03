
import torch
import torch.nn as nn
from .base import *


class UNet(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, in_ch=3, out_ch=1):
        super(UNet, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = DoubleConvBlock(in_ch, filters[0])
        self.Conv2 = DoubleConvBlock(filters[0], filters[1])
        self.Conv3 = DoubleConvBlock(filters[1], filters[2])
        self.Conv4 = DoubleConvBlock(filters[2], filters[3])
        self.Conv5 = DoubleConvBlock(filters[3], filters[4])

        self.Up5 = UpConv(filters[4], filters[3])
        self.Up_conv5 = DoubleConvBlock(filters[4], filters[3])

        self.Up4 = UpConv(filters[3], filters[2])
        self.Up_conv4 = DoubleConvBlock(filters[3], filters[2])

        self.Up3 = UpConv(filters[2], filters[1])
        self.Up_conv3 = DoubleConvBlock(filters[2], filters[1])

        self.Up2 = UpConv(filters[1], filters[0])
        self.Up_conv2 = DoubleConvBlock(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch,
                              kernel_size=1, stride=1, padding=0)

       # self.active = torch.nn.Sigmoid()

    def forward(self, x):

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)
        return out

    def name(self):
        return "Unet"


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    image = torch.rand((1, 3, 512, 512)).to(DEVICE)
    model = UNet().to(DEVICE)
    res = model(image)
    print(res.shape)
