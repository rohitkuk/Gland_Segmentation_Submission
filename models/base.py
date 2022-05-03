import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_block = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.conv_block(x)
        return x


class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.double_conv_block = nn.Sequential(
            ConvBlock(self.in_channels, self.out_channels),
            ConvBlock(self.out_channels, self.out_channels),
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.double_conv_block(x)
        return x


class UpConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(UpConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Up Conv block
        self.up_conv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            ConvBlock(self.in_channels, self.out_channels),

        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.up_conv(x)
        return x


class AttentionBlock(nn.Module):

    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,
                      stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,
                      stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out


class NestedBlock(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels):
        super(NestedBlock, self).__init__()

        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels

        self.nest_block = nn.Sequential(
            ConvBlock(self.in_channels, self.mid_channels),
            ConvBlock(self.mid_channels, self.out_channels),
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        output = self.nest_block(x)
        return output


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    image = torch.rand((1, 3, 522, 775)).to(DEVICE)

    model = DoubleConvBlock(3, 1).to(DEVICE)
    res = model(image)
    print(res.shape)
