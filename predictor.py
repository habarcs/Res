"""
This is the original U-net model from the paper U-Net: Convolutional Networks for Biomedical Image Segmentation
"""

import torch


class UNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = ConvBlock(3, 64)
        self.down1 = DownBlock(64, 128)
        self.down2 = DownBlock(128, 256)
        self.down3 = DownBlock(256, 512)
        self.down4 = DownBlock(512, 1024)

        self.up1 = UpBlock(1024, 512)
        self.up2 = UpBlock(512, 256)
        self.up3 = UpBlock(256, 128)
        self.up4 = UpBlock(128, 64)

        self.out = torch.nn.Conv2d(64, 3, 1)

    def forward(self, x):
        assert x.shape[-1] % 8 == 0 and x.shape[-2] % 8 == 0, (
            "the input is not a multiple of 8, the output won't match the input"
        )
        x1 = self.first(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x4, x5)
        x = self.up2(x3, x)
        x = self.up3(x2, x)
        x = self.up4(x1, x)
        return self.out(x)


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, padding="same"),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, 3, padding="same"),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


class DownBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(torch.nn.MaxPool2d(2, 2), ConvBlock(in_channels, out_channels))

    def forward(self, x):
        return self.layers(x)


class UpBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.up = torch.nn.Sequential(torch.nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2))
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, side, down):
        # crop tensor coming from side to the size of the upscaled down tensor to be able to concat them
        # it should be only needed if the input is not divisible by 8
        down = self.up(down)
        h_target, w_target = down.shape[-2:]
        h_source, w_source = side.shape[-2:]
        h_start = (h_source - h_target) // 2
        w_start = (w_source - w_target) // 2
        cropped = side[..., h_start : h_start + h_target, w_start : w_start + w_target]
        concat = torch.cat((down, cropped), dim=1)
        out = self.conv(concat)
        return out
