from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = torch.nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResNet34Encoder(nn.Module):
    def __init__(self, in_channels: int = 1, pretrained: bool = False):
        super().__init__()
        backbone = models.resnet34(pretrained=pretrained)
        if in_channels != 3:
            old_conv = backbone.conv1
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )
            with torch.no_grad():
                if in_channels == 1:
                    new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
                else:
                    repeat_factor = (in_channels + 2) // 3
                    repeated = old_conv.weight.repeat(1, repeat_factor, 1, 1)[:, :in_channels, :, :]
                    new_conv.weight.copy_(repeated / float(repeat_factor))
            backbone.conv1 = new_conv
        self.backbone = backbone

    def forward(self, x: torch.Tensor):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        c1 = self.backbone.relu(x)
        x = self.backbone.maxpool(c1)
        c2 = self.backbone.layer1(x)
        c3 = self.backbone.layer2(c2)
        c4 = self.backbone.layer3(c3)
        c5 = self.backbone.layer4(c4)
        return c1, c2, c3, c4, c5

    def encoder_state_dict(self):
        state = {}
        for key, value in self.backbone.state_dict().items():
            state[f"resnet.{key}"] = value
        return state


class SSLReconstructionNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 3, encoder_pretrained: bool = False):
        super().__init__()
        self.encoder = ResNet34Encoder(in_channels=in_channels, pretrained=encoder_pretrained)
        self.u5 = UpBlock(512, 256, 256)
        self.u6 = UpBlock(256, 128, 128)
        self.u7 = UpBlock(128, 64, 64)
        self.u8 = UpBlock(64, 64, 64)
        self.head = nn.ConvTranspose2d(64, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1, c2, c3, c4, c5 = self.encoder(x)
        x = self.u5(c5, c4)
        x = self.u6(x, c3)
        x = self.u7(x, c2)
        x = self.u8(x, c1)
        x = self.head(x)
        return torch.sigmoid(x)

    def export_encoder_state(self):
        return self.encoder.encoder_state_dict()
