"""Width-scaled pre-activation ResNet-18 used in the paper."""

import torch
from torch import nn
from torch.nn import functional as F


class PreActBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, bias=False
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if self.shortcut is not None else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        return out + shortcut


class PreActResNet18(nn.Module):
    def __init__(self, width: int, num_classes: int = 10):
        super().__init__()
        self.in_channels = width
        self.conv1 = nn.Conv2d(3, width, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(width, blocks=2, stride=1)
        self.layer2 = self._make_layer(2 * width, blocks=2, stride=2)
        self.layer3 = self._make_layer(4 * width, blocks=2, stride=2)
        self.layer4 = self._make_layer(8 * width, blocks=2, stride=2)
        self.linear = nn.Linear(8 * width, num_classes)

    def _make_layer(self, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for block_stride in strides:
            layers.append(PreActBlock(self.in_channels, out_channels, block_stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        return self.linear(torch.flatten(out, 1))


class NormalizedModel(nn.Module):
    def __init__(self, network: nn.Module, mean, std):
        super().__init__()
        self.network = network
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32)[None, :, None, None])
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32)[None, :, None, None])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network((x - self.mean) / self.std)


def make_resnet18(width: int, mean, std) -> NormalizedModel:
    return NormalizedModel(PreActResNet18(width=width), mean=mean, std=std)

