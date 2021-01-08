from collections import OrderedDict

import torch
from torch import nn as nn

from repalette.models.nn import ConvBlock


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1",
                        ConvBlock(
                            3 + 18,
                            64,
                            kernel_size=4,
                            stride=2,
                        ),
                    ),
                    (
                        "conv2",
                        ConvBlock(
                            64,
                            128,
                            kernel_size=4,
                            stride=2,
                        ),
                    ),
                    (
                        "conv3",
                        ConvBlock(
                            128,
                            256,
                            kernel_size=4,
                            stride=2,
                        ),
                    ),
                    (
                        "conv4",
                        ConvBlock(
                            256,
                            512,
                            kernel_size=4,
                            stride=2,
                        ),
                    ),
                    (
                        "fc",
                        nn.AdaptiveAvgPool2d(1),
                    ),
                    ("flatten", nn.Flatten()),
                    ("activ", nn.Sigmoid()),
                ]
            )
        )

    def forward(self, x, palette):
        device = next(self.parameters()).device
        batch_size, _, height, width = x.size()
        palette_dupl = palette[:, :, None, None] * torch.ones(
            batch_size,
            18,
            height,
            width,
            device=device,
        )
        x = torch.cat([x, palette_dupl], dim=1)
        return self.model(x)
