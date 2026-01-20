import math

from torch import nn


class ECA(nn.Module):
    """
    ECA: Efficient Channel Attention
    paper: "ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks" (CVPR 2020)

    support input:
      - 4D: [B, C, H, W]
      - 3D: [B, C, L]
    """
    def __init__(self, channels: int, gamma: int = 2, beta: int = 1):
        super().__init__()
        t = int(abs((math.log2(channels) + beta) / gamma))
        k = t if t % 2 else t + 1
        k = max(k, 3)  

        self.avg_pool = nn.AdaptiveAvgPool2d(1)  
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv2d(channels, channels-1, 1, bias=True)

    def forward(self, x):
        """
        x: [B, C, H, W]
        """

        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c, 1).transpose(1, 2)
        y = self.conv(y)
        y = self.sigmoid(y.transpose(1, 2).view(b, c, 1, 1))

        return self.conv1(x * y)
