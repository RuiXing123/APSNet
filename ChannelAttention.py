import math

from torch import nn


class ECA(nn.Module):
    """
    ECA: Efficient Channel Attention
    论文: "ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks" (CVPR 2020)

    支持输入:
      - 4D: [B, C, H, W]
      - 3D: [B, C, L]  (例如1D特征/时序)
    """
    def __init__(self, channels: int, gamma: int = 2, beta: int = 1):
        super().__init__()
        # 自适应核大小 k (需为奇数)
        t = int(abs((math.log2(channels) + beta) / gamma))
        k = t if t % 2 else t + 1
        k = max(k, 3)  # 至少为3，经验更稳

        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 对4D输入用；3D输入会改用 F.adaptive_avg_pool1d
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv2d(channels, channels-1, 1, bias=True)

    def forward(self, x):
        """
        x: [B, C, H, W]
        返回: 同形状，按通道加权后的张量
        """

        b, c, _, _ = x.shape
        # 全局平均池化 -> [B, C, 1, 1] -> [B, 1, C]
        y = self.avg_pool(x).view(b, c, 1).transpose(1, 2)
        # 1D卷积在通道轴上建模局部跨通道交互 -> [B, 1, C]
        y = self.conv(y)
        # 映射回 [B, C, 1, 1]
        y = self.sigmoid(y.transpose(1, 2).view(b, c, 1, 1))
        return self.conv1(x * y)