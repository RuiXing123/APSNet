import torch
from torch import nn


class FrequencySeparation(nn.Module):
    def __init__(self, in_channels, low_pass_size=12):
        super(FrequencySeparation, self).__init__()
        self.low_pass_size = low_pass_size
        self.channel_reduce = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        mask = torch.zeros((H, W), device=x.device)
        crow, ccol = H // 2, W // 2
        size = self.low_pass_size
        mask[crow-size:crow+size, ccol-size:ccol+size] = 1
        mask = mask[None, None, :, :]

        x_fft = torch.fft.fft2(x)
        x_fft_shifted = torch.fft.fftshift(x_fft)
        high_freq_fft = x_fft_shifted * (1 - mask)
        # low_freq_fft = x_fft_shifted * (mask)
        high_freq = torch.fft.ifft2(torch.fft.ifftshift(high_freq_fft)).real  # [B, C, H, W]

        high_freq = self.channel_reduce(high_freq)  # [B, 1, H, W]

        return high_freq