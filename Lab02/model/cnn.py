"""
Lab02 - your colorization network lives here.

Interface (fixed - the grading script imports exactly this):

    from model.cnn import ColorizationCNN
    model = ColorizationCNN()
    ab_norm = model(L_norm)

    L_norm  : (B, 1, 32, 32) float tensor, ground-truth L* / 100
    ab_norm : (B, 2, 32, 32) float tensor, predicted a*b* / 110

Keep the class name, the constructor's default arguments, and both shapes.
Everything inside is yours~
"""

import torch
import torch.nn as nn


def _block(cin: int, cout: int) -> nn.Sequential:
    """Two 3x3 conv + BN + ReLU layers."""
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class ColorizationCNN(nn.Module):
    """Fully-convolutional U-Net: 32 -> 16 -> 8 -> 4 -> 8 -> 16 -> 32."""

    def __init__(self, base_channels: int = 32):
        super().__init__()
        self.base_channels = base_channels
        c = base_channels
        # encoder (strided 3x3 convs downsample, no pooling layers)
        self.enc1 = _block(1, c)                                     # 32x32
        self.down1 = nn.Conv2d(c, 2 * c, 3, stride=2, padding=1)     # 16x16
        self.enc2 = _block(2 * c, 2 * c)
        self.down2 = nn.Conv2d(2 * c, 4 * c, 3, stride=2, padding=1)  # 8x8
        self.enc3 = _block(4 * c, 4 * c)
        self.down3 = nn.Conv2d(4 * c, 8 * c, 3, stride=2, padding=1)  # 4x4
        # bottleneck: dilated convs give a large receptive field for global colour context
        self.mid = nn.Sequential(
            nn.Conv2d(8 * c, 8 * c, 3, padding=1, bias=False),
            nn.BatchNorm2d(8 * c), nn.ReLU(inplace=True),
            nn.Conv2d(8 * c, 8 * c, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(8 * c), nn.ReLU(inplace=True),
            nn.Conv2d(8 * c, 8 * c, 3, padding=1, bias=False),
            nn.BatchNorm2d(8 * c), nn.ReLU(inplace=True),
        )
        # decoder (transposed convs upsample, skip connections by concatenation)
        self.up3 = nn.ConvTranspose2d(8 * c, 4 * c, 2, stride=2)
        self.dec3 = _block(8 * c, 4 * c)
        self.up2 = nn.ConvTranspose2d(4 * c, 2 * c, 2, stride=2)
        self.dec2 = _block(4 * c, 2 * c)
        self.up1 = nn.ConvTranspose2d(2 * c, c, 2, stride=2)
        self.dec1 = _block(2 * c, c)
        self.head = nn.Sequential(nn.Conv2d(c, 2, 1), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        m = self.mid(self.down3(e3))
        d3 = self.dec3(torch.cat([self.up3(m), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


if __name__ == "__main__":
    net = ColorizationCNN()
    n = sum(p.numel() for p in net.parameters() if p.requires_grad)
    y = net(torch.zeros(4, 1, 32, 32))
    print(f"Trainable parameters: {n:,}")
    print(f"Output shape: {tuple(y.shape)}  (expected (4, 2, 32, 32))")
    assert y.shape == (4, 2, 32, 32), "Output shape is wrong!"
