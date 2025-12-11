import torch
import torch.nn as nn

# ---------------------------------------
# Double Conv Block (Conv → ReLU → Conv → ReLU)
# ---------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# ---------------------------------------
# SoftCleanNet (fast hologram removal UNet)
# ---------------------------------------
class SoftCleanNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Down
        self.down1 = DoubleConv(1, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        # Middle
        self.middle = DoubleConv(64, 128)

        # Up
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv_up1 = DoubleConv(128, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv_up2 = DoubleConv(64, 32)

        # Output
        self.final = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        # Down
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        mid = self.middle(self.pool2(d2))

        # Up
        u1 = self.up1(mid)
        u1 = torch.cat([u1, d2], dim=1)
        u1 = self.conv_up1(u1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, d1], dim=1)
        u2 = self.conv_up2(u2)

        return self.final(u2)
