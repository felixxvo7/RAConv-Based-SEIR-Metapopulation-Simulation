import torch
import torch.nn as nn


class ResBlock3D(nn.Module):
    """
    ResBlock that follows Fig. 4 in the paper as closely as possible.

    Structure:
        Input
          -> Conv3D
          -> BatchNorm
             |------------------------------ shortcut #1 -----------------------------|
          -> Conv3D
          -> BatchNorm
          -> ReLU
          -> Conv3D
          -> BatchNorm
          -> Add(shortcut #1)
          -> ReLU
             |------------------------------ shortcut #2 -----------------------------|
          -> Conv3D
          -> BatchNorm
          -> ReLU
          -> Conv3D
          -> BatchNorm
          -> Add(shortcut #2)
          -> ReLU
          -> Output

    Notes:
    - This block has 5 Conv3D layers total, matching the paper figure.
    - The first Conv3D changes channels from in_channels -> out_channels.
    - The remaining 4 Conv3D layers keep out_channels -> out_channels.
    - Because both shortcuts branch AFTER the first Conv3D + BN, they already have
      out_channels, so no 1x1x1 projection is needed inside this block.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm3d(out_channels)

        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.conv3 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn3 = nn.BatchNorm3d(out_channels)

        self.conv4 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn4 = nn.BatchNorm3d(out_channels)

        self.conv5 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn5 = nn.BatchNorm3d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)

        shortcut1 = x

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x = x + shortcut1
        x = self.relu(x)

        shortcut2 = x

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)

        x = x + shortcut2
        x = self.relu(x)

        return x


class ResConv3D(nn.Module):
    """
    Example stack using the paper-style ResBlock.
    """
    def __init__(self):
        super().__init__()

        self.block1 = ResBlock3D(1, 64)
        self.block2 = ResBlock3D(64, 96)
        self.block3 = ResBlock3D(96, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x


if __name__ == "__main__":
    x = torch.randn(2, 1, 14, 16, 16)  # (batch, channels, time, height, width)

    block = ResBlock3D(1, 64)
    y = block(x)
    print("Single block input shape: ", x.shape)
    print("Single block output shape:", y.shape)   # expected: (2, 64, 14, 16, 16)

    model = ResConv3D()
    z = model(x)
    print("Full model output shape:  ", z.shape)   # expected: (2, 128, 14, 16, 16)

    loss = z.mean()
    loss.backward()
    print("Backward pass worked.")
