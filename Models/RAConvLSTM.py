import torch
import torch.nn as nn

from ResBlock.ResBlock import ResBlock3D
from AConvLSTM.AConvLSTM import AConvLSTMLayers


class RAConv(nn.Module):
    def __init__(self, in_channels=1, out_steps=4):
        """
        RAConv Implementation using Autoregressive Prediction.

        Args:
            in_channels (int): Number of input channels (usually 1).
            out_steps (int): Number of future steps to predict (Q).
        """
        super(RAConv, self).__init__()

        self.conv_init = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )

        self.res_block1 = ResBlock3D(64, 64)
        self.res_block2 = ResBlock3D(64, 96)
        self.res_block3 = ResBlock3D(96, 128)

        self.aconv_lstm = AConvLSTMLayers(
            input_channels=128,
            hidden_channels=[256, 256],
            kernel_size=[3, 3],
            num_layers=2,
            use_attention=True
        )

        self.out_steps = out_steps

    def forward(self, x):
        x = self.conv_init(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        feat_vol = self.res_block3(x)

        # Permute for LSTM: (B, 128, T, H, W) -> (B, T, 128, H, W)
        feat_seq = feat_vol.permute(0, 2, 1, 3, 4)

        _, last_state_list = self.aconv_lstm(feat_seq)

        # Use the 128-channel feature map of the last time step,
        # matching the channel dimension the LSTM expects as input.
        last_feature_map = feat_seq[:, -1, :, :, :]

        predictions = self.aconv_lstm.predict_future(
            last_state_list=last_state_list,
            Q=self.out_steps,
            first_input=last_feature_map
        )

        return predictions


if __name__ == "__main__":
    P = 14
    Q = 4
    dummy_input = torch.randn(2, 1, P, 16, 16)

    model = RAConv(in_channels=1, out_steps=Q)
    output = model(dummy_input)

    print(f"Input shape (B, C, P, H, W): {dummy_input.shape}")
    print(f"Output shape (B, Q, 1, H, W): {output.shape}")
