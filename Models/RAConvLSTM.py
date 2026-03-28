import torch
import torch.nn as nn
from ResBlock.ResBlock import ResBlock3D
from ConvLSTM.AConvLSTM import AConvLSTMLayers

class RAConv(nn.Module):
    def __init__(self, in_channels=1, out_steps=4):
        """
        RAConv Implementation using Autoregressive Prediction.
        
        Args:
            in_channels (int): Number of input channels (usually 1).
            out_steps (int): Number of future steps to predict (Q).
        """
        super(RAConv, self).__init__()

        # --- 1. Initial 3D Convolutional Layer ---
        self.conv_init = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )

        # --- 2. ResConv3D Module (Spatial-Temporal Feature Extraction) ---
        self.res_block1 = ResBlock3D(64, 64)
        self.res_block2 = ResBlock3D(64, 96)
        self.res_block3 = ResBlock3D(96, 128)

        # --- 3. AConvLSTM Module ---
        # Note: AConvLSTMLayers already contains the 256->1 output_conv 
        # needed for the predict_future method.
        self.aconv_lstm = AConvLSTMLayers(
            input_channels=128,
            hidden_channels=[256, 256],
            kernel_size=[3, 3],
            num_layers=2,
            use_attention=True
        )
        
        self.out_steps = out_steps

    def forward(self, x):
        # --- Step 1 & 2: 3D Residual Learning ---
        # x starts as (B, 1, T, H, W)
        x = self.conv_init(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        feat_vol = self.res_block3(x) # Shape: (B, 128, T, H, W)

        # --- Step 3: Recurrent Encoding ---
        # Permute for LSTM: (B, 128, T, H, W) -> (B, T, 128, H, W)
        feat_seq = feat_vol.permute(0, 2, 1, 3, 4)
        
        # Get hidden states by processing the whole history
        _, last_state_list = self.aconv_lstm(feat_seq)

        # NEW: Get the 128-channel feature map of ONLY the last time step
        # This matches the 128-channel input the LSTM expects.
        last_feature_map = feat_seq[:, -1, :, :, :] # Shape: (B, 128, H, W)

        # --- Step 4: Autoregressive Future Prediction ---
        # Pass the 128-channel feature map, not the 1-channel raw frame
        predictions = self.aconv_lstm.predict_future(
            last_state_list=last_state_list, 
            Q=self.out_steps, 
            first_input_feature=last_feature_map
        )

        return predictions

# Example Usage
if __name__ == "__main__":
    # Parameters: Batch=2, Channel=1, TimeSteps=8, Grid=16x16
    P = 14  # Historical steps
    Q = 4  # Prediction steps
    dummy_input = torch.randn(2, 1, P, 16, 16)
    
    model = RAConv(in_channels=1, out_steps=Q)
    
    # The output will be the predicted sequence
    output = model(dummy_input)
    
    print(f"Input shape (B, C, P, H, W): {dummy_input.shape}")
    print(f"Output shape (B, Q, 1, H, W): {output.shape}")