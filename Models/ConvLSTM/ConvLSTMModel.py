'''
ConvLSTM with Conv2d output layer (decoder) for frame prediction.

ConvLSTM → Conv2d → prediction
'''

from ConvLSTM import ConvLSTM
import torch
import torch.nn as nn
import torch.optim as optim

class ConvLSTMModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.convlstm = ConvLSTM(
            input_channels=1,
            hidden_channels=16,
            kernel_size=3
        )

        # Extra conv layer to map hidden state to prediction
        self.conv_out = nn.Conv2d(16, 1, kernel_size=3, padding=1)

    def forward(self, x):

        # (B, T, hidden, H, W)
        output = self.convlstm(x)

        # last timestep
        h_last = output[:, -1]

        prediction = self.conv_out(h_last)

        return prediction

# --- Training setup ---
def main():
    # Hyperparameters
    batch_size = 4
    seq_len = 10
    channels = 1
    hidden_channels = 16
    height, width = 32, 32
    epochs = 30  # keep small for demo
    lr = 1e-3
    
    model = ConvLSTMModel()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):

        x = torch.randn(batch_size, seq_len, channels, height, width)

        # predict last frame
        target = x[:, -1]

        pred = model(x)

        loss = criterion(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    main()
