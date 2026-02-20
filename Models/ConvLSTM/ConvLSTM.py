# ConvLSTM

import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super().__init__()

        padding = kernel_size // 2
        self.hidden_channels = hidden_channels

        # W_x* terms (input convolutions)
        self.Wxi = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=padding, bias=True)
        self.Wxf = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=padding, bias=True)
        self.Wxc = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=padding, bias=True)
        self.Wxo = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=padding, bias=True)

        # W_h* terms (hidden state convolutions)
        self.Whi = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding, bias=False)
        self.Whf = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding, bias=False)
        self.Whc = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding, bias=False)
        self.Who = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding, bias=False)

        # Peephole connections (Hadamard weights)
        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def init_hidden(self, batch_size, spatial_size, device=None):
        height, width = spatial_size
        device = device or next(self.parameters()).device

        H = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
        C = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)

        # Initialize peephole parameters once spatial size is known
        if self.Wci is None:
            self.Wci = nn.Parameter(torch.zeros(1, self.hidden_channels, height, width, device=device))
            self.Wcf = nn.Parameter(torch.zeros(1, self.hidden_channels, height, width, device=device))
            self.Wco = nn.Parameter(torch.zeros(1, self.hidden_channels, height, width, device=device))

        return H, C

    def forward(self, X, H_prev, C_prev):
        # i_t = σ(Wxi*Xt + Whi*Ht−1 + Wci◦Ct−1 + bi)
        i = torch.sigmoid(
            self.Wxi(X)
            + self.Whi(H_prev)
            + self.Wci * C_prev
        )

        # f_t = σ(Wxf*Xt + Whf*Ht−1 + Wcf◦Ct−1 + bf)
        f = torch.sigmoid(
            self.Wxf(X)
            + self.Whf(H_prev)
            + self.Wcf * C_prev
        )

        # C_t = f_t◦C_t−1 + i_t◦tanh(Wxc*Xt + Whc*Ht−1 + bc)
        C = f * C_prev + i * torch.tanh(
            self.Wxc(X) + self.Whc(H_prev)
        )

        # o_t = σ(Wxo*Xt + Who*Ht−1 + Wco◦Ct + bo)
        o = torch.sigmoid(
            self.Wxo(X)
            + self.Who(H_prev)
            + self.Wco * C
        )

        # H_t = o_t◦tanh(C_t)
        H = o * torch.tanh(C)

        return H, C
    
class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super().__init__()
        self.cell = ConvLSTMCell(input_channels, hidden_channels, kernel_size)

    def forward(self, X):
        # X: (batch, time, channels, height, width)
        B, T, C, H, W = X.shape

        H_t, C_t = self.cell.init_hidden(B, (H, W), X.device)

        outputs = []

        for t in range(T):
            H_t, C_t = self.cell(X[:, t], H_t, C_t)
            outputs.append(H_t)

        return torch.stack(outputs, dim=1)
