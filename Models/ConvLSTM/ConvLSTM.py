"""
ConvLSTM

Acknowledgement: This file is modified upon the implementation of https://github.com/KL4805/ConvLSTM-Pytorch?tab=readme-ov-file
"""

import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_channels: int
            Number of channels of input tensor.
        hidden_channels: int
            Number of channels of hidden state.
        kernel_size: int
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
    """
    def __init__(self, input_channels, hidden_channels, kernel_size, bias):
        super().__init__()

        padding = kernel_size // 2
        self.hidden_channels = hidden_channels

        # W_x* terms (input convolutions)
        self.Wxi = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=padding, bias=bias)
        self.Wxf = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=padding, bias=bias)
        self.Wxc = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=padding, bias=bias)
        self.Wxo = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=padding, bias=bias)

        # W_h* terms (hidden state convolutions)
        self.Whi = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding, bias=bias)
        self.Whf = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding, bias=bias)
        self.Whc = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding, bias=bias)
        self.Who = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding, bias=bias)

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


class ConvLSTMLayers(nn.Module):
    """
    Parameters:
        input_channels: Number of channels in input
        hidden_channels: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        bias: Bias or no bias in Convolution
    Input:
        A tensor of size B, T, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    """

    def __init__(self, input_channels, hidden_channels=[256, 256], kernel_size=[3, 3], num_layers=2, bias=True):
        super().__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.bias = bias
        self.num_layers = num_layers # Fixed to 2 layers as per Figure 6

        # Other than first layer, input channels = hidden channels of previous layer
        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_channels = self.input_channels if i == 0 else self.hidden_channels[i - 1]

            cell_list.append(ConvLSTMCell(input_channels=cur_input_channels,
                                          hidden_channels=self.hidden_channels[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

        # 256 to 1 channel mapping for final prediction layer
        # # Layer 1: Captures long-term spatial-temporal dependencies 
        # # Configured with 256 hidden channels
        # self.convlstm1 = ConvLSTMCell(
        #     input_channels=input_channels, 
        #     hidden_channels=256, 
        #     kernel_size=kernel_size,
        #     bias=True
        # )
        
        # # Layer 2: Final prediction layer 
        # # While Figure 6 labels both as 256, the final hidden state must match
        # # the 1-channel ground truth to be the "final predicted result"
        # self.convlstm2 = ConvLSTMCell(
        #     input_channels=256, 
        #     hidden_channels=1, 
        #     kernel_size=kernel_size, 
        #     bias=True
        # )

        # self.cell_list = nn.ModuleList()
        # self.cell_list.append(self.convlstm1)
        # self.cell_list.append(self.convlstm2)

    def forward(self, input_tensor):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        hidden_state = []
        for i in range(self.num_layers):
            hidden_state.append(self.cell_list[i].init_hidden(b, (h, w)))

        layer_output_list = []
        last_state_list = []

        # T/sequence length
        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](
                    X=cur_layer_input[:, t, :, :, :],
                    H_prev=h,
                    C_prev=c
                )
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        return layer_output_list, last_state_list


# Test
import torch.optim as optim
def main():
    # Simple test run of ConvLSTMLayers
    # model = ConvLSTMLayers(input_channels=64)
    # x = torch.rand(2, 10, 64, 16, 16)
    # out = model(x)
    
    # Train 2 layer ConvLSTM (first: 64 -> 256, second: 256 -> 64) on random data to predict last frame from previous frames
    # Hyperparameters
    batch_size = 1
    seq_len = 10
    input_channels = 64
    hidden_channels = 256
    height, width = 16, 16
    epochs = 10  # keep small for demo
    lr = 1e-3
    
    model = ConvLSTMLayers(input_channels, [hidden_channels, input_channels], [3, 3], 2, True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    x = torch.rand(batch_size, seq_len, input_channels, height, width)
    for epoch in range(epochs):

        # predict last frame
        target = x[:, -1]

        pred = model(x)[0][-1]

        loss = criterion(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    main()