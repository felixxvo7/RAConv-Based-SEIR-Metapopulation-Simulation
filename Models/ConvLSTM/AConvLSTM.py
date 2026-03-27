import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# ConvLSTMCell  (AConvLSTM cell — exactly as in Eqs 7-9 of the paper)
# ---------------------------------------------------------------------------

class ConvLSTMCell(nn.Module):
    """
    Single AConvLSTM recurrent cell.

    Gate equations (paper Eqs 7-9 and surrounding text):

      Input gate  — spatial attention, NO peephole in Zt:
        Zt = Wi * tanh(Wxi*X + Whi*H + bi)
        it = softmax-like attention of Zt   (div by spatial max per channel)

      Forget gate — standard sigmoid WITH peephole on C_{t-1}:
        ft = σ(Wxf*X + Whf*H + Wcf⊙C + bf)

      Cell candidate:
        g  = tanh(Wxc*X + Whc*H + bc)
        C' = f⊙C + i⊙g

      Output gate — spatial attention WITH peephole on C' (updated cell):
        Zt_o = Wo * tanh(Wxo*X + Who*H + Wco⊙C' + bo)
        ot   = attention(Zt_o)

      Hidden:
        H' = o ⊙ tanh(C')
    """

    def __init__(self, input_channels: int, hidden_channels: int,
                 kernel_size: int = 3, bias: bool = True):
        super().__init__()
        p = kernel_size // 2
        self.hidden_channels = hidden_channels

        # ---- input-to-hidden convolutions ----
        self.Wxi = nn.Conv2d(input_channels,  hidden_channels, kernel_size, padding=p, bias=bias)
        self.Wxf = nn.Conv2d(input_channels,  hidden_channels, kernel_size, padding=p, bias=bias)
        self.Wxc = nn.Conv2d(input_channels,  hidden_channels, kernel_size, padding=p, bias=bias)
        self.Wxo = nn.Conv2d(input_channels,  hidden_channels, kernel_size, padding=p, bias=bias)

        # ---- hidden-to-hidden convolutions ----
        self.Whi = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=p, bias=False)
        self.Whf = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=p, bias=False)
        self.Whc = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=p, bias=False)
        self.Who = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=p, bias=False)

        # ---- peephole weights (lazy-init in init_hidden) ----
        # Only forget gate and output gate have peepholes (as written in the paper).
        # Input gate Zt formula (Eq 7) has NO Wci term.
        self.Wcf = None
        self.Wco = None

        # ---- attention projection kernels (1×1 conv, Wi and Wo in paper) ----
        self.Wi_att = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=True)
        self.Wo_att = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=True)

    # ------------------------------------------------------------------
    def init_hidden(self, batch_size: int, spatial_size: tuple,
                    device: torch.device = None):
        h, w = spatial_size
        device = device or next(self.parameters()).device
        H = torch.zeros(batch_size, self.hidden_channels, h, w, device=device)
        C = torch.zeros(batch_size, self.hidden_channels, h, w, device=device)
        if self.Wcf is None:
            self.Wcf = nn.Parameter(
                torch.zeros(1, self.hidden_channels, h, w, device=device))
            self.Wco = nn.Parameter(
                torch.zeros(1, self.hidden_channels, h, w, device=device))
        return H, C

    # ------------------------------------------------------------------
    @staticmethod
    def _attention(Z: torch.Tensor) -> torch.Tensor:
        """
        Eq 8: A^ij(h) = exp(Z^ij(h)) / max_{i',j'} exp(Z^{i'j'}(h))
        Division by spatial max (not sum) → peak weight = 1, others in (0,1].
        """
        e = torch.exp(Z)
        denom = e.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        return e / denom

    # ------------------------------------------------------------------
    def forward(self, X: torch.Tensor,
                H_prev: torch.Tensor,
                C_prev: torch.Tensor):
        # -- Input gate: attention, NO peephole (Eq 7) --
        Zt_i = self.Wi_att(torch.tanh(self.Wxi(X) + self.Whi(H_prev)))
        i    = self._attention(Zt_i)

        # -- Forget gate: sigmoid WITH peephole on C_{t-1} --
        f = torch.sigmoid(self.Wxf(X) + self.Whf(H_prev) + self.Wcf * C_prev)

        # -- Cell candidate --
        g = torch.tanh(self.Wxc(X) + self.Whc(H_prev))

        # -- Cell update --
        C_new = f * C_prev + i * g

        # -- Output gate: attention WITH peephole on C_new (updated cell) --
        Zt_o = self.Wo_att(torch.tanh(self.Wxo(X) + self.Who(H_prev) + self.Wco * C_new))
        o    = self._attention(Zt_o)

        # -- Hidden state --
        H_new = o * torch.tanh(C_new)
        return H_new, C_new


# ---------------------------------------------------------------------------
# ConvLSTMLayers  — two stacked AConvLSTM cells
# ---------------------------------------------------------------------------

class ConvLSTMLayers(nn.Module):
    """
    Two stacked AConvLSTM layers.
    Input:  [B, T, C, H, W]
    Output: (layer_output_list, last_state_list)
      layer_output_list[l] : [B, T, hidden_ch, H, W]
      last_state_list[l]   : [H_final, C_final]
    """

    def __init__(self, input_channels: int,
                 hidden_channels: list = None,
                 kernel_size: list = None,
                 num_layers: int = 2,
                 bias: bool = True):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = [256, 256]
        if kernel_size is None:
            kernel_size = [3, 3]

        self.num_layers      = num_layers
        self.hidden_channels = hidden_channels

        cells = []
        for i in range(num_layers):
            in_ch = input_channels if i == 0 else hidden_channels[i - 1]
            cells.append(ConvLSTMCell(in_ch, hidden_channels[i],
                                      kernel_size[i], bias))
        self.cell_list = nn.ModuleList(cells)

    def forward(self, x: torch.Tensor):
        b, T, _, h, w = x.size()
        hidden = [self.cell_list[i].init_hidden(b, (h, w))
                  for i in range(self.num_layers)]

        layer_out_list  = []
        last_state_list = []
        cur_input = x

        for l in range(self.num_layers):
            hh, cc    = hidden[l]
            step_outs = []
            for t in range(T):
                hh, cc = self.cell_list[l](
                    X=cur_input[:, t], H_prev=hh, C_prev=cc)
                step_outs.append(hh)
            layer_out  = torch.stack(step_outs, dim=1)   # [B, T, ch, H, W]
            cur_input  = layer_out
            layer_out_list.append(layer_out)
            last_state_list.append([hh, cc])

        return layer_out_list, last_state_list
"""
ConvLSTM with attention (AConvLSTM-style)

Only the input and output gates are changed to use an attention mechanism.
"""

import torch
import torch.nn as nn


class AConvLSTMCell(nn.Module):
    """a
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
    use_attention: bool
        Whether to use attention-based gates or standard sigmoid gates.
    """

    def __init__(self, input_channels, hidden_channels, kernel_size, bias, use_attention=True):
        super().__init__()

        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.use_attention = use_attention

        # W_x*
        self.Wxi = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=padding, bias=bias)
        self.Wxf = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=padding, bias=bias)
        self.Wxc = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=padding, bias=bias)
        self.Wxo = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=padding, bias=bias)

        # W_h*
        self.Whi = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding, bias=bias)
        self.Whf = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding, bias=bias)
        self.Whc = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding, bias=bias)
        self.Who = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding, bias=bias)

        # Peepholes
        self.Wci = None
        self.Wcf = None
        self.Wco = None

        if self.use_attention:
            # Attention conv for input gate (Wi in the paper)
            self.Wi_att = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, padding=0, bias=True)
            # Attention conv for output gate (Wo in the paper)
            self.Wo_att = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, padding=0, bias=True)

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

    # Shared helper to turn a feature map into attention gate via exp / max
    def _attention_gate(self, Z):
        """
        Z: [B, C, H, W]
        Returns attention tensor A in (0, 1] by channel-wise max-normalization.
        Numerically stable: subtract spatial max before exp to prevent overflow.
        """
        Z_max = Z.amax(dim=(-2, -1), keepdim=True)
        Z_exp = torch.exp(Z - Z_max)
        max_per_channel = Z_exp.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        return Z_exp / max_per_channel

    def forward(self, X, H_prev, C_prev):
        # ======================= INPUT GATE =======================
        # i_t attention gate
        if self.use_attention:
            # pre-activation for input gate including peephole, then tanh as in Eq. (7)
            pre_i = self.Wxi(X) + self.Whi(H_prev)
            pre_i = torch.tanh(pre_i)                     # tanh before attention projection
            # project with Wi_att and build attention via exp / max
            Z_i = self.Wi_att(pre_i)
            i = self._attention_gate(Z_i)                 # attention-based input gate
        else:
            # i_t = s(Wxi*Xt + Whi*Ht-1 + Wci?Ct-1 + bi)
            i = torch.sigmoid(self.Wxi(X) + self.Whi(H_prev) + self.Wci * C_prev)

        # ======================= FORGET GATE =======================
        # f_t = s(Wxf*Xt + Whf*Ht-1 + Wcf?Ct-1 + bf)
        f = torch.sigmoid(
            self.Wxf(X)
            + self.Whf(H_prev)
            + self.Wcf * C_prev
        )

        # ======================= CELL UPDATE =======================
        # C_t candidate = tanh(Wxc*Xt + Whc*Ht-1 + bc)
        g = torch.tanh(self.Wxc(X) + self.Whc(H_prev))
        # C_t = f_t?C_t-1 + i_t?g_t
        C = f * C_prev + i * g

        # ======================= OUTPUT GATE =======================
        if self.use_attention:
            # pre-activation for output gate including peephole, then tanh
            pre_o = self.Wxo(X) + self.Who(H_prev) + self.Wco * C
            pre_o = torch.tanh(pre_o)                    # tanh before attention projection
            # project with Wo_att and build attention gate
            Z_o = self.Wo_att(pre_o)
            o = self._attention_gate(Z_o)                # attention-based output gate
        else:
            # o_t = s(Wxo*Xt + Who*Ht-1 + Wco?Ct + bo)
            o = torch.sigmoid(
                self.Wxo(X)
                + self.Who(H_prev)
                + self.Wco * C
            )

        # H_t = o_t?tanh(C_t)
        H = o * torch.tanh(C)

        return H, C


class AConvLSTMLayers(nn.Module):
    """
    Parameters:
        input_channels: Number of channels in input
        hidden_channels: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        bias: Bias or no bias in Convolution
        use_attention: Whether to use attention-based gates or standard sigmoid gates
    Input:
        A tensor of size B, T, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    """

    def __init__(self, input_channels, hidden_channels=[256,256], kernel_size=[3,3], num_layers=2, bias=True, use_attention=True):
        super().__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.bias = bias
        self.num_layers = num_layers # Fixed to 2 layers as per Figure 6
        self.use_attention = use_attention
        self.output_conv = nn.Conv2d(self.hidden_channels[-1], 1, kernel_size=1)

        # Other than first layer, input channels = hidden channels of previous layer
        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_channels = self.input_channels if i == 0 else self.hidden_channels[i - 1]

            cell_list.append(AConvLSTMCell(input_channels=cur_input_channels,
                                          hidden_channels=self.hidden_channels[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias,
                                          use_attention=self.use_attention))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor):
        """
        Parameters
        ----------
        input_tensor:
            5-D Tensor of shape (b, t, c, h, w)
        Returns
        -------
        layer_output_list, last_state_list
        """
        b, _, _, h, w = input_tensor.size()

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

    def predict_future(self, last_state_list, Q, first_input):
        """
        Generate Q future steps using autoregressive roll-forward.

        Parameters
        ----------
        last_state_list: list of (h, c) for each layer
        Q: number of future steps
        first_input: (B, C, H, W) ? usually last frame of input

        Returns
        -------
        predictions: (B, Q, 1, H, W)
        """

        cur_input = first_input
        predictions = []

        # Copy states (important to avoid modifying original)
        hidden_states = [(h.clone(), c.clone()) for (h, c) in last_state_list]

        for _ in range(Q):
            new_hidden_states = []

            x = cur_input  # input to first layer

            # pass through stacked ConvLSTM layers
            for layer_idx in range(self.num_layers):
                h, c = hidden_states[layer_idx]

                h, c = self.cell_list[layer_idx](
                    X=x,
                    H_prev=h,
                    C_prev=c
                )

                new_hidden_states.append((h, c))
                x = h  # output becomes next layer input

            # Project to prediction (256 ? 1)
            pred = self.output_conv(x)

            predictions.append(pred)

            # autoregressive: use prediction as next input
            cur_input = pred

            hidden_states = new_hidden_states

        predictions = torch.stack(predictions, dim=1)  # (B, Q, 1, H, W)
        return predictions
