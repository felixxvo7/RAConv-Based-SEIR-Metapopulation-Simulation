"""
Attention-based ConvLSTM (AConvLSTM)

Architecture: Conv3D → AConvLSTM×2 → output

Only the input and output gates use an attention mechanism
(exp / spatial-max normalisation) instead of standard sigmoid gates.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# AConvLSTMCell  — single recurrent cell
# ---------------------------------------------------------------------------

class AConvLSTMCell(nn.Module):
    """
    Single AConvLSTM recurrent cell.

    Parameters
    ----------
    input_channels : int
    hidden_channels : int
    kernel_size : int
    bias : bool
    use_attention : bool
        If True, input and output gates use exp/max attention instead of sigmoid.
    """

    def __init__(self, input_channels, hidden_channels, kernel_size,
                 bias, use_attention=True):
        super().__init__()

        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.use_attention = use_attention

        # input-to-hidden
        self.Wxi = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=padding, bias=bias)
        self.Wxf = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=padding, bias=bias)
        self.Wxc = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=padding, bias=bias)
        self.Wxo = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=padding, bias=bias)

        # hidden-to-hidden
        self.Whi = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding, bias=bias)
        self.Whf = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding, bias=bias)
        self.Whc = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding, bias=bias)
        self.Who = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding, bias=bias)

        # peepholes — shape resolved at first init_hidden call
        self.Wci = None
        self.Wcf = None
        self.Wco = None

        if self.use_attention:
            self.Wi_att = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=True)
            self.Wo_att = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=True)

    # ------------------------------------------------------------------
    def init_hidden(self, batch_size, spatial_size, device=None):
        h, w = spatial_size
        device = device or next(self.parameters()).device
        H = torch.zeros(batch_size, self.hidden_channels, h, w, device=device)
        C = torch.zeros(batch_size, self.hidden_channels, h, w, device=device)
        if self.Wci is None:
            self.Wci = nn.Parameter(torch.zeros(1, self.hidden_channels, h, w, device=device))
            self.Wcf = nn.Parameter(torch.zeros(1, self.hidden_channels, h, w, device=device))
            self.Wco = nn.Parameter(torch.zeros(1, self.hidden_channels, h, w, device=device))
        return H, C

    # ------------------------------------------------------------------
    @staticmethod
    def _attention_gate(Z):
        """
        A^ij(h) = exp(Z^ij(h) - max) / max_exp   →  values in (0, 1]
        Numerically stable via subtract-max before exp.
        """
        Z_max = Z.amax(dim=(-2, -1), keepdim=True)
        Z_exp = torch.exp(Z - Z_max)
        return Z_exp / Z_exp.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-6)

    # ------------------------------------------------------------------
    def forward(self, X, H_prev, C_prev):
        # Input gate
        if self.use_attention:
            pre_i = torch.tanh(self.Wxi(X) + self.Whi(H_prev))
            i = self._attention_gate(self.Wi_att(pre_i))
        else:
            i = torch.sigmoid(self.Wxi(X) + self.Whi(H_prev) + self.Wci * C_prev)

        # Forget gate (sigmoid + peephole on C_prev)
        f = torch.sigmoid(self.Wxf(X) + self.Whf(H_prev) + self.Wcf * C_prev)

        # Cell candidate & update
        g = torch.tanh(self.Wxc(X) + self.Whc(H_prev))
        C = f * C_prev + i * g

        # Output gate
        if self.use_attention:
            pre_o = torch.tanh(self.Wxo(X) + self.Who(H_prev) + self.Wco * C)
            o = self._attention_gate(self.Wo_att(pre_o))
        else:
            o = torch.sigmoid(self.Wxo(X) + self.Who(H_prev) + self.Wco * C)

        H = o * torch.tanh(C)
        return H, C


# ---------------------------------------------------------------------------
# AConvLSTMLayers  — stacked recurrent cells (pure encoder, no output head)
# ---------------------------------------------------------------------------

class AConvLSTMLayers(nn.Module):
    """
    Stacks ``num_layers`` AConvLSTMCell modules.

    Input : (B, T, C, H, W)
    Output: (layer_output_list, last_state_list)
        layer_output_list[l] : (B, T, hidden_channels[l], H, W)
        last_state_list[l]   : [H_final, C_final]
    """

    def __init__(self, input_channels, hidden_channels=None, kernel_size=None,
                 num_layers=2, bias=True, use_attention=True):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = [256, 256]
        if kernel_size is None:
            kernel_size = [3, 3]

        self.num_layers      = num_layers
        self.hidden_channels = hidden_channels

        cell_list = []
        for i in range(num_layers):
            in_ch = input_channels if i == 0 else hidden_channels[i - 1]
            cell_list.append(
                AConvLSTMCell(in_ch, hidden_channels[i], kernel_size[i],
                              bias, use_attention)
            )
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor):
        """input_tensor: (B, T, C, H, W)"""
        b, T, _, h, w = input_tensor.size()

        hidden = [self.cell_list[i].init_hidden(b, (h, w))
                  for i in range(self.num_layers)]

        layer_output_list = []
        last_state_list   = []
        cur_input = input_tensor

        for l in range(self.num_layers):
            hh, cc    = hidden[l]
            step_outs = []
            for t in range(T):
                hh, cc = self.cell_list[l](
                    X=cur_input[:, t], H_prev=hh, C_prev=cc)
                step_outs.append(hh)
            layer_out = torch.stack(step_outs, dim=1)   # (B, T, ch, H, W)
            cur_input = layer_out
            layer_output_list.append(layer_out)
            last_state_list.append([hh, cc])

        return layer_output_list, last_state_list


# ---------------------------------------------------------------------------
# AConvLSTMModel  — full pipeline: Conv3D → AConvLSTM×2 → output
# ---------------------------------------------------------------------------

class AConvLSTMModel(nn.Module):
    """
    Full model::

        (B, T, 1, H, W)
          ──► Conv3D  (temporal + spatial feature extraction)
          ──► AConvLSTM × num_layers  (recurrent encoding)
          ──► Conv2D 1×1  (output projection to 1 channel)

    The Conv3D frontend uses ``kernel=(3,3,3)`` with ``padding=(1,1,1)``
    to preserve the (T, H, W) dimensions.

    For autoregressive future prediction a lightweight ``frame_proj``
    (Conv2D) maps individual frames — which may have a different channel
    count than the Conv3D output — into the same feature space, so that
    the recurrent cells receive consistent input at every step.

    Parameters
    ----------
    in_channels : int
        Channels in the raw input frames (typically 1).
    conv3d_channels : int
        Output channels of the Conv3D frontend (and input to AConvLSTM).
    hidden_channels : list[int]
        Hidden channels per AConvLSTM layer.
    kernel_size : list[int]
        Convolutional kernel sizes per AConvLSTM layer.
    num_layers : int
        Number of stacked AConvLSTM layers.
    bias : bool
    use_attention : bool
    """

    def __init__(self, in_channels=1, conv3d_channels=32,
                 hidden_channels=None, kernel_size=None,
                 num_layers=2, bias=True, use_attention=True):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = [256, 256]
        if kernel_size is None:
            kernel_size = [3, 3]

        # ── Conv3D frontend ──────────────────────────────────────────────
        # Input layout for nn.Conv3d: (B, C, D, H, W) where D = T
        self.conv3d = nn.Conv3d(
            in_channels, conv3d_channels,
            kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=bias
        )
        self.conv3d_act = nn.ReLU(inplace=True)

        # ── Single-frame projection for autoregressive prediction ────────
        # Maps (B, in_channels, H, W) → (B, conv3d_channels, H, W)
        self.frame_proj = nn.Sequential(
            nn.Conv2d(in_channels, conv3d_channels,
                      kernel_size=3, padding=1, bias=bias),
            nn.ReLU(inplace=True),
        )

        # ── Recurrent stack ──────────────────────────────────────────────
        self.aconvlstm = AConvLSTMLayers(
            input_channels=conv3d_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            num_layers=num_layers,
            bias=bias,
            use_attention=use_attention,
        )

        # ── Output projection ────────────────────────────────────────────
        self.output_conv = nn.Conv2d(hidden_channels[-1], 1, kernel_size=1)

    # ------------------------------------------------------------------
    def forward(self, x):
        """
        Parameters
        ----------
        x : (B, T, C, H, W)

        Returns
        -------
        layer_output_list, last_state_list  — same contract as AConvLSTMLayers
        """
        B, T, C, H, W = x.shape
        # (B, T, C, H, W) → (B, C, T, H, W) for Conv3D
        feat = self.conv3d_act(self.conv3d(x.permute(0, 2, 1, 3, 4)))
        # (B, conv3d_ch, T, H, W) → (B, T, conv3d_ch, H, W) for AConvLSTM
        feat = feat.permute(0, 2, 1, 3, 4)
        return self.aconvlstm(feat)

    # ------------------------------------------------------------------
    def predict_future(self, last_state_list, Q, first_input):
        """
        Autoregressive roll-forward for Q steps.

        Parameters
        ----------
        last_state_list : list of [H, C] tensors  (from forward())
        Q : int  — number of future steps
        first_input : (B, C, H, W)  — last raw frame of the input sequence

        Returns
        -------
        predictions : (B, Q, 1, H, W)
        """
        cur_input = self.frame_proj(first_input)          # (B, conv3d_ch, H, W)
        hidden_states = [(h.clone(), c.clone()) for (h, c) in last_state_list]
        predictions = []

        for _ in range(Q):
            new_states = []
            x = cur_input
            for l in range(self.aconvlstm.num_layers):
                h, c = hidden_states[l]
                h, c = self.aconvlstm.cell_list[l](X=x, H_prev=h, C_prev=c)
                new_states.append((h, c))
                x = h

            pred = self.output_conv(x)                    # (B, 1, H, W)
            predictions.append(pred)
            cur_input = self.frame_proj(pred)             # project for next step
            hidden_states = new_states

        return torch.stack(predictions, dim=1)            # (B, Q, 1, H, W)
