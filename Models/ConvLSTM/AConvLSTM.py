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
