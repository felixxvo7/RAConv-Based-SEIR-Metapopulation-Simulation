"""
RAConv: Residual 3D Convolutional Network + Attention ConvLSTM.

Paper: Cellular Traffic Prediction Using Deep Convolutional Neural
       Network with Attention Mechanism.
       Wang & Wong, ICC 2022.

Architecture (exactly as described in the paper, Figure 6):

  INPUT  [B, P, 1, H, W]
    │
  Conv3D (64 ch, 3×3×3) → BN → ReLU
    │
  ResBlock (64 ch)
    │
  ResBlock (96 ch)   ← 1×1×1 projection 64→96 between blocks
    │
  ResBlock (128 ch)  ← 1×1×1 projection 96→128 between blocks
    │
  AConvLSTM layer 1 (256 ch)
    │
  AConvLSTM layer 2 (256 ch)
    │
  Conv2d 1×1  (256 → Q)   ← direct multi-step output projection
    │
  OUTPUT [B, Q, 1, H, W]

Key paper equations (Section III-B-2, Eqs 7-9):
  Input gate (attention):
    Zt   = Wi * tanh(Wxi*Xt + Whi*Ht-1 + bi)       (no peephole on Zt)
    A^ij = exp(Z^ij) / max_{i',j'} exp(Z^{i'j'})   per channel
    it   = {A^ij(h)}

  Forget gate (standard sigmoid with peephole):
    ft = σ(Wxf*Xt + Whf*Ht-1 + Wcf⊙Ct-1 + bf)

  Cell:
    Ct = ft⊙Ct-1 + it⊙tanh(Wxc*Xt + Whc*Ht-1 + bc)

  Output gate (attention, peephole on Ct):
    Zt_o = Wi_o * tanh(Wxo*Xt + Who*Ht-1 + Wco⊙Ct + bo)
    ot   = attention(Zt_o)

  Hidden:
    Ht = ot ⊙ tanh(Ct)

The paper's output: the FINAL hidden state of the last AConvLSTM layer
is projected directly to Q channels via a 1×1 Conv2d — NO autoregressive
decoder.  "The output of the last layer is the final predicted results."
(Section III-B-3, paper page 2343).
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# ResBlock3D
# ---------------------------------------------------------------------------
# Five 3×3×3 Conv3D layers with an identity shortcut.
# No spatial / temporal downsampling — stride=1, same-padding throughout.
# ---------------------------------------------------------------------------

class ResBlock3D(nn.Module):
    """One residual building block with five Conv3D layers (Fig. 4 of paper)."""

    def __init__(self, channels: int):
        super().__init__()

        def _conv_bn(apply_relu: bool):
            layers = [
                nn.Conv3d(channels, channels,
                          kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm3d(channels),
            ]
            if apply_relu:
                layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        self.layer1 = _conv_bn(True)
        self.layer2 = _conv_bn(True)
        self.layer3 = _conv_bn(True)
        self.layer4 = _conv_bn(True)
        self.layer5 = _conv_bn(False)   # no ReLU; shortcut added before final ReLU
        self.relu   = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.relu(out + identity)
        return out


# ---------------------------------------------------------------------------
# ResConv3D
# ---------------------------------------------------------------------------
# Channel progression: 1 → 64 → 96 → 128  (Figure 6).
# Resolution [T, H, W] is preserved throughout.
# ---------------------------------------------------------------------------

class ResConv3D(nn.Module):
    """3D residual convolutional encoder (ResConv3D module)."""

    def __init__(self, in_channels: int = 1, hidden_3d: list = None):
        super().__init__()
        if hidden_3d is None:
            hidden_3d = [64, 96, 128]
        c0, c1, c2 = hidden_3d

        # Entry Conv3D 3×3×3: lifts raw input to c0 channels.
        self.init_conv = nn.Sequential(
            nn.Conv3d(in_channels, c0,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(c0),
            nn.ReLU(inplace=True),
        )

        self.res1  = ResBlock3D(c0)
        self.proj1 = nn.Sequential(
            nn.Conv3d(c0, c1, kernel_size=1, bias=False),
            nn.BatchNorm3d(c1),
            nn.ReLU(inplace=True),
        )
        self.res2  = ResBlock3D(c1)
        self.proj2 = nn.Sequential(
            nn.Conv3d(c1, c2, kernel_size=1, bias=False),
            nn.BatchNorm3d(c2),
            nn.ReLU(inplace=True),
        )
        self.res3  = ResBlock3D(c2)
        self.out_channels = c2  # 128

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_ch, T, H, W]
        out = self.init_conv(x)     # [B, 64,  T, H, W]
        out = self.res1(out)        # [B, 64,  T, H, W]
        out = self.proj1(out)       # [B, 96,  T, H, W]
        out = self.res2(out)        # [B, 96,  T, H, W]
        out = self.proj2(out)       # [B, 128, T, H, W]
        out = self.res3(out)        # [B, 128, T, H, W]
        return out


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


# ---------------------------------------------------------------------------
# Weight initialisation
# ---------------------------------------------------------------------------

def init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.Conv3d)):
        nn.init.kaiming_normal_(module.weight, mode='fan_in',
                                nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.constant_(module.weight, 1.0)
        nn.init.constant_(module.bias,   0.0)


# ---------------------------------------------------------------------------
# RAConv  — full end-to-end model
# ---------------------------------------------------------------------------
# Output:  the final hidden state of layer-2 AConvLSTM is projected by a
#          1×1 Conv2d directly to Q channels, then reshaped.
#          "The output of the last layer is the final predicted results."
#          (Section III-B-3).  No autoregressive decoder.
# ---------------------------------------------------------------------------

class RAConv(nn.Module):
    """
    Full RAConv model (paper-faithful).

    Input  : [B, P, 1, H, W]
    Output : [B, Q, 1, H, W]
    """

    def __init__(self,
                 in_channels: int = 1,
                 hidden_3d: list   = None,
                 hidden_lstm: int  = 256,
                 P: int = 8,
                 Q: int = 4):
        super().__init__()
        if hidden_3d is None:
            hidden_3d = [64, 96, 128]

        self.P = P
        self.Q = Q

        # --- Stage 1: 3-D residual feature extractor ---
        self.resconv3d  = ResConv3D(in_channels=in_channels, hidden_3d=hidden_3d)
        res_out_ch      = self.resconv3d.out_channels   # 128

        # --- Stage 2: two-layer AConvLSTM encoder ---
        self.encoder_lstm = ConvLSTMLayers(
            input_channels  = res_out_ch,
            hidden_channels = [hidden_lstm, hidden_lstm],
            kernel_size     = [3, 3],
            num_layers      = 2,
            bias            = True,
        )

        # --- Stage 3: direct output projection (paper Fig. 6) ---
        # The final hidden-state [B, 256, H, W] is mapped to Q channels,
        # then each channel becomes one predicted frame.
        self.out_conv = nn.Conv2d(hidden_lstm, Q, kernel_size=1, bias=True)

        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, P, 1, H, W]
        B, T, C, H, W = x.shape

        # Conv3D expects [B, C, T, H, W]
        x3d      = x.permute(0, 2, 1, 3, 4)        # [B, 1,   P, H, W]
        feat     = self.resconv3d(x3d)              # [B, 128, P, H, W]
        feat_seq = feat.permute(0, 2, 1, 3, 4)     # [B, P, 128, H, W]

        _, last_states = self.encoder_lstm(feat_seq)

        # Final hidden state of the top LSTM layer
        h_final = last_states[-1][0]                # [B, 256, H, W]

        # Project to Q channels then split into Q single-channel frames.
        # out_conv: [B, 256, H, W] → [B, Q, H, W]
        out = self.out_conv(h_final)                # [B, Q,   H, W]
        out = out.unsqueeze(2)                      # [B, Q, 1, H, W]
        return out


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def main():
    P, Q    = 8, 4
    B, H, W = 2, 16, 16

    model = RAConv(in_channels=1, hidden_3d=[64, 96, 128],
                   hidden_lstm=256, P=P, Q=Q)

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total:,}")

    x = torch.rand(B, P, 1, H, W)
    print(f"Input  shape: {list(x.shape)}")

    with torch.no_grad():
        y = model(x)

    print(f"Output shape: {list(y.shape)}")
    assert y.shape == (B, Q, 1, H, W), f"Expected {(B, Q, 1, H, W)}, got {tuple(y.shape)}"
    print("Smoke-test passed.")


if __name__ == "__main__":
    main()
