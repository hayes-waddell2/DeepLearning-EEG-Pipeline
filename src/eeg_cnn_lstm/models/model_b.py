"""
@file model.py
@brief CNN + LSTM binary classifier for preprocessed EEG epochs.

@details
Convolutional Recurrent Neural Network (CRNN) for the TUH EEG Abnormal
Corpus (TUAB) classification task. The CNN front-end acts as a feature
extractor over the raw time-series, reducing each 2500-sample window to
a shorter sequence of 128-dim feature vectors. The bidirectional LSTM
models how those features evolve across the window. A small MLP head
projects the LSTM's final hidden state to a single logit suitable for
`torch.nn.BCEWithLogitsLoss`.

@par Tensor shapes (with default config):
@verbatim
Input epoch:                  (B, 19, 2500)
After CNN block 1 (pool 4):   (B, 32, 625)
After CNN block 2 (pool 4):   (B, 64, 156)
After CNN block 3 (pool 2):   (B, 128, 78)
Permute for LSTM:             (B, 78, 128)
LSTM h_n  (bidir, 2 layers):  (4, B, 128)
Last fwd + bwd hidden concat: (B, 256)
Head output (logit):          (B,)
@endverbatim

@note The model returns logits, not probabilities. `BCEWithLogitsLoss`
      applies the sigmoid internally for numerical stability; do not add
      a sigmoid before the loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


# -------------
# Configuration
# -------------


@dataclass(frozen=True)
class ModelConfig:
    """
    @brief Hyperparameters that define the CNN+LSTM architecture.

    @var n_channels Number of input EEG channels (19, the 10-20 montage).
    @var n_timesteps Number of input time samples per epoch (2500 = 10 s @ 250 Hz).
    @var conv_channels Output channels of the three CNN blocks, in order.
    @var conv_kernels Kernel sizes for the three CNN blocks, in order. All odd
         so that "same" padding (kernel_size // 2) keeps the time dimension
         unchanged at the conv step; pooling alone reduces it.
    @var pool_sizes MaxPool stride for each block. The product is the total
         downsampling factor of the LSTM input sequence length
         (defaults: 4 * 4 * 2 = 32x, so 2500 -> 78).
    @var conv_dropout Dropout probability after each CNN block.
    @var lstm_hidden Hidden size of the LSTM (per direction).
    @var lstm_layers Number of stacked LSTM layers.
    @var lstm_bidirectional Whether to use a bidirectional LSTM.
    @var lstm_dropout Inter-layer dropout (only active when lstm_layers > 1).
    @var head_hidden Hidden size of the MLP classification head.
    @var head_dropout Dropout probability inside the MLP head.
    """

    n_channels: int = 19
    n_timesteps: int = 2500
    conv_channels: tuple[int, int, int] = (32, 64, 128)
    conv_kernels: tuple[int, int, int] = (7, 5, 3)
    pool_sizes: tuple[int, int, int] = (4, 4, 2)
    conv_dropout: float = 0.3
    lstm_hidden: int = 128
    lstm_layers: int = 2
    lstm_bidirectional: bool = True
    lstm_dropout: float = 0.3
    head_hidden: int = 64
    head_dropout: float = 0.3


# ---------
# CNN block
# ---------


class _ConvBlock(nn.Module):
    """
    @brief One convolutional block: Conv1d -> BatchNorm -> ELU -> Dropout -> MaxPool.

    @details
    The Conv1d uses "same" padding (`kernel_size // 2`) so only the trailing
    MaxPool reduces the time dimension. ELU is used as the nonlinearity for
    its smooth gradients on EEG-scale signals. Bias is omitted from the conv
    layer because BatchNorm's affine parameters subsume it.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pool_size: int,
        dropout: float,
    ) -> None:
        """
        @brief Build the layers for one CNN block.

        @param in_channels Number of input channels to the Conv1d.
        @param out_channels Number of output feature maps.
        @param kernel_size Conv kernel size; must be odd for symmetric padding.
        @param pool_size MaxPool kernel and stride; divides time dim by this factor.
        @param dropout Dropout probability applied after the activation.
        """
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels, out_channels, kernel_size,
                padding=padding, bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(pool_size),
        )
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        @brief Apply the conv block to a (B, C_in, T) input.

        @param x Input tensor of shape (batch, in_channels, time).
        @return Output tensor of shape (batch, out_channels, time // pool_size).
        """
        return self.net(x)


# ----------
# Full model
# ----------


class CNN_LSTM(nn.Module):
    """
    @brief CNN + bidirectional LSTM binary classifier for EEG epochs.

    @details
    Forward takes a batch of preprocessed EEG epochs of shape
    (B, n_channels, n_timesteps) and returns a 1-D tensor of shape (B,)
    containing one logit per sample. Pair with `BCEWithLogitsLoss`.

    @par Why this architecture:
      - The CNN learns local spatial-temporal features (cross-channel
        patterns + short waveform morphology) without hand-crafted
        spectral features.
      - The bidirectional LSTM models how those features evolve across
        the 10-second window. Useful for transient abnormalities that
        a global spectral summary would miss.
      - Bidirectional is appropriate because we are not in an online
        setting; the model can attend to both sides of any event.
    """
    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        """
        @brief Build the CNN feature extractor, LSTM, and MLP head.

        @param config Optional `ModelConfig`. If `None`, defaults are used
               (3 CNN blocks 32/64/128, 2-layer BiLSTM hidden 128, MLP 64->1).
        """
        super().__init__()
        cfg = config or ModelConfig()
        self.config = cfg

        # CNN feature extractor
        blocks: list[nn.Module] = []
        in_ch = cfg.n_channels
        for out_ch, k, p in zip(cfg.conv_channels, cfg.conv_kernels, cfg.pool_sizes):
            blocks.append(_ConvBlock(in_ch, out_ch, k, p, cfg.conv_dropout))
            in_ch = out_ch
        self.cnn = nn.Sequential(*blocks)

        # LSTM model
        self.lstm = nn.LSTM(
            input_size=cfg.conv_channels[-1],
            hidden_size=cfg.lstm_hidden,
            num_layers=cfg.lstm_layers,
            batch_first=True,
            bidirectional=cfg.lstm_bidirectional,
            dropout=cfg.lstm_dropout if cfg.lstm_layers > 1 else 0.0
        )

        n_directions = 2 if cfg.lstm_bidirectional else 1
        lstm_out_dim = cfg.lstm_hidden * n_directions

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(lstm_out_dim, cfg.head_hidden),
            nn.ELU(),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(cfg.head_hidden, 1),
        )

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        @brief Run a forward pass on a batch of EEG epochs.

        @param x Input tensor of shape (batch, n_channels, n_timesteps),
                 dtype float32. Expected to already be per-channel z-scored
                 by `TUABEpochDataset`.
        @return 1-D tensor of shape (batch,) of unnormalized logits. Apply
                `torch.sigmoid` for probabilities, or pass straight to
                `BCEWithLogitsLoss`.
        """
        feats = self.cnn(x)
        feats = feats.transpose(1, 2)

        # LSTM returns (output, (h_n, c_n)). h_n shape:
        #   (num_layers * num_directions, batch, hidden_size)
        # Layout for bidirectional: h_n[layer * 2 + direction], so
        #   h_n[-2] = last layer forward, h_n[-1] = last layer backward.
        _, (h_n, _) = self.lstm(feats)

        if self.config.lstm_bidirectional:
            forward_last = h_n[-2]
            backward_last = h_n[-1]
            summary = torch.cat([forward_last, backward_last], dim=1)
        else: 
            summary = h_n[-1]
        
        logits = self.head(summary).squeeze(-1)
        return logits
    

    def num_parameters(self, trainable_only: bool = True) -> int:
        """
        @brief Count parameters in the model.

        @param trainable_only If True (default), only count parameters with
               `requires_grad=True`.
        @return Total parameter count.
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


# ----------------------
# Smoke test entry point
# ----------------------


def _main() -> None:
    """
    @brief Smoke test: build the model, run a fake batch, print shapes and param count.

    @details
    Run as `python -m src.models.model` from the project root. Verifies the
    model constructs cleanly and produces the expected output shape on a fake
    batch matching the dataset's contract: (B, 19, 2500) -> (B,).
    """
    model = CNN_LSTM()
    print(f"Trainable parameters: {model.num_parameters():,}")

    fake = torch.rand(4, 19, 2500)
    with torch.no_grad():
        logits = model(fake)
    print(f"Input shape:  {tuple(fake.shape)}")
    print(f"Output shape: {tuple(logits.shape)}  (expected: (4,))")
    print(f"Logit sample: {[round(v, 4) for v in logits.tolist()]}")

if __name__ == "__main__":
    _main()