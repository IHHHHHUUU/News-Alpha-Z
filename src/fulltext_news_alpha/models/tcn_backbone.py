"""Temporal convolutional backbone for 30 trading-day lookback windows."""

from __future__ import annotations

from collections.abc import Sequence

from fulltext_news_alpha.models._torch import require_torch

torch, nn = require_torch()


class CausalConv1d(nn.Module):
    """1D convolution with left padding so step ``t`` cannot see future steps."""

    def __init__(self, input_dim: int, output_dim: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        if dilation <= 0:
            raise ValueError("dilation must be positive")
        self.left_padding = (int(kernel_size) - 1) * int(dilation)
        self.conv = nn.Conv1d(
            int(input_dim),
            int(output_dim),
            kernel_size=int(kernel_size),
            dilation=int(dilation),
        )

    def forward(self, values):
        padded = nn.functional.pad(values, (self.left_padding, 0))
        return self.conv(padded)


class TemporalBlock(nn.Module):
    """Residual dilated TCN block."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(input_dim, hidden_dim, kernel_size, dilation)
        self.conv2 = CausalConv1d(hidden_dim, hidden_dim, kernel_size, dilation)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(float(dropout))
        self.residual = (
            nn.Identity()
            if int(input_dim) == int(hidden_dim)
            else nn.Conv1d(int(input_dim), int(hidden_dim), kernel_size=1)
        )
        self.norm = nn.LayerNorm(int(hidden_dim))
        self.dilation = int(dilation)

    def forward(self, values):
        residual = self.residual(values)
        out = self.conv1(values)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.activation(out + residual)
        out = self.dropout(out)
        return self.norm(out.transpose(1, 2)).transpose(1, 2)


class TCNBackbone(nn.Module):
    """Encode ``[batch, time, dim]`` sequences into one target-day state."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        kernel_size: int = 3,
        dilations: Sequence[int] = (1, 2, 4, 8),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if not dilations:
            raise ValueError("dilations must contain at least one layer")
        layers: list[nn.Module] = []
        current_dim = int(input_dim)
        for dilation in dilations:
            layers.append(
                TemporalBlock(
                    input_dim=current_dim,
                    hidden_dim=int(hidden_dim),
                    kernel_size=int(kernel_size),
                    dilation=int(dilation),
                    dropout=float(dropout),
                )
            )
            current_dim = int(hidden_dim)
        self.layers = nn.ModuleList(layers)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.kernel_size = int(kernel_size)
        self.dilations = tuple(int(value) for value in dilations)
        self.dropout_p = float(dropout)

    def forward(self, sequence, sequence_mask=None):
        """Return ``(encoded_sequence, last_state)``.

        ``sequence`` is expected as ``[batch, time, dim]``. If ``sequence_mask``
        is provided, padded timesteps are zeroed before and after the TCN; the
        returned state is gathered from the last valid timestep per row.
        """

        if sequence.ndim != 3:
            raise ValueError("sequence must have shape [batch, time, dim]")
        values = sequence
        mask = None
        if sequence_mask is not None:
            mask = sequence_mask.bool()
            values = values * mask.unsqueeze(-1).to(values.dtype)
        out = values.transpose(1, 2)
        for layer in self.layers:
            out = layer(out)
            if mask is not None:
                out = out * mask.unsqueeze(1).to(out.dtype)
        encoded = out.transpose(1, 2)
        if mask is None:
            return encoded, encoded[:, -1, :]
        lengths = mask.long().sum(dim=1).clamp(min=1)
        gather_index = (lengths - 1).view(-1, 1, 1).expand(-1, 1, encoded.shape[-1])
        last_state = encoded.gather(dim=1, index=gather_index).squeeze(1)
        return encoded, last_state
