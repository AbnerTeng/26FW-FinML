import torch
from torch import nn


class GRUModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc_out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x.shape: [B, T, S, F]
        """
        b, t, s, f = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b * s, t, f)
        gru_hidden = self.gru(x)[0]  # [b * s, t, h]
        gru_last_hidden = gru_hidden[:, -1, :].reshape(b, s, -1)
        out = self.fc_out(gru_last_hidden)  # [b, s, 1]
        out = out.squeeze(-1)

        return out
