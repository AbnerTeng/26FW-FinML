"""
Custom loss functions for financial ML
"""

import torch
import torch.nn as nn


class PairMSELoss(nn.Module):
    """Pairwise MSE Loss with ranking component"""

    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse_loss = self.mse(pred, target)

        # Ranking loss component
        pred_diff = pred.unsqueeze(0) - pred.unsqueeze(1)
        target_diff = target.unsqueeze(0) - target.unsqueeze(1)
        ranking_loss = torch.mean(
            (torch.sign(pred_diff) - torch.sign(target_diff)) ** 2
        )

        return self.alpha * mse_loss + (1 - self.alpha) * ranking_loss


class SpearmanCorr(nn.Module):
    """Spearman Correlation Loss"""

    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_rank = torch.argsort(torch.argsort(pred, dim=0), dim=0).float()
        target_rank = torch.argsort(torch.argsort(target, dim=0), dim=0).float()

        pred_centered = pred_rank - pred_rank.mean()
        target_centered = target_rank - target_rank.mean()

        pearson = (pred_centered * target_centered).sum() / (
            torch.sqrt((pred_centered**2).sum() * (target_centered**2).sum()) + 1e-6
        )

        return 1 - pearson
