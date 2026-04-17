from typing import Dict, Optional

import numpy as np
import torch


def calculate_portfolio_returns_sliding(
    predictions: Optional[np.ndarray],
    next_ret: Optional[np.ndarray],
    k: int,
    prediction_windows: int,
    hard_top_k: bool = True,
    equal_weight: bool = False,
) -> np.ndarray:
    """
    predictions.shape [b, s, t, a]
    next_ret.shape [b, s, 1]

    return.shape [prediction_windows, batch]
    """
    preds = np.array(predictions)
    next_ret = next_ret.squeeze(-1)  # [b, s]
    batch, _, n_targets, _ = preds.shape
    returns = np.zeros((n_targets, prediction_windows, batch))

    for t in range(n_targets):
        spec_pred = preds[:, :, t, :]
        window_len = prediction_windows

        for day in range(0, batch):
            row_idx = day % window_len
            time_range = min(window_len, batch - day)
            this_day_alphas = spec_pred[day]  # [s, a]
            desc_indices = np.argsort(this_day_alphas, axis=0)[::-1]
            top_k_stocks = list(set(desc_indices[:k].flatten().tolist()))
            bottom_k_stocks = list(set(desc_indices[-k:].flatten().tolist()))
            top_k_alphas = this_day_alphas[top_k_stocks].sum(axis=1)
            bottom_k_alphas = this_day_alphas[bottom_k_stocks].sum(axis=1)

            if equal_weight:
                weights_top = np.ones_like(top_k_alphas)
                if hard_top_k:
                    weights_top[np.argsort(top_k_alphas, axis=-1)][:-k] = 0.0
                weights_top = weights_top / weights_top.sum(axis=-1)
            else:
                if hard_top_k:
                    top_k_alphas[np.argsort(top_k_alphas, axis=-1)][:-k] = -np.inf
                    bottom_k_alphas[np.argsort(bottom_k_alphas, axis=-1)][k:] = np.inf
                weights_top = torch.tensor(top_k_alphas).softmax(-1).numpy()

            weighted_returns_top = (
                next_ret[day : day + time_range, top_k_stocks]
                * weights_top[np.newaxis, :]
            )
            top_mean_return = np.sum(weighted_returns_top, axis=1)
            returns[t, row_idx, day : day + time_range] = top_mean_return

    return returns.mean(axis=0)


def get_metrics(returns: np.ndarray) -> Dict[str, float]:
    mr = np.mean(returns)
    ar = mr * 252
    sr = (mr / np.std(returns)) * np.sqrt(252)
    mdd = np.max(np.maximum.accumulate(returns.cumsum()) - returns.cumsum())
    cr = ar / mdd if mdd != 0 else np.inf

    return {
        "AR": ar,
        "SR": sr,
        "MDD": -1 * mdd,
        "CR": cr,
    }


def wandb_recorder(
    wandb, epoch, alpha, train_loss, valid_loss, valid_metrics, gate_dict=None
) -> None:
    g_final, g_sys, g_unsys = None, None, None
    wandb_log = {
        "epoch": epoch,
        "alpha": alpha,
        "train/loss": train_loss,
        "valid/loss": valid_loss,
    }
    for k, v in valid_metrics.items():
        wandb_log[f"valid/{k}"] = v

    if gate_dict is not None:
        g_sys = gate_dict["gate_sys"].detach()
        g_unsys = gate_dict["gate_unsys"].detach()
        g_final = gate_dict["final_gate"].detach()
        wandb_log["Regime/Market_Mean"] = g_sys.mean().item()
        wandb_log["Regime/Market_Std"] = g_sys.std().item()
        wandb_log["Regime/Stock_Deviation_Magnitude"] = g_unsys.abs().mean().item()
        wandb_log["Gate/Final_Saturation"] = (g_final - 0.5).abs().mean().item() * 2

    wandb.log(wandb_log)

    if g_unsys is not None and g_final is not None and epoch % 100 == 0:
        wandb.log(
            {
                "Gate/Distribution_Final": wandb.Histogram(g_final.cpu().numpy()),
                "Gate/Distribution_Unsys": wandb.Histogram(g_unsys.cpu().numpy()),
            },
            step=epoch,
        )
