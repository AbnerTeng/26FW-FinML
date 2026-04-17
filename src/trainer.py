from typing import Any, Optional

import pandas as pd
import numpy as np
import torch
import wandb
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from .eval_utils import calculate_portfolio_returns_sliding, get_metrics, wandb_recorder
from .utils import EarlyStopping


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_type: str,
        loss_fn: nn.Module,
        expr_name: str,
        device: str,
        k: int,
        target: int,
        model_path: str,
        wandb_project: Optional[Any] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_type = loss_type
        self.loss_fn = loss_fn
        self.expr_name = expr_name
        self.device = device
        self.k = k
        self.target = target
        self.model_path = model_path
        self.wandb = wandb_project

    def _load_model(self, model_path: str) -> None:
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)

    def train(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        n_epochs: int,
        patience: int,
        full_batch_size: int = 256,
    ) -> None:
        early_stopping = EarlyStopping(
            model_path=self.model_path, patience=patience, verbose=True, metric="cumret"
        )
        if train_loader.batch_size:
            accumulate_steps = int(full_batch_size / train_loader.batch_size)
        else:
            raise ValueError("train_loader must have a defined batch_size")

        with tqdm(total=n_epochs, desc="Training Progress", unit="epoch") as pbar:
            for epoch in range(n_epochs):
                self.model.train(True)
                epoch_loss: float = 0.0

                for i, (tr_feat, tr_label, _) in enumerate(train_loader):
                    batch_loss: torch.Tensor = torch.tensor(0.0, device=self.device)
                    tr_feat, tr_label = tr_feat.to(self.device), tr_label.to(
                        self.device
                    )
                    if tr_label.ndim == 3:
                        tr_label = tr_label.squeeze(-1)

                    model_output = self.model(tr_feat)
                    tr_pred = (
                        model_output[0]
                        if isinstance(model_output, tuple)
                        else model_output
                    )
                    batch_loss = self.loss_fn(tr_pred, tr_label)

                    if self.loss_type not in ["pairmse", "mse"]:
                        batch_loss *= -1

                    batch_loss = batch_loss / accumulate_steps
                    batch_loss.backward()

                    if (i + 1) % accumulate_steps == 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_norm=1.0, norm_type=2
                        )
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                    epoch_loss += batch_loss.item() * accumulate_steps

                if len(train_loader) % accumulate_steps != 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0, norm_type=2
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                epoch_loss /= len(train_loader)

                self.model.eval()

                epoch_val_loss: float = 0.0

                with torch.no_grad():
                    full_val_pred, full_val_next_ret = [], []
                    gate_sys_list, gate_unsys_list, final_gate_list = [], [], []

                    for val_feat, val_label, val_next_ret in valid_loader:
                        val_feat, val_label = (
                            val_feat.to(self.device),
                            val_label.to(self.device),
                        )
                        if val_label.ndim == 3:
                            val_label = val_label.squeeze(-1)
                        model_output = self.model(val_feat)

                        if isinstance(model_output, tuple):
                            val_pred, gate_dict = model_output
                            gate_sys_list.append(
                                gate_dict["systematic_weight"].detach()
                            )
                            gate_unsys_list.append(
                                gate_dict["unsystematic_weight"].detach()
                            )
                            final_gate_list.append(gate_dict["final_gate"].detach())
                        else:
                            val_pred = model_output

                        val_loss = self.loss_fn(val_pred, val_label)

                        if self.loss_type not in ["pairmse", "mse"]:
                            val_loss *= -1

                        epoch_val_loss += val_loss.item()
                        full_val_pred.append(val_pred.unsqueeze(-1).unsqueeze(-1).cpu().numpy())
                        full_val_next_ret.append(
                            val_next_ret.cpu().numpy()
                            if val_next_ret is not None
                            else None
                        )

                    epoch_val_loss /= len(valid_loader)
                    full_val_pred = np.concatenate(full_val_pred, axis=0)
                    full_val_next_ret = (
                        np.concatenate(full_val_next_ret, axis=0)
                        if full_val_next_ret[0] is not None
                        else None
                    )
                    sliding_rets = calculate_portfolio_returns_sliding(
                        full_val_pred,
                        full_val_next_ret,
                        k=self.k,
                        prediction_windows=self.target,
                    )
                    average_metrics = pd.DataFrame(
                        [get_metrics(sliding_rets[i]) for i in range(len(sliding_rets))]
                    ).mean(axis=0)

                    if isinstance(average_metrics, pd.Series) or isinstance(
                        average_metrics, pd.DataFrame
                    ):
                        average_metrics = average_metrics.to_dict()

                epoch_gate_dict = None

                if self.wandb is not None:
                    wandb_recorder(
                        wandb=self.wandb,
                        epoch=epoch,
                        alpha=(
                            self.model.gate_score.item()
                            if hasattr(self.model, "gate_score")
                            else 0.0
                        ),
                        train_loss=epoch_loss,
                        valid_loss=epoch_val_loss,  # No validation loss in this context
                        valid_metrics=average_metrics,
                        gate_dict=epoch_gate_dict,
                    )

                early_stopping(average_metrics["AR"], self.expr_name, self.model)
                pbar.set_postfix(
                    {
                        "Epoch": f"{epoch+1}/{n_epochs}",
                        "Train Loss": f"{epoch_loss:.4f}",
                        "Val Loss": f"{epoch_val_loss:.4f}",
                        "Val AR": f"{average_metrics['AR']:.4f}",
                        "Early Stopping Patience": f"{early_stopping.counter}/{patience}",
                    }
                )
                pbar.update(1)

        torch.save(self.model.state_dict(), f"model/{self.expr_name}/{self.model_path}")

        print("Training complete.")

    @torch.no_grad()
    def eval(
        self,
        eval_loader: DataLoader,
        inference_only: bool = False,
        test_dates: list = [],
    ) -> np.ndarray:
        if inference_only:
            self._load_model(f"model/{self.expr_name}/{self.model_path}")

        self.model.eval()
        full_test_pred, full_test_next_ret = [], []

        with torch.no_grad():
            for te_feat, te_label, te_next_ret in tqdm(eval_loader):
                te_feat, te_label = te_feat.to(self.device), te_label.to(self.device)
                model_output = self.model(te_feat)

                if isinstance(model_output, tuple):
                    te_pred = model_output[0]
                else:
                    te_pred = model_output

                if len(te_pred.shape) == 2:
                    te_pred = te_pred.unsqueeze(-1).unsqueeze(-1)

                full_test_pred.append(te_pred.cpu().numpy())
                full_test_next_ret.append(
                    te_next_ret.cpu().numpy() if te_next_ret is not None else None
                )

            full_test_pred = np.concatenate(full_test_pred, axis=0)
            full_test_next_ret = (
                np.concatenate(full_test_next_ret, axis=0)
                if full_test_next_ret[0] is not None
                else None
            )
            sliding_rets = calculate_portfolio_returns_sliding(
                full_test_pred,
                full_test_next_ret,
                k=self.k,
                prediction_windows=self.target,
            )
            average_metrics = pd.DataFrame(
                [get_metrics(sliding_rets[i]) for i in range(len(sliding_rets))]
            ).mean(axis=0)
            if isinstance(average_metrics, pd.Series) or isinstance(
                average_metrics, pd.DataFrame
            ):
                average_metrics = average_metrics.to_dict()

        print(f"Test Metrics:\n {average_metrics}")

        if self.wandb is not None:
            data = [
                [x, y] for (x, y) in zip(test_dates, sliding_rets.mean(axis=0).cumsum())
            ]
            cumrets = self.wandb.Table(
                data=data, columns=["Date", "Cumulative Returns"]
            )
            wandb_log = {
                "test/loss": None,  # No test loss in this context
                "test/cumrets": self.wandb.plot.line(
                    cumrets,
                    "Date",
                    "Cumulative Returns",
                    title="Testing Cumulative Returns",
                ),
            }
            for metric_name, v in average_metrics.items():
                wandb_log[f"test/{metric_name}"] = v

            self.wandb.log(wandb_log)

        print("Testing complete.")

        return full_test_pred
