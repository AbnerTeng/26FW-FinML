import os
import logging

import torch
import numpy as np
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from .constants import FEATURES
from .datasets import TSDataset
from .models import model_mapper
from .models.loss import PairMSELoss, SpearmanCorr
from .trainer import Trainer
from .utils import get_array, load_data, preprocess, rolling_norm_sequence, seed_all

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    TARGET = f"cumret_{cfg.target_range}"

    if cfg.device == "mps":  # For Apple Silicon devices, ensure MPS is available
        torch.backends.mps.is_available = lambda: True
        torch.backends.mps.is_built = lambda: True

    data = load_data(cfg.data_path)
    train, valid, test = preprocess(
        data,
        cfg.ma_windows,
        cfg.target_range,
        cfg.date_column,
        cfg.train_start,
        cfg.val_start,
        cfg.test_start,
        cfg.test_end,
        cfg.add_daily_norm,
        norm_target=False,
    )
    assert len(train) > 0, "Train set is empty."
    assert (
        len(train) == len(valid) == len(test)
    ), "Train, valid, and test sets must have the same number of stocks."
    test_dates = test[list(test.keys())[0]][cfg.date_column].values.tolist()
    used_features = FEATURES

    train_feat, train_label = get_array(train, used_features, TARGET, False)
    valid_feat, valid_label, valid_next_ret = get_array(
        valid, used_features, TARGET, True
    )
    test_feat, test_label, test_next_ret = get_array(test, used_features, TARGET, True)
    roll_train_feat, roll_train_label, _ = rolling_norm_sequence(
        train_feat,
        train_label,
        None,
        seq_len=cfg.seq_len,
        norm_axis=0,
        norm_index=cfg.norm_index,
        norm_type=cfg.norm_type,
    )
    roll_valid_feat, roll_valid_label, roll_valid_next_ret = rolling_norm_sequence(
        valid_feat,
        valid_label,
        valid_next_ret,
        seq_len=cfg.seq_len,
        norm_axis=0,
        norm_index=cfg.norm_index,
        norm_type=cfg.norm_type,
    )
    roll_test_feat, roll_test_label, roll_test_next_ret = rolling_norm_sequence(
        test_feat,
        test_label,
        test_next_ret,
        seq_len=cfg.seq_len,
        norm_axis=0,
        norm_index=cfg.norm_index,
        norm_type=cfg.norm_type,
    )
    wandb_config = OmegaConf.to_container(cfg, resolve=True)

    if cfg.wandb:
        group_name = f"{cfg.market}_{cfg.loss}"
        run_name = f"{cfg.expr_name}_seed{cfg.seed}"
        wandb.init(
            project=cfg.wandb_project_name,
            group=group_name,
            job_type="seed_search",
            name=run_name,
            config=wandb_config,
        )

    seed_all(cfg.seed)
    model_path = f"last_s{cfg.seed}.pt"
    pred_path = f"pred_s{cfg.seed}.npy"

    if os.path.exists(f"model/{cfg.expr_name}/{model_path}"):
        log.info("Model %s already exists. Skipping training.", model_path)
        if cfg.wandb:
            wandb.finish()
        exit(0)

    train_dataset = TSDataset(roll_train_feat, roll_train_label, None)
    valid_dataset = TSDataset(roll_valid_feat, roll_valid_label, roll_valid_next_ret)
    test_dataset = TSDataset(roll_test_feat, roll_test_label, roll_test_next_ret)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
    )
    model_conf = OmegaConf.to_container(cfg.model[cfg.model_name], resolve=True)
    log.info("Model config: %s", model_conf)
    model = model_mapper[cfg.model_name](**model_conf).to(cfg.device)

    if cfg.loss == "mse":
        loss_func = torch.nn.MSELoss()
    elif cfg.loss == "pairmse":
        loss_func = PairMSELoss(alpha=cfg.get("alpha", 0.5))
    elif cfg.loss == "spearman":
        loss_func = SpearmanCorr()
    else:
        raise ValueError(f"Unknown loss type: {cfg.loss}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=0.0,  # 1e-5,
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_type=cfg.loss,
        loss_fn=loss_func,
        expr_name=cfg.expr_name,
        device=cfg.device,
        k=cfg.k,
        target=cfg.target_range,
        model_path=model_path,
        wandb_project=None if not cfg.wandb else wandb,
    )
    if not cfg.inference_only:
        log.info("Starting training...")
        trainer.train(
            train_loader=train_loader,
            valid_loader=valid_loader,
            n_epochs=cfg.n_epochs,
            patience=cfg.patience,
        )

    test_preds = trainer.eval(
        eval_loader=test_loader,
        inference_only=cfg.inference_only,
        test_dates=test_dates,
    )
    if not os.path.exists(f"output/{cfg.expr_name}"):
        os.makedirs(f"output/{cfg.expr_name}", exist_ok=True)

    np.save(f"output/{cfg.expr_name}/{pred_path}", test_preds)

    if cfg.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
