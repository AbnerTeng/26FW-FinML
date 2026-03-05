import os
import random
import pickle
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

log = logging.getLogger(__name__)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(filepath: str) -> Dict[str, pd.DataFrame]:
    with open(filepath, "rb") as f:
        data = pickle.load(f)

    return data


def compute_trends(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    for w in windows:
        rolling_mean = pd.DataFrame(df["close"].rolling(window=w).mean())
        df[f"trend_{w}"] = rolling_mean.div(df["close"]) - 1

    return df


def compute_cumret(df: pd.DataFrame, target_range: int) -> pd.DataFrame:
    rolling_sum = pd.DataFrame(df["ret"].rolling(window=target_range).sum())
    label = rolling_sum.shift(-target_range).fillna(0)
    mask = label.abs() > 0.5
    mean_label = label[~mask].mean()
    label[mask] = mean_label

    return label


def compute_daily_norm(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["open", "high", "low", "close", "volume"]:
        df[f"daily_norm_{col}"] = df[col].pct_change().fillna(0)

    return df


def preprocess(
    data: Dict[str, pd.DataFrame],
    ma_windows: List[int],
    target_range: int = 5,
    date_column: str = "date",
    train_start: str = "2010-01-01",
    val_start: str = "2020-01-01",
    test_start: str = "2021-01-01",
    test_end: str = "2022-01-01",
    add_daily_norm: bool = True,
    norm_target: bool = False,
) -> Tuple[Dict[str, pd.DataFrame], ...]:
    """
    data.shape = (n_dates, OHLCV + date))

    ma_windows & target_range can be adjusted based on the strategy.
    """
    train, valid, test = {}, {}, {}

    for stock_id, stock_data in tqdm(
        data.items(), desc="Preprocessing data", leave=False
    ):
        if stock_data.empty:
            continue

        else:
            stock_data.columns = stock_data.columns.str.lower()
            stock_data[date_column] = stock_data[date_column].astype(str)

            if (stock_data[date_column].iloc[0] > train_start) or (
                stock_data[date_column].iloc[-1] < test_end
            ):
                continue

            else:
                nas = (
                    stock_data[["open", "high", "low", "close", "volume"]]
                    .isna()
                    .sum()
                    .sum()
                )
                na_percent = nas / (stock_data.shape[0] * 5)

                if na_percent < 0.03:
                    stock_data_copy = stock_data.copy()
                    stock_data_copy["ret"] = (
                        stock_data_copy["close"].pct_change().fillna(0)
                    )
                    stock_data_copy["next_ret"] = stock_data_copy["ret"].shift(-1)
                    stock_data_copy[date_column] = pd.to_datetime(
                        stock_data_copy[date_column]
                    )

                    if len(ma_windows) > 0:
                        stock_data_copy = compute_trends(stock_data_copy, ma_windows)

                    if add_daily_norm:
                        stock_data_copy = compute_daily_norm(stock_data_copy)

                    masks = {
                        "train": stock_data_copy[date_column] < val_start,
                        "valid": (stock_data_copy[date_column] >= val_start)
                        & (stock_data_copy[date_column] < test_start),
                        "test": stock_data_copy[date_column] >= test_start,
                    }

                    for split_name, mask in masks.items():
                        split_df = stock_data_copy[mask].copy()

                        if norm_target:
                            split_df[f"cumret_{target_range}"] = (
                                compute_cumret(split_df, target_range) / target_range
                            )
                        else:
                            split_df[f"cumret_{target_range}"] = compute_cumret(
                                split_df, target_range
                            )
                        split_df.fillna(0, inplace=True)

                        if split_name == "train":
                            train[stock_id] = split_df
                        elif split_name == "valid":
                            valid[stock_id] = split_df
                        else:
                            test[stock_id] = split_df

    return train, valid, test


def get_array(
    data: Dict[str, pd.DataFrame],
    feats: List[str],
    target: str,
    get_next_ret: bool = False,
) -> Tuple[np.ndarray, ...]:
    """
    feat_matrix.shape = (n_dates, n_stocks, n_feats)
    label_matrix.shape = (n_dates, n_stocks, 1)
    next_ret_matrix.shape = (n_dates, n_stocks, 1)
    """
    n_dates = data[list(data.keys())[0]].shape[0]
    n_stocks = len(data)
    n_feats = len(feats)

    assert n_dates > 0, "Data must contain at least one date"

    feat_matrix = np.empty((n_dates, n_stocks, n_feats), dtype=np.float32)
    label_matrix = np.empty((n_dates, n_stocks, 1), dtype=np.float32)
    next_ret_matrix = np.empty((n_dates, n_stocks, 1), dtype=np.float32)

    for i, (_, stock_data) in enumerate(
        tqdm(data.items(), desc="Converting data to arrays", leave=False)
    ):
        feat_matrix[:, i] = stock_data[feats].values
        label_matrix[:, i, 0] = stock_data[target].values

        if get_next_ret:
            next_ret_matrix[:, i, 0] = stock_data["next_ret"].values

    feat_matrix = np.nan_to_num(feat_matrix, nan=0, posinf=0, neginf=0)
    label_matrix = np.nan_to_num(label_matrix, nan=0, posinf=0, neginf=0)

    if get_next_ret:
        return (
            feat_matrix,
            np.clip(label_matrix, -1, 1),
            np.clip(next_ret_matrix, -1, 1),
        )
    else:
        return feat_matrix, np.clip(label_matrix, -1, 1)


def max_norm(feat_matrix: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    feat_matrix.shape = (seq_len, n_stocks, n_feats)
    """
    max_val = np.nanmax(feat_matrix, axis=axis, keepdims=True)
    max_val[max_val == 0] = 1  # Avoid division by zero
    normalized_matrix = feat_matrix / max_val

    normalized_matrix = np.nan_to_num(
        normalized_matrix,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    assert np.all(
        (normalized_matrix >= 0) & (normalized_matrix <= 1)
    ), "Normalized matrix values should be in the range [0, 1]"
    assert not np.isnan(
        normalized_matrix
    ).any(), "NaN values found in normalized matrix"

    return normalized_matrix


def std_norm(feat_matrix: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    feat_matrix.shape = (seq_len, n_stocks, n_feats)
    """
    if feat_matrix.shape[0] == 0:
        return feat_matrix

    mean = np.nanmean(feat_matrix, axis=axis, keepdims=True)
    std = np.nanstd(feat_matrix, axis=axis, keepdims=True)

    std[std == 0] = 1  # Avoid division by zero
    normalized_matrix = (feat_matrix - mean) / std

    normalized_matrix = np.nan_to_num(
        normalized_matrix,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    assert not np.isnan(
        normalized_matrix
    ).any(), "NaN values found in normalized matrix"

    return normalized_matrix


def first_norm(feat_matrix: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    feat_matrix.shape = (seq_len, n_stocks, n_feats)
    """
    if feat_matrix.shape[0] == 0:
        return feat_matrix

    first_val = np.expand_dims(feat_matrix[axis], axis=0)
    normalized_matrix = (feat_matrix - first_val) / (first_val + 1e-8)

    assert not np.isnan(
        normalized_matrix
    ).any(), "NaN values found in normalized matrix"

    return normalized_matrix


def rolling_norm_sequence(
    feat_matrix: np.ndarray,
    label_matrix: np.ndarray,
    next_ret_matrix: Optional[np.ndarray],
    seq_len: int,
    norm_axis: int,
    norm_index: int,
    norm_type: str = "std",
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    feat_seq.shape = (n_dates - seq_len + 1, seq_len, n_stocks, n_feats)
    """
    if seq_len <= 1:
        return feat_matrix, label_matrix, next_ret_matrix

    n_dates = feat_matrix.shape[0]
    seq_feat = np.empty(
        (n_dates - seq_len + 1, seq_len, feat_matrix.shape[1], feat_matrix.shape[2]),
        dtype=feat_matrix.dtype,
    )

    for i in range(n_dates - seq_len + 1):
        normed_slice = np.empty_like(feat_matrix[i : i + seq_len])

        if norm_type == "max":
            normed_slice[:, :, :norm_index] = max_norm(
                feat_matrix[i : i + seq_len, :, :norm_index], axis=norm_axis
            )
        elif norm_type == "std":
            normed_slice[:, :, :norm_index] = std_norm(
                feat_matrix[i : i + seq_len, :, :norm_index], axis=norm_axis
            )
        elif norm_type == "first":
            normed_slice[:, :, :norm_index] = first_norm(
                feat_matrix[i : i + seq_len, :, :norm_index], axis=norm_axis
            )
        elif norm_type == "none":
            normed_slice[:, :, :norm_index] = feat_matrix[
                i : i + seq_len, :, :norm_index
            ]
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}")

        normed_slice[:, :, norm_index:] = np.nan_to_num(
            feat_matrix[i : i + seq_len, :, norm_index:],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        seq_feat[i] = normed_slice

    assert not np.isnan(
        seq_feat
    ).any(), "NaN values found in rolling normalized sequence"

    return (
        seq_feat,
        label_matrix[seq_len - 1 :],
        next_ret_matrix[seq_len - 1 :] if next_ret_matrix is not None else None,
    )


class EarlyStopping:
    def __init__(
        self,
        model_path: str,
        patience: int = 1000,
        verbose: bool = True,
        metric: str = "cumret",
    ) -> None:
        self.patience = patience
        self.verbose = verbose
        self.metric = metric
        self.model_path = model_path
        self.counter = 0
        self.best_score = 0
        self.early_stop = False
        self.loss_min = float("inf")

    def __call__(self, loss: Union[int, float], expr_name: str, model: Any) -> None:
        score = loss

        early_stop_criteria = (
            (score < self.best_score)
            if self.metric == "cumret"
            else (score > self.best_score)
        )

        if self.best_score == 0:
            self.best_score = score
            self.save_checkpoint(loss, expr_name, model)

        elif early_stop_criteria:
            self.counter += 1

            if self.verbose:
                log.info(
                    "EarlyStopping counter: %d out of %d", self.counter, self.patience
                )

            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.save_checkpoint(loss, expr_name, model)
            self.counter = 0

    def save_checkpoint(self, loss: float, expr_name: str, model: Any) -> None:
        if self.verbose:
            log.info(
                "Validation %s Increased (%.6f --> %.6f). Saving model...",
                self.metric,
                self.loss_min,
                loss,
            )

        if not os.path.exists(f"model/{expr_name}"):
            os.makedirs(f"model/{expr_name}")

        torch.save(model.state_dict(), f"model/{expr_name}/{self.model_path}")
        self.loss_min = loss
