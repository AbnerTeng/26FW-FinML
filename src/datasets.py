from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class TSDataset(Dataset):
    def __init__(
        self, feat: np.ndarray, label: np.ndarray, next_ret: Optional[np.ndarray] = None
    ) -> None:
        self.feat = feat
        self.label = label
        self.next_ret = next_ret

    def __len__(self) -> int:
        return self.feat.shape[0]

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        feat = torch.tensor(self.feat[index], dtype=torch.float32)
        label = torch.tensor(self.label[index], dtype=torch.float32)

        if self.next_ret is not None:
            next_ret = torch.tensor(self.next_ret[index], dtype=torch.float32)

            return feat, label, next_ret

        return feat, label, torch.tensor([])
