"""
Model registry for FinML
"""

from .seq_model import GRUModel
from .loss import PairMSELoss, SpearmanCorr

# Model mapper - register your models here
model_mapper = {
    "gru": GRUModel,
    # Add other models here as they are implemented
}

__all__ = ["model_mapper", "GRUModel", "PairMSELoss", "SpearmanCorr"]
