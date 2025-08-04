from torch import nn


class BaseModel(nn.Module):
    """Base class for all models."""

    def __init__(self):
        super().__init__()
