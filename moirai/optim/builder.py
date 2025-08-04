# pylint: disable=inconsistent-quotes
from typing import Dict, Type, Union
from torch import nn, optim
from torch.optim import lr_scheduler


import moirai.optim.custom.optim as moirai_optim
import moirai.optim.custom.lr_scheduler as moirai_lr_scheduler


def build_optim(
        optim_cfg: Dict[str, Union[str, Type[optim.Optimizer], Dict]],
        model: nn.Module
) -> optim.Optimizer:
    """
    Build an optimizer instance from configuration.

    Expected config format:
    {
        'TYPE': (str or type) Optimizer class or name (e.g., 'Adam', 'SGD', or a custom type),
        'PARAM': (Dict) Initialization parameters excluding `params`
    }

    Example:
        optim_cfg = {
            'TYPE': 'Adam',
            'PARAM': {
                'lr': 1e-3,
                'betas': (0.9, 0.99),
                'eps': 1e-8,
                'weight_decay': 0
            }
        }

    Args:
        optim_cfg (Dict): Optimizer configuration dictionary.
        model (nn.Module): Model whose parameters will be optimized.

    Returns:
        optim.Optimizer: Instantiated optimizer.
    """

    # Resolve optimizer class from type or string
    optim_type = optim_cfg.get('TYPE')
    if isinstance(optim_type, type):
        optimizer_cls = optim_type
    elif isinstance(optim_type, str):
        # Try resolving from torch.optim
        if hasattr(optim, optim_type):
            optimizer_cls = getattr(optim, optim_type)
        # Fallback to custom optimizer module
        elif hasattr(moirai_optim, optim_type):
            optimizer_cls = getattr(moirai_optim, optim_type)
        else:
            raise ValueError(f"Unknown optimizer type: '{optim_type}'. "
                             f"Neither torch.optim nor custom.optim has this class.")
    else:
        raise TypeError("`TYPE` in optim_cfg must be a string or a class type.")

    optim_param = dict(optim_cfg.get('PARAM', {}))

    return optimizer_cls(model.parameters(), **optim_param)


def build_lr_scheduler(
    cfg: Dict, optimizer: optim.Optimizer
) -> lr_scheduler._LRScheduler:
    """
    Build a learning rate scheduler from the given configuration.

    Configuration Format:
    {
        'TYPE': str or type,
            Scheduler name or class. Can be from `torch.optim.lr_scheduler` or a custom scheduler.
        'PARAM': Dict,
            Initialization parameters (excluding the `optimizer`).
    }

    Example:
        cfg = {
            'TYPE': 'MultiStepLR',
            'PARAM': {
                'milestones': [100, 200, 300],
                'gamma': 0.1
            }
        }

    Args:
        cfg (Dict): Scheduler configuration.
        optimizer (optim.Optimizer): Optimizer instance.

    Returns:
        lr_scheduler._LRScheduler: Initialized scheduler.
    """
    scheduler_type = cfg['TYPE']
    if not isinstance(scheduler_type, type):
        scheduler_type = (
            getattr(lr_scheduler, scheduler_type, None)
            or getattr(moirai_lr_scheduler, scheduler_type)
        )

    scheduler_param = cfg['PARAM'].copy()
    scheduler_param['optimizer'] = optimizer
    return scheduler_type(**scheduler_param)
