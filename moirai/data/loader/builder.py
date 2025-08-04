# pylint: disable=inconsistent-quotes
from typing import Dict
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from .perfetcher import DataLoaderX
from moirai.dist import get_rank, get_world_size


def build_data_loader(
    dataset: Dataset,
    data_cfg: Dict,
    distributed: bool = False
) -> DataLoader:
    """
    Build a PyTorch DataLoader (with optional DDP support).

    Args:
        dataset (Dataset): The dataset instance.
        data_cfg (Dict): Configuration dictionary with optional keys:
            - BATCH_SIZE (int): Mini-batch size (default: 1)
            - SHUFFLE (bool): Whether to shuffle data (default: False)
            - NUM_WORKERS (int): Number of subprocesses to use for data loading (default: 0)
            - PIN_MEMORY (bool): Whether to pin memory during loading (default: False)
            - PREFETCH (bool): Whether to use background data prefetching (default: False)
            - COLLATE_FN (Callable, optional): Custom collate function.
        distributed (bool): Whether to use DistributedSampler (DDP mode).

    Returns:
        DataLoader: A PyTorch DataLoader instance.
    """
    batch_size = data_cfg.get("BATCH_SIZE", 1)
    shuffle = data_cfg.get("SHUFFLE", False)
    num_workers = data_cfg.get("NUM_WORKERS", 0)
    pin_memory = data_cfg.get("PIN_MEMORY", False)
    use_prefetch = data_cfg.get("PREFETCH", False)
    collate_fn = data_cfg.get("COLLATE_FN", None)

    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=shuffle,
        )
        shuffle = False  # Must be False when using DistributedSampler

    data_loader_cls = DataLoaderX if use_prefetch else DataLoader

    return data_loader_cls(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )


def build_data_loader_ddp(dataset: Dataset, data_cfg: Dict):
    """Build ddp dataloader from `data_cfg`
    `data_cfg` is part of config which defines fields about data, such as `CFG.TRAIN.DATA`

    structure of `data_cfg` is
    {
        'BATCH_SIZE': (int, optional) batch size of data loader (default: ``1``),
        'SHUFFLE': (bool, optional) data reshuffled option (default: ``False``),
        'NUM_WORKERS': (int, optional) num workers for data loader (default: ``0``),
        'PIN_MEMORY': (bool, optional) pin_memory option (default: ``False``),
        'PREFETCH': (bool, optional) set to ``True`` to use `BackgroundGenerator` (default: ``False``)
            need to install `prefetch_generator`, see https://pypi.org/project/prefetch_generator/
    }

    Args:
        dataset (Dataset): dataset defined by user
        data_cfg (Dict): data config

    Returns:
        data loader
    """

    ddp_sampler = DistributedSampler(
        dataset,
        get_world_size(),
        get_rank(),
        shuffle=data_cfg.get('SHUFFLE', False)
    )
    return (DataLoaderX if data_cfg.get('PREFETCH', False) else DataLoader)(
        dataset,
        collate_fn=data_cfg.get('COLLATE_FN', None),
        batch_size=data_cfg.get('BATCH_SIZE', 1),
        shuffle=False,
        sampler=ddp_sampler,
        num_workers=data_cfg.get('NUM_WORKERS', 0),
        pin_memory=data_cfg.get('PIN_MEMORY', False)
    )

