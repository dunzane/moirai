# pylint: disable=inconsistent-quotes,too-many-positional-arguments
import functools
import random
from typing import Callable, Optional, Union, Dict, Any

import torch

from moirai.device import get_device_count, get_device_type, set_device_type, set_device

# default master rank
MASTER_RANK = 0


def get_rank() -> int:
    """Get the rank of the current process group.

    If DDP is initialized, return `torch.distributed.get_rank()`.
    Else return 0

    Returns:
        rank (int)
    """

    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return 0


def get_local_rank() -> int:
    """Get the local rank of a current process group in multiple compute nodes.

    Returns:
        local_rank (int)
    """

    return get_rank() % get_device_count() if get_device_count() != 0 else 0


def get_world_size() -> int:
    """Get the number of processes in the current process group.

    If DDP is initialized, return ```torch.distributed.get_world_size()```.
    Else return 1

    Returns:
        world_size (int)
    """

    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    else:
        return 1


def is_rank(rank: int) -> bool:
    """Checking if the rank of the current process group is equal to ```rank```.

    Notes:
        ```rank``` must be less than ```world_size```

    Args:
        rank (int): rank

    Returns:
        result (bool)
    """

    if rank >= get_world_size():
        raise ValueError('Rank is out of range')

    return get_rank() == rank


def is_master() -> bool:
    """Checking if a current process is a master process.

    The rank of master process is ```MASTER_RANK```

    Returns:
        result (bool)
    """

    return is_rank(MASTER_RANK)


def master_only(func):
    """A function decorator that the function is only executed in the master process.

    Examples:
        @master_only
        def func(x):
            return 2 ** x

    Args:
        func: function

    Returns:
        wrapper func
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_master():
            return func(*args, **kwargs)

    return wrapper


def dist_wrap(
        func: Callable[..., Any],
        node_num: int = 1,
        device_num: int = 1,
        node_rank: int = 0,
        dist_backend: Optional[Union[str, torch.distributed.Backend]] = None,
        init_method: Optional[str] = None
) -> Callable[..., Any]:
    """Wrap a function for distributed training using torch.multiprocessing.spawn."""

    _check_dist_args(node_num, device_num, node_rank)

    world_size = node_num * device_num

    if world_size <= 1:
        return func

    dist_backend = dist_backend or 'nccl'
    if init_method is None:
        if node_num == 1:
            port = random.randint(50000, 65000)
            init_method = f'tcp://127.0.0.1:{port}'
        else:
            raise ValueError("init_method must be specified when using multiple nodes.")

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        dist_params = {
            'device_type': get_device_type(),
            'device_num': device_num,
            'node_rank': node_rank,
            'world_size': world_size,
            'dist_backend': dist_backend,
            'init_method': init_method
        }

        torch.multiprocessing.spawn(
            dist_func,
            args=(dist_params, func, args, kwargs),
            nprocs=device_num,
            join=True
        )

    return wrapper


def dist_func(local_rank: int, dist_params: Dict[str, Any], func: Callable, args: Any, kwargs: Any):
    """Function launched by each distributed process."""
    rank = dist_params['device_num'] * dist_params['node_rank'] + local_rank
    world_size = dist_params['world_size']

    print(
        f"[Distributed] Launching process: "
        f"world_size={world_size}, rank={rank}, local_rank={local_rank}, "
        f"backend={dist_params['dist_backend']}, init_method={dist_params['init_method']}"
    )

    set_device_type(dist_params['device_type'])
    set_device(local_rank)

    torch.distributed.init_process_group(
        backend=dist_params['dist_backend'],
        init_method=dist_params['init_method'],
        rank=rank,
        world_size=world_size
    )

    try:
        func(*args, **kwargs)
    finally:
        torch.distributed.destroy_process_group()


def _check_dist_args(node_num: int, device_num: int, node_rank: int):
    """Basic validation for distributed arguments."""
    if node_num < 1:
        raise ValueError('The node_num must be greater than 1!')
    if device_num < 0:
        raise ValueError('The device_num must be greater than 0!')
    if node_rank >= node_num:
        raise ValueError("node_rank must be < node_num.")
    if device_num != get_device_count():
        raise RuntimeError(
            f"Mismatch in device count: expected {device_num}, "
            f"but torch.cuda.device_count() = {get_device_count()}."
        )
