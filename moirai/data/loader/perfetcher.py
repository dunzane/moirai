# pylint: disable=raise-missing-from
import threading
from queue import Queue
from typing import Iterator, Optional

import torch
from torch.utils.data import DataLoader

from moirai.device import data_to_device
from moirai import device


class BackgroundGenerator(threading.Thread):
    """
    BackgroundGenerator

    Wraps a generator into a background-threaded generator using a queue.

    Args:
        generator (Iterator): Any iterable that yields data (e.g., a PyTorch DataLoader).
        max_prefetch (int): Maximum number of prefetch items to store in the queue.
                            Default is 1. Set to -1 for unlimited prefetch (not recommended).
    """

    def __init__(self, generator: Iterator, max_prefetch: int = 1):
        super().__init__()
        self.generator = generator
        self.queue = Queue(maxsize=0 if max_prefetch == -1 else max_prefetch)
        self.daemon = True
        self.start()

    def run(self):
        try:
            for item in self.generator:
                self.queue.put(item)
        finally:
            self.queue.put(None)  # Signal end-of-data

    def __iter__(self):
        return self

    def __next__(self):
        item = self.queue.get()
        if item is None:
            raise StopIteration
        return item


class DataLoaderX(DataLoader):
    """
    DataLoader with background prefetch using BackgroundGenerator.

    Usage:
        data_loader = DataLoaderX(...)
        for batch in data_loader:
            ...
    """

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class DevicePrefetcher:
    """
    Prefetches data batches from CPU to GPU asynchronously using CUDA streams.

    Usage:
        prefetcher = DevicePrefetcher(data_loader)
        for batch in prefetcher:
            ...  # batch is already on GPU
    """

    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.stream = torch.cuda.Stream()
        self.batch_data: Optional[dict] = None
        self.data_loader_iter: Optional[Iterator] = None

    def preload(self):
        try:
            self.batch_data = next(self.data_loader_iter)
            with device.stream(self.stream):
                self.batch_data = data_to_device(self.batch_data)
        except StopIteration:
            self.batch_data = None
        except Exception as e:
            # Ensure error messages are informative
            raise RuntimeError(f"Error during preloading batch: {e}")

    def __next__(self):
        if self.batch_data is None:
            raise StopIteration

        device.current_stream().wait_stream(self.stream)
        batch = self.batch_data
        self.preload()
        return batch

    def __iter__(self):
        self.data_loader_iter = iter(self.data_loader)
        self.preload()
        return self

    def __len__(self):
        return len(self.data_loader)
