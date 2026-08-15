import queue
import threading

import numpy as np
import torch


def load_dataset(path: str) -> np.ndarray:
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(dataset: np.ndarray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    starts = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    x = np.stack([dataset[s : s + context_length] for s in starts]).astype(np.int64)
    y = np.stack([dataset[s + 1 : s + 1 + context_length] for s in starts]).astype(np.int64)
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    if device.startswith("cuda"):
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y


class BatchPrefetcher:
    def __init__(self, dataset: np.ndarray, batch_size: int, context_length: int, device: str, queue_size: int = 3):
        self.dataset = dataset
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device
        self.queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._produce, daemon=True)
        self.thread.start()

    def _produce(self) -> None:
        while not self.stop_event.is_set():
            batch = get_batch(self.dataset, self.batch_size, self.context_length, self.device)
            self.queue.put(batch)

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.queue.get()

    def close(self) -> None:
        self.stop_event.set()
        while not self.queue.empty():
            self.queue.get_nowait()
        self.thread.join(timeout=1.0)
