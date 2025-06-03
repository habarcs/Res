from typing import Iterator, Sized
import torch
from torch.utils import data

from datapipe.data_aug import create_row_transforms
import config
from datapipe.datasets import DiffusionDataset
import random


class InfiniteRandomSampler(data.Sampler[int]):
    """
    A random infinite sampler with replacement
    """

    def __init__(self, data_source: Sized) -> None:
        super().__init__()
        self.n = len(data_source)

    def __iter__(self) -> Iterator[int]:
        yield random.randrange(0, self.n)


def data_loader_from_config(
    cfg: config.DataCfg,
) -> tuple[data.DataLoader, data.DataLoader, data.DataLoader]:
    hq_transform, lq_transform = create_row_transforms(
        cfg.image_size, cfg.scale_factor, cfg.mean, cfg.std
    )
    dataset = DiffusionDataset(cfg.data_dir, lq_transform, hq_transform)
    split_generator = torch.Generator().manual_seed(
        cfg.split_seed
    )  # assure that split is the same everytime
    train, val, test = data.random_split(
        dataset, [cfg.train_ratio, cfg.val_ratio, cfg.test_ratio], split_generator
    )
    train_sampler = InfiniteRandomSampler(train)
    train_loader = data.DataLoader(
        train, cfg.batch_size, sampler=train_sampler, num_workers=cfg.num_workers
    )
    val_loader = data.DataLoader(val, cfg.batch_size, num_workers=cfg.num_workers)
    test_loader = data.DataLoader(test, cfg.batch_size, num_workers=cfg.num_workers)
    return train_loader, val_loader, test_loader
