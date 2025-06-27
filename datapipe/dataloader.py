from typing import Iterator, Sized
import torch
from torch.utils import data

from datapipe.data_aug import (
    create_image_classification_transform,
    create_row_transforms,
)
import config
from datapipe.datasets import DiffusionDataset
from torchvision.datasets import ImageFolder


class InfiniteRandomSampler(data.Sampler[int]):
    """
    A random infinite sampler with replacement
    """

    def __init__(self, data_source: Sized) -> None:
        super().__init__()
        self.n = len(data_source)

    def __iter__(self) -> Iterator[int]:
        while True:
            yield from torch.randint(high=self.n, size=(32,)).tolist()


def data_loader_from_config(
    cfg: config.DataCfg,
) -> tuple[
    data.DataLoader[DiffusionDataset],
    data.DataLoader[DiffusionDataset],
    data.DataLoader[DiffusionDataset],
    list[str],
]:
    hq_transform, lq_transform = create_row_transforms(
        cfg.image_size, cfg.scale_factor, cfg.mean, cfg.std, cfg.grayscale
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
    return train_loader, val_loader, test_loader, dataset.classes


def classfication_data_loader_from_config(
    cfg: config.ClassifierDataCfg,
) -> tuple[
    data.DataLoader[ImageFolder],
    data.DataLoader[ImageFolder],
    data.DataLoader[ImageFolder],
    list[str],
]:
    transform = create_image_classification_transform(cfg.image_size, cfg.mean, cfg.std)

    dataset = ImageFolder(cfg.data_dir, transform, None)

    split_generator = torch.Generator().manual_seed(
        cfg.split_seed
    )  # assure that split is the same everytime
    train, val, test = data.random_split(
        dataset, [cfg.train_ratio, cfg.val_ratio, cfg.test_ratio], split_generator
    )

    train_loader = data.DataLoader(
        train, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )

    val_loader = data.DataLoader(val, cfg.batch_size, num_workers=cfg.num_workers)

    test_loader = data.DataLoader(train, cfg.batch_size, num_workers=cfg.num_workers)

    return train_loader, val_loader, test_loader, dataset.classes
