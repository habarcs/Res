from pathlib import Path
import torch
from torch.utils import data
from torchvision import datasets
from torchvision.transforms import v2

from datapipe.data_aug import create_row_transforms


class DiffusionDataset(data.Dataset):
    def __init__(
        self, root: Path | str, train: bool, hq_transform: v2.Transform | None, lq_transform: v2.Transform | None
    ) -> None:
        super().__init__()
        self.root = root
        self.train = train
        self.hq_transform = hq_transform
        self.lq_transform = lq_transform
        self.dataset = datasets.CIFAR10(root, train, download=True)  # TODO swap

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        base_img, _ = self.dataset[idx]
        hq_img = self.hq_transform(base_img) if self.hq_transform else base_img
        lq_img = self.lq_transform(base_img) if self.lq_transform else base_img
        return hq_img, lq_img


def create_dataloader(
    dataset: data.Dataset, batch_size: int, num_workers: int, sampler: data.Sampler | None = None
) -> data.DataLoader:
    return data.DataLoader(dataset, batch_size, sampler=sampler, num_workers=num_workers)


def create_cifar_dataloaders(
    data_dir: Path | str,
    img_size: int,
    sf: int,
    mean: float,
    std: float,
    batch_size: int,
    num_workers: int,
    num_batches: int,
) -> tuple[data.DataLoader, data.DataLoader]:
    hq_transform, lq_transform = create_row_transforms(img_size, sf, mean, std)
    train_dataset = DiffusionDataset(data_dir, True, hq_transform, lq_transform)
    val_dataset = DiffusionDataset(data_dir, False, hq_transform, lq_transform)
    train_sampler = data.RandomSampler(train_dataset, replacement=True)
    train_loader = data.DataLoader(train_dataset, batch_size, sampler=train_sampler, num_workers=num_workers)
    val_loader = data.DataLoader(val_dataset, batch_size, num_workers=num_workers)
    return train_loader, val_loader
