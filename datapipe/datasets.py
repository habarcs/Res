from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader


class DiffusionDataset(ImageFolder):
    def __init__(
        self,
        root: Union[str, Path],
        lq_transform: Optional[Callable] = None,
        hq_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        allow_empty: bool = False,
    ):
        super().__init__(
            root,
            None,
            target_transform,
            loader,
            is_valid_file,
            allow_empty,
        )
        self.lq_transform = lq_transform
        self.hq_transform = hq_transform

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (hq_image, lq_image, class)
        """
        path, target = self.imgs[index]
        sample = self.loader(path)
        if self.hq_transform is not None:
            hq = self.hq_transform(sample)
        else:
            hq = sample
        if self.lq_transform is not None:
            lq = self.lq_transform(sample)
        else:
            lq = sample
        if self.target_transform is not None:
            target = self.target_transform(target)

        return hq, lq, target
