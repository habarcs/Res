from typing import Sequence
from torchvision.transforms import InterpolationMode, v2
from torchvision import transforms
import torch


def create_row_transforms(
    size: int, scale: int, mean: float, std: float, grayscale: bool
) -> tuple[v2.Transform, v2.Transform]:
    base_transform = [
        v2.PILToTensor(),
        v2.Resize(size),
        v2.CenterCrop(size),
    ]
    if grayscale:
        base_transform.append(v2.Grayscale(1))

    normalizer = [v2.ToDtype(torch.float32, scale=True), v2.Normalize((mean,), (std,))]
    row_scaler = [
        v2.Lambda(lambda img: img[..., 0::scale, :]),
        v2.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST),
    ]

    hq_transform = v2.Compose(base_transform + normalizer)
    lq_transform = v2.Compose(base_transform + row_scaler + normalizer)
    return hq_transform, lq_transform


def create_image_classification_transform(
    size: int, mean: Sequence[float], std: Sequence[float]
) -> v2.Transform:
    transform = v2.Compose(
        [
            v2.Resize(size, InterpolationMode.BICUBIC, antialias=True),
            v2.CenterCrop(size),
            v2.PILToTensor(),
            v2.ConvertImageDtype(),
            v2.Normalize(mean=mean, std=std),
        ]
    )
    return transform
