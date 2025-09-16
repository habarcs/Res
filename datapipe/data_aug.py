from typing import Sequence

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode, v2
from transforms.gaussian_shadow import GaussianShadow
from transforms.haze import HazeArtifact
from transforms.attenuation import DepthAttenuation


def create_row_transforms(
    size: int,
    scale: int,
    mean: float,
    std: float,
    grayscale: bool,
    ultrasound_transforms: bool,
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

    random_apply = [ultrasound_augmentation()] if ultrasound_transforms else []

    hq_transform = v2.Compose(base_transform + normalizer)
    lq_transform = v2.Compose(base_transform + row_scaler + random_apply + normalizer)
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


def ultrasound_augmentation() -> v2.Transform:
    brightness = v2.ColorJitter(
        brightness=[-0.2, 0.2],
    )
    contrast = v2.ColorJitter(
        contrast=[-0.2, 0.2],
    )
    noise = v2.GaussianNoise(mean=0.0, sigma=0.0225)

    haze = HazeArtifact()
    depth_attenuation = DepthAttenuation()
    gauss_shadow = GaussianShadow()

    random = v2.RandomChoice(
        [brightness, contrast, noise, haze, depth_attenuation, gauss_shadow]
    )
    return random
