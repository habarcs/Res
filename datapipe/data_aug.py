from torchvision.transforms import v2
from torchvision import transforms
import torch


def create_row_transforms(
    size: int, scale: int, mean: float, std: float
) -> tuple[v2.Transform, v2.Transform]:
    base_transform = v2.Compose(
        [
            v2.PILToTensor(),
            v2.Resize(size),
            v2.CenterCrop(size),
        ]
    )
    normalizer = v2.Compose(
        [v2.ToDtype(torch.float32, scale=True), v2.Normalize((mean,), (std,))]
    )
    row_scaler = v2.Compose(
        [
            v2.Lambda(lambda img: img[..., 0::scale, :]),
            v2.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST),
        ]
    )

    hq_transform = v2.Compose([base_transform, normalizer])
    lq_transform = v2.Compose([base_transform, row_scaler, normalizer])
    return hq_transform, lq_transform
