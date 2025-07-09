from upscaler.resshift_unet.unet import UNetModelSwin
from torch.nn import Module


def create_resshift_model(image_size: int, grayscale: bool) -> Module:
    in_channels = 1 if grayscale else 3
    out_channels = in_channels
    return UNetModelSwin(
        image_size=image_size,
        in_channels=in_channels,
        model_channels=160,
        out_channels=out_channels,
        num_res_blocks=[2, 2, 2, 2],
        attention_resolutions=[64, 32, 16, 8],
        channel_mult=[1, 2, 2, 4],
        dims=2,
        use_scale_shift_norm=True,
        swin_embed_dim=192,
        mlp_ratio=4,
        lq_size=image_size,
    )
