from typing import Callable, List, cast
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders.mix_transformer import Attention
import torch
from torchvision.models.swin_transformer import ShiftedWindowAttentionV2

import config


class SWINAttention(ShiftedWindowAttentionV2):
    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:  # pyright: ignore[reportIncompatibleMethodOverride]
        return super().forward(x)


class SmpModel(torch.nn.Module):
    def __init__(
        self,
        arch: str,
        encoder_name: str,
        encoder_weights: str | None,
        input_dim: int,
        T: int,
        t_embedding_dim: int,
        shifting_window: bool,
    ) -> None:
        super().__init__()
        assert t_embedding_dim > 0
        assert T > 0

        in_channels = 2 * input_dim  # lq and x_t concat
        out_channels = input_dim
        smp_model = smp.create_model(
            arch,
            encoder_name,
            encoder_weights,
            in_channels=in_channels,
            classes=out_channels,
        )
        self.encoder = cast(torch.nn.Module, smp_model.encoder)
        self.decoder = cast(torch.nn.Module, smp_model.decoder)
        self.encoder_out_channels = cast(List[int], self.encoder.out_channels)
        self.segmentation_head = cast(torch.nn.Module, smp_model.segmentation_head)
        self.check_input_shape = cast(Callable, smp_model.check_input_shape)

        self.t_embedding = torch.nn.Embedding(T, t_embedding_dim)

        self.timestep_decoder_projections = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(t_embedding_dim, dim),
                    torch.nn.SiLU(),
                )
                for dim in self.encoder_out_channels
            ]
        )
        if shifting_window:
            self.replace_attention_with_swin_attention()

    def replace_attention_with_swin_attention(self):
        for module in self.modules():
            for name, child in module.named_children():
                if isinstance(child, Attention):
                    shifted_attention = SWINAttention(
                        child.dim,
                        [8, 8],
                        [4, 4],
                        child.num_heads,
                    )
                    setattr(module, name, shifted_attention)

    def forward(
        self, x_t: torch.Tensor, lq: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        assert x_t.shape == lq.shape
        assert x_t.dim() == 4
        assert t.shape == (x_t.shape[0],)
        x = torch.cat((x_t, lq), dim=1)
        self.check_input_shape(x)

        t_embedded = self.t_embedding(
            t - 1
        )  # timesteps are between 1 - T, but for embedding need to be 0 - T-1

        x = x + self.timestep_decoder_projections[0](t_embedded).unsqueeze(2).unsqueeze(
            3
        )

        features = self.encoder(x)
        time_embedded_features = [
            out + t_encoder(t_embedded).unsqueeze(2).unsqueeze(3)
            for out, t_encoder in zip(
                features, self.timestep_decoder_projections, strict=True
            )
        ]

        decoder_output = self.decoder(time_embedded_features)

        preds = self.segmentation_head(decoder_output)

        return preds

    @classmethod
    def from_config(
        cls,
        model_cfg: config.ModelCfg,
        data_cfg: config.DataCfg,
        diff_cfg: config.DiffusionCfg,
        vqgan_cfg: config.VQGANCfg,
    ):
        if model_cfg.autoencoder:
            input_dim = vqgan_cfg.embed_dim
        elif data_cfg.grayscale:
            input_dim = 1
        else:
            input_dim = 3
        return cls(
            model_cfg.arch,
            model_cfg.encoder,
            model_cfg.encoder_weights,
            input_dim,
            diff_cfg.T,
            model_cfg.t_embedding_dim,
            model_cfg.swin_attention,
        )
