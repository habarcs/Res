from typing import Callable, List, cast
import segmentation_models_pytorch as smp
import torch

import config


class SmpModel(torch.nn.Module):
    def __init__(
        self,
        arch: str,
        encoder_name: str,
        encoder_weights: str | None,
        grayscale: bool,
        T: int,
        t_embedding_dim: int,
    ) -> None:
        super().__init__()
        assert t_embedding_dim > 0
        assert T > 0

        in_channels = 2 if grayscale else 6
        out_channels = 1 if grayscale else 3
        smp_model = smp.create_model(
            arch,
            encoder_name,
            encoder_weights,
            in_channels=in_channels,
            classes=out_channels,
        )
        self.encoder = cast(torch.nn.Module, smp_model.encoder)
        self.decoder = cast(torch.nn.Module, smp_model.decoder)
        self.encoder_out_channels = cast(List[int], self.encoder._out_channels)
        self.segmentation_head = cast(torch.nn.Module, smp_model.segmentation_head)
        self.check_input_shape = cast(Callable, smp_model.check_input_shape)

        self.t_embedding = torch.nn.Embedding(T, t_embedding_dim)

        self.timestep_encoder_projections = [
            torch.nn.Sequential(
                torch.nn.Linear(t_embedding_dim, dim),
                torch.nn.SiLU(),
            )
            for dim in self.encoder_out_channels
        ]

    def forward(
        self, x_t: torch.Tensor, lq: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        assert x_t.shape == lq.shape
        assert x_t.dim() == 4
        assert t.shape == (x_t.shape[0],)
        x = torch.cat((x_t, lq), dim=1)
        self.check_input_shape(x)

        t_embedded = self.t_embedding(t - 1) # timesteps are between 1 - T, but for embedding need to be 0 - T-1

        features = self.encoder(x)
        time_embedded_features = [
            out + t_encoder(t_embedded).unsqueeze(2).unsqueeze(3)
            for out, t_encoder in zip(features, self.timestep_encoder_projections)
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
    ):
        return cls(
            model_cfg.arch,
            model_cfg.encoder,
            model_cfg.encoder_weights,
            data_cfg.grayscale,
            diff_cfg.T,
            model_cfg.t_embedding_dim,
        )
