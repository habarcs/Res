from typing import Self
from classifier.cls_model import ClsModel
import config
import torch
from loss.perceptual_loss import PerceptualLoss


class CombinedLoss(torch.nn.Module):
    def __init__(self, lambda_coef: float, perceptual_loss: torch.nn.Module) -> None:
        super().__init__()
        self.perceptual_loss = perceptual_loss
        self.lambda_coef = lambda_coef
        self.mse_loss = torch.nn.MSELoss()

        self._last_mse = 0.0
        self._last_percep = 0.0

    def forward(self, hq: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        assert hq.shape == gt.shape
        mse = self.mse_loss(hq, gt)
        percep = self.perceptual_loss(hq, gt)
        combined = mse + (self.lambda_coef * percep)

        self._last_mse = mse.item()
        self._last_percep = percep.item()

        return combined

    @property
    def last_mse(self) -> float:
        return self._last_mse

    @property
    def last_percep(self) -> float:
        return self._last_percep

    @classmethod
    def from_config(cls, cfg: config.LossCfg, num_classes: int) -> Self:
        classifier = ClsModel(num_classes)

        p_loss = PerceptualLoss(classifier, cfg.perceptual_loss_weights)
        return cls(cfg.perceptual_coef, p_loss)
