from typing import Self
from classifier.cls_model import ClsModel
import config
import torch
from loss.perceptual_loss import PerceptualLoss


class CombinedLoss(torch.nn.Module):
    def __init__(self, lambda_coef: float, perceptual_loss: torch.nn.Module) -> None:
        self.perceptual_loss = perceptual_loss
        self.lambda_coef = lambda_coef
        self.mse_loss = torch.nn.MSELoss()

    def forward(
        self, hq: torch.Tensor, gt: torch.Tensor
    ) -> tuple[torch.Tensor, float, float]:
        assert hq.shape == gt.shape
        mse = self.mse_loss(hq, gt)
        percep = self.perceptual_loss(hq, gt)
        combined = mse + (self.lambda_coef * percep)
        return combined, percep.item(), mse.item()

    @classmethod
    def from_config(cls, cfg: config.LossCfg, num_classes: int) -> Self:
        classifier = ClsModel(num_classes)

        p_loss = PerceptualLoss(classifier, cfg.perceptual_loss_weights)
        return cls(cfg.perceptual_coef, p_loss)
