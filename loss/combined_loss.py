import config
import torch
from classifier.cls_model import ClsModel
from loss.perceptual_loss import PerceptualLoss


class CombinedLoss(torch.nn.Module):
    def __init__(self, lambda_coef: float, other_loss: torch.nn.Module):
        self.other_loss = other_loss
        self.lambda_coef = lambda_coef
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, hq: torch.Tensor, gt: torch.Tensor):
        assert hq.shape == gt.shape
        return self.mse_loss(hq, gt) + self.lambda_coef * self.other_loss(hq, gt)

    @classmethod
    def from_config(cls, cfg: config.LossCfg):
        classifier = ClsModel.from_weights(cfg.perceptual_model_path)
        p_loss = PerceptualLoss(classifier, cfg.perceptual_loss_weights)
        return cls(cfg.perceptual_coef, p_loss)
