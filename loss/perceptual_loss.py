from typing import Sequence

import torch
from torch.nn import MSELoss
from torchvision.models.feature_extraction import create_feature_extractor

from classifier.cls_model import ClsModel


class PerceptualLoss(torch.nn.Module):
    def __init__(self, classifier: ClsModel, weights: Sequence[float]):
        super().__init__()
        assert len(weights) == classifier.num_layers
        assert all(weight >= 0 for weight in weights)
        self.classifier = classifier
        self.classifier.eval()
        for param in classifier.parameters():
            param.requires_grad = False
        return_nodes = {
            f"backbone.features.{i}": f"features.{i}" for i in range(self.classifier.num_layers)
        }
        self.extractor = create_feature_extractor(self.classifier, return_nodes)
        self.mse_loss = MSELoss()
        self.normalized_weights = [weight / sum(weights) for weight in weights]

    def forward(self, hq: torch.Tensor, gt: torch.Tensor) -> float:
        assert hq.shape == gt.shape
        loss = 0.0
        hq_features = self.extractor(hq)
        gt_features = self.extractor(gt)
        for hq_f, gt_f, weight in zip(
            hq_features, gt_features, self.normalized_weights
        ):
            loss += weight * self.mse_loss(hq_f, gt_f)
        return loss
