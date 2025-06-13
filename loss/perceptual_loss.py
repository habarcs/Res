import torch
import classifier.cls_model
from torchvision.models.feature_extraction import create_feature_extractor
from torch.nn import MSELoss

class PerceptualLoss(torch.nn.Module):
    def __init__(self, classifier: classifier.cls_model.ClsModel):
        super().__init__()
        self.classifier = classifier
        self.classifier.eval()
        for param in classifier.parameters():
            param.requires_grad=False
        return_nodes = {f"features.{i}": f"features.{i}" for i in range(8)}
        self.extractor = create_feature_extractor(self.classifier, return_nodes)
        self.mse_loss = MSELoss()
        
                
    def forward(self, hq: torch.Tensor, gt: torch.Tensor) -> float:
        assert hq.shape == gt.shape
        loss = 0.0
        hq_features = self.extractor(hq)
        gt_features = self.extractor(gt)
        for hq_f, gt_f in zip(hq_features, gt_features):
            loss += self.mse_loss(hq_f, gt_f)
        return loss
