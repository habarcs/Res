import torch

class CombinedLoss(torch.nn.Module):
    def __init__(self, lambda_coef: float, other_loss: torch.nn.Module):
        self.other_loss = other_loss
        self.lambda_coef = lambda_coef
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, hq:torch.Tensor, gt:torch.Tensor):
        assert hq.shape == gt.shape
        return self.mse_loss(hq, gt) + self.lambda_coef * self.other_loss(hq, gt)
