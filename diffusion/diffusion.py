"""
This file implements all the components needed for the residual shifting, the equations for the reverse and forward process (training and sampling)
"""

from typing import Callable
from torch import Tensor
import torch
import math
import config
from diffusion.shifting_sequence import create_shifting_seq


class Diffusion:
    def __init__(
        self,
        kappa: float,
        T: int,
        shifting_seq: Callable[[int], float],
    ) -> None:
        """
        Initiates the diffusion class
        kappa: hyperparameter controlling the noise intensity
        T: int,
        shifting_sec: this function is generating the shifting sequence
        """
        assert kappa >= 0
        assert T > 0

        self.kappa = kappa
        self.shifting_seq = shifting_seq
        self.T = T

    @classmethod
    def from_config(cls, cfg: config.DiffusionCfg):
        return cls(cfg.kappa, cfg.T, create_shifting_seq(cfg.T, cfg.p))

    def forward_process(self, lq: Tensor, hq: Tensor, t: Tensor) -> Tensor:
        """
        Given the low quality and high quality images and a time-step it calculates the forward process and returns x_t
        q(x_t|x_0, y_0) = N(x_t; x_0 + eta_t * e_0, kappa^2 * eta_t * I)
        lq is y_0, hq is x_0
        """
        assert lq.shape == hq.shape
        assert lq.dim() == 4
        assert t.dim() == 1
        assert t.shape[0] == lq.shape[0]
        assert t.min().item() > 0 and t.max().item() <= self.T

        e_0 = lq - hq
        eta_t = self.__shift_t(t)
        mean = hq + (eta_t.reshape(-1, 1, 1, 1) * e_0)
        std = self.kappa * eta_t.sqrt()
        return torch.normal(mean, std.reshape(-1, 1, 1, 1))

    @torch.no_grad()
    def reverse_process(
        self, lq: Tensor, f_theta: torch.nn.Module, collect_progress: bool
    ) -> tuple[Tensor, list[Tensor]]:
        """
        Takes a low quality image y and return x_0 by running the reverse process
        lq is the low quality image
        f_theta is a model trained to predict x_0 from lq, x_t and t
        """
        assert lq.dim() == 4

        t_dim = (lq.shape[0],)
        eta_T = self.shifting_seq(self.T)
        x_t = torch.normal(lq, self.kappa * math.sqrt(eta_T))
        progress = []

        for t in range(self.T, 1, -1):  # T, T-1, ... 3, 2
            e = torch.randn_like(x_t)
            eta_t = self.shifting_seq(t)
            eta_t_1 = self.shifting_seq(t - 1)
            alpha_t = eta_t - eta_t_1
            x_0 = f_theta(x_t, lq, torch.full(t_dim, t))
            mean = ((eta_t_1 / eta_t) * x_t) + ((alpha_t / eta_t) * x_0)
            x_t = mean + self.kappa * math.sqrt(eta_t_1 * alpha_t / eta_t) * e
            if collect_progress and t in [self.T, 2, (self.T + 1) // 2]:
                progress.append(x_t)

        return f_theta(x_t, lq, torch.full(t_dim, 1, device=lq.device)), progress

    def sample_timesteps(self, batch_size) -> Tensor:
        """
        Samples batch number of timesteps from an uniform distribution between 1 and T
        """
        return torch.randint(1, self.T + 1, (batch_size,))

    def __shift_t(self, t: Tensor) -> Tensor:
        """
        Helper function that allows shifting seq to also be used on tensors
        """
        # type manipulation is needed, because map only works on tensors of the same type, t is ints but the shifting seq is floats
        # map_ only works on cpu tensors, but final tensor will be on same device as t
        return torch.zeros(t.size(), dtype=torch.float).map_(
            t.cpu().float(), lambda _, x: self.shifting_seq(int(x))
        ).to(t.device)
