"""
This file implements all the components needed for the residual shifting, the equations for the reverse and forward process (training and sampling)
"""

from typing import Callable
from torch import Tensor
import torch
import math


class Diffusion:
    def __init__(
        self,
        kappa: float,
        shifting_seq: Callable[[int], float],
        T: int,
        input_dim: tuple[int, int, int, int],
    ) -> None:
        """
        Initiates the diffusion class
        kappa: hyperparameter controlling the noise intensity
        shifting_sec: this function is generating the shifting sequence
        T: is the number of degradation steps, the length of the Markov chain
        input_dim: the dimension of the images,  B x C x H x W
        """
        assert kappa > 0
        assert T > 0
        assert min(input_dim) > 0
        self.kappa = kappa
        self.kappa_square = kappa * kappa
        self.shifting_seq = shifting_seq
        self.T = T
        self.input_dim = input_dim

    def forward_process(self, lq: Tensor, hq: Tensor, t: Tensor | None = None) -> Tensor:
        """
        Given the low quality and high quality images and a time-step it calculates the forward process and returns x_t
        q(x_t|x_0, y_0) = N(x_t; x_0 + eta_t * e_0, kappa^2 * eta_t * I)
        lq is y_0, hq is x_0
        """
        assert lq.shape == self.input_dim
        assert hq.shape == self.input_dim
        if t:
            assert t.shape == (self.input_dim[0], 1, 1, 1)
            assert t.min().item() > 0 and t.max().item() <= self.T
        else:
            t = self.sample_timesteps()

        e_0 = hq - lq
        eta_t = self._shift_t(t)
        mean = hq + (eta_t * e_0)
        return torch.normal(mean, self.get_variance(t).sqrt())

    def reverse_process(self, lq: Tensor, f_theta: torch.nn.Module) -> Tensor:
        """
        Takes a low quality image y and return x_0 by running the reverse process
        lq is the low quality image
        f_theta is a model trained to predict x_0 from lq, x_t and t
        """
        assert lq.shape == self.input_dim
        x_t = torch.normal(lq, self.get_variance(self.T).sqrt())

        for t in range(self.T, 0, -1):
            if t > 1:
                e = torch.randn_like(x_t)
                eta_t = self.shifting_seq(t)
                eta_t_1 = self.shifting_seq(t - 1)
                alpha_t = eta_t - eta_t_1
                x_0 = f_theta(x_t, lq, t).detach()
                mean = ((eta_t_1 / eta_t) * x_t) + ((alpha_t / eta_t) * x_0)
                x_t = mean + self.kappa * math.sqrt(eta_t_1 * alpha_t / eta_t) * e
            else:
                x_t = f_theta(x_t, lq, t).detach()

        return x_t

    def sample_timesteps(self) -> Tensor:
        """
        Samples batch number of timesteps from an uniform distribution between 1 and T
        """
        return torch.randint(1, self.T + 1, (self.input_dim[0], 1, 1, 1))

    def get_variance(self, t: int | Tensor) -> Tensor:
        """
        Returns the variance at timestep t
        kappa^2 * eta_t * I
        """
        if isinstance(t, int):
            assert 0 < t <= self.T
        else:
            assert t.shape == (self.input_dim[0], 1, 1, 1)
            assert t.min().item() > 0 and t.max().item() <= self.T
        b, c, h, w = self.input_dim
        identity = torch.eye(h, w).unsqueeze(0).repeat((c, 1, 1)).unsqueeze(0).repeat((b, 1, 1, 1))
        eta_t = self._shift_t(t)
        variance = self.kappa_square * eta_t * identity
        return variance

    def _shift_t(self, t: int|Tensor)-> float|Tensor:
        """
        Helper function that allows shifting seq to also be used on tensors
        if t is int, returns a single float
        else if t is a tensor of ints returns a tensor of floats
        """
        if isinstance(t, int):
            return self.shifting_seq(t)
        # type manipulation is needed, because map only works on tensors of the same type, t is ints but the shifting seq is floats
        return torch.zeros_like(t, dtype=torch.float).map_(t.float(), lambda _, x: self.shifting_seq(int(x)))
