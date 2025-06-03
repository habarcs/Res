from typing import Callable
import math


def create_shifting_seq(T: int, p: float) -> Callable[[int], float]:
    """
    this implements the shifting sequence proposed by the authors in the paper
    T is the number of timesteps
    p is a hyperparameter affecting the sequence
    """
    assert T > 0
    assert 1 >= p >= 0

    def shifting_seq(t: int) -> float:
        """
        returns eta_t the noise intensity
        """
        assert 0 < t <= T
        if t == T:
            return 0.999
        if t == 1:
            return 0.001
        beta_t = (T - 1) * (((t - 1) / (T - 1)) ** p)
        b_0 = math.exp(1 / (2 * (T - 1)) * math.log(0.999 / 0.001))
        sqrt_eta_t = math.sqrt(0.001) * (b_0**beta_t)
        return sqrt_eta_t**2

    return shifting_seq
