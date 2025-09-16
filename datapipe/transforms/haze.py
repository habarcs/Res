from typing import Any, Dict, List
import torch
from torchvision.transforms import v2
from torchvision.utils import save_image


class HazeArtifact(v2.Transform):
    def __init__(
        self,
        radius: tuple[float, float] = (0.05, 0.95),
        sigma: tuple[float, float] = (0.0, 0.1),
    ) -> None:
        super().__init__()
        self.radius = radius
        self.sigma = sigma

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        _, H, W = inpt.shape
        x = torch.tile(torch.linspace(0, 1, W), (H, 1))
        y = torch.tile(torch.linspace(0, 1, H).unsqueeze(1), (1, W))
        d = torch.sqrt(torch.pow(x - 0.5, 2) + torch.pow(y, 2))
        u = params["u"]
        r = params["r"]
        sigma = params["sigma"]

        haze = 0.5 * u * torch.exp(-torch.pow(d - r, 2) / (2 * torch.pow(sigma, 2)))
        return torch.mul(haze, input)

    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        params = dict()
        params["u"] = torch.rand((1))
        params["r"] = torch.distributions.Uniform(*self.radius, True).sample()
        params["sigma"] = torch.distributions.Uniform(*self.sigma, True).sample()
        return params


if __name__ == "__main__":
    haze = HazeArtifact()
    input = torch.ones(3, 224, 224)
    save_image(haze(input), "haze.png")
