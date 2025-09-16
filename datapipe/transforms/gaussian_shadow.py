from typing import Any, Dict, List
import torch
from torchvision.transforms import v2
from torchvision.utils import save_image


class GaussianShadow(v2.Transform):
    def __init__(
        self,
        shadow_size: tuple[float, float] = (0.01, 0.2),
        shadow_strength: tuple[float, float] = (0.25, 0.8),
    ) -> None:
        super().__init__()
        self.shadow_size = shadow_size
        self.shadow_strength = shadow_strength

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        _, H, W = inpt.shape
        x = torch.tile(torch.linspace(0, 1, W), (H, 1))
        y = torch.tile(torch.linspace(0, 1, H).unsqueeze(1), (1, W))
        mu_x = x[0, params["x_center"]]
        mu_y = y[params["y_center"], 0]
        s = params["strength"]
        sigma_x = params["x_dim"]
        sigma_y = params["y_dim"]

        shad = 1 - (
            s
            * torch.exp(
                -(torch.pow(x - mu_x, 2) / (2 * torch.pow(sigma_x, 2)))
                - (torch.pow(y - mu_y, 2) / (2 * torch.pow(sigma_y, 2)))
            )
        )
        result = torch.mul(inpt, shad)
        return result

    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        _, H, W = flat_inputs[0].shape
        params = dict()
        params["x_center"] = torch.randint(0, W, [1])
        params["y_center"] = torch.randint(0, H, [1])
        uni = torch.distributions.Uniform(*self.shadow_size, True)
        params["x_dim"] = uni.sample()
        params["y_dim"] = uni.sample()
        params["strength"] = torch.distributions.Uniform(
            *self.shadow_strength, True
        ).sample()
        return params


if __name__ == "__main__":
    shadow = GaussianShadow()
    input = torch.ones(3, 224, 224)
    save_image(shadow(input), "shadow.png")
