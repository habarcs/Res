from typing import Any, Dict, List
from torchvision.transforms import v2
import torch
from torchvision.utils import save_image


class DepthAttenuation(v2.Transform):
    def __init__(
        self, max_atten: float = 0.0, atten_rate: tuple[float, float] = (0.0, 3.0)
    ) -> None:
        super().__init__()
        self.max_atten = max_atten
        self.atten_rate = atten_rate

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        _, H, W = inpt.shape
        x = torch.tile(torch.linspace(0, 1, W), (H, 1))
        y = torch.tile(torch.linspace(0, 1, H).unsqueeze(1), (1, W))
        d = torch.sqrt(torch.pow(x - 0.5, 2) + torch.pow(y, 2))
        rate = params["attenuation_rate"]
        atten = (1 - self.max_atten) * torch.exp(-1 * rate * d) + self.max_atten
        return torch.mul(atten, inpt)

    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        params = dict()
        params["attenuation_rate"] = torch.distributions.Uniform(
            *self.atten_rate, validate_args=True
        ).sample()
        return params


if __name__ == "__main__":
    depth = DepthAttenuation()
    input = torch.zeros([3, 224, 244])
    depth(input)
