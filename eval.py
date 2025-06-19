import argparse
import config
import torch
from unittest.mock import Mock

from datapipe.dataloader import data_loader_from_config
from diffusion.diffusion import Diffusion
from training.saver import load_state
from training.trainer import eval_loop
from upscaler.ema_model import ema_model_from_config
from upscaler.smp_model import SmpModel


def get_args() -> tuple[str, bool]:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--no-ema", action="store_false")
    args = parser.parse_args()
    return args.model_path, args.no_ema


def evaluate_model(model_path: str, no_ema: bool = False):
    data_cfg = config.DataCfg()
    model_cfg = config.ModelCfg()
    diffusion_cfg = config.DiffusionCfg()
    training_cfg = config.TrainingCfg()
    ema_cfg = config.EMAModelCfg()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(2025)

    _, _, test_loader = data_loader_from_config(data_cfg)
    model = SmpModel.from_config(model_cfg, data_cfg, diffusion_cfg)
    ema_model = ema_model_from_config(model, ema_cfg)
    diffusor = Diffusion.from_config(diffusion_cfg)

    ema_model_loaded = load_state(model_path, model, ema_model)

    loss_fn = torch.nn.MSELoss()

    eval_model = ema_model if not no_ema and ema_model and ema_model_loaded else model
    loss = eval_loop(
        training_cfg,
        Mock(),
        "Test",
        0,
        device,
        diffusor,
        test_loader,
        eval_model,
        loss_fn,
    )
    print(f"Final test loss: {loss}")


if __name__ == "__main__":
    model_path, no_ema = get_args()
    evaluate_model(model_path, no_ema)
