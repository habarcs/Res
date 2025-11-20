import argparse
import config
from pathlib import Path
import torch

from datapipe.dataloader import data_loader_from_config
from diffusion.diffusion import Diffusion
from loss.combined_loss import CombinedLoss
from taming.models.vqgan import VQModel
from training.saver import load_state
from training.trainer import eval_loop
from upscaler.smp_model import SmpModel
from torch.utils.tensorboard import SummaryWriter


def get_args() -> tuple[str, str, bool]:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--run-id", default="")
    args = parser.parse_args()
    return args.model_path, args.run_id, args.no_ema


def evaluate_model(model_path: str, run_id: str, no_ema: bool):
    data_cfg = config.DataCfg()
    model_cfg = config.ModelCfg()
    diffusion_cfg = config.DiffusionCfg()
    training_cfg = config.TrainingCfg()
    loss_cfg = config.LossCfg()
    autoencoder_cfg = config.VQGANCfg()

    if run_id:
        log_path = training_cfg.save_dir / run_id / "eval_log"
    else:
        log_path = Path(model_path).parent / "eval_log"
    logger = SummaryWriter(log_path)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(2025)

    _, _, test_loader, classes = data_loader_from_config(data_cfg)
    model = SmpModel.from_config(model_cfg, data_cfg, diffusion_cfg, autoencoder_cfg)
    load_state(model_path, model, not no_ema)
    model.to(device)

    diffusor = Diffusion.from_config(diffusion_cfg)

    combined_loss = CombinedLoss.from_config(loss_cfg, len(classes)).to(device)

    if model_cfg.autoencoder:
        autoencoder = VQModel(
            autoencoder_cfg.ddconfig,
            None,
            autoencoder_cfg.n_embed,
            autoencoder_cfg.embed_dim,
            model_cfg.autoencoder_model_path,
        )
        for param in autoencoder.parameters():
            param.requires_grad = False
        autoencoder.eval()
        autoencoder.to(device)
    else:
        autoencoder = None

    if training_cfg.compile:
        torch.set_float32_matmul_precision("high")
        model.compile()
        combined_loss.compile()

    loss = eval_loop(
        logger,
        "Test",
        0,
        device,
        autoencoder,
        diffusor,
        test_loader,
        model,
        combined_loss,
        data_cfg.mean,
        std=data_cfg.std,
    )
    print(f"Final test loss: {loss}")


if __name__ == "__main__":
    model_path, run_id, no_ema = get_args()
    evaluate_model(model_path, run_id, no_ema)
