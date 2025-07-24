import argparse
from dataclasses import asdict

import torch
from torch.utils.tensorboard import SummaryWriter

import config
from datapipe.dataloader import data_loader_from_config
from diffusion.diffusion import Diffusion
from loss.combined_loss import CombinedLoss
from training.trainer import train_loop
from ema.ema_model import ema_model_from_config
from upscaler.smp_model import SmpModel


def get_optional_run_id() -> str | None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    return parser.parse_args().run_id


def train():
    data_cfg = config.DataCfg()
    model_cfg = config.ModelCfg()
    diffusion_cfg = config.DiffusionCfg()
    training_cfg = config.TrainingCfg()
    run_id = get_optional_run_id()
    if run_id:
        training_cfg.run_id = run_id
    ema_cfg = config.EMACfg()
    loss_cfg = config.LossCfg()
    logger = SummaryWriter(training_cfg.save_dir / training_cfg.run_id / "log")

    for cfg in (data_cfg, model_cfg, diffusion_cfg, training_cfg, ema_cfg, loss_cfg):
        logger.add_text(cfg.__class__.__name__, str(asdict(cfg)))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(2025)

    train_loader, val_loader, _, classes = data_loader_from_config(data_cfg)
    model = SmpModel.from_config(model_cfg, data_cfg, diffusion_cfg).to(device)
    ema_model = ema_model_from_config(model, ema_cfg, device)
    diffusor = Diffusion.from_config(diffusion_cfg)

    combined_loss = CombinedLoss.from_config(loss_cfg, len(classes)).to(device)

    if training_cfg.compile:
        torch.set_float32_matmul_precision("high")
        model.compile()
        if ema_model:
            ema_model.compile()
        combined_loss.compile()

    optimizer = torch.optim.Adam(model.parameters(), training_cfg.lr_start)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        training_cfg.iterations // training_cfg.scheduler_freq,
        training_cfg.lr_end,
    )

    train_loop(
        training_cfg,
        device,
        logger,
        diffusor,
        train_loader,
        val_loader,
        model,
        ema_model,
        combined_loss,
        optimizer,
        scheduler,
    )
    logger.close()


if __name__ == "__main__":
    train()
