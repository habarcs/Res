from pathlib import Path
import torch
import torchvision

import config


def save_state(
    cfg: config.TrainingCfg,
    id: str,
    model: torch.nn.Module,
    ema_model: torch.nn.Module | None,
    optimizer: torch.optim.Optimizer,
):
    path = cfg.save_dir / cfg.run_id / "models"
    path.mkdir(parents=True, exist_ok=True)
    file = path / f"{id}_all.pth"
    torch.save(
        {
            "config": cfg.todict(),
            "model": model.state_dict(),
            "ema_model": ema_model.state_dict() if ema_model else None,
            "optimizer": optimizer.state_dict(),
        },
        file,
    )


def save_images(
    cfg: config.TrainingCfg,
    id: str,
    hq: torch.Tensor,
    lq: torch.Tensor,
    pred: torch.Tensor,
    progress: list[torch.Tensor],
):
    path = cfg.save_dir / cfg.run_id / "images"
    path.mkdir(parents=True, exist_ok=True)
    file = path / f"{id}.png"
    torchvision.utils.save_image([hq, lq] + progress + [pred], file)


def load_state(
    file: Path,
    model: torch.nn.Module | None,
    ema_model: torch.nn.Module | None,
    optimizer: torch.optim.Optimizer | None,
) -> config.TrainingCfg:
    checkpoint = torch.load(file)
    if model and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    if ema_model and "ema_model" in checkpoint:
        ema_model.load_state_dict(checkpoint["ema_model"])
    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if "config" in checkpoint:
        return config.TrainingCfg(**checkpoint["config"])
    return config.TrainingCfg()
