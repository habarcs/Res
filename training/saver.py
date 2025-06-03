from pathlib import Path
import torch

import config

       
def save_state(
    cfg: config.TrainingCfg,
    id: str,
    model: torch.nn.Module,
    ema_model: torch.nn.Module | None,
    optimizer: torch.optim.Optimizer,
):
    cfg.save_path.mkdir(parents=True, exist_ok=True)
    file = cfg.save_path / f"{cfg.run_id}_{id}.pth"
    torch.save(
        {
            "config": cfg.todict(),
            "model": model.state_dict(),
            "ema_model": ema_model.state_dict() if ema_model else None,
            "optimizer": optimizer.state_dict(),
        },
        file,
    )


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
