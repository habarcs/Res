from pathlib import Path
import torch

import config


def save_state(
    cfg: config.TrainingCfg,
    batch_id: str,
    test_loss: float,
    model: torch.nn.Module,
    ema_model: torch.nn.Module | None,
):
    path = cfg.save_dir / cfg.run_id / "models"
    path.mkdir(parents=True, exist_ok=True)
    formatted_loss = f"{test_loss:.2e}".replace(".", "_").replace("-", "m")
    formatted_batch_id = f"{batch_id:07d}"
    file = path / f"{formatted_batch_id}_diffusion_{formatted_loss}.pth"
    torch.save(
        {
            "model": model.state_dict() if model else None,
            "ema_model": ema_model.state_dict() if ema_model else None,
        },
        file,
    )


def load_state(
    file: Path,
    model: torch.nn.Module | None,
    ema_model: torch.nn.Module | None,
) -> None:
    checkpoint = torch.load(file)
    if model and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    if ema_model and "ema_model" in checkpoint:
        ema_model.load_state_dict(checkpoint["ema_model"])
