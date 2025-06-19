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
    formatted_loss = f"{test_loss:.2e}"
    formatted_batch_id = f"{batch_id:07d}"
    file = path / f"{formatted_batch_id}_diffusion_{formatted_loss}.pth"
    torch.save(
        {
            "model": model.state_dict() if model else None,
            "ema_model": ema_model.state_dict() if ema_model else None,
            "loss": test_loss,
        },
        file,
    )


def load_state(
    file: Path | str,
    model: torch.nn.Module,
    ema_model: torch.nn.Module | None,
) -> bool:
    checkpoint = torch.load(file, weights_only=True)
    ema_model_loaded = False
    assert "model" in checkpoint
    model.load_state_dict(checkpoint["model"])
    if ema_model and "ema_model" in checkpoint:
        ema_model.load_state_dict(checkpoint["ema_model"])
        ema_model_loaded = True
    print(f"Starting with a model of validation loss {checkpoint['loss']}")
    return ema_model_loaded
