from pathlib import Path
import torch

import config


def save_state(
    cfg: config.TrainingCfg,
    batch_id: str,
    test_loss: float,
    model: torch.nn.Module,
    ema_model: torch.nn.Module | None,
) -> None:
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
    load_ema: bool,
) -> None:
    checkpoint = torch.load(file, weights_only=True)
    assert "model" in checkpoint
    if "ema_model" in checkpoint and load_ema:
        model.load_state_dict(checkpoint["ema_model"])
        print("Ema model loaded")
    else:
        model.load_state_dict(checkpoint["model"])
        print("Model loaded")
    print(f"Starting with a model of validation loss {checkpoint['loss']}")
