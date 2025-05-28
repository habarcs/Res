from pathlib import Path
import torch

MODEL_FILE_NAME = "models.pth"


def save_state(
    save_dir: Path | str,
    model: torch.nn.Module,
    ema_model: torch.nn.Module | None,
    optimizer: torch.optim.Optimizer,
):
    if isinstance(save_dir, str):
        save_dir = Path(save_dir)
    file = save_dir / MODEL_FILE_NAME
    torch.save(
        {
            "model": model.state_dict(),
            "ema_model": ema_model.state_dict() if ema_model else None,
            "optimizer": optimizer.state_dict(),
        },
        file,
    )


def load_state(
    save_dir: Path | str,
    model: torch.nn.Module,
    ema_model: torch.nn.Module | None,
    optimizer: torch.optim.Optimizer,
) -> None:
    if isinstance(save_dir, str):
        save_dir = Path(save_dir)
    file = save_dir / MODEL_FILE_NAME
    checkpoint = torch.load(file)
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    if ema_model and "ema_model" in checkpoint:
        ema_model.load_state_dict(checkpoint["ema_model"])
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
