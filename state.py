from pathlib import Path
import torch

MODEL_FILE_NAME = "models.pth"


def save_state(
    batch: int,
    save_dir: Path | str,
    model: torch.nn.Module,
    ema_model: torch.nn.Module | None,
    optimizer: torch.optim.Optimizer,
):
    if isinstance(save_dir, str):
        save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    file = save_dir / (str(batch) + "_" + MODEL_FILE_NAME)
    torch.save(
        {
            "model": model.state_dict(),
            "ema_model": ema_model.state_dict() if ema_model else None,
            "optimizer": optimizer.state_dict(),
        },
        file,
    )


def load_state(
    load_path: Path | str,
    model: torch.nn.Module | None,
    ema_model: torch.nn.Module | None,
    optimizer: torch.optim.Optimizer | None,
) -> None:
    checkpoint = torch.load(load_path)
    if model and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    if ema_model and "ema_model" in checkpoint:
        ema_model.load_state_dict(checkpoint["ema_model"])
    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
