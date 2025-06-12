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
        },
        file,
    )


def find_best_model(
    cfg: config.TrainingCfg | config.ClassifierFineTuneCfg, lower_better: bool
) -> Path:
    """
    Finds the best model based on the name of the save file, it expects, the loss or acc to be before the extension
    """
    save_path = cfg.save_dir / cfg.run_id / "models"
    files = list(save_path.glob("*.pth"))
    metrics = [float(str(file.stem).split("_")[-1]) for file in files]
    arg = metrics.index(min(metrics)) if lower_better else metrics.index(max(metrics))
    return files[arg]


def load_state(
    file: Path,
    model: torch.nn.Module,
    ema_model: torch.nn.Module | None,
) -> bool:
    checkpoint = torch.load(file)
    ema_model_loaded = False
    assert "model" in checkpoint
    model.load_state_dict(checkpoint["model"])
    if ema_model and "ema_model" in checkpoint:
        ema_model.load_state_dict(checkpoint["ema_model"])
        ema_model_loaded = True
    return ema_model_loaded
