from pathlib import Path
from typing import Sized
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torch

import config


class ClsModel(torch.nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.weights = torchvision.models.Swin_V2_T_Weights.DEFAULT
        self.backbone = torchvision.models.swin_v2_t(weights=self.weights)
        # replace classifier head, to have right dimensions, needed, because of pre-trained weights
        num_features = self.backbone.head.in_features
        self.backbone.head = torch.nn.Linear(num_features, self.num_classes)
        self.num_layers = len(self.backbone.features)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone(image)


def train_step(
    train_loader: DataLoader,
    logger: SummaryWriter,
    device: torch.device,
    model: ClsModel,
    ema_model: AveragedModel | None,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    model.train()
    assert isinstance(train_loader.dataset, Sized)
    dataset_size = len(train_loader.dataset)
    assert train_loader.batch_size
    batch_size = train_loader.batch_size
    for batch, (images, targets) in enumerate(train_loader):
        images, targets = images.to(device), targets.to(device)

        preds = model(images)
        loss = loss_fn(preds, targets)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if ema_model:
            ema_model.update_parameters(model)

        current = batch * batch_size + len(images)
        print(f"Train: loss: {loss.item():>7f}  [{current:>5d}/{dataset_size:>5d}]")
        logger.add_scalar("Train/loss", loss.item(), current)


@torch.no_grad()
def test_step(
    dataloader: DataLoader,
    logger: SummaryWriter,
    device: torch.device,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    label: str = "Val",
    epoch: int = 0,
):
    assert isinstance(dataloader.dataset, Sized)
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0.0, 0

    for images, targets in dataloader:
        images, targets = images.to(device), targets.to(device)
        preds = model(images)
        test_loss += loss_fn(preds, targets).item()
        correct += (preds.argmax(1) == targets).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    acc = 100 * correct
    logger.add_scalar(f"{label}/loss", test_loss, epoch + 1)
    logger.add_scalar(f"{label}/acc", acc, epoch + 1)
    print(
        f"{label} epoch {epoch + 1}: Accuracy: {(acc):>0.1f}%, Avg loss: {test_loss:>8f}\n"
    )
    return acc


def save_model(
    model: ClsModel,
    ema_model: AveragedModel | None,
    cfg: config.ClassifierFineTuneCfg,
    epoch_id: int,
    acc: float,
):
    path = cfg.save_dir / cfg.run_id / "models"
    path.mkdir(parents=True, exist_ok=True)
    file = path / f"{epoch_id:03d}_classifier_{acc:06.2f}.pth"
    state = {
        "model": model.state_dict(),
        "ema_model": ema_model.state_dict() if ema_model else None,
        "num_classes": model.num_classes,
        "acc": acc,
    }
    torch.save(state, file)


def load_model(model: ClsModel, path: Path | str, load_ema: bool) -> None:
    state = torch.load(path, weights_only=True)
    if "ema_model" in state and load_ema:
        model.load_state_dict(state["ema_model"])
        print("Loaded EMA model")
    else:
        model.load_state_dict(state["model"])
        print("Loaded model")
    print(f"Starting with a model of validation accuracy {state['acc']}")
