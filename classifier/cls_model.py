from unittest.mock import Mock
from pathlib import Path
from typing import Self
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torch
from datapipe.dataloader import classfication_data_loader_from_config
from training.saver import find_best_model

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

    @classmethod
    def from_weights(cls, path: Path | str) -> Self:
        state = torch.load(path)
        model = cls(state["num_classes"])
        model.load_state_dict(state["weights"])
        return model


def eval_best(
    fine_tune_cfg: config.ClassifierFineTuneCfg, data_cfg: config.ClassifierDataCfg
):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    _, _, test_loader, _ = classfication_data_loader_from_config(data_cfg)
    best_model_path = find_best_model(fine_tune_cfg, False)
    model = ClsModel.from_weights(best_model_path)
    loss_fn = torch.nn.CrossEntropyLoss()

    acc = _test_step(test_loader, Mock(), device, model, loss_fn)
    print(f"Final test accuracy: {acc}")


def fine_tune(
    fine_tune_cfg: config.ClassifierFineTuneCfg, data_cfg: config.ClassifierDataCfg
):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger = SummaryWriter(fine_tune_cfg.save_dir / fine_tune_cfg.run_id / "log")
    train_loader, val_loader, _, classes = classfication_data_loader_from_config(
        data_cfg
    )
    model = ClsModel(len(classes))
    optimizer = AdamW(model.parameters(), lr=fine_tune_cfg.starting_lr)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=fine_tune_cfg.epochs, eta_min=fine_tune_cfg.ending_lr
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(fine_tune_cfg.epochs):
        print(f"Epoch {epoch + 1}/{fine_tune_cfg.epochs}")
        _train_step(train_loader, logger, device, model, loss_fn, optimizer)
        acc = _test_step(val_loader, logger, device, model, loss_fn)
        _save_model(model, fine_tune_cfg, epoch + 1, acc)
        scheduler.step()


def _train_step(train_loader, logger, device, model, loss_fn, optimizer):
    model.train()
    dataset_size = len(train_loader.dataset)
    for batch, (images, _, targets) in enumerate(train_loader):
        images, targets = images.to(device), targets.to(device)

        preds = model(images)
        loss = loss_fn(preds, targets)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        current = (batch + 1) * len(images)
        print(f"Train: loss: {loss.item():>7f}  [{current:>5d}/{dataset_size:>5d}]")
        logger.add_scalar("Train/loss", loss.item())


@torch.no_grad()
def _test_step(dataloader, logger, device, model, loss_fn, label="Val"):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0.0, 0

    for images, _, targets in dataloader:
        images, targets = images.to(device), targets.to(device)
        preds = model(images)
        test_loss += loss_fn(preds, targets).item()
        correct += (preds.argmax(1) == targets).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    acc = 100 * correct
    logger.add_scalar(f"{label}/loss", test_loss)
    logger.add_scalar(f"{label}/acc", acc)
    print(f"{label} Error: \n Accuracy: {(acc):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return acc


def _save_model(
    model: ClsModel, cfg: config.ClassifierFineTuneCfg, epoch_id: int, acc: float
):
    path = cfg.save_dir / cfg.run_id / "models"
    path.mkdir(parents=True, exist_ok=True)
    file = path / f"{epoch_id:03d}_classifier_{acc:06.2f}.pth"
    state = {"weights": model.state_dict, "num_classes": model.num_classes}
    torch.save(state, file)
