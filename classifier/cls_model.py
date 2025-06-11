from pathlib import Path
from typing import Self, cast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
import torch
from datapipe.dataloader import classfication_data_loader_from_config

import config
from datapipe.datasets import DiffusionDataset


class ClsModel(torch.nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.weights = torchvision.models.Swin_V2_T_Weights.DEFAULT
        self.backbone = torchvision.models.swin_v2_t(
            self.weights
        )
        # replace classifier head, to have right dimensions, needed, because of pre-trained weights
        num_features = self.backbone.head.in_features
        self.backbone.head = torch.nn.Linear(num_features, self.num_classes)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone(image)

    @classmethod
    def from_weights(cls, path: Path|str) -> Self:
        state = torch.load(path)
        model = cls(state["num_classes"])
        model.load_state_dict(state["weights"])
        return model

def save_model(self, cfg: config.ClassifierFineTuneCfg, id: str):
    path = cfg.save_dir / cfg.run_id / "models"
    path.mkdir(parents=True, exist_ok=True)
    file = path / f"{id}_classifier.pth"
    state = {
        "weights": self.state_dict,
        "num_classes": self.num_classes
    }
    torch.save(state, file)
        

def fine_tune():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    fine_tune_cfg = config.ClassifierFineTuneCfg()
    data_cfg = config.ClassifierDataCfg()
    train_loader, val_loader, test_loader = classfication_data_loader_from_config(data_cfg)
    dataset = cast(DiffusionDataset, train_loader.dataset)
    model = ClsModel(len(dataset.classes))
    optimizer = AdamW(model.parameters(), lr=fine_tune_cfg.starting_lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=fine_tune_cfg.epochs, eta_min=fine_tune_cfg.ending_lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    for epoch in range(fine_tune_cfg.epochs):
        print(f"Epoch {epoch+1}/{fine_tune_cfg.epochs}")
        _train_step(train_loader, device, model, loss_fn, optimizer)
        _test_step(val_loader, device, model, loss_fn)
        scheduler.step()

    _test_step(test_loader, device, model, loss_fn, "Test")
    
def _train_step(train_loader, device, model, loss_fn, optimizer):
    model.train()
    dataset_size = len(train_loader.dataset)
    for batch, (images, _, targets) in enumerate(train_loader):
        images, targets = images.to(device), targets.to(device)

        preds = model(images)
        loss = loss_fn(preds, targets)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            current = (batch +1) * len(images)
            print(
                f"Train: loss: {loss.item():>7f}  [{current:>5d}/{dataset_size:>5d}]"
            )

def _test_step(dataloader, device, model, loss_fn, label="Validation"):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for images, _, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            preds = model(images)
            test_loss += loss_fn(preds, targets).item()
            correct += (preds.argmax(1) == targets).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"{label} Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
