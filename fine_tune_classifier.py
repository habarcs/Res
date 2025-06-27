import argparse

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import config
from classifier.cls_model import ClsModel, save_model, test_step, train_step
from datapipe.dataloader import classfication_data_loader_from_config
from upscaler.ema_model import ema_model_from_config


def get_optional_run_id() -> str | None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    return parser.parse_args().run_id


def fine_tune(
    fine_tune_cfg: config.ClassifierFineTuneCfg,
    data_cfg: config.ClassifierDataCfg,
    ema_cfg: config.EMACfg,
):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(2025)

    logger = SummaryWriter(fine_tune_cfg.save_dir / fine_tune_cfg.run_id / "log")

    train_loader, val_loader, _, classes = classfication_data_loader_from_config(
        data_cfg
    )

    model = ClsModel(len(classes)).to(device)
    ema_model = ema_model_from_config(model, ema_cfg, device)

    if fine_tune_cfg.compile:
        torch.set_float32_matmul_precision("high")
        model.compile()
        if ema_model:
            ema_model.compile()

    optimizer = AdamW(model.parameters(), lr=fine_tune_cfg.starting_lr)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=fine_tune_cfg.epochs, eta_min=fine_tune_cfg.ending_lr
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(fine_tune_cfg.epochs):
        print(f"Epoch {epoch + 1}/{fine_tune_cfg.epochs}")
        train_step(train_loader, logger, device, model, ema_model, loss_fn, optimizer)
        val_model = ema_model if ema_model else model
        acc = test_step(val_loader, logger, device, val_model, loss_fn, epoch=epoch)
        save_model(model, ema_model, fine_tune_cfg, epoch + 1, acc)
        scheduler.step()


if __name__ == "__main__":
    fine_tune_cfg = config.ClassifierFineTuneCfg()
    data_cfg = config.ClassifierDataCfg()
    ema_cfg = config.EMACfg()
    run_id = get_optional_run_id()
    if run_id:
        fine_tune_cfg.run_id = run_id
    fine_tune(fine_tune_cfg, data_cfg, ema_cfg)
