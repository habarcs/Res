from pathlib import Path
from typing import Sized
from torch.utils import data
from torch import nn, optim
import torch
from diffusion import Diffusion
from torch.optim.swa_utils import AveragedModel

from state import load_state, save_state


def train_loop(
    device: torch.device,
    save_dir: Path | str,
    diffusor: Diffusion,
    dataloader: data.DataLoader,
    val_dataloader: data.DataLoader | None,
    model: nn.Module,
    ema_model: AveragedModel | None,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    save_freq: int,
    val_freq:int,
    log_freq:int,
    load=False
):
    assert dataloader.batch_size
    assert isinstance(dataloader.dataset, Sized)

    if load:
        load_state(save_dir, model, ema_model, optimizer)

    model.train()

    size = len(dataloader.dataset)

    for batch, (lq, hq) in enumerate(dataloader):
        lq.to(device)
        hq.to(device)
        t = diffusor.sample_timesteps().to(device)
        x_t = diffusor.forward_process(lq, hq, t)
        pred = model(x_t, lq, t)
        loss = loss_fn(pred, hq)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if ema_model:
            ema_model.update_parameters(model)

        if batch and batch % log_freq == 0:
            current = batch * dataloader.batch_size + len(lq)
            print(f"Train~~ loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")
            # TODO log tensorboard
        if batch and val_dataloader and batch % val_freq ==0:
            test_loop(device, diffusor, val_dataloader, model, loss_fn)
        if batch and save_freq and batch % save_freq == 0:
            save_state(save_dir, model, ema_model, optimizer)


def test_loop(
    device: torch.device, diffusor: Diffusion, dataloader: data.DataLoader, model: nn.Module, loss_fn: nn.Module
):
    assert dataloader.batch_size
    assert isinstance(dataloader.dataset, Sized)
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for batch, (lq, hq) in enumerate(dataloader):
            lq.to(device)
            hq.to(device)
            pred = diffusor.reverse_process(lq, model)
            loss = loss_fn(pred, hq).item()
            test_loss += loss
            current = batch * dataloader.batch_size + len(lq)
            print(f"Val~~~~ loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    test_loss /= num_batches
    print(f"Val~~~~ Avg loss: {test_loss:>8f} \n")
