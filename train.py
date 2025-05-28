from pathlib import Path
from torch.utils import data
from torch import nn, optim
import torch
from diffusion import Diffusion
from torch.optim.swa_utils import AveragedModel

from state import save_state


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
    val_freq: int,
    log_freq: int,
):
    num_batches = len(dataloader)

    model.train()
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

        if (batch + 1) % log_freq == 0:
            print(f"Train~~ loss: {loss.item():>7f}  [{batch + 1:>5d}/{num_batches:>5d}]")
            # TODO log tensorboard
        if val_dataloader and (batch + 1) % val_freq == 0:
            test_loop(device, diffusor, val_dataloader, model, loss_fn, log_freq)
        if save_freq and (batch + 1) % save_freq == 0:
            save_state(batch + 1, save_dir, model, ema_model, optimizer)


def test_loop(
    device: torch.device,
    diffusor: Diffusion,
    dataloader: data.DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    log_freq: int,
):
    num_batches = len(dataloader)
    test_loss = 0

    model.eval()
    with torch.no_grad():
        for batch, (lq, hq) in enumerate(dataloader):
            lq.to(device)
            hq.to(device)
            pred = diffusor.reverse_process(lq, model)
            loss = loss_fn(pred, hq).item()
            test_loss += loss
            if (batch + 1) % log_freq == 0:
                print(f"Val~~~~ loss: {loss:>7f}  [{batch + 1:>5d}/{num_batches:>5d}]")
                # TODO log tensorboard

    test_loss /= num_batches
    print(f"Val~~~~ Avg loss: {test_loss:>8f} \n")
