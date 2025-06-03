from torch.utils import data
from torch import nn, optim
import torch
import config
from diffusion.diffusion import Diffusion
from torch.optim.swa_utils import AveragedModel

from training.saver import save_state, save_images


def train_loop(
    cfg: config.TrainingCfg,
    device: torch.device,
    diffusor: Diffusion,
    train_dataloader: data.DataLoader,
    val_dataloader: data.DataLoader | None,
    test_dataloader: data.DataLoader | None,
    model: nn.Module,
    ema_model: AveragedModel | None,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    start_iteration: int = 0,
):
    model.to(device)
    if ema_model:
        ema_model.to(device)

    train_iterator = iter(train_dataloader)
    for batch in range(start_iteration, cfg.iterations):
        model.train()
        hq, lq, _ = next(train_iterator)
        lq.to(device)
        hq.to(device)
        t = diffusor.sample_timesteps(lq.shape[0]).to(device)
        x_t = diffusor.forward_process(lq, hq, t)
        pred = model(x_t, lq, t)
        loss = loss_fn(pred, hq)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if ema_model:
            ema_model.update_parameters(model)

        if cfg.scheduler_freq and (batch + 1) % cfg.scheduler_freq == 0:
            scheduler.step()
        if cfg.log_freq and (batch + 1) % cfg.log_freq == 0:
            print(
                f"Train: loss: {loss.item():>7f}  [{batch + 1:>5d}/{cfg.iterations:>5d}]"
            )
            # TODO log tensorboard
        if cfg.val_freq and val_dataloader and (batch + 1) % cfg.val_freq == 0:
            val_model = ema_model if ema_model else model
            eval_loop(cfg, "Val", device, diffusor, val_dataloader, val_model, loss_fn)
        if cfg.save_freq and (batch + 1) % cfg.save_freq == 0:
            save_state(cfg, str(batch + 1), model, ema_model, optimizer, scheduler)

    if test_dataloader:
        test_model = ema_model if ema_model else model
        eval_loop(cfg, "Test", device, diffusor, test_dataloader, test_model, loss_fn)


def eval_loop(
    cfg: config.TrainingCfg,
    split: str,
    device: torch.device,
    diffusor: Diffusion,
    dataloader: data.DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
):
    num_batches = len(dataloader)
    test_loss = 0

    model.eval()
    model.to(device)
    with torch.no_grad():
        for batch, (hq, lq, _) in enumerate(dataloader):
            lq.to(device)
            hq.to(device)
            pred, progress = diffusor.reverse_process(lq, model, True)
            loss = loss_fn(pred, hq).item()
            test_loss += loss
            save_images(cfg, f"{split}_{batch + 1}", hq, lq, pred, progress)
            if cfg.log_freq and (batch + 1) % cfg.log_freq == 0:
                print(f"{split}: loss: {loss:>7f}  [{batch + 1:>5d}/{num_batches:>5d}]")
                # TODO log tensorboard

    test_loss /= num_batches
    print(f"{split}: Avg loss: {test_loss:>8f} \n")
