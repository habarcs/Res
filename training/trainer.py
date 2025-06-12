from torch.utils import data
from torch import nn, optim
import torch
from torch.utils.tensorboard import SummaryWriter
import config
from diffusion.diffusion import Diffusion
from torch.optim.swa_utils import AveragedModel

from training.saver import save_state


def train_loop(
    cfg: config.TrainingCfg,
    device: torch.device,
    logger: SummaryWriter,
    diffusor: Diffusion,
    train_dataloader: data.DataLoader,
    val_dataloader: data.DataLoader,
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

        logger.add_scalar("Train/loss", loss.item(), batch + 1)

        if cfg.scheduler_freq and (batch + 1) % cfg.scheduler_freq == 0:
            scheduler.step()
        if cfg.val_freq and (batch + 1) % cfg.val_freq == 0:
            val_model = ema_model if ema_model else model
            val_loss = eval_loop(cfg, logger, "Val", batch + 1, device, diffusor, val_dataloader, val_model, loss_fn)
            save_state(cfg, str(batch + 1), val_loss, model, ema_model)


@torch.no_grad()
def eval_loop(
    cfg: config.TrainingCfg,
    logger: SummaryWriter,
    split: str,
    iteration: int,
    device: torch.device,
    diffusor: Diffusion,
    dataloader: data.DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
) -> float:
    num_batches = len(dataloader)
    test_loss = 0.0
    batch_size = dataloader.batch_size
    assert isinstance(batch_size, int)

    model.eval()
    model.to(device)
    for batch, (hq, lq, _) in enumerate(dataloader):
        lq.to(device)
        hq.to(device)
        pred, progress = diffusor.reverse_process(lq, model, True)
        loss = loss_fn(pred, hq).item()
        test_loss += loss
        for i in range(len(lq)):
            image_id = batch_size * batch + i
            images = [lq[i] + [p[i] for p in progress] + pred[i] + hq[i]]
            logger.add_images(f"{split}/{image_id}", torch.stack(images))
        logger.add_scalar(f"{split}/loss", loss)

    test_loss /= num_batches
    logger.add_scalar(f"{split}/avgloss", test_loss, iteration)
    return test_loss
