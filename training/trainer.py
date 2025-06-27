from torch.utils import data
from torch import nn, optim
import torch
from torch.utils.tensorboard import SummaryWriter
import config
from diffusion.diffusion import Diffusion
from torch.optim.swa_utils import AveragedModel

from loss.combined_loss import CombinedLoss
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
    loss_fn: CombinedLoss,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    start_iteration: int = 0,
):
    train_iterator = iter(train_dataloader)
    for iteration in range(start_iteration, cfg.iterations):
        model.train()
        hq, lq = next(train_iterator)
        lq, hq = lq.to(device), hq.to(device)
        t = diffusor.sample_timesteps(lq.shape[0]).to(device)
        x_t = diffusor.forward_process(lq, hq, t)
        pred = model(x_t, lq, t)
        loss, mse, percep = loss_fn(pred, hq)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if ema_model:
            ema_model.update_parameters(model)

        logger.add_scalar("Train/loss", loss.item(), iteration + 1)
        logger.add_scalar("Train/mseloss", mse, iteration + 1)
        logger.add_scalar("Train/perceploss", percep, iteration + 1)
        print(f"Train {iteration + 1} loss: {loss.item()}")

        if cfg.scheduler_freq and (iteration + 1) % cfg.scheduler_freq == 0:
            scheduler.step()
        if cfg.val_freq and (iteration + 1) % cfg.val_freq == 0:
            val_model = ema_model if ema_model else model
            val_loss = eval_loop(
                cfg,
                logger,
                "Val",
                iteration,
                device,
                diffusor,
                val_dataloader,
                val_model,
                loss_fn,
            )
            save_state(cfg, str(iteration + 1), val_loss, model, ema_model)


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
    loss_fn: CombinedLoss,
) -> float:
    num_batches = len(dataloader)
    test_loss = 0.0
    batch_size = dataloader.batch_size
    assert isinstance(batch_size, int)

    model.eval()
    for batch, (hq, lq) in enumerate(dataloader):
        lq, hq = lq.to(device), hq.to(device)
        pred, progress = diffusor.reverse_process(lq, model, True)
        loss, _, _ = loss_fn(pred, hq)
        test_loss += loss.item()
        for i in range(len(lq)):
            image_id = batch_size * batch + i
            images = [lq[i] + [p[i] for p in progress] + pred[i] + hq[i]]
            logger.add_images(f"{split}/{image_id}", torch.stack(images), iteration + 1)
        print(f"{split} {batch + 1} loss: {loss}")

    test_loss /= num_batches
    logger.add_scalar(f"{split}/avgloss", test_loss, iteration + 1)
    return test_loss
