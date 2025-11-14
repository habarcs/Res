from torch.utils import data
from torch import nn, optim
import torch
from torch.utils.tensorboard import SummaryWriter
import config
from diffusion.diffusion import Diffusion
from torch.optim.swa_utils import AveragedModel
from torch.nn.functional import l1_loss
from piq import psnr, ssim, LPIPS
from loss.combined_loss import CombinedLoss
from taming.models.vqgan import VQModel
from training.saver import save_state
from torchvision.transforms.v2.functional import to_dtype


def train_loop(
    cfg: config.TrainingCfg,
    device: torch.device,
    logger: SummaryWriter,
    diffusor: Diffusion,
    train_dataloader: data.DataLoader,
    val_dataloader: data.DataLoader,
    test_dataloader: data.DataLoader,
    model: nn.Module,
    ema_model: AveragedModel | None,
    autoencoder: VQModel | None,
    loss_fn: CombinedLoss | torch.nn.MSELoss,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    start_iteration: int = 0,
    eval_combined_loss: CombinedLoss | None = None,
):
    val_loss_fn = (
        loss_fn
        if eval_combined_loss is None or isinstance(loss_fn, CombinedLoss)
        else eval_combined_loss
    )
    train_iterator = iter(train_dataloader)
    for iteration in range(start_iteration, cfg.iterations):
        model.train()
        hq, lq = next(train_iterator)
        lq, hq = lq.to(device), hq.to(device)
        if autoencoder:
            z_lq, _, _ = autoencoder.encode(lq)
            z_hq, _, _ = autoencoder.encode(hq)
            t = diffusor.sample_timesteps(z_lq.shape[0]).to(device)
            z_x_t = diffusor.forward_process(z_lq, z_hq, t, device)
            z_pred = model(z_x_t, z_lq, t)
            loss_autoencoder = torch.nn.functional.mse_loss(z_pred, z_hq)
            loss_autoencoder.backward()
            pred = autoencoder.decode(z_pred)

            logger.add_scalar(
                "Train/latentmseloss", loss_autoencoder.item(), iteration + 1
            )
        else:
            t = diffusor.sample_timesteps(lq.shape[0]).to(device)
            x_t = diffusor.forward_process(lq, hq, t, device)
            pred = model(x_t, lq, t)

        loss = loss_fn(pred, hq)
        if not autoencoder:
            loss.backward()

        if (iteration + 1) % cfg.backprop_freq == 0:
            optimizer.step()
            optimizer.zero_grad()
            if ema_model:
                ema_model.update_parameters(model)

        logger.add_scalar("Train/loss", loss.item(), iteration + 1)
        if isinstance(loss_fn, CombinedLoss):
            logger.add_scalar("Train/mseloss", loss_fn.last_mse, iteration + 1)
            logger.add_scalar("Train/perceploss", loss_fn.last_percep, iteration + 1)
        logger.add_scalar("Train/lrrate", scheduler.get_last_lr()[0], iteration + 1)

        print(f"Train {iteration + 1} loss: {loss.item()}")

        if cfg.scheduler_freq and (iteration + 1) % cfg.scheduler_freq == 0:
            scheduler.step()
        if cfg.val_freq and (iteration + 1) % cfg.val_freq == 0:
            val_model = ema_model if ema_model else model
            val_loss = eval_loop(
                logger,
                "Val",
                iteration,
                device,
                autoencoder,
                diffusor,
                val_dataloader,
                val_model,
                val_loss_fn,
            )
            save_state(cfg, iteration + 1, val_loss, model, ema_model)

    test_model = ema_model if ema_model else model
    eval_loop(
        logger,
        "Test",
        cfg.iterations,
        device,
        autoencoder,
        diffusor,
        test_dataloader,
        test_model,
        val_loss_fn,
    )


@torch.no_grad()
def eval_loop(
    logger: SummaryWriter,
    split: str,
    iteration: int,
    device: torch.device,
    autoencoder: VQModel | None,
    diffusor: Diffusion,
    dataloader: data.DataLoader,
    model: nn.Module,
    loss_fn: CombinedLoss | torch.nn.MSELoss,
) -> float:
    num_batches = len(dataloader)
    loss_total = 0.0
    mse_total = 0.0
    percep_total = 0.0
    mae_total = 0.0
    psnr_total = 0.0
    fid_total = 0.0
    ssim_total = 0.0
    lpips_total = 0.0
    autoencoder_loss_total = 0.0
    batch_size = dataloader.batch_size
    assert isinstance(batch_size, int)

    lpips = LPIPS()

    model.eval()
    for batch, (hq, lq) in enumerate(dataloader):
        lq, hq = lq.to(device), hq.to(device)
        if autoencoder:
            z_lq, _, _ = autoencoder.encode(lq)
            z_hq, _, _ = autoencoder.encode(hq)
            z_pred, progress = diffusor.reverse_process(z_lq, model, True, device)
            autoencoder_loss_total += torch.nn.functional.mse_loss(z_pred, z_hq).item()
            pred = autoencoder.decode(z_pred)
            images = (
                [lq[0]]
                + [autoencoder.decode(p)[0] for p in progress]
                + [pred[0]]
                + [hq[0]]
            )
        else:
            pred, progress = diffusor.reverse_process(lq, model, True, device)
            images = [lq[0]] + [p[0] for p in progress] + [pred[0]] + [hq[0]]
        loss = loss_fn(pred, hq)

        loss_total += loss.item()
        if isinstance(loss_fn, CombinedLoss):
            mse_total += loss_fn.last_mse
            percep_total += loss_fn.last_percep
        mae_total += l1_loss(pred, hq).item()
        psnr_total += psnr(
            to_dtype(pred, torch.uint8, scale=True),
            to_dtype(hq, torch.uint8, scale=True),
            data_range=255,
        ).item()
        ssim_total += ssim(
            to_dtype(pred, torch.uint8, scale=True),
            to_dtype(hq, torch.uint8, scale=True),
            data_range=255,
        ).item()  # pyright: ignore[reportAttributeAccessIssue] full=false only returns one value
        lpips_total += lpips(pred, hq).item()

        logger.add_images(
            f"{split}/{batch_size * batch}", torch.stack(images), iteration + 1
        )

        print(f"{split} {batch + 1} loss: {loss}")

    logger.add_scalar(f"{split}/avg_loss", loss_total / num_batches, iteration + 1)
    logger.add_scalar(f"{split}/avg_mse", mse_total / num_batches, iteration + 1)
    logger.add_scalar(f"{split}/avg_percep", percep_total / num_batches, iteration + 1)
    logger.add_scalar(f"{split}/avg_mae", mae_total / num_batches, iteration + 1)
    logger.add_scalar(f"{split}/avg_psnr", psnr_total / num_batches, iteration + 1)
    logger.add_scalar(f"{split}/avg_fid", fid_total / num_batches, iteration + 1)
    logger.add_scalar(f"{split}/avg_ssim", ssim_total / num_batches, iteration + 1)
    logger.add_scalar(f"{split}/avg_lpips", lpips_total / num_batches, iteration + 1)
    if autoencoder:
        logger.add_scalar(
            f"{split}/avg_autoencoder_loss",
            autoencoder_loss_total / num_batches,
            iteration + 1,
        )
    return loss_total / num_batches
