from datapipe.dataloader import data_loader_from_config
from diffusion.diffusion import Diffusion
import torch
import config
from models.unet_simple import UNet
from models.ema_model import ema_model_from_config
from training.trainer import train_loop


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(2025)

    model = UNet()
    ema_model = ema_model_from_config(model, config.EMAModelCfg())
    diffusor = Diffusion.from_config(config.DiffusionCfg())
    loss_fn = torch.nn.MSELoss()
    training_cfg = config.TrainingCfg()
    optimizer = torch.optim.Adam(model.parameters(), training_cfg.lr_start)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        training_cfg.iterations // training_cfg.scheduler_freq,
        training_cfg.lr_end,
    )
    train_loader, val_loader, test_loader = data_loader_from_config(config.DataCfg())

    train_loop(
        training_cfg,
        device,
        diffusor,
        train_loader,
        val_loader,
        test_loader,
        model,
        ema_model,
        loss_fn,
        optimizer,
        scheduler,
    )


if __name__ == "__main__":
    main()
