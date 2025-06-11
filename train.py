from datapipe.dataloader import data_loader_from_config
from diffusion.diffusion import Diffusion
import torch
import config
from upscaler.ema_model import ema_model_from_config
from upscaler.smp_model import SmpModel
from training.trainer import train_loop


def main():
    data_cfg = config.DataCfg()
    model_cfg = config.ModelCfg()
    diffusion_cfg = config.DiffusionCfg()
    training_cfg = config.TrainingCfg()
    ema_cfg = config.EMAModelCfg()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(2025)

    train_loader, val_loader, test_loader = data_loader_from_config(data_cfg)
    model = SmpModel.from_config(model_cfg, data_cfg, diffusion_cfg)
    ema_model = ema_model_from_config(model, ema_cfg)
    diffusor = Diffusion.from_config(diffusion_cfg)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), training_cfg.lr_start)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        training_cfg.iterations // training_cfg.scheduler_freq,
        training_cfg.lr_end,
    )

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
