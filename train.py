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
    optimizer = torch.optim.Adam(model.parameters())
    train_loader, val_loader, test_loader = data_loader_from_config(config.DataCfg())

    train_loop(
        config.TrainingCfg(),
        device,
        diffusor,
        train_loader,
        val_loader,
        test_loader,
        model,
        ema_model,
        loss_fn,
        optimizer,
    )


if __name__ == "__main__":
    main()
