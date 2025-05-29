from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from models.unet_simple import UNet
from datapipe.dataloader import create_cifar_dataloaders
from diffusion.diffusion import Diffusion
from diffusion.shifting_sequence import create_shifting_seq
from training.train import train_loop
import torch
from datetime import datetime

if __name__ == "__main__":
    run_id = datetime.now().strftime("%Y-%m-%dT%H:%M")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(2025)

    model = UNet().to(device)
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999)).to(device)
    train_loader, val_loader = create_cifar_dataloaders("data", img_size=32, sf=4, mean=0.5, std=0.5, batch_size=4, num_workers=4, num_batches=1_000)
    diffusor = Diffusion(0.5, create_shifting_seq(5, 0.2), 5, (4, 3, 32, 32))
    loss_fn = torch.nn.MSELoss()
    optim = torch.optim.Adam(model.parameters())

    train_loop(device, "checkpoint/" + run_id, diffusor, train_loader, val_loader, model, ema_model, loss_fn, optim, 10, 250, 1)

