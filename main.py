from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from predictor import UNet
from dataloader import create_cifar_dataloaders
from diffusion import Diffusion
from shifting_sequence import create_shifting_seq
from train import test_loop, train_loop
import torch

if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(2025)

    model = UNet().to(device)
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999)).to(device)
    train_loader, val_loader = create_cifar_dataloaders("data", img_size=32, sf=4, mean=0.5, std=0.5, batch_size=12, num_batches=10_000)
    diffusor = Diffusion(0.5, create_shifting_seq(10, 0.2), 10, (12, 3, 32, 32))
    loss_fn = torch.nn.MSELoss()
    optim = torch.optim.Adam(model.parameters())

    train_loop(device, diffusor, train_loader, val_loader, model, ema_model, loss_fn, optim)

    # TODO if there is test dataloader
    # test_loop(device, diffusor, test_loader, ema_model, loss_fn)
