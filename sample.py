import argparse

import torch
from datapipe.dataloader import create_cifar_dataloaders
from diffusion.diffusion import Diffusion
from models.unet_simple import UNet
from diffusion.shifting_sequence import create_shifting_seq
from models import load_state
from pathlib import Path
from torchvision.utils import save_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Diffusion smapler",
        description="Takes in images, creates a low resolution version, upscales and compares with original",
    )
    parser.add_argument("model_path")
    parser.add_argument("out_dir")

    args = parser.parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = UNet().to(device)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    load_state(args.model_path, model, None, None)

    diffusor = Diffusion(0.5, create_shifting_seq(5, 0.2), 5, (4, 3, 32, 32))
    
    _, test_loader = create_cifar_dataloaders("data", img_size=32, sf=4, mean=0.5, std=0.5, batch_size=4, num_workers=4, num_batches=1_000)

    
    model.eval()
    with torch.no_grad():
        hq, lq = next(iter(test_loader))
        hq.to(device)
        lq.to(device)
        pred = diffusor.reverse_process(lq, model)
    save_image(lq, out / "lq.png")    
    save_image(pred, out / "pr.png")    
    save_image(hq, out / "hq.png")    

