import argparse

import torch
from pytorch_lightning import Trainer
from torchvision.utils import save_image

import config
from datapipe.dataloader import (
    autoencoder_data_loader_from_config,
)
from taming.models.vqgan import VQModel
from taming.modules.losses.vqperceptual import VQLPIPSWithDiscriminator
from training.saver import unnormalize


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_path")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--fast", action="store_true")
    return parser.parse_args()


def eval(cfg: config.VQGANCfg, args):
    torch.set_float32_matmul_precision("high")
    loss = VQLPIPSWithDiscriminator(**cfg.lossconfig)
    autoencoder = VQModel(
        cfg.ddconfig,
        loss,
        cfg.n_embed,
        cfg.embed_dim,
        image_key=0,
        ckpt_path=args.ckpt_path,
    )  # pyright: ignore[reportArgumentType]
    autoencoder.learning_rate = cfg.batch_size * cfg.base_learning_rate  # pyright: ignore[reportArgumentType]

    train_loader, val_loader, test_loader = autoencoder_data_loader_from_config(cfg)

    trainer = Trainer(
        deterministic=True,
        default_root_dir=cfg.save_dir,
        logger=True,
    )

    if args.fast:
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(test_loader.dataset, list(range(100)))
        )

    trainer.validate(autoencoder, test_loader)
    if not args.save_images:
        return

    (cfg.save_dir / cfg.run_id).mkdir(parents=True, exist_ok=True)

    autoencoder.eval()
    with torch.no_grad():
        for i, (b, _) in enumerate(test_loader):
            rec, _ = autoencoder(b)
            save_image(
                [
                    unnormalize(b[0], cfg.mean[0], cfg.std[0]),
                    unnormalize(rec[0], cfg.mean[0], cfg.std[0]),
                ],
                cfg.save_dir / cfg.run_id / f"diff_{i}.png",
            )


if __name__ == "__main__":
    cfg = config.VQGANCfg()
    args = get_args()
    if args.run_id:
        cfg.run_id = args.run_id
    if args.save_images:
        print(f"Images will be saved in {cfg.save_dir / cfg.run_id}")
    eval(cfg, args)
