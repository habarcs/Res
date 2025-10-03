import argparse
from pytorch_lightning import Trainer

import config
from taming.models.vqgan import VQModel
from taming.modules.losses.vqperceptual import VQLPIPSWithDiscriminator

from datapipe.dataloader import (
    autoencoder_data_loader_from_config,
)


def get_optional_run_id() -> str | None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    return parser.parse_args().run_id


def train(cfg: config.VQGANCfg):
    loss = VQLPIPSWithDiscriminator(**cfg.lossconfig)
    autoencoder = VQModel(cfg.ddconfig, loss, cfg.n_embed, cfg.embed_dim)

    train_loader, val_loader, test_loader = autoencoder_data_loader_from_config(cfg)

    trainer = Trainer(deterministic=True, default_root_dir=cfg.save_dir, logger=True)
    trainer.fit(autoencoder, train_loader, val_loader)

    trainer.test(autoencoder, test_loader)


if __name__ == "__main__":
    run_id = get_optional_run_id()
    cfg = config.VQGANCfg()
    if run_id:
        cfg.run_id = run_id
    train(cfg)
