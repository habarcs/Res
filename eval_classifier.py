import argparse
from unittest.mock import Mock

import torch

import config
from classifier.cls_model import ClsModel, load_model, test_step
from datapipe.dataloader import classfication_data_loader_from_config


def eval_model(
    fine_tune_cfg: config.ClassifierFineTuneCfg,
    data_cfg: config.ClassifierDataCfg,
    model_path: str,
    no_ema: bool,
):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(2025)

    _, _, test_loader, classes = classfication_data_loader_from_config(data_cfg)
    model = ClsModel(len(classes))
    load_model(model, model_path, not no_ema)

    if fine_tune_cfg.compile:
        torch.set_float32_matmul_precision("high")
        model.compile()

    loss_fn = torch.nn.CrossEntropyLoss()

    acc = test_step(test_loader, Mock(), device, model, loss_fn)
    print(f"Final test accuracy: {acc}")


def get_args() -> tuple[str, bool]:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--no-ema", action="store_true")
    args = parser.parse_args()
    return args.model_path, args.no_ema


if __name__ == "__main__":
    fine_tune_cfg = config.ClassifierFineTuneCfg()
    data_cfg = config.ClassifierDataCfg()
    model_path, no_ema = get_args()
    eval_model(fine_tune_cfg, data_cfg, model_path, no_ema)
