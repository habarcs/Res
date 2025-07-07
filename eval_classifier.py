import argparse
from pathlib import Path

import torch

import config
from classifier.cls_model import ClsModel, load_model, test_step
from datapipe.dataloader import classfication_data_loader_from_config
from torch.utils.tensorboard import SummaryWriter


def eval_model(
    fine_tune_cfg: config.ClassifierFineTuneCfg,
    data_cfg: config.ClassifierDataCfg,
    model_path: str,
    run_id: str,
    no_ema: bool,
):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(2025)

    if run_id:
        log_path = fine_tune_cfg.save_dir / run_id / "eval_log"
    else:
        log_path = Path(model_path).parent / "eval_log"
    logger = SummaryWriter(log_path)

    _, _, test_loader, classes = classfication_data_loader_from_config(data_cfg)
    model = ClsModel(len(classes))
    load_model(model, model_path, not no_ema)
    model.to(device)

    if fine_tune_cfg.compile:
        torch.set_float32_matmul_precision("high")
        model.compile()

    loss_fn = torch.nn.CrossEntropyLoss()

    acc = test_step(test_loader, logger, device, model, loss_fn)
    print(f"Final test accuracy: {acc}")


def get_args() -> tuple[str, str, bool]:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--run-id", default="")
    args = parser.parse_args()
    return args.model_path, args.run_id, args.no_ema


if __name__ == "__main__":
    fine_tune_cfg = config.ClassifierFineTuneCfg()
    data_cfg = config.ClassifierDataCfg()
    model_path, run_id, no_ema = get_args()
    eval_model(fine_tune_cfg, data_cfg, model_path, run_id, no_ema)
