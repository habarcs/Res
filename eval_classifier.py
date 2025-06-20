import argparse
from classifier.cls_model import eval_model
import config


def get_args() -> tuple[str, bool]:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--no-ema", action="store_false")
    args = parser.parse_args()
    return args.model_path, args.no_ema


if __name__ == "__main__":
    fine_tune_cfg = config.ClassifierFineTuneCfg()
    data_cfg = config.ClassifierDataCfg()
    model_path, no_ema = get_args()
    eval_model(fine_tune_cfg, data_cfg, model_path, no_ema)
