import argparse
from classifier.cls_model import eval_model
import config


def get_model_path() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    return parser.parse_args().model_path


if __name__ == "__main__":
    fine_tune_cfg = config.ClassifierFineTuneCfg()
    data_cfg = config.ClassifierDataCfg()
    eval_model(fine_tune_cfg, data_cfg, get_model_path())
