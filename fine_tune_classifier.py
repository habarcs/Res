import argparse
from classifier.cls_model import fine_tune
import config


def get_optional_run_id() -> str | None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    return parser.parse_args().run_id


if __name__ == "__main__":
    fine_tune_cfg = config.ClassifierFineTuneCfg()
    data_cfg = config.ClassifierDataCfg()
    ema_cfg = config.EMAModelCfg()
    run_id = get_optional_run_id()
    if run_id:
        fine_tune_cfg.run_id = run_id
    fine_tune(fine_tune_cfg, data_cfg, ema_cfg)
