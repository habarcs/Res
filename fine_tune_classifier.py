from classifier.cls_model import fine_tune
import config

if __name__ == "__main__":
    fine_tune_cfg = config.ClassifierFineTuneCfg()
    data_cfg = config.ClassifierDataCfg()
    fine_tune(fine_tune_cfg, data_cfg)
