from classifier.cls_model import eval_best
import config

if __name__ == "__main__":
    fine_tune_cfg = config.ClassifierFineTuneCfg()
    data_cfg = config.ClassifierDataCfg()
    eval_best(fine_tune_cfg, data_cfg)
