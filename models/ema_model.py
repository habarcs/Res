from torch.optim.swa_utils import AveragedModel, get_ema_avg_fn
import config
import torch


def ema_model_from_config(model: torch.nn.Module, cfg: config.EMAModelCfg) -> AveragedModel|None:
    if config.EMAModelCfg.enabled:
        return AveragedModel(model, avg_fn=get_ema_avg_fn(config.EMAModelCfg.decay), use_buffers=True)
    return None
