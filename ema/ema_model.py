from torch.optim.swa_utils import AveragedModel, get_ema_avg_fn
import config
import torch


def ema_model_from_config(
    model: torch.nn.Module,
    cfg: config.EMACfg | config.ClassifierEMACfg,
    device: torch.device | None = None,
) -> AveragedModel | None:
    if cfg.enabled:
        ema_model = AveragedModel(
            model, avg_fn=get_ema_avg_fn(cfg.decay), use_buffers=True
        )
        if device:
            return ema_model.to(device)
        return ema_model
    return None
