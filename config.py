from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from collections.abc import Sequence


@dataclass
class EMACfg:
    enabled: bool = True
    decay: float = 0.999


@dataclass
class DiffusionCfg:
    p: float = 0.3
    T: int = 15
    kappa: float = 1.0


@dataclass
class TrainingCfg:
    # one iteration is one batch, so everything here depends on batch_size defined in datacfg
    iterations: int = 100_000
    scheduler_freq: int = 1000
    val_freq: int = 10_000
    backprop_freq: int = 1  # for gradient accumulation
    run_id: str = "diff_" + datetime.now().isoformat(timespec="minutes")
    save_dir: Path = Path("out")
    lr_start: float = 5e-5
    lr_end: float = 2e-5
    compile: bool = False


@dataclass
class BaseDataCfg:
    # dataset
    data_dir: Path = Path("/home/marton/work/rin2d/radiology_ai/US")
    image_size: int = 224

    # training split the following settings should not be changed
    split_seed: int = 42
    train_ratio: float = 0.9
    val_ratio: float = 0.05
    test_ratio: float = 0.05


@dataclass
class DataCfg(BaseDataCfg):
    scale_factor: int = 4
    mean: float = 0.5
    std: float = 0.5
    grayscale: bool = True
    batch_size: int = 64
    num_workers: int = 4
    ultrasound_augmentation: bool = True


@dataclass
class ModelCfg:
    # for possible architectures see https://smp.readthedocs.io/en/latest/models.html
    arch: str = (
        "unet"  # some modification of the code may be neccessary if changed
    )
    # for possible encoder and encoder weights options see https://smp.readthedocs.io/en/latest/encoders.html
    # This encoder refers to the downsampling block, not the latent diffusion autoencoder
    encoder: str = (
        "mit_b1"  # some modification of the code may be neccessary if changed
    )
    encoder_weights: str | None = None
    swin_attention: bool = True
    t_embedding_dim: int = 32


@dataclass
class LossCfg:
    perceptual_model_path: Path = Path("")
    use_ema: bool = True
    perceptual_coef: float = 0.05
    perceptual_loss_weights: Sequence[float] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)


@dataclass
class AutoencoderCfg:
    enabled: bool = True
    arch: str = ""
    weights: str = ""


@dataclass
class ClassifierDataCfg(BaseDataCfg):
    mean: Sequence[float] = (0.485, 0.456, 0.406)
    std: Sequence[float] = (0.229, 0.224, 0.225)
    batch_size: int = 1024
    num_workers: int = 4


@dataclass
class ClassifierFineTuneCfg:
    epochs: int = 30
    starting_lr: float = 4e-5
    ending_lr: float = 1e-8
    run_id: str = "cls_" + datetime.now().isoformat(timespec="minutes")
    save_dir: Path = Path("out")
    compile: bool = False


@dataclass
class ClassifierEMACfg:
    enabled: bool = True
    decay: float = 0.999
