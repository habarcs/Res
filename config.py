from dataclasses import dataclass, field
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
    T: int = 10
    kappa: float = 1.0


@dataclass
class TrainingCfg:
    # one iteration is one batch, so everything here depends on batch_size defined in datacfg
    iterations: int = 50_000
    scheduler_freq: int = 1000
    val_freq: int = 5_000
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
    grayscale: bool = False
    batch_size: int = 32
    num_workers: int = 4
    ultrasound_augmentation: bool = False


@dataclass
class ModelCfg:
    # for possible architectures see https://smp.readthedocs.io/en/latest/models.html
    arch: str = "unet"  # some modification of the code may be neccessary if changed
    # for possible encoder and encoder weights options see https://smp.readthedocs.io/en/latest/encoders.html
    # This encoder refers to the downsampling block, not the latent diffusion autoencoder
    encoder: str = (
        "mit_b1"  # some modification of the code may be neccessary if changed
    )
    encoder_weights: str | None = None
    swin_attention: bool = False
    t_embedding_dim: int = 32
    autoencoder: bool = False
    autoencoder_model_path: Path = Path("")


@dataclass
class LossCfg:
    use_percpetual_loss: bool = False
    perceptual_model_path: Path = Path("")
    use_ema: bool = True
    perceptual_coef: float = 0.05
    perceptual_loss_weights: Sequence[float] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)


@dataclass
class ClassifierDataCfg(BaseDataCfg):
    mean: Sequence[float] = (0.485, 0.456, 0.406)
    std: Sequence[float] = (0.229, 0.224, 0.225)
    batch_size: int = 128
    num_workers: int = 4


@dataclass
class ClassifierTrainingCfg:
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


@dataclass
class VQGANCfg(BaseDataCfg):
    mean: Sequence[float] = (0.5, 0.5, 0.5)
    std: Sequence[float] = (0.5, 0.5, 0.5)
    num_workers: int = 4

    embed_dim: int = 3
    n_embed: int = 8192
    base_learning_rate: float = 4.5e-6

    ddconfig: dict = field(
        default_factory=lambda: {
            "double_z": False,
            "z_channels": 3,
            "resolution": 224,
            "in_channels": 3,
            "out_ch": 3,
            "ch": 128,
            "ch_mult": [1, 2, 4],  # num_down = len(ch_mult)-1
            "num_res_blocks": 2,
            "attn_resolutions": [],
            "dropout": 0.0,
        }
    )

    # disriminator loss
    lossconfig: dict = field(
        default_factory=lambda: {
            "disc_conditional": False,
            "disc_in_channels": 3,
            "disc_start": 10000,
            "disc_weight": 0.75,
            "codebook_weight": 1.0,
        }
    )

    # data
    batch_size = 12
    num_epochs = 1000
    run_id: str = "auto_" + datetime.now().isoformat(timespec="minutes")
    save_dir: Path = Path("out")
