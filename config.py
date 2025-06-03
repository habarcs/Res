from dataclasses import dataclass
import dataclasses
from pathlib import Path
from datetime import datetime


@dataclass
class EMAModelCfg:
    enabled: bool = True
    decay: float = 0.999


@dataclass
class DiffusionCfg:
    p: float = 0.3
    T: int = 15
    kappa: float = 2.0


@dataclass
class TrainingCfg:
    # one iteration is one batch, so everything here depends on batch_size defined in datacfg
    iterations: int = 100_000
    scheduler_freq: int = 5000
    save_freq: int = 50
    val_freq: int = 5000
    log_freq: int = 50
    run_id: str = datetime.now().isoformat(timespec="minutes")
    save_dir: Path = Path("checkpoint")
    lr_start: float = 5e-5
    lr_end: float = 2e-5

    def todict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class DataCfg:
    data_dir: Path = Path("/home/marton/work/rin2d/radiology_ai/US")
    image_size: int = 224
    scale_factor: int = 4
    mean: float = 0.5
    std: float = 0.5
    batch_size: int = 64
    num_workers: int = 4
    split_seed: int = 42
    train_ratio: float = 0.9
    val_ratio: float = 0.05
    test_ratio: float = 0.05
