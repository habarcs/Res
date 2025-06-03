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
    p: float = 0.4
    T: int = 10
    kappa: float = 0.3


@dataclass
class TrainingCfg:
    # one iteration is one batch, so everything here depends on batch_size defined in datacfg
    iterations: int = 100000
    save_freq: int = 250
    val_freq: int = 250
    log_freq: int = 1
    run_id: str = datetime.now().isoformat(timespec="minutes")
    save_dir: Path = Path("checkpoint")

    def todict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class DataCfg:
    data_dir: Path = Path("/home/marton/work/rin2d/radiology_ai/US")
    image_size: int = 224
    scale_factor: int = 4
    mean: float = 0.5
    std: float = 0.5
    batch_size: int = 12
    num_workers: int = 4
    split_seed: int = 42
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
