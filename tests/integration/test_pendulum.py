import hydra
from omegaconf import DictConfig

from src.main import main


@hydra.main(version_base=None, config_path="./../../config", config_name="multibodypendulum")
def test_multibodypendulum(cfg: DictConfig) -> None:
    cfg.run.epochs = 1
    main(cfg)
