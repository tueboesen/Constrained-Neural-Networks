from src.batch_jobs import job_runner
from src.main import main

import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="./../../config", config_name="multibodypendulum")
def test_multibodypendulum(cfg: DictConfig) -> None:
    cfg.run.epochs = 1
    main(cfg)
    


