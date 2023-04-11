from src.batch_jobs import job_runner
from src.main import main

import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="./../config", config_name="multibodypendulum_penalty")
def objective(cfg: DictConfig) -> None:

    loss = main(cfg)
    return loss

if __name__ == "__main__":
    objective()



