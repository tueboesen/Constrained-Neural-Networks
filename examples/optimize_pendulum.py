from src.batch_jobs import job_runner
from src.main import main

import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="./../config", config_name="pendulum_optuna")
def objective(cfg: DictConfig) -> None:
    loss = main(cfg)
    # print(OmegaConf.to_yaml(cfg))
    return loss

if __name__ == "__main__":
    objective()



