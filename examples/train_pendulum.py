from src.batch_jobs import job_runner
from src.main import main

import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="./../config", config_name="pendulum")
def my_app(cfg: DictConfig) -> None:
    main(cfg)
    # print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    my_app()