from src.main import main

import hydra
from omegaconf import DictConfig, OmegaConf

# @hydra.main(version_base=None, config_path="./../config", config_name="multibodypendulum_tables")
@hydra.main(version_base=None, config_path="./../config", config_name="multibodypendulum_penalty_sweep")
def my_app(cfg: DictConfig) -> None:
    # print(cfg.run.seed)
    # return
    main(cfg)
    # print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    my_app()