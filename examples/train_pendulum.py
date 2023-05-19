import hydra
from omegaconf import DictConfig

from src.main import main


@hydra.main(version_base=None, config_path="./../config", config_name="multibodypendulum_tables")
# @hydra.main(version_base=None, config_path="./../config", config_name="multibodypendulum")
def my_app(cfg: DictConfig) -> None:
    # print(cfg.run.seed)
    # return
    main(cfg)
    # print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    my_app()
