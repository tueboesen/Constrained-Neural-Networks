import hydra
from omegaconf import DictConfig

from src.main import main


@hydra.main(version_base=None, config_path="./../config", config_name="water")
def my_app(cfg: DictConfig) -> None:
    main(cfg)
    # print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    my_app()
